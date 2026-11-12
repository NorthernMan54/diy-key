#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Outbound mTLS WebSocket ("ucp4") client -- the ongoing connection a real AI
Key opens to the console after ``POST /api/adopt`` (see
``stage1_http/adopt_http.py``, which writes ``adopt_state.json`` for this
script to pick up).

The envelope is two concatenated records, each with an 8-byte header:
``type u8, format u8, compressed u8, reserved u8, length u32 BE``. The length
is 32-bit: a 16-bit read silently truncates any payload over 64 KiB (the
controller reads it at offset 4 as a u32). ``format=3`` means the body is raw
bytes rather than JSON.

AI-Key RPC handlers are the live-confirmed ones from ``../../protocol.md`` §3;
unknown actions are acked with ``{}`` and logged so the console's own
``errors.log`` can teach us new shapes.
"""
import argparse
import concurrent.futures
import json
import logging
import os
import queue
import re
import shutil
import ssl
import struct
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import parse_qsl, urlsplit

import requests
import urllib3
from websockets.sync.client import connect

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core import config  # noqa: E402
from system_stats import collect_system_metrics  # noqa: E402
import clip_fetch  # noqa: E402
import firmware  # noqa: E402
import ram_tagger  # noqa: E402
import nl_search  # noqa: E402
import clip_embed  # noqa: E402
import reid_model  # noqa: E402
import second_verifier  # noqa: E402
import face_recog  # noqa: E402
import lpr  # noqa: E402
import face_enhance  # noqa: E402
import vlm_describe  # noqa: E402
import scene_memory  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CONSOLE_INTERNAL_API_PORT = 7444

log = logging.getLogger("ucp4-client")

# UCP4 record header (8 bytes): type, format, compressed, reserved, u32 length.
RECORD_TYPE_HEAD = 1
RECORD_TYPE_BODY = 2
RECORD_FORMAT_JSON = 1
RECORD_FORMAT_RAW = 3

# --state-dir (or None for the historical stage1_http/ path). Resolved in
# main(); kept as plain paths here so importing this module has no side effects.
STATE_DIR = None
PROCESS_START = time.time()  # getInfo "uptime" counts from here

# What a real AI Key advertises in getInfo featureFlags (Protect's aiprocessor model lists
# them); the UI enables features, e.g. person second-stage verification, from these. Built
# from the environment at each getInfo so they say what this instance can actually do.
# Leaving a flag out is not "off": for an AI Key Protect fills an absent
# supportFaceRecognition/LicensePlateRecognition/AiSummary/RecognizeAnything/FaceEnhancement
# (and RetroactiveProcessing from fw 1.3.9) with enabled=true (Protect M/643837:52-67), so an
# unsupported feature is sent explicitly as {"enabled": false}.
_OFF_VALUES = ("", "0", "off", "false", "no")


def _truthy(value):
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _feature(enabled):
    return {"enabled": bool(enabled), "version": "v1"}


def _deep_flags(env, vlm):
    on = vlm and _truthy(env.get("DIYKEY_DEEP_MODE"))
    return {"supportDeepMode": on, "supportVlm": on}


def _feature_flags(env=None):
    """featureFlags for getInfo (aiMode added by the caller). Every capability defaults to
    off unless its DIYKEY_* key enables it."""
    env = os.environ if env is None else env
    ram = bool((env.get("DIYKEY_RAM_MODEL") or "").strip())
    # Only the backends vlm_describe implements; any other value would advertise a summary
    # that never arrives.
    vlm = (env.get("DIYKEY_VLM") or "").strip().lower() in ("mlx", "transformers")
    return {
        # Unchanged from the always-on list this replaced.
        "supportPoseEstimation": _feature(True),
        # Gates the Person class in Second Stage Verification: Protect builds the NVR's in-memory
        # aiFeatureFlag.supportPersonReId from adopted AI Keys' getInfo (live 2026-10-29 on
        # Mac B: Person unselectable until advertised, then a browser reload). Live 2026-11-07
        # (Protect 7.3.74): turning it off hid Person again. No ReID model is needed for SSV
        # itself; DIYKEY_PERSON_REID=0 turns it off.
        "supportPersonReId": _feature((env.get("DIYKEY_PERSON_REID") or "1").strip().lower()
                                      not in _OFF_VALUES),
        "supportFaceEnhancement": _feature(_truthy(env.get("DIYKEY_FACE_ENHANCE"))),
        "supportRetroactiveProcessing": _feature(ram),
        "supportImageSearch": _feature(_truthy(env.get("DIYKEY_CLIP"))),
        "supportRecognizeAnything": _feature(ram),
        "supportFaceRecognition": _feature(_truthy(env.get("DIYKEY_FACE"))),
        "supportLicensePlateRecognition": _feature(_truthy(env.get("DIYKEY_LPR"))),
        "supportAiSummary": _feature(vlm),
        # Deep Understanding (Protect M/59223:106-117, M/643837:69-72: plain booleans, absent
        # reads false; supportVlm is the legacy alias, kept in sync). isDeepModeCapable
        # (M/387031:24) is what lets Protect send modelMode "deep" and, once the NVR's deepUnderstandingSettings
        # are on, RAM is switched off on this key (M/288811:12-14, M/851272:119) and the
        # work arrives as RequestAI :7968/describe instead. So it is opt-in
        # (DIYKEY_DEEP_MODE=1) and needs a working VLM: advertising it without one would trade
        # working recognizeKeyFrames for descriptions that never come.
        **_deep_flags(env, vlm),
        # fastReverification (Second Stage Verification from per-thumbnail JPEGs instead of a
        # clip) runs second_verifier, which needs the detector DIYKEY_RAM_MODEL loads. The
        # firmware advertises it (aikey-fw 2.2.8 ai-key-base-file featureflags.json); Protect
        # 7.3.70 has no such attribute (M/59223) so this only matters on a newer Protect.
        "supportReverification": _feature(ram),
        # The firmware's own description: "AI Key presents its TLS client certificate on
        # outbound Protect internal aiprocessors API requests". Protect only uses it to refuse
        # an internal request that presents no matching certificate (M/761379:83-104), while
        # the peer-address fallback still serves a device with no stored fingerprint. Every
        # DIY-Key call to :7444 (clip/image fetches, callbacks) passes cert=(CERT_PATH,
        # KEY_PATH), so enabling it changes nothing about how Protect connects.
        "supportAiProcessorInternalMtls": _feature(True),
        # Absent reads false (attr defaultValue false, M/59223): sent explicitly so it is
        # visibly off. A real key blinks its LED on `locate`, which DIY-Key can't do.
        "supportLocate": False,
    }


# Set by changeAiInferAgentSettings {modelMode}; Protect reads it back as featureFlags.aiMode.
_ai_mode = "basic"
# The rest of changeAiInferAgentSettings: updateAiSettings sends enableFaceRecognize,
# enableLprRecognize, enableRAM, enableSTT, enableFaceEnhance, deepModeSupported and region
# (no modelMode) on every connect and whenever the NVR's AI settings change (Protect
# M/288811:15-26, M/738405:27-42). Read with ai_setting_enabled().
_ai_settings = {}
# latestFirmwareVersion from the console's last firmwareState: the update target when the
# updateFirmware uri doesn't name one.
_latest_fw = None
# How long a simulated firmware update "installs" before the fake reboot.
FAKE_UPDATE_S = float(os.environ.get("DIYKEY_FAKE_UPDATE_S") or 15)


def _fw_version(device_info):
    """Reported firmware: the version a simulated update pinned, else the configured one."""
    return config.pinned_fw_version(STATE_DIR) or device_info["version"]


def _version_from_update_uri(uri):
    """Target version out of the console's update URL, or None: service.js builds
    /internal/update/<platform>/<product>/<firmware|software>/<version>/<mac>, or the
    same fields as a query string (getUpdateUrl)."""
    if not uri:
        return None
    try:
        parts = urlsplit(uri)
    except ValueError:
        return None
    segments = [seg for seg in parts.path.split("/") if seg]
    if len(segments) >= 3 and segments[-3] in ("firmware", "software"):
        return segments[-2]
    return dict(parse_qsl(parts.query)).get("version") or None


def _simulate_update(ws, target, body, mac_nosep):
    """Keep a copy of the firmware (unless DIYKEY_FW_DOWNLOAD=0), pretend to flash, pin the
    new version, then drop the connection as the reboot. The reconnect reports the pinned
    version, which ends Protect's update. Nothing is ever installed."""
    t0 = time.monotonic()
    if os.environ.get("DIYKEY_FW_DOWNLOAD", "1") != "0" and body.get("uri"):
        log.info("updateFirmware: downloading %s to %s", target, firmware.firmware_dir())
        firmware.download(body["uri"], target, body.get("md5"), _callback_headers(mac_nosep),
                          (CERT_PATH, KEY_PATH))
    wait = max(0.0, FAKE_UPDATE_S - (time.monotonic() - t0))
    log.info("updateFirmware: simulating install of %s for %.0fs", target, wait)
    time.sleep(wait)
    config.pin_fw_version(STATE_DIR, target)
    log.info("updateFirmware: now on %s; simulated reboot (closing ucp4)", target)
    try:
        ws.close()
    except Exception:  # noqa: BLE001 -- already closed is fine
        pass


def _psql_path():
    """psql from DIYKEY_PG_BIN, else PATH, else Homebrew's postgresql@17 (macOS service)."""
    pg_bin = os.environ.get("DIYKEY_PG_BIN")
    if pg_bin:
        return os.path.join(pg_bin, "psql")
    return shutil.which("psql") or "/opt/homebrew/opt/postgresql@17/bin/psql"


def _pg_password_plan(new_password):
    """(argv, stdin) that gives role ``unifi-protect`` the password Protect rotated to, or
    (None, reason) when this host has no reachable PostgreSQL.

    macOS service: DIYKEY_PG_SOCKET=<instance>/pg, psql runs as the login user.
    Linux system PostgreSQL: none needed -- as root psql runs as the postgres OS user over
    /var/run/postgresql, so peer auth lets it ALTER ROLE. DIYKEY_PG_USER overrides the OS
    user (any user that can authenticate as a superuser works).
    """
    sock = os.environ.get("DIYKEY_PG_SOCKET")
    user = os.environ.get("DIYKEY_PG_USER")
    if sys.platform != "darwin" and not sock and os.geteuid() == 0:
        sock = "/var/run/postgresql"
        user = user or "postgres"
    if not sock and not user:
        return None, "no DIYKEY_PG_SOCKET/DIYKEY_PG_USER"
    cmd = [_psql_path(), "-d", "postgres", "-q", "-v", "ON_ERROR_STOP=1"]
    if sock:
        cmd += ["-h", sock]
    runuser = shutil.which("runuser")
    if user and os.geteuid() == 0 and user != "root" and runuser:
        cmd = [runuser, "-u", user, "--"] + cmd
    elif user and user != "root":
        cmd += ["-U", user]
    # The password never goes in argv (it would show in ps): quoted as a SQL literal instead,
    # doubling embedded quotes, with backslash escapes off so it is stored verbatim.
    literal = "'" + new_password.replace("'", "''") + "'"
    sql = ("SET standard_conforming_strings = on;\n"
           f'ALTER ROLE "unifi-protect" PASSWORD {literal};\n')
    return cmd, sql


def _apply_pg_password(new_password):
    """Give the instance's PostgreSQL role the password Protect just rotated to, so its
    first candidate works."""
    if not new_password:
        return
    cmd, payload = _pg_password_plan(new_password)
    if cmd is None:
        log.info("changeUserPassword: %s; PostgreSQL password left as is", payload)
        return
    try:
        r = subprocess.run(cmd, input=payload, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("changeUserPassword: psql failed: %r", e)
        return
    if r.returncode:
        log.warning("changeUserPassword: psql exited %d: %s", r.returncode, r.stderr.strip())
    else:
        log.info("changeUserPassword: applied the rotated password to PostgreSQL")
ADOPT_STATE_FILE = os.path.join(config.DEFAULT_STATE_DIR, "adopt_state.json")
CERT_PATH = os.path.join(config.DEFAULT_STATE_DIR, "server.crt")
KEY_PATH = os.path.join(config.DEFAULT_STATE_DIR, "server.key")


def pack_record(record_type: int, payload) -> bytes:
    if isinstance(payload, (bytes, bytearray)):
        body = bytes(payload)
        rec_format = RECORD_FORMAT_RAW
    else:
        body = json.dumps(payload).encode()
        rec_format = RECORD_FORMAT_JSON
    header = struct.pack(">BBBBI", record_type, rec_format, 0, 0, len(body))
    return header + body


def pack_message(head: dict, body) -> bytes:
    return pack_record(RECORD_TYPE_HEAD, head) + pack_record(RECORD_TYPE_BODY, body)


def unpack_all_records(data: bytes):
    """[(record_type, format, obj)] for every complete record in a frame."""
    records = []
    off = 0
    while off < len(data):
        if len(data) - off < 8:
            log.warning("trailing %d bytes, not enough for a record header", len(data) - off)
            break
        rtype, rec_format, _compressed, _reserved, length = struct.unpack(
            ">BBBBI", data[off:off + 8])
        off += 8
        payload = data[off:off + length]
        off += length
        if rec_format == RECORD_FORMAT_RAW:
            obj = {"_raw_bytes": payload}
        else:
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                obj = {"_raw": payload.decode(errors="replace")}
        records.append((rtype, rec_format, obj))
    return records


def _console_host(fallback_host_port):
    """The adopted console's host: adopt_state hosts[0], else the configured fallback."""
    try:
        with open(ADOPT_STATE_FILE) as f:
            hosts = json.load(f).get("hosts") or []
        if hosts:
            return _split_host_port(hosts[0])[0]
    except (OSError, ValueError, TypeError):
        pass
    return _split_host_port(fallback_host_port)[0] if fallback_host_port else None


def _nl_search_reply(head, body, device_info):
    """Reply body for an nl-search request; empty fields on failure, like the AI Key."""
    action = head.get("action")
    body = body or {}
    try:
        if action == "NL_PARSE":
            reply = nl_search.parse(body.get("querySentence") or "")
            log.info("NL_PARSE %r (model %s) -> tags %s types %s time %s embed %d",
                     body.get("querySentence"), body.get("model"),
                     [t["matchedWord"] for t in reply["keyTags"]], reply["objectTypes"],
                     reply.get("timeTag"), len(reply["txtEmbed"]))
            return reply
        if action == "IMAGE_SEARCH":
            import cv2
            import numpy as np
            resp = requests.get(body.get("imgUri") or "", headers=_callback_headers(
                device_info["mac_nosep"]), cert=(CERT_PATH, KEY_PATH), verify=False, timeout=30)
            img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR) \
                if resp.status_code == 200 else None
            reply = nl_search.image_search(img)
            log.info("IMAGE_SEARCH %s -> embed %d", resp.status_code, len(reply["imgEmbed"]))
            return reply
    except Exception as e:  # noqa: BLE001 -- Protect waits for a reply either way
        log.warning("nl-search %s failed: %r", action, e)
        return {"imgEmbed": []} if action == "IMAGE_SEARCH" else {"keyTags": [], "objectTypes": []}
    log.warning("nl-search: unhandled action %r", action)
    return {}


def _nl_search_loop(fallback_host_port, device_info):
    """Hold the AI Key's second WebSocket (wss://<console>:7443/wss/nl-search/v1) open while
    adopted, so Protect routes Find Anything here (see nl_search.py)."""
    backoff = 5
    while True:
        host = _console_host(fallback_host_port) if config.is_adopted(STATE_DIR) else None
        if not host:
            time.sleep(10)
            continue
        url = f"wss://{host}:{nl_search.PORT}{nl_search.PATH}"
        extra = {}
        if device_info.get("bind_ip") not in (None, "", "0.0.0.0"):
            extra["source_address"] = (device_info["bind_ip"], 0)
        try:
            with connect(url, subprotocols=["ucp4"],
                         additional_headers={"x-ident": device_info["mac_nosep"]},
                         ssl=build_ssl_context(), open_timeout=15, **extra) as ws:
                log.info("nl-search connected to %s", url)
                global _nl_ws
                _nl_ws = ws
                backoff = 5
                for raw in ws:
                    if isinstance(raw, str):
                        continue
                    records = unpack_all_records(raw)
                    if len(records) < 2:
                        continue
                    head, body = records[0][2], records[1][2]
                    if head.get("type") != "request" or not head.get("id"):
                        continue
                    reply = _nl_search_reply(head, body, device_info)
                    ws.send(pack_message(
                        {"id": head["id"], "type": "response", "action": head.get("action"),
                         "timestamp": int(time.time() * 1000), "error": None, "errorCode": 0},
                        reply))
        except Exception as e:  # noqa: BLE001 -- reconnect on anything
            log.warning("nl-search connection to %s ended: %r -- retrying in %ds", url, e, backoff)
        finally:
            _nl_ws = None
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)


def wait_for_adopt_state(fallback_host_port, poll_interval=1.0, fallback_after=10.0):
    """Wait for stage1_http to write adopt_state.json from a fresh adopt POST.

    If the device is already adopted server-side (restart with no fresh POST),
    the controller may never re-POST; after the grace period fall back to a
    tokenless reconnect to the configured console host (the controller's
    record has our pinned fingerprint and accepts that).
    """
    log.info("waiting for %s to appear (written by stage1_http on a successful adopt POST)...",
             ADOPT_STATE_FILE)
    waited = 0.0
    if not config.is_adopted(STATE_DIR):
        # Empty/absent state (never adopted, or factory-reset by Protect): only a fresh adopt
        # POST may start a session. Live 2026-11-07: tokenless reconnects after a removal kept an
        # unadopted key showing lastSeen and receiving recognizeKeyFrames.
        fallback_host_port = None
        log.info("not adopted (adopt_state empty) -- waiting for a fresh adopt POST; "
                 "tokenless reconnect disabled")
    while True:
        if os.path.exists(ADOPT_STATE_FILE):
            state = config.read_adopt_state(STATE_DIR)
            if state.get("hosts"):
                return state
        if fallback_host_port and waited >= fallback_after:
            log.info("no adopt POST seen after %.0fs -- falling back to tokenless reconnect "
                     "to %s (device may already be adopted server-side)",
                     waited, fallback_host_port)
            return {"hosts": [fallback_host_port], "token": None}
        time.sleep(poll_interval)
        waited += poll_interval


def build_ssl_context():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # per spec: device does not validate the console cert
    ctx.load_cert_chain(CERT_PATH, KEY_PATH)  # our mTLS client identity
    return ctx


def ai_setting_enabled(name, default=False):
    """Whether Protect's last changeAiInferAgentSettings turned `name` on, e.g.
    "enableFaceRecognize" or "enableLprRecognize" (Protect M/288811:15-26). `default` until
    Protect has sent it; it does so on every connect (M/738405:27-42)."""
    value = _ai_settings.get(name)
    return default if value is None else bool(value)


# recognizeKeyFrames/RequestAI run on a small worker pool instead of a thread each: a burst
# of events (or a retroactive-processing backfill) would otherwise fetch and tag dozens of
# clips at once. Protect stops dispatching to a processor whose queue is over 200 (Protect
# M/851272:100-106), so a full queue here means something is wrong; drop rather than grow.
# Tasks that waited past DIYKEY_TASK_DROP_AGE_S (default 900 s, the "up to 15 minutes" delay
# of a full queue in Ubiquiti's AI Key FAQ) are discarded when dequeued. Protect has not given
# up yet: it waits 30 min for a tagging task, 3 min for describe/embeddings (M/630689:25
# resolveTaskTimeoutMs), so a dropped tagging task stays in progress there until that timeout.
_task_lock = threading.Lock()
_task_q = None
# queued/running count every task; queued_<group>/running_<group> also count the tasks of a
# group getTaskQueueInfo reports on its own ("ram": recognizeKeyFrames, "enhance": enhanceImage).
_task_stats = {"queued": 0, "queued_ram": 0, "queued_enhance": 0, "running": 0,
               "running_ram": 0, "running_enhance": 0, "dropped": 0}
# A recognizeKeyFrames task also counts under "face"/"lpr" when it will run that model
# (UI_AUTO_FACE_RECOGNITION / UI_AUTO_LPR_RECOGNITION in getTaskQueueInfo).


def _task_group(tag):
    """Queue items carry the old `ram` bool, a group name or a tuple of group names (a task
    can belong to several: recognizeKeyFrames is "ram" plus "face"/"lpr"); True means "ram"."""
    if tag is True:
        return ("ram",)
    if isinstance(tag, (tuple, list)):
        return tuple(tag)
    return (tag,) if tag else ()


def _count(state, group, n):
    """Adjust `state` ("queued"/"running") and its group counters; caller holds _task_lock."""
    _task_stats[state] += n
    for g in _task_group(group):
        key = f"{state}_{g}"
        _task_stats[key] = _task_stats.get(key, 0) + n


def _env_number(name, default, cast=int):
    try:
        return cast(os.environ.get(name) or default)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, os.environ.get(name), default)
        return cast(default)


_task_ctx = threading.local()  # wait_ms: how long the worker's current task sat in the queue


def _task_worker(q):
    drop_age = _env_number("DIYKEY_TASK_DROP_AGE_S", 900, float)
    while True:
        queued_at, kind, fn, args, tag = q.get()
        group = _task_group(tag)
        age = time.monotonic() - queued_at
        with _task_lock:
            _count("queued", group, -1)
            stale = age > drop_age
            if stale:
                _task_stats["dropped"] += 1
            else:
                _count("running", group, 1)
        if stale:
            log.warning("%s task waited %.0fs (> DIYKEY_TASK_DROP_AGE_S=%.0f); dropped",
                        kind, age, drop_age)
            continue
        _task_ctx.wait_ms = age * 1000
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 -- one bad task must not kill the worker
            log.exception("%s task failed", kind)
        finally:
            with _task_lock:
                _count("running", group, -1)


def _task_queue():
    """The task queue, with its DIYKEY_TASK_WORKERS workers started on first use."""
    global _task_q
    with _task_lock:
        if _task_q is None:
            _task_q = queue.Queue(maxsize=max(1, _env_number("DIYKEY_TASK_QUEUE_MAX", 200)))
            workers = max(1, _env_number("DIYKEY_TASK_WORKERS", 1))
            for n in range(workers):
                threading.Thread(target=_task_worker, args=(_task_q,), daemon=True,
                                 name=f"diykey-task-{n}").start()
            log.info("task queue: %d worker(s), max %d queued", workers, _task_q.maxsize)
        return _task_q


_summary_q = None


def _summary_queue():
    """The automatic-summary queue and its one worker, started on first use. Live 2026-11-09:
    a summary was about 30 s of each 60 s event on the only task worker, and 7 events waited
    past DIYKEY_TASK_DROP_AGE_S and were dropped."""
    global _summary_q
    with _task_lock:
        if _summary_q is None:
            _summary_q = queue.Queue(maxsize=max(1, _env_number("DIYKEY_TASK_QUEUE_MAX", 200)))
            threading.Thread(target=_task_worker, args=(_summary_q,), daemon=True,
                             name="diykey-summary").start()
        return _summary_q


def _submit_task(kind, fn, args, ram=False, group=None, extra_groups=(), q=None):
    """Queue fn(*args); False (and a warning) when the queue is full. ram=True is
    group="ram"; a group's tasks are also counted under queued_/running_<group>.
    extra_groups are further groups to count the task under (see _task_group). q is the
    queue to use, by default the main task queue."""
    q = _task_queue() if q is None else q
    group = _task_group("ram" if ram else group) + tuple(extra_groups)
    with _task_lock:
        try:
            q.put_nowait((time.monotonic(), kind, fn, args, group))
        except queue.Full:
            _task_stats["dropped"] += 1
            log.warning("task queue full (%d); dropped %s", q.maxsize, kind)
            return False
        _count("queued", group, 1)
    return True


_fast_lane = 0


def _fast_lane_count():
    with _task_lock:
        return _fast_lane


def _fast_lane_run(fn, *args):
    """Run fn(*args) counted in UI_TASK_FAST_LANE (on-demand summary thread body)."""
    global _fast_lane
    with _task_lock:
        _fast_lane += 1
    try:
        fn(*args)
    finally:
        with _task_lock:
            _fast_lane -= 1


def task_queue_stats():
    with _task_lock:
        return dict(_task_stats)


class ActionError(Exception):
    """handle_action's way to fail a request: _consume sends it as the response head's
    errorCode/error (ucpv4_send_response's {"errorCode":%d,"error":"%m"} shape)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


UNKNOWN_ACTION_CODE = -95
# Admin actions the firmware routes (ui-websocketd + syswrapper) that DIY-Key has nothing to do
# for; Protect expects them to succeed. locate stays an ack although supportLocate is false.
ACK_ONLY_ACTIONS = frozenset({
    "getSupportInfo", "setInfo", "updateTimezone", "sshService", "locate", "reboot",
    "updateLcmSettings", "setConsoleInfo",
    "speechToText", "RequestTts"})
# Actions handled by the if/elif chains in handle_action (the ones above are acks too).
HANDLED_ACTIONS = frozenset({
    "getInfo", "firmwareState", "updateFirmware", "networkStatus", "changeUserPassword",
    "changeAiInferAgentSettings", "diskInfo", "systemMetrics", "getTaskQueueInfo",
    "recognizeKeyFrames", "fastReverification", "enhanceImage", "enhanceImageGroup",
    "RequestAI", "factoryReset", "restoreDefault"})
_ack_only_logged = set()
# Tokens are single-use: send X-Token only the first time a value is seen. Module-level so a
# factory reset can forget them (a re-adopt may legitimately reuse nothing, but stale entries
# would also suppress a repeated token).
_used_tokens = set()
# The live nl-search socket, so a factory reset can drop it (the loop only re-checks adoption
# between connections).
_nl_ws = None
# Set by a factoryReset/restoreDefault; _consume closes the ucp4 socket after the reply is sent.
_reset_pending = threading.Event()


def _factory_reset(action):
    """Un-adopt locally, like the firmware's restore_default (aikey-fw 2.2.8 syswrapper.sh
    restore_default ~L4334: `rm -rf /etc/persistent/cfg/*` drops controller.cfg, i.e. the
    adoption; resets the ui/ubnt credentials and the Protect PostgreSQL credential; then the
    unit reboots and comes back unadopted, X-Adopted false). Live 2026-11-07 22:32:47 (Protect
    7.3.74): removing the AI Key sent factoryReset {} then changeAiInferAgentSettings; an ack-only
    DIY-Key kept adopt_state.json, fell back to a tokenless reconnect and stayed 'connected' to an
    console that had forgotten it. We keep server.crt/server.key and the fw_version pin: the
    firmware's wipe covers cfg/httpd/ulcmd/db, not its identity or installed firmware version,
    and a fresh adopt POST pins our existing fingerprint."""
    log.info("action %r: factory reset -- clearing adoption state, dropping used tokens, closing "
             "ucp4 and nl-search; waiting for a fresh adopt POST (no tokenless reconnect)", action)
    config.clear_adopt_state(STATE_DIR)
    _used_tokens.clear()
    _reset_pending.set()
    ws = _nl_ws
    if ws is not None:
        try:
            ws.close()
        except Exception:  # noqa: BLE001
            pass


def handle_action(head, body, device_info):
    global _ai_mode, _latest_fw
    action = head.get("action")
    log.info("<<< controller action=%r head=%s body=%s", action, head, body)

    if action == "getInfo":
        return {
            "mac": device_info["mac_nosep"],
            "type": device_info["type"],
            # Firmware getInfo format (ui-websocketd strings): {"type":"%s","sysid":"%s",
            # "version":"%s","mac":"%W","uptime":%u,"poeType":"%s","storageSize":%u,
            # "featureFlags":%s}: sysid is the same "0xa5f0" string as x-sysid, uptime whole
            # seconds, storageSize a NUMBER here (diskInfo's is the string "%d").
            "sysid": device_info.get("sysid") or "0xa5f0",
            "uptime": int(time.time() - PROCESS_START),
            "storageSize": 256,
            "version": _fw_version(device_info),
            # Read live so ResetToDefaults/readopt can't leave this stale.
            "adopted": config.is_adopted(STATE_DIR),
            "guid": device_info["guid"],
            "deviceId": device_info["device_id"],
            # The console's host, not ours (used to build upload URLs).
            "connectionHost": device_info.get("console_host") or device_info["ip"],
            "poeType": "802.3bt",  # Protect sets requiredPoeType 802.3bt for an AI Key
            "featureFlags": {**(flags := _feature_flags()),
                             # Protect clamps anything but "deep" to basic (M/643837:72) and only
                             # sends deep to a capable key, so never echo "deep" we don't support.
                             "aiMode": _ai_mode if flags["supportDeepMode"] else "basic"},
        }
    if action == "firmwareState":
        _latest_fw = (body or {}).get("latestFirmwareVersion") or _latest_fw
        return {}
    if action == "updateFirmware":
        target = _version_from_update_uri((body or {}).get("uri")) or _latest_fw
        if not target:
            log.warning("updateFirmware without a target version (uri=%r); ignoring",
                        (body or {}).get("uri"))
            return {}
        if device_info.get("ws") is not None:
            threading.Thread(target=_simulate_update,
                             args=(device_info["ws"], target, body, device_info["mac_nosep"]),
                             daemon=True).start()
        else:
            config.pin_fw_version(STATE_DIR, target)
        return {}
    if action == "networkStatus":
        # Protect reads linkSpeedMbps (and ssid/rssi, wifi only).
        return {"ip": device_info["ip"], "connected": True, "linkSpeedMbps": 1000}
    if action == "changeUserPassword":
        _apply_pg_password((body or {}).get("passwordNew"))
        return {}
    if action == "changeAiInferAgentSettings":
        body = body or {}
        if body.get("modelMode"):
            _ai_mode = body["modelMode"]
            log.info("AI mode set to %s", _ai_mode)
        settings = {k: v for k, v in body.items() if k != "modelMode"}
        if settings:
            _ai_settings.update(settings)
            log.info("AI settings now %s", _ai_settings)
        return {}
    if action == "diskInfo":
        # Protect parses storageSize as a numeric string into featureFlags.storageSize;
        # 128 makes it the UP-AI-KEY-128 variant, anything else the 256 GB OrinNX256.
        return {"storageSize": "256"}
    if action == "systemMetrics":
        # CONFIRMED schema: a flat top-level array of {name, value} numbers. Values are
        # FRACTIONS 0-1 like the firmware's (aikey-fw 2.2.8 uaibd/hwmon/jtop_systemMetrics.py:
        # cpu user/100, gpu load/100, RAM used/tot, rounded to 4 places) and no NPU entry
        # (the real key reports CPU, GPU, Memory only). Protect stores the numbers untouched
        # (M/606323:29-45) and its one consumer reads PsqlUsage against 0.8
        # (M/617109:19-21), so fractions are the scale it expects; the UI never reads them.
        raw = collect_system_metrics()
        ram_total = raw["ram"]["totalKilobytes"]
        ram_avail = raw["ram"]["availableKilobytes"]
        ram_used = ((ram_total - ram_avail) / ram_total) if ram_total else 0.0
        metrics = [
            {"name": "CPU", "value": round((raw["cpu"]["load"] or 0.0) / 100, 4)},
            {"name": "GPU", "value": round((raw["gpu"]["load"] or 0.0) / 100, 4)},
            {"name": "Memory", "value": round(ram_used, 4)},
        ]
        log.info("systemMetrics -> %s", metrics)
        return {"systemMetrics": metrics}
    if action == "getTaskQueueInfo":
        # CONFIRMED schema: six numeric counters. `{}` fails validation silently on every
        # ~5 s poll (Protect M/691057:234-240). UI_TASK_NUM is what this key really holds
        # (queued + running), which Protect uses to pick the least-loaded processor;
        # UI_AUTO_RAM the recognizeKeyFrames share of it, UI_MANUAL_FACE_ENHANCE the
        # enhanceImage share (the Faces page "Enhance" button: Protect M/851272:188-206). We
        # run no STT queue and never enhance on our own (UI_AUTO_FACE_ENHANCE).
        stats = task_queue_stats()
        return {
            "UI_AUDIO_RAM": 0,
            "UI_AUTO_FACE_ENHANCE": 0,
            # Firmware's ui-websocketd getTaskQueueInfo also names UI_AUTO_FACE_RECOGNITION,
            # UI_AUTO_LPR_RECOGNITION and UI_TASK_FAST_LANE (strings of aikey-fw 2.2.8
            # ui-websocketd). Face/LPR: recognizeKeyFrames tasks that will run that model.
            "UI_AUTO_FACE_RECOGNITION": (stats.get("queued_face", 0)
                                         + stats.get("running_face", 0)),
            "UI_AUTO_LPR_RECOGNITION": (stats.get("queued_lpr", 0)
                                        + stats.get("running_lpr", 0)),
            # Fast lane = on-demand work that bypasses the queue. Ours are the on-demand
            # RequestAI summary threads, counted in _fast_lane while in flight.
            "UI_TASK_FAST_LANE": _fast_lane_count(),
            "UI_AUTO_RAM": stats.get("queued_ram", 0) + stats.get("running_ram", 0),
            "UI_AUTO_STT": 0,
            "UI_MANUAL_FACE_ENHANCE": (stats.get("queued_enhance", 0)
                                       + stats.get("running_enhance", 0)),
            "UI_TASK_NUM": stats["queued"] + stats["running"],
        }
    if action == "recognizeKeyFrames":
        # Confirmed live: the console asks us to deep-analyse a clip, then we
        # POST the result (multipart field `ram`) to the console's resUrl.
        # Face/LPR share of the queue for getTaskQueueInfo: the models run when _wanted_metas
        # has entries for them (same gates as _handle_recognize).
        wanted = _wanted_metas(body or {})
        extra = tuple(g for g in ("face", "lpr") if g in wanted)
        fetch = _start_prefetch(device_info["console_host"], device_info["mac_nosep"], body)
        if not _submit_task("recognizeKeyFrames", _handle_recognize,
                            (device_info["console_host"], device_info["mac_nosep"], body,
                             fetch), ram=True,
                            **({"extra_groups": extra} if extra else {})) and fetch:
            fetch.cancel()
    elif action == "fastReverification":
        # Second Stage Verification from per-thumbnail JPEGs (or a clip when Protect falls
        # back), aikey-fw 2.2.8 syswrapper.sh:3324-3640. Same queue as the other model work.
        _submit_task("fastReverification", _handle_fast_reverification,
                     (device_info["console_host"], device_info["mac_nosep"], body))
    elif action == "enhanceImage":
        # Faces page "Enhance" (Protect M/851272:188-206): ack now, fetch/enhance/upload the
        # face JPEG from the queue; Protect learns the outcome from the resUrl upload.
        _submit_task("enhanceImage", _handle_enhance_image,
                     (device_info["console_host"], device_info["mac_nosep"], body),
                     group="enhance")
    elif action == "enhanceImageGroup":
        # Faces page group portrait (firmware download_image_to_enhance_group_from_json,
        # SW:1672-1704): same queue and UI_MANUAL_FACE_ENHANCE share as enhanceImage.
        _submit_task("enhanceImageGroup", _handle_enhance_image_group,
                     (device_info["console_host"], device_info["mac_nosep"], body),
                     group="enhance")
    elif action == "RequestAI" and str((body or {}).get("targetUri") or "").endswith(
            DESCRIBE_URI):
        # Deep-mode describe (M/851272:218-255, timeoutMs 30 s): runs on the shared worker
        # pool; in deep mode RAM is off (M/851272:119) so that queue holds only these.
        _submit_task("describe", _handle_describe,
                     (device_info["console_host"], device_info["mac_nosep"], body))
    elif action == "RequestAI" and str((body or {}).get("targetUri") or "").endswith(
            EMBED_URI):
        _submit_task("generateEmbeddings", _handle_generate_embeddings,
                     (device_info["console_host"], device_info["mac_nosep"], body))
    elif action == "RequestAI" and str((body or {}).get("targetUri") or "").endswith(
            VLM_ON_DEMAND_URI):
        # On-demand AI Summary (visionToLanguage, Protect M/309156:20-47). requestAi gives up
        # after 15 s (M/490990:11-24), so it gets its own thread: queued behind a burst of
        # recognizeKeyFrames clips on the default single worker it would always time out.
        threading.Thread(target=_fast_lane_run,
                         args=(_on_demand_summary, device_info["console_host"],
                               device_info["mac_nosep"], body),
                         daemon=True, name="diykey-vlm-on-demand").start()
    elif action == "RequestAI" and str((body or {}).get("targetUri") or "").endswith(
            SEGMENT_URI):
        # Tap-to-find (fetchSegments, M/370427). Same 15 s requestAi limit as the summary.
        threading.Thread(target=_fast_lane_run,
                         args=(_segment_video, device_info["console_host"],
                               device_info["mac_nosep"], body),
                         daemon=True, name="diykey-segment-video").start()
    elif action == "RequestAI":
        # targetUri (":7788/v1/models/<model>/inference") means "call your own
        # local gateway": there is no console callback into :7788.
        _submit_task("RequestAI", _post_request_ai_result, (device_info["mac_nosep"], body))
    elif action in ("factoryReset", "restoreDefault"):
        _factory_reset(action)
    elif action in ACK_ONLY_ACTIONS:
        # The firmware handles these (syswrapper/ui-websocketd) but they mean nothing to a
        # software key; an empty ack keeps Protect's RPC from failing. Said once per name.
        if action not in _ack_only_logged:
            _ack_only_logged.add(action)
            log.info("action %r is ack-only (nothing to do on DIY-Key)", action)
    elif action not in HANDLED_ACTIONS:
        # The firmware's router logs "Unknown action: %r" and fails the request with
        # errorCode -95 (EOPNOTSUPP; aikey-fw 2.2.8 ui-websocketd aikey_route_on_request
        # falls through with w20=-95, echoed by ucpv4_echo_response). An empty ack would
        # tell Protect an unimplemented feature worked.
        log.warning("Unknown action: %r (replying errorCode %d)", action, UNKNOWN_ACTION_CODE)
        raise ActionError(UNKNOWN_ACTION_CODE, f"Unknown action: {action}")
    return {}


def _second_verifier_result(params: dict) -> dict:
    """Same placeholder shape as stage4_modelserver/model_server.py's
    _second_verifier_result. Schema confirmed from the real
    models/second_verifier.py; values stay placeholders until real inference."""
    rows = []
    for entry in params.get("thumbnailMeta") or []:
        ts = entry.get("ts")
        for roi in entry.get("roi", []):
            object_type = str(roi.get("objectType") or "").lower()
            rows.append({
                "thumbnailMs": ts,
                "trackerID": int(roi.get("trackerID", roi.get("trackerId", -1))),  # schema: number
                "objectType": object_type,
                "isValidDetection": True,
                "detectedAs": object_type,
                "modelRunStatus": "skipped",
                "iouScore": None,
                "detectionConfidence": 0.0,
                "detectionConfidenceList": [],
            })
    return {
        "cameraId": params.get("camera", ""),
        "eventId": params.get("event", ""),
        "status": "success",
        "scoreThresholdApplied": params.get("score_threshold", 0.5),
        "preProcessMs": 0,
        "inferMs": 0,
        "timeElapsedMs": 0,
        "verificationResults": rows,
    }


def _callback_headers(mac_nosep):
    return {
        "x-ident": mac_nosep.upper(),
        "x-type": "UP-AI-KEY",
        "x-sysid": "0xa5f0",
    }


def _verify_from_clip(params, mac_nosep):
    """Second Stage Verification on the event's own frames: fetch reqUrl (the export around
    the thumbnails) as MP4, grab each thumbnail time, classify each box. None when there is
    no tagger model or the clip/frames can't be had, so the placeholder goes back instead."""
    tagger = _get_tagger()
    req_url = params.get("reqUrl")
    if not tagger or not req_url or not STATE_DIR:
        return None
    t0 = time.monotonic()
    moments = sorted({e.get("ts") for e in params.get("thumbnailMeta") or [] if e.get("ts")}
                     | set(params.get("thumbnailMs") or []))
    if not moments:
        return None
    start = dict(parse_qsl(urlsplit(req_url).query)).get("start")
    start = int(start) if start and start.isdigit() else moments[0]
    clips_dir = os.path.join(STATE_DIR, "clips")
    clip = clip_fetch.fetch_clip(clip_fetch.mp4_url("", CONSOLE_INTERNAL_API_PORT, req_url),
                                 _callback_headers(mac_nosep), (CERT_PATH, KEY_PATH), clips_dir,
                                 f"{params.get('event') or 'event'}-verify")
    if not clip:
        return None
    frames = clip_fetch.extract_frames(clip, start, moments)
    clip_fetch.prune(clips_dir)
    if not frames:
        return None
    try:
        result = second_verifier.verify(tagger, frames, params)
    except Exception as e:  # noqa: BLE001 -- fall back to the placeholder
        log.warning("verification failed for event %s: %r", params.get("event"), e)
        return None
    pre_ms = int((time.monotonic() - t0) * 1000) - result["inferMs"]
    result["preProcessMs"] = max(0, pre_ms)
    result["timeElapsedMs"] = result["preProcessMs"] + result["inferMs"]
    return result


def _post_request_ai_result(mac_nosep, task_body):
    target_uri = task_body.get("targetUri", "")
    res_url = task_body.get("resUrl")
    payload = task_body.get("payload", {})
    action = payload.get("action", "classify")
    params = payload.get("params", {})

    model_name = "unknown"
    m = re.search(r"/v1/models/([^/]+)/inference", target_uri)
    if m:
        model_name = m.group(1)

    if model_name == "second_verifier_mlabel":
        result = _verify_from_clip(params, mac_nosep) or _second_verifier_result(params)
    else:
        log.warning("RequestAI for unhandled model %r, sending empty result", model_name)
        result = {}

    response_body = {
        "model": model_name,
        "action": action,
        "inference_time_ms": 0.0,
        "result": result,
    }

    if not res_url:
        log.warning("RequestAI task had no resUrl, nothing to post (result=%s)", response_body)
        return
    try:
        resp = requests.post(res_url, headers=_callback_headers(mac_nosep), json=response_body,
                             cert=(CERT_PATH, KEY_PATH), verify=False, timeout=15)
        log.info("POST %s -> %s: %s", res_url, resp.status_code, resp.text[:2000])
    except requests.RequestException as e:
        log.warning("POST %s failed: %r", res_url, e)


def _handle_fast_reverification(console_host, mac_nosep, body):
    """fastReverification (aikey-fw 2.2.8 syswrapper.sh:3324-3640): re-classify the boxes of
    thumbnailMeta and POST the verdict as multipart field `reverification` to resUrl.

    reqUrl is a list of image URLs, one per thumbnailMs (fast path, downloaded as-is), or a
    single clip URL (fallback: grab each thumbnail time from it). The firmware drops a
    thumbnail whose download fails or that has no thumbnailMeta entry, aborts when the list
    and thumbnailMs lengths differ, and hands the frames to its ReverificationProc model;
    here the frames go through second_verifier (the same boxes-vs-detector logic as
    RequestAI/second_verifier_mlabel) and the flat reverificationInferenceSchema result is
    posted. With no detector or no frames at all the placeholder (every box left valid,
    modelRunStatus "skipped") goes back, as for RequestAI, so Protect's task still ends."""
    body = body or {}
    camera_id, event_id = body.get("camera"), body.get("event")
    res_url, req_url = body.get("resUrl"), body.get("reqUrl")
    moments = list(body.get("thumbnailMs") or [])
    metas = {e.get("ts"): e for e in body.get("thumbnailMeta") or []}
    if not res_url or not STATE_DIR:
        log.warning("fastReverification without resUrl (or state dir): %s", body)
        return
    if isinstance(req_url, list) and len(req_url) != len(moments):
        log.warning("fastReverification %s: reqUrl has %d images but thumbnailMs %d; aborting "
                    "like the firmware", event_id, len(req_url), len(moments))
        return
    params = {"camera": camera_id, "event": event_id,
              "thumbnailMeta": [metas[ts] for ts in moments if ts in metas] if moments
              else list(metas.values()),
              "score_threshold": body.get("score_threshold", 0.8)}
    headers, cert = _callback_headers(mac_nosep), (CERT_PATH, KEY_PATH)
    clips_dir = os.path.join(STATE_DIR, "clips")
    frames = {}
    if isinstance(req_url, list):
        out_dir = os.path.join(clips_dir, f"{re.sub(r'[^A-Za-z0-9_.-]', '_', str(event_id))}"
                                          "-reverify-images")
        for ts, url in zip(moments, req_url):
            if not url or ts not in metas:
                continue
            path = clip_fetch.fetch_image(_abs_url(console_host, url), headers, cert,
                                          os.path.join(out_dir, f"{camera_id}_{ts}.jpg"))
            if path:
                frames[ts] = path
    elif isinstance(req_url, str) and req_url and moments:
        clip = clip_fetch.fetch_clip(clip_fetch.mp4_url(console_host, CONSOLE_INTERNAL_API_PORT,
                                                        req_url),
                                     headers, cert, clips_dir, f"{event_id}-reverify")
        if clip:
            start = body.get("start")
            frames = clip_fetch.extract_frames(clip, int(start) if start is not None
                                               else min(moments), sorted(moments))
    clip_fetch.prune(clips_dir)
    result = None
    tagger = _get_tagger()
    if tagger:
        t0 = time.monotonic()
        try:
            result = second_verifier.verify(tagger, frames, params)
            result["timeElapsedMs"] = int((time.monotonic() - t0) * 1000)
        except Exception as e:  # noqa: BLE001 -- fall back to the placeholder
            log.warning("fastReverification failed for event %s: %r", event_id, e)
    result = result or _second_verifier_result(params)
    log.info("fastReverification %s: %d/%d frames, %d verdicts", event_id, len(frames),
             len(moments), len(result["verificationResults"]))
    url = _abs_url(console_host, res_url)
    files = {"reverification": ("reverification.json", json.dumps(result), "application/json")}
    try:
        resp = requests.post(url, headers=headers, files=files, cert=cert, verify=False,
                             timeout=30)
        log.info("POST %s [reverification] -> %s: %s", url, resp.status_code, resp.text[:500])
    except requests.RequestException as e:
        log.warning("POST %s failed: %r", url, e)


_tagger = None
_tagger_lock = threading.Lock()
_tagger_warned = False


def _footprint_mb():
    """This process's memory in MB as Activity Monitor counts it (macOS phys_footprint, which
    includes compressed and swapped pages), or None. Live 2026-11-10 15:57: Mac A's main
    process was 13 GB footprint but 2 MB RSS on a swapping 16 GB Mac, so RSS hides it."""
    try:
        if sys.platform == "darwin":
            import ctypes
            buf = ctypes.create_string_buffer(512)
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            if libproc.proc_pid_rusage(os.getpid(), 2, buf) != 0:  # RUSAGE_INFO_V2
                return None
            # rusage_info_v2: 16-byte uuid, then uint64s; ri_phys_footprint is the 8th.
            return struct.unpack_from("<Q", buf.raw, 16 + 7 * 8)[0] / 2 ** 20
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:  # noqa: BLE001 -- only for the log line
        return None
    return None


def _mem_note(before):
    """", +1700 MB (process 5200 MB)" for a warm-up log line; "" when unknown. Other threads
    (tasks, the other warm-ups' caches) can add to the delta."""
    after = _footprint_mb()
    if before is None or after is None:
        return ""
    return f", {after - before:+.0f} MB (process {after:.0f} MB)"


def _warm_one(name, fn):
    try:
        t0, m0 = time.monotonic(), _footprint_mb()
        fn()
        log.info("%s warm-up took %.1f s%s", name, time.monotonic() - t0, _mem_note(m0))
    except Exception as e:  # noqa: BLE001 -- the first task will load it or report it
        log.warning("%s warm-up failed: %r", name, e)


def _warm_models():
    """Load every enabled model at startup, one at a time, RAM++ (the slowest) last. Live
    2026-11-09 15:45: an on-demand summary loaded SmolVLM first, took 18 s against Protect's
    15 s limit and got "Invalid upload token"."""
    import cv2
    import numpy as np
    blank = np.zeros((256, 256, 3), np.uint8)
    log.info("warm-up start: process %s MB", f"{_footprint_mb() or 0:.0f}")
    if clip_embed.enabled():
        _warm_one("CLIP", lambda: clip_embed.text_embed("warm up"))
        # The not-an-animal labels: ~90 text encodings, otherwise paid by the first animal crop.
        _warm_one("CLIP labels", lambda: clip_embed.label_matrix(
            ram_tagger.ANIMAL_LABELS + ram_tagger.NOT_ANIMAL))
    if vlm_describe.enabled():
        _warm_one("VLM", lambda: vlm_describe.describe(blank))
    if reid_model.enabled():
        _warm_one("re-ID", lambda: reid_model.embed(blank))
    if _truthy(os.environ.get("DIYKEY_FACE")):
        _warm_one("face", lambda: (face_recog._models.infer_fn() or (lambda _: None))(blank))
    if _truthy(os.environ.get("DIYKEY_FACE_ENHANCE")):
        _warm_one("face enhance",
                  lambda: face_enhance.enhance(cv2.imencode(".jpg", blank)[1].tobytes()))
    if _truthy(os.environ.get("DIYKEY_LPR")) and lpr.lpr_available():
        _warm_one("LPR", lambda: lpr._alpr_reader(blank))
    _warm_ram_plus()


def _warm_ram_plus():
    """Load RAM++ at startup with one blank frame, so the first event doesn't pay for it. Live
    2026-11-09: 1 min 41 s from CoreML's cache, inside the first tagged event after a restart."""
    try:
        m0 = _footprint_mb()  # with the tagger: YOLO's session is small next to RAM++
        tagger = _get_tagger()
        if tagger and tagger.plus:
            import numpy as np
            t0 = time.monotonic()
            tagger.plus.tags(np.zeros((tagger.plus.SIZE, tagger.plus.SIZE, 3), np.uint8))
            log.info("RAM++ warm-up took %.0f s%s", time.monotonic() - t0, _mem_note(m0))
    except Exception as e:  # noqa: BLE001 -- the first event will load it or report it
        log.warning("RAM++ warm-up failed: %r", e)


def _get_tagger():
    """The RAM tagger for DIYKEY_RAM_MODEL, loaded once; None when unset or broken."""
    global _tagger, _tagger_warned
    path = os.environ.get("DIYKEY_RAM_MODEL")
    if not path:
        # Say it once: without a model recognizeKeyFrames has no RAM result to send,
        # which otherwise looks like successful processing in the log.
        with _tagger_lock:
            if not _tagger_warned:
                _tagger_warned = True
                log.warning("RAM tagging disabled: DIYKEY_RAM_MODEL is unset -- "
                            "recognizeKeyFrames will send no RAM result")
        return None
    with _tagger_lock:
        if _tagger is None:
            try:
                _tagger = ram_tagger.RamTagger(path)
            except Exception as e:  # noqa: BLE001 -- a bad model must not stop results
                log.warning("RAM tagger for %s failed to load: %r", path, e)
                _tagger = False
        return _tagger or None


class _StageTimer:
    """Milliseconds per named stage of one task; mark(name) closes the stage just finished."""

    def __init__(self):
        self.start = self.last = time.monotonic()
        self.stages = {}
        self.frames = 0
        self.detail = ""  # inside the tags stage: ram_tagger.prof_format

    def mark(self, name):
        now = time.monotonic()
        self.stages[name] = self.stages.get(name, 0) + (now - self.last) * 1000
        self.last = now

    def total_ms(self):
        return (time.monotonic() - self.start) * 1000


def _handle_recognize(console_host, mac_nosep, body, fetch=None):
    """_recognize, then one "timing" line per event, even if it failed. Live 2026-11-09: events
    waited up to 842 s in the queue, and that could only be worked out from unrelated log
    lines. wait = queue, total = the work after it; a stage that did not run is left out.
    With a prefetch (fetch, a Future of _fetch_media), clip= and frames= are how long those
    took on the prefetch pool, outside total; fetchwait= is the part the worker still waited."""
    timer = _StageTimer()
    try:
        _recognize(console_host, mac_nosep, body, timer, fetch)
    finally:
        objects = sorted({str(t).lower() for key in ("roiMeta", "thumbnailMeta")
                          for t in (e["roi"].get("objectType") or
                                    (e["roi"].get("attributes") or {}).get("objectType")
                                    for e in _flatten_meta(body.get(key))) if t})
        metas = [k for k in ("personMeta", "vehicleMeta", "faceMeta") if body.get(k)]
        log.info("event %s: timing ms camera=%s ramType=%s objects=%s metas=%s frames=%d "
                 "wait=%d total=%d %s", body.get("event"), body.get("camera"),
                 body.get("ramType"), ",".join(objects) or "-", ",".join(metas) or "-",
                 timer.frames, getattr(_task_ctx, "wait_ms", 0), timer.total_ms(),
                 " ".join([f"{name}={ms:.0f}" for name, ms in timer.stages.items()]
                          + ([timer.detail] if timer.detail else [])))


def _frame_moments(body):
    """(ram_moments, metas, meta_start, face_meta, meta_moments) for a video task: the key
    moments/thumbnails RAM tags, and the extra face/plate grabs."""
    start = int(body["start"])
    ram_moments = set(body.get("keyMoments") or []) | set(body.get("thumbnailMs") or [])
    metas = _wanted_metas(body)
    meta_start = _meta_start_ms(metas, start)
    face_meta = _flatten_meta(body.get("faceMeta")) if "face" in metas else []
    meta_moments = {_meta_wall_ms(e["ts"], meta_start)
                    for entries in (*metas.values(), face_meta) for e in entries}
    return ram_moments, metas, meta_start, face_meta, meta_moments


def _fetch_media(console_host, mac_nosep, body):
    """Download the task's clip and grab its frames: the console and ffmpeg part of an event,
    no models. {clip, frames, clip_ms, frames_ms}."""
    t = time.monotonic()
    clips_dir = os.path.join(STATE_DIR, "clips")
    url = clip_fetch.mp4_url(console_host, CONSOLE_INTERNAL_API_PORT, body.get("reqUrl"))
    clip = clip_fetch.fetch_clip(url, _callback_headers(mac_nosep), (CERT_PATH, KEY_PATH),
                                 clips_dir, body.get("event"))
    clip_ms = (time.monotonic() - t) * 1000
    t = time.monotonic()
    frames = {}
    if clip and body.get("start") is not None:
        ram_moments, *_, meta_moments = _frame_moments(body)
        # One ffmpeg pass list for both: the meta frames are extra grabs, not key moments.
        frames = clip_fetch.extract_frames(clip, int(body["start"]),
                                           sorted(ram_moments | meta_moments))
    return {"clip": clip, "frames": frames, "clip_ms": clip_ms,
            "frames_ms": (time.monotonic() - t) * 1000}


_prefetch_pool = None


def _start_prefetch(console_host, mac_nosep, body):
    """Start _fetch_media for a video task on the prefetch pool (DIYKEY_PREFETCH_WORKERS,
    default 2, 0 = off) and return its Future, or None. Downloads then overlap the one task
    worker's model work; the models stay one event at a time. Live 2026-11-10 08:40: an event
    waited 26.5 s on the single worker while the one ahead of it spent 39 s downloading."""
    global _prefetch_pool
    body = body or {}
    workers = _env_number("DIYKEY_PREFETCH_WORKERS", 2)
    if workers <= 0 or not body.get("reqUrl") or not STATE_DIR \
            or body.get("ramType") in ("image", "multipleImages"):
        return None
    with _task_lock:
        if _prefetch_pool is None:
            _prefetch_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="diykey-prefetch")
        pool = _prefetch_pool
    return pool.submit(_fetch_media, console_host, mac_nosep, body)


def _recognize(console_host, mac_nosep, body, timer, fetch=None):
    """Fetch the task's clip (or take the prefetched one), tag its key-moment and thumbnail
    frames, add face/plate parts when asked, and POST the result; then, for postVLM, a second
    POST with the summary. Every step is best effort: on a failure that part is left out."""
    camera_id, event_id = body.get("camera"), body.get("event")
    result = None
    parts, images, frames = {}, {}, {}
    scene = None
    req_url = body.get("reqUrl")
    if body.get("ramType") in ("image", "multipleImages") and STATE_DIR:
        # Retroactive processing (and image RAM): cropped JPEGs by imageId, no clip.
        ram_tagger.prof_reset()
        try:
            result = _recognize_images(console_host, mac_nosep, body)
        finally:
            timer.mark("tags")
            timer.detail = ram_tagger.prof_format(ram_tagger.prof_take())
    elif req_url and STATE_DIR:
        t0 = time.monotonic()
        clips_dir = os.path.join(STATE_DIR, "clips")
        media = None
        if fetch is not None:
            try:
                media = fetch.result()
            except Exception as e:  # noqa: BLE001 -- fetch again here instead
                log.warning("event %s: prefetch failed (%r); fetching now", event_id, e)
            timer.mark("fetchwait")
        if media is None:
            media = _fetch_media(console_host, mac_nosep, body)
            timer.mark("clip")  # the whole fetch; split below
            timer.stages["clip"] -= media["frames_ms"]
            timer.stages["frames"] = media["frames_ms"]
        else:
            # preProcessMs: the download and frame grabs happened before this worker started.
            t0 -= (media["clip_ms"] + media["frames_ms"]) / 1000
            timer.stages["clip"] = media["clip_ms"]
            timer.stages["frames"] = media["frames_ms"]
        clip = media["clip"]
        if clip and body.get("start") is not None:
            ram_moments, metas, meta_start, face_meta, meta_moments = _frame_moments(body)
            frames = media["frames"]
            timer.frames = len(frames)
            ram_frames = {ms: p for ms, p in frames.items() if ms in ram_moments}
            log.info("event %s: %d/%d frames (%d for face/lpr)", event_id, len(frames),
                     len(ram_moments | meta_moments), len(meta_moments - ram_moments))
            tagger = _get_tagger()
            timer.mark("load")
            if tagger and ram_frames:
                scene = scene_memory.observe(STATE_DIR, tagger, camera_id, event_id, ram_frames,
                                             _event_triggers(body))
                timer.mark("scene")
                ram_tagger.prof_reset()
                try:
                    # Only Protect's key moments/thumbnails: a keyMomentsTags row per extra
                    # face/plate grab would be a key moment Protect never asked about.
                    result = ram_tagger.build_result(
                        tagger, camera_id, event_id, ram_frames,
                        pre_ms=(time.monotonic() - t0) * 1000,
                        objects=_video_objects(body, frames), roi_meta=body.get("roiMeta"))
                    result = scene_memory.filter_result(result, scene)
                    log.info("event %s: %s", event_id, result["description"])
                except Exception as e:  # noqa: BLE001
                    log.warning("RAM tagging failed for event %s: %r", event_id, e)
                timer.mark("tags")
                timer.detail = ram_tagger.prof_format(ram_tagger.prof_take())
            if metas:
                _add_face_lpr(parts, images, metas, frames, meta_start, camera_id, event_id,
                              face_meta=face_meta)
                timer.mark("face")
        clip_fetch.prune(clips_dir)
    _post_recognize_result(console_host, mac_nosep, body.get("resUrl"), camera_id, event_id,
                           result, parts=parts or None, images=images or None)
    timer.mark("post")
    if body.get("postVLM") and vlm_describe.enabled():
        if _truthy(os.environ.get("DIYKEY_VLM_QUEUE", "1")):
            _submit_task("summary", _run_summary, (console_host, mac_nosep, body, frames, scene),
                         q=_summary_queue())
        else:
            _post_vlm_description(console_host, mac_nosep, body, frames, scene)
            timer.mark("summary")


def _run_summary(console_host, mac_nosep, body, frames, scene):
    """The summary POST on its own queue, with its own "timing" line."""
    t0 = time.monotonic()
    try:
        _post_vlm_description(console_host, mac_nosep, body, frames, scene)
    finally:
        log.info("event %s: timing ms summary camera=%s wait=%d total=%d", body.get("event"),
                 body.get("camera"), getattr(_task_ctx, "wait_ms", 0),
                 (time.monotonic() - t0) * 1000)


# Epoch ms from Sep 2001 up; anything smaller is an offset into the clip. Protect's own ts
# fields (keyMeta, thumbnailMeta, personMeta: Protect M/346308:89-100) all share one
# z.number() with nothing saying which; keyMoments/thumbnailMs are epoch ms (they seek
# against body.start), so epoch is the expected case and the offset branch a guard.
EPOCH_MS_MIN = 10 ** 12


def _flatten_meta(meta):
    """personMeta/vehicleMeta entries with one roi each. The schema makes roi one object
    (Protect M/346308:76-80, 96-97) but Protect's own linkAiKeyFacesToPersons walks a
    personMeta roi as an array (M/321055:78-90); accept both."""
    out = []
    for entry in meta or []:
        if not isinstance(entry, dict) or entry.get("ts") is None:
            continue
        roi = entry.get("roi")
        if isinstance(roi, list):
            out.extend({**entry, "roi": r} for r in roi if isinstance(r, dict))
        elif isinstance(roi, dict):
            out.append(entry)
    return out


_ROI_PERSON = ("person",)
_ROI_VEHICLE = ("vehicle", "car", "truck", "bus", "motorcycle")


def _roi_derived_metas(body):
    """{"person": [...], "vehicle": [...]} personMeta/vehicleMeta-shaped entries built from
    roiMeta, only for ramType 'videoWithRecognition' and only for the kinds the body lacks.

    Live 21:04 2026-11-07, Protect 7.3.74 (doorbell camera under "Legacy Camera
    Enhancement"): videoWithRecognition carries only roiMeta (and thumbnailMeta), no
    personMeta/vehicleMeta/faceMeta, each roi {name, coord, trackerID, trackerId,
    attributes: {objectType}|None, confidence, objectType}. An explicit personMeta/
    vehicleMeta always wins; the AI-setting gates stay in _wanted_metas."""
    if body.get("ramType") != "videoWithRecognition":
        return {}
    out = {}
    for kind, key, types in (("person", "personMeta", _ROI_PERSON),
                             ("vehicle", "vehicleMeta", _ROI_VEHICLE)):
        if body.get(key):
            continue
        found = []
        for e in _flatten_meta(body.get("roiMeta")):
            r = e["roi"]
            otype = r.get("objectType") or (r.get("attributes") or {}).get("objectType")
            if otype in types and r.get("coord") is not None:
                tid = r.get("trackerId", r.get("trackerID"))
                found.append({"ts": e["ts"], "roi": {"coord": r["coord"], "objectType": otype,
                                                     "trackerId": tid}})
        if found:
            out[kind] = found
    if out:
        log.info("event %s: derived %d person / %d vehicle rois from roiMeta",
                 body.get("event"), len(out.get("person", ())), len(out.get("vehicle", ())))
    return out


def _wanted_metas(body):
    """{"face": personMeta, "lpr": vehicleMeta} for the features this task should run.

    Each needs the DIYKEY_* key (what we advertise) and Protect's AI setting: the firmware
    deletes personMeta unless enableFaceRecognize (or pose) is on and vehicleMeta unless
    enableLprRecognize is (aikey-fw 2.2.8 syswrapper.sh:2456-2482), and Protect only adds them
    for cameras with the feature enabled on this processor (M/851272:421-425). faceMeta
    (Protect's own face boxes) is passed separately as face_meta (see _handle_recognize)."""
    metas = {}
    derived = _roi_derived_metas(body)
    # Live 2026-11-07, Protect 7.3.74: Settings > Intelligence > "Legacy Camera Enhancement"
    # (face + plates for older cameras) was ON for All cameras with the Face Recognition
    # legal permission ticked, yet changeAiInferAgentSettings kept sending
    # enableFaceRecognize False (flipped from True at 21:01:36 while that card was edited;
    # re-sent False 21:05:30, 21:10:47). Tasks for those cameras arrive with ramType
    # 'videoWithRecognition' (21:04, doorbell 6a568fa0): Protect already chose them for
    # recognition, so the per-processor flags are not consulted there. Other ramTypes keep
    # the 7.3.70 gates (explicit metas + enable flags).
    legacy = body.get("ramType") == "videoWithRecognition"
    if _truthy(os.environ.get("DIYKEY_FACE")) and (
            legacy or ai_setting_enabled("enableFaceRecognize")):
        person = _flatten_meta(body.get("personMeta"))
        if not person and "person" in derived:
            person = derived["person"]
        if person:
            metas["face"] = person
    if _truthy(os.environ.get("DIYKEY_LPR")) and (
            legacy or ai_setting_enabled("enableLprRecognize")):
        vehicle = _flatten_meta(body.get("vehicleMeta"))
        if not vehicle and "vehicle" in derived:
            vehicle = derived["vehicle"]
        if vehicle:
            metas["lpr"] = vehicle
    return metas


def _meta_start_ms(metas, start):
    """start_ms for face_recog/lpr: None when the meta ts are epoch ms, so their
    clockBestWall is the ts itself, the same epoch ms RAM sends as keyMomentMs; the clip
    start when they look like offsets (then wall = start + ts). Unverified live which one
    Protect sends; decided per event so a snapshot's wall never mixes the two."""
    ts = [float(e["ts"]) for entries in metas.values() for e in entries]
    return None if not ts or min(ts) >= EPOCH_MS_MIN else int(start)


def _meta_wall_ms(ts, meta_start):
    return int(float(ts)) + (meta_start or 0)


def _add_face_lpr(parts, images, metas, frames, meta_start, camera_id, event_id,
                  face_meta=None):
    """Run face_recog/lpr on the grabbed frames and add their parts and JPEGs. face_meta is
    Protect's own camera face boxes (body faceMeta, personMeta-shaped, roi flattened)."""
    import cv2
    bgr = {}  # ts as Protect sent it (both modules look frames up by that) -> BGR frame
    for entries in (*metas.values(), face_meta or []):
        for e in entries:
            if e["ts"] in bgr:
                continue
            path = frames.get(_meta_wall_ms(e["ts"], meta_start))
            img = cv2.imread(path) if path else None
            if img is not None:
                bgr[e["ts"]] = img
    for name, build in (("face", face_recog.build_face_result), ("lpr", lpr.build_lpr)):
        if name not in metas:
            continue
        try:
            if name == "face" and face_meta:
                try:
                    out = build(camera_id, event_id, bgr, metas[name], start_ms=meta_start,
                                face_meta=face_meta)
                except TypeError:  # face_recog without the face_meta keyword yet
                    out = build(camera_id, event_id, bgr, metas[name], start_ms=meta_start)
            else:
                out = build(camera_id, event_id, bgr, metas[name], start_ms=meta_start)
        except Exception as e:  # noqa: BLE001 -- the other parts still go
            log.warning("%s failed for event %s: %r", name, event_id, e)
            continue
        if not out:
            log.info("event %s: no %s result", event_id, name)
            continue
        part, jpegs = out
        parts[name] = part
        _merge_images(images, jpegs, name, event_id)


def _merge_images(images, new, source, event_id):
    """Add `new` JPEGs ({key: bytes}) to `images` without replacing one already there: the
    multipart field is the snapshot trackerID, and face (+100000) / plate (+200000) ids are
    offset to stay clear of RAM's tracker ids, so a collision means an id scheme broke."""
    for key, jpeg in (new or {}).items():
        key = str(key)
        if key in images:
            log.warning("event %s: %s image %s collides with an earlier part's; kept the first",
                        event_id, source, key)
            continue
        images[key] = jpeg


def _vlm_frame(body, frames):
    """The still for the auto summary: the firmware's VLM agent uses the RAM key frame, else
    the first thumbnail frame, else the first key-moment frame (aikey-fw 2.2.8 ui-vlm-agent
    app.py:343-348). We keep no "best" frame, so: first thumbnail, else the middle key moment."""
    for ms in body.get("thumbnailMs") or []:
        if ms in frames:
            return frames[ms]
    moments = [ms for ms in sorted(body.get("keyMoments") or []) if ms in frames]
    return frames[moments[len(moments) // 2]] if moments else None


# What triggered the event, for the auto summary. The firmware's auto summary gets no object
# type, box or history (fw 2.2.8 ui-vlm-agent app.py: one fixed prompt, one frame); Protect's Deep
# Understanding is the precedent for describing a crop of one object. Live 2026-11-08 Driveway:
# passing cars were summarised as "Red car parked in driveway" from the whole frame.
VLM_CROP_SCALE = 3.0  # Driveway's boxes land about 1 s off fast cars, so a tight crop misses them
VLM_EDGE_MARGIN = 20      # on 0..1000: a box this close to the frame edge is cut off or too near
VLM_MAX_BOX_AREA = 0.4    # fraction of the frame: a box bigger than this is right at the lens
VLM_EDGE_VEHICLE_SCALE = 1.5  # see _vlm_event_crop
_VLM_KIND = {"person": "person", "vehicle": "vehicle", "car": "vehicle", "truck": "vehicle",
             "bus": "vehicle", "motorcycle": "vehicle", "bicycle": "vehicle", "animal": "animal",
             "dog": "animal", "cat": "animal", "bird": "animal"}
_VLM_SUBJECT = {"person": ("person", "the person"), "vehicle": ("vehicle", "it"),
                "animal": ("animal", "it")}


def _valid_coord(coord):
    """[x, y, w, h] floats (0..1000) with a real area, or None."""
    try:
        x, y, w, h = (float(v) for v in coord[:4])
    except (TypeError, ValueError):
        return None
    return [x, y, w, h] if w > 0 and h > 0 else None


def _event_triggers(body):
    """[{ts, coord, objectType, trackerID}]: the objects that triggered this event, from
    thumbnailMeta, else roiMeta (roi a dict or a list; coord [x, y, w, h] on 0..1000)."""
    for key in ("thumbnailMeta", "roiMeta"):
        found = []
        for entry in _flatten_meta(body.get(key)):
            roi = entry["roi"]
            coord = _valid_coord(roi.get("coord"))
            try:
                ts = int(float(entry["ts"]))
            except (TypeError, ValueError):
                continue
            if coord:
                otype = roi.get("objectType") or (roi.get("attributes") or {}).get("objectType")
                found.append({"ts": ts, "coord": coord,
                              "objectType": str(otype).lower() if otype else None,
                              "trackerID": roi.get("trackerID", roi.get("trackerId"))})
        if found:
            return found
    return []


def _vlm_box_is_clear(coord):
    """False for a box touching the frame edge or filling most of it. Live 2026-11-08 19:36
    Doorbell: the largest box [41, 111, 544, 880] was the walker right beside the camera, blown
    out by the IR light with the head blurred, and the summary said "walking into a dark room";
    the standing frame a second earlier had a clear box."""
    return not _vlm_box_at_edge(coord) and coord[2] * coord[3] <= VLM_MAX_BOX_AREA * 1000 * 1000


def _vlm_box_at_edge(coord):
    x, y, w, h = coord
    return (x < VLM_EDGE_MARGIN or y < VLM_EDGE_MARGIN or x + w > 1000 - VLM_EDGE_MARGIN
            or y + h > 1000 - VLM_EDGE_MARGIN)


def _vlm_trigger(triggers, frames):
    """The trigger box to describe, from those we have a frame for: a person if there is one,
    then the largest clear box, or the largest of any if none is clear. None without a frame."""
    usable = [t for t in triggers if t["ts"] in frames]
    group = [t for t in usable if t["objectType"] == "person"] or usable
    clear = [t for t in group if _vlm_box_is_clear(t["coord"])]
    return max(clear or group, key=lambda t: t["coord"][2] * t["coord"][3], default=None)


def _vlm_event_crop(img, coord, object_type=None):
    """The box expanded to VLM_CROP_SCALE times its size about its centre, clamped to the frame
    and below the timestamp band the whole-frame path crops off; None if nothing is left.

    A vehicle box at the frame edge gets VLM_EDGE_VEHICLE_SCALE and keeps the band. Live
    2026-11-09 07:49 Driveway: a garbage truck at the top left, box [0, 8, 212, 201]; the 3x crop
    took in the parked red car and the band cut the truck in half, and the summary said "A red
    car driving down the road"."""
    h, w = img.shape[:2]
    edge_vehicle = _VLM_KIND.get(object_type) == "vehicle" and _vlm_box_at_edge(coord)
    scale = VLM_EDGE_VEHICLE_SCALE if edge_vehicle else VLM_CROP_SCALE
    top = 0 if edge_vehicle else int(h * vlm_describe.CROP_TOP_FRAC)
    x, y, bw, bh = coord[0] * w / 1000, coord[1] * h / 1000, coord[2] * w / 1000, coord[3] * h / 1000
    cx, cy, half_w, half_h = x + bw / 2, y + bh / 2, bw * scale / 2, bh * scale / 2
    x0, x1 = max(0, int(cx - half_w)), min(w, int(cx + half_w))
    y0, y1 = max(top, int(cy - half_h)), min(h, int(cy + half_h))
    return img[y0:y1, x0:x1] if x1 - x0 >= 8 and y1 - y0 >= 8 else None


def _vlm_context(trigger):
    noun, pronoun = _VLM_SUBJECT.get(_VLM_KIND.get(trigger["objectType"]), ("object", "it"))
    return (f"This event was triggered by a moving {noun}. Describe {pronoun} and what "
            f"{'they are' if noun == 'person' else 'it is'} doing. "
            "Ignore parked or stationary objects.")


def _vlm_event_subject(body, frames):
    """(crop, context) for the object that triggered the event, or (None, None) to keep the
    firmware's whole-frame summary (setting off, no usable box, or no frame for it)."""
    import cv2
    if not _truthy(os.environ.get("DIYKEY_VLM_EVENT_CONTEXT", "1")):
        return None, None
    trigger = _vlm_trigger(_event_triggers(body), frames)
    img = cv2.imread(frames[trigger["ts"]]) if trigger else None
    crop = _vlm_event_crop(img, trigger["coord"], trigger["objectType"]) if img is not None else None
    return (crop, _vlm_context(trigger)) if crop is not None else (None, None)


def _post_vlm_description(console_host, mac_nosep, body, frames, scene=None):
    """postVLM: the AI Key's VLM agent posts a second multipart after the RAM result, with
    only ram={eventId, status, description} (ui-vlm-agent app.py:376-405). Protect reads a
    ram without keyMomentsTags as a description update to the event (M/315148:99-105,
    M/618986:72-74 -> saveRamDescriptionEnhancement M/543268:18-29). cameraId is added: the
    firmware omits it, but Protect's description schema requires it (M/315148:99-104).

    Summarises the triggering object when the task names one (see _vlm_event_subject), and
    `scene` (scene_memory) adds what changed and what to leave out. Only the description
    string differs: no fields are added."""
    import cv2
    camera_id, event_id = body.get("camera"), body.get("event")
    img, context = _vlm_event_subject(body, frames)
    if img is not None:
        text = vlm_describe.describe(img, context=context, crop_top=False)
    else:
        path = _vlm_frame(body, frames)
        img = cv2.imread(path) if path else None
        if img is None:
            log.info("event %s: postVLM but no frame to describe", event_id)
            return
        context = scene_memory.ignore_context(scene)
        text = vlm_describe.describe(img, context=context) if context else vlm_describe.describe(img)
    if not text:
        log.info("event %s: postVLM but no description", event_id)
        return
    text = scene_memory.summary_text(text, scene)
    log.info("event %s: summary %r (context %r)", event_id, text, context)
    _post_recognize_result(console_host, mac_nosep, body.get("resUrl"), camera_id, event_id,
                           {"cameraId": camera_id, "eventId": event_id, "status": "success",
                            "description": text})


def _abs_url(console_host, url):
    """A console-relative callback URL made absolute on the internal API port."""
    return url if url.startswith("http") else \
        f"https://{console_host}:{CONSOLE_INTERNAL_API_PORT}{url}"


def _handle_enhance_image(console_host, mac_nosep, body):
    """enhanceImage: fetch reqUrl (the face JPEG), enhance it, upload to resUrl.

    The upload is the firmware's upload_image (aikey-fw 2.2.8 syswrapper.sh:3716-3735):
    multipart file + camera + type=face + smartDetectObject, which Protect's
    /aiprocessors/image/enhanced requires (400 without smartDetectObject, type != face or no
    file: M/761379:150-170). Like the firmware, the original goes up when enhancement is off or
    fails (syswrapper.sh:1623-1628) -- face_enhance.enhance already returns it on failure."""
    body = body or {}
    req_url, obj = body.get("reqUrl"), body.get("smartDetectObject")
    res_url = body.get("resUrl") or "/internal/aiprocessors/image/enhanced"
    if not req_url or not obj or not STATE_DIR:
        log.warning("enhanceImage without reqUrl/smartDetectObject (or state dir): %s", body)
        return
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(body.get("imageId") or obj))
    got = _fetch_and_enhance(console_host, mac_nosep, req_url, name)
    if not got:
        return
    original, jpeg = got
    files = {
        "file": (f"{name}.jpg", jpeg, "image/jpeg"),
        "camera": (None, str(body.get("camera") or "")),
        "type": (None, "face", "text/plain"),
        "smartDetectObject": (None, str(obj), "text/plain"),
    }
    _post_enhanced("enhanceImage", obj, console_host, mac_nosep, res_url, files, original, jpeg)


def _fetch_and_enhance(console_host, mac_nosep, req_url, name):
    """(original, jpeg) for the face image at req_url, or None when the fetch fails."""
    path = os.path.join(STATE_DIR, "clips", "enhance", f"{name}.jpg")
    if not clip_fetch.fetch_image(_abs_url(console_host, req_url), _callback_headers(mac_nosep),
                                  (CERT_PATH, KEY_PATH), path):
        return None
    try:
        with open(path, "rb") as f:
            original = f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    jpeg = face_enhance.enhance(original) if _truthy(os.environ.get("DIYKEY_FACE_ENHANCE")) \
        else original
    return original, jpeg


def _post_enhanced(kind, ident, console_host, mac_nosep, res_url, files, original, jpeg):
    url = _abs_url(console_host, res_url)
    try:
        resp = requests.post(url, headers=_callback_headers(mac_nosep), files=files,
                             cert=(CERT_PATH, KEY_PATH), verify=False, timeout=15)
        log.info("%s %s (%s, %d -> %d bytes): POST %s -> %s: %s", kind, ident,
                 "enhanced" if jpeg is not original else "original", len(original), len(jpeg),
                 url, resp.status_code, resp.text[:300])
    except requests.RequestException as e:
        log.warning("%s POST %s failed: %r", kind, url, e)


def _handle_enhance_image_group(console_host, mac_nosep, body):
    """enhanceImageGroup: the face-group portrait variant of enhanceImage.

    Firmware download_image_to_enhance_group_from_json (aikey-fw 2.2.8 syswrapper.sh:1672-1704)
    reads reqUrl, resUrl, smartDetectObjectGroup and imageUpdatedAt, downloads reqUrl and
    upload_image_group (SW:3747-3780) posts multipart file + imageUpdatedAt + type=face +
    smartDetectObjectGroup. No camera and no smartDetectObject (the single-object route needs
    both, M/761379:150-170). imageUpdatedAt is echoed verbatim so Protect can tell whether the
    group's image changed while we worked. The original goes up when enhancement is off or
    fails, as for enhanceImage (SW:1884-1888). Protect 7.3.70 has no dispatcher for this
    action (not found in M); it exists in the firmware, so it is handled for a newer Protect."""
    body = body or {}
    req_url, group = body.get("reqUrl"), body.get("smartDetectObjectGroup")
    res_url = body.get("resUrl")
    if not req_url or not group or not res_url or not STATE_DIR:
        log.warning("enhanceImageGroup without reqUrl/resUrl/smartDetectObjectGroup (or state "
                    "dir): %s", body)
        return
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"group-{group}")
    got = _fetch_and_enhance(console_host, mac_nosep, req_url, name)
    if not got:
        return
    original, jpeg = got
    files = {
        "file": (f"{name}.jpg", jpeg, "image/jpeg"),
        "imageUpdatedAt": (None, str(body.get("imageUpdatedAt") if
                                     body.get("imageUpdatedAt") is not None else "")),
        "type": (None, "face", "text/plain"),
        "smartDetectObjectGroup": (None, str(group), "text/plain"),
    }
    _post_enhanced("enhanceImageGroup", group, console_host, mac_nosep, res_url, files,
                   original, jpeg)


# Deep Understanding targets (Protect M/851272:196-255).
DESCRIBE_URI = ":7968/describe"
EMBED_URI = ":7445/generate-embeddings"
SEGMENT_URI = ":7445/segment-video"
DESCRIBE_MODEL = "diykey-vlm"


def _describe_frame(console_host, mac_nosep, payload):
    """The BGR still a describe task is about: the middle object of the first video that
    yields a frame (Protect sends `videos` [{reqUrl, objects:[{ts, coord, ...}]}] from the
    close pass and `images` [{reqUrl, thumbnailId}] from the open pass, M/851272:222-247;
    reqUrl's start/end are epoch ms like every export), else the first fetched image."""
    import cv2
    clips_dir = os.path.join(STATE_DIR, "clips")
    hdr, cert = _callback_headers(mac_nosep), (CERT_PATH, KEY_PATH)
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", str(payload.get("event") or "describe"))
    img = None
    for i, video in enumerate(payload.get("videos") or []):
        req = video.get("reqUrl")
        objs = sorted((o for o in video.get("objects") or [] if o.get("ts") is not None),
                      key=lambda o: o["ts"])
        start = dict(parse_qsl(urlsplit(req or "").query)).get("start")
        if not req or not objs or not start:
            continue
        clip = clip_fetch.fetch_clip(clip_fetch.mp4_url(console_host, CONSOLE_INTERNAL_API_PORT,
                                                        req), hdr, cert, clips_dir,
                                     f"{tag}-describe{i}")
        if not clip:
            continue
        moment = int(objs[len(objs) // 2]["ts"])
        frames = clip_fetch.extract_frames(clip, int(start), [moment])
        img = cv2.imread(frames[moment]) if moment in frames else None
        if img is not None:
            break
    if img is None:
        for i, image in enumerate(payload.get("images") or []):
            req = image.get("reqUrl")
            path = os.path.join(clips_dir, "describe", f"{tag}-{i}.jpg")
            if req and clip_fetch.fetch_image(_abs_url(console_host, req), hdr, cert, path):
                img = cv2.imread(path)
                if img is not None:
                    break
    clip_fetch.prune(clips_dir)
    return img


def _describe_labels(img):
    """ram_tagger tags (Protect's `labels`, sanitised on its side, M/402297:62) when
    DIYKEY_RAM_MODEL loads, else None. Colour/night tags are kept: they are the search
    vocabulary."""
    tagger = _get_tagger()
    if not tagger:
        return None
    try:
        labels = [t["tag"] for t in tagger.tag_frame(img)]
    except Exception as e:  # noqa: BLE001 -- labels are optional
        log.warning("describe: tagging failed: %r", e)
        return None
    return labels or None


def _handle_describe(console_host, mac_nosep, task_body):
    """RequestAI :7968/describe (deep mode): describe one representative frame and POST
    {camera, event, pass, description, labels?, descEmbedding?, model} to resUrl
    (/internal/aiprocessors/descriptions/<taskId>, schema M/761379:420-466). An empty
    description is how a failure is reported: saveDescriptions then logs "leaving session for
    retry" and completes the task instead of waiting out the 30 s timeout (M/402297:42-52).
    descEmbedding is the CLIP text embedding of the description, only with DIYKEY_CLIP=1 and
    DIYKEY_DESC_EMBED=1: Protect's column dimension for it is not in M (CLIP here is
    768-d, the same space nl_search queries use), so a mismatch would reject saves; unverified."""
    payload = (task_body or {}).get("payload") or {}
    res_url = (task_body or {}).get("resUrl")
    body = {"camera": payload.get("camera"), "event": payload.get("event"),
            "pass": payload.get("pass"), "description": "", "model": DESCRIBE_MODEL}
    if not res_url or not STATE_DIR:
        log.warning("describe without resUrl (or state dir): %s", task_body)
        return
    if not vlm_describe.enabled():
        body["failed"] = [{"reason": "AI summary is not enabled on this processor"}]
    else:
        img = _describe_frame(console_host, mac_nosep, payload)
        text = vlm_describe.describe(img) if img is not None else None
        if not text:
            body["failed"] = [{"reason": "no frame" if img is None else "no description"}]
        else:
            body["description"] = text
            labels = _describe_labels(img)
            if labels:
                body["labels"] = labels
            if clip_embed.enabled() and _truthy(os.environ.get("DIYKEY_DESC_EMBED")):
                vec = clip_embed.text_embed(text)
                if vec:
                    body["descEmbedding"] = vec
    url = _abs_url(console_host, res_url)
    try:
        resp = requests.post(url, headers=_callback_headers(mac_nosep), json=body,
                             cert=(CERT_PATH, KEY_PATH), verify=False, timeout=15)
        log.info("describe %s/%s: POST %s -> %s: %s", body["event"], body["pass"], url,
                 resp.status_code, resp.text[:300])
    except requests.RequestException as e:
        log.warning("describe POST %s failed: %r", url, e)


def _handle_generate_embeddings(console_host, mac_nosep, task_body):
    """RequestAI :7445/generate-embeddings (M/851272:212-225): payload {camera, event,
    images:[{reqUrl, thumbnailId, objectId, ...}]}; reply POSTs {camera, event, embeddings:
    [{objectId, objectType, reidEmbed[1024], model, dim}], failed:[{objectId, reason}]} to resUrl
    (schema M/761379:376-400; saveEmbeddings M/579892:21-70 keeps only vectors for person objects
    of that event and groups them into sessions). Each crop gets reid_model.embed, the same
    vector space as thumbnailTags reidEmbed. Without DIYKEY_REID_MODEL, or per crop that cannot
    be fetched/embedded, objects are reported failed so the task completes at once instead of
    timing out."""
    import cv2
    payload = (task_body or {}).get("payload") or {}
    res_url = (task_body or {}).get("resUrl")
    if not res_url:
        return
    embeddings, failed = [], []
    on = reid_model.enabled() and bool(STATE_DIR)
    hdr, cert = _callback_headers(mac_nosep), (CERT_PATH, KEY_PATH)
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", str(payload.get("event") or "embed"))
    for i, im in enumerate(payload.get("images") or []):
        oid = im.get("objectId")
        if not on:
            failed.append({"objectId": oid, "reason": "no re-ID model"})
            continue
        vec = None
        req = im.get("reqUrl")
        path = os.path.join(STATE_DIR, "clips", "embed", f"{tag}-{i}.jpg")
        if req and clip_fetch.fetch_image(_abs_url(console_host, req), hdr, cert, path):
            try:
                vec = reid_model.embed(cv2.imread(path))
            except Exception as e:  # noqa: BLE001 -- one bad crop must not sink the task
                log.warning("generateEmbeddings: embed failed for %s: %r", oid, e)
        if vec:
            embeddings.append({"objectId": oid, "objectType": im.get("objectType") or "person",
                               "reidEmbed": vec, "model": reid_model.MODEL_NAME,
                               "dim": len(vec)})
        else:
            failed.append({"objectId": oid, "reason": "no crop" if not req else "embed failed"})
    body = {"camera": payload.get("camera"), "event": payload.get("event"),
            "embeddings": embeddings, "failed": failed}
    url = _abs_url(console_host, res_url)
    try:
        resp = requests.post(url, headers=hdr, json=body, cert=cert, verify=False, timeout=15)
        log.info("generateEmbeddings: %d embedded, %d failed: POST %s -> %s",
                 len(embeddings), len(failed), url, resp.status_code)
    except requests.RequestException as e:
        log.warning("generateEmbeddings POST %s failed: %r", url, e)


SEGMENT_MIN_CONF = 0.5  # trt_engine.py segment(conf_threshold=0.5)


def _segment_results(tagger, img):
    """fetchSegmentsResponseSchema results (M/370427) for the people in a frame. Coordinates are
    0..1000 and bbox is x1,y1,x2,y2 like the firmware (trt_engine.py postprocess). The firmware
    outlines each person with SmartSegment.engine (NVIDIA-only); here the outline is the
    detector's box, so a tap selects the rectangle rather than the silhouette."""
    h, w = img.shape[:2]
    results = []
    for label, score, (x, y, bw, bh) in tagger.detect(img):
        if label != "person" or score < SEGMENT_MIN_CONF:
            continue
        x0, y0, x1, y1 = (max(0, int(x)), max(0, int(y)), min(w, int(x + bw)), min(h, int(y + bh)))
        vec = reid_model.embed(img[y0:y1, x0:x1])
        if not vec:
            continue
        box = [int(x0 * 1000 / w), int(y0 * 1000 / h), int(x1 * 1000 / w), int(y1 * 1000 / h)]
        contour = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
        results.append({"embedding": vec, "bbox": box, "contour": contour, "contours": [contour],
                        "objectType": "person"})
    return results


def _segment_video(console_host, mac_nosep, task_body):
    """RequestAI :7445/segment-video (M/126498 reidentifyFrame, firmware agent.py /segment-video):
    payload {cameraId, timestamp, videoUrl}; POST {"results": [...]} or {"error": ...} to resUrl.
    Like the firmware, only +-1 s of the +-5 s export is fetched."""
    import shutil

    import cv2
    payload = (task_body or {}).get("payload") or {}
    res_url = (task_body or {}).get("resUrl")
    if not res_url:
        return
    hdr, cert = _callback_headers(mac_nosep), (CERT_PATH, KEY_PATH)
    out, clip = {"error": "Failed to download video"}, None
    try:
        tagger = _get_tagger()
        ts, url = int(payload["timestamp"]), str(payload["videoUrl"])
        if tagger is None or not reid_model.enabled() or not STATE_DIR:
            out = {"error": "no re-ID model"}
        else:
            url = url.replace(f"start={ts - 5000}", f"start={ts - 1000}")
            url = url.replace(f"end={ts + 5000}", f"end={ts + 1000}")
            clip = clip_fetch.fetch_clip(
                clip_fetch.mp4_url(console_host, CONSOLE_INTERNAL_API_PORT, url), hdr, cert,
                os.path.join(STATE_DIR, "clips"), f"segment-{payload.get('cameraId')}-{ts}")
            frames = clip_fetch.extract_frames(clip, ts - 1000, [ts]) if clip else {}
            img = cv2.imread(frames[ts]) if ts in frames else None
            if img is not None:
                out = {"results": _segment_results(tagger, img)}
    except Exception as e:  # noqa: BLE001 -- Protect is waiting on this reply
        log.warning("segment-video failed: %r", e)
        out = {"error": "Internal server error"}
    finally:
        if clip:
            for path in (clip, clip + ".start"):
                try:
                    os.remove(path)
                except OSError:
                    pass
            shutil.rmtree(os.path.splitext(clip)[0], ignore_errors=True)
    try:
        resp = requests.post(_abs_url(console_host, res_url), headers=hdr, json=out, cert=cert,
                             verify=False, timeout=15)
        log.info("segment-video: %s -> %s", len(out.get("results", [])) if "results" in out
                 else out["error"], resp.status_code)
    except requests.RequestException as e:
        log.warning("segment-video POST failed: %r", e)


# requestAi's visionToLanguage target (Protect M/490990:11).
VLM_ON_DEMAND_URI = ":7968/on_demand_inference"


def _on_demand_frame_ms(timestamp, start):
    """Clip moment for visionToLanguage's timestamp. Protect builds the export as
    timestamp +/- 5 s (M/309156:22-30), so timestamp is epoch ms; a small value is taken as
    an offset from the export start in case another caller sends one (unverified)."""
    ts = int(float(timestamp))
    return ts if ts >= EPOCH_MS_MIN else int(start) + ts


def _summarize_clip(console_host, mac_nosep, payload):
    """(description, None) or (None, error) for an on-demand summary payload."""
    import cv2
    if not vlm_describe.enabled():
        return None, "AI summary is not enabled on this processor"
    video_url, ts = payload.get("videoUrl"), payload.get("timestamp")
    if not video_url or ts is None or not STATE_DIR:
        return None, "missing videoUrl or timestamp"
    start = dict(parse_qsl(urlsplit(video_url).query)).get("start")
    try:
        start = int(start) if start else int(float(ts)) - 5000
        moment = _on_demand_frame_ms(ts, start)
    except (TypeError, ValueError):
        return None, f"bad timestamp {ts!r}"
    clips_dir = os.path.join(STATE_DIR, "clips")
    clip = clip_fetch.fetch_clip(
        clip_fetch.mp4_url(console_host, CONSOLE_INTERNAL_API_PORT, video_url),
        _callback_headers(mac_nosep), (CERT_PATH, KEY_PATH), clips_dir,
        f"{payload.get('eventId') or 'summary'}-summary")
    if not clip:
        return None, "could not fetch the clip"
    frames = clip_fetch.extract_frames(clip, start, [moment])
    img = cv2.imread(frames[moment]) if moment in frames else None
    clip_fetch.prune(clips_dir)
    if img is None:
        return None, "no frame at the timestamp"
    text = vlm_describe.describe(img)
    return (text, None) if text else (None, "no description")


def _on_demand_summary(console_host, mac_nosep, task_body):
    """RequestAI visionToLanguage: describe the frame at payload.timestamp and POST
    {"description": text} to resUrl. That is Protect's camera-upload callback, which resolves
    the waiting request with the body only when Content-Type is exactly application/json
    (M/854445:82-90) and then parses {description: string} (M/309156:18, 50-52) -- not the
    {model, result} wrapper the :7788 models get. Failures send {"error": ...}, which fails
    that parse, so Protect reports the summary as failed instead of waiting out 15 s."""
    res_url = task_body.get("resUrl")
    payload = task_body.get("payload") or {}
    try:
        text, error = _summarize_clip(console_host, mac_nosep, payload)
    except Exception as e:  # noqa: BLE001 -- Protect is waiting for an answer
        log.warning("on-demand summary failed: %r", e)
        text, error = None, "summary failed"
    reply = {"description": text} if text else {"error": error}
    log.info("on-demand summary camera %s at %s: %s", payload.get("cameraId"),
             payload.get("timestamp"), reply)
    if not res_url:
        log.warning("on-demand summary had no resUrl, nothing to post")
        return
    url = _abs_url(console_host, res_url)
    try:
        resp = requests.post(url, data=json.dumps(reply),
                             headers={**_callback_headers(mac_nosep),
                                      "Content-Type": "application/json"},
                             cert=(CERT_PATH, KEY_PATH), verify=False, timeout=15)
        log.info("POST %s -> %s: %s", url, resp.status_code, resp.text[:300])
    except requests.RequestException as e:
        log.warning("POST %s failed: %r", url, e)


def _video_objects(body, frames):
    """Tracked objects of a video RAM task, cropped out of the frame at their thumbnail time."""
    import cv2
    objects = []
    for entry in body.get("thumbnailMeta") or []:
        ts = entry.get("ts")
        img = cv2.imread(frames[ts]) if ts in frames else None
        for roi in entry.get("roi") or []:
            tracker = roi.get("trackerID", roi.get("trackerId"))
            if img is None or tracker is None or not roi.get("coord"):
                continue
            objects.append({"trackerID": tracker, "keyMomentMs": ts,
                            "img": ram_tagger.crop_box(img, roi["coord"]), "hint": roi})
    return objects


def _recognize_images(console_host, mac_nosep, body):
    """RAM result for an image task: ramType "image" (reqUrl + keyMoment) or "multipleImages"
    (images[] with their own reqUrl and timestamps). None when nothing could be tagged."""
    tagger = _get_tagger()
    if not tagger:
        return None
    t0 = time.monotonic()
    event_id = body.get("event") or "event"
    if body.get("ramType") == "image":
        entries = [body]
    else:
        entries = list(body.get("images") or [])
    items = [(e.get("reqUrl"), e.get("keyMoment") or body.get("keyMoment") or body.get("start"),
              e.get("imageId"), e) for e in entries]
    clips_dir = os.path.join(STATE_DIR, "clips")
    frames, hints, objects = {}, {}, []
    for n, (url, ms, image_id, entry) in enumerate(items):
        if not url or ms is None:
            continue
        url = url if url.startswith("http") else \
            f"https://{console_host}:{CONSOLE_INTERNAL_API_PORT}{url}"
        path = os.path.join(clips_dir, f"{event_id}-images", f"{image_id or n}.jpg")
        if clip_fetch.fetch_image(url, _callback_headers(mac_nosep), (CERT_PATH, KEY_PATH), path):
            frames[int(ms)] = path
            if entry.get("objectType") or entry.get("attributes"):
                hints[int(ms)] = entry
            tracker = entry.get("trackerId", entry.get("trackerID"))
            if tracker is not None:
                import cv2
                objects.append({"trackerID": tracker, "keyMomentMs": int(ms),
                                "img": cv2.imread(path), "hint": entry})
    clip_fetch.prune(clips_dir)
    log.info("event %s: %d/%d images (%s)", event_id, len(frames), len(items), body.get("ramType"))
    if not frames:
        return None
    try:
        result = ram_tagger.build_result(tagger, body.get("camera"), body.get("event"), frames,
                                         pre_ms=(time.monotonic() - t0) * 1000, hints=hints,
                                         objects=objects)
    except Exception as e:  # noqa: BLE001
        log.warning("RAM image tagging failed for event %s: %r", event_id, e)
        return None
    log.info("event %s: %s", event_id, result["description"])
    return result


def _post_recognize_result(console_host, mac_nosep, res_url, camera_id, event_id, result=None,
                           parts=None, images=None):
    """POST our recognize-anything result to resUrl.

    CONFIRMED from the real firmware's syswrapper.sh: this is multipart/form-data with the
    JSON under form field ``ram`` (optionally with per-tracker JPEG fields), not a plain JSON
    POST. `result` is a ramInferenceSchema dict from ram_tagger.build_result, or None.
    `parts` adds other JSON parts by field name (e.g. {"face": ..., "lpr": ...}) and
    `images` per-tracker JPEGs ({trackerID: jpeg_bytes}, field str(trackerID)).

    Protect validates each JSON part against its schema and rejects the whole multipart if
    any fails, or if their eventIds differ (Protect M/761379:343-375), so every part gets
    this task's eventId. There is no placeholder ram any more: one without `description`
    failed ramInferenceSchema (M/315148) and took face/lpr down with it. Without a RAM
    result the ram part is left out and Protect keys the save on lpr/face's eventId
    instead (M/618986:51). With no JSON part at all nothing is POSTed: an empty upload has
    no eventId, so Protect's save returns before touching the task (M/618986:46-55) -- the
    POST would change nothing, and the task simply times out on Protect's side as before.
    """
    if not res_url:
        log.warning("recognizeKeyFrames task had no resUrl, nothing to post")
        return
    json_parts = {}
    for name, part in (("ram", result), *((parts or {}).items())):
        if not part:
            continue
        part = dict(part)
        if part.get("eventId") not in (None, event_id):
            log.warning("%s part for event %s carried eventId %r; replacing it",
                        name, event_id, part["eventId"])
        part["eventId"] = event_id
        json_parts[name] = part
    if not json_parts:
        log.info("event %s: no ram/face/lpr result; nothing to POST to %s", event_id, res_url)
        return
    url = res_url if res_url.startswith("http") else \
        f"https://{console_host}:{CONSOLE_INTERNAL_API_PORT}{res_url}"
    files = {name: (f"{name}.json", json.dumps(part), "application/json")
             for name, part in json_parts.items()}
    for tracker_id, jpeg in (images or {}).items():
        if jpeg:
            files[str(tracker_id)] = (f"{tracker_id}.jpg", jpeg, "image/jpeg")
    try:
        resp = requests.post(url, headers=_callback_headers(mac_nosep), files=files,
                             cert=(CERT_PATH, KEY_PATH), verify=False, timeout=15)
        log.info("POST %s [%s] -> %s: %s", url, ", ".join(files), resp.status_code,
                 resp.text[:2000])
    except requests.RequestException as e:
        log.warning("POST %s failed: %r", url, e)


def _watch_for_reconnect(ws, this_connection_token, stop_event, poll_interval=2.0):
    """A fresh adopt POST means the console wants a new WSS session carrying the
    new token and will not necessarily close the old one. Any token change --
    including clearing on un-adopt -- makes X-Adopted stale, so tear down."""
    while not stop_event.is_set():
        time.sleep(poll_interval)
        current = config.read_token(STATE_DIR)
        if current != this_connection_token:
            log.info("adopt token changed (%s -> %s) -- closing this connection "
                     "to force a fresh one", this_connection_token, current)
            try:
                ws.close()
            except Exception:
                pass
            return


def run(host: str, port: int, token: str, send_token_header: bool, device_info: dict,
        adopted: bool = True):
    device_info = {**device_info, "console_host": host}
    url = f"wss://{host}:{port}/"
    headers = {
        "X-Ident": device_info["mac_nosep"],
        "X-Mode": "0",
        "X-Type": device_info["type"],
        "X-Sysid": device_info["sysid"],
        "X-Adopted": "true" if adopted else "false",
        "X-Ip": device_info["ip"],
        "X-Device-Id": device_info["device_id"],
        "X-Version": _fw_version(device_info),
        "X-Guid": device_info["guid"],
    }
    if send_token_header:
        headers["X-Token"] = token

    log.info("connecting to %s with headers=%s", url, headers)
    ctx = build_ssl_context()

    # Leave from the bound address too, so a second IP on a shared host is the
    # one the console sees (websockets passes this to socket.create_connection).
    extra = {}
    if device_info.get("bind_ip") not in (None, "", "0.0.0.0"):
        extra["source_address"] = (device_info["bind_ip"], 0)

    with connect(url, subprotocols=["ucp4"], additional_headers=headers,
                 ssl=ctx, open_timeout=15, **extra) as ws:
        log.info("WSS CONNECTED (subprotocol=%s)", ws.subprotocol)
        device_info["ws"] = ws  # this connection's copy; updateFirmware closes it to "reboot"

        stop_watch = threading.Event()
        watcher = threading.Thread(target=_watch_for_reconnect, args=(ws, token, stop_watch),
                                   daemon=True)
        watcher.start()

        req_id = str(uuid.uuid4())
        msg = pack_message(
            {"timestamp": int(time.time() * 1000), "type": "request",
             "action": "getConsoleInfo", "id": req_id},
            {},
        )
        ws.send(msg)
        log.info(">>> sent getConsoleInfo request id=%s", req_id)

        try:
            _send_time_sync(ws)
            threading.Thread(target=_time_sync_loop, args=(ws, stop_watch), daemon=True,
                             name="diykey-timesync").start()
            _consume(ws, device_info)
        finally:
            stop_watch.set()


# Firmware ucpv4_time_sync_monitor sends request action "timeSync" body {"t0":<ms>}
# (ui-websocketd format string {"t0":%ld%03ld}); Protect's devices.timeSync handler answers
# the same id with {t0, t1: Protect's receive Date.now(), t2: its send Date.now()}
# (M/322759_timeSync.js, routed from M/188339 devices.websockets:211-219). The poll
# interval is not in the binary's strings; DIYKEY_TIMESYNC_S (default 60) stands in.
_time_sync_pending = {}  # request id -> t0 ms; drained by _on_time_sync_response
_time_sync_lock = threading.Lock()


def _send_time_sync(ws):
    req_id = str(uuid.uuid4())
    t0 = int(time.time() * 1000)
    with _time_sync_lock:
        _time_sync_pending[req_id] = t0
        if len(_time_sync_pending) > 8:  # unanswered ones must not pile up
            _time_sync_pending.pop(next(iter(_time_sync_pending)))
    ws.send(pack_message({"timestamp": t0, "type": "request", "action": "timeSync",
                          "id": req_id}, {"t0": t0}))


def _on_time_sync_response(head, body):
    """True when `head` answers one of our timeSync requests (then logs the NTP-style offset:
    positive = Protect's clock ahead of ours); other responses (getConsoleInfo) fall through."""
    with _time_sync_lock:
        t0 = _time_sync_pending.pop(head.get("id"), None)
    if t0 is None:
        return False
    t3 = int(time.time() * 1000)
    try:
        t1, t2 = float(body["t1"]), float(body["t2"])
    except (KeyError, TypeError, ValueError):
        log.debug("timeSync response without t1/t2: %s", body)
        return True
    offset = ((t1 - t0) + (t2 - t3)) / 2
    delay = (t3 - t0) - (t2 - t1)
    log.debug("timeSync: offset %+.0f ms, round trip %.0f ms", offset, delay)
    global _time_sync_warned
    if abs(offset) > 2000 and not _time_sync_warned:
        _time_sync_warned = True
        log.info("timeSync: clock differs from the console by %+.1f s", offset / 1000)
    return True


_time_sync_warned = False


def _time_sync_loop(ws, stop_event):
    interval = max(1.0, _env_number("DIYKEY_TIMESYNC_S", 60, float))
    while not stop_event.wait(interval):
        try:
            _send_time_sync(ws)
        except Exception:  # noqa: BLE001 -- the connection is closing
            return


def _consume(ws, device_info):
    for raw in ws:
        if isinstance(raw, str):
            log.warning("unexpected TEXT frame (protocol says binary-only): %r", raw[:200])
            continue
        records = unpack_all_records(raw)
        if len(records) < 2:
            log.warning("frame with <2 records: %s", records)
            continue
        head = records[0][2]
        body = records[1][2]

        if head.get("type") == "response":
            if _on_time_sync_response(head, body):
                continue
            log.info("<<< response to our request id=%s errorCode=%s body=%s",
                     head.get("id"), head.get("errorCode"), body)
            continue

        error = None
        try:
            reply_body = handle_action(head, body, device_info)
        except ActionError as e:
            reply_body, error = {}, e
        if head.get("id"):
            reply_head = {"timestamp": int(time.time() * 1000), "type": "response",
                          "action": head.get("action"), "id": head["id"], "errorCode": 0}
            if error:
                reply_head.update(errorCode=error.code, error=error.message)
            reply = pack_message(reply_head, reply_body)
            ws.send(reply)
            log.info(">>> replied to action=%s id=%s body=%s",
                     head.get("action"), head["id"], reply_body)
        if _reset_pending.is_set():
            _reset_pending.clear()
            log.info("closing ucp4 connection after %s", head.get("action"))
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
            return


def _split_host_port(host_port, default_port=7442):
    if ":" in str(host_port):
        host, port_s = str(host_port).rsplit(":", 1)
        return host, int(port_s)
    return str(host_port), default_port


def advanced_alarm_gaps(env=None):
    """What AI Key Advanced alarms (Alarm Manager natural-language rules) lack, by env name.
    Protect parses each rule sentence once over nl-search into txtEmbed + ramTagIds
    (M/253677_aiprocessors.js:17-45), then on every RAM save matches only objects that carry
    an embedding (M/12387_aiprocessors.js:213-231) and share a tag id with the rule
    (M/718745_aiprocessors.js:23,33). So it needs: CLIP (imgEmbed and txtEmbed), the RAM
    detector (thumbnailTags at all), Protect's tag list (query tags it won't drop; the
    built-in list is a safe subset) and the nl-search socket (NL_PARSE)."""
    env = os.environ if env is None else env
    missing = []
    if (env.get("DIYKEY_CLIP") or "0") != "1":
        missing.append("DIYKEY_CLIP=1")
    if not (env.get("DIYKEY_RAM_MODEL") or "").strip():
        missing.append("DIYKEY_RAM_MODEL")
    if not (env.get("DIYKEY_RAM_TAGS") or "").strip():
        missing.append("DIYKEY_RAM_TAGS")
    if (env.get("DIYKEY_NL_SEARCH") or "1") == "0":
        missing.append("DIYKEY_NL_SEARCH (set to 0)")
    return missing


def main():
    cfg = config.load_config_and_logging(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__)
    config.add_common_flags(ap)
    ap.add_argument("--iface", default=None,
                    help="interface to derive identity from (never bound); defaults to "
                         "cfg [network] iface, else the default-route interface")
    ap.add_argument("--bind-ip", default="0.0.0.0",
                    help="local IP to report as our own (X-Ip etc); outbound client, "
                         "so no socket is bound to it")
    ap.add_argument("--mac", default=None)
    ap.add_argument("--type", default=cfg["platform"],
                    help="catalog model string (ucp4 X-Type)")
    ap.add_argument("--version", default=cfg["fw_version"])
    ap.add_argument("--device-id", default=cfg["device_id"])
    ap.add_argument("--guid", default=cfg["guid"])
    ap.add_argument("--fallback-host-port", default=None,
                    help="host:port to reconnect to (tokenless) if no fresh adopt POST "
                         "shows up within --fallback-after seconds")
    ap.add_argument("--fallback-after", type=float, default=10.0)
    ap.add_argument("--state-dir", default=None,
                    help="where adopt_state.json/server.crt/server.key live -- defaults "
                         "to stage1_http/; set this to isolate another instance")
    args = ap.parse_args()
    args.iface = config.resolve_iface(args.iface, cfg)

    global STATE_DIR, ADOPT_STATE_FILE, CERT_PATH, KEY_PATH
    STATE_DIR = config.resolve_state_dir(args.state_dir)
    paths = config.state_paths(STATE_DIR)
    ADOPT_STATE_FILE, CERT_PATH, KEY_PATH = (
        paths["adopt_state"], paths["cert"], paths["key"])

    device_info = {
        "mac_nosep": config.resolve_mac(args.iface, args.mac, cfg).lower(),
        "type": args.type,
        "version": args.version,
        "device_id": args.device_id,
        "guid": args.guid,
        "ip": config.advertised_ip(config.resolve_ip(args.iface, cfg, args.bind_ip)),
        "bind_ip": args.bind_ip,
        "sysid": f"0x{config.sysid_int(cfg):04x}",
    }

    missing = advanced_alarm_gaps()
    if missing:
        log.warning("AI Key Advanced alarms (natural-language Alarm Manager rules) won't fire "
                    "reliably -- missing: %s", ", ".join(missing))
    threading.Thread(target=_warm_models, daemon=True, name="diykey-warm").start()
    if os.environ.get("DIYKEY_NL_SEARCH", "1") != "0":
        threading.Thread(target=_nl_search_loop, args=(args.fallback_host_port, device_info),
                         daemon=True).start()

    used_tokens = _used_tokens
    backoff = 2

    while True:
        state = wait_for_adopt_state(args.fallback_host_port, fallback_after=args.fallback_after)
        hosts = state.get("hosts") or []
        token = state.get("token")
        # Canonical adopted predicate (token or hosts), matching getInfo,
        # /api/info and discovery.
        adopted = config.is_adopted(STATE_DIR)
        if not hosts:
            log.warning("adopt state had no hosts, retrying in 5s")
            time.sleep(5)
            continue
        try:
            host, port = _split_host_port(hosts[0])
        except (TypeError, ValueError):
            log.warning("malformed adopt state host %r; retrying in 5s", hosts[0])
            time.sleep(5)
            continue

        send_token = token is not None and token not in used_tokens
        if token is not None:
            used_tokens.add(token)
        if token is not None and not send_token:
            log.info("token already used on a prior connection -- reconnecting tokenless "
                     "(fingerprint should already be pinned)")
        _reset_pending.clear()
        try:
            run(host, port, token, send_token, device_info, adopted=adopted)
            if not config.is_adopted(STATE_DIR):
                log.info("ucp4 connection closed after factory reset; waiting for a fresh adopt POST")
                continue
            log.warning("ucp4 connection closed cleanly by peer; reconnecting in 3s")
            backoff = 2
            time.sleep(3)
        except Exception as e:
            if not config.is_adopted(STATE_DIR):
                log.info("ucp4 connection ended after un-adopt (%r); waiting for a fresh adopt POST", e)
                continue
            log.warning("connection ended: %r -- reconnecting in %ds", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


if __name__ == "__main__":
    main()
