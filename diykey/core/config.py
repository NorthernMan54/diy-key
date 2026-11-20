#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Shared config, device identity and adoption state for the AI Key emulator.

Single source of truth read from ``diykey.cfg`` (INI); CLI flags win over cfg
values. Defaults to ``<diykey>/diykey.cfg``. Every stage also accepts
``--state-dir`` to keep its adopt token/certs apart from other instances.
"""
import argparse
import configparser
import ipaddress
import json
import logging
import os
import re
import subprocess

# core/ -> diykey/ (the directory holding run_all.py and the stages).
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_DIR, "diykey.cfg")
# Historical location of adopt_state.json/server.crt/server.key; keeping it as
# the default preserves an already-adopted device's pinned TLS fingerprint.
DEFAULT_STATE_DIR = os.path.join(PROJECT_DIR, "stage1_http")

# Real Ubiquiti OUI; a non-Ubiquiti one makes the controller report "Unknown".
UBNT_OUI = "FCECDA"

DEFAULTS = {
    # [identity]
    "mac": "",
    "hostname": "diy-key",
    # Canonical type in ucp4 X-Type, /api/info and discovery TLV 0x0C.
    "platform": "UP-AI-KEY",
    "discovery_platform": "UP-AI-KEY",
    "sysid": "0xa5f0",
    # ucp4 X-Version / getInfo "version".
    "fw_version": "2.2.6",
    # Discovery advertises a longer firmware string than ucp4's "version".
    "discovery_fw_version": "aikey.rk3588.v2.2.6.0.emu.260901.0000",
    # Override per instance; keep stable once adopted (changing it forces re-adopt).
    "device_id": "00000000-0000-4000-8000-000000000001",
    "guid": "00000000-0000-4000-8000-000000000002",
    # [network]
    "iface": "",                # blank = auto-detect (see resolve_iface)
    "parent_iface": "",         # macvlan parent; blank = [network] iface / auto-detect
    "mode": "dhcp",             # "dhcp" | "static"
    "ip": "",
    "netmask": "255.255.255.0",
    "gateway": "",
    # [console]
    "host": "",                 # set to the Protect console's IP
    "port": "7442",
    # [logging]
    "debug": "false",
}

# Env overrides applied after diykey.cfg (container/unit-friendly identity and
# console config, e.g. -e DIYKEY_CONSOLE_HOST=...).
ENV_OVERRIDES = {
    "DIYKEY_MAC": "mac",
    "DIYKEY_HOSTNAME": "hostname",
    "DIYKEY_DEVICE_ID": "device_id",
    "DIYKEY_GUID": "guid",
    "DIYKEY_SYSID": "sysid",
    "DIYKEY_PLATFORM": "platform",
    "DIYKEY_DISCOVERY_PLATFORM": "discovery_platform",
    "DIYKEY_FW_VERSION": "fw_version",
    "DIYKEY_DISCOVERY_FW_VERSION": "discovery_fw_version",
    "DIYKEY_IFACE": "iface",
    "DIYKEY_PARENT_IFACE": "parent_iface",
    "DIYKEY_NETWORK_MODE": "mode",
    "DIYKEY_NETWORK_IP": "ip",
    "DIYKEY_NETWORK_GATEWAY": "gateway",
    "DIYKEY_CONSOLE_HOST": "host",
    "DIYKEY_CONSOLE_PORT": "port",
    "DIYKEY_DEBUG": "debug",
}


def load_config(path=None):
    """Parse diykey.cfg into merged settings; a missing file falls back to
    DEFAULTS. Non-empty ENV_OVERRIDES win over the file."""
    cfg = dict(DEFAULTS)
    path = path or DEFAULT_CONFIG_PATH
    parser = configparser.ConfigParser(interpolation=None)
    if os.path.exists(path):
        parser.read(path)
        for section in parser.sections():
            for key, value in parser.items(section):
                cfg[key] = value.strip()
    for env_name, key in ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value:
            cfg[key] = value
    return cfg


def get_bool(cfg, key):
    return str(cfg.get(key, "")).strip().lower() in ("1", "true", "yes", "on")


# Secrets that appear in logged payloads/headers: the console's password,
# adoption tokens and console-session cookies.
_REDACTIONS = [
    # URL userinfo, greedily through the last '@' so a password containing '@'
    # cannot leave its tail in the log.
    (re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s'\"]+@"), r"\1<redacted>@"),
    # Query/form credentials.
    (re.compile(r"([?&](?:password|passwd|username|user|token)=)[^&\s'\"]+", re.I),
     r"\1<redacted>"),
    (re.compile(r"""(['"](?:password|passwd|newPassword|new_password|passwordOld|passwordNew|[\w-]*token|adoptionCode)['"]\s*:\s*['"])[^'"]+""", re.I),
     r"\1<redacted>"),
    # "token changed (old -> new)" before the generic token-value rule.
    (re.compile(r"(\btoken changed \()[^)]*"), r"\1<redacted>"),
    # "token: X" / "newPassword: X" (colon or equals), including short values.
    (re.compile(r"(\b(?:token|newPassword|new_password)\b\s*[:=]\s*)[^\s,;)}\]]+", re.I),
     r"\1<redacted>"),
    # A long token-like value after a space ("token FAKE... already used").
    (re.compile(r"(\b(?:token|newPassword|new_password)\b[^A-Za-z0-9<(]{1,12})[A-Za-z0-9_+/.=-]{12,}"), r"\1<redacted>"),
]


def redact(text):
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class RedactFilter(logging.Filter):
    """Rewrites each record's final message through redact()."""

    def filter(self, record):
        message = record.getMessage()
        redacted = redact(message)
        if redacted != message:
            record.msg, record.args = redacted, None
        return True


def apply_logging(debug=False):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    # On the handlers, not the root logger: logger filters skip records
    # propagated from child loggers.
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, RedactFilter) for f in handler.filters):
            handler.addFilter(RedactFilter())
    # DIYKEY_DEBUG_LOGGERS=diykey.lpr,diykey.x: debug for just those loggers
    # (handlers have no level, so their records pass at INFO root).
    for name in os.environ.get("DIYKEY_DEBUG_LOGGERS", "").split(","):
        if name.strip():
            logging.getLogger(name.strip()).setLevel(logging.DEBUG)


def sysid_int(cfg):
    return int(str(cfg.get("sysid", "0xa5f0")), 0)


def add_common_flags(ap):
    ap.add_argument("--config", default=None,
                    help="path to diykey.cfg (default: <diykey>/diykey.cfg)")
    ap.add_argument("--debug", action="store_true", default=None,
                    help="enable debug logging (overrides [logging] debug=true)")


def load_config_and_logging(argv=None):
    """Pre-parse --config/--debug (position-independent), load cfg and apply
    the debug log level; callers must also declare both via add_common_flags()."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--debug", action="store_true", default=None)
    known, _ = pre.parse_known_args(argv if argv is not None else None)
    cfg = load_config(known.config)
    debug = known.debug if known.debug is not None else get_bool(cfg, "debug")
    apply_logging(debug)
    return cfg


def state_dir_path(state_dir=None):
    """The directory holding this instance's adopt token and TLS keypair
    (``--state-dir``, else the historical stage1_http/ path). No side effects:
    readers must be able to point at a directory that does not exist yet."""
    return os.path.abspath(state_dir or DEFAULT_STATE_DIR)


def resolve_state_dir(state_dir=None):
    """Create and return the state directory (for writers/stages)."""
    path = state_dir_path(state_dir)
    os.makedirs(path, exist_ok=True)
    return path


def adopt_state_path(state_dir=None):
    return os.path.join(state_dir_path(state_dir), "adopt_state.json")


def state_paths(state_dir=None):
    """Paths every stage must agree on for one instance; creates the dir."""
    path = resolve_state_dir(state_dir)
    return {
        "state_dir": path,
        "adopt_state": os.path.join(path, "adopt_state.json"),
        "cert": os.path.join(path, "server.crt"),
        "key": os.path.join(path, "server.key"),
    }


def _run(cmd):
    """stdout of a short command, or "" if it's missing/fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def _ifconfig_field(iface, key):
    """First value after `key` in BSD/macOS `ifconfig <iface>` (no /sys, no `ip`)."""
    for line in _run(["ifconfig", iface]).splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == key:
            return fields[1]
    return ""


def _iface_mac(iface):
    try:
        with open(f"/sys/class/net/{iface}/address") as f:
            return f.read().strip().upper().replace(":", "")
    except OSError:
        pass
    # macOS masks `ifconfig` ether as 02:00:00:00:00:00 for most processes;
    # networksetup doesn't.
    out = _run(["networksetup", "-getmacaddress", iface]).split()
    mac = out[2] if len(out) > 2 and out[:2] == ["Ethernet", "Address:"] else ""
    mac = mac or _ifconfig_field(iface, "ether")
    mac = mac.upper().replace(":", "")
    return "" if mac == "020000000000" else mac


def _default_iface():
    """Best-effort auto-detect: the default-route interface, else the first
    non-loopback up interface. Returns "" if unknown."""
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:
                fields = line.split()
                # dest 0.0.0.0 with RTF_UP (0x1) = the default-route device
                if len(fields) >= 4 and fields[1] == "00000000" and int(fields[3], 16) & 0x1:
                    return fields[0]
    except (OSError, ValueError):
        pass
    try:
        for name in sorted(os.listdir("/sys/class/net")):
            if name == "lo":
                continue
            try:
                with open(f"/sys/class/net/{name}/operstate") as fh:
                    if fh.read().strip() in ("up", "unknown"):
                        return name
            except OSError:
                continue
    except OSError:
        pass
    # macOS/BSD: no /proc or /sys, ask the routing table.
    for line in _run(["route", "-n", "get", "default"]).splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "interface" and value.strip():
            return value.strip()
    return ""


def resolve_iface(iface_override=None, cfg=None):
    """Interface to bind/derive identity from. Precedence: explicit --iface,
    then cfg [network] iface, then auto-detect."""
    cfg = cfg or {}
    explicit = iface_override or cfg.get("iface") or ""
    return explicit or _default_iface() or "eth0"


def resolve_parent_iface(cfg=None):
    """Physical interface a macvlan instance is created on top of. Precedence:
    cfg [network] parent_iface, then [network] iface, then auto-detect."""
    cfg = cfg or {}
    return cfg.get("parent_iface") or resolve_iface(None, cfg)


def resolve_mac(iface, mac_override=None, cfg=None):
    """12-hex-char MAC (uppercase, no separators). Precedence: --mac, cfg
    [identity] mac, then the real iface MAC with the Ubiquiti OUI prepended."""
    cfg = cfg or {}
    raw = mac_override or cfg.get("mac") or ""
    if raw:
        return raw.replace(":", "").replace("-", "").upper()
    real = _iface_mac(iface)
    if real and len(real) == 12:
        return UBNT_OUI + real[6:]
    return "000000000000"


def _iface_ip(iface):
    """The interface's currently-assigned IPv4, or None."""
    try:
        out = subprocess.run(["ip", "-4", "-json", "addr", "show", iface],
                             capture_output=True, text=True, timeout=5)
        data = json.loads(out.stdout or "[]")
        for link in data:
            for addr in link.get("addr_info", []):
                if addr.get("family") == "inet":
                    return addr["local"]
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError, KeyError):
        pass
    # BSD/macOS ifconfig; reject busybox's "addr:1.2.3.4" form rather than misparse it.
    ip = _ifconfig_field(iface, "inet")
    try:
        return str(ipaddress.IPv4Address(ip))
    except ValueError:
        return None


def resolve_ip(iface, cfg=None, bind_ip=None):
    """IP to report/bind to. Precedence: --bind-ip, then static cfg ip (when
    mode == static), then the interface's live IP; "0.0.0.0" as last resort."""
    if bind_ip and bind_ip != "0.0.0.0":
        return bind_ip
    cfg = cfg or {}
    if str(cfg.get("mode", "")).lower() == "static" and cfg.get("ip"):
        return cfg["ip"]
    return _iface_ip(iface) or "0.0.0.0"


def advertised_ip(ip):
    """The IP to tell Protect, when that isn't the one bound: DIYKEY_ADVERTISE_IP,
    for a NAT'd/forwarded setup. Only what's reported changes, never what's bound."""
    return (os.environ.get("DIYKEY_ADVERTISE_IP") or "").strip() or ip


def netmask_to_prefix(netmask):
    """Dotted-quad netmask -> CIDR prefix length (255.255.255.0 -> 24)."""
    if not netmask:
        return None
    try:
        octets = [int(x) for x in str(netmask).split(".")]
        if len(octets) != 4 or any(o < 0 or o > 255 for o in octets):
            return None
        value = sum(o << (8 * (3 - i)) for i, o in enumerate(octets))
        inverted = (~value) & 0xFFFFFFFF
        if inverted & (inverted + 1):  # non-contiguous mask
            return None
        return bin(value).count("1")
    except ValueError:
        return None


def atomic_write_json(path, obj):
    """Write obj as JSON atomically (tmp + fsync + os.replace), mode 0600, so
    readers never see a torn write and a power cut cannot lose the rename."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    if parent:
        try:
            dfd = os.open(parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dfd)
        except OSError:
            pass
        finally:
            os.close(dfd)


def read_adopt_state(state_dir=None):
    try:
        with open(adopt_state_path(state_dir)) as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_token(state_dir=None):
    token = read_adopt_state(state_dir).get("token")
    return token or None


def is_adopted(state_dir=None):
    """Best-effort local signal that the console adopted us: an adopt token or
    a console host list is present. Read live, never cached."""
    state = read_adopt_state(state_dir)
    return bool(state.get("token")) or bool(state.get("hosts"))


def console_id_bytes(state_dir=None):
    """16 raw bytes of the adopted console's anonymousDeviceId UUID, or None."""
    console_id = read_adopt_state(state_dir).get("consoleId")
    if not console_id:
        return None
    try:
        return bytes.fromhex(str(console_id).replace("-", ""))
    except ValueError:
        return None


def write_adopt_state(state_dir, payload):
    """Persist a validated adopt payload (the console's ``mgmt``/flat body)."""
    atomic_write_json(adopt_state_path(state_dir), payload)


def clear_adopt_state(state_dir=None):
    """Un-adopt locally. Overwrite with {} rather than deleting so every stage
    sees "not adopted" instead of ENOENT and racing a stale cache."""
    atomic_write_json(adopt_state_path(state_dir), {})


# Factory version; a simulated update pins the real one in <state>/fw_version.json, read live
# so discovery, /api/info and ucp4 agree without a restart.
def fw_version_path(state_dir=None):
    return os.path.join(state_dir_path(state_dir), "fw_version.json")


def pinned_fw_version(state_dir=None):
    """The version a simulated update installed, or None."""
    try:
        with open(fw_version_path(state_dir)) as f:
            version = json.load(f).get("version")
        return str(version) if version else None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def pin_fw_version(state_dir, version):
    atomic_write_json(fw_version_path(state_dir), {"version": version})


def effective_fw_version(cfg, state_dir=None):
    """Short version for /api/info, getInfo and ucp4 X-Version (e.g. "2.2.8")."""
    return pinned_fw_version(state_dir) or cfg["fw_version"]


def effective_discovery_fw_version(cfg, state_dir=None):
    """Discovery's long firmware string, with its vX.Y.Z swapped for a pinned version."""
    configured = cfg["discovery_fw_version"]
    pinned = pinned_fw_version(state_dir)
    if not pinned:
        return configured
    swapped, n = re.subn(r"v\d+\.\d+\.\d+", f"v{pinned}", configured, count=1)
    return swapped if n else configured
