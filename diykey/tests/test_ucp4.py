# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Unit tests for ucp4_client.py: wire framing, state and RPC handlers."""
import concurrent.futures
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ucp4_client  # noqa: E402
from core import config  # noqa: E402

# The service environment must not leak into tests; each test sets what it needs.
for _key in [k for k in os.environ if k.startswith("DIYKEY_")]:
    os.environ.pop(_key, None)


class TestPackRecord(unittest.TestCase):
    def test_header_shape(self):
        rec = ucp4_client.pack_record(3, {"a": 1})
        rtype, rec_format, compressed, reserved, length = struct.unpack(">BBBBI", rec[:8])
        self.assertEqual((rtype, rec_format, compressed, reserved), (3, 1, 0, 0))
        self.assertEqual(length, len(rec) - 8)
        self.assertEqual(json.loads(rec[8:]), {"a": 1})

    def test_u32_length_survives_large_payload(self):
        payload = {"blob": "x" * 70000}
        rec = ucp4_client.pack_record(2, payload)
        self.assertEqual(struct.unpack(">I", rec[4:8])[0], len(rec) - 8)
        self.assertEqual(ucp4_client.unpack_all_records(rec)[0][2], payload)

    def test_raw_body_format(self):
        blob = bytes(range(256)) * 400
        rec = ucp4_client.pack_record(2, blob)
        rtype, rec_format, _c, _r, length = struct.unpack(">BBBBI", rec[:8])
        self.assertEqual((rtype, rec_format, length), (2, 3, len(blob)))
        self.assertEqual(rec[8:], blob)
        self.assertEqual(ucp4_client.unpack_all_records(rec)[0][2], {"_raw_bytes": blob})

    def test_pack_message_head_then_body(self):
        data = ucp4_client.pack_message({"h": 1}, {"b": 2})
        recs = ucp4_client.unpack_all_records(data)
        self.assertEqual([r[0] for r in recs], [1, 2])
        self.assertEqual([r[2] for r in recs], [{"h": 1}, {"b": 2}])


class TestUnpack(unittest.TestCase):
    def test_multiple_and_trailing(self):
        data = (ucp4_client.pack_record(1, {"x": "y"})
                + ucp4_client.pack_record(2, {"n": 42})
                + b"\x01\x02\x03")
        recs = ucp4_client.unpack_all_records(data)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[1][2], {"n": 42})

    def test_non_json_payload_is_raw(self):
        body = b"not json"
        data = struct.pack(">BBBBI", 7, 1, 0, 0, len(body)) + body
        self.assertEqual(ucp4_client.unpack_all_records(data), [(7, 1, {"_raw": "not json"})])

    def test_empty(self):
        self.assertEqual(ucp4_client.unpack_all_records(b""), [])


class _StateDirTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self._orig_state = ucp4_client.STATE_DIR
        self._orig_file = ucp4_client.ADOPT_STATE_FILE
        ucp4_client.STATE_DIR = self.d
        ucp4_client.ADOPT_STATE_FILE = os.path.join(self.d, "adopt_state.json")
        self.addCleanup(self._restore)

    def _restore(self):
        ucp4_client.STATE_DIR = self._orig_state
        ucp4_client.ADOPT_STATE_FILE = self._orig_file

    def _adopt(self, token="tok", extra=None):
        config.write_adopt_state(self.d, {"hosts": ["10.0.0.5:7442"], "token": token,
                                          **(extra or {})})


class TestHandleAction(_StateDirTest):
    DEVICE = {
        "mac_nosep": "fcecdaaabbcc",
        "type": "UP-AI-KEY",
        "version": "2.2.6",
        "guid": "00000000-0000-4000-8000-000000000002",
        "device_id": "00000000-0000-4000-8000-000000000001",
        "ip": "10.0.0.9",
    }

    def _call(self, action, body=None):
        return ucp4_client.handle_action({"action": action}, body or {}, dict(self.DEVICE))

    def test_get_info_shape(self):
        self._adopt()
        reply = ucp4_client.handle_action(
            {"action": "getInfo"}, {}, {**self.DEVICE, "console_host": "10.0.0.5"})
        self.assertEqual(reply["mac"], "fcecdaaabbcc")
        self.assertEqual(reply["type"], "UP-AI-KEY")
        self.assertEqual(reply["version"], "2.2.6")
        self.assertTrue(reply["adopted"])
        self.assertEqual(reply["deviceId"], "00000000-0000-4000-8000-000000000001")
        self.assertEqual(reply["connectionHost"], "10.0.0.5")
        self.assertEqual(reply["poeType"], "802.3bt")

    def test_get_info_adopted_is_live(self):
        reply = self._call("getInfo")
        self.assertFalse(reply["adopted"])
        self._adopt()
        self.assertTrue(self._call("getInfo")["adopted"])
        config.clear_adopt_state(self.d)
        self.assertFalse(self._call("getInfo")["adopted"])

    def test_network_status(self):
        self.assertEqual(self._call("networkStatus"),
                         {"ip": "10.0.0.9", "connected": True, "linkSpeedMbps": 1000})

    def test_disk_info(self):
        self.assertEqual(self._call("diskInfo"), {"storageSize": "256"})

    _FLAG_ENV = ("DIYKEY_PERSON_REID", "DIYKEY_FACE", "DIYKEY_LPR", "DIYKEY_FACE_ENHANCE", "DIYKEY_VLM",
                 "DIYKEY_CLIP", "DIYKEY_RAM_MODEL", "DIYKEY_DEEP_MODE")

    def _flags(self, **env):
        clean = {k: v for k, v in os.environ.items() if k not in self._FLAG_ENV}
        with mock.patch.dict(os.environ, {**clean, **env}, clear=True):
            return self._call("getInfo")["featureFlags"]

    def test_get_info_feature_flags_default_off(self):
        # Absent flags read as enabled in Protect (M/643837:52-67): every one must be sent.
        flags = self._flags()
        off = {"enabled": False, "version": "v1"}
        for name in ("supportFaceEnhancement", "supportRetroactiveProcessing",
                     "supportImageSearch", "supportRecognizeAnything", "supportFaceRecognition",
                     "supportLicensePlateRecognition", "supportAiSummary"):
            self.assertEqual(flags[name], off, name)
        self.assertEqual(flags["supportPoseEstimation"], {"enabled": True, "version": "v1"})
        # On by default: it gates Person in Second Stage Verification.
        self.assertEqual(flags["supportPersonReId"], {"enabled": True, "version": "v1"})
        self.assertEqual(flags["aiMode"], "basic")

    def test_get_info_feature_flags_from_env(self):
        flags = self._flags(DIYKEY_FACE="1", DIYKEY_LPR="true", DIYKEY_FACE_ENHANCE="yes",
                            DIYKEY_VLM="mlx", DIYKEY_CLIP="1", DIYKEY_RAM_MODEL="/m.onnx")
        for name in ("supportFaceEnhancement", "supportRetroactiveProcessing",
                     "supportImageSearch", "supportRecognizeAnything", "supportFaceRecognition",
                     "supportLicensePlateRecognition", "supportAiSummary"):
            self.assertTrue(flags[name]["enabled"], name)
        self.assertFalse(self._flags(DIYKEY_PERSON_REID="0")["supportPersonReId"]["enabled"])

    def test_advertise_all_matches_the_firmware_set(self):
        flags = self._flags(DIYKEY_ADVERTISE_ALL="1")
        for name in ("supportFaceEnhancement", "supportRetroactiveProcessing",
                     "supportImageSearch", "supportRecognizeAnything",
                     "supportFaceRecognition", "supportLicensePlateRecognition",
                     "supportAiSummary", "supportReverification"):
            self.assertTrue(flags[name]["enabled"], name)
        self.assertFalse(flags["supportLocate"])

    def test_feature_flags_falsy_values(self):
        flags = self._flags(DIYKEY_FACE="0", DIYKEY_LPR="", DIYKEY_VLM="off", DIYKEY_RAM_MODEL="")
        for name in ("supportFaceRecognition", "supportLicensePlateRecognition",
                     "supportAiSummary", "supportRecognizeAnything",
                     "supportRetroactiveProcessing"):
            self.assertFalse(flags[name]["enabled"], name)
        self.assertFalse(self._flags(DIYKEY_VLM="0")["supportAiSummary"]["enabled"])
        self.assertFalse(self._flags(DIYKEY_VLM="qwen2-vl")["supportAiSummary"]["enabled"])

    def test_model_mode_reported_in_get_info(self):
        self.addCleanup(setattr, ucp4_client, "_ai_mode", "basic")
        self._call("changeAiInferAgentSettings", {"modelMode": "deep"})
        # Not capable (no DIYKEY_DEEP_MODE): never echo a mode we can't run.
        self.assertEqual(self._call("getInfo")["featureFlags"]["aiMode"], "basic")
        flags = self._flags(DIYKEY_VLM="mlx", DIYKEY_DEEP_MODE="1")
        self.assertEqual(flags["aiMode"], "deep")

    def test_deep_mode_capability_needs_vlm_and_opt_in(self):
        for env, want in (({}, False), ({"DIYKEY_VLM": "mlx"}, False),
                          ({"DIYKEY_DEEP_MODE": "1"}, False),
                          ({"DIYKEY_VLM": "mlx", "DIYKEY_DEEP_MODE": "1"}, True)):
            flags = self._flags(**env)
            self.assertIs(flags["supportDeepMode"], want, env)
            self.assertIs(flags["supportVlm"], want, env)

    def test_ai_settings_stored(self):
        # updateAiSettings sends these without modelMode (Protect M/288811:15-26).
        self.addCleanup(ucp4_client._ai_settings.clear)
        self.assertFalse(ucp4_client.ai_setting_enabled("enableFaceRecognize"))
        self.assertTrue(ucp4_client.ai_setting_enabled("enableFaceRecognize", default=True))
        reply = self._call("changeAiInferAgentSettings", {
            "deepModeSupported": True, "enableFaceRecognize": True,
            "enableLprRecognize": False, "enableRAM": True, "region": "CA"})
        self.assertEqual(reply, {})
        self.assertTrue(ucp4_client.ai_setting_enabled("enableFaceRecognize"))
        self.assertFalse(ucp4_client.ai_setting_enabled("enableLprRecognize", default=True))
        self.assertEqual(ucp4_client._ai_mode, "basic")
        self._call("changeAiInferAgentSettings", {"enableLprRecognize": True})
        self.assertTrue(ucp4_client.ai_setting_enabled("enableLprRecognize"))
        self.assertTrue(ucp4_client.ai_setting_enabled("enableFaceRecognize"))

    def test_change_password_without_pg_socket(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DIYKEY_PG_SOCKET", None)
            self.assertEqual(self._call("changeUserPassword", {"username": "ui", "passwordOld": "a", "passwordNew": "b"}), {})

    def test_pg_password_connects_as_cluster_owner(self):
        # Under the root daemon psql's default role would be "root", which doesn't exist.
        run = mock.Mock(return_value=mock.Mock(returncode=0, stderr=""))
        with mock.patch.dict(os.environ, {"DIYKEY_PG_SOCKET": "/pg", "DIYKEY_PG_USER": "owner"}), \
                mock.patch.object(ucp4_client.subprocess, "run", run):
            ucp4_client._apply_pg_password("new")
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("-U") + 1], "owner")

    def test_pg_password_default_role_without_owner(self):
        run = mock.Mock(return_value=mock.Mock(returncode=0, stderr=""))
        env = {k: v for k, v in os.environ.items() if k != "DIYKEY_PG_USER"}
        env["DIYKEY_PG_SOCKET"] = "/pg"
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(ucp4_client.subprocess, "run", run):
            ucp4_client._apply_pg_password("new")
        self.assertNotIn("-U", run.call_args.args[0])

    def test_task_queue_shape(self):
        with mock.patch.object(ucp4_client, "_task_stats", dict.fromkeys(
                ("queued", "queued_ram", "running", "running_ram", "dropped"), 0)):
            reply = self._call("getTaskQueueInfo")
        self.assertEqual(set(reply), {
            "UI_AUDIO_RAM", "UI_AUTO_FACE_ENHANCE", "UI_AUTO_RAM", "UI_AUTO_STT",
            "UI_MANUAL_FACE_ENHANCE", "UI_TASK_NUM", "UI_AUTO_FACE_RECOGNITION",
            "UI_AUTO_LPR_RECOGNITION", "UI_TASK_FAST_LANE"})
        self.assertTrue(all(v == 0 for v in reply.values()))

    def test_task_queue_reports_true_depth(self):
        stats = {"queued": 3, "queued_ram": 2, "running": 1, "running_ram": 1, "dropped": 0}
        with mock.patch.object(ucp4_client, "_task_stats", stats):
            reply = self._call("getTaskQueueInfo")
        self.assertEqual(reply["UI_TASK_NUM"], 4)
        self.assertEqual(reply["UI_AUTO_RAM"], 3)
        self.assertEqual(reply["UI_AUDIO_RAM"], 0)

    def test_task_queue_face_lpr_fast_lane(self):
        stats = {"queued": 3, "queued_ram": 3, "queued_face": 2, "running": 1, "running_ram": 1,
                 "running_face": 1, "running_lpr": 1, "dropped": 0}
        with mock.patch.object(ucp4_client, "_task_stats", stats), \
                mock.patch.object(ucp4_client, "_fast_lane", 2):
            reply = self._call("getTaskQueueInfo")
        self.assertEqual((reply["UI_AUTO_FACE_RECOGNITION"], reply["UI_AUTO_LPR_RECOGNITION"],
                          reply["UI_TASK_FAST_LANE"]), (3, 1, 2))

    def test_recognize_counts_face_group(self):
        submit = mock.Mock()
        body = {"personMeta": [{"ts": 1, "roi": {"coord": [1, 2, 3, 4]}}]}
        env = {"DIYKEY_FACE": "1"}
        ucp4_client._ai_settings.update(enableFaceRecognize=True)
        self.addCleanup(ucp4_client._ai_settings.clear)
        with mock.patch.dict(os.environ, env or {}), mock.patch.object(ucp4_client, "_submit_task", submit):
            ucp4_client.handle_action({"action": "recognizeKeyFrames"}, body,
                                      {"console_host": "h", "mac_nosep": "m"})
        self.assertEqual(submit.call_args.kwargs["extra_groups"], ("face",))

    def test_get_info_sysid_uptime_storage(self):
        reply = ucp4_client.handle_action(
            {"action": "getInfo"}, {}, {**self.DEVICE, "sysid": "0xa5f0"})
        self.assertEqual(reply["sysid"], "0xa5f0")
        self.assertEqual(reply["storageSize"], 256)
        self.assertIsInstance(reply["uptime"], int)
        self.assertGreaterEqual(reply["uptime"], 0)

    def test_system_metrics_flat_array(self):
        reply = self._call("systemMetrics")
        self.assertEqual([m["name"] for m in reply["systemMetrics"]], ["CPU", "GPU", "Memory"])
        for item in reply["systemMetrics"]:
            self.assertEqual(set(item), {"name", "value"})
            self.assertIsInstance(item["value"], (int, float))

    def test_system_metrics_ai_feature_tops(self):
        raw = {"cpu": {"load": 10.0}, "gpu": {"load": 0.0},
               "ram": {"totalKilobytes": 1000, "availableKilobytes": 500}}
        ucp4_client._ai_settings.update({"enableRAM": True, "enableSTT": False,
                                         "enableLprRecognize": True,
                                         "enableFaceRecognize": False})
        self.addCleanup(ucp4_client._ai_settings.clear)
        with mock.patch.object(ucp4_client, "collect_system_metrics", return_value=raw):
            reply = self._call("systemMetrics")
        self.assertEqual(reply["aiFeatureMetrics"], {
            "person-reverification": 1.1, "vehicle-reverification": 1.1,
            "animal-reverification": 1.1, "object-indexing": 7.5,
            "ai-summary": 12.5, "lpr-legacy": 1.2})

    def test_system_metrics_are_fractions(self):
        # Firmware jtop_systemMetrics.py sends load/100; Protect compares PsqlUsage to 0.8.
        raw = {"cpu": {"load": 25.0}, "gpu": {"load": None}, "npu": {"load": 90.0},
               "ram": {"totalKilobytes": 1000, "availableKilobytes": 250}}
        with mock.patch.object(ucp4_client, "collect_system_metrics", return_value=raw):
            metrics = {m["name"]: m["value"] for m in self._call("systemMetrics")["systemMetrics"]}
        self.assertEqual(metrics, {"CPU": 0.25, "GPU": 0.0, "Memory": 0.75})

    def test_ack_only_actions_reply_empty_and_log_once(self):
        ucp4_client._ack_only_logged.discard("sshService")
        self.addCleanup(ucp4_client._ack_only_logged.discard, "sshService")
        with self.assertLogs("ucp4-client", "INFO") as cm:
            self.assertEqual(self._call("sshService"), {})
            self.assertEqual(self._call("sshService"), {})
        self.assertEqual(sum("ack-only" in m for m in cm.output), 1)
        for action in ucp4_client.ACK_ONLY_ACTIONS:
            self.assertEqual(self._call(action), {}, action)
        self.assertNotIn("factoryReset", ucp4_client.ACK_ONLY_ACTIONS)
        self.assertNotIn("restoreDefault", ucp4_client.ACK_ONLY_ACTIONS)
        self.assertNotIn("reboot", ucp4_client.ACK_ONLY_ACTIONS)

    def test_speech_to_text_is_not_acknowledged(self):
        with self.assertRaises(ucp4_client.ActionError):
            self._call("speechToText")

    def test_reboot_requests_a_real_restart(self):
        device = {**self.DEVICE, "ws": mock.Mock()}
        with mock.patch.object(ucp4_client, "_request_restart") as restart:
            self.assertEqual(ucp4_client.handle_action({"action": "reboot"}, {}, device), {})
        restart.assert_called_once()

    def test_unknown_action_is_an_error(self):
        with self.assertLogs("ucp4-client", "WARNING"), \
                self.assertRaises(ucp4_client.ActionError) as cm:
            self._call("playTts")
        self.assertEqual((cm.exception.code, cm.exception.message),
                         (-95, "Unknown action: playTts"))

    def test_consume_sends_error_head_for_unknown_action(self):
        sent = []
        frames = [ucp4_client.pack_message(
            {"type": "request", "action": "bogus", "id": "r1"}, {})]
        ws = mock.Mock(send=sent.append)
        ws.__iter__ = lambda self: iter(frames)
        with self.assertLogs("ucp4-client", "WARNING"):
            ucp4_client._consume(ws, dict(self.DEVICE))
        head, body = (r[2] for r in ucp4_client.unpack_all_records(sent[0]))
        self.assertEqual((head["id"], head["type"], head["errorCode"], head["error"]),
                         ("r1", "response", -95, "Unknown action: bogus"))
        self.assertEqual(body, {})

    def test_feature_flags_reverification_follows_ram_model(self):
        self.assertFalse(self._flags()["supportReverification"]["enabled"])
        self.assertTrue(self._flags(DIYKEY_RAM_MODEL="/m.onnx")["supportReverification"]["enabled"])

    def test_feature_flags_mtls_and_locate(self):
        flags = self._flags()
        self.assertEqual(flags["supportAiProcessorInternalMtls"], {"enabled": True, "version": "v1"})
        self.assertIs(flags["supportLocate"], False)


class TestTaskQueue(unittest.TestCase):
    """The bounded recognizeKeyFrames/RequestAI pool (replaces a thread per task)."""

    def setUp(self):
        self._orig = (ucp4_client._task_q, dict(ucp4_client._task_stats))
        ucp4_client._task_q = None
        for k in ucp4_client._task_stats:
            ucp4_client._task_stats[k] = 0
        self.addCleanup(self._restore)

    def _drain(self, *gates):
        # Release gated tasks and wait for the workers to finish them, so none outlives
        # the test and decrements the counters after _restore/the next setUp zeroed them.
        import time
        for g in gates:
            g.set()
        q = ucp4_client._task_q
        for _ in range(1000):
            st = ucp4_client.task_queue_stats()
            if (q is None or q.empty()) and st["queued"] <= 0 and st["running"] == 0:
                break
            time.sleep(0.01)

    def _restore(self):
        ucp4_client._task_q = self._orig[0]
        ucp4_client._task_stats.clear()
        ucp4_client._task_stats.update(self._orig[1])

    def _env(self, **kw):
        return mock.patch.dict(os.environ, {k: str(v) for k, v in kw.items()})

    def test_runs_in_order_and_counts_running(self):
        import threading
        gate, seen, started = threading.Event(), [], threading.Event()
        self.addCleanup(self._drain, gate)

        def task(n):
            started.set()
            gate.wait(5)
            seen.append(n)

        with self._env(DIYKEY_TASK_WORKERS=1, DIYKEY_TASK_QUEUE_MAX=10):
            ucp4_client._submit_task("recognizeKeyFrames", task, (1,), ram=True)
            self.assertTrue(started.wait(5))
            ucp4_client._submit_task("RequestAI", task, (2,))
            ucp4_client._submit_task("recognizeKeyFrames", task, (3,), ram=True)
            stats = ucp4_client.task_queue_stats()
            self.assertEqual((stats["running"], stats["running_ram"]), (1, 1))
            self.assertEqual((stats["queued"], stats["queued_ram"]), (2, 1))
            gate.set()
            for _ in range(500):
                if len(seen) == 3 and ucp4_client.task_queue_stats()["running"] == 0:
                    break
                import time
                time.sleep(0.01)
        self.assertEqual(seen, [1, 2, 3])
        self.assertEqual(ucp4_client.task_queue_stats()["queued"], 0)

    def test_full_queue_drops_with_warning(self):
        import threading
        gate, started = threading.Event(), threading.Event()
        self.addCleanup(self._drain, gate)

        def block():
            started.set()
            gate.wait(5)

        with self._env(DIYKEY_TASK_WORKERS=1, DIYKEY_TASK_QUEUE_MAX=2):
            ucp4_client._submit_task("RequestAI", block, ())
            self.assertTrue(started.wait(5))
            self.assertTrue(ucp4_client._submit_task("RequestAI", block, ()))
            self.assertTrue(ucp4_client._submit_task("RequestAI", block, ()))
            with self.assertLogs("ucp4-client", "WARNING") as cm:
                self.assertFalse(ucp4_client._submit_task("recognizeKeyFrames", block, (),
                                                          ram=True))
        self.assertIn("queue full", cm.output[0])
        stats = ucp4_client.task_queue_stats()
        self.assertEqual((stats["queued"], stats["queued_ram"], stats["dropped"]), (2, 0, 1))

    def test_stale_task_is_discarded(self):
        import queue
        q = queue.Queue()
        ran = []
        ucp4_client._task_stats.update(queued=1, queued_ram=1)
        q.put((ucp4_client.time.monotonic() - 1000, "recognizeKeyFrames",
               lambda: ran.append(1), (), True))

        def stop(*_a, **_k):
            raise SystemExit  # ends the worker loop once the queue is empty

        with self._env(DIYKEY_TASK_DROP_AGE_S=900), \
                mock.patch.object(q, "get", side_effect=[q.get_nowait(), SystemExit]), \
                self.assertLogs("ucp4-client", "WARNING") as cm:
            with self.assertRaises(SystemExit):
                ucp4_client._task_worker(q)
        self.assertEqual(ran, [])
        self.assertIn("dropped", cm.output[0])
        stats = ucp4_client.task_queue_stats()
        self.assertEqual((stats["queued"], stats["queued_ram"], stats["running"],
                          stats["dropped"]), (0, 0, 0, 1))

    def test_failing_task_does_not_kill_worker(self):
        import queue
        q = queue.Queue()
        ran = []
        ucp4_client._task_stats.update(queued=2)
        q.put((ucp4_client.time.monotonic(), "RequestAI", lambda: 1 / 0, (), False))
        q.put((ucp4_client.time.monotonic(), "RequestAI", lambda: ran.append(1), (), False))
        items = [q.get_nowait(), q.get_nowait(), SystemExit]
        with mock.patch.object(q, "get", side_effect=items), \
                self.assertLogs("ucp4-client", "ERROR"):
            with self.assertRaises(SystemExit):
                ucp4_client._task_worker(q)
        self.assertEqual(ran, [1])
        self.assertEqual(ucp4_client.task_queue_stats()["running"], 0)

    def test_handle_action_queues_instead_of_threading(self):
        submit = mock.Mock(return_value=True)
        device = {"console_host": "10.0.0.5", "mac_nosep": "fcecdaaabbcc"}
        with mock.patch.object(ucp4_client, "_submit_task", submit):
            self.assertEqual(ucp4_client.handle_action(
                {"action": "recognizeKeyFrames"}, {"event": "e1"}, device), {})
            ucp4_client.handle_action({"action": "RequestAI"}, {"resUrl": "x"}, device)
        (kind1, fn1, _a1), kw1 = submit.call_args_list[0]
        self.assertEqual((kind1, fn1, kw1), ("recognizeKeyFrames",
                                             ucp4_client._handle_recognize, {"ram": True}))
        (kind2, fn2, _a2), kw2 = submit.call_args_list[1]
        self.assertEqual((kind2, fn2, kw2), ("RequestAI", ucp4_client._post_request_ai_result, {}))


class TestPostRecognizeResult(unittest.TestCase):
    """Protect rejects the whole multipart if any JSON part fails its schema or the eventIds
    differ (M/761379:343-375)."""

    def _post(self, **kw):
        post = mock.Mock(return_value=mock.Mock(status_code=200, text="{}"))
        with mock.patch.object(ucp4_client.requests, "post", post):
            ucp4_client._post_recognize_result("10.0.0.5", "fcecdaaabbcc",
                                               "/internal/aiprocessors/recognize-anything",
                                               "cam1", "ev1", **kw)
        return post

    def test_ram_result_posted(self):
        post = self._post(result={"cameraId": "cam1", "eventId": "ev1", "description": "a car"})
        files = post.call_args.kwargs["files"]
        self.assertEqual(set(files), {"ram"})
        name, data, ctype = files["ram"]
        self.assertEqual((name, ctype), ("ram.json", "application/json"))
        self.assertEqual(json.loads(data)["description"], "a car")
        self.assertTrue(post.call_args.args[0].startswith("https://10.0.0.5:7444/"))

    def test_no_placeholder_ram(self):
        # The old placeholder had no description and failed ramInferenceSchema (M/315148).
        post = self._post(parts={"face": {"cameraId": "cam1", "eventId": "ev1", "faces": []}})
        files = post.call_args.kwargs["files"]
        self.assertEqual(set(files), {"face"})

    def test_nothing_to_send_skips_post(self):
        post = self._post()
        post.assert_not_called()
        post = self._post(images={7: b"jpeg"})  # images alone reference nothing
        post.assert_not_called()

    def test_parts_and_images_share_event_id(self):
        post = self._post(result={"eventId": "ev1", "description": "d"},
                          parts={"face": {"eventId": "other"}, "lpr": {"plates": []}, "empty": None},
                          images={7: b"\xff\xd8jpeg", "8": b""})
        files = post.call_args.kwargs["files"]
        self.assertEqual(set(files), {"ram", "face", "lpr", "7"})
        for name in ("ram", "face", "lpr"):
            self.assertEqual(json.loads(files[name][1])["eventId"], "ev1", name)
        self.assertEqual(files["7"], ("7.jpg", b"\xff\xd8jpeg", "image/jpeg"))

    def test_caller_dict_not_mutated(self):
        face = {"eventId": "other"}
        self._post(parts={"face": face})
        self.assertEqual(face, {"eventId": "other"})

    def test_no_res_url(self):
        post = mock.Mock()
        with mock.patch.object(ucp4_client.requests, "post", post):
            ucp4_client._post_recognize_result("h", "m", None, "c", "e", {"description": "d"})
        post.assert_not_called()


class TestVersionFromUpdateUri(unittest.TestCase):
    def test_version_from_path(self):
        self.assertEqual(ucp4_client._version_from_update_uri(
            "https://h:7443/internal/update/rk3588/AI-Key/firmware/2.2.8/FCECDA000001"),
            "2.2.8")

    def test_version_from_query(self):
        self.assertEqual(ucp4_client._version_from_update_uri(
            "https://h/x?version=2.2.9&mac=1"),
            "2.2.9")

    def test_empty_uri(self):
        self.assertIsNone(ucp4_client._version_from_update_uri(""))


class TestUpdateFirmware(_StateDirTest):
    DEVICE = {
        "mac_nosep": "fcecdaaabbcc",
        "type": "UP-AI-KEY",
        "version": "2.2.6",
        "guid": "00000000-0000-4000-8000-000000000002",
        "device_id": "00000000-0000-4000-8000-000000000001",
        "ip": "10.0.0.9",
    }

    def setUp(self):
        super().setUp()
        self.addCleanup(setattr, ucp4_client, "_latest_fw", None)

    def _call(self, action, body=None):
        return ucp4_client.handle_action({"action": action}, body or {}, dict(self.DEVICE))

    def test_update_pins_version_without_ws(self):
        self.assertEqual(self._call("updateFirmware",
            {"uri": "https://h/internal/update/p/AI-Key/firmware/2.2.8/MAC"}), {})
        self.assertEqual(self._call("getInfo")["version"], "2.2.8")

    def test_simulate_update_restarts_the_service(self):
        ws = mock.Mock()
        with mock.patch.object(ucp4_client, "FAKE_UPDATE_S", 0), \
                mock.patch.object(ucp4_client.config, "pin_fw_version"), \
                mock.patch.object(ucp4_client, "_request_restart") as restart:
            ucp4_client._simulate_update(ws, "2.2.8", {}, "mac")
        restart.assert_called_once()
        self.assertIs(restart.call_args.args[1], ws)

    def test_update_uses_latest_from_firmware_state(self):
        self._call("firmwareState", {"latestFirmwareVersion": "2.3.0"})
        self.assertEqual(self._call("updateFirmware",
            {"uri": "https://h/blob.bin"}), {})
        self.assertEqual(self._call("getInfo")["version"], "2.3.0")


class TestAdoptStateWait(_StateDirTest):
    def test_returns_existing_state(self):
        self._adopt()
        state = ucp4_client.wait_for_adopt_state(None, poll_interval=0.01, fallback_after=0.01)
        self.assertEqual(state["token"], "tok")

    def test_falls_back_when_adopted_without_hosts(self):
        config.write_adopt_state(self.d, {"token": "t"})
        state = ucp4_client.wait_for_adopt_state("10.0.0.5:7442", poll_interval=0.01,
                                                fallback_after=0.02)
        self.assertEqual(state, {"hosts": ["10.0.0.5:7442"], "token": None})

    def test_no_fallback_when_state_empty(self):
        config.clear_adopt_state(self.d)
        calls = []
        real_sleep = ucp4_client.time.sleep
        def sleep(t):
            calls.append(t)
            if len(calls) > 5:
                config.write_adopt_state(self.d,
                                         {"hosts": ["1.2.3.4:7442"], "token": "n"})
            real_sleep(0)
        with mock.patch.object(ucp4_client.time, "sleep", sleep):
            state = ucp4_client.wait_for_adopt_state("10.0.0.5:7442", poll_interval=0.01,
                                                    fallback_after=0.02)
        self.assertEqual(state["hosts"], ["1.2.3.4:7442"])

    def test_factory_reset_clears_state_and_closes(self):
        self._adopt()
        ucp4_client._used_tokens.add("tok")
        self.addCleanup(ucp4_client._reset_pending.clear)
        nl = mock.Mock()
        with mock.patch.object(ucp4_client, "_nl_ws", nl):
            for action in ("factoryReset", "restoreDefault"):
                self._adopt()
                self.assertEqual(ucp4_client.handle_action(
                    {"action": action}, {}, {}), {})
                self.assertFalse(config.is_adopted(self.d))
        self.assertEqual(ucp4_client._used_tokens, set())
        self.assertTrue(ucp4_client._reset_pending.is_set())
        nl.close.assert_called()

    def test_consume_closes_after_factory_reset_reply(self):
        self._adopt()
        sent = []
        ws = mock.MagicMock()
        ws.__iter__.return_value = iter([ucp4_client.pack_message(
            {"type": "request", "action": "factoryReset", "id": "r1"}, {})])
        ws.send.side_effect = sent.append
        ucp4_client._consume(ws, {})
        self.assertEqual(len(sent), 1)
        ws.close.assert_called_once()
        self.assertFalse(config.is_adopted(self.d))


class TestPgPasswordPlan(unittest.TestCase):
    """changeUserPassword: the rotated password must reach psql, never argv."""

    def _plan(self, env, platform, euid=1000, which=None):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("DIYKEY_PG")}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True), \
                mock.patch.object(ucp4_client.sys, "platform", platform), \
                mock.patch.object(ucp4_client.os, "geteuid", return_value=euid), \
                mock.patch.object(ucp4_client.shutil, "which", side_effect=which):
            return ucp4_client._pg_password_plan("s3cret")

    def test_macos_instance_socket(self):
        cmd, sql = self._plan({"DIYKEY_PG_SOCKET": "/tmp/pg"}, "darwin", which=lambda n: None)
        self.assertEqual(cmd[0], "/opt/homebrew/opt/postgresql@17/bin/psql")
        self.assertEqual(cmd[cmd.index("-h") + 1], "/tmp/pg")
        self.assertIn("ALTER ROLE \"unifi-protect\" PASSWORD 's3cret'", sql)
        self.assertNotIn("s3cret", " ".join(cmd))  # ps would show it

    def test_linux_root_uses_the_system_socket_as_postgres(self):
        cmd, sql = self._plan({}, "linux", euid=0, which=lambda n: {
            "psql": "/usr/bin/psql", "runuser": "/usr/sbin/runuser"}.get(n))
        self.assertEqual(cmd[:4], ["/usr/sbin/runuser", "-u", "postgres", "--"])
        self.assertEqual(cmd[4], "/usr/bin/psql")
        self.assertEqual(cmd[cmd.index("-h") + 1], "/var/run/postgresql")
        self.assertIn("PASSWORD 's3cret'", sql)

    def test_no_postgres_configured(self):
        cmd, reason = self._plan({}, "darwin")
        self.assertIsNone(cmd)
        self.assertIn("DIYKEY_PG_SOCKET", reason)

    def test_embedded_quote_is_doubled(self):
        with mock.patch.dict(os.environ, {"DIYKEY_PG_SOCKET": "/tmp/pg"}, clear=False), \
                mock.patch.object(ucp4_client.sys, "platform", "darwin"):
            _, sql = ucp4_client._pg_password_plan("a'b")
        self.assertIn("PASSWORD 'a''b'", sql)


class TestTaggerConfig(unittest.TestCase):
    def test_missing_model_is_reported_once(self):
        clean = {k: v for k, v in os.environ.items() if k != "DIYKEY_RAM_MODEL"}
        with mock.patch.dict(os.environ, clean, clear=True), \
                mock.patch.object(ucp4_client, "_tagger_warned", False):
            with self.assertLogs("ucp4-client", level="WARNING") as logs:
                self.assertIsNone(ucp4_client._get_tagger())
                self.assertIsNone(ucp4_client._get_tagger())
        self.assertEqual(len([line for line in logs.output if "RAM tagging disabled" in line]), 1)


class TestSplitHostPort(unittest.TestCase):
    def test_with_port(self):
        self.assertEqual(ucp4_client._split_host_port("10.0.0.5:7442"), ("10.0.0.5", 7442))

    def test_default_port(self):
        self.assertEqual(ucp4_client._split_host_port("10.0.0.5"), ("10.0.0.5", 7442))


class _RecognizeTest(_StateDirTest):
    """recognizeKeyFrames with the clip fetch, ffmpeg and models mocked out."""
    START = 1_760_000_000_000

    def setUp(self):
        super().setUp()
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        self.addCleanup(ucp4_client._ai_settings.clear)
        self.extracted = []

        def extract(clip, start, moments):
            self.extracted.append((start, list(moments)))
            out = {}
            for ms in moments:
                path = os.path.join(self.d, f"{int(ms)}.jpg")
                cv2.imwrite(path, np.full((40, 60, 3), 100, np.uint8))
                out[int(ms)] = path
            return out

        self.post = mock.Mock(return_value=mock.Mock(status_code=200, text="{}"))
        for target, attr, value in (
                (ucp4_client.clip_fetch, "fetch_clip", mock.Mock(return_value="/x.mp4")),
                (ucp4_client.clip_fetch, "extract_frames", extract),
                (ucp4_client.clip_fetch, "prune", mock.Mock()),
                (ucp4_client, "_get_tagger", mock.Mock(return_value=None)),
                (ucp4_client.requests, "post", self.post)):
            p = mock.patch.object(target, attr, value)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.dict(os.environ, {"DIYKEY_VLM_QUEUE": "0"})
        p.start()
        self.addCleanup(p.stop)

    def _body(self, **kw):
        s = self.START
        return {"camera": "cam1", "event": "ev1", "start": s, "end": s + 10000,
                "reqUrl": "/internal/aiprocessors/video/export?start=1&format=ubv",
                "resUrl": "/internal/aiprocessors/recognize-anything",
                "keyMoments": [s + 1000, s + 2000, s + 3000], "thumbnailMs": [s + 2000], **kw}

    def _run(self, body, env=None, settings=None, face=None, lpr=None):
        ucp4_client._ai_settings.update(settings or {})
        self.face = mock.Mock(return_value=face)
        self.lpr = mock.Mock(return_value=lpr)
        with mock.patch.dict(os.environ, env or {}), \
                mock.patch.object(ucp4_client.face_recog, "build_face_result", self.face), \
                mock.patch.object(ucp4_client.lpr, "build_lpr", self.lpr):
            ucp4_client._handle_recognize("10.0.0.5", "fcecdaaabbcc", body)

    def _files(self, n=0):
        return self.post.call_args_list[n].kwargs["files"]


class TestRecognizeTiming(_RecognizeTest):
    """One greppable "timing" line per event: long-running events can be found from the log."""

    def _timing(self, cm):
        lines = [r.getMessage() for r in cm.records if "timing ms" in r.getMessage()]
        self.assertEqual(len(lines), 1)
        return lines[0]

    def test_stages_that_ran_are_logged_with_the_queue_wait(self):
        ucp4_client._task_ctx.wait_ms = 1234
        self.addCleanup(delattr, ucp4_client._task_ctx, "wait_ms")
        with self.assertLogs(ucp4_client.log, "INFO") as cm:
            self._run(self._body())
        line = self._timing(cm)
        self.assertIn("event ev1: timing ms camera=cam1 ramType=None objects=- metas=- frames=3 "
                      "wait=1234 total=", line)
        for stage in ("clip=", "frames=", "load=", "post="):
            self.assertIn(stage, line)
        for stage in ("tags=", "face=", "summary="):  # no tagger, no metas, no postVLM here
            self.assertNotIn(stage, line)

    def test_says_what_kind_of_event_it_was(self):
        roi = {"coord": [1, 2, 3, 4], "objectType": "person", "trackerID": 1}
        face = {"coord": [1, 2, 3, 4], "objectType": "face", "trackerID": 2}
        body = self._body(ramType="video", roiMeta=[{"ts": self.START, "roi": [roi]}],
                          thumbnailMeta=[{"ts": self.START, "roi": [face]}],
                          personMeta=[{"ts": self.START, "roi": roi}])
        with self.assertLogs(ucp4_client.log, "INFO") as cm:
            self._run(body)
        self.assertIn("ramType=video objects=face,person metas=personMeta frames=", self._timing(cm))

    def test_logged_when_the_task_fails(self):
        ucp4_client.clip_fetch.fetch_clip.side_effect = RuntimeError("boom")
        with self.assertLogs(ucp4_client.log, "INFO") as cm, self.assertRaises(RuntimeError):
            self._run(self._body())
        self.assertNotIn("clip=", self._timing(cm))

    def test_tags_stage_breakdown_is_appended(self):
        ram = {"cameraId": "cam1", "eventId": "ev1", "description": "a car", "keyMomentsTags": []}

        def build(*a, **k):
            ucp4_client.ram_tagger._prof_add("crops", None)
            ucp4_client.ram_tagger._prof_add("ramplus", time.monotonic())
            return ram

        with mock.patch.object(ucp4_client, "_get_tagger", return_value=mock.Mock()), \
                mock.patch.object(ucp4_client.ram_tagger, "build_result", side_effect=build), \
                self.assertLogs(ucp4_client.log, "INFO") as cm:
            self._run(self._body())
        line = self._timing(cm)
        self.assertRegex(line, r" tags=\d+ .*crops=1 ramplus=\d+/1$")

    def test_worker_records_the_wait_before_running_the_task(self):
        seen, done = [], threading.Event()

        def task():
            seen.append(ucp4_client._task_ctx.wait_ms)
            done.set()

        self.assertTrue(ucp4_client._submit_task("timing-test", task, ()))
        self.assertTrue(done.wait(5))
        self.assertGreaterEqual(seen[0], 0)


class TestFootprint(unittest.TestCase):
    """Warm-up lines say what each model added (live 2026-11-10: 13 GB, 2 MB RSS)."""

    def test_footprint_is_this_process(self):
        mb = ucp4_client._footprint_mb()
        self.assertIsNotNone(mb)
        self.assertGreater(mb, 1)

    def test_note_shows_the_delta_and_total(self):
        with mock.patch.object(ucp4_client, "_footprint_mb", return_value=5200.0):
            self.assertEqual(ucp4_client._mem_note(3500.0), ", +1700 MB (process 5200 MB)")
            self.assertEqual(ucp4_client._mem_note(None), "")

    def test_warm_one_logs_it(self):
        with mock.patch.object(ucp4_client, "_footprint_mb", side_effect=[100.0, 400.0]), \
                self.assertLogs(ucp4_client.log, "INFO") as cm:
            ucp4_client._warm_one("CLIP", lambda: None)
        self.assertRegex(cm.output[0], r"CLIP warm-up took [0-9.]+ s, \+300 MB \(process 400 MB\)")


class TestPrefetch(_RecognizeTest):
    """Clip download + frame grabs start when the task arrives, on their own pool; the one
    task worker only runs the models (live 2026-11-10 08:40: 26.5 s queued behind a 39 s
    download)."""

    def _timing(self, cm):
        return next(r.getMessage() for r in cm.records if "timing ms" in r.getMessage())

    def test_prefetched_clip_and_frames_are_used(self):
        body = self._body()
        fut = concurrent.futures.Future()
        fut.set_result(ucp4_client._fetch_media("10.0.0.5", "fcecdaaabbcc", body))
        ucp4_client.clip_fetch.fetch_clip.reset_mock()
        n_extract = len(self.extracted)
        with self.assertLogs(ucp4_client.log, "INFO") as cm, \
                mock.patch.object(ucp4_client.face_recog, "build_face_result"), \
                mock.patch.object(ucp4_client.lpr, "build_lpr"):
            ucp4_client._handle_recognize("10.0.0.5", "fcecdaaabbcc", body, fut)
        ucp4_client.clip_fetch.fetch_clip.assert_not_called()
        self.assertEqual(len(self.extracted), n_extract)
        line = self._timing(cm)
        self.assertRegex(line, r"frames=3 wait=\d+ total=\d+ fetchwait=\d+ clip=\d+ frames=\d+")

    def test_failed_prefetch_fetches_again(self):
        fut = concurrent.futures.Future()
        fut.set_exception(RuntimeError("boom"))
        with self.assertLogs(ucp4_client.log, "INFO") as cm, \
                mock.patch.object(ucp4_client.face_recog, "build_face_result"), \
                mock.patch.object(ucp4_client.lpr, "build_lpr"):
            ucp4_client._handle_recognize("10.0.0.5", "fcecdaaabbcc", self._body(), fut)
        ucp4_client.clip_fetch.fetch_clip.assert_called_once()
        self.assertTrue(any("prefetch failed" in r.getMessage() for r in cm.records))
        self.assertIn("frames=3 wait=", self._timing(cm))

    def test_started_for_video_tasks_only(self):
        self.addCleanup(setattr, ucp4_client, "_prefetch_pool", None)
        fut = ucp4_client._start_prefetch("h", "m", self._body())
        self.assertEqual(len(fut.result(5)["frames"]), 3)
        self.assertIsNone(ucp4_client._start_prefetch("h", "m", self._body(ramType="image")))
        self.assertIsNone(ucp4_client._start_prefetch("h", "m", self._body(reqUrl=None)))
        with mock.patch.dict(os.environ, {"DIYKEY_PREFETCH_WORKERS": "0"}):
            self.assertIsNone(ucp4_client._start_prefetch("h", "m", self._body()))

    def test_full_queue_does_not_start_a_prefetch(self):
        with mock.patch.object(ucp4_client, "_start_prefetch") as start, \
                mock.patch.object(ucp4_client, "_submit_task", return_value=False) as submit:
            ucp4_client.handle_action({"action": "recognizeKeyFrames"}, self._body(),
                                      {"console_host": "h", "mac_nosep": "m"})
        start.assert_not_called()
        self.assertIsInstance(submit.call_args.args[2][3], ucp4_client._Prefetch)

    def test_admitted_task_starts_the_prefetch(self):
        fut = mock.Mock()
        with mock.patch.object(ucp4_client, "_start_prefetch", return_value=fut) as start, \
                mock.patch.object(ucp4_client, "_submit_task", return_value=True) as submit:
            ucp4_client.handle_action({"action": "recognizeKeyFrames"}, self._body(),
                                      {"console_host": "h", "mac_nosep": "m"})
        start.assert_called_once()
        holder = submit.call_args.args[2][3]
        self.assertIs(holder.result(), fut.result())


class TestSummaryQueue(_RecognizeTest):
    """Live 2026-11-09: a summary was half of each event on the only task worker."""

    def test_queued_after_the_tags_are_posted(self):
        with mock.patch.object(ucp4_client, "_submit_task", return_value=True) as submit, \
                mock.patch.object(ucp4_client, "_summary_queue", return_value="Q"), \
                mock.patch.object(ucp4_client, "_post_vlm_description") as post_vlm, \
                mock.patch.object(ucp4_client, "_post_recognize_result") as post_tags:
            self._run(self._body(postVLM=True), {"DIYKEY_VLM": "mlx", "DIYKEY_VLM_QUEUE": "1"})
        post_tags.assert_called_once()
        post_vlm.assert_not_called()
        self.assertEqual(submit.call_args.args[:2], ("summary", ucp4_client._run_summary))
        self.assertEqual(submit.call_args.kwargs["q"], "Q")

    def test_off_writes_it_on_the_main_worker(self):
        with mock.patch.object(ucp4_client, "_submit_task") as submit, \
                mock.patch.object(ucp4_client, "_post_vlm_description") as post_vlm:
            self._run(self._body(postVLM=True), {"DIYKEY_VLM": "mlx"})
        post_vlm.assert_called_once()
        submit.assert_not_called()

    def test_separate_from_the_main_queue(self):
        self.assertIsNot(ucp4_client._summary_queue(), ucp4_client._task_queue())

    def test_summary_has_its_own_timing_line_even_if_it_fails(self):
        with mock.patch.object(ucp4_client, "_post_vlm_description", side_effect=RuntimeError), \
                self.assertLogs(ucp4_client.log, "INFO") as cm, self.assertRaises(RuntimeError):
            ucp4_client._run_summary("h", "m", {"event": "e", "camera": "c"}, {}, None)
        self.assertIn("event e: timing ms summary camera=c wait=", cm.output[-1])


class TestWarmModels(unittest.TestCase):
    ALL = {"DIYKEY_FACE": "1", "DIYKEY_FACE_ENHANCE": "1", "DIYKEY_LPR": "1"}

    def _run(self, env, on=True, describe=None):
        order = []
        step = lambda name: mock.Mock(side_effect=lambda *a: order.append(name))
        c = ucp4_client
        infer = step("face")
        with mock.patch.dict(os.environ, env or {}), \
                mock.patch.object(c.clip_embed, "enabled", return_value=on), \
                mock.patch.object(c.clip_embed, "text_embed", step("clip")), \
                mock.patch.object(c.clip_embed, "label_matrix", step("labels")), \
                mock.patch.object(c.vlm_describe, "enabled", return_value=on), \
                mock.patch.object(c.vlm_describe, "describe", describe or step("vlm")), \
                mock.patch.object(c.reid_model, "enabled", return_value=on), \
                mock.patch.object(c.reid_model, "embed", step("reid")), \
                mock.patch.object(c.face_recog._models, "infer_fn", return_value=infer), \
                mock.patch.object(c.face_enhance, "enhance", step("enhance")), \
                mock.patch.object(c.lpr, "lpr_available", return_value=True), \
                mock.patch.object(c.lpr, "_alpr_reader", step("lpr")), \
                mock.patch.object(c, "_warm_ram_plus", step("ram")):
            c._warm_models()
        return order

    def test_every_enabled_model_with_ram_plus_last(self):
        self.assertEqual(self._run(self.ALL),
                         ["clip", "labels", "vlm", "reid", "face", "enhance", "lpr", "ram"])

    def test_disabled_models_are_skipped(self):
        env = {k: "0" for k in self.ALL}
        self.assertEqual(self._run(env, on=False), ["ram"])

    def test_one_failure_does_not_stop_the_rest(self):
        with self.assertLogs(ucp4_client.log, "WARNING") as cm:
            order = self._run(self.ALL, describe=mock.Mock(side_effect=RuntimeError("x")))
        self.assertIn("VLM warm-up failed", cm.output[0])
        self.assertEqual(order[-1], "ram")
        self.assertIn("reid", order)


class TestWarmRamPlus(unittest.TestCase):
    def test_runs_one_blank_frame_through_ram_plus(self):
        tagger = mock.Mock()
        tagger.plus.SIZE = 384
        with mock.patch.object(ucp4_client, "_get_tagger", return_value=tagger):
            ucp4_client._warm_ram_plus()
        tagger.plus.tags.assert_called_once()
        self.assertEqual(tagger.plus.tags.call_args.args[0].shape, (384, 384, 3))

    def test_nothing_without_ram_plus(self):
        for tagger in (None, mock.Mock(plus=None)):
            with mock.patch.object(ucp4_client, "_get_tagger", return_value=tagger):
                ucp4_client._warm_ram_plus()

    def test_failure_is_only_a_warning(self):
        with mock.patch.object(ucp4_client, "_get_tagger", side_effect=RuntimeError("x")), \
                self.assertLogs(ucp4_client.log, "WARNING"):
            ucp4_client._warm_ram_plus()


class TestRecognizeFaceLpr(_RecognizeTest):
    ON ={"DIYKEY_FACE": "1", "DIYKEY_LPR": "1"}
    BOTH = {"enableFaceRecognize": True, "enableLprRecognize": True}

    def _metas(self, offset=0):
        s = 0 if offset else self.START
        return {
            "personMeta": [{"ts": s + 2500, "roi": {"coord": [1, 2, 3, 4],
                                                    "objectType": "person", "trackerId": 5}}],
            "vehicleMeta": [{"ts": s + 1000, "roi": {"coord": [5, 6, 7, 8],
                                                     "objectType": "vehicle", "trackerId": 9}}],
        }

    def test_face_and_lpr_parts_and_images(self):
        face = ({"cameraId": "cam1", "eventId": "ev1", "faceSnapshots": []}, {"100005": b"F"})
        plate = ({"cameraId": "cam1", "eventId": "ev1", "lprSnapshots": []}, {"200009": b"P"})
        self._run(self._body(**self._metas()), self.ON, self.BOTH, face, plate)
        # personMeta ts is grabbed as well, but RAM moments stay Protect's own.
        start, moments = self.extracted[0]
        self.assertIn(self.START + 2500, moments)
        files = self._files()
        self.assertEqual(set(files), {"face", "lpr", "100005", "200009"})
        self.assertEqual(files["100005"], ("100005.jpg", b"F", "image/jpeg"))
        # Epoch-ms meta ts: frames keyed by the ts Protect sent, start_ms None so
        # clockBestWall is that same epoch ms (as RAM's keyMomentMs).
        args, kw = self.face.call_args
        # One decoded dict for both models.
        self.assertEqual(set(args[2]), {self.START + 2500, self.START + 1000})
        self.assertEqual(args[2][self.START + 2500].shape, (40, 60, 3))
        self.assertIsNone(kw["start_ms"])
        self.assertIs(self.lpr.call_args.args[2], args[2])

    def test_off_without_protect_setting(self):
        # Firmware drops the metas unless the enable flag is set (syswrapper.sh:2456-2482).
        self._run(self._body(**self._metas()), self.ON, {"enableLprRecognize": False})
        self.face.assert_not_called()
        self.lpr.assert_not_called()
        self.assertNotIn(self.START + 2500, self.extracted[0][1])

    def test_off_without_env(self):
        self._run(self._body(**self._metas()), {"DIYKEY_FACE": "0", "DIYKEY_LPR": ""}, self.BOTH)
        self.face.assert_not_called()
        self.lpr.assert_not_called()

    def test_offset_ts_seeks_from_start(self):
        self._run(self._body(**self._metas(offset=True)), self.ON, self.BOTH)
        self.assertIn(self.START + 2500, self.extracted[0][1])
        args, kw = self.face.call_args
        self.assertEqual(set(args[2]), {2500, 1000})
        self.assertEqual(kw["start_ms"], self.START)

    def test_roi_array_is_flattened(self):
        metas = self._metas()
        metas["personMeta"][0]["roi"] = [metas["personMeta"][0]["roi"],
                                         {"coord": [9, 9, 9, 9], "objectType": "person",
                                          "trackerId": 6}]
        self._run(self._body(**metas), self.ON, self.BOTH)
        meta = self.face.call_args.args[3]
        self.assertEqual([m["roi"]["trackerId"] for m in meta], [5, 6])

    def _roi_body(self, ram_type="videoWithRecognition", **extra):
        s = self.START
        roi = [{"name": "a", "coord": [1, 2, 3, 4], "trackerID": 5, "trackerId": 5,
                "attributes": {"objectType": "person"}, "confidence": 90, "objectType": None},
               {"name": "b", "coord": [5, 6, 7, 8], "trackerId": 9, "objectType": "car"}]
        return self._body(ramType=ram_type, roiMeta=[{"ts": s + 2500, "roi": roi}], **extra)

    def test_roi_meta_derives_person_and_vehicle(self):
        with self.assertLogs("ucp4-client", "INFO") as cm:
            self._run(self._roi_body(), self.ON, self.BOTH)
        self.assertTrue(any("derived 1 person / 1 vehicle" in m for m in cm.output))
        self.assertEqual(self.face.call_args.args[3],
                         [{"ts": self.START + 2500,
                           "roi": {"coord": [1, 2, 3, 4], "objectType": "person",
                                   "trackerId": 5,
                                   "attributes": {"objectType": "person"},
                                   "confidence": 90}}])
        self.assertEqual(self.lpr.call_args.args[3][0]["roi"]["objectType"], "car")

    def test_roi_meta_ignored_for_plain_video(self):
        self._run(self._roi_body("video"), self.ON, self.BOTH)
        self.face.assert_not_called()
        self.lpr.assert_not_called()

    def test_video_with_recognition_ignores_enable_flags(self):
        # Live 2026-11-07 7.3.74: Legacy Camera Enhancement on, flags still False.
        off = {"enableFaceRecognize": False, "enableLprRecognize": False}
        self._run(self._roi_body(), self.ON, off)
        self.face.assert_called_once()
        self.lpr.assert_called_once()

    def test_video_with_recognition_still_needs_env_keys(self):
        self._run(self._roi_body(), {"DIYKEY_FACE": "0", "DIYKEY_LPR": "0"}, self.BOTH)
        self.face.assert_not_called()
        self.lpr.assert_not_called()

    def test_plain_video_respects_gates(self):
        self._run(self._body(**self._metas()), self.ON, {"enableFaceRecognize": False,
                                                         "enableLprRecognize": True})
        self.face.assert_not_called()
        self.lpr.assert_called_once()
        self._run(self._body(ramType="video", **self._metas()), self.ON,
                  {"enableFaceRecognize": False, "enableLprRecognize": False})
        self.assertEqual(self.face.call_count, 0)

    def test_explicit_person_meta_wins_over_roi_meta(self):
        metas = self._metas()
        metas["personMeta"][0]["roi"]["trackerId"] = 77
        self._run(self._roi_body(**metas), self.ON, self.BOTH)
        self.assertEqual(self.face.call_args.args[3][0]["roi"]["trackerId"], 77)
        # vehicleMeta also explicit here.
        self.assertEqual(self.lpr.call_args.args[3][0]["roi"]["trackerId"], 9)

    def test_model_failure_keeps_other_part(self):
        plate = ({"cameraId": "cam1", "eventId": "ev1", "lprSnapshots": []}, {})
        ucp4_client._ai_settings.update(self.BOTH)
        with mock.patch.dict(os.environ, self.ON), \
                mock.patch.object(ucp4_client.face_recog, "build_face_result",
                                  side_effect=RuntimeError("boom")), \
                mock.patch.object(ucp4_client.lpr, "build_lpr", return_value=plate), \
                self.assertLogs("ucp4-client", "WARNING"):
            ucp4_client._handle_recognize("10.0.0.5", "fcecdaaabbcc", self._body(**self._metas()))
        self.assertEqual(set(self._files()), {"lpr"})

    def test_image_collision_keeps_first(self):
        images = {"7": b"ram"}
        with self.assertLogs("ucp4-client", "WARNING") as cm:
            ucp4_client._merge_images(images, {7: b"face", 8: b"f8"}, "face", "ev1")
        self.assertEqual(images, {"7": b"ram", "8": b"f8"})
        self.assertIn("collides", cm.output[0])


class TestRecognizeFaceMeta(TestRecognizeFaceLpr):
    def test_face_meta_passed_and_frame_grabbed(self):
        s = self.START
        fm = [{"ts": s + 2600, "roi": [{"coord": [1, 1, 2, 2], "trackerId": 5}]}]
        face = ({"cameraId": "cam1", "eventId": "ev1", "faceSnapshots": []}, {})
        self._run(self._body(**self._metas(), faceMeta=fm), self.ON, self.BOTH, face)
        self.assertIn(s + 2600, self.extracted[0][1])
        got = self.face.call_args.kwargs["face_meta"]
        self.assertEqual(got, [{"ts": s + 2600, "roi": {"coord": [1, 1, 2, 2], "trackerId": 5}}])
        self.assertIn(s + 2600, self.face.call_args.args[2])  # frame decoded for that ts

    def test_no_face_meta_keyword_when_absent_or_unsupported(self):
        self._run(self._body(**self._metas()), self.ON, self.BOTH)
        self.assertNotIn("face_meta", self.face.call_args.kwargs)
        fm = [{"ts": self.START + 2600, "roi": {"coord": [1, 1, 2, 2]}}]
        calls = []

        def old(camera, event, frames, meta, start_ms=None):  # no face_meta keyword yet
            calls.append(meta)
            return None
        ucp4_client._ai_settings.update(self.BOTH)
        with mock.patch.dict(os.environ, self.ON), \
                mock.patch.object(ucp4_client.face_recog, "build_face_result", old), \
                mock.patch.object(ucp4_client.lpr, "build_lpr", mock.Mock(return_value=None)):
            ucp4_client._handle_recognize("10.0.0.5", "fcecdaaabbcc",
                                          self._body(**self._metas(), faceMeta=fm))
        self.assertEqual(len(calls), 1)


class TestTimeSync(unittest.TestCase):
    def setUp(self):
        ucp4_client._time_sync_pending.clear()
        ucp4_client._time_sync_warned = False

    def test_request_shape_and_offset(self):
        ws = mock.Mock()
        ucp4_client._send_time_sync(ws)
        head, body = [r[2] for r in ucp4_client.unpack_all_records(ws.send.call_args.args[0])]
        self.assertEqual((head["type"], head["action"]), ("request", "timeSync"))
        self.assertEqual(set(body), {"t0"})
        now = int(time.time() * 1000)
        resp = {"id": head["id"]}
        with self.assertLogs("ucp4-client", "INFO") as cm:  # Protect 5 s ahead
            self.assertTrue(ucp4_client._on_time_sync_response(
                resp, {"t0": body["t0"], "t1": now + 5000, "t2": now + 5000}))
        self.assertIn("5.0 s", cm.output[0])
        self.assertFalse(ucp4_client._time_sync_pending)

    def test_other_responses_fall_through(self):
        self.assertFalse(ucp4_client._on_time_sync_response({"id": "x"}, {}))


class TestRecognizePostVlm(_RecognizeTest):
    def test_description_posted_after_result(self):
        tagger = mock.Mock()
        ram = {"cameraId": "cam1", "eventId": "ev1", "description": "a car",
               "keyMomentsTags": []}
        with mock.patch.object(ucp4_client, "_get_tagger", return_value=tagger), \
                mock.patch.object(ucp4_client.ram_tagger, "build_result",
                                  return_value=ram), \
                mock.patch.object(ucp4_client.vlm_describe, "describe",
                                  return_value="A man walks a dog.") as describe:
            self._run(self._body(postVLM=True), {"DIYKEY_VLM": "mlx"})
        self.assertEqual(self.post.call_count, 2)
        self.assertIn("keyMomentsTags", json.loads(self._files(0)["ram"][1]))
        second = self._files(1)
        self.assertEqual(set(second), {"ram"})
        self.assertEqual(json.loads(second["ram"][1]),
                         {"cameraId": "cam1", "eventId": "ev1", "status": "success",
                          "description": "A man walks a dog."})
        describe.assert_called_once()

    def test_no_second_post_without_post_vlm_or_backend(self):
        with mock.patch.object(ucp4_client.vlm_describe, "describe") as describe:
            self._run(self._body(), {"DIYKEY_VLM": "mlx"})
            self._run(self._body(postVLM=True), {"DIYKEY_VLM": "off"})
        describe.assert_not_called()

    def test_frame_choice(self):
        frames = {1: "a", 2: "b", 3: "c", 4: "d"}
        self.assertEqual(ucp4_client._vlm_frame({"thumbnailMs": [9, 3], "keyMoments": [1]},
                                                frames), "c")
        self.assertEqual(ucp4_client._vlm_frame({"keyMoments": [4, 1, 2]}, frames), "b")
        self.assertIsNone(ucp4_client._vlm_frame({"keyMoments": [7]}, frames))


class TestVlmEventContext(_RecognizeTest):
    """Live 2026-11-08 Driveway: passing cars summarised as "Red car parked in driveway"."""
    MS = 1_790_000_000_000

    def _frames(self, w=1000, h=1000):
        path = os.path.join(self.d, "still.png")
        img = self.np.full((h, w, 3), 50, self.np.uint8)
        self.cv2.imwrite(path, img)
        return {self.MS: path}

    def _meta(self, coord=(400, 400, 100, 100), otype="vehicle", key="thumbnailMeta", as_list=True):
        roi = {"coord": list(coord), "objectType": otype, "trackerID": 3}
        return {"camera": "c", "event": "e", "resUrl": "/r", "keyMoments": [self.MS],
                key: [{"ts": self.MS, "roi": [roi] if as_list else roi}]}

    def _summarise(self, body, frames=None, scene=None, env=None):
        with mock.patch.dict(os.environ, env or {}), \
                mock.patch.object(ucp4_client.vlm_describe, "describe",
                                  return_value="A car drives past.") as describe:
            ucp4_client._post_vlm_description("h", "m", body, frames or self._frames(), scene)
        return describe

    def test_describes_a_crop_three_times_the_box(self):
        describe = self._summarise(self._meta())
        img = describe.call_args.args[0]
        self.assertEqual(img.shape[:2], (300, 300))
        self.assertIn("moving vehicle", describe.call_args.kwargs["context"])
        self.assertIn("Ignore parked or stationary objects", describe.call_args.kwargs["context"])
        self.assertFalse(describe.call_args.kwargs["crop_top"])

    def test_crop_clamped_to_the_frame_and_below_the_timestamp(self):
        img = self.np.zeros((1000, 1000, 3), self.np.uint8)
        crop = ucp4_client._vlm_event_crop(img, [0, 20, 100, 100])
        self.assertEqual(crop.shape[:2], (120, 200))  # y: 100 (the band) to 220; x: 0 to 200

    def test_vehicle_at_the_edge_gets_a_tight_crop_and_keeps_the_band(self):
        img = self.np.zeros((1000, 1000, 3), self.np.uint8)
        crop = ucp4_client._vlm_event_crop(img, [0, 8, 212, 201], "truck")
        self.assertEqual(crop.shape[:2], (259, 265))  # 1.5x, y from 0, not from the band

    def test_person_at_the_edge_keeps_the_wide_crop(self):
        img = self.np.zeros((1000, 1000, 3), self.np.uint8)
        crop = ucp4_client._vlm_event_crop(img, [0, 8, 212, 201], "person")
        self.assertEqual(crop.shape[:2], (310, 424))  # 3x, y from the band at 100

    def test_vehicle_away_from_the_edge_keeps_the_wide_crop(self):
        img = self.np.zeros((1000, 1000, 3), self.np.uint8)
        crop = ucp4_client._vlm_event_crop(img, [400, 400, 100, 100], "vehicle")
        self.assertEqual(crop.shape[:2], (300, 300))

    def test_summary_of_an_edge_vehicle_uses_the_tight_crop(self):
        describe = self._summarise(self._meta(coord=(0, 8, 212, 201)))
        self.assertEqual(describe.call_args.args[0].shape[:2], (259, 265))

    def test_roi_meta_when_there_is_no_thumbnail_meta(self):
        describe = self._summarise(self._meta(key="roiMeta", as_list=False))
        self.assertEqual(describe.call_args.args[0].shape[:2], (300, 300))

    def test_prefers_a_person_then_the_largest(self):
        frames = {self.MS: "x"}
        big = {"ts": self.MS, "coord": [0, 0, 500, 500], "objectType": "vehicle"}
        person = {"ts": self.MS, "coord": [0, 0, 10, 10], "objectType": "person"}
        small = {"ts": self.MS, "coord": [0, 0, 5, 5], "objectType": "person"}
        self.assertIs(ucp4_client._vlm_trigger([big, small, person], frames), person)
        self.assertIs(ucp4_client._vlm_trigger([big], frames), big)

    def _boxes(self, *coords, otype="person"):
        return [{"ts": self.MS + i, "coord": list(c), "objectType": otype}
                for i, c in enumerate(coords)]

    def test_live_doorbell_walk_by_skips_the_blown_out_close_up(self):
        # 2026-11-08 19:36 Doorbell thumbnailMeta, in Protect's order.
        boxes = self._boxes((348, 334, 326, 589), (41, 111, 544, 880), (255, 305, 462, 681))
        frames = {t["ts"]: "x" for t in boxes}
        self.assertIs(ucp4_client._vlm_trigger(boxes, frames), boxes[0])

    def test_largest_of_the_clear_boxes(self):
        boxes = self._boxes((100, 100, 100, 100), (300, 300, 300, 300))
        self.assertIs(ucp4_client._vlm_trigger(boxes, {t["ts"]: "x" for t in boxes}), boxes[1])

    def test_a_box_at_any_edge_or_too_big_is_not_clear(self):
        for coord in ((0, 300, 100, 100), (300, 5, 100, 100), (900, 300, 95, 100),
                      (300, 900, 100, 95), (100, 100, 700, 700)):
            self.assertFalse(ucp4_client._vlm_box_is_clear(list(coord)), coord)
        self.assertTrue(ucp4_client._vlm_box_is_clear([100, 100, 500, 500]))

    def test_falls_back_to_the_largest_when_none_is_clear(self):
        boxes = self._boxes((0, 0, 100, 100), (0, 0, 300, 300))
        self.assertIs(ucp4_client._vlm_trigger(boxes, {t["ts"]: "x" for t in boxes}), boxes[1])

    def test_a_person_at_the_edge_still_beats_a_clear_vehicle(self):
        person = self._boxes((0, 0, 100, 100))
        vehicle = self._boxes((300, 300, 200, 200), otype="vehicle")
        vehicle[0]["ts"] = self.MS + 9
        frames = {self.MS: "x", self.MS + 9: "x"}
        self.assertIs(ucp4_client._vlm_trigger(vehicle + person, frames), person[0])

    def test_person_context(self):
        describe = self._summarise(self._meta(otype="person"))
        self.assertIn("moving person", describe.call_args.kwargs["context"])
        self.assertIn("what they are doing", describe.call_args.kwargs["context"])

    def test_no_usable_box_keeps_the_whole_frame(self):
        body = self._meta()
        body["thumbnailMeta"][0]["roi"][0]["coord"] = [1, 2, 0, 0]
        describe = self._summarise(body)
        self.assertEqual(describe.call_args.args[0].shape[:2], (1000, 1000))
        self.assertEqual(describe.call_args.kwargs, {})

    def test_box_without_a_frame_keeps_the_whole_frame(self):
        body = self._meta()
        body["thumbnailMeta"][0]["ts"] = self.MS + 5
        describe = self._summarise(body)
        self.assertEqual(describe.call_args.kwargs, {})

    def test_setting_off_keeps_the_whole_frame(self):
        describe = self._summarise(self._meta(), env={"DIYKEY_VLM_EVENT_CONTEXT": "0"})
        self.assertEqual(describe.call_args.args[0].shape[:2], (1000, 1000))
        self.assertEqual(describe.call_args.kwargs, {})

    def test_whole_frame_fallback_names_what_to_ignore(self):
        scene = ucp4_client.scene_memory.EventScene()
        scene.ignore = ["the parked red car"]
        body = self._meta()
        del body["thumbnailMeta"]
        describe = self._summarise(body, scene=scene)
        self.assertIn("Ignore the parked red car", describe.call_args.kwargs["context"])

    def test_changes_go_in_the_description_only(self):
        scene = ucp4_client.scene_memory.EventScene()
        scene.changes = ["The parked car has moved."]
        scene.still = ["The person is still there."]
        self._summarise(self._meta(), scene=scene)
        body = json.loads(self.post.call_args.kwargs["files"]["ram"][1])
        self.assertEqual(body, {"cameraId": "c", "eventId": "e", "status": "success",
                                "description": "The parked car has moved. A car drives past. "
                                               "The person is still there."})


class TestSceneMemoryWiring(_RecognizeTest):
    def _tagged(self, env, **observe):
        tagger = mock.Mock()
        ram = {"cameraId": "cam1", "eventId": "ev1", "description": "Car detected.",
               "keyMomentsTags": [{"keyMomentMs": 1, "tags": [{"tag": "car", "confScore": 0.9}]}]}
        with mock.patch.object(ucp4_client, "_get_tagger", return_value=tagger), \
                mock.patch.object(ucp4_client.ram_tagger, "build_result", return_value=ram), \
                mock.patch.object(ucp4_client.scene_memory, "observe", **observe) as obs:
            self._run(self._body(), env)
        return obs

    def test_a_learned_scene_filters_the_upload(self):
        scene = ucp4_client.scene_memory.EventScene()
        scene.hide = {"vehicle"}
        self._tagged({}, return_value=scene)
        sent = json.loads(self._files(0)["ram"][1])
        self.assertEqual(sent["keyMomentsTags"][0]["tags"], [])
        self.assertEqual(sent["description"], "Motion detected.")

    def test_a_crash_in_scene_memory_still_uploads(self):
        with mock.patch.object(ucp4_client.scene_memory, "_observe",
                               side_effect=RuntimeError("boom")), \
                mock.patch.dict(os.environ, {"DIYKEY_SCENE_MEMORY": "1"}):
            tagger = mock.Mock()
            ram = {"cameraId": "cam1", "eventId": "ev1", "description": "Car detected.",
                   "keyMomentsTags": []}
            with mock.patch.object(ucp4_client, "_get_tagger", return_value=tagger), \
                    mock.patch.object(ucp4_client.ram_tagger, "build_result", return_value=ram), \
                    self.assertLogs("diykey.scene", "WARNING"):
                self._run(self._body(), {"DIYKEY_SCENE_MEMORY": "1"})
        self.assertEqual(json.loads(self._files(0)["ram"][1])["description"], "Car detected.")

    def test_off_by_default_the_upload_is_unchanged(self):
        tagger = mock.Mock()
        ram = {"cameraId": "cam1", "eventId": "ev1", "description": "Car detected.",
               "keyMomentsTags": []}
        with mock.patch.object(ucp4_client, "_get_tagger", return_value=tagger), \
                mock.patch.object(ucp4_client.ram_tagger, "build_result", return_value=ram):
            self._run(self._body(), {})
        tagger.detect.assert_not_called()
        self.assertEqual(json.loads(self._files(0)["ram"][1])["description"], "Car detected.")


class TestEnhanceImage(_StateDirTest):
    BODY = {"reqUrl": "/internal/aiprocessors/image/img1?type=face&camera=cam1"
                      "&smartDetectObject=obj1",
            "resUrl": "/internal/aiprocessors/image/enhanced", "imageId": "img1",
            "type": "face", "camera": "cam1", "smartDetectObject": "obj1"}

    def _run(self, env, body=None):
        def fetch(url, headers, cert, path):
            self.fetched = url
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"orig")
            return path

        post = mock.Mock(return_value=mock.Mock(status_code=200, text='{"bytes":3}'))
        self.enhance = mock.Mock(return_value=b"new")
        with mock.patch.dict(os.environ, env or {}), \
                mock.patch.object(ucp4_client.clip_fetch, "fetch_image", fetch), \
                mock.patch.object(ucp4_client.face_enhance, "enhance", self.enhance), \
                mock.patch.object(ucp4_client.requests, "post", post):
            ucp4_client._handle_enhance_image("10.0.0.5", "fcecdaaabbcc", body or self.BODY)
        return post

    def test_enhanced_upload(self):
        post = self._run({"DIYKEY_FACE_ENHANCE": "1"})
        self.assertEqual(self.fetched, "https://10.0.0.5:7444" + self.BODY["reqUrl"])
        self.enhance.assert_called_once_with(b"orig")
        url = post.call_args.args[0]
        self.assertEqual(url, "https://10.0.0.5:7444/internal/aiprocessors/image/enhanced")
        files = post.call_args.kwargs["files"]
        # syswrapper.sh upload_image: file, camera, type=face, smartDetectObject.
        self.assertEqual(files["file"], ("img1.jpg", b"new", "image/jpeg"))
        self.assertEqual(files["camera"], (None, "cam1"))
        self.assertEqual(files["type"], (None, "face", "text/plain"))
        self.assertEqual(files["smartDetectObject"], (None, "obj1", "text/plain"))
        self.assertEqual(post.call_args.kwargs["headers"]["x-ident"], "FCECDAAABBCC")
        self.assertEqual(os.listdir(os.path.join(self.d, "clips", "enhance")), [])

    def test_original_when_enhance_off(self):
        post = self._run({"DIYKEY_FACE_ENHANCE": "0"})
        self.enhance.assert_not_called()
        self.assertEqual(post.call_args.kwargs["files"]["file"][1], b"orig")

    def test_no_smart_detect_object(self):
        # Protect 400s without it (M/761379:150-170): don't fetch or post.
        with self.assertLogs("ucp4-client", "WARNING"):
            post = self._run({}, {**self.BODY, "smartDetectObject": None})
        post.assert_not_called()

    def test_action_queued_and_counted(self):
        submit = mock.Mock(return_value=True)
        device = {"console_host": "10.0.0.5", "mac_nosep": "fcecdaaabbcc"}
        with mock.patch.object(ucp4_client, "_submit_task", submit):
            self.assertEqual(ucp4_client.handle_action({"action": "enhanceImage"}, self.BODY,
                                                       device), {})
        (kind, fn, args), kw = submit.call_args
        self.assertEqual((kind, fn, kw), ("enhanceImage", ucp4_client._handle_enhance_image,
                                          {"group": "enhance"}))
        stats = {"queued": 3, "queued_ram": 1, "queued_enhance": 2, "running": 1,
                 "running_ram": 0, "running_enhance": 1, "dropped": 0}
        with mock.patch.object(ucp4_client, "_task_stats", stats):
            reply = ucp4_client.handle_action({"action": "getTaskQueueInfo"}, {}, device)
        self.assertEqual((reply["UI_MANUAL_FACE_ENHANCE"], reply["UI_AUTO_RAM"],
                          reply["UI_TASK_NUM"]), (3, 1, 4))


class TestEnhanceGroupCounting(unittest.TestCase):
    def test_worker_counts_group(self):
        import queue
        orig = dict(ucp4_client._task_stats)
        self.addCleanup(lambda: (ucp4_client._task_stats.clear(),
                                 ucp4_client._task_stats.update(orig)))
        for k in ucp4_client._task_stats:
            ucp4_client._task_stats[k] = 0
        ucp4_client._task_stats.update(queued=1, queued_enhance=1)
        seen = []
        q = queue.Queue()
        q.put((ucp4_client.time.monotonic(), "enhanceImage",
               lambda: seen.append(dict(ucp4_client._task_stats)), (), "enhance"))
        with mock.patch.object(q, "get", side_effect=[q.get_nowait(), SystemExit]):
            with self.assertRaises(SystemExit):
                ucp4_client._task_worker(q)
        self.assertEqual((seen[0]["running_enhance"], seen[0]["queued_enhance"]), (1, 0))
        self.assertEqual(ucp4_client._task_stats["running_enhance"], 0)


class TestOnDemandSummary(_StateDirTest):
    TS = 1_760_000_005_000
    TASK = {"targetUri": ":7968/on_demand_inference", "timeoutMs": 15000,
            "resUrl": "https://10.0.0.5:7444/camera-upload/tok",
            "payload": {"cameraId": "cam1", "eventId": "u1", "timestamp": TS,
                        "videoUrl": "/internal/aiprocessors/video/export?camera=cam1"
                                    f"&start={TS - 5000}&end={TS + 5000}&format=ubv"}}

    def _run(self, env, describe="A dog runs."):
        import cv2
        import numpy as np
        self.extracted = []

        def extract(clip, start, moments):
            self.extracted.append((start, list(moments)))
            path = os.path.join(self.d, "f.jpg")
            cv2.imwrite(path, np.zeros((20, 20, 3), np.uint8))
            return {int(moments[0]): path}

        post = mock.Mock(return_value=mock.Mock(status_code=200, text="{}"))
        self.fetch = mock.Mock(return_value="/x.mp4")
        with mock.patch.dict(os.environ, env or {}), \
                mock.patch.object(ucp4_client.clip_fetch, "fetch_clip", self.fetch), \
                mock.patch.object(ucp4_client.clip_fetch, "extract_frames", extract), \
                mock.patch.object(ucp4_client.clip_fetch, "prune", mock.Mock()), \
                mock.patch.object(ucp4_client.vlm_describe, "describe", return_value=describe), \
                mock.patch.object(ucp4_client.requests, "post", post):
            ucp4_client._on_demand_summary("10.0.0.5", "fcecdaaabbcc", self.TASK)
        return post

    def test_description_json(self):
        post = self._run({"DIYKEY_VLM": "mlx"})
        self.assertIn("format=mp4", self.fetch.call_args.args[0])
        self.assertEqual(self.extracted, [(self.TS - 5000, [self.TS])])
        self.assertEqual(post.call_args.args[0], self.TASK["resUrl"])
        # camera-upload resolves only on exactly application/json (M/854445:82-90).
        self.assertEqual(post.call_args.kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(json.loads(post.call_args.kwargs["data"]), {"description": "A dog runs."})

    def test_error_when_off_or_empty(self):
        post = self._run({"DIYKEY_VLM": "off"})
        self.fetch.assert_not_called()
        self.assertIn("error", json.loads(post.call_args.kwargs["data"]))
        post = self._run({"DIYKEY_VLM": "mlx"}, describe=None)
        self.assertEqual(json.loads(post.call_args.kwargs["data"]), {"error": "no description"})

    def test_offset_timestamp(self):
        self.assertEqual(ucp4_client._on_demand_frame_ms(self.TS, 0), self.TS)
        self.assertEqual(ucp4_client._on_demand_frame_ms(5000, self.TS - 5000), self.TS)

    def test_runs_on_own_thread_not_queue(self):
        submit, thread = mock.Mock(), mock.Mock()
        device = {"console_host": "10.0.0.5", "mac_nosep": "fcecdaaabbcc"}
        with mock.patch.object(ucp4_client, "_submit_task", submit), \
                mock.patch.object(ucp4_client.threading, "Thread", thread):
            self.assertEqual(ucp4_client.handle_action({"action": "RequestAI"}, self.TASK,
                                                       device), {})
        submit.assert_not_called()
        self.assertIs(thread.call_args.kwargs["target"], ucp4_client._fast_lane_run)
        self.assertIs(thread.call_args.kwargs["args"][0], ucp4_client._on_demand_summary)
        thread.return_value.start.assert_called_once()

    def test_fast_lane_counted_while_running(self):
        seen = []
        ucp4_client._fast_lane_run(lambda: seen.append(ucp4_client._fast_lane_count()))
        self.assertEqual((seen, ucp4_client._fast_lane_count()), ([1], 0))


class TestFastReverification(_StateDirTest):
    """fastReverification (aikey-fw syswrapper.sh:3324-3640) with downloads and model mocked."""
    URL = "/internal/aiprocessors/video/export?start=1&end=2"
    META = [{"ts": 1000, "roi": [{"objectType": "person", "trackerID": 7,
                                  "coord": [0, 0, 500, 500], "confidence": 80}]},
            {"ts": 2000, "roi": [{"objectType": "person", "trackerID": 7,
                                  "coord": [0, 0, 500, 500], "confidence": 80}]}]

    def setUp(self):
        super().setUp()
        self.post = mock.Mock(return_value=mock.Mock(status_code=200, text="{}"))
        self.fetched = []

        def fetch_image(url, headers, cert, path):
            self.fetched.append(url)
            if url.endswith("bad"):
                return None
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"jpg")
            return path

        self.tagger = object()
        self.verify = mock.Mock(side_effect=lambda t, frames, params: {
            "cameraId": params["camera"], "eventId": params["event"], "status": "success",
            "inferMs": 5, "verificationResults": [{"thumbnailMs": ts} for ts in frames]})
        for target, attr, value in (
                (ucp4_client.clip_fetch, "fetch_image", fetch_image),
                (ucp4_client.clip_fetch, "prune", mock.Mock()),
                (ucp4_client, "_get_tagger", mock.Mock(return_value=self.tagger)),
                (ucp4_client.second_verifier, "verify", self.verify),
                (ucp4_client.requests, "post", self.post)):
            p = mock.patch.object(target, attr, value)
            p.start()
            self.addCleanup(p.stop)

    def _body(self, **kw):
        return {"camera": "cam1", "event": "ev1", "start": 500, "end": 2500,
                "reqUrl": ["/internal/aiprocessors/image/a", "/internal/aiprocessors/image/b"],
                "resUrl": "/internal/aiprocessors/reverification", "ramType": "fastReverification",
                "thumbnailMs": [1000, 2000], "thumbnailMeta": self.META, **kw}

    def _run(self, body):
        ucp4_client._handle_fast_reverification("10.0.0.5", "fcecdaaabbcc", body)

    def test_image_mode_posts_reverification_field(self):
        self._run(self._body())
        self.assertEqual(self.fetched, ["https://10.0.0.5:7444/internal/aiprocessors/image/a",
                                        "https://10.0.0.5:7444/internal/aiprocessors/image/b"])
        self.assertEqual(sorted(self.verify.call_args.args[1]), [1000, 2000])
        self.assertEqual(self.verify.call_args.args[2]["score_threshold"], 0.8)
        self.assertEqual(self.post.call_args.args[0],
                         "https://10.0.0.5:7444/internal/aiprocessors/reverification")
        kw = self.post.call_args.kwargs
        self.assertEqual(kw["headers"], {"x-ident": "FCECDAAABBCC", "x-type": "UP-AI-KEY",
                                         "x-sysid": "0xa5f0"})
        self.assertEqual(list(kw["files"]), ["reverification"])
        name, payload, ctype = kw["files"]["reverification"]
        self.assertEqual(ctype, "application/json")
        self.assertEqual(json.loads(payload)["eventId"], "ev1")
        self.assertIn("cert", kw)

    def test_failed_download_and_missing_meta_are_skipped(self):
        body = self._body(reqUrl=["/x/bad", "/x/ok"])
        self._run(body)
        self.assertEqual(sorted(self.verify.call_args.args[1]), [2000])
        self._run(self._body(thumbnailMeta=self.META[:1]))
        self.assertEqual(sorted(self.verify.call_args.args[1]), [1000])

    def test_length_mismatch_aborts_without_post(self):
        with self.assertLogs("ucp4-client", "WARNING"):
            self._run(self._body(reqUrl=["/x/a"]))
        self.post.assert_not_called()

    def test_video_mode_extracts_frames(self):
        def extract(clip, start, moments):
            self.extract_args = (clip, start, moments)
            return {}
        with mock.patch.object(ucp4_client.clip_fetch, "fetch_clip",
                               mock.Mock(return_value="/x.mp4")) as fc, \
                mock.patch.object(ucp4_client.clip_fetch, "extract_frames", extract):
            self._run(self._body(reqUrl=self.URL))
        self.assertEqual(self.extract_args, ("/x.mp4", 500, [1000, 2000]))
        self.assertTrue(fc.call_args.args[0].startswith("https://10.0.0.5:7444/internal/"))
        self.post.assert_called_once()

    def test_placeholder_without_detector(self):
        with mock.patch.object(ucp4_client, "_get_tagger", return_value=None):
            self._run(self._body())
        result = json.loads(self.post.call_args.kwargs["files"]["reverification"][1])
        self.assertEqual({r["modelRunStatus"] for r in result["verificationResults"]},
                         {"skipped"})
        self.verify.assert_not_called()

    def test_action_queued_through_task_pool(self):
        submit = mock.Mock(return_value=True)
        device = {"console_host": "10.0.0.5", "mac_nosep": "fcecdaaabbcc"}
        with mock.patch.object(ucp4_client, "_submit_task", submit):
            self.assertEqual(ucp4_client.handle_action({"action": "fastReverification"},
                                                       self._body(), device), {})
        self.assertEqual(submit.call_args.args[:2],
                         ("fastReverification", ucp4_client._handle_fast_reverification))


class TestEnhanceImageGroup(_StateDirTest):
    BODY = {"reqUrl": "/internal/aiprocessors/recognition/face/groups/g1/image?t=5",
            "resUrl": "/internal/aiprocessors/image/enhanced-group",
            "smartDetectObjectGroup": "g1", "imageUpdatedAt": 1760000000000}

    def _run(self, env, body=None):
        def fetch(url, headers, cert, path):
            self.fetched = url
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"orig")
            return path

        post = mock.Mock(return_value=mock.Mock(status_code=200, text="{}"))
        with mock.patch.dict(os.environ, env or {}), \
                mock.patch.object(ucp4_client.clip_fetch, "fetch_image", fetch), \
                mock.patch.object(ucp4_client.face_enhance, "enhance",
                                  mock.Mock(return_value=b"new")), \
                mock.patch.object(ucp4_client.requests, "post", post):
            ucp4_client._handle_enhance_image_group("10.0.0.5", "fcecdaaabbcc",
                                                    body or self.BODY)
        return post

    def test_upload_fields_follow_firmware(self):
        post = self._run({"DIYKEY_FACE_ENHANCE": "1"})
        self.assertEqual(post.call_args.args[0],
                         "https://10.0.0.5:7444/internal/aiprocessors/image/enhanced-group")
        files = post.call_args.kwargs["files"]
        # syswrapper.sh upload_image_group: file, imageUpdatedAt, type=face, group; no camera.
        self.assertEqual(files["file"][1], b"new")
        self.assertEqual(files["imageUpdatedAt"], (None, "1760000000000"))
        self.assertEqual(files["type"], (None, "face", "text/plain"))
        self.assertEqual(files["smartDetectObjectGroup"], (None, "g1", "text/plain"))
        self.assertNotIn("camera", files)

    def test_original_when_enhance_off(self):
        self.assertEqual(self._run({"DIYKEY_FACE_ENHANCE": "0"})
                         .call_args.kwargs["files"]["file"][1], b"orig")

    def test_missing_group_or_resurl_is_skipped(self):
        for bad in ({**self.BODY, "smartDetectObjectGroup": None},
                    {**self.BODY, "resUrl": None}):
            with self.assertLogs("ucp4-client", "WARNING"):
                self._run({}, bad).assert_not_called()

    def test_action_queued_counted_and_not_ack_only(self):
        self.assertNotIn("enhanceImageGroup", ucp4_client.ACK_ONLY_ACTIONS)
        submit = mock.Mock(return_value=True)
        device = {"console_host": "10.0.0.5", "mac_nosep": "fcecdaaabbcc"}
        with mock.patch.object(ucp4_client, "_submit_task", submit):
            ucp4_client.handle_action({"action": "enhanceImageGroup"}, self.BODY, device)
        (kind, fn, _args), kw = submit.call_args
        self.assertEqual((kind, fn, kw), ("enhanceImageGroup",
                                          ucp4_client._handle_enhance_image_group,
                                          {"group": "enhance"}))


class TestDeepDescribe(_StateDirTest):
    TASK = {"targetUri": ":7968/describe", "timeoutMs": 30000,
            "resUrl": "/internal/aiprocessors/descriptions/t1",
            "payload": {"camera": "cam1", "event": "ev1", "pass": "close",
                        "promptProfile": "session-v1",
                        "videos": [{"reqUrl": "/internal/aiprocessors/video/export?camera=cam1"
                                              "&start=1000&end=9000&format=ubv",
                                    "objects": [{"ts": 2000}, {"ts": 4000}, {"ts": 3000}]}]}}

    def _run(self, task=None, text="A person walks.", env=None, tags=None, vec=None,
             clip="/c.mp4", frame=True):
        import numpy as np
        post = mock.Mock(return_value=mock.Mock(status_code=200, text="{}"))
        tagger = mock.Mock()
        tagger.tag_frame.return_value = [{"tag": t} for t in tags or []]
        with mock.patch.dict(os.environ, env or {}), \
                mock.patch.object(ucp4_client.clip_fetch, "fetch_clip",
                                  mock.Mock(return_value=clip)) as fc, \
                mock.patch.object(ucp4_client.clip_fetch, "extract_frames",
                                  mock.Mock(return_value={3000: "/f.jpg"})) as ef, \
                mock.patch.object(ucp4_client.clip_fetch, "prune"), \
                mock.patch("cv2.imread", return_value=np.zeros((4, 4, 3), "uint8")
                           if frame else None), \
                mock.patch.object(ucp4_client.vlm_describe, "enabled", return_value=True), \
                mock.patch.object(ucp4_client.vlm_describe, "describe",
                                  mock.Mock(return_value=text)), \
                mock.patch.object(ucp4_client, "_get_tagger",
                                  mock.Mock(return_value=tagger if tags is not None else None)), \
                mock.patch.object(ucp4_client.clip_embed, "text_embed",
                                  mock.Mock(return_value=vec)), \
                mock.patch.object(ucp4_client.requests, "post", post):
            ucp4_client._handle_describe("10.0.0.5", "fcecdaaabbcc", task or self.TASK)
        self.fetch_clip, self.extract = fc, ef
        return post

    def test_describes_middle_object_frame(self):
        post = self._run(tags=["person", "night"])
        self.assertEqual(post.call_args.args[0],
                         "https://10.0.0.5:7444/internal/aiprocessors/descriptions/t1")
        self.assertEqual(self.extract.call_args.args[1:], (1000, [3000]))  # median ts
        self.assertIn("format=mp4", self.fetch_clip.call_args.args[0])
        body = post.call_args.kwargs["json"]
        self.assertEqual(body, {"camera": "cam1", "event": "ev1", "pass": "close",
                                "description": "A person walks.", "model": "diykey-vlm",
                                "labels": ["person", "night"]})

    def test_desc_embedding_needs_clip_and_opt_in(self):
        env = {"DIYKEY_CLIP": "1", "DIYKEY_DESC_EMBED": "1"}
        self.assertEqual(self._run(env=env, vec=[0.5] * 384).call_args.kwargs["json"]
                         ["descEmbedding"], [0.5] * 384)
        self.assertNotIn("descEmbedding", self._run(env={"DIYKEY_CLIP": "1"},
                                                    vec=[1.0]).call_args.kwargs["json"])
        self.assertNotIn("descEmbedding", self._run(env={"DIYKEY_DESC_EMBED": "1"},
                                                    vec=[1.0]).call_args.kwargs["json"])

    def test_desc_embedding_wrong_dimension_is_omitted(self):
        env = {"DIYKEY_CLIP": "1", "DIYKEY_DESC_EMBED": "1"}
        body = self._run(env=env, vec=[0.5] * 768).call_args.kwargs["json"]
        self.assertNotIn("descEmbedding", body)

    def test_images_variant_and_failure_reports_empty_description(self):
        task = {**self.TASK, "payload": {"camera": "cam1", "event": "ev1", "pass": "open",
                                         "images": [{"reqUrl": "/internal/aiprocessors/image/i1",
                                                     "thumbnailId": "i1"}]}}
        import numpy as np
        fetched = []

        def fetch(url, headers, cert, path):
            fetched.append(url)
            return path

        with mock.patch.object(ucp4_client.clip_fetch, "fetch_image", fetch):
            post = self._run(task=task)
        self.assertEqual(fetched, ["https://10.0.0.5:7444/internal/aiprocessors/image/i1"])
        self.assertEqual(post.call_args.kwargs["json"]["description"], "A person walks.")
        # No frame -> empty description + failed: Protect completes the task for a retry.
        body = self._run(frame=False, clip=None).call_args.kwargs["json"]
        self.assertEqual(body["description"], "")
        self.assertTrue(body["failed"])
        body = self._run(text=None).call_args.kwargs["json"]
        self.assertEqual((body["description"], body["failed"][0]["reason"]),
                         ("", "no description"))

    def test_action_dispatch(self):
        submit = mock.Mock(return_value=True)
        device = {"console_host": "10.0.0.5", "mac_nosep": "fcecdaaabbcc"}
        with mock.patch.object(ucp4_client, "_submit_task", submit):
            ucp4_client.handle_action({"action": "RequestAI"}, self.TASK, device)
            ucp4_client.handle_action(
                {"action": "RequestAI"},
                {"targetUri": ":7445/generate-embeddings", "resUrl": "/x"}, device)
        self.assertEqual([c.args[0] for c in submit.call_args_list],
                         ["describe", "generateEmbeddings"])


class TestGenerateEmbeddings(unittest.TestCase):
    def test_reports_failed_without_a_reid_model(self):
        post = mock.Mock(return_value=mock.Mock(status_code=200, text="{}"))
        task = {"targetUri": ":7445/generate-embeddings",
                "resUrl": "/internal/aiprocessors/embeddings/t9",
                "payload": {"camera": "cam1", "event": "ev1",
                            "images": [{"objectId": "o1", "thumbnailId": "i1"}]}}
        with mock.patch.object(ucp4_client.requests, "post", post):
            ucp4_client._handle_generate_embeddings("10.0.0.5", "fcecdaaabbcc", task)
        self.assertEqual(post.call_args.args[0],
                         "https://10.0.0.5:7444/internal/aiprocessors/embeddings/t9")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["embeddings"], [])
        self.assertEqual(body["failed"], [{"objectId": "o1", "reason": "no re-ID model"}])


class TestReidEmbed(unittest.TestCase):
    def _task(self):
        return {"resUrl": "/internal/aiprocessors/embeddings/t9",
                "payload": {"camera": "cam1", "event": "ev1",
                            "images": [{"objectId": "o1", "reqUrl": "/internal/aiprocessors/image/i1"},
                                       {"objectId": "o2", "reqUrl": "/internal/aiprocessors/image/i2"}]}}

    def test_handler_reply_shape(self):
        post = mock.Mock(return_value=mock.Mock(status_code=200, text="{}"))
        vecs = iter([[0.1] * 512, None])
        with mock.patch.dict(os.environ, {"DIYKEY_REID_MODEL": "/m.pth"}), \
                mock.patch.object(ucp4_client, "STATE_DIR", "/tmp/x"), \
                mock.patch.object(ucp4_client.clip_fetch, "fetch_image", return_value="/tmp/x.jpg"), \
                mock.patch("cv2.imread", return_value=object()), \
                mock.patch.object(ucp4_client.reid_model, "embed_raw", lambda _i: next(vecs)), \
                mock.patch.object(ucp4_client.requests, "post", post):
            ucp4_client._handle_generate_embeddings("10.0.0.5", "fcecdaaabbcc", self._task())
        body = post.call_args.kwargs["json"]
        self.assertEqual(len(body["embeddings"]), 1)
        e = body["embeddings"][0]
        self.assertEqual((e["objectId"], e["dim"], len(e["reidEmbed"])), ("o1", 512, 512))
        self.assertEqual(body["failed"], [{"objectId": "o2", "reason": "embed failed"}])


class TestRoiShapes(unittest.TestCase):
    """thumbnailMeta.roi is one object in Protect's schema; some paths send a list."""

    def test_video_objects_accepts_one_roi_object(self):
        import numpy as np
        body = {"thumbnailMeta": [{"ts": 1000,
                                   "roi": {"coord": [10, 10, 50, 50], "trackerID": 7}}]}
        with mock.patch("cv2.imread", return_value=np.zeros((100, 100, 3), np.uint8)):
            objs = ucp4_client._video_objects(body, {1000: "f.jpg"})
        self.assertEqual([o["trackerID"] for o in objs], [7])

    def test_placeholder_accepts_one_roi_object(self):
        out = ucp4_client._second_verifier_result({
            "camera": "c", "event": "e",
            "thumbnailMeta": [{"ts": 1, "roi": {"objectType": "person", "trackerID": 7}}]})
        self.assertEqual(out["verificationResults"][0]["trackerID"], 7)

    def test_second_verifier_accepts_one_roi_object(self):
        out = ucp4_client.second_verifier.verify(None, {}, {
            "event": "e",
            "thumbnailMeta": [{"ts": 1, "roi": {"objectType": "person", "trackerID": 7}}]})
        self.assertEqual(out["verificationResults"][0]["trackerID"], 7)

    def test_legacy_body_types_derive_a_vehicle(self):
        for otype in ("suv", "van", "bicycle", "pickup truck"):
            body = {"ramType": "videoWithRecognition",
                    "roiMeta": [{"ts": 1, "roi": [{"coord": [1, 2, 3, 4],
                                                   "objectType": otype, "trackerId": 9}]}]}
            derived = ucp4_client._roi_derived_metas(body)
            self.assertEqual(derived["vehicle"][0]["roi"]["objectType"], otype)


class TestVerifyWindow(unittest.TestCase):
    def test_covering_export_is_unchanged(self):
        url = "/proxy/protect/api/video/export?camera=c&start=1000&end=9000"
        self.assertEqual(ucp4_client._verify_window(url, [2000, 5000]), (url, 1000))

    def test_zero_length_export_is_widened(self):
        # Live 2026-11-08 13:48: start == end, thumbnail 4.75 s later.
        url, start = ucp4_client._verify_window("/x/export?camera=c&start=1000&end=1000", [5750])
        self.assertEqual(start, 3750)
        self.assertIn("start=3750", url)
        self.assertIn("end=7750", url)
        self.assertIn("camera=c", url)

    def test_export_without_times_is_unchanged(self):
        self.assertEqual(ucp4_client._verify_window("/x/export?camera=c", [5]), ("/x/export?camera=c", 5))


class TestRequestAiUrl(unittest.TestCase):
    def test_posts_a_relative_resurl_on_the_console(self):
        post = mock.Mock(return_value=mock.Mock(status_code=200, text="{}"))
        body = {"resUrl": "/internal/aiprocessors/reverify",
                "targetUri": ":7444/v1/models/second_verifier_mlabel/inference",
                "payload": {"action": "classify", "params": {}}}
        with mock.patch.object(ucp4_client.requests, "post", post), \
                mock.patch.object(ucp4_client, "_verify_from_clip", return_value=None):
            ucp4_client._post_request_ai_result("10.0.0.5", "fec", body)
        self.assertEqual(post.call_args.args[0],
                         "https://10.0.0.5:7444/internal/aiprocessors/reverify")

    def test_verify_from_clip_uses_the_console_host(self):
        with mock.patch.object(ucp4_client, "STATE_DIR", "/tmp/x"), \
                mock.patch.object(ucp4_client, "_get_tagger", return_value=object()), \
                mock.patch.object(ucp4_client.clip_fetch, "fetch_clip",
                                  return_value=None) as fetch:
            self.assertIsNone(ucp4_client._verify_from_clip(
                {"event": "e", "reqUrl": "/internal/x?start=1",
                 "thumbnailMeta": [{"ts": 1, "roi": {"objectType": "person", "trackerID": 7}}]},
                "fec", "10.0.0.5"))
        self.assertEqual(fetch.call_args.args[0],
                         "https://10.0.0.5:7444/internal/x?start=1&format=mp4")


class TestRequestAiModels(unittest.TestCase):
    def _post(self, target, params=None):
        post = mock.Mock(return_value=mock.Mock(status_code=200, text="{}"))
        body = {"resUrl": "/internal/aiprocessors/x", "targetUri": target,
                "payload": {"action": "classify", "params": params or {}}}
        with mock.patch.object(ucp4_client.requests, "post", post), \
                mock.patch.object(ucp4_client, "_verify_from_clip", return_value=None):
            ucp4_client._post_request_ai_result("10.0.0.5", "fec", body)
        return post.call_args.kwargs["json"]

    def test_ppe_mlabel_uses_the_verifier_schema(self):
        out = self._post(":7788/v1/models/ppe_mlabel/inference", {
            "thumbnailMeta": [{"ts": 1, "roi": {"objectType": "person", "trackerID": 7}}]})
        self.assertEqual(out["model"], "ppe_mlabel")
        self.assertEqual(out["result"]["verificationResults"][0]["trackerID"], 7)

    def test_unhandled_model_reports_an_error(self):
        out = self._post(":7788/v1/models/foo/inference")
        self.assertIn("error", out["result"])

    def test_search_reranking_reports_unsupported(self):
        out = self._post(":7445/search-reranking")
        self.assertIn("error", out["result"])


class TestSegmentImage(unittest.TestCase):
    def test_posts_results(self):
        task = {"resUrl": "/internal/aiprocessors/segments",
                "payload": {"imageUrl": "/internal/image/i1"}}
        post = mock.Mock(return_value=mock.Mock(status_code=200, text="{}"))
        with mock.patch.object(ucp4_client, "_get_tagger", return_value=object()), \
                mock.patch.object(ucp4_client.reid_model, "enabled", return_value=True), \
                mock.patch.object(ucp4_client, "STATE_DIR", "/tmp/x"), \
                mock.patch.object(ucp4_client.clip_fetch, "fetch_image", return_value=True), \
                mock.patch("cv2.imread", return_value=object()), \
                mock.patch.object(ucp4_client, "_segment_results",
                                  return_value=[{"bbox": [0, 0, 1, 1]}]), \
                mock.patch.object(ucp4_client.requests, "post", post):
            ucp4_client._segment_image("10.0.0.5", "fec", task)
        self.assertEqual(post.call_args.kwargs["json"],
                         {"results": [{"bbox": [0, 0, 1, 1]}]})
        self.assertEqual(post.call_args.args[0],
                         "https://10.0.0.5:7444/internal/aiprocessors/segments")

    def test_no_model_reports_an_error(self):
        task = {"resUrl": "/x", "payload": {"imageUrl": "/i"}}
        post = mock.Mock(return_value=mock.Mock(status_code=200, text="{}"))
        with mock.patch.object(ucp4_client, "_get_tagger", return_value=None), \
                mock.patch.object(ucp4_client.requests, "post", post):
            ucp4_client._segment_image("10.0.0.5", "fec", task)
        self.assertEqual(post.call_args.kwargs["json"], {"error": "no re-ID model"})


class TestNlSearchReply(unittest.TestCase):
    def test_image_search_absolutises_img_uri(self):
        resp = mock.Mock(status_code=200, content=b"")
        device = {"mac_nosep": "fec", "console_host": "10.0.0.5"}
        with mock.patch.object(ucp4_client.requests, "get", return_value=resp) as get, \
                mock.patch.object(ucp4_client.nl_search, "image_search",
                                  return_value={"imgEmbed": []}), \
                mock.patch("cv2.imdecode", return_value=None):
            reply = ucp4_client._nl_search_reply(
                {"action": "IMAGE_SEARCH"}, {"imgUri": "/internal/image/i1"}, device)
        self.assertEqual(reply, {"imgEmbed": []})
        self.assertEqual(get.call_args.args[0],
                         "https://10.0.0.5:7444/internal/image/i1")


if __name__ == "__main__":
    unittest.main()
