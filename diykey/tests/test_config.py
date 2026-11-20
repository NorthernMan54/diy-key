# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Unit tests for core/config.py's pure logic (no hardware, no network)."""
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import config  # noqa: E402


class TestNetmask(unittest.TestCase):
    def test_common_masks(self):
        self.assertEqual(config.netmask_to_prefix("255.255.255.0"), 24)
        self.assertEqual(config.netmask_to_prefix("255.255.0.0"), 16)
        self.assertEqual(config.netmask_to_prefix("255.0.0.0"), 8)
        self.assertEqual(config.netmask_to_prefix("255.255.255.255"), 32)

    def test_invalid(self):
        self.assertIsNone(config.netmask_to_prefix(""))
        self.assertIsNone(config.netmask_to_prefix("not-a-mask"))
        self.assertIsNone(config.netmask_to_prefix("255.255.255"))
        self.assertIsNone(config.netmask_to_prefix("255.0.255.0"))


class TestMisc(unittest.TestCase):
    def test_get_bool(self):
        for v in ("1", "true", "TRUE", "yes", "on", " On "):
            self.assertTrue(config.get_bool({"x": v}, "x"), v)
        for v in ("0", "false", "no", "off", ""):
            self.assertFalse(config.get_bool({"x": v}, "x"), v)
        self.assertFalse(config.get_bool({}, "missing"))

    def test_sysid_int(self):
        self.assertEqual(config.sysid_int({"sysid": "0xa5f0"}), 0xA5F0)
        self.assertEqual(config.sysid_int({"sysid": "42480"}), 42480)

    def test_redact_payload_secrets(self):
        text = '{"token": "abc123", "password": "hunter2", "hosts": ["1.2.3.4:7442"]}'
        out = config.redact(text)
        self.assertNotIn("abc123", out)
        self.assertNotIn("hunter2", out)
        self.assertIn("1.2.3.4:7442", out)

    def test_env_overrides_win(self):
        with mock.patch.dict(os.environ, {"DIYKEY_MAC": "fcecdaaabbcc"}):
            self.assertEqual(config.load_config("/nonexistent")["mac"], "fcecdaaabbcc")

    def test_load_config_file(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "x.cfg")
        with open(path, "w") as f:
            f.write("[identity]\nhostname = from-file\nsysid = 0x1234\n")
        cfg = config.load_config(path)
        self.assertEqual(cfg["hostname"], "from-file")
        self.assertEqual(cfg["sysid"], "0x1234")
        # untouched keys keep their defaults
        self.assertEqual(cfg["platform"], "UP-AI-KEY")


class TestResolveIdentity(unittest.TestCase):
    def test_iface_precedence(self):
        self.assertEqual(config.resolve_iface("dk-main", {"iface": "eth0"}), "dk-main")
        self.assertEqual(config.resolve_iface(None, {"iface": "enP4p65s0"}), "enP4p65s0")
        self.assertIsInstance(config.resolve_iface(None, {"iface": ""}), str)

    def test_parent_precedence(self):
        self.assertEqual(config.resolve_parent_iface(
            {"parent_iface": "enP4p65s0", "iface": "eth0"}), "enP4p65s0")
        self.assertEqual(config.resolve_parent_iface(
            {"parent_iface": "", "iface": "enP4p65s0"}), "enP4p65s0")

    def test_mac_override_strips_separators(self):
        self.assertEqual(config.resolve_mac("eth0", "fc:ec:da:00:00:01", {}),
                         "FCECDA000001")
        self.assertEqual(config.resolve_mac("eth0", None, {"mac": "fc-ec-da-00-00-02"}),
                         "FCECDA000002")

    def test_mac_prepends_ubnt_oui(self):
        with mock.patch.object(config, "_iface_mac", return_value="001122334455"):
            self.assertEqual(config.resolve_mac("eth0", None, {}), "FCECDA334455")

    def test_mac_unknown(self):
        with mock.patch.object(config, "_iface_mac", return_value=""):
            self.assertEqual(config.resolve_mac("eth0", None, {}), "000000000000")

    def test_ip_precedence(self):
        with mock.patch.object(config, "_iface_ip", return_value="10.0.0.9"):
            self.assertEqual(config.resolve_ip("eth0", {}, "10.0.0.5"), "10.0.0.5")
            self.assertEqual(config.resolve_ip("eth0", {"mode": "static", "ip": "10.0.0.7"}),
                             "10.0.0.7")
            self.assertEqual(config.resolve_ip("eth0", {}, "0.0.0.0"), "10.0.0.9")
        with mock.patch.object(config, "_iface_ip", return_value=None):
            self.assertEqual(config.resolve_ip("eth0", {}, None), "0.0.0.0")

    def test_advertised_ip_env(self):
        with mock.patch.dict(os.environ, {"DIYKEY_ADVERTISE_IP": "192.0.2.50"}):
            self.assertEqual(config.advertised_ip("10.0.0.9"), "192.0.2.50")
        with mock.patch.dict(os.environ, {"DIYKEY_ADVERTISE_IP": ""}):
            self.assertEqual(config.advertised_ip("10.0.0.9"), "10.0.0.9")


class TestStateFiles(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)

    def test_atomic_write_is_private_and_complete(self):
        path = os.path.join(self.d, "sub", "state.json")
        config.atomic_write_json(path, {"hosts": ["10.0.0.5:7442"], "token": "t"})
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        with open(path) as f:
            self.assertEqual(json.load(f)["token"], "t")
        self.assertEqual([n for n in os.listdir(os.path.dirname(path)) if n.endswith(".tmp")], [])

    def test_adopt_state_lifecycle(self):
        self.assertFalse(config.is_adopted(self.d))
        self.assertIsNone(config.read_token(self.d))
        config.write_adopt_state(self.d, {
            "hosts": ["10.0.0.5:7442"], "token": "tok",
            "consoleId": "00000000-0000-4000-8000-000000000003"})
        self.assertTrue(config.is_adopted(self.d))
        self.assertEqual(config.read_token(self.d), "tok")
        self.assertEqual(config.console_id_bytes(self.d),
                         bytes.fromhex("00000000000040008000000000000003"))
        config.clear_adopt_state(self.d)
        self.assertFalse(config.is_adopted(self.d))
        # cleared state is still a readable file, not ENOENT
        with open(config.adopt_state_path(self.d)) as f:
            self.assertEqual(json.load(f), {})

    def test_hosts_alone_counts_as_adopted(self):
        config.write_adopt_state(self.d, {"hosts": ["10.0.0.5:7442"]})
        self.assertTrue(config.is_adopted(self.d))

    def test_corrupt_state_is_not_adopted(self):
        with open(os.path.join(self.d, "adopt_state.json"), "w") as f:
            f.write("{not json")
        self.assertFalse(config.is_adopted(self.d))
        self.assertEqual(config.read_adopt_state(self.d), {})

    def test_state_paths(self):
        paths = config.state_paths(self.d)
        self.assertEqual(paths["state_dir"], os.path.abspath(self.d))
        self.assertEqual(paths["cert"], os.path.join(os.path.abspath(self.d), "server.crt"))
        self.assertTrue(os.path.isdir(self.d))


class TestFirmwarePin(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)

    def test_pinned_is_none_when_empty(self):
        self.assertIsNone(config.pinned_fw_version(self.d))

    def test_pin_and_read(self):
        config.pin_fw_version(self.d, "2.2.8")
        self.assertEqual(config.pinned_fw_version(self.d), "2.2.8")

    def test_effective_uses_config_before_pin(self):
        cfg = {"fw_version": "2.2.6"}
        self.assertEqual(config.effective_fw_version(cfg, self.d), "2.2.6")
        config.pin_fw_version(self.d, "2.2.8")
        self.assertEqual(config.effective_fw_version(cfg, self.d), "2.2.8")

    def test_effective_discovery_swaps_version(self):
        cfg = {"discovery_fw_version": "aikey.rk3588.v2.2.6.0.emu.260901.0000"}
        self.assertEqual(config.effective_discovery_fw_version(cfg, self.d),
                         "aikey.rk3588.v2.2.6.0.emu.260901.0000")
        config.pin_fw_version(self.d, "2.2.8")
        self.assertEqual(config.effective_discovery_fw_version(cfg, self.d),
                         "aikey.rk3588.v2.2.8.0.emu.260901.0000")


if __name__ == "__main__":
    unittest.main()


class TestDebugLoggers(unittest.TestCase):
    def test_named_loggers_get_debug(self):
        import logging
        from unittest import mock
        from diykey.core import config
        with mock.patch.dict(os.environ, {"DIYKEY_DEBUG_LOGGERS": "diykey.t1, diykey.t2"}):
            config.apply_logging()
        self.assertEqual(logging.getLogger("diykey.t1").level, logging.DEBUG)
        self.assertEqual(logging.getLogger("diykey.t2").level, logging.DEBUG)
        self.assertEqual(logging.getLogger("diykey.t3").level, logging.NOTSET)
