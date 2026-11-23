# SPDX-License-Identifier: AGPL-3.0-or-later
"""Idle unload: the setting, the sweep, and the VLM / face-enhance unload hooks."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3_wss"))

import face_enhance  # noqa: E402
import idle_unload  # noqa: E402
import vlm_describe  # noqa: E402


class TestSetting(unittest.TestCase):
    def test_minutes_to_seconds(self):
        self.assertEqual(idle_unload.idle_seconds({"DIYKEY_IDLE_UNLOAD_MIN": "60"}), 3600)
        self.assertEqual(idle_unload.idle_seconds({"DIYKEY_IDLE_UNLOAD_MIN": "0.5"}), 30)

    def test_off_when_unset_zero_or_bad(self):
        for v in ("", "0", "-5", "soon"):
            self.assertEqual(idle_unload.idle_seconds({"DIYKEY_IDLE_UNLOAD_MIN": v}), 0, v)
        self.assertEqual(idle_unload.idle_seconds({}), 0)

    def test_start_does_nothing_when_off(self):
        self.assertIsNone(idle_unload.start([("x", lambda *_: True)], idle_s=0))


class TestSweep(unittest.TestCase):
    def test_frees_caches_only_when_something_unloaded(self):
        with mock.patch.object(idle_unload, "free_accelerator_caches") as free:
            self.assertEqual(idle_unload.sweep([("a", lambda *_: False)], 60), [])
            free.assert_not_called()
            done = idle_unload.sweep([("a", lambda *_: True), ("b", lambda *_: False)], 60,
                                     footprint=lambda: 1000.0)
        self.assertEqual(done, ["a"])
        free.assert_called_once()

    def test_a_failing_unloader_does_not_stop_the_others(self):
        def boom(*_):
            raise RuntimeError("x")
        with mock.patch.object(idle_unload, "free_accelerator_caches"):
            self.assertEqual(idle_unload.sweep([("a", boom), ("b", lambda *_: True)], 60),
                             ["b"])


class TestVlmUnload(unittest.TestCase):
    def setUp(self):
        self._saved = dict(vlm_describe._loaded), vlm_describe._last_used
        vlm_describe._loaded.clear()
        vlm_describe._loaded[("mlx", "m", 1)] = lambda *a, **k: "x"

    def tearDown(self):
        vlm_describe._loaded.clear()
        vlm_describe._loaded.update(self._saved[0])
        vlm_describe._last_used = self._saved[1]

    def test_unloads_after_idle(self):
        vlm_describe._last_used = 100.0
        self.assertFalse(vlm_describe.unload_if_idle(60, now=150.0))
        self.assertTrue(vlm_describe.loaded())
        self.assertTrue(vlm_describe.unload_if_idle(60, now=161.0))
        self.assertFalse(vlm_describe.loaded())
        self.assertFalse(vlm_describe.unload_if_idle(60, now=161.0))  # nothing left

    def test_never_while_in_use(self):
        vlm_describe._last_used = 0.0
        with vlm_describe._lock:
            self.assertFalse(vlm_describe.unload_if_idle(60, now=1e9))
        self.assertTrue(vlm_describe.loaded())

    def test_describe_marks_use(self):
        vlm_describe._last_used = 0.0
        import numpy as np
        vlm_describe.describe(np.zeros((64, 64, 3), np.uint8), backend=lambda rgb, p: "A car.")
        self.assertGreater(vlm_describe._last_used, 0.0)

    def test_prewarm_skips_when_loaded_or_off(self):
        self.assertIsNone(vlm_describe.prewarm())  # loaded
        vlm_describe._loaded.clear()
        with mock.patch.dict(os.environ, {"DIYKEY_VLM": "off"}):
            self.assertIsNone(vlm_describe.prewarm())

    def test_prewarm_loads_when_unloaded(self):
        vlm_describe._loaded.clear()
        with mock.patch.dict(os.environ, {"DIYKEY_VLM": "mlx", "DIYKEY_VLM_MODEL": "m"}), \
                mock.patch.object(vlm_describe, "_get_backend") as get:
            vlm_describe.prewarm().join(2)
        get.assert_called_once_with("mlx", "m")


class TestFaceEnhanceUnload(unittest.TestCase):
    def setUp(self):
        self._saved = face_enhance._session, face_enhance._last_used
        face_enhance._session = object()

    def tearDown(self):
        face_enhance._session, face_enhance._last_used = self._saved

    def test_unloads_after_idle_only(self):
        face_enhance._last_used = 100.0
        self.assertFalse(face_enhance.unload_if_idle(60, now=150.0))
        self.assertTrue(face_enhance.unload_if_idle(60, now=161.0))
        self.assertIsNone(face_enhance._session)

    def test_never_while_in_use(self):
        face_enhance._last_used = 0.0
        with face_enhance._lock:
            self.assertFalse(face_enhance.unload_if_idle(60, now=1e9))
        self.assertIsNotNone(face_enhance._session)


if __name__ == "__main__":
    unittest.main()
