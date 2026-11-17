# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for stage3_wss/vlm_describe.py with a fake backend (no model loads)."""
import os
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import vlm_describe as vd  # noqa: E402


def _frame(w, h):
    f = np.zeros((h, w, 3), np.uint8)
    f[:, :, 0] = 255  # pure blue in BGR
    return f


class PreprocessTest(unittest.TestCase):
    def test_crops_top_ten_percent_and_converts_to_rgb(self):
        f = _frame(640, 480)
        f[:48] = 7  # the "timestamp" band must vanish
        out = vd.preprocess(f)
        self.assertEqual(out.shape, (432, 640, 3))
        self.assertTrue((out[:, :, 2] == 255).all())  # blue now last (RGB)
        self.assertFalse((out == 7).any())

    def test_caps_long_side_at_1280(self):
        # 3840x2160 -> crop 216 -> 3840x1944 -> x(1280/3840) -> 1280x648
        out = vd.preprocess(_frame(3840, 2160))
        self.assertEqual(out.shape[:2], (648, 1280))

    def test_small_frame_not_upscaled(self):
        self.assertEqual(vd.preprocess(_frame(320, 200)).shape[:2], (180, 320))


class PostprocessTest(unittest.TestCase):
    def test_strips_redundant_phrase_and_capitalises(self):
        self.assertEqual(vd.postprocess("The image shows a man walking a dog"),
                         "A man walking a dog.")
        self.assertEqual(vd.postprocess("The image depicts a red car."),
                         "A red car.")

    def test_dedupes_comma_phrases(self):
        self.assertEqual(vd.postprocess("a man, a car, a truck, a car, a car"),
                         "A man, a car, a truck.")
        # Firmware capitalises before deduping: the first phrase is now "A car".
        self.assertEqual(vd.postprocess("a car, a car, a car"), "A car, a car.")

    def test_empty(self):
        self.assertEqual(vd.postprocess("The image captures "), "")
        self.assertEqual(vd.postprocess(None), "")


class DescribeTest(unittest.TestCase):
    def test_off_returns_none(self):
        with mock.patch.dict(os.environ, {"DIYKEY_VLM": "off"}):
            self.assertIsNone(vd.describe(_frame(64, 64)))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(vd.describe(_frame(64, 64)))

    def test_missing_deps_returns_none(self):
        with mock.patch.dict(os.environ, {"DIYKEY_VLM": "mlx",
                                          "DIYKEY_VLM_MODEL": "x/none"}), \
                mock.patch.object(vd, "_mlx_backend", side_effect=ImportError("no")):
            vd._failed.discard(("mlx", "x/none"))
            self.assertIsNone(vd.describe(_frame(64, 64)))

    def test_fake_backend_gets_rgb_and_prompt(self):
        seen = {}

        def fake(rgb, prompt):
            seen["shape"], seen["prompt"] = rgb.shape, prompt
            return "The image features a person at the door"
        self.assertEqual(vd.describe(_frame(100, 100), backend=fake),
                         "A person at the door.")
        self.assertEqual(seen["shape"], (90, 100, 3))
        self.assertIn("30 words", seen["prompt"])

    def test_exception_returns_none(self):
        def boom(rgb, prompt):
            raise RuntimeError("MPS out of memory")
        self.assertIsNone(vd.describe(_frame(64, 64), backend=boom))

    def test_lock_serialises(self):
        active, peak, lock = [0], [0], threading.Lock()

        def slow(rgb, prompt):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return "a cat"
        results = []
        ts = [threading.Thread(target=lambda: results.append(
            vd.describe(_frame(32, 32), backend=slow))) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(peak[0], 1)
        self.assertEqual(results, ["A cat."] * 4)


class ContextTest(unittest.TestCase):
    def test_context_is_appended_for_that_call_only(self):
        prompts = []

        def backend(rgb, prompt):
            prompts.append(prompt)
            return "A car."
        f = _frame(64, 64)
        vd.describe(f, backend=backend, context="This event was triggered by a moving vehicle.")
        vd.describe(f, backend=backend)
        self.assertEqual(prompts[0], vd.PROMPT + "\nThis event was triggered by a moving vehicle.")
        self.assertEqual(prompts[1], vd.PROMPT)

    def test_a_crop_keeps_its_top(self):
        shapes = []
        f = _frame(64, 100)
        vd.describe(f, backend=lambda rgb, p: shapes.append(rgb.shape) or "A car.", crop_top=False)
        self.assertEqual(shapes[0][:2], (100, 64))
        vd.describe(f, backend=lambda rgb, p: shapes.append(rgb.shape) or "A car.")
        self.assertEqual(shapes[1][:2], (90, 64))


if __name__ == "__main__":
    unittest.main()


class TestAttributes(unittest.TestCase):
    """vd.attributes/parse_attribute_reply: Protect's describe-prompt JSON."""

    def test_parse_fenced_json_and_drop_placeholders(self):
        text = ('Here you go:\n```json\n{"description": "a man", "labels": '
                '["top:hoodie", "topColor:red", "species:species", "bare", "top:hoodie", 7]}\n```')
        self.assertEqual(vd.parse_attribute_reply(text), ["top:hoodie", "topColor:red"])

    def test_parse_garbage(self):
        for text in (None, "", "no json", "{not json}", '{"labels": "x"}', "[1]"):
            self.assertEqual(vd.parse_attribute_reply(text), [])

    def test_attributes_passes_system_and_token_limit(self):
        import numpy as np
        backend = mock.Mock(return_value='{"description": "d", "labels": ["carry:box"]}')
        out = vd.attributes(np.zeros((40, 20, 3), dtype=np.uint8), "SYS", "USER",
                                      backend=backend)
        self.assertEqual(out, ["carry:box"])
        self.assertEqual(backend.call_args.args[1], "USER")
        self.assertEqual(backend.call_args.kwargs, {"system": "SYS", "max_tokens": 384})

    def test_attributes_error_is_empty(self):
        import numpy as np
        out = vd.attributes(np.zeros((4, 4, 3), dtype=np.uint8), "S", "U",
                                      backend=mock.Mock(side_effect=RuntimeError("boom")))
        self.assertEqual(out, [])
