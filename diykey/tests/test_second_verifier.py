# SPDX-License-Identifier: AGPL-3.0-or-later
"""second_verifier: box maths and verdicts that follow the firmware's verifier (fw 2.2.8
ui-infer-gateway second_verifier.py: keep the detection unless sure of another type; kept
confidence min(det% + max(2, prob%/20), 99); "skipped" without a frame)."""
import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import second_verifier as sv  # noqa: E402


class FakeTagger:
    """detect() -> [(label, score, (x, y, w, h) px)], the same on frame and crop."""
    def __init__(self, dets):
        self.dets = dets

    def detect(self, img):
        return self.dets


class BoxTest(unittest.TestCase):
    def test_box_px_is_top_left_0_to_1000(self):
        self.assertEqual(sv.box_px([100, 200, 500, 250], 1920, 1080), (192, 216, 960, 270))

    def test_iou(self):
        self.assertAlmostEqual(sv.iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(sv.iou((0, 0, 10, 10), (5, 0, 10, 10)), 50 / 150)
        self.assertEqual(sv.iou((0, 0, 10, 10), (20, 20, 5, 5)), 0.0)
        self.assertEqual(sv.iou((0, 0, 0, 0), (0, 0, 0, 0)), 0.0)

    def test_protect_type(self):
        self.assertEqual(sv.protect_type("truck"), "vehicle")
        self.assertEqual(sv.protect_type("dog"), "animal")
        self.assertIsNone(sv.protect_type("chair"))


class VerdictTest(unittest.TestCase):
    def test_kept_confidence_formula(self):
        self.assertEqual(sv.kept_confidence(80, 0.9), 0.845)  # 80 + 90/20
        self.assertEqual(sv.kept_confidence(80, 0.2), 0.82)   # floor of 2
        self.assertEqual(sv.kept_confidence(98, 1.0), 0.99)   # capped at 99

    def test_agreeing_type_is_kept(self):
        self.assertEqual(sv.verdict("person", 70, "person", 0.6, 0.8), (True, "person", 0.73))

    def test_nothing_found_keeps_the_detection(self):
        # The old code said invalid at 0.05; the firmware has no "nothing there" class.
        self.assertEqual(sv.verdict("animal", 64, None, 0.0, 0.8), (True, "animal", 0.64))

    def test_weak_other_type_is_kept(self):
        self.assertEqual(sv.verdict("person", 70, "vehicle", 0.5, 0.8), (True, "person", 0.7))

    def test_sure_other_type_wins(self):
        self.assertEqual(sv.verdict("animal", 82, "vehicle", 0.91, 0.8), (False, "vehicle", 0.91))

    def test_override_env(self):
        with mock.patch.dict(os.environ, {"DIYKEY_VERIFY_OVERRIDE": "0.995"}):
            self.assertEqual(sv.verdict("animal", 82, "vehicle", 0.91)[0], True)
        with mock.patch.dict(os.environ, {"DIYKEY_VERIFY_OVERRIDE": "junk"}):
            self.assertEqual(sv.override_min(), 0.8)


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "f.jpg")
        cv2.imwrite(self.path, np.zeros((100, 100, 3), np.uint8))
        self.params = {"event": "e", "thumbnailMeta": [{"ts": 5, "roi": [
            {"coord": [100, 100, 500, 500], "objectType": "person", "trackerID": 3,
             "confidence": 70}]}]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_frame_is_skipped_and_kept(self):
        row = sv.verify(None, {}, self.params)["verificationResults"][0]
        self.assertEqual((row["isValidDetection"], row["detectedAs"], row["modelRunStatus"],
                          row["detectionConfidence"]), (True, "person", "skipped", 0.7))

    def test_matched_on_the_frame(self):
        tagger = FakeTagger([("person", 0.9, (10, 10, 50, 50))])
        row = sv.verify(tagger, {5: self.path}, self.params)["verificationResults"][0]
        self.assertEqual((row["isValidDetection"], row["detectedAs"], row["modelRunStatus"],
                          row["detectionConfidence"], row["iouScore"]),
                         (True, "person", "success:frame", 0.745, 1.0))

    def test_no_match_keeps(self):
        row = sv.verify(FakeTagger([]), {5: self.path}, self.params)["verificationResults"][0]
        self.assertEqual((row["isValidDetection"], row["detectedAs"], row["modelRunStatus"]),
                         (True, "person", "noMatch"))


if __name__ == "__main__":
    unittest.main()
