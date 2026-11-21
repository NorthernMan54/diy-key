# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for stage3_wss/lpr.py with a mocked plate reader (no fast-alpr needed)."""
import importlib.util
import os
import sys
import unittest
from unittest import mock

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import lpr  # noqa: E402

HAS_CV2 = importlib.util.find_spec("cv2") is not None


def _frame(bgr=(0, 0, 200)):
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    f[:] = bgr
    return f


def _meta(ts, tid, coord=(250, 250, 500, 500), obj="vehicle"):
    return {"ts": ts, "roi": {"coord": list(coord), "objectType": obj, "trackerId": tid}}


def _reader(table):
    """Reader returning table[call index] for successive calls."""
    calls = iter(table)
    return lambda crop: next(calls)


BOX = (10, 10, 110, 40)  # 100 px wide, over PLATE_MIN_PX


def check_descriptor(tc, d):
    """Mirror of Protect's descriptor zod schema (M/315148:26-38 all required; M/116837:33-49
    coord 4 finite numbers, zones/lines number arrays)."""
    tc.assertEqual(set(d) - {"attributes"}, {"boxColor", "confidenceLevel", "coord", "coord3d",
                                             "lines", "name", "objectType", "stationary",
                                             "tag", "trackerID", "zones"})
    tc.assertEqual(len(d["coord"]), 4)
    tc.assertTrue(all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in d["coord"]))
    tc.assertEqual(d["zones"], [])
    tc.assertEqual(d["lines"], [])
    tc.assertIsInstance(d["stationary"], bool)
    tc.assertIsInstance(d["trackerID"], int)


class BuildLprTest(unittest.TestCase):
    @unittest.skipUnless(HAS_CV2, "opencv not installed")
    def test_schema(self):
        frames = {1000: _frame()}
        out = lpr.build_lpr("cam1", "ev1", frames, [_meta(1000, 7)], start_ms=1_700_000_000_000,
                            reader=_reader([[("abc123", 0.9, BOX)]]))
        self.assertIsNotNone(out)
        j, jpegs = out
        for k, t in (("cameraId", str), ("eventId", str), ("eventTracks", list),
                     ("lprAttrs", dict), ("lprSnapshots", list), ("inferMs", int),
                     ("preProcessMs", int), ("status", str), ("timeElapsedMs", int)):
            self.assertIsInstance(j[k], t, k)
        self.assertEqual(j["status"], "success")
        self.assertEqual(len(j["eventTracks"]), 1)
        trk, = j["eventTracks"]
        self.assertEqual(trk["timestamp"], 1000)
        d, = trk["descriptors"]
        check_descriptor(self, d)
        self.assertEqual(d["objectType"], "licensePlate")
        self.assertEqual(d["name"], "ABC123")
        self.assertEqual(d["trackerID"], 7 + lpr.PLATE_TRACKER_OFFSET)
        self.assertEqual(d["confidenceLevel"], 90)
        a = j["lprAttrs"]["7"]
        self.assertEqual(a["matchedName"], "ABC123")
        # Only the vehicle carries the plate: no separate plate object, so no entry for it.
        self.assertEqual(list(j["lprAttrs"]), ["7"])
        self.assertEqual(a["namesTopK"], ["ABC123"])
        self.assertEqual(a["objectType"], "vehicle")
        self.assertEqual(a["topKCandidate"], [{"name": "ABC123", "confidence": 90.0}])
        self.assertEqual(a["color"]["val"], "red")  # BGR (0,0,200) is red
        self.assertNotIn("vehicleType", a)          # objectType alone gives no type
        # No plate snapshot: Protect would save it as a licensePlate object that is never
        # linked to the plate's named group, an unnamed second card (Driveway 2026-11-21).
        self.assertEqual(j["lprSnapshots"], [])
        self.assertEqual(jpegs, {})
        # attributes.confidence is the card's confidence; no roi confidence -> the plate read's.
        self.assertEqual(a["confidence"], 90)

    def test_best_read_per_tracker(self):
        frames = {1: _frame(), 2: _frame(), 3: _frame()}
        meta = [_meta(1, 5), _meta(2, 5), _meta(3, 9)]
        reads = [[("AB1234", 0.6, BOX)], [("AB1284", 0.8, BOX), ("XY", 0.99, BOX)],
                 [("ZZZ999", 0.7, BOX)]]
        j, jpegs = lpr.build_lpr("c", "e", frames, meta, reader=_reader(reads))
        self.assertEqual(j["lprAttrs"]["5"]["matchedName"], "AB1284")
        self.assertEqual(j["lprAttrs"]["5"]["namesTopK"], ["AB1284", "AB1234"])
        self.assertEqual(j["lprAttrs"]["9"]["matchedName"], "ZZZ999")
        self.assertEqual(set(j["lprAttrs"]), {"5", "9"})
        self.assertEqual((j["lprSnapshots"], jpegs), ([], {}))

    def test_narrow_plate_and_non_vehicle_skipped(self):
        frames = {1: _frame(), 2: _frame()}
        meta = [_meta(1, 1), _meta(2, 2, obj="person")]
        out = lpr.build_lpr("c", "e", frames, meta,
                            reader=_reader([[("ABC123", 0.9, (0, 0, 30, 10))]]))
        self.assertIsNone(out)

    def test_legacy_roi_vehicle_types_are_read(self):
        """roiMeta names a vehicle by body type (car/truck/...), not always "vehicle"."""
        for obj in ("vehicle", "car", "truck", "bus", "motorcycle", "bicycle"):
            with self.subTest(obj=obj):
                out = lpr.build_lpr("c", "e", {1: _frame()}, [_meta(1, 7, obj=obj)],
                                    reader=_reader([[("ABC123", 0.9, BOX)]]))
                self.assertIsNotNone(out)
                self.assertEqual(out[0]["lprAttrs"]["7"]["matchedName"], "ABC123")

    def test_other_object_types_are_not_read(self):
        for obj in ("person", "animal", "dog", "crack"):
            with self.subTest(obj=obj):
                self.assertIsNone(lpr.build_lpr("c", "e", {1: _frame()},
                                                [_meta(1, 7, obj=obj)],
                                                reader=_reader([[("ABC123", 0.9, BOX)]])))

    def test_vehicle_type_and_colour_from_roi_attributes(self):
        m = _meta(1, 3)
        m["roi"]["attributes"] = {"vehicleType": {"val": "suv", "confidence": 88},
                                  "color": {"val": "silver", "confidence": 95}}
        j, _ = lpr.build_lpr("c", "e", {1: _frame()}, [m],
                             reader=_reader([[("SUV001", 0.9, BOX)]]))
        attrs = j["lprAttrs"]["3"]
        self.assertEqual(attrs["vehicleType"], {"val": "suv", "confidence": 88})
        self.assertEqual(attrs["color"], {"val": "gray", "confidence": 95})  # silver -> gray

    def test_vehicle_keeps_camera_confidence(self):
        m = _meta(1, 3)
        m["roi"]["attributes"] = {"confidence": 91}
        j, _ = lpr.build_lpr("c", "e", {1: _frame()}, [m],
                             reader=_reader([[("CLVA684", 0.874, BOX)]]))
        self.assertEqual(j["lprAttrs"]["3"]["confidence"], 91)
        self.assertEqual(j["lprAttrs"]["3"]["matchedName"], "CLVA684")

    def test_vehicle_type_from_label(self):
        m = _meta(1, 3)
        m["roi"]["label"] = "truck"
        j, _ = lpr.build_lpr("c", "e", {1: _frame()}, [m],
                             reader=_reader([[("TRK001", 0.9, BOX)]]))
        self.assertEqual(j["lprAttrs"]["3"]["vehicleType"]["val"], "truck")

    def test_none_when_empty(self):
        self.assertIsNone(lpr.build_lpr("c", "e", {}, [], reader=lambda c: []))
        self.assertIsNone(lpr.build_lpr("c", "e", {1: _frame()}, [_meta(1, 1)],
                                        reader=lambda c: []))
        # missing frame
        self.assertIsNone(lpr.build_lpr("c", "e", {}, [_meta(1, 1)],
                                        reader=lambda c: [("ABC123", 0.9, BOX)]))

    def test_none_when_deps_missing_logs_once(self):
        with mock.patch.object(lpr, "lpr_available", return_value=False), \
                mock.patch.object(lpr, "_unavailable_logged", False), \
                self.assertLogs("diykey.lpr", "WARNING") as cm:
            self.assertIsNone(lpr.build_lpr("c", "e", {1: _frame()}, [_meta(1, 1)]))
            self.assertIsNone(lpr.build_lpr("c", "e", {1: _frame()}, [_meta(1, 1)]))
        self.assertEqual(len(cm.output), 1)


class OcrConfidenceTest(unittest.TestCase):
    def test_flared_plate_with_one_unreadable_char_is_dropped(self):
        # IR flare: the detector found the plate, but one character is a guess.
        self.assertIsNone(lpr.ocr_confidence("ABC123", [0.95, 0.9, 0.12, 0.9, 0.9, 0.9], 0.4))

    def test_pad_slots_do_not_count(self):
        # Trailing pad slots are near-certain; only the text's own slots are averaged.
        probs = [0.5, 0.6, 0.7, 0.99, 0.99, 0.99, 0.99]
        self.assertAlmostEqual(lpr.ocr_confidence("ABC", probs, 0.4), 0.6)
        self.assertIsNone(lpr.ocr_confidence("ABC", [0.3, 0.99, 0.99, 0.99], 0.4))

    def test_scalar_and_empty(self):
        self.assertEqual(lpr.ocr_confidence("ABC", 0.8, 0.4), 0.8)
        self.assertIsNone(lpr.ocr_confidence("ABC", 0.2, 0.4))
        self.assertIsNone(lpr.ocr_confidence("", [0.9], 0.4))
        self.assertIsNone(lpr.ocr_confidence("ABC", [], 0.4))


if __name__ == "__main__":
    unittest.main()


class _FakeClip:
    def __init__(self, probs, enabled=True):
        self.probs, self._enabled, self.embedded = probs, enabled, 0

    def enabled(self):
        return self._enabled

    def image_embed(self, img_bgr):
        self.embedded += 1
        return [1.0, 0.0]

    def label_probs(self, vec, labels, templates=None):
        assert templates == lpr.CLIP_COLOR_TEMPLATES
        return {k: self.probs.get(k, 0.0) for k in labels}


class ClipColourTest(unittest.TestCase):
    COORD = [250, 250, 500, 500]

    def _colour(self, fake, bgr=(200, 120, 60)):
        with mock.patch.dict(sys.modules, {"clip_embed": fake}):
            return lpr.vehicle_attributes(None, _frame(bgr)[..., ::-1], self.COORD)

    def test_sky_blue_pixels_lose_to_clip_neutral_and_silver_folds_into_gray(self):
        # Sky reflected on a silver hood: pixels say blue; CLIP's silver+gray wins.
        fake = _FakeClip({"silver": 0.3, "gray": 0.25, "blue": 0.4, "white": 0.05})
        self.assertEqual(self._colour(fake)["color"], {"val": "gray", "confidence": 55})
        self.assertEqual(fake.embedded, 2)  # centre crop + padded square

    def test_clip_vivid_loses_to_pixel_neutral(self):
        # White van among autumn leaves: CLIP leans brown, pixels say gray.
        fake = _FakeClip({"brown": 0.5, "gray": 0.3, "white": 0.2})
        colour = self._colour(fake, bgr=(150, 150, 160))["color"]
        self.assertEqual(colour["val"], "gray")

    def test_clip_wins_when_both_vivid(self):
        fake = _FakeClip({"green": 0.7, "blue": 0.3})
        self.assertEqual(self._colour(fake)["color"], {"val": "green", "confidence": 70})

    def test_greyscale_ir_crop_keeps_pixel_heuristic(self):
        fake = _FakeClip({"blue": 1.0})
        self.assertEqual(self._colour(fake, bgr=(128, 128, 128))["color"]["val"], "gray")
        self.assertEqual(fake.embedded, 0)

    def test_clip_off_keeps_pixel_heuristic(self):
        fake = _FakeClip({"red": 1.0}, enabled=False)
        self.assertEqual(self._colour(fake)["color"]["val"], "blue")  # BGR (200,120,60)
        self.assertEqual(fake.embedded, 0)

    def test_roi_colour_still_wins(self):
        fake = _FakeClip({"red": 1.0})
        with mock.patch.dict(sys.modules, {"clip_embed": fake}):
            a = lpr.vehicle_attributes(None, _frame()[..., ::-1], self.COORD,
                                       {"color": {"val": "white", "confidence": 80}})
        self.assertEqual(a["color"], {"val": "white", "confidence": 80})
        self.assertEqual(fake.embedded, 0)
