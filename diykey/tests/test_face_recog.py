# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for face_recog.py: faceInferenceSchema shape, best-face selection, square
crops and face tracker ids. infer is mocked, so no model files are needed."""
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

import cv2  # noqa: E402
import face_recog  # noqa: E402


def emb(seed, norm=8.0):
    v = np.random.default_rng(seed).standard_normal(128)
    return (v / np.linalg.norm(v) * norm).astype(np.float32).reshape(1, 128)


def frame(w=640, h=480):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def meta(ts, tid, coord=(250, 100, 300, 800), obj="person"):
    return {"ts": ts, "roi": {"coord": list(coord), "objectType": obj, "trackerId": tid}}


class FaceResultTest(unittest.TestCase):
    def setUp(self):  # these use small synthetic faces; FaceMinSizeTest covers the limit
        patcher = mock.patch.object(face_recog, "FACE_MIN_PX", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_schema_keys_and_types(self):
        infer = mock.Mock(return_value=[((30, 20, 40, 50), 0.9, emb(1))])
        res = face_recog.build_face_result("cam1", "ev1", {1000: frame()}, [meta(1000, 7)],
                                           start_ms=1_700_000_000_000, infer=infer)
        self.assertIsNotNone(res)
        face_json, jpegs = res
        self.assertEqual(set(face_json), {"cameraId", "eventId", "eventTracks", "faceAttrs",
                                          "faceSnapshots", "inferMs", "preProcessMs",
                                          "status", "timeElapsedMs"})
        self.assertEqual(face_json["cameraId"], "cam1")
        self.assertEqual(face_json["eventId"], "ev1")
        trk, = face_json["eventTracks"]
        self.assertEqual(trk["timestamp"], 1000)
        d, = trk["descriptors"]
        self.assertEqual(set(d) - {"attributes"}, {"boxColor", "confidenceLevel", "coord",
                         "coord3d", "lines", "name", "objectType", "stationary", "tag",
                         "trackerID", "zones"})
        self.assertEqual(d["objectType"], "face")
        self.assertEqual(d["trackerID"], 7 + face_recog.FACE_TRACKER_OFFSET)
        self.assertEqual(len(d["coord"]), 4)
        self.assertTrue(all(0 <= v <= 1000 for v in d["coord"]))
        self.assertEqual(d["confidenceLevel"], 90)
        self.assertEqual(d["attributes"]["faceMask"]["val"], "none")
        self.assertEqual(face_json["status"], "success")
        for k in ("inferMs", "preProcessMs", "timeElapsedMs"):
            self.assertIsInstance(face_json[k], int)
        fid = 7 + face_recog.FACE_TRACKER_OFFSET
        attrs = face_json["faceAttrs"][str(fid)]
        self.assertEqual(len(attrs["faceEmbed"]), 128)
        self.assertTrue(all(isinstance(v, float) for v in attrs["faceEmbed"]))
        # Unit length, as a real AI Port's faceEmbed is (live 2026-11-21).
        self.assertAlmostEqual(float(np.linalg.norm(attrs["faceEmbed"])), 1.0, places=4)
        self.assertEqual(attrs["faceMask"], {"confidence": face_recog.FACE_MASK_CONF, "val": "none"})
        self.assertEqual(attrs["matchedName"], "")
        self.assertEqual(attrs["namesTopK"], [])
        self.assertEqual(attrs["topKCandidate"], [])
        self.assertEqual(attrs["objectType"], "face")
        self.assertIs(type(attrs["linkedPersonTrackerID"]), int)
        self.assertEqual(attrs["linkedPersonTrackerID"], 7)
        snap, = face_json["faceSnapshots"]
        self.assertEqual(snap["trackerID"], fid)
        self.assertEqual(snap["smartDetectSnapshotType"], "face")
        for k in ("smartDetectHeatmap", "smartDetectSnapshotName"):
            self.assertEqual(snap[k], "")
        # firmware local path; the multipart part name is str(trackerID) (SW:4019-4026)
        self.assertEqual(snap["smartDetectSnapshot"], f"/tmp/cam1_{fid}.jpg")
        self.assertEqual(snap["clockBestMonotonic"], 1000)
        self.assertEqual(snap["clockBestWall"], 1_700_000_000_000 + 1000)
        self.assertEqual(set(jpegs), {str(fid)})
        self.assertTrue(jpegs[str(fid)].startswith(b"\xff\xd8"))

    def test_wall_clock_without_start(self):
        infer = mock.Mock(return_value=[((30, 20, 40, 50), 0.9, emb(1))])
        face_json, _ = face_recog.build_face_result("c", "e", {5: frame()}, [meta(5, 1)],
                                                    infer=infer)
        self.assertEqual(face_json["faceSnapshots"][0]["clockBestWall"], 5)

    def test_best_face_per_tracker(self):
        e_small, e_big, e_later, e_other = emb(1), emb(2), emb(3), emb(4)
        frames = {100: frame(), 200: frame(), 300: frame()}
        results = {
            # frame 100: two faces, the bigger wins inside the frame
            100: [((10, 10, 20, 20), 0.95, e_small), ((10, 10, 60, 60), 0.85, e_big)],
            # frame 200: smaller than frame 100's best -> loses across frames
            200: [((10, 10, 30, 30), 0.99, e_later)],
            # frame 300: another tracker
            300: [((10, 10, 40, 40), 0.9, e_other)],
        }
        calls = iter([100, 200, 300])
        infer = lambda crop: results[next(calls)]  # noqa: E731
        face_json, jpegs = face_recog.build_face_result(
            "c", "e", frames, [meta(100, 3), meta(200, 3), meta(300, 9)], infer=infer)
        off = face_recog.FACE_TRACKER_OFFSET
        self.assertEqual(set(face_json["faceAttrs"]), {str(3 + off), str(9 + off)})
        self.assertEqual(face_json["faceAttrs"][str(3 + off)]["faceEmbed"],
                         face_recog._embedding_list(e_big))
        self.assertEqual(face_json["faceAttrs"][str(9 + off)]["linkedPersonTrackerID"], 9)
        snaps = {s["trackerID"]: s for s in face_json["faceSnapshots"]}
        self.assertEqual(snaps[3 + off]["clockBestMonotonic"], 100)
        self.assertEqual(len(jpegs), 2)

    def test_square_crop(self):
        infer = mock.Mock(return_value=[((5, 5, 30, 60), 0.9, emb(1))])
        # person at the frame's left edge: the square must shift, not shrink or go non-square
        _, jpegs = face_recog.build_face_result("c", "e", {1: frame(320, 240)},
                                                [meta(1, 2, coord=(0, 0, 400, 900))], infer=infer)
        img = cv2.imdecode(np.frombuffer(next(iter(jpegs.values())), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[0], img.shape[1])
        self.assertEqual(img.shape[0], round(60 * face_recog.THUMB_SCALE))

    def test_square_box_clamps_to_small_frame(self):
        x1, y1, x2, y2 = face_recog.square_box(0, 0, 100, 200, 120, 90)
        self.assertEqual((x2 - x1, y2 - y1), (90, 90))
        self.assertTrue(0 <= x1 and x2 <= 120 and 0 <= y1 and y2 <= 90)

    def test_offset_ids(self):
        infer = mock.Mock(return_value=[((30, 20, 40, 50), 0.9, emb(1))])
        face_json, jpegs = face_recog.build_face_result(
            "c", "e", {1: frame(), 2: frame()}, [meta(1, 0), meta(2, 42)], infer=infer)
        off = face_recog.FACE_TRACKER_OFFSET
        self.assertEqual(off, 100000)
        self.assertEqual(sorted(s["trackerID"] for s in face_json["faceSnapshots"]),
                         [off, 42 + off])
        self.assertEqual(set(jpegs), {str(off), str(42 + off)})

    def test_none_when_no_faces(self):
        self.assertIsNone(face_recog.build_face_result(
            "c", "e", {1: frame()}, [meta(1, 1)], infer=lambda crop: []))
        self.assertIsNone(face_recog.build_face_result("c", "e", {}, [], infer=lambda c: []))
        # frame missing for the entry
        self.assertIsNone(face_recog.build_face_result(
            "c", "e", {}, [meta(1, 1)], infer=lambda c: [((1, 1, 30, 30), 0.9, emb(1))]))

    def test_weak_embedding_and_non_person_dropped(self):
        weak = lambda crop: [((10, 10, 40, 40), 0.9, emb(1, norm=3.0))]  # noqa: E731
        self.assertIsNone(face_recog.build_face_result("c", "e", {1: frame()}, [meta(1, 1)],
                                                       infer=weak))
        good = mock.Mock(return_value=[((10, 10, 40, 40), 0.9, emb(1))])
        self.assertIsNone(face_recog.build_face_result(
            "c", "e", {1: frame()}, [meta(1, 1, obj="vehicle")], infer=good))
        good.assert_not_called()

    def test_ts_key_type_mismatch(self):
        infer = mock.Mock(return_value=[((30, 20, 40, 50), 0.9, emb(1))])
        self.assertIsNotNone(face_recog.build_face_result("c", "e", {"1000": frame()},
                                                          [meta(1000, 1)], infer=infer))

    def test_none_when_models_missing(self):
        with mock.patch.dict(os.environ, {"DIYKEY_FACE_MODELS": "/nonexistent/face-models"}), \
                mock.patch.object(face_recog, "_models", face_recog._Models()):
            with self.assertLogs("diykey.face", level="WARNING") as cm:
                self.assertIsNone(face_recog.build_face_result("c", "e", {1: frame()},
                                                               [meta(1, 1)]))
            self.assertEqual(len(cm.output), 1)
            # logged once: a second call stays quiet
            with mock.patch.object(face_recog.log, "warning") as warn:
                self.assertIsNone(face_recog.build_face_result("c", "e", {1: frame()},
                                                               [meta(1, 1)]))
                warn.assert_not_called()


class FaceMinSizeTest(unittest.TestCase):
    """Live 17:07 Backyard: a 20 px face reached Protect as a smeared new subject."""

    def _run(self, width):
        def infer(crop):
            return [((1, 1, width, width), 0.9, emb(1))]
        return face_recog.build_face_result("c", "e", {1: frame()}, [meta(1, 3)], infer=infer)

    def test_small_face_skipped(self):
        with mock.patch.object(face_recog, "FACE_MIN_PX", 40):
            self.assertIsNone(self._run(22))

    def test_large_enough_face_kept(self):
        with mock.patch.object(face_recog, "FACE_MIN_PX", 40):
            self.assertIsNotNone(self._run(60))

    def test_zero_disables(self):
        with mock.patch.object(face_recog, "FACE_MIN_PX", 0):
            self.assertIsNotNone(self._run(22))


class AlignmentQualityTest(unittest.TestCase):
    """The cheap stand-in for the AI Key's SmartFace_lmk + QA models."""

    def test_perfect_template_has_no_error(self):
        # YuNet order: right eye, left eye, nose, right mouth, left mouth
        pts = face_recog.ARCFACE_DST[[1, 0, 2, 4, 3]]
        rms, max_err = face_recog.alignment_quality(pts)
        self.assertIsNotNone(rms)
        self.assertLess(rms, 0.01)
        self.assertLess(max_err, 0.01)

    def test_moved_landmark_is_flagged(self):
        pts = face_recog.ARCFACE_DST[[1, 0, 2, 4, 3]].copy()
        pts[2, 0] += 120.0
        rms, max_err = face_recog.alignment_quality(pts)
        self.assertGreater(rms, face_recog.FACE_MAX_ALIGN_RMS)
        self.assertGreater(max_err, face_recog.FACE_MAX_LANDMARK_ERR)

    def test_bad_shape_is_rejected(self):
        self.assertEqual(face_recog.alignment_quality([1, 2, 3, 4]), (None, None))
        self.assertEqual(face_recog.alignment_quality([float("nan")] * 10), (None, None))


class FaceMetaTest(unittest.TestCase):
    def setUp(self):  # these use small synthetic faces; FaceMinSizeTest covers the limit
        patcher = mock.patch.object(face_recog, "FACE_MIN_PX", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    """Protect's camera face boxes are used directly: no YuNet, SFace-only embed."""

    def test_face_box_aligned_via_yunet_in_expanded_box(self):
        infer = mock.Mock(return_value=[((10, 10, 30, 30), 0.9, emb(2))])
        embed = mock.Mock(return_value=emb(3))
        fm = [{"ts": 1000, "roi": [{"coord": [350, 120, 100, 130], "trackerId": 55}]}]
        res = face_recog.build_face_result("c", "e", {1000: frame()}, [meta(1000, 7)],
                                           infer=infer, face_meta=fm, embed=embed)
        embed.assert_not_called()
        crop = infer.call_args[0][0]
        self.assertGreater(crop.shape[1], 64)  # expanded past the 64 px box
        self.assertEqual(res[0]["faceAttrs"][str(7 + face_recog.FACE_TRACKER_OFFSET)]
                         ["linkedPersonTrackerID"], 7)

    def test_face_box_fallback_unaligned_when_yunet_finds_nothing(self):
        # The old behaviour, now opt-in: an unaligned resize is a poor embedding, so it is
        # off unless DIYKEY_FACE_UNALIGNED=1.
        infer = mock.Mock(return_value=[])
        embed = mock.Mock(return_value=emb(3))
        fm = [{"ts": 1000, "roi": [{"coord": [350, 120, 100, 130], "trackerId": 55}]}]
        pm = [meta(1000, 7), meta(1000, 8, coord=(700, 100, 200, 800))]
        with mock.patch.object(face_recog, "FACE_UNALIGNED", False):
            self.assertIsNone(face_recog.build_face_result(
                "c", "e", {1000: frame()}, pm, infer=infer, face_meta=fm, embed=embed))
        embed.assert_not_called()
        with mock.patch.object(face_recog, "FACE_UNALIGNED", True):
            res = face_recog.build_face_result("c", "e", {1000: frame()}, pm, infer=infer,
                                               face_meta=fm, embed=embed)
        face_json, _ = res
        crop = embed.call_args[0][0]
        self.assertEqual(crop.shape[:2], (63, 64))
        fid = 7 + face_recog.FACE_TRACKER_OFFSET
        self.assertEqual(list(face_json["faceAttrs"]), [str(fid)])
        self.assertEqual(face_json["faceAttrs"][str(fid)]["linkedPersonTrackerID"], 7)
        d = face_json["eventTracks"][0]["descriptors"][0]
        for got, want in zip(d["coord"], (350, 120, 100, 130)):
            self.assertAlmostEqual(got, want, delta=3)  # pixel truncation

    def test_unlinked_box_dropped_and_other_ts_use_person_search(self):
        infer = mock.Mock(return_value=[((30, 20, 40, 50), 0.9, emb(2))])
        embed = mock.Mock(return_value=emb(3))
        fm = [{"ts": 1000, "roi": {"coord": [900, 900, 50, 50], "trackerId": 1}}]
        res = face_recog.build_face_result(
            "c", "e", {1000: frame(), 2000: frame()}, [meta(1000, 7), meta(2000, 7)],
            infer=infer, face_meta=fm, embed=embed)
        embed.assert_not_called()
        self.assertEqual(infer.call_count, 1 + 1)  # ts 1000 not covered, ts 2000 too
        self.assertIsNotNone(res)


class EmbeddingListTest(unittest.TestCase):
    def test_normalised(self):
        self.assertEqual(face_recog._embedding_list(np.array([[3.0, 4.0]], np.float32)),
                         [0.6, 0.8])

    def test_zero_vector_left_alone(self):
        self.assertEqual(face_recog._embedding_list(np.zeros(2)), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

