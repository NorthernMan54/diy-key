# SPDX-License-Identifier: AGPL-3.0-or-later
"""face_enhance: shape/range contract and the firmware's return-the-original
fallback (aikey-fw syswrapper.sh:1623-1628)."""
import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

from stage3_wss import face_enhance as fe  # noqa: E402

HAS_ORT = importlib.util.find_spec("onnxruntime") is not None


def _jpeg(w, h):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = (30, 120, 220)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class _Input:
    name = "input"


class FakeSession:
    """Checks the GFPGAN input contract, then inverts (or echoes) it."""

    def __init__(self, invert=True, fail=False):
        self.invert, self.fail, self.calls = invert, fail, []

    def get_inputs(self):
        return [_Input()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, outputs, feeds):
        if self.fail:
            raise RuntimeError("boom")
        x = feeds["input"]
        self.calls.append(x)
        return [-x if self.invert else x]


def _reset():
    fe._session, fe._load_failed, fe._key = None, False, None
    fe._yunet, fe._yunet_key = None, None
    fe._logged.clear()


def _face_img(w=200, h=240):
    """Flat background with a bright face-sized ellipse; landmarks placed like a frontal face."""
    img = np.full((h, w, 3), (40, 90, 40), np.uint8)
    cx, cy = w // 2, h // 2
    cv2.ellipse(img, (cx, cy), (45, 60), 0, 0, 360, (150, 170, 210), -1)
    lm = np.array([[cx - 20, cy - 15], [cx + 20, cy - 15], [cx, cy + 5],
                   [cx - 17, cy + 28], [cx + 17, cy + 28]], np.float32)
    return img, lm


class FaceEnhanceTest(unittest.TestCase):
    """The padded whole-crop path (no face found)."""

    def setUp(self):
        _reset()
        p = mock.patch.object(fe, "_detect_landmarks", return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        _reset()

    def test_output_is_jpeg_same_size_and_input_contract(self):
        sess = FakeSession()
        src = _jpeg(120, 160)
        out = fe.enhance(src, session=sess)
        self.assertTrue(out.startswith(b"\xff\xd8"))
        img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape, (160, 120, 3))
        x = sess.calls[0]
        self.assertEqual((x.shape, x.dtype), ((1, 3, 512, 512), np.float32))
        self.assertGreaterEqual(x.min(), -1.0)
        self.assertLessEqual(x.max(), 1.0)

    def test_identity_session_roughly_preserves_pixels(self):
        src = _jpeg(200, 100)
        out = fe.enhance(src, session=FakeSession(invert=False))
        a = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR).astype(int)
        b = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR).astype(int)
        self.assertEqual(a.shape, b.shape)
        self.assertLess(np.abs(a - b).mean(), 6)

    def test_missing_model_returns_original(self):
        src = _jpeg(64, 64)
        with mock.patch.dict(os.environ, {fe.ENV_MODEL: "/nonexistent/gfpgan.onnx"}):
            self.assertIs(fe.enhance(src), src)
            self.assertIs(fe.enhance(src), src)
        self.assertIn("no_model", fe._logged)

    def test_decode_error_returns_original(self):
        bad = b"not a jpeg"
        self.assertIs(fe.enhance(bad, session=FakeSession()), bad)
        self.assertIs(fe.enhance(b"", session=FakeSession()), b"")

    def test_session_exception_returns_original(self):
        src = _jpeg(80, 90)
        self.assertIs(fe.enhance(src, session=FakeSession(fail=True)), src)
        self.assertIn("run", fe._logged)

    @unittest.skipUnless(HAS_ORT, "onnxruntime not installed")
    def test_failed_load_retries_when_model_file_changes(self):
        src = _jpeg(80, 90)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gfpgan.onnx")
            with open(path, "wb") as f:
                f.write(b"corrupt")
            sessions = [RuntimeError("bad model"), FakeSession()]

            def _make(*_a, **_k):
                s = sessions.pop(0)
                if isinstance(s, Exception):
                    raise s
                return s

            with mock.patch.dict(os.environ, {fe.ENV_MODEL: path}), \
                    mock.patch("onnxruntime.get_available_providers",
                               return_value=["CPUExecutionProvider"]), \
                    mock.patch("onnxruntime.InferenceSession", side_effect=_make) as ctor:
                # First load fails: pass-through, and no retry while the file is unchanged.
                self.assertIs(fe.enhance(src), src)
                self.assertIs(fe.enhance(src), src)
                self.assertEqual(ctor.call_count, 1)
                # Replacing the file (new mtime) triggers a fresh load that succeeds.
                with open(path, "wb") as f:
                    f.write(b"good model")
                st = os.stat(path)
                os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
                out = fe.enhance(src)
                self.assertNotEqual(out, src)
                self.assertEqual(ctor.call_count, 2)
                self.assertIsNotNone(fe._session)


class AlignedEnhanceTest(unittest.TestCase):
    """GFPGAN wants FFHQ-aligned faces: warp to the template, restore, paste back feathered."""

    def setUp(self):
        _reset()

    def tearDown(self):
        _reset()

    def test_frontal_landmarks_fit_template(self):
        _, lm = _face_img()
        m = fe._align_transform(lm)
        self.assertIsNotNone(m)
        # Eyes land near the template's eyes.
        got = cv2.transform(lm[None], m)[0]
        self.assertLess(np.abs(got[:2] - fe.FFHQ_512[:2]).max(), 25)

    def test_mirrored_or_bad_landmarks_are_rejected(self):
        _, lm = _face_img()
        profile = lm.copy()
        profile[:, 0] = lm[0, 0]  # every point on one vertical line, like a side view
        self.assertIsNone(fe._align_transform(profile))
        self.assertIsNone(fe._align_transform(np.full((5, 2), np.nan, np.float32)))
        self.assertIsNone(fe._align_transform(np.zeros((3, 2), np.float32)))

    def test_aligned_changes_face_keeps_background_and_size(self):
        img, lm = _face_img()
        sess = FakeSession()  # inverts: any pixel it touches changes a lot
        with mock.patch.object(fe, "_detect_landmarks", return_value=lm):
            out = fe._run(sess, img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(sess.calls[0].shape, (1, 3, 512, 512))
        h, w = img.shape[:2]
        diff = np.abs(out.astype(int) - img.astype(int)).sum(axis=2)
        self.assertGreater(diff[h // 2, w // 2], 100)  # the face was replaced
        self.assertEqual(diff[:8, :8].max(), 0)  # corners untouched
        self.assertEqual(diff[-8:, -8:].max(), 0)

    def test_identity_model_roughly_round_trips(self):
        img, lm = _face_img()
        with mock.patch.object(fe, "_detect_landmarks", return_value=lm):
            out = fe._run(FakeSession(invert=False), img)
        self.assertLess(np.abs(out.astype(int) - img.astype(int)).mean(), 3)

    def test_align_off_uses_padded_path(self):
        img, lm = _face_img()
        with mock.patch.object(fe, "_detect_landmarks", return_value=lm) as det, \
                mock.patch.dict(os.environ, {fe.ENV_ALIGN: "0"}):
            fe._run(FakeSession(), img)
        det.assert_not_called()

    def test_missing_yunet_warns_once_and_finds_nothing(self):
        with mock.patch.dict(os.environ, {"DIYKEY_FACE_MODELS": "/nonexistent"}):
            self.assertIsNone(fe._detect_landmarks(_face_img()[0]))
            self.assertIsNone(fe._detect_landmarks(_face_img()[0]))
        self.assertIn("no_yunet", fe._logged)


class ProviderChoiceTest(unittest.TestCase):
    """DIYKEY_FACE_ENHANCE_PROVIDER: Mac B 2026-11-10, MLProgram ran GFPGAN as one CoreML
    partition at +779-1,239 MB vs the default format's +3,568 MB, same pixels as CPU."""
    BOTH = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    def setUp(self):
        _reset()

    def tearDown(self):
        _reset()

    def _provs(self, value, avail=None):
        env = {} if value is None else {fe.ENV_PROVIDER: value}
        with mock.patch.dict(os.environ, env):
            if value is None:
                os.environ.pop(fe.ENV_PROVIDER, None)
            return fe._providers("/m/gfpgan_1.4.onnx", avail or self.BOTH)

    def test_default_is_unchanged_coreml_then_cpu(self):
        self.assertEqual(self._provs(None), self.BOTH)
        self.assertEqual(self._provs("coreml"), self.BOTH)
        self.assertEqual(self._provs(""), self.BOTH)

    def test_mlprogram_options_and_cache_next_to_model(self):
        provs = self._provs("MLProgram")
        name, opts = provs[0]
        self.assertEqual(name, "CoreMLExecutionProvider")
        self.assertEqual(opts["ModelFormat"], "MLProgram")
        self.assertEqual(opts["ModelCacheDirectory"], "/m/.coreml_cache")
        self.assertEqual(provs[1:], ["CPUExecutionProvider"])

    def test_cpu_leaves_coreml_out(self):
        self.assertEqual(self._provs("cpu"), ["CPUExecutionProvider"])

    def test_no_coreml_available_means_cpu_even_for_mlprogram(self):
        self.assertEqual(self._provs("mlprogram", ["CPUExecutionProvider"]),
                         ["CPUExecutionProvider"])

    def test_unknown_value_warns_once_and_uses_coreml(self):
        self.assertEqual(self._provs("ane"), self.BOTH)
        self.assertIn("provider", fe._logged)

    @unittest.skipUnless(HAS_ORT, "onnxruntime not installed")
    def test_load_passes_the_choice_to_onnxruntime(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gfpgan.onnx")
            open(path, "wb").close()
            with mock.patch.dict(os.environ, {fe.ENV_MODEL: path, fe.ENV_PROVIDER: "mlprogram"}), \
                    mock.patch("onnxruntime.get_available_providers", return_value=self.BOTH), \
                    mock.patch("onnxruntime.InferenceSession",
                               return_value=FakeSession()) as ctor:
                fe.enhance(_jpeg(64, 64))
            provs = ctor.call_args.kwargs["providers"]
            self.assertEqual(provs[0][1]["ModelFormat"], "MLProgram")
            self.assertEqual(provs[0][1]["ModelCacheDirectory"], os.path.join(d, ".coreml_cache"))


if __name__ == "__main__":
    unittest.main()
