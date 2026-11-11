# SPDX-License-Identifier: AGPL-3.0-or-later
"""face_enhance: shape/range contract and the firmware's return-the-original
fallback (aikey-fw syswrapper.sh:1623-1628)."""
import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from stage3_wss import face_enhance as fe


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
    fe._logged.clear()


class FaceEnhanceTest(unittest.TestCase):
    def setUp(self):
        _reset()

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


class ProviderChoiceTest(unittest.TestCase):
    """DIYKEY_FACE_ENHANCE_PROVIDER: Saul 2026-11-10, MLProgram ran GFPGAN as one CoreML
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
