# SPDX-License-Identifier: AGPL-3.0-or-later
"""Face restoration for Protect's Faces page "Enhance" (ucp4 enhanceImage).

Protect dispatches ``enhanceImage`` with ``reqUrl=/internal/aiprocessors/image/
<imageId>?type=..&camera=..&smartDetectObject=..`` and ``resUrl=/internal/
aiprocessors/image/enhanced`` (M/851272_RamDispatchType_AiFeatureSettings_
AiProcessorDispatcher.js:188-206); the AI Key fetches the face JPEG, enhances it
and POSTs a JPEG back to resUrl, which internalApi stores against the object
(M/761379_internalApi.js:150-170). M = ~/Code/up/service-expanded/modules.

The firmware never fails the task: if ``enhance_image`` produces an empty file it
copies the original and uploads that instead (aikey-fw 2.2.8 ui-websocketd
syswrapper.sh:1623-1628, "Face Enhance download image" / "upload image"). This
module keeps the same contract: ``enhance()`` always returns JPEG bytes, the
enhanced face when it can, otherwise the input unchanged.

Model: GFPGAN v1.4 (TencentARC, Apache-2.0, github.com/TencentARC/GFPGAN). Use an
ONNX export of GFPGANv1.4.pth, e.g. the community export in
github.com/facefusion/facefusion-assets releases ("gfpgan_1.4.onnx") or produce
one yourself with the GFPGAN repo's torch.onnx.export at 1x3x512x512. Contract
assumed here: input 1x3x512x512 float32 RGB in [-1, 1], output the same shape and
range. Point DIYKEY_FACE_ENHANCE_MODEL at the .onnx file; unset or missing means
enhancement is a pass-through. Keep the Apache-2.0 notice with the weights.

Do NOT swap in CodeFormer or GPEN: CodeFormer is under the S-Lab License 1.0 and
GPEN's weights are research/non-commercial only; neither is compatible with
shipping alongside this AGPL project.

Pipeline: decode -> pad to square (edge replicate, so the face isn't framed by a
black border the GAN would "restore") -> resize to 512 -> run -> resize back to
the padded size -> crop the pad -> encode JPEG q=95. Protect's face crops are
small and rarely square, hence the pad rather than a stretch.

onnxruntime tries CoreMLExecutionProvider (Apple Silicon) then CPU, as
DIYKEY_FACE_ENHANCE_PROVIDER picks: "coreml" (default, CoreML's default
NeuralNetwork format), "mlprogram" (CoreML MLProgram, cached compile next to the
model) or "cpu". Measured on Mac B (M1 Pro, 16 GB) 2026-11-10, five Driveway/doorbell
face crops: the default format splits GFPGAN into 29 CoreML partitions (492 of 557
nodes), loads in 6.7 s, ends +3,568 MB and runs 1.25 s per face; MLProgram takes all
557 nodes in one partition, loads in 1.0 s from its cache (2.6 s cold), ends
+779-1,239 MB and runs 0.48 s; CPU ends +1,300 MB at 1.33 s. MLProgram output
matched CPU to within 1 grey level (PSNR > 90 dB), the default format to 0.06 mean
(FP16), none visible. On Mac A the default format added +5,284 MB at
warm-up, MLProgram +757 MB in a standalone test. docs/memory-speed-2026-11.md.
The session loads lazily on first use and is shared by threads under a lock (ucp4 tasks run
concurrently; an InferenceSession run is not something we want to interleave
with a half-finished load).
"""
import logging
import os
import threading

import cv2
import numpy as np

log = logging.getLogger("diykey.face_enhance")

SIZE = 512
JPEG_QUALITY = 95
ENV_MODEL = "DIYKEY_FACE_ENHANCE_MODEL"
ENV_PROVIDER = "DIYKEY_FACE_ENHANCE_PROVIDER"
PROVIDER_CHOICES = ("coreml", "mlprogram", "cpu")

_lock = threading.Lock()
_session = None
_load_failed = False
_key = None  # (path, mtime_ns) the current session or failure came from
_logged = set()


def _warn_once(kind, msg, *args):
    """One log line per failure kind, so a missing model doesn't spam every task."""
    if kind in _logged:
        return
    _logged.add(kind)
    log.warning(msg, *args)


def _model_key():
    """(path, mtime_ns) of the configured model; one os.stat per call. A missing
    file gives mtime None, so the key changes when the file appears."""
    path = os.environ.get(ENV_MODEL, "")
    try:
        return (path, os.stat(path).st_mtime_ns) if path else (path, None)
    except OSError:
        return (path, None)


def _providers(path, avail):
    """onnxruntime providers for DIYKEY_FACE_ENHANCE_PROVIDER (see the module docstring), or
    None for onnxruntime's own default. An unknown value warns once and means "coreml"."""
    choice = os.environ.get(ENV_PROVIDER, "coreml").strip().lower() or "coreml"
    if choice not in PROVIDER_CHOICES:
        _warn_once("provider", "face enhance: %s=%r unknown (want %s); using coreml",
                   ENV_PROVIDER, choice, "/".join(PROVIDER_CHOICES))
        choice = "coreml"
    coreml = "CoreMLExecutionProvider" in avail and choice != "cpu"
    if coreml and choice == "mlprogram":
        return [("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
            "ModelCacheDirectory": os.path.join(os.path.dirname(os.path.abspath(path)),
                                                ".coreml_cache")}),
                "CPUExecutionProvider"]
    providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
                 if p in avail and (coreml or p != "CoreMLExecutionProvider")]
    return providers or None


def _load_session():
    """Lazy ONNX load; caller holds _lock. Returns None when no usable model.
    A failed load is retried only when the model path or its mtime changes, so
    replacing the file (or fixing a missing one) takes effect without a restart."""
    global _session, _load_failed, _key
    key = _model_key()
    if key != _key:
        _key = key
        _session = None
        _load_failed = False
    if _session is not None or _load_failed:
        return _session
    path = key[0]
    if not path or not os.path.isfile(path):
        _warn_once("no_model", "face enhance: no model (%s=%r); returning originals",
                   ENV_MODEL, path)
        _load_failed = True
        return None
    try:
        import onnxruntime as ort
        _session = ort.InferenceSession(path, providers=_providers(
            path, ort.get_available_providers()))
        log.info("face enhance: loaded %s (%s)", path, _session.get_providers())
    except Exception as e:  # noqa: BLE001 - any load failure means pass-through
        _warn_once("load", "face enhance: load %s failed: %s", path, e)
        _load_failed = True
        _session = None
    return _session


def _run(session, img_bgr):
    """Enhance one decoded BGR image; returns a BGR image of the same size."""
    h, w = img_bgr.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    sq = cv2.copyMakeBorder(img_bgr, top, side - h - top, left, side - w - left,
                            cv2.BORDER_REPLICATE)
    x = cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC if side < SIZE
                   else cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    name = session.get_inputs()[0].name if hasattr(session, "get_inputs") else "input"
    out = session.run(None, {name: x})[0]
    out = np.asarray(out, dtype=np.float32).reshape(3, SIZE, SIZE).transpose(1, 2, 0)
    out = ((np.clip(out, -1.0, 1.0) + 1.0) * 127.5).round().astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    out = cv2.resize(out, (side, side), interpolation=cv2.INTER_AREA if side < SIZE
                     else cv2.INTER_CUBIC)
    return out[top:top + h, left:left + w]


def enhance(jpeg_bytes, session=None):
    """Return enhanced JPEG bytes, or ``jpeg_bytes`` unchanged on any failure
    (firmware's upload-the-original fallback, syswrapper.sh:1623-1628)."""
    try:
        arr = np.frombuffer(jpeg_bytes or b"", dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
    except Exception:  # noqa: BLE001
        img = None
    if img is None:
        _warn_once("decode", "face enhance: could not decode image (%d bytes)",
                   len(jpeg_bytes or b""))
        return jpeg_bytes
    with _lock:
        sess = session if session is not None else _load_session()
        if sess is None:
            return jpeg_bytes
        try:
            out = _run(sess, img)
        except Exception as e:  # noqa: BLE001
            _warn_once("run", "face enhance: inference failed: %s", e)
            return jpeg_bytes
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        _warn_once("encode", "face enhance: JPEG encode failed")
        return jpeg_bytes
    return buf.tobytes()
