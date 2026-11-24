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

Pipeline (aligned, the normal case): decode -> YuNet's 5 landmarks -> similarity
warp onto GFPGAN's FFHQ 512 template -> run -> warp the restored face back -> blend
it into the original crop through a feathered face mask -> encode JPEG q=95. The
output keeps the input's size. GFPGAN is trained on FFHQ-aligned faces (eyes and
mouth at fixed points); fed an unaligned crop with hair and background it invents
detail. The firmware's enhancer aligns too: ubnt_ai_infer_agent 0.0.24.4 (AI Key
2.2.8) runs a face detector, UIFaceLandmark and GetRefAnchor before its encoder/GAN
pair (outputs style_feat, enc_feat_32..256, the GFPGAN layout), then a FaceRefiner
pass we have no open equivalent for.

Fallback, when YuNet (DIYKEY_FACE_MODELS, as face_recog) is missing, finds no face
or the landmarks don't fit the template: pad to square (edge replicate, so the face
isn't framed by a black border the GAN would "restore") -> resize to 512 -> run ->
resize back -> crop the pad. DIYKEY_FACE_ENHANCE_ALIGN=0 forces the fallback.

YuNet's row order is right eye, left eye, nose, right mouth corner, left mouth
corner, the subject's right being the image's left (checked on 12 live Driveway
crops 2026-11-23: "right eye" x is always the smaller). That is the template's
image-left-first order, so the landmarks map straight on with no reorder; OpenCV's
FaceRecognizerSF.alignCrop does the same against ArcFace's template.

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
import time

import cv2
import numpy as np

log = logging.getLogger("diykey.face_enhance")

SIZE = 512
JPEG_QUALITY = 95
ENV_MODEL = "DIYKEY_FACE_ENHANCE_MODEL"
ENV_PROVIDER = "DIYKEY_FACE_ENHANCE_PROVIDER"
ENV_ALIGN = "DIYKEY_FACE_ENHANCE_ALIGN"
PROVIDER_CHOICES = ("coreml", "mlprogram", "cpu")

# GFPGAN's 5-point FFHQ template for a 512x512 face (facexlib FaceRestoreHelper, the
# helper GFPGAN's own inference uses): image-left eye, image-right eye, nose,
# image-left mouth corner, image-right mouth corner.
FFHQ_512 = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.19366],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
], dtype=np.float32)
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
# Protect only sends face objects here, so a lower floor than recognition's 0.8.
YUNET_SCORE = 0.6
# YuNet misses small faces; upscale crops whose short side is under this before detecting.
DETECT_MIN_SIDE = 320
# Landmarks that fit the template worse than this (RMS px at 512) are a profile or a bad
# detection; aligning those does more harm than the padded fallback.
MAX_ALIGN_RMS = 40.0

_lock = threading.Lock()
_session = None
_load_failed = False
_key = None  # (path, mtime_ns) the current session or failure came from
_logged = set()
_last_used = 0.0  # monotonic time of the last enhance(), for idle_unload
_yunet = None
_yunet_key = None  # (path, mtime_ns) the detector or its failure came from


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


def _align_enabled():
    return (os.environ.get(ENV_ALIGN) or "1").strip().lower() not in ("0", "false", "no", "off")


def _yunet_path():
    d = os.environ.get("DIYKEY_FACE_MODELS") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    return os.path.join(d, YUNET_FILE)


def _detector():
    """Lazy YuNet; caller holds _lock (cv2 detectors aren't re-entrant). None when the model
    is missing or fails to load; retried when its path or mtime changes."""
    global _yunet, _yunet_key
    path = _yunet_path()
    try:
        key = (path, os.stat(path).st_mtime_ns)
    except OSError:
        key = (path, None)
    if key == _yunet_key:
        return _yunet
    _yunet_key, _yunet = key, None
    if key[1] is None:
        _warn_once("no_yunet", "face enhance: no YuNet at %s; enhancing unaligned", path)
        return None
    try:
        _yunet = cv2.FaceDetectorYN.create(path, "", (320, 320), YUNET_SCORE)
    except Exception as e:  # noqa: BLE001
        _warn_once("yunet", "face enhance: YuNet load %s failed: %s; enhancing unaligned",
                   path, e)
    return _yunet


def _detect_landmarks(img_bgr):
    """5x2 float32 landmarks (YuNet order) of the most prominent face, or None."""
    det = _detector()
    if det is None:
        return None
    h, w = img_bgr.shape[:2]
    scale = max(1.0, DETECT_MIN_SIDE / float(min(h, w)))
    img = img_bgr if scale == 1.0 else cv2.resize(
        img_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    det.setInputSize((img.shape[1], img.shape[0]))
    _, found = det.detect(img)
    if found is None or len(found) == 0:
        return None
    best = max(found, key=lambda f: float(f[2] * f[3] * f[14]))
    return (np.asarray(best[4:14], dtype=np.float32).reshape(5, 2) / scale).astype(np.float32)


def _align_transform(landmarks):
    """Similarity transform crop -> FFHQ 512 template, or None when the fit is poor."""
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (5, 2) or not np.all(np.isfinite(pts)):
        return None
    m, _ = cv2.estimateAffinePartial2D(pts, FFHQ_512, method=cv2.LMEDS)
    if m is None:
        return None
    err = np.linalg.norm(cv2.transform(pts[None], m)[0] - FFHQ_512, axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))
    if rms > MAX_ALIGN_RMS:
        log.debug("face enhance: landmarks fit the template at %.1f px RMS; unaligned", rms)
        return None
    return m


def _gfpgan(session, face512_bgr):
    """One 512x512 BGR face through the model; returns 512x512 BGR uint8."""
    x = cv2.cvtColor(face512_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    name = session.get_inputs()[0].name if hasattr(session, "get_inputs") else "input"
    out = session.run(None, {name: x})[0]
    out = np.asarray(out, dtype=np.float32).reshape(3, SIZE, SIZE).transpose(1, 2, 0)
    out = ((np.clip(out, -1.0, 1.0) + 1.0) * 127.5).round().astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _run_aligned(session, img_bgr, m):
    """Warp to the template, restore, warp back and blend through a feathered mask (the
    facexlib paste_faces_to_input_image recipe, at the input's own size)."""
    h, w = img_bgr.shape[:2]
    face = cv2.warpAffine(img_bgr, m, (SIZE, SIZE), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
    restored = _gfpgan(session, face)
    inv = cv2.invertAffineTransform(m)
    back = cv2.warpAffine(restored, inv, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(np.ones((SIZE, SIZE), np.float32), inv, (w, h),
                          flags=cv2.INTER_LINEAR, borderValue=0.0)
    edge = max(1, int(np.sqrt(float(mask.sum()))) // 20)
    mask = cv2.erode(mask, np.ones((2 * edge + 1, 2 * edge + 1), np.uint8))
    mask = cv2.GaussianBlur(mask, (4 * edge + 1, 4 * edge + 1), 0)[..., None]
    out = mask * back.astype(np.float32) + (1.0 - mask) * img_bgr.astype(np.float32)
    return np.clip(out, 0, 255).round().astype(np.uint8)


def _run(session, img_bgr):
    """Enhance one decoded BGR image; returns a BGR image of the same size. Aligned when
    YuNet finds a face that fits the template, else the padded whole-crop path."""
    if _align_enabled():
        lm = _detect_landmarks(img_bgr)
        m = _align_transform(lm) if lm is not None else None
        if m is not None:
            return _run_aligned(session, img_bgr, m)
    return _run_padded(session, img_bgr)


def _run_padded(session, img_bgr):
    """Fallback: the whole crop padded to square and resized to 512."""
    h, w = img_bgr.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    sq = cv2.copyMakeBorder(img_bgr, top, side - h - top, left, side - w - left,
                            cv2.BORDER_REPLICATE)
    x = cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC if side < SIZE
                   else cv2.INTER_AREA)
    out = _gfpgan(session, x)
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
    global _last_used
    with _lock:
        _last_used = time.monotonic()
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


def unload_if_idle(idle_s, now=None):
    """Drop the ONNX session when unused for idle_s (idle_unload); True when dropped. Skips
    while an enhance holds the lock; the next enhance() loads it again (_key is kept, so a
    failed load is not retried until the file changes)."""
    global _session
    now = time.monotonic() if now is None else now
    if _session is None or now - _last_used < idle_s or not _lock.acquire(blocking=False):
        return False
    try:
        if _session is None or now - _last_used < idle_s:
            return False
        _session = None
        return True
    finally:
        _lock.release()
