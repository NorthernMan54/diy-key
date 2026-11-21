# SPDX-License-Identifier: AGPL-3.0-or-later
"""Face detection + embedding for a recognizeFaces result (faceInferenceSchema).

Protect hands the AI Key a person event's key frames with per-frame person boxes
(person_meta: [{ts, roi:{coord:[x,y,w,h] 0..1000, objectType, trackerId}}], Protect
M/346308:76-100) and expects back faceInferenceSchema (M/315148:117-127): one faceAttrs entry
per face tracker carrying the 128-d faceEmbed its face service clusters on, plus one
faceSnapshots entry per face whose JPEG goes up as the face thumbnail.

Detection and embedding are ported from pi-port's diyport/enrich.py (same author, AGPL):
OpenCV zoo YuNet (face_detection_yunet_2023mar.onnx, MIT) finds faces in the top of the person
box, SFace (face_recognition_sface_2021dec.onnx, Apache-2.0) embeds the aligned face. No
gallery matching here: Protect names faces itself from the embeddings (matchedName stays "").

Models come from DIYKEY_FACE_MODELS (a directory holding both .onnx files; default
<diykey>/models). pi-port keeps the same two files in diyport/models, so pointing
DIYKEY_FACE_MODELS there works. They load lazily on the first call; without cv2 or the model
files build_face_result returns None and logs once.
"""
import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger("diykey.face")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"

COORD_END = 1000  # Protect roi coords are 0..1000 of the frame, top-left origin

# Face trackerID = person trackerId + this. Protect links the face back to its person through
# linkedPersonTrackerID (M/315148:39-58), so the face id only has to be unique within the
# event. It must also not collide with the trackerID keys our RAM result (thumbnailTags /
# searchSnapshot) uses for the same event, which are the person/vehicle tracker ids
# themselves; Protect's own tracker ids are small (tens to hundreds), so +100000 keeps clear.
# Unverified live: no capture of a real AI Key's face tracker ids yet.
FACE_TRACKER_OFFSET = 100000

# YuNet score floor; pi-port's AIPORT_FACE_THRESH default.
FACE_SCORE_THRESH = float(os.environ.get("DIYKEY_FACE_THRESH") or 0.8)
# SFace vectors shorter than this are poor faces (backs of heads 3.3-4.3, turned away 5.9,
# real faces 6.7-8.8: pi-port live 12:36-12:40 on Driveway); don't send them, a weak
# embedding makes a bad cluster in Protect. 0 keeps them all.
FACE_EMBED_MIN_NORM = float(os.environ.get("DIYKEY_FACE_EMBED_MIN_NORM") or 7.0)
# Narrowest face (frame pixels) worth sending. Live 2026-11-08 17:07 on Backyard a 19-22 px face
# scored 0.83, went up at 80% and Protect filed it as a new unmatched subject: SFace expects
# 112 px, so that much enlargement leaves no identity. The firmware gates on model-based quality
# instead (VerifyFaceQuality, EstimateFaceQuality); this approximates it. 0 keeps them all.
FACE_MIN_PX = int(os.environ.get("DIYKEY_FACE_MIN_PX") or 40)
# Faces sit in the top of a person box; search only this fraction of it (pi-port top_frac).
PERSON_TOP_FRAC = 0.6
# Person box grown by this factor before the search: the key frame can lag the box a little
# (pi-port CROP_MARGIN, "snapshot can lag the 640 detection frame by up to a second").
CROP_MARGIN = 1.3
# Square thumbnail side = face box's longer side * this, so the crop shows the whole head.
THUMB_SCALE = 1.6
JPEG_QUALITY = 90

# faceMask. Values: the firmware's mask classifier (SmartFace_mask, "Face Mask Classifier"
# model path in ubnt_ai_infer_agent strings) labels "none" / "face_mask"; _AddFaceMaskAttributeTag
# logs "Enhance mask tag with conf {} detected" and Protect's attribute is {val: string,
# confidence: number} with no enum (M/315148:21,42; M/482984:68). Protect's own sample reads
# faceMask {val:"face", confidence:70} (M/116837:159), so the vocabulary is loosely checked;
# "none" is the unmasked label we are sure the classifier emits. Confidence is 0-100 like every
# Protect attribute (M/116837:126,157). No permissively licensed small mask ONNX is bundled and
# a colour heuristic on a 40 px face is worse than a constant, so every face reports "none"
# at a realistic classifier-like 90 rather than a 1.0 that reads as 1%. DIYKEY_FACE_MASK_CONF
# overrides it.
FACE_MASK_VAL = "none"
FACE_MASK_CONF = float(os.environ.get("DIYKEY_FACE_MASK_CONF") or 90)
# A Protect-supplied face box has no YuNet score; this stands in for eventTracks confidence.
FACE_BOX_CONF = 90
FACE_BOX_MARGIN = 1.4


def model_dir():
    return os.environ.get("DIYKEY_FACE_MODELS") or os.path.join(_ROOT, "models")


def embedding_norm(embedding):
    return 0.0 if embedding is None else float(np.linalg.norm(np.asarray(embedding, dtype=np.float64)))


def weak_embedding(embedding):
    """True if an SFace vector is missing or too short (a poor face) to send."""
    return embedding is None or embedding_norm(embedding) < FACE_EMBED_MIN_NORM


def _embedding_list(emb):
    """SFace feature (1x128 float32) -> JSON-safe, L2-normalised list for faceEmbed.

    A real AI Port's faceEmbed has length 1.000 (live, 2026-11-21); raw SFace is about 7-9.
    Protect's face service compares by dot product, so raw vectors scored far past its
    threshold. The weak-face gate (weak_embedding) runs on the raw vector before this."""
    v = np.asarray(emb, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    if n > 0 and np.isfinite(n):
        v = v / n
    return [round(float(x), 6) for x in v]


def _expand_px(coord, img_w, img_h, margin=CROP_MARGIN, top_frac=1.0):
    """0..1000 [x,y,w,h] -> clamped pixel (x1,y1,x2,y2), grown by margin about its centre;
    top_frac keeps only the upper part. Ported from pi-port enrich._expand_px. (Not
    ram_tagger.crop_box: we need the crop's offset to map faces back into the frame, and
    importing ram_tagger would pull cv2/clip_embed in before we can report them missing.)"""
    x, y, w, h = (float(v) for v in coord[:4])
    h = h * top_frac
    cx, cy = x + w / 2, y + h / 2
    w, h = w * margin, h * margin
    x1 = int(max(0, (cx - w / 2) * img_w / COORD_END))
    y1 = int(max(0, (cy - h / 2) * img_h / COORD_END))
    x2 = int(min(img_w, (cx + w / 2) * img_w / COORD_END))
    y2 = int(min(img_h, (cy + h / 2) * img_h / COORD_END))
    return x1, y1, x2, y2


def square_box(fx, fy, fw, fh, img_w, img_h, scale=THUMB_SCALE):
    """A square (x1,y1,x2,y2) centred on the face, shifted (not shrunk) to stay inside the
    frame unless the frame is smaller. Protect sets the thumbnail's isRealCropped only when
    width == height (M/337954:117-138), so the crop must be exactly square."""
    side = int(round(max(fw, fh) * scale))
    side = max(1, min(side, img_w, img_h))
    cx, cy = fx + fw / 2, fy + fh / 2
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x1 = min(max(0, x1), img_w - side)
    y1 = min(max(0, y1), img_h - side)
    return x1, y1, x1 + side, y1 + side


class _Models:
    """Lazily-loaded YuNet + SFace, shared across threads (cv2 models aren't re-entrant)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._yunet = None
        self._sface = None
        self._failed = False
        self._failed_key = None

    def _key(self):
        d = model_dir()

        def mtime(name):
            try:
                return os.path.getmtime(os.path.join(d, name))
            except OSError:
                return None

        return (d, mtime(YUNET_FILE), mtime(SFACE_FILE))

    def infer_fn(self):
        """The default infer callable, or None (logged once) without cv2 or the models."""
        if self._failed:
            if self._failed_key == self._key():
                return None
            self._failed = False
            self._yunet = self._sface = None
        with self._lock:
            if self._yunet is None:
                try:
                    import cv2
                    d = model_dir()
                    yunet_path, sface_path = os.path.join(d, YUNET_FILE), os.path.join(d, SFACE_FILE)
                    missing = [p for p in (yunet_path, sface_path) if not os.path.exists(p)]
                    if missing:
                        raise FileNotFoundError(", ".join(missing))
                    self._yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320),
                                                            FACE_SCORE_THRESH)
                    self._sface = cv2.FaceRecognizerSF.create(sface_path, "")
                except Exception as e:
                    self._failed = True
                    self._failed_key = self._key()
                    log.warning("face recognition disabled until the models change (set "
                                "DIYKEY_FACE_MODELS to a dir with %s and %s): %s",
                                YUNET_FILE, SFACE_FILE, e)
                    return None
        return self._infer

    def embed_fn(self):
        """crop -> SFace embedding without detection (Protect's own face box), or None."""
        if self.infer_fn() is None:
            return None
        return self._embed

    def _embed(self, crop):
        """BGR face crop -> 1x128 SFace feature. No landmarks without YuNet, so the box is
        resized to SFace's 112x112 input unaligned (alignCrop needs the 15-value face row)."""
        import cv2
        with self._lock:
            return self._sface.feature(cv2.resize(crop, (112, 112)))

    def _infer(self, crop):
        """BGR crop -> [((x,y,w,h) crop px, score, embedding)]."""
        out = []
        with self._lock:
            self._yunet.setInputSize((crop.shape[1], crop.shape[0]))
            _, found = self._yunet.detect(crop)
            if found is None:
                return out
            for f in found:
                emb = self._sface.feature(self._sface.alignCrop(crop, f))
                out.append(((float(f[0]), float(f[1]), float(f[2]), float(f[3])),
                            float(f[14]), emb))
        return out


_models = _Models()


def _frame_for(frames, ts):
    """frames keyed by ts as Protect sent it; tolerate int/str/float key mismatches."""
    if ts in frames:
        return frames[ts]
    if str(ts) in frames:
        return frames[str(ts)]
    try:
        n = float(ts)
    except (TypeError, ValueError):
        return None
    for k, v in frames.items():
        try:
            if float(k) == n:
                return v
        except (TypeError, ValueError):
            continue
    return None


def _rois(entry):
    """roi may be one object (M/346308:96) or a list (M/321055 personMeta handles lists)."""
    roi = entry.get("roi")
    if isinstance(roi, dict):
        return [roi]
    return [r for r in (roi or []) if isinstance(r, dict)]


def _valid_coord(coord):
    return (isinstance(coord, (list, tuple)) and len(coord) >= 4
            and all(isinstance(v, (int, float)) and np.isfinite(v) for v in coord[:4]))


def _overlap_frac(face, person):
    """Fraction of the face box [x,y,w,h] inside the person box: Protect's own link rule
    (M/321055 s(): intersection / face area, accepted at >= 0.5)."""
    fx, fy, fw, fh = face
    px, py, pw, ph = person
    iw = min(fx + fw, px + pw) - max(fx, px)
    ih = min(fy + fh, py + ph) - max(fy, py)
    return 0.0 if fw * fh <= 0 or iw <= 0 or ih <= 0 else iw * ih / (fw * fh)


def face_descriptor(fid, coord, conf):
    """One faceInferenceSchema eventTracks descriptor (all fields required, M/315148:26-38;
    coord = 4 finite numbers and zones/lines number arrays per M/116837:33-49). Protect's
    linkAiKeyFacesToPersons (M/321055) reads objectType face + coord only, and only when
    linkedPersonTrackerID is absent; we still send both (a missing link falls back to this)."""
    return {"attributes": {"faceMask": {"confidence": FACE_MASK_CONF, "val": FACE_MASK_VAL}},
            "boxColor": "cyan", "confidenceLevel": int(round(conf)), "coord": coord,
            "coord3d": [], "lines": [], "name": "", "objectType": "face",
            "stationary": False, "tag": "", "trackerID": fid, "zones": []}


def build_face_result(camera_id, event_id, frames, person_meta, start_ms=None, infer=None,
                      face_meta=None, embed=None):
    """faceInferenceSchema result for a person event, or None if no usable face.

    frames: {ts: BGR ndarray}; person_meta: [{ts, roi:{coord, objectType, trackerId}}].
    Each entry's person box is searched for faces; the best (score * area) face is embedded
    and the best one across frames kept per person trackerId. infer(crop) -> [(bbox, score,
    embedding)] replaces YuNet+SFace (tests). Returns (face_json, {str(face_tracker_id):
    jpeg bytes}).

    face_meta: Protect's camera face boxes [{ts, roi:{coord:[x,y,w,h] 0..1000, trackerId,
    ...}}] (roi may be a list). At those ts the boxes are cropped directly,     YuNet runs inside the box grown 40% (aligned embed), falling back to embed(crop) -> embedding, an unaligned resize, when it finds nothing; each
    box links to the one person box at that ts that contains it (>= 0.5 of the face, as
    M/321055). Person frames without face_meta use the person-box search above."""
    t0 = time.monotonic()
    try:
        import cv2
    except Exception:
        if not getattr(build_face_result, "_warned_cv2", False):
            build_face_result._warned_cv2 = True
            log.warning("face recognition disabled: opencv is not installed")
        return None
    if infer is None:
        infer = _models.infer_fn()
        if infer is None:
            return None

    face_boxes = {}  # ts -> [coord] from Protect's camera
    for entry in face_meta or []:
        if entry.get("ts") is None:
            continue
        for roi in _rois(entry):
            if _valid_coord(roi.get("coord")):
                face_boxes.setdefault(float(entry["ts"]), []).append(roi["coord"][:4])
    pre_s = infer_s = 0.0
    best = {}  # person trackerId -> (quality, ts, frame, (fx,fy,fw,fh) frame px, embedding, conf 0-100)
    persons_at = {}  # ts -> [(tid, coord)]
    for entry in person_meta or []:
        for roi in _rois(entry):
            if (roi.get("objectType") or "person") == "person" and roi.get("trackerId") is not None \
                    and _valid_coord(roi.get("coord")) and entry.get("ts") is not None:
                persons_at.setdefault(float(entry["ts"]), []).append(
                    (int(roi["trackerId"]), [float(v) for v in roi["coord"][:4]]))

    # Protect's camera face boxes first; a person frame they cover skips the YuNet search.
    covered = set()
    for fts, boxes in face_boxes.items():
        frame = _frame_for(frames or {}, fts)
        if frame is None:
            continue
        ih, iw = frame.shape[:2]
        for fc in boxes:
            fc = [float(v) for v in fc]
            links = [tid for tid, pc in persons_at.get(fts, ()) if _overlap_frac(fc, pc) >= 0.5]
            if len(links) != 1:
                continue  # none or ambiguous: Protect itself would not link it either
            tp = time.monotonic()
            # Grow the camera's box 40% and let YuNet find the face inside it: the aligned
            # SFace crop embeds far better than a plain resize (the aligned path made the
            # live face match). Unaligned resize only when YuNet finds nothing.
            x1, y1, x2, y2 = _expand_px(fc, iw, ih, margin=FACE_BOX_MARGIN)
            bx1, by1 = max(0, int(fc[0] * iw / COORD_END)), max(0, int(fc[1] * ih / COORD_END))
            bx2, by2 = min(iw, int((fc[0] + fc[2]) * iw / COORD_END)), \
                min(ih, int((fc[1] + fc[3]) * ih / COORD_END))
            pre_s += time.monotonic() - tp
            if bx2 - bx1 < max(20, FACE_MIN_PX) or by2 - by1 < 20:
                continue
            ti = time.monotonic()
            emb, box = None, (bx1, by1, bx2 - bx1, by2 - by1)
            try:
                found = [f for f in (infer(frame[y1:y2, x1:x2]) or [])
                         if f[0][2] >= max(1, FACE_MIN_PX) and f[0][3] > 0
                         and not weak_embedding(f[2])]
                if found:
                    (dx, dy, dw, dh), _, emb = max(found, key=lambda f: f[1] * f[0][2] * f[0][3])
                    box = (x1 + dx, y1 + dy, dw, dh)
                else:
                    if embed is None:
                        embed = _models.embed_fn()
                    emb = embed(frame[by1:by2, bx1:bx2]) if embed else None
            except Exception:
                log.exception("face embed failed on camera %s event %s", camera_id, event_id)
            infer_s += time.monotonic() - ti
            covered.add(fts)
            if emb is None or weak_embedding(emb):
                continue
            q = FACE_BOX_CONF / 100.0 * box[2] * box[3]
            tid = links[0]
            if tid not in best or q > best[tid][0]:
                best[tid] = (q, fts, frame, box, emb, FACE_BOX_CONF)

    for entry in person_meta or []:
        if entry.get("ts") is not None and float(entry["ts"]) in covered:
            continue
        roi = entry.get("roi") or {}
        if isinstance(roi, list):
            roi = roi[0] if roi and isinstance(roi[0], dict) else {}
        if (roi.get("objectType") or "person") != "person":
            continue
        tid, coord = roi.get("trackerId"), roi.get("coord")
        if tid is None or not coord or len(coord) < 4:
            continue
        frame = _frame_for(frames or {}, entry.get("ts"))
        if frame is None:
            continue
        tp = time.monotonic()
        ih, iw = frame.shape[:2]
        x1, y1, x2, y2 = _expand_px(coord, iw, ih, top_frac=PERSON_TOP_FRAC)
        crop = frame[y1:y2, x1:x2]
        pre_s += time.monotonic() - tp
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            continue
        ti = time.monotonic()
        try:
            found = infer(crop) or []
        except Exception:
            log.exception("face inference failed on camera %s event %s", camera_id, event_id)
            found = []
        infer_s += time.monotonic() - ti
        top = None
        for (fx, fy, fw, fh), score, emb in found:
            if fw < max(1, FACE_MIN_PX) or fh <= 0 or weak_embedding(emb):
                continue
            q = float(score) * fw * fh
            if top is None or q > top[0]:
                top = (q, (x1 + fx, y1 + fy, fw, fh), emb, float(score) * 100)
        if top is None:
            continue
        tid = int(tid)
        if tid not in best or top[0] > best[tid][0]:
            best[tid] = (top[0], entry.get("ts"), frame, top[1], top[2], top[3])

    if not best:
        return None

    face_attrs, snapshots, jpegs, tracks = {}, [], {}, []
    for tid, (_, ts, frame, (fx, fy, fw, fh), emb, conf) in sorted(best.items()):
        tp = time.monotonic()
        ih, iw = frame.shape[:2]
        sx1, sy1, sx2, sy2 = square_box(fx, fy, fw, fh, iw, ih)
        ok, buf = cv2.imencode(".jpg", frame[sy1:sy2, sx1:sx2],
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        pre_s += time.monotonic() - tp
        if not ok:
            continue
        fid = tid + FACE_TRACKER_OFFSET
        face_attrs[str(fid)] = {
            "faceEmbed": _embedding_list(emb),
            "faceMask": {"confidence": FACE_MASK_CONF, "val": FACE_MASK_VAL},
            "matchedName": "",
            "namesTopK": [],
            "objectType": "face",
            "topKCandidate": [],
            "linkedPersonTrackerID": tid,
        }
        ts_num = float(ts) if ts is not None else 0.0
        # clockBestWall: person_meta ts looks event-relative ms in the captures we have, so
        # wall = start_ms + ts when the caller knows the event start; otherwise ts is passed
        # through as-is (correct if it is already epoch ms). Unverified which Protect sends.
        wall = start_ms + ts_num if start_ms is not None else ts_num
        snapshots.append({
            "clockBestMonotonic": int(ts_num),
            "clockBestWall": int(wall),
            "smartDetectHeatmap": "",
            # smartDetectSnapshot is the firmware's LOCAL file path; syswrapper only tests it with `-f`
            # and uploads it as multipart part named str(trackerID) (SW:4019-4026), so Protect never
            # sees the path (M/337954:92 only copies it into the parsed snapshot). Mirror the
            # firmware's /tmp/<camera>_<id>.jpg. Name stays "": Protect uses it as the object's
            # display name (M/482984:245) and, on a vehicle snapshot, as the plate flag (M/204401:43).
            # Heatmap: nothing in SW or the save code reads it; "" until a capture shows otherwise.
            "smartDetectSnapshot": f"/tmp/{camera_id}_{fid}.jpg",
            "smartDetectSnapshotName": "",
            "smartDetectSnapshotType": "face",
            "trackerID": fid,
        })
        jpegs[str(fid)] = buf.tobytes()
        # eventTracks timestamp = the frame ts exactly as Protect's personMeta carries it
        # (M/321055 looks it up by number equality), not wall time.
        coord = [round(max(0.0, min(float(COORD_END), v)), 1) for v in
                 (fx * COORD_END / iw, fy * COORD_END / ih,
                  fw * COORD_END / iw, fh * COORD_END / ih)]
        tracks.append({"timestamp": int(ts_num) if ts_num == int(ts_num) else ts_num,
                       "descriptors": [face_descriptor(fid, coord, conf)]})

    if not face_attrs:
        return None
    face_json = {
        "cameraId": camera_id,
        "eventId": event_id,
        "eventTracks": tracks,
        "faceAttrs": face_attrs,
        "faceSnapshots": snapshots,
        "inferMs": int(round(infer_s * 1000)),
        "preProcessMs": int(round(pre_s * 1000)),
        "status": "success",
        "timeElapsedMs": int(round((time.monotonic() - t0) * 1000)),
    }
    log.info("faces: camera %s event %s -> %d face(s) %s", camera_id, event_id,
             len(face_attrs), sorted(face_attrs))
    return face_json, jpegs
