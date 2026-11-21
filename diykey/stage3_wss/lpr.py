# SPDX-License-Identifier: AGPL-3.0-or-later
"""Licence-plate reads for an AI Key LPR result (Protect's lprInferenceSchema).

Ported from pi-port/diy-port (same author, AGPL): diyport/enrich.py vehicle_attributes /
dominant_color (vehicle colour + type) and Enricher.plates (fast-alpr: YOLOv9 plate detector
+ CCT OCR, both MIT). Here the input is the event's vehicle track metadata, not a live
detection: per trackerId the best plate read over all of its frames is reported.

Shape (Protect M/315148:59-71 lprAttrs record, :106-115 lprInferenceSchema; M =
up/service-expanded/modules):
  {cameraId, eventId, eventTracks: [{timestamp, descriptors: [plate]}], lprAttrs: {<vehicle trackerId>: {matchedName,
   namesTopK, objectType, topKCandidate: [{name, confidence}], color?, vehicleType?}},
   lprSnapshots: [snapshotSchema], inferMs, preProcessMs, status, timeElapsedMs}
lprAttrs is keyed by the *vehicle* trackerId because Protect saves the plate onto the
smart-detect object whose attributes.trackerId matches (M/785414:36-58), and repeated under
the plate snapshot's trackerID, whose entry becomes the plate object's attributes (:88-94).
"""
import importlib.util
import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger("diykey.lpr")

COORD_END = 1000  # roi coords are normalised 0..1000 (M/346308:76-80)

# Snapshot trackerID = vehicle trackerId + this. Unknown whether the real firmware reuses the
# vehicle id for the plate snapshot; the offset keeps the plate JPEG's multipart key from
# colliding with a vehicle snapshot of the same tracker. lprAttrs stays on the vehicle id.
PLATE_TRACKER_OFFSET = 200000

# enrich.py:73-79. Plate-detector score floor; plates narrower than PLATE_MIN_PX (crop pixels)
# aren't reported (pi-port live: every read under 75 px was wrong).
PLATE_SCORE_THRESH = float(os.environ.get("DIYKEY_LPR_THRESH", "0.5"))
PLATE_MIN_PX = int(os.environ.get("DIYKEY_LPR_MIN_PX") or 60)
# OCR floor: a plate whose least certain character scores under this is not reported. The
# detector finds plate *shapes* even when IR flare or blur leaves the characters unreadable,
# and the OCR always emits some text; one unreadable character makes the whole read wrong,
# so the weakest character decides (2026-11-19: an IR-flared Driveway plate was sent).
PLATE_OCR_MIN = float(os.environ.get("DIYKEY_LPR_OCR_MIN") or 0.4)
# enrich.py CROP_MARGIN 1.3 grows the box 15% each side (the frame can lag the box).
CROP_MARGIN = 0.15
TOP_K = 5

# vehicleMeta says "vehicle"; legacy roiMeta names the body type (car/truck/...). Both are
# vehicles for plate reading; person/animal are not.
VEHICLE_OBJECT_TYPES = frozenset({
    "vehicle", "car", "truck", "bus", "motorcycle", "motorbike", "bicycle", "bike", "van",
    "suv", "pickup truck",
})

# --- vehicle type + colour (enrich.py:33-46, 123-192) ----------------------

# COCO label -> Protect vehicleType; Protect knows car/suv/truck/van/bus/motorcycle/bike.
VEHICLE_TYPES = {
    "car": "car", "truck": "truck", "bus": "bus", "motorbike": "motorcycle",
    "motorcycle": "motorcycle", "bicycle": "bike",
}
# Protect's colour vocabulary; the classifier's "silver" has no Protect value.
PROTECT_COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
                  "black", "white", "gray"}
COLOR_ALIASES = {"silver": "gray"}


def _rgb_to_hsv(rgb):
    """uint8 Nx3 RGB -> float Nx3 HSV (h in degrees, s/v in 0-1)."""
    c = rgb.astype(np.float32) / 255.0
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    v = c.max(axis=1)
    mn = c.min(axis=1)
    d = v - mn
    s = np.where(v > 0, d / np.maximum(v, 1e-6), 0.0)
    h = np.zeros_like(v)
    nz = d > 1e-6
    rm = nz & (v == r)
    gm = nz & (v == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = (60 * ((g[rm] - b[rm]) / d[rm])) % 360
    h[gm] = 60 * ((b[gm] - r[gm]) / d[gm]) + 120
    h[bm] = 60 * ((r[bm] - g[bm]) / d[bm]) + 240
    return np.stack([h, s, v], axis=1)


def _pixel_colors(hsv):
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    names = np.full(len(h), "gray", dtype=object)
    chroma = s >= 0.25
    names[~chroma & (v >= 0.8)] = "white"
    names[~chroma & (v >= 0.55) & (v < 0.8)] = "silver"
    names[v < 0.2] = "black"
    hue = chroma & (v >= 0.2)
    bins = [(0, 15, "red"), (15, 40, "orange"), (40, 70, "yellow"), (70, 170, "green"),
            (170, 260, "blue"), (260, 330, "purple"), (330, 361, "red")]
    for lo, hi, name in bins:
        names[hue & (h >= lo) & (h < hi)] = name
    # dark, desaturated orange/red reads as brown
    names[hue & (h < 40) & (v < 0.45)] = "brown"
    return names


def dominant_color(frame_rgb, coord):
    """(color, confidence 0-100) for the central body of a box; None if the crop is empty."""
    fh, fw = frame_rgb.shape[:2]
    x, y, w, h = (float(v) for v in coord[:4])
    # Central region: skips road/background at the edges and glass at the top.
    x1 = int((x + 0.2 * w) * fw / COORD_END)
    x2 = int((x + 0.8 * w) * fw / COORD_END)
    y1 = int((y + 0.35 * h) * fh / COORD_END)
    y2 = int((y + 0.85 * h) * fh / COORD_END)
    crop = frame_rgb[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
    if crop.size == 0:
        return None
    names = _pixel_colors(_rgb_to_hsv(crop.reshape(-1, 3)))
    values, counts = np.unique(names, return_counts=True)
    i = int(np.argmax(counts))
    return str(values[i]), int(round(100 * counts[i] / counts.sum()))


# CLIP colour. Pixel hue can't tell paint from reflections: on 2026-11-21 a silver Lexus on the
# Driveway read "blue" (57-74 %) because its hood mirrored a clear sky (hood pixels 99-100 % blue,
# median saturation ~0.5), and neither white balance nor a saturation cutoff fixes that without
# losing real blue cars. CLIP judges the whole car, but alone it is brittle on small crops: a
# white van on the road flipped gray <-> brown on a JPEG re-save, and CLIP's 224 px centre crop
# only sees the middle of a wide box. So: average CLIP over the centre crop and the box padded to
# a square, then (choose_color) where CLIP and the pixels disagree on whether the car is neutral
# (white/gray/black) at all, take the neutral answer -- a wrong neutral colour reads less wrong
# than a wrong vivid one, and coloured light/reflections push both methods toward vivid.
# Protect has no silver; it folds into gray like COLOR_ALIASES. Day frames only: an IR/greyscale
# crop has no colour for CLIP to see, so the pixel heuristic (white/gray/black) stays there.
CLIP_COLOR_LABELS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
                     "black", "white", "gray", "silver")
CLIP_COLOR_TEMPLATES = ("a photo of a {} car.", "a photo of a {} vehicle.")
NEUTRAL_COLORS = frozenset({"white", "gray", "black"})
GRAYSCALE_MAX_SAT = 0.08  # median HSV saturation of an IR night crop is ~0


def _box_crop(frame_rgb, coord):
    fh, fw = frame_rgb.shape[:2]
    x, y, w, h = (float(v) for v in coord[:4])
    x1, x2 = max(0, int(x * fw / COORD_END)), min(fw, int((x + w) * fw / COORD_END))
    y1, y2 = max(0, int(y * fh / COORD_END)), min(fh, int((y + h) * fh / COORD_END))
    return frame_rgb[y1:y2, x1:x2]


def _pad_square(img, value=128):
    h, w = img.shape[:2]
    side = max(h, w)
    out = np.full((side, side, 3), value, dtype=img.dtype)
    y, x = (side - h) // 2, (side - w) // 2
    out[y:y + h, x:x + w] = img
    return out


def clip_color_probs(frame_rgb, coord):
    """{Protect colour: probability} from CLIP zero-shot over the vehicle box (mean of the centre
    crop and the box padded square), or None when CLIP is off or unavailable, the crop is empty,
    or the crop is greyscale (IR)."""
    try:
        import clip_embed
    except ImportError:
        return None
    if not clip_embed.enabled():
        return None
    crop = _box_crop(frame_rgb, coord)
    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    sample = crop[::4, ::4].reshape(-1, 3)
    if float(np.median(_rgb_to_hsv(sample)[:, 1])) < GRAYSCALE_MAX_SAT:
        return None
    bgr = np.ascontiguousarray(crop[..., ::-1])  # image_embed wants BGR
    merged = {}
    views = (bgr, _pad_square(bgr))
    for view in views:
        probs = clip_embed.label_probs(clip_embed.image_embed(view), list(CLIP_COLOR_LABELS),
                                       CLIP_COLOR_TEMPLATES)
        if not probs:
            return None
        for name, p in probs.items():
            name = COLOR_ALIASES.get(name, name)
            merged[name] = merged.get(name, 0.0) + p / len(views)
    return merged


def choose_color(clip_probs, pixel):
    """(color, confidence 0-100) from CLIP's probabilities and the pixel heuristic's
    (color, confidence); either may be None. CLIP wins, except that when exactly one of the two
    calls the car neutral (white/gray/black), the neutral answer wins."""
    if pixel:
        pixel = (COLOR_ALIASES.get(pixel[0], pixel[0]), pixel[1])
    if not clip_probs:
        return pixel
    name = max(clip_probs, key=clip_probs.get)
    clip = (name, int(round(100 * clip_probs[name])))
    if pixel and pixel[0] != name and pixel[0] in NEUTRAL_COLORS and name not in NEUTRAL_COLORS:
        return pixel
    return clip


def vehicle_attributes(label, frame_rgb, coord, roi_attrs=None):
    """{vehicleType?, color?} as {val, confidence}, or None. Confidence is 0-100 as in
    pi-port's descriptor attributes (scale there came from the Protect decompile). A camera
    vehicleType/color in the ROI wins over the detector label; otherwise colour comes from
    choose_color (CLIP, with the pixel heuristic as fallback and neutral tiebreak)."""
    roi_attrs = roi_attrs or {}
    attrs = {}
    vt = roi_attrs.get("vehicleType") or {}
    vtype = vt.get("val") or VEHICLE_TYPES.get((label or "").strip())
    if vtype:
        attrs["vehicleType"] = {"val": str(vtype).lower(),
                                "confidence": vt.get("confidence", 70)}
    color_attr = roi_attrs.get("color") or {}
    if color_attr.get("val"):
        name = COLOR_ALIASES.get(str(color_attr["val"]).lower(), str(color_attr["val"]).lower())
        if name in PROTECT_COLORS:
            attrs["color"] = {"val": name,
                              "confidence": color_attr.get("confidence", 70)}
    elif frame_rgb is not None:
        probs = clip_color_probs(frame_rgb, coord)
        pixel = dominant_color(frame_rgb, coord)
        color = choose_color(probs, pixel)
        log.debug("vehicle colour %s (clip %s, pixel %s)", color,
                  None if probs is None else
                  {k: round(v, 2) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3]},
                  pixel)
        if color:
            name = COLOR_ALIASES.get(color[0], color[0])
            if name in PROTECT_COLORS:
                attrs["color"] = {"val": name, "confidence": color[1]}
    return attrs or None


# --- fast-alpr reader (enrich.py:241-299) ----------------------------------

_lock = threading.Lock()
_alpr = None
_unavailable_logged = False


def _have(mod):
    return importlib.util.find_spec(mod) is not None


def lpr_available():
    return _have("fast_alpr") and _have("cv2")


def _alpr_reader(crop):
    """crop (BGR) -> [(text, confidence 0-1, (x1, y1, x2, y2) crop px)]. Lazy-loads fast-alpr
    once; the lock also serialises predict(), which isn't known to be thread-safe."""
    global _alpr
    out = []
    with _lock:
        if _alpr is None:
            from fast_alpr import ALPR
            # Both on CPU: the CCT OCR graph and the YOLOv9 dynamic NMS output don't work
            # under Core ML (enrich.py:244-247).
            _alpr = ALPR(detector_providers=["CPUExecutionProvider"],
                         ocr_providers=["CPUExecutionProvider"])
        results = _alpr.predict(crop)
    for r in results:
        if r.detection.confidence < PLATE_SCORE_THRESH or not r.ocr or not r.ocr.text:
            continue
        conf = ocr_confidence(r.ocr.text, r.ocr.confidence)
        if conf is None:
            log.debug("plate %r dropped: OCR character confidence under %.2f (%s)",
                      r.ocr.text, PLATE_OCR_MIN, r.ocr.confidence)
            continue
        b = r.detection.bounding_box
        out.append((r.ocr.text, min(float(r.detection.confidence), conf),
                    (int(b.x1), int(b.y1), int(b.x2), int(b.y2))))
    return out


def ocr_confidence(text, probs, floor=None):
    """Mean OCR confidence over the plate's own characters, or None when any of them is
    under `floor` (default PLATE_OCR_MIN) or there are none.

    fast-plate-ocr returns one probability per model slot, trailing pad slots included, and
    strips the pad from the text; the pad slots are near-certain, so averaging them all lets
    an unreadable plate look confident. Only the first len(text) slots are the characters."""
    floor = PLATE_OCR_MIN if floor is None else floor
    if isinstance(probs, (list, tuple, np.ndarray)):
        chars = [float(p) for p in list(probs)[:len(text or "")]]
    else:
        chars = [float(probs)] if text else []
    if not chars or min(chars) < floor:
        return None
    return float(np.mean(chars))


def _crop(img, coord, margin):
    try:
        from ram_tagger import crop_box
    except ImportError:
        crop_box = None
    if crop_box is not None:
        return crop_box(img, coord, margin=margin)
    h, w = img.shape[:2]
    x, y, bw, bh = (float(v) * s / COORD_END for v, s in zip(coord[:4], (w, h, w, h)))
    x0, y0 = max(0, int(x - bw * margin)), max(0, int(y - bh * margin))
    x1, y1 = min(w, int(x + bw * (1 + margin))), min(h, int(y + bh * (1 + margin)))
    return img[y0:y1, x0:x1] if x1 - x0 >= 8 and y1 - y0 >= 8 else None


def _crop_origin(img, coord, margin):
    """Top-left (x0, y0) frame pixel of _crop's fallback crop, to map the plate bbox (crop px)
    back into frame px for eventTracks. ram_tagger.crop_box is assumed to use the same
    margin-grown clamp; if it ever differs the track box is off by the margin only."""
    h, w = img.shape[:2]
    x, y, bw, bh = (float(v) * s / COORD_END for v, s in zip(coord[:4], (w, h, w, h)))
    return max(0, int(x - bw * margin)), max(0, int(y - bh * margin))


def _norm_box(x1, y1, x2, y2, fw, fh):
    """Frame-px box -> [x, y, w, h] in 0..1000 (the descriptor coord, M/116837:30-31,
    smartDetectTrackPayload.coord "normalized to 0-1000 of the frame")."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    return [round(x1 * COORD_END / fw, 1), round(y1 * COORD_END / fh, 1),
            round((x2 - x1) * COORD_END / fw, 1), round((y2 - y1) * COORD_END / fh, 1)]


def _jpeg(img):
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes() if ok else b""
    except Exception:
        log.debug("plate JPEG encode failed", exc_info=True)
        return b""


def plate_descriptor(snap_id, text, conf, coord):
    """One lprInferenceSchema eventTracks descriptor (all fields required, M/315148:26-38;
    Protect's smartDetectObjectDescriptorAjvParse also wants coord = 4 finite numbers and
    zones/lines = number arrays, M/116837:33-49). objectType licensePlate with the plate
    text as name, as firmware lprSnapshots name the object; confidenceLevel 0-100.
    Protect's face linker (M/321055) matches eventTracks by timestamp to personMeta ts, so
    the track timestamp is the frame ts as given, not wall time."""
    return {"boxColor": "cyan", "confidenceLevel": int(round(100 * conf)), "coord": coord,
            "coord3d": [], "lines": [], "name": text, "objectType": "licensePlate",
            "stationary": False, "tag": "", "trackerID": snap_id, "zones": []}


def build_lpr(camera_id, event_id, frames, vehicle_meta, start_ms=None, reader=None):
    """(lpr_json, jpegs) for an event's vehicle tracks, or None when no plate was read.

    frames: {ts: BGR ndarray}. vehicle_meta: [{ts, roi: {coord: [x, y, w, h] 0..1000,
    objectType, trackerId}}] (M/346308:76-80, 96: roi is a single object). reader: optional
    callable(crop) -> [(text, confidence, bbox)] replacing fast-alpr (tests).
    jpegs: {str(snapshot trackerID): plate crop JPEG}, snapshot id = trackerId + offset.
    """
    global _unavailable_logged
    t0 = time.monotonic()
    if reader is None:
        if not lpr_available():
            if not _unavailable_logged:
                log.warning("fast-alpr/opencv not installed -- licence plates disabled")
                _unavailable_logged = True
            return None
        reader = _alpr_reader
    pre_s = infer_s = 0.0
    best = {}    # trackerId -> dict(text, conf, ts, coord, frame, plate, label)
    cands = {}   # trackerId -> {text: best confidence}
    for m in vehicle_meta or ():
        roi = m.get("roi") or {}
        tid, coord, ts = roi.get("trackerId"), roi.get("coord"), m.get("ts")
        if tid is None or not coord or ts is None:
            continue
        if str(roi.get("objectType") or "vehicle").lower() not in VEHICLE_OBJECT_TYPES:
            continue
        frame = frames.get(ts)
        if frame is None:
            continue
        p0 = time.monotonic()
        crop = _crop(frame, coord, CROP_MARGIN)
        pre_s += time.monotonic() - p0
        if crop is None or crop.shape[0] < 16 or crop.shape[1] < 16:
            continue
        i0 = time.monotonic()
        try:
            reads = reader(crop) or []
        except Exception:
            log.exception("plate read failed on tracker %s", tid)
            reads = []
        infer_s += time.monotonic() - i0
        tid = int(tid)
        for text, conf, bbox in reads:
            text = (text or "").replace("_", "").strip().upper()
            if len(text) < 3:
                continue
            x1, y1, x2, y2 = (int(v) for v in bbox)
            if x2 - x1 < PLATE_MIN_PX:
                log.debug("plate %r skipped: %d px wide, under %d px", text, x2 - x1,
                          PLATE_MIN_PX)
                continue
            conf = float(conf)
            c = cands.setdefault(tid, {})
            c[text] = max(conf, c.get(text, 0.0))
            if tid not in best or conf > best[tid]["conf"]:
                ox, oy = _crop_origin(frame, coord, CROP_MARGIN)
                best[tid] = {"text": text, "conf": conf, "ts": ts, "coord": coord,
                             "frame": frame, "box_px": (ox + x1, oy + y1, ox + x2, oy + y2),
                             "label": roi.get("label") or m.get("label"),
                             "attrs": roi.get("attributes") or {},
                             "plate": crop[max(0, y1):y2, max(0, x1):x2]}
    if not best:
        return None

    attrs, snaps, jpegs, tracks = {}, [], {}, []
    for tid, b in best.items():
        top = sorted(cands[tid].items(), key=lambda kv: -kv[1])[:TOP_K]
        entry = {
            "matchedName": b["text"],
            "namesTopK": [t for t, _ in top],
            "objectType": "vehicle",
            # 0-100, same as color/vehicleType: every Protect attribute confidence is a
            # percentage (smartDetectTrackPayload.confidence "(0-100)" M/116837:126; sample
            # attributes color 95 / vehicleType 88 M/116837:157-158; descriptor
            # confidenceLevel is the same scale). topKCandidate sits in the same attribute
            # family (482984:31 {name, confidence}) and 785414 only copies it, so a 0..1
            # fast-alpr score would read as ~1% there. No Protect code compares it.
            "topKCandidate": [{"name": t, "confidence": round(100 * c, 1)} for t, c in top],
        }
        va = vehicle_attributes(b["label"], b["frame"][..., ::-1], b["coord"], b["attrs"])
        if va:
            entry.update(va)
        plate_conf = int(round(100 * b["conf"]))
        # Protect REPLACES the vehicle object's attributes with lprAttrs[vehicle id]
        # (M/785414:47-55), dropping the camera's confidence, and the detected-thumbnail card
        # takes its confidence from attributes.confidence (M/482984:249). attributeSchema is
        # passthrough, so a 0-100 confidence survives: the camera's own when the roi carries
        # one, else the plate read's. The plate object gets the plate read's.
        roi_conf = b["attrs"].get("confidence") if isinstance(b["attrs"], dict) else None
        attrs[str(tid)] = {**entry, "confidence": roi_conf if isinstance(roi_conf, (int, float))
                           else plate_conf}
        snap_id = tid + PLATE_TRACKER_OFFSET
        # Protect gives the plate's own smartDetectObject the attributes of
        # lprAttrs[snapshot trackerID] (M/785414:88-94), so the plate number has to sit under
        # the snapshot id too, or the plate record is saved with no name.
        attrs[str(snap_id)] = {**entry, "confidence": plate_conf}
        wall = int(start_ms) + int(b["ts"]) if start_ms is not None else int(b["ts"])
        snaps.append({
            "clockBestMonotonic": int(b["ts"]),
            "clockBestWall": wall,
            "smartDetectHeatmap": "",
            # smartDetectSnapshot is the firmware's LOCAL file path; syswrapper only tests it with `-f`
            # and uploads it as multipart part named str(trackerID) (SW:4019-4026), so Protect never
            # sees the path (M/337954:92 only copies it into the parsed snapshot). Mirror the
            # firmware's /tmp/<camera>_<id>.jpg. Name = the plate number: Protect labels the
            # plate's detected-thumbnail card with it (M/482984:245; "" left the card with no
            # number, Driveway 2026-11-21). The name doubles as a plate flag only on a *vehicle*
            # snapshot (M/204401:43); this one is typed licensePlate.
            # Heatmap: nothing in SW or the save code reads it; "" until a capture shows otherwise.
            "smartDetectSnapshot": f"/tmp/{camera_id}_{snap_id}.jpg",
            "smartDetectSnapshotName": b["text"],
            "smartDetectSnapshotType": "licensePlate",
            "trackerID": snap_id,
        })
        fh, fw = b["frame"].shape[:2]
        tracks.append({"timestamp": int(b["ts"]), "descriptors": [plate_descriptor(
            snap_id, b["text"], b["conf"], _norm_box(*b["box_px"], fw, fh))]})
        jpegs[str(snap_id)] = _jpeg(b["plate"]) if b["plate"].size else b""
    lpr_json = {
        "cameraId": camera_id,
        "eventId": event_id,
        "eventTracks": tracks,
        "lprAttrs": attrs,
        "lprSnapshots": snaps,
        "inferMs": int(round(infer_s * 1000)),
        "preProcessMs": int(round(pre_s * 1000)),
        "status": "success",
        "timeElapsedMs": int(round((time.monotonic() - t0) * 1000)),
    }
    return lpr_json, jpegs
