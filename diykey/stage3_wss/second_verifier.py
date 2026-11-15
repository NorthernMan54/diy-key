# SPDX-License-Identifier: AGPL-3.0-or-later
"""Second Stage Verification (RequestAI -> second_verifier_mlabel).

Protect sends the export URL for the event plus thumbnailMeta: per thumbnail
time, the tracked objects with ``coord`` = [x, y, w, h] of the box's top-left
corner in 0..1000 of the frame (checked against a live person box). For each
object we say what is really in that box: Protect sets the thumbnail and smart
object to ``detectedAs`` (person / vehicle / animal) with confidence
round(100 * detectionConfidence), keeping the old ones as preReverification*.

The box is matched against the COCO detector (ram_tagger.RamTagger) on the
full frame by IoU; when nothing overlaps, the box (with a margin) is run on its
own, which helps small or IR-washed objects.

Verdicts follow the firmware's verifier (ui-infer-gateway second_verifier.py, fw 2.2.8): it
has no "nothing there" class, so it keeps the camera's detection unless the model is sure of
another type (top class prob >= 0.995), and reports min(det% + max(2, prob% / 20), 99) as the
confidence of a kept one. Our detector's scores are YOLO confidences, not a classifier's
softmax, and rarely pass 0.995, so another type wins at DIYKEY_VERIFY_OVERRIDE (default 0.8).
No frame gives modelRunStatus "skipped", as the firmware's placeholder does.
"""
import logging
import os
import time

import cv2

log = logging.getLogger("diykey.verify")

# COCO label -> Protect object type.
VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"}
ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
MIN_IOU = 0.3
CROP_MARGIN = 0.3


def override_min():
    """Detector score at which another type replaces the camera's (DIYKEY_VERIFY_OVERRIDE)."""
    try:
        return float(os.environ.get("DIYKEY_VERIFY_OVERRIDE") or 0.8)
    except ValueError:
        return 0.8


def kept_confidence(det_pct, prob):
    """The firmware's confidence for a detection it keeps: min(det% + max(2, prob% / 20), 99),
    as 0..1. det_pct: the camera's confidence (0..100); prob: the verifier's score (0..1)."""
    return round(min(float(det_pct or 0) + max(2.0, prob * 100 / 20), 99.0) / 100, 3)


def verdict(claimed, det_pct, kind, score, override=None):
    """(isValidDetection, detectedAs, detectionConfidence) for one box: the detector's best
    (kind, score) against the camera's claimed type and confidence."""
    override = override_min() if override is None else override
    if kind and kind != claimed and score >= override:
        return False, kind, round(score, 3)
    if kind == claimed:
        return True, claimed, kept_confidence(det_pct, score)
    return True, claimed, round(float(det_pct or 0) / 100, 3)


def protect_type(label):
    if label == "person":
        return "person"
    if label in VEHICLES:
        return "vehicle"
    if label in ANIMALS:
        return "animal"
    return None


def _rois(entry):
    """roi is one object in Protect's schema but a list on some paths."""
    roi = entry.get("roi")
    if isinstance(roi, dict):
        return [roi]
    return [r for r in roi or () if isinstance(r, dict)]


def box_px(coord, width, height):
    """[x, y, w, h] in 0..1000 -> pixel (x, y, w, h)."""
    x, y, w, h = (float(v) for v in coord[:4])
    return x * width / 1000, y * height / 1000, w * width / 1000, h * height / 1000


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _crop_best(tagger, img, box):
    """Best (type, score) the detector finds inside the box plus a margin, or (None, 0)."""
    h, w = img.shape[:2]
    x, y, bw, bh = box
    mx, my = bw * CROP_MARGIN, bh * CROP_MARGIN
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(w, int(x + bw + mx)), min(h, int(y + bh + my))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, 0.0
    best = (None, 0.0)
    for label, score, _ in tagger.detect(img[y0:y1, x0:x1]):
        kind = protect_type(label)
        if kind and score > best[1]:
            best = (kind, score)
    return best


def classify(tagger, img, roi):
    """One verificationResults row's verdict for a roi: (detectedAs, confidence, iou, how)."""
    h, w = img.shape[:2]
    box = box_px(roi.get("coord") or [0, 0, 0, 0], w, h)
    best = (None, 0.0, 0.0)
    for label, score, det in tagger.detect(img):
        kind = protect_type(label)
        overlap = iou(box, det)
        if kind and overlap >= MIN_IOU and score > best[1]:
            best = (kind, score, overlap)
    if best[0]:
        return best[0], best[1], best[2], "frame"
    kind, score = _crop_best(tagger, img, box)
    if kind:
        return kind, score, None, "crop"
    return None, 0.0, None, "none"


def verify(tagger, frames, params):
    """reverificationInferenceSchema result from {thumbnail_ms: jpeg_path} and the task params."""
    t0 = time.monotonic()
    rows = []
    for entry in params.get("thumbnailMeta") or []:
        ts = entry.get("ts")
        img = cv2.imread(frames[ts]) if ts in frames else None
        for roi in _rois(entry):
            claimed = str(roi.get("objectType") or (roi.get("attributes") or {}).get("objectType")
                          or "").lower()
            tracker = roi.get("trackerID", roi.get("trackerId"))
            row = {"thumbnailMs": ts, "trackerID": int(tracker) if tracker is not None else -1,
                   "objectType": claimed, "iouScore": None, "detectionConfidenceList": []}
            det_pct = roi.get("confidence") or 0
            if img is None:
                row.update(isValidDetection=True, detectedAs=claimed, modelRunStatus="skipped",
                           detectionConfidence=round(det_pct / 100, 3))
            else:
                kind, score, overlap, how = classify(tagger, img, roi)
                valid, detected, conf = verdict(claimed, det_pct, kind, score)
                row.update(isValidDetection=valid, detectedAs=detected, detectionConfidence=conf,
                           iouScore=overlap, modelRunStatus=f"success:{how}" if kind
                           else "noMatch")
                if kind and kind != detected:
                    log.info("verify %s @%s: saw %s %.2f under DIYKEY_VERIFY_OVERRIDE; kept %s",
                             params.get("event"), ts, kind, score, claimed)
            rows.append(row)
            log.info("verify %s @%s tracker %s: claimed %s -> %s %.2f (%s)", params.get("event"),
                     ts, row["trackerID"], claimed, row["detectedAs"], row["detectionConfidence"],
                     row["modelRunStatus"])
    infer_ms = int((time.monotonic() - t0) * 1000)
    return {"cameraId": params.get("camera", ""), "eventId": params.get("event", ""),
            "status": "success", "scoreThresholdApplied": params.get("score_threshold", 0.8),
            "preProcessMs": 0, "inferMs": infer_ms, "timeElapsedMs": infer_ms,
            "verificationResults": rows}
