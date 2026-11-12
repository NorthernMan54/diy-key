# SPDX-License-Identifier: AGPL-3.0-or-later
"""Event meta helpers for recognizeKeyFrames: flatten personMeta/vehicleMeta, derive them
from roiMeta for 'videoWithRecognition' tasks, and map meta timestamps to wall clock.

Split out of ucp4_client.py (pure functions; the AI-setting gates stay in
ucp4_client._wanted_metas). Sources are cited per function.
"""
import logging

log = logging.getLogger("ucp4-client")

EPOCH_MS_MIN = 10 ** 12


def _flatten_meta(meta):
    """personMeta/vehicleMeta entries with one roi each. The schema makes roi one object
    (Protect M/346308:76-80, 96-97) but Protect's own linkAiKeyFacesToPersons walks a
    personMeta roi as an array (M/321055:78-90); accept both."""
    out = []
    for entry in meta or []:
        if not isinstance(entry, dict) or entry.get("ts") is None:
            continue
        roi = entry.get("roi")
        if isinstance(roi, list):
            out.extend({**entry, "roi": r} for r in roi if isinstance(r, dict))
        elif isinstance(roi, dict):
            out.append(entry)
    return out


_ROI_PERSON = ("person",)
_ROI_VEHICLE = ("vehicle", "car", "truck", "bus", "motorcycle")


def _roi_derived_metas(body):
    """{"person": [...], "vehicle": [...]} personMeta/vehicleMeta-shaped entries built from
    roiMeta, only for ramType 'videoWithRecognition' and only for the kinds the body lacks.

    Live 21:04 2026-11-07, Protect 7.3.74 (doorbell camera under "Legacy Camera
    Enhancement"): videoWithRecognition carries only roiMeta (and thumbnailMeta), no
    personMeta/vehicleMeta/faceMeta, each roi {name, coord, trackerID, trackerId,
    attributes: {objectType}|None, confidence, objectType}. An explicit personMeta/
    vehicleMeta always wins; the AI-setting gates stay in _wanted_metas."""
    if body.get("ramType") != "videoWithRecognition":
        return {}
    out = {}
    for kind, key, types in (("person", "personMeta", _ROI_PERSON),
                             ("vehicle", "vehicleMeta", _ROI_VEHICLE)):
        if body.get(key):
            continue
        found = []
        for e in _flatten_meta(body.get("roiMeta")):
            r = e["roi"]
            otype = r.get("objectType") or (r.get("attributes") or {}).get("objectType")
            if otype in types and r.get("coord") is not None:
                tid = r.get("trackerId", r.get("trackerID"))
                found.append({"ts": e["ts"], "roi": {"coord": r["coord"], "objectType": otype,
                                                     "trackerId": tid}})
        if found:
            out[kind] = found
    if out:
        log.info("event %s: derived %d person / %d vehicle rois from roiMeta",
                 body.get("event"), len(out.get("person", ())), len(out.get("vehicle", ())))
    return out


def _meta_start_ms(metas, start):
    """start_ms for face_recog/lpr: None when the meta ts are epoch ms, so their
    clockBestWall is the ts itself, the same epoch ms RAM sends as keyMomentMs; the clip
    start when they look like offsets (then wall = start + ts). Unverified live which one
    Protect sends; decided per event so a snapshot's wall never mixes the two."""
    ts = [float(e["ts"]) for entries in metas.values() for e in entries]
    return None if not ts or min(ts) >= EPOCH_MS_MIN else int(start)


def _meta_wall_ms(ts, meta_start):
    return int(float(ts)) + (meta_start or 0)
