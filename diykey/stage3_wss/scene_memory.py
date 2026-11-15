# SPDX-License-Identifier: AGPL-3.0-or-later
"""What is always in the picture, per camera, so summaries and tags follow what happened.

Live 2026-11-08 Driveway: cars passing on the road were summarised as "Red car parked in
driveway" and every event got a "car" tag, because the model describes the parked car in a whole
frame. Nothing in the AI Key does this: no firmware (2.2.8 ui-vlm-agent app.py) or Protect code
keeps a memory of the scene, so this is new behaviour, off unless DIYKEY_SCENE_MEMORY=1. What
Protect receives keeps its existing format, the description string and the whole-frame tags.

Learning happens once per recognizeKeyFrames event, from RamTagger.detect() boxes on that event's
frames. Each object is stored with its box (0..1000) and a fingerprint: OSNet for people,
CLIP for everything else. An object becomes:

  stationary  seen in the same spot in N events over a minimum time span (DIYKEY_SCENE_*).
              Vehicles need box IoU and fingerprint to match and last until they move. Animals
              and people match more loosely, weighting appearance, and expire when unseen.
  fixture     same spot and same look for 7 days (statues and posters detected as people).
              Ignored, and dropped silently the moment its look changes.

Things this module does beyond the rules, each to stop a specific false result:
  - Fingerprints are kept per lighting mode (colour day / greyscale IR). CLIP puts a night crop
    far from the same object by day (see CLAUDE.md "Traps"), so comparing across modes would
    report every parked car as replaced at dusk. An object not yet seen in this mode matches on
    a strict box alone and its fingerprint for the mode is added.
  - In an event with several frames, something seen in only one or two of them is not stationary
    evidence: that is what a passing car looks like.
  - A vehicle or animal the detector misses is looked for by fingerprinting its spot before it is
    counted missing, so one detector miss at 320 px does not report a move. Not for people: OSNet
    gives high cosines on non-person crops, so an empty chair would look like the person.

An object overlapping the event's own trigger box is never stationary evidence, and its spot is
not "visible" for the missing rule.

Limits: it learns only from frames at events, so changes are noticed at the next event, and a
moved camera resets what is learned (the old objects are reported missing once).

Every public function catches its own errors, logs them and returns a neutral value: this module
must never stop tagging or the upload.
"""
import collections
import json
import logging
import math
import os
import re
import threading
import uuid

import cv2
import numpy as np

import clip_embed
import ram_tagger
import reid_model
from core import config

log = logging.getLogger("diykey.scene")

SUBDIR = "scene_memory"
VERSION = 1
# Clusters per event that get a fingerprint (largest first) and per-camera bound on the file.
MAX_FINGERPRINTS = 16
MAX_ENTRIES = 40
MAX_EVENT_IDS = 64
CLUSTER_IOU = 0.5           # the same object across one event's frames
FP_EMA = 0.2                # how fast a matched object's fingerprint follows slow change
BLIND_IOU = 0.5             # box match needed when there is no fingerprint for this lighting

VEHICLE, ANIMAL, PERSON = "vehicle", "animal", "person"
_KINDS = {
    VEHICLE: VEHICLE,
    ANIMAL: ANIMAL,
    PERSON: PERSON,
    # car/truck/suv/... and dog/cat/bird from the same alias map the tagger uses.
    **ram_tagger.TYPE_ALIASES,
    **{k: ANIMAL for k in ram_tagger.FAMILY["animal"]},
}

# Whole-frame tags hidden with their kind; matched on the tag's last word ("white car", "sports
# car"; "car park" stays).
_HIDE_WORDS = {
    VEHICLE: set(ram_tagger.KIND_WORDS["vehicle"])
             | {"minivan", "pickup", "limo", "coupe", "hatchback", "automobile"},
    ANIMAL: set(ram_tagger.KIND_WORDS["animal"]),
}

STATIONARY, FIXTURE, CANDIDATE = "stationary", "fixture", "candidate"

Rule = collections.namedtuple("Rule", "events span_ms iou sim score expire_ms")

_locks_guard = threading.Lock()
_locks = {}                 # camera id -> lock: one event per camera at a time
_warned = set()


class EventScene:
    """What one event learned, for the tag filter and the summary."""
    __slots__ = ("changes", "still", "ignore", "hide")

    def __init__(self):
        self.changes = []   # sentences to prepend: "The parked red car has moved."
        self.still = []     # sentences to add: "The person is still there."
        self.ignore = []    # what a whole-frame summary should leave out
        self.hide = set()   # kinds whose words come out of the whole-frame tags


def enabled():
    return (os.environ.get("DIYKEY_SCENE_MEMORY") or "").strip().lower() in ("1", "true", "yes", "on")


def _warn_once(key, msg, *args):
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


def _num(suffix, default):
    raw = os.environ.get("DIYKEY_SCENE_" + suffix)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        _warn_once(suffix, "DIYKEY_SCENE_%s=%r is not a number; using %s", suffix, raw, default)
        return default


def _rules():
    """Per-kind thresholds. Starting guesses, each one a DIYKEY_SCENE_* setting to tune."""
    def rule(prefix, minutes, iou, sim, score, expire):
        return Rule(int(_num(prefix + "_EVENTS", 3)), _num(prefix + "_MINUTES", minutes) * 60000,
                    _num(prefix + "_IOU", iou), _num(prefix + "_SIM", sim),
                    None if score is None else _num(prefix + "_SCORE", score),
                    None if expire is None else _num(prefix + "_EXPIRE_MINUTES", expire) * 60000)
    return {VEHICLE: rule("VEHICLE", 30, 0.6, 0.85, None, None),
            ANIMAL: rule("ANIMAL", 10, 0.2, 0.75, 0.7, 60),
            PERSON: rule("PERSON", 10, 0.2, 0.7, 0.6, 15)}


def _iou(a, b):
    inter = _intersection(a, b)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _intersection(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _overlaps(a, b, ratio):
    """True when the overlap covers at least `ratio` of the smaller box."""
    smaller = min(a[2] * a[3], b[2] * b[3])
    return smaller > 0 and _intersection(a, b) / smaller >= ratio


def _unit(vec):
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else None


def _cos(a, b):
    return float(np.dot(np.asarray(a, np.float32), np.asarray(b, np.float32)))


def _stored(vec):
    return [round(float(v), 5) for v in vec]


def _mode(night):
    return "night" if night else "day"


def _default_embed(kind, crop):
    return reid_model.embed(crop) if kind == PERSON else clip_embed.image_embed(crop)


def _default_namer(tagger, kind, crop):
    return ram_tagger.tag_crop(tagger, crop, {"objectType": kind})


def _safe_camera(camera_id):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(camera_id))


def _path(state_dir, camera_id):
    return os.path.join(state_dir, SUBDIR, _safe_camera(camera_id) + ".json")


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") == VERSION and isinstance(data.get("entries"), list):
            data.setdefault("events", [])
            data.setdefault("last_event_ms", 0)
            return data
        log.warning("scene memory %s has an unknown format; starting again", path)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, AttributeError) as e:
        log.warning("scene memory %s unreadable (%r); starting again", path, e)
    return {"version": VERSION, "entries": [], "events": [], "last_event_ms": 0}


def _camera_lock(camera_id):
    with _locks_guard:
        return _locks.setdefault(str(camera_id), threading.Lock())


def _kind(label):
    return _KINDS.get(str(label or "").lower())


def _match_sim(entry, mode, fp):
    stored = entry["fp"].get(mode)
    return None if stored is None or fp is None else _cos(stored, fp)


def _matches(entry, rule, iou, sim, weight, fixture_iou):
    if entry["state"] == FIXTURE:  # "drops out the moment its look changes"
        return iou >= fixture_iou and (sim is None or sim >= rule.sim)
    if iou < rule.iou:
        return False
    if sim is None:
        return iou >= max(rule.iou, BLIND_IOU)
    if rule.score is None:
        return sim >= rule.sim
    return (1 - weight) * iou + weight * sim >= rule.score


def _noun_colour(kind, tags, label):
    names = [t["tag"] for t in tags or ()]
    colour = next((n.split()[0] for n in names if n.split()[0] in ram_tagger.COLOURS), None)
    words = _HIDE_WORDS.get(kind, ())
    noun = next((n.split()[-1] for n in names
                 if n.split()[-1] in words and n.split()[-1] not in (VEHICLE, ANIMAL)), None)
    noun = noun or label or kind
    return f"{colour} {noun}" if colour else noun


def _change_sentence(entry):
    name = entry.get("name") or entry["kind"]
    if entry["kind"] == VEHICLE:
        return f"The parked {name} has moved."
    if entry["kind"] == ANIMAL:
        return f"The {name} that was resting has moved."
    return "The person who was seated has left."


def _expire(entries, now, rules, keep_ms):
    """Forget stale stationary status. People and animals lose it after their unseen limit but
    keep their record a little longer, or a statue seen only at hours-apart events could never
    reach the fixture age."""
    out = []
    for e in entries:
        rule = rules.get(e.get("kind"))
        if rule is None:
            continue
        age = now - e["last_seen"]
        if rule.expire_ms is not None and age > rule.expire_ms and e["state"] != FIXTURE:
            # Not missing=0: with events further apart than the limit this runs every time, and
            # something that has gone would never reach the missing count.
            e.update(state=CANDIDATE, count=0, first_seen=None)
        lasting = e["state"] == FIXTURE or (e["state"] == STATIONARY and e["kind"] == VEHICLE)
        if lasting or age <= keep_ms:
            out.append(e)
    return out


def _detect_clusters(tagger, images):
    """One record per object across this event's frames, boxes on 0..1000."""
    clusters = []
    for ms, img, night in images:
        h, w = img.shape[:2]
        for label, score, rect in tagger.detect(img):
            kind = _kind(label)
            if kind is None:
                continue
            box = [rect[0] * 1000 / w, rect[1] * 1000 / h, rect[2] * 1000 / w, rect[3] * 1000 / h]
            for c in clusters:
                if c["kind"] == kind and ms not in c["frames"] and _iou(c["box"], box) >= CLUSTER_IOU:
                    c["frames"].add(ms)
                    if score > c["score"]:
                        c.update(box=box, score=score, label=label, img=img, night=night)
                    break
            else:
                clusters.append({"kind": kind, "label": label, "score": score, "box": box,
                                 "frames": {ms}, "img": img, "night": night, "fp": None,
                                 "crop": None})
    return clusters


def _fingerprint_clusters(clusters, embed):
    for c in sorted(clusters, key=lambda c: -c["box"][2] * c["box"][3])[:MAX_FINGERPRINTS]:
        c["crop"] = ram_tagger.crop_box(c["img"], c["box"], margin=0)
        if c["crop"] is None:
            continue
        vec = embed(c["kind"], c["crop"])
        if vec is None:
            _warn_once("embed-" + c["kind"], "no fingerprint for %s objects: is its model "
                       "enabled? They will not be learned", c["kind"])
            continue
        c["fp"] = _unit(vec)


def _assign(entries, clusters, rules, weight, fixture_iou):
    """Greedy best-first pairing of remembered objects with this event's objects."""
    pairs = []
    for i, e in enumerate(entries):
        rule = rules[e["kind"]]
        for j, c in enumerate(clusters):
            if c["kind"] != e["kind"] or c["fp"] is None:
                continue
            iou = _iou(e["box"], c["box"])
            sim = _match_sim(e, _mode(c["night"]), c["fp"])
            if _matches(e, rule, iou, sim, weight, fixture_iou):
                pairs.append((iou + (sim if sim is not None else 0), i, j, iou, sim))
    pairs.sort(reverse=True)
    by_entry, used = {}, set()
    for _, i, j, iou, sim in pairs:
        if i not in by_entry and j not in used:
            by_entry[i] = (j, iou, sim)
            used.add(j)
    return by_entry, used


def _record_evidence(entry, cluster, rule, iou, sim, now, opts, tagger, namer):
    """This event counts: update the object and promote it if it has earned it."""
    entry["count"] += 1
    if entry.get("first_seen") is None:
        entry["first_seen"] = now
    mode = _mode(cluster["night"])
    old = entry["fp"].get(mode)
    entry["fp"][mode] = _stored(cluster["fp"] if old is None else
                                _unit((1 - FP_EMA) * np.asarray(old, np.float32)
                                      + FP_EMA * cluster["fp"]))
    # The fixture clock wants the strict same spot, same look, and survives a person's expiry.
    if iou >= opts["fixture_iou"] and (sim is None or sim >= rule.sim):
        entry["fixture_events"] = entry.get("fixture_events", 0) + 1
    else:
        entry.update(fixture_since=now, fixture_events=1)
    if entry["state"] != FIXTURE and now - entry["fixture_since"] >= opts["fixture_ms"] \
            and entry["fixture_events"] >= opts["fixture_events"]:
        entry["state"] = FIXTURE
        log.info("scene: %s at %s is now a fixture", entry["label"], _round_box(entry["box"]))
    elif entry["state"] == CANDIDATE and entry["count"] >= rule.events \
            and now - entry["first_seen"] >= rule.span_ms:
        entry["state"] = STATIONARY
        entry["name"] = _name_object(entry, cluster, tagger, namer)
        log.info("scene: %s at %s is now stationary", entry["name"], _round_box(entry["box"]))


def _name_object(entry, cluster, tagger, namer):
    """Colour and type for the sentences, from tags on its crop. Best effort."""
    try:
        if cluster["crop"] is not None and entry["kind"] != PERSON:
            return _noun_colour(entry["kind"], namer(tagger, entry["kind"], cluster["crop"]),
                                entry["label"])
    except Exception as e:  # noqa: BLE001 -- a name is only ever a nicer sentence
        log.info("scene: could not name %s: %r", entry["label"], e)
    return entry["label"] if entry["kind"] != PERSON else PERSON


def _round_box(box):
    return [round(v) for v in box]


def _new_entry(cluster, now):
    return {"id": uuid.uuid4().hex[:8], "kind": cluster["kind"], "label": cluster["label"],
            "box": [round(v, 1) for v in cluster["box"]],
            "fp": {_mode(cluster["night"]): _stored(cluster["fp"])},
            "first_seen": now, "last_seen": now, "count": 1, "state": CANDIDATE, "missing": 0,
            "fixture_since": now, "fixture_events": 1}


def _spot_crop_sim(entry, images, embed):
    """Cosine between the object and what is at its spot now; None when that can't be said (no
    crop, or no fingerprint for this lighting)."""
    _, img, night = images[-1]
    stored = entry["fp"].get(_mode(night))
    crop = ram_tagger.crop_box(img, entry["box"], margin=0)
    if stored is None or crop is None:
        return None
    vec = embed(entry["kind"], crop)
    unit = _unit(vec) if vec is not None else None
    return None if unit is None else _cos(stored, unit)


def _trigger_says_moved(entry, rule, triggers, by_ms, embed, trigger_fp):
    """The event's own trigger is this object, somewhere other than its spot."""
    for t in triggers:
        if t["kind"] != entry["kind"] or _iou(t["coord"], entry["box"]) >= rule.iou:
            continue
        frame = by_ms.get(t["ts"])
        if frame is None:
            continue
        key = (t["ts"], tuple(t["coord"]))
        if key not in trigger_fp:
            crop = ram_tagger.crop_box(frame[0], t["coord"], margin=0)
            vec = embed(t["kind"], crop) if crop is not None else None
            trigger_fp[key] = _unit(vec) if vec is not None else None
        stored = entry["fp"].get(_mode(frame[1]))
        if trigger_fp[key] is not None and stored is not None \
                and _cos(stored, trigger_fp[key]) >= rule.sim:
            return True
    return False


def _describe_present(entries, scene):
    """Sentences and ignore phrases for stationary objects seen in this event."""
    for e in entries:
        name = e.get("name") or e["kind"]
        if e["state"] == FIXTURE:
            what = "motionless figure" if e["kind"] == PERSON else name
            line, target = f"the {what} that is always in the picture", scene.ignore
        elif e["kind"] == VEHICLE:
            line, target = f"the parked {name}", scene.ignore
        elif e["kind"] == ANIMAL:
            line, target = f"The {name} is still resting.", scene.still
        else:  # never hidden, even after hours
            line, target = "The person is still there.", scene.still
        if line not in target:
            target.append(line)


def _observe(state_dir, tagger, camera_id, event_id, frames, triggers, embed, namer):
    images = []
    for ms in sorted(frames):
        img = cv2.imread(frames[ms])
        if img is not None:
            images.append((int(ms), img, bool(ram_tagger.is_greyscale(img))))
    if not images:
        return None
    triggers = [{**t, "kind": _kind(t.get("objectType"))} for t in triggers or ()]
    rules = _rules()
    now = images[-1][0]
    weight = _num("APPEARANCE_WEIGHT", 0.75)
    overlap = _num("TRIGGER_OVERLAP", 0.3)
    missing_limit = int(_num("MISSING_EVENTS", 2))
    opts = {"fixture_ms": _num("FIXTURE_DAYS", 7) * 86400000,
            "fixture_iou": _num("FIXTURE_IOU", 0.6),
            "fixture_events": int(_num("FIXTURE_EVENTS", 7))}
    keep_ms = _num("KEEP_DAYS", 3) * 86400000
    path = _path(state_dir, camera_id)
    with _camera_lock(camera_id):
        store = _load(path)
        if event_id in store["events"]:
            log.info("event %s: scene memory already saw this event", event_id)
            return None
        last_ms = int(store.get("last_event_ms") or 0)
        if now < last_ms:
            log.warning("event %s: scene clock %s is older than the last learned event (%s); "
                        "not learning it", event_id, now, last_ms)
            return None
        store["last_event_ms"] = now
        entries = _expire(store["entries"], now, rules, keep_ms)
        clusters = _detect_clusters(tagger, images)
        _fingerprint_clusters(clusters, embed)
        frames_needed = 1 if len(images) < 2 else max(2, math.ceil(len(images) / 2))
        for c in clusters:
            c["steady"] = len(c["frames"]) >= frames_needed
            c["on_trigger"] = any(_overlaps(c["box"], t["coord"], overlap) for t in triggers)
        by_entry, used = _assign(entries, clusters, rules, weight, opts["fixture_iou"])
        scene = EventScene()
        by_ms = {ms: (img, night) for ms, img, night in images}
        trigger_fp, present, rescued_kinds, removed = {}, [], set(), set()
        stationary_clusters = set()
        for i, e in enumerate(entries):
            rule = rules[e["kind"]]
            if i in by_entry:
                j, iou, sim = by_entry[i]
                c = clusters[j]
                e.update(last_seen=now, missing=0)
                if c["steady"] and not c["on_trigger"]:
                    _record_evidence(e, c, rule, iou, sim, now, opts, tagger, namer)
                if e["state"] in (STATIONARY, FIXTURE):
                    present.append(e)
                    stationary_clusters.add(j)
                continue
            # Not matched. Still there, replaced, gone, or simply hidden behind something?
            spot_sim = _spot_crop_sim(e, images, embed) if e["kind"] != PERSON else None
            if spot_sim is not None and spot_sim >= rule.sim:
                e.update(last_seen=now, missing=0)  # the detector missed it
                if e["state"] in (STATIONARY, FIXTURE):
                    present.append(e)
                    rescued_kinds.add(e["kind"])
                continue
            different = any(j not in used and c["kind"] == e["kind"] and c["fp"] is not None
                            and c["steady"] and not c["on_trigger"]
                            and _iou(e["box"], c["box"]) >= rule.iou
                            and _match_sim(e, _mode(c["night"]), c["fp"]) is not None
                            for j, c in enumerate(clusters))
            moved = e["state"] == STATIONARY and _trigger_says_moved(
                e, rule, triggers, by_ms, embed, trigger_fp)
            if different or moved:
                _forget(e, scene, removed, "different look" if different else "trigger moved")
                continue
            hidden = any(_overlaps(e["box"], t["coord"], overlap) for t in triggers) \
                or any(_overlaps(e["box"], c["box"], overlap) for c in clusters)
            if hidden:
                continue
            if spot_sim is None and e["kind"] != PERSON:
                continue  # no fingerprint for this lighting yet: can't call it missing
            e["missing"] += 1
            if e["missing"] >= missing_limit:
                _forget(e, scene, removed, "missing")
        for j, c in enumerate(clusters):
            if j not in used and c["fp"] is not None and c["steady"] and not c["on_trigger"]:
                entries.append(_new_entry(c, now))
        entries = [e for e in entries if e["id"] not in removed]
        _describe_present(present, scene)
        _decide_hidden(scene, clusters, stationary_clusters, rescued_kinds, triggers)
        entries.sort(key=lambda e: (e["state"] == CANDIDATE, -e["last_seen"]))
        store["entries"] = entries[:MAX_ENTRIES]
        store["events"] = (store["events"] + [event_id])[-MAX_EVENT_IDS:]
        config.atomic_write_json(path, store)
    log.info("event %s: scene: %d objects seen, %d remembered, hide=%s, ignore=%s, still=%s, "
             "changes=%s", event_id, len(clusters), len(store["entries"]), sorted(scene.hide),
             scene.ignore, scene.still, scene.changes)
    return scene


def _forget(entry, scene, removed, why):
    """Remove an object, saying so unless it was only ever a candidate or a fixture."""
    removed.add(entry["id"])
    if entry["state"] == STATIONARY:
        sentence = _change_sentence(entry)
        if sentence not in scene.changes:
            scene.changes.append(sentence)
    log.info("scene: forgot %s %s at %s (%s)", entry["state"], entry["label"],
             _round_box(entry["box"]), why)


def _decide_hidden(scene, clusters, stationary_clusters, rescued_kinds, triggers):
    """Hide a kind's whole-frame tags only when every one detected is stationary and the trigger
    is known to be something else. People are never hidden."""
    if not triggers or any(t["kind"] is None for t in triggers):
        return
    trigger_kinds = {t["kind"] for t in triggers}
    for kind in (VEHICLE, ANIMAL):
        if kind in trigger_kinds:
            continue
        seen = [j for j, c in enumerate(clusters) if c["kind"] == kind]
        if (seen or kind in rescued_kinds) and all(j in stationary_clusters for j in seen):
            scene.hide.add(kind)


def observe(state_dir, tagger, camera_id, event_id, frames, triggers, embed=None, namer=None):
    """Learn from one event's frames ({ms: path}); the EventScene, or None when off or on error.
    triggers: [{ts, coord, objectType}] from the task's thumbnailMeta/roiMeta."""
    if not enabled() or not state_dir or not camera_id or not event_id or not frames:
        return None
    try:
        return _observe(state_dir, tagger, camera_id, event_id, frames, triggers,
                        embed or _default_embed, namer or _default_namer)
    except Exception as e:  # noqa: BLE001 -- never stops tagging or the upload
        log.warning("scene memory failed for event %s: %r", event_id, e)
        return None


def _keeps_tag(tag, hide):
    last = tag.split()[-1] if tag else ""
    return not any(last in _HIDE_WORDS[kind] for kind in hide)


def filter_result(result, scene):
    """A RAM result with hidden kinds taken out of its whole-frame tags and the changes added to
    its description. thumbnailTags (tracked objects) are left as they are. Returns `result`
    unchanged on any problem."""
    if not scene or not result:
        return result
    try:
        out = dict(result)
        if scene.hide and out.get("keyMomentsTags"):
            moments, merged, dropped = [], {}, set()
            for km in out["keyMomentsTags"]:
                tags = [t for t in km.get("tags") or () if _keeps_tag(t["tag"], scene.hide)]
                dropped |= {t["tag"] for t in km.get("tags") or ()} - {t["tag"] for t in tags}
                moments.append({**km, "tags": tags})
                for t in tags:
                    merged[t["tag"]] = max(merged.get(t["tag"], 0), t["confScore"])
            out["keyMomentsTags"] = moments
            ordered = sorted(merged.items(), key=lambda kv: -kv[1])
            out["description"] = ram_tagger.describe(
                [{"tag": t, "confScore": s} for t, s in ordered], out.get("thumbnailTags"))
            log.info("event %s: scene memory removed tags %s (description was %r)",
                     out.get("eventId"), sorted(dropped), result.get("description"))
        out["description"] = summary_text(out.get("description"), scene)
        return out
    except Exception as e:  # noqa: BLE001 -- the unfiltered result still goes
        log.warning("scene memory could not filter result: %r", e)
        return result


def summary_text(text, scene):
    """The summary with what changed in front and what is still there after it."""
    if not scene or not text:
        return text
    try:
        return " ".join([*scene.changes, text, *scene.still])
    except Exception as e:  # noqa: BLE001
        log.warning("scene memory could not extend summary: %r", e)
        return text


def ignore_context(scene):
    """Prompt line for a whole-frame summary naming what to leave out, or None."""
    try:
        if not scene or not scene.ignore:
            return None
        items = scene.ignore
        listed = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
        return f"Ignore {listed}. Do not mention them."
    except Exception as e:  # noqa: BLE001
        log.warning("scene memory could not build a prompt: %r", e)
        return None
