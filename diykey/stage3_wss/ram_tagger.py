# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tag key-moment frames for a recognizeKeyFrames (RAM) result.

Protect stores RAM results by tag *name* from its fixed vocabulary
(app/fixtures/ram_tags.csv on the console, ~7.3k tags), keeps every
key-moment tag on the event and the description as ramDescription. Here the
tags come from a COCO object detector (an Ultralytics-style anchor-free ONNX
export, [1, 84, N] output, e.g. Frigate's yolov9-s-320.onnx), plus "night"
for IR (greyscale) frames. COCO names are already vocabulary tags except the
ones mapped or dropped below.

DIYKEY_RAM_PLUS_MODEL (path to an ONNX export of RAM++, Recognize Anything Plus, Apache-2.0)
adds the attributes COCO has no class for (gender, clothing, accessories, sedan/SUV...): see
RamPlus. COCO still supplies the boxes and its tags; RAM++ runs on the whole frame and on every
person/vehicle crop and its tags are merged in.
"""
import ast
import logging
import os
import threading
import time

import cv2
import numpy as np

import clip_embed
import reid_model

log = logging.getLogger("diykey.ram")

# Per-model time inside one task's tags stage, for its timing log line. Live 2026-11-09: tags was
# 24 s median of 36 s per event and swung 8.7-43 s on the same camera with the same 2 frames, but
# one timer could not say whether YOLO, RAM++, CLIP or the crop count was to blame. Thread-local:
# each task runs build_result on its own worker thread; nothing is counted until prof_reset().
_prof = threading.local()


def prof_reset():
    _prof.stats = {}


def prof_take():
    """{name: [ms, calls]} since prof_reset() on this thread; counting stops."""
    stats = getattr(_prof, "stats", None) or {}
    _prof.stats = None
    return stats


def _prof_add(name, t0, calls=1):
    stats = getattr(_prof, "stats", None)
    if stats is not None:
        s = stats.setdefault(name, [0.0, 0])
        s[0] += (time.monotonic() - t0) * 1000 if t0 is not None else 0
        s[1] += calls


_PROF_COUNTS = ("ramframes", "crops")  # counted, not timed


def prof_format(stats):
    """'ramframes=2 crops=2 yolo=310/4 ramplus=9120/4 ...': counts first, then ms/calls."""
    counts = [f"{k}={stats[k][1]}" for k in _PROF_COUNTS if k in stats]
    return " ".join(counts + [f"{k}={ms:.0f}/{n}" for k, (ms, n) in stats.items()
                              if k not in _PROF_COUNTS])

# COCO name -> Protect RAM tag; None = no supported tag for it.
COCO_TO_RAM = {
    "airplane": "plane", "suitcase": "luggage", "skis": "ski", "toilet": "toilet bowl",
    "tv": "television", "refrigerator": "fridge", "potted plant": "plant",
    "dining table": "table", "cell phone": "phone",
    "fire hydrant": None, "teddy bear": None, "sports ball": None, "hot dog": None,
    "scissors": None, "hair drier": None,
}
NIGHT_TAG = "night"
# Camera attribute values -> the fixture's name. Protect drops a thumbnail tag that is not an
# enabled ram_tags.csv row ("Some tags are not found in ramTagBidirectionalMap",
# M/12387_aiprocessors.js:181); the fixture has "gray" and "bicycle" but no "grey"/"bike"
# (lpr.py's vehicleType list includes "bike").
ATTR_TO_RAM = {"grey": "gray", "bike": "bicycle"}
# Protect object type -> the tag every one of its crops carries in thumbnailTags. Those tags are
# the object row's ramTagIds, and an alarm rule only matches a row sharing a tag id with it
# (M/718745_aiprocessors.js:23,33); nl_search.parse adds the same umbrella tag to queries naming
# an object of that type, so a "suv" or "deer" rule can still match a crop the detector called
# "car" or nothing at all.
UMBRELLA = {"person": "person", "vehicle": "vehicle", "animal": "animal"}
# ...but only when one of our own models backs the type up (claimed_type_supported). Live
# 2026-11-09 18:35:45 Driveway (event fe952e5c, tracker 2818): diy-port called a crack junction in
# the pavement "animal" (82, box static for all 21 s of the clip); RAM++ tagged the frame crack 0.88
# and found no animal on the crop, yet the row went out tagged animal 0.82, so animal alarm rules and
# tag search matched a crack. 14 Driveway "animal" detections that week, most with crack tags.
# Words that back each type when RAM++ finds them on the crop: the ones that are both RAM++ tags
# and enabled rows of Protect's ram_tags.csv (7.3.70; RamPlus.tags drops the rest).
KIND_WORDS = {
    "person": frozenset({"person", "man", "woman", "boy", "girl", "child", "pedestrian", "baby",
                         "toddler"}),
    "vehicle": frozenset({"car", "vehicle", "sedan", "suv", "truck", "van", "bus", "motorcycle",
                          "bicycle", "taxi", "police car", "ambulance", "fire truck", "school bus",
                          "tow truck", "jeep", "scooter"}),
    "animal": frozenset({"animal", "dog", "cat", "bird", "deer", "squirrel", "raccoon", "fox",
                         "rabbit", "horse", "cow", "sheep", "bear", "coyote", "goose", "duck", "pet",
                         "wildlife", "rat", "mouse", "crow", "pigeon", "turkey", "moose", "wolf",
                         "hen", "chicken", "owl", "eagle", "insect", "butterfly", "frog", "snake",
                         "turtle"}),
}


# Species on an animal crop. plus_filter keeps only words Protect turns into a filter label, which
# has no species, so a RAM++ "cat" never reached the object row that search queries. Live
# 2026-11-09 18:27:07 Doorbell (event 9be8c45f, tracker 74537): the summary said "A cat
# walking through a gate" and the whole frame was tagged cat 0.90, but the object went out as
# animal/night/bear 0.62 (COCO's guess), so Find Anything and Spotlight found no cat.
# A crop's species is kept only when the event's whole-frame tags name it too: on tiny crops
# RAM++ invents animals (2026-11-09 18:35 Driveway crack crop: bird 0.83, skier, diver; the whole
# frame said crack and pavement). A size floor would not separate the two (crack crop ~205x105 px).
# Without whole frames (image tasks), the crop's short side must reach DIYKEY_SPECIES_MIN_PX.
SPECIES = KIND_WORDS["animal"] - {"animal", "pet", "wildlife"}


# CLIP as a second opinion on a claimed animal. Real crops 2026-11-09 (3 frames each, labels =
# SPECIES + "animal" + NOT_ANIMAL): the 18:35 Driveway crack read "crack in the pavement" 0.99 in
# every frame, where COCO said airplane and RAM++ bird/skier/person; two Backyard raccoons (02:24,
# 02:30) never had a non-animal label above 0.30. Species from CLIP alone were right in only 3 of
# 6 raccoon frames (bear/cat otherwise), so CLIP vetoes here and does not name species.
NOT_ANIMAL = ("crack in the pavement", "stain on the pavement", "empty yard", "empty driveway",
              "shadow", "pile of leaves", "tree stump", "garbage bin", "blanket", "person", "car",
              "bicycle")
ANIMAL_LABELS = tuple(sorted(SPECIES)) + ("animal",)


def clip_veto_threshold():
    """DIYKEY_CLIP_VETO: the CLIP probability of "not an animal" that removes a claimed animal's
    animal tags (default 0.9; 0 turns the check off)."""
    try:
        return float(os.environ.get("DIYKEY_CLIP_VETO") or 0.9)
    except ValueError:
        return 0.9


def not_animal_prob(image_vec):
    """CLIP's probability that a crop is one of NOT_ANIMAL rather than an animal; None without
    CLIP."""
    probs = clip_embed.label_probs(image_vec, ANIMAL_LABELS + NOT_ANIMAL)
    return None if probs is None else sum(probs[label] for label in NOT_ANIMAL)


def _species_min_px():
    try:
        return int(os.environ.get("DIYKEY_SPECIES_MIN_PX") or 160)
    except ValueError:
        return 160


def agreed_species(raw_plus, img_bgr, frame_tags):
    """{species: score} from a crop's RAM++ tags that may stand: named by the event's whole
    frames too, or (no frames, frame_tags None) from a crop big enough to trust on its own."""
    found = {t: s for t, s in (raw_plus or {}).items() if t in SPECIES}
    if frame_tags is not None:
        return {t: s for t, s in found.items() if t in frame_tags}
    return found if min(img_bgr.shape[:2]) >= _species_min_px() else {}


def _trust_claimed_type():
    """DIYKEY_TRUST_CLAIMED_TYPE=1: tag every crop with Protect's type, as before 2026-11-09."""
    return (os.environ.get("DIYKEY_TRUST_CLAIMED_TYPE") or "").strip().lower() in (
        "1", "true", "yes", "on")


# RAM++ tags only meaningful on one kind of object: dropped from crops of another kind (a "man"
# on a car crop is the driver behind the glass; Protect would file the car under gender:male).
PLUS_PERSON_ONLY = {"man", "woman", "boy", "girl"}
PLUS_VEHICLE_ONLY = {"sedan", "suv", "truck", "van", "bus", "motorcycle", "bicycle", "pickup truck",
                     "minivan", "taxi", "limo"}
# Also kept on a vehicle crop though Protect makes no filter label of them. Live 2026-11-09 Driveway
# clips: only a crop reached the threshold (police car 0.95 against 0.62 on the whole frame, school
# bus 1.00 against 0.66), and none of 35 ordinary vehicle crops did.
PLUS_VEHICLE_KINDS = {"police car", "ambulance", "fire truck", "school bus", "tow truck"}
# Used when DIYKEY_RAM_TAGS is unset: the RAM++ tags Protect's convertRamTagsToLabels turns into a
# Find Anything filter (gender, clothing type, accessory, vehicle type), all enabled rows of
# ram_tags.csv on 7.3.70 (587258_convertRamTagsToLabels.js, 594367_*.js).
PLUS_FALLBACK_VOCAB = frozenset({
    "man", "woman", "boy", "girl", "shirt", "hoodie", "coat", "suit", "pant", "short", "skirt",
    "glasses", "hat", "backpack", "handbag", "helmet", "mask", "sedan", "suv", "truck", "van",
    "bus", "motorcycle", "bicycle", "person", "car", "dog", "cat", "bird", "night",
})


def _protect_vocab():
    """Enabled tag names of Protect's ram_tags.csv (DIYKEY_RAM_TAGS), or the fallback set."""
    path = os.path.expanduser(os.environ.get("DIYKEY_RAM_TAGS") or "")
    names = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 3 and parts[2].strip().lower() == "true":
                    names.add(parts[1].strip().lower())
    except OSError:
        pass
    return names or set(PLUS_FALLBACK_VOCAB)


class RamPlus:
    """RAM++ (swin-L, 384x384) image tagging from an ONNX export: image [1,3,384,384] (RGB,
    ImageNet-normalised) -> sigmoid probabilities [1, 4585], one per tag of ram_plus_tags.txt.
    Next to the model: ram_plus_tags.txt (RAM++'s ram_tag_list.txt) and ram_plus_thresholds.txt
    (its ram_tag_list_threshold.txt, one per-class threshold per line; a tag is output when its
    probability exceeds its own threshold, not a global one). Lazy: the session loads on first use
    (CoreML on Apple Silicon, else CPU), under one lock. Names are lowercased and only tags that
    are enabled rows of Protect's ram_tags.csv are kept."""
    SIZE = 384
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path, vocab=None, max_tags=None):
        self.model_path = os.path.expanduser(model_path)
        self.vocab = vocab
        self.max_tags = int(max_tags if max_tags is not None
                            else os.environ.get("DIYKEY_RAM_PLUS_MAX") or 16)
        self._lock = threading.Lock()
        self._session = None
        self._failed = False

    def _load(self):
        import onnxruntime as ort
        d = os.path.dirname(self.model_path)
        with open(os.path.join(d, "ram_plus_tags.txt"), encoding="utf-8") as f:
            names = [ln.strip().lower() for ln in f.read().splitlines()]
        with open(os.path.join(d, "ram_plus_thresholds.txt"), encoding="utf-8") as f:
            thr = [float(ln) for ln in f.read().split()]
        if len(thr) != len(names):
            raise ValueError(f"{len(names)} RAM++ tags but {len(thr)} thresholds")
        vocab = self.vocab if self.vocab is not None else _protect_vocab()
        self.names, self.thresholds = names, np.array(thr, np.float32)
        self.keep = np.array([n in vocab for n in names])
        provs = ["CPUExecutionProvider"]
        if os.environ.get("DIYKEY_RAM_PLUS_PROVIDER", "coreml").lower() == "coreml" \
                and "CoreMLExecutionProvider" in ort.get_available_providers():
            provs.insert(0, ("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
                "ModelCacheDirectory": os.path.join(d, ".coreml_cache")}))
        self._session = ort.InferenceSession(self.model_path, providers=provs)
        log.info("RAM++: %s (%s, %d of %d tags in Protect's vocabulary)", self.model_path,
                 self._session.get_providers()[0], int(self.keep.sum()), len(names))

    def _prep(self, img_bgr):
        h, w = img_bgr.shape[:2]
        interp = cv2.INTER_AREA if max(h, w) > self.SIZE else cv2.INTER_LINEAR
        x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (self.SIZE, self.SIZE),
                       interpolation=interp).astype(np.float32) / 255
        return ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None]

    def tags(self, img_bgr):
        """{tag: probability} for one image: tags above their per-class threshold that Protect
        knows, best max_tags by probability. {} when the model cannot load."""
        t0 = time.monotonic()
        try:
            return self._tags(img_bgr)
        finally:
            _prof_add("ramplus", t0)  # includes waiting for the lock (warm-up, scene memory)

    def _tags(self, img_bgr):
        with self._lock:
            if self._failed:
                return {}
            try:
                if self._session is None:
                    self._load()
                probs = self._session.run(None, {self._session.get_inputs()[0].name:
                                                 self._prep(img_bgr)})[0][0]
            except Exception as e:  # noqa: BLE001 -- RAM++ is an extra; COCO tags still go out
                log.warning("RAM++ failed (%r); disabled", e)
                self._failed = True
                return {}
        hit = np.nonzero(self.keep & (probs > self.thresholds))[0]
        hit = hit[np.argsort(-probs[hit])][:self.max_tags]
        return {self.names[i]: float(probs[i]) for i in hit}


# Words Protect's convertRamTagsToLabels turns into a filter label (594367_*.js lists), optionally
# behind a colour: gender, clothing types, accessories, vehicle types.
_LABEL_NOUNS = {"man", "boy", "woman", "girl", "shirt", "hoodie", "coat", "suit", "skirt", "pant",
                "short", "glasses", "glass", "hat", "backpack", "handbag", "helmet", "mask",
                "bike", "motorcycle", "sedan", "suv", "truck", "van", "bus", "clothing"}
_LABEL_COLOURS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black",
                  "white", "gray", "grey"}


def label_tag(tag):
    """True when Protect maps this tag to a Find Anything label (or it is an umbrella tag)."""
    first, _, rest = tag.partition(" ")
    noun = rest if first in _LABEL_COLOURS else tag
    return noun in _LABEL_NOUNS or tag in UMBRELLA.values()


def plus_filter(plus_tags, object_type):
    """RAM++ tags that fit a crop of this Protect object type (see PLUS_PERSON_ONLY). On a
    person/vehicle/animal crop only label-producing tags survive: live 2026-11-07 doorbell person
    crop got oven/food/baker/tray, which would pollute Find Anything."""
    kinds = PLUS_VEHICLE_KINDS if object_type == "vehicle" else ()
    out = {t: v for t, v in plus_tags.items() if label_tag(t) or t in kinds} \
        if object_type else dict(plus_tags)
    special = sorted((t for t in out if t in kinds), key=out.get)
    for t in special[:-1]:  # one vehicle, one kind: the 12:56 police car also scored ambulance 0.66
        del out[t]
    if object_type and object_type != "person":
        for t in PLUS_PERSON_ONLY:
            out.pop(t, None)
    if object_type and object_type != "vehicle":
        for t in PLUS_VEHICLE_ONLY:
            out.pop(t, None)
    return out


class RamTagger:
    def __init__(self, model_path, conf=0.35, iou=0.5, plus_path=None):
        import onnxruntime as ort
        plus_path = plus_path if plus_path is not None else os.environ.get("DIYKEY_RAM_PLUS_MODEL")
        self.plus = RamPlus(plus_path) if plus_path else None
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        names = self.session.get_modelmeta().custom_metadata_map.get("names")
        self.labels = ast.literal_eval(names) if names else {}
        self.conf, self.iou = conf, iou
        log.info("RAM tagger: %s (%dpx, %d classes)", model_path, self.size, len(self.labels))

    def _letterbox(self, img):
        h, w = img.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        return canvas, scale, left, top

    def detect(self, img_bgr):
        """[(label, score, (x, y, w, h) in image pixels)], after NMS."""
        t0 = time.monotonic()
        canvas, scale, left, top = self._letterbox(img_bgr)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
        out = self.session.run(None, {self.input_name: x})[0][0].T  # [N, 4 + classes]
        _prof_add("yolo", t0)
        scores = out[:, 4:]
        cls = scores.argmax(1)
        best = scores[np.arange(len(cls)), cls]
        keep = best >= self.conf
        boxes, best, cls = out[keep, :4], best[keep], cls[keep]
        rects = [[float((cx - bw / 2 - left) / scale), float((cy - bh / 2 - top) / scale),
                  float(bw / scale), float(bh / scale)] for cx, cy, bw, bh in boxes]
        idx = cv2.dnn.NMSBoxes(rects, best.tolist(), self.conf, self.iou) if rects else []
        return [(self.labels.get(int(cls[i]), str(int(cls[i]))), float(best[i]), tuple(rects[i]))
                for i in np.array(idx).flatten()]

    def tag_frame(self, img_bgr, plus=True):
        """[{tag, confScore}] for one frame, best score per tag. plus=False leaves RAM++ out
        (tag_crop runs it itself, after the family filter)."""
        found = {}
        for label, score, _ in self.detect(img_bgr):
            tag = COCO_TO_RAM.get(label, label)
            if tag and score > found.get(tag, 0):
                found[tag] = score
        if self.plus and plus:
            for tag, score in self.plus.tags(img_bgr).items():
                found[tag] = max(found.get(tag, 0), score)
        if is_greyscale(img_bgr):
            found[NIGHT_TAG] = 0.9
        return [{"tag": t, "confScore": round(s, 3)}
                for t, s in sorted(found.items(), key=lambda kv: -kv[1])]


# Detector labels that are consistent with each Protect object type (crops are of one object).
FAMILY = {
    "person": {"person"},
    "vehicle": {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"},
    "animal": {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
               "giraffe"},
}


def claimed_type_supported(object_type, coco_tags, plus_tags, plus_ran, camera_typed=False):
    """Whether a crop may carry Protect's claimed type as a tag (tag_crop's object_type and
    thumbnail_tags' umbrella). Yes when COCO found a label of that family on the crop, RAM++ found
    one of KIND_WORDS, or the camera classified the vehicle itself; also whenever RAM++ did not
    run (no second opinion), for types without a word list, and with DIYKEY_TRUST_CLAIMED_TYPE=1.
    No only when RAM++ looked at the crop and saw nothing of that kind (see KIND_WORDS)."""
    words = KIND_WORDS.get(object_type)
    if words is None or not plus_ran or camera_typed or _trust_claimed_type():
        return True
    family_tags = {COCO_TO_RAM.get(lbl, lbl) for lbl in FAMILY.get(object_type, ())}
    return any(t in family_tags for t in coco_tags) or any(t in words for t in plus_tags or ())


def tag_crop(tagger, img_bgr, hint, frame_tags=None):
    """Tags for a crop of one tracked object. hint is the task's image entry: objectType plus the
    camera's attributes (vehicleType, color). Detector labels outside the object's family are
    dropped (car crops read as boat/train), and the camera's own attributes are added.
    frame_tags: tag names from the event's whole frames, or None (see agreed_species)."""
    return crop_tags(tagger, img_bgr, hint, frame_tags)[0]


def crop_tags(tagger, img_bgr, hint, frame_tags=None):
    """(tag_crop's tags, claimed_type_supported for this crop)."""
    object_type = str(hint.get("objectType") or (hint.get("attributes") or {}).get("objectType")
                      or "").lower()
    family = FAMILY.get(object_type)
    attrs = hint.get("attributes") or {}
    family_tags = {COCO_TO_RAM.get(lbl, lbl) for lbl in family or ()}
    found = {}
    plus = getattr(tagger, "plus", None)
    coco = tagger.tag_frame(img_bgr, plus=False) if plus else tagger.tag_frame(img_bgr)
    for t in coco:
        if t["tag"] == NIGHT_TAG or family is None:
            found[t["tag"]] = t["confScore"]
        elif t["tag"] in family_tags and not (attrs.get("vehicleType") or {}).get("val") \
                and not any(k in family_tags for k in found):
            found[t["tag"]] = t["confScore"]  # one object: only the best label of its family
    raw_plus = {}
    if plus:
        # RAM++ on the crop: gender, clothing, accessories, vehicle body type. Added after the
        # family filter above (which exists to stop COCO's wrong-class guesses on crops).
        raw_plus = plus.tags(img_bgr)
        extra = plus_filter(raw_plus, object_type)
        if (attrs.get("vehicleType") or {}).get("val"):  # the camera already named the body type
            extra = {t: v for t, v in extra.items() if t not in PLUS_VEHICLE_ONLY}
        for tag, score in extra.items():
            found[tag] = max(found.get(tag, 0), round(score, 3))
    species = agreed_species(raw_plus, img_bgr, frame_tags) if object_type == "animal" else {}
    if species:
        # One animal, one species: RAM++'s agreed word replaces COCO's family guess (bear on a cat).
        best = max(species, key=species.get)
        for t in family_tags:
            found.pop(t, None)
        found[best] = round(species[best], 3)
    for key in ("vehicleType", "color"):
        val = (attrs.get(key) or {}).get("val")
        if val:
            conf = round(min(1.0, (attrs[key].get("confidence") or 0) / 100), 3)
            tag = ATTR_TO_RAM.get(str(val).lower(), str(val).lower())
            found[tag] = max(found.get(tag, 0), conf)
    # RAM++ species words back "animal" only under the same agreement rule.
    evidence = {t: s for t, s in raw_plus.items() if t not in SPECIES or t in species} \
        if object_type == "animal" else raw_plus
    supported = claimed_type_supported(
        object_type, [t["tag"] for t in coco], evidence,
        plus_ran=bool(plus) and not getattr(plus, "_failed", False),
        camera_typed=object_type == "vehicle" and bool((attrs.get("vehicleType") or {}).get("val")))
    if supported and object_type and object_type not in ("vehicle",) and object_type not in found:
        found[object_type] = round((hint.get("confidence") or 0) / 100, 3)
    tags = [{"tag": t, "confScore": c} for t, c in sorted(found.items(), key=lambda kv: -kv[1])]
    return tags, supported


def is_greyscale(img_bgr, tol=3.0):
    """IR night frames come through with (nearly) equal channels."""
    small = cv2.resize(img_bgr, (64, 36)).astype(np.int16)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    return float(np.mean(np.abs(b - g)) + np.mean(np.abs(g - r))) / 2 < tol


COLOURS = {"white", "black", "grey", "gray", "silver", "red", "blue", "green", "yellow",
           "orange", "brown", "purple", "pink", "beige", "gold"}


# What a lone colour tag can describe. It is the vehicle's (camera attribute or RAM++ on the
# frame); clothing colours come as one tag ("black jacket"). Live 2026-11-10 08:43-08:45 Driveway:
# put on the first thing it gave "Red road, car" and, once scene memory dropped the parked car,
# "Red woman".
_COLOURABLE = {"car", "vehicle"} | PLUS_VEHICLE_ONLY | PLUS_VEHICLE_KINDS


def describe(tags):
    """One sentence from the tags, e.g. "Person and car detected at night." or "Person and
    white car detected." (a colour tag goes in front of the first vehicle; with none it is left
    out)."""
    names = [t["tag"] for t in tags if t["tag"] != NIGHT_TAG]
    colours = [n for n in names if n in COLOURS]
    things = [n for n in names if n not in COLOURS]
    target = next((i for i, n in enumerate(things) if n in _COLOURABLE), None)
    if colours and target is not None:
        things[target] = f"{colours[0]} {things[target]}"
    when = " at night" if any(t["tag"] == NIGHT_TAG for t in tags) else ""
    if not things:
        return f"Motion detected{when}."
    listed = things[0] if len(things) == 1 else ", ".join(things[:-1]) + " and " + things[-1]
    return f"{listed[0].upper()}{listed[1:]} detected{when}."


def crop_box(img, coord, margin=0.1):
    """The box [x, y, w, h] (0..1000 of the frame, top-left) cut out of img, or None."""
    h, w = img.shape[:2]
    x, y, bw, bh = (float(v) for v in coord[:4])
    x, y, bw, bh = x * w / 1000, y * h / 1000, bw * w / 1000, bh * h / 1000
    x0, y0 = max(0, int(x - bw * margin)), max(0, int(y - bh * margin))
    x1, y1 = min(w, int(x + bw * (1 + margin))), min(h, int(y + bh * (1 + margin)))
    return img[y0:y1, x0:x1] if x1 - x0 >= 8 and y1 - y0 >= 8 else None


def thumbnail_tags(tagger, objects, frame_tags=None):
    """thumbnailTags for tracked objects: [{trackerID, keyMomentMs, img, hint}] ->
    [{trackerID, keyMomentMs, tags, imgEmbed?, reidEmbed?}]. Protect stores an object's row in
    the AI Key's ramDetections (what search queries) only when it carries imgEmbed, matched to
    the smart-detect object by trackerId and keyMomentMs == detectedAt. reidEmbed is what
    tap-to-find searches (ramDetections.reidEmbedding, M/499374:40); the firmware fills it for
    one object class only (FUN_0030d234 `*param_4 == 1`, infer agent 0.0.24.4) and its segmenter
    is person-only (trt_engine.py class_pred), so people only here too."""
    out = []
    for o in objects:
        if o.get("img") is None:
            continue
        _prof_add("crops", None)
        hint = o.get("hint") or {}
        tags, supported = crop_tags(tagger, o["img"], hint, frame_tags)
        entry = {"trackerID": int(o["trackerID"]), "keyMomentMs": int(o["keyMomentMs"]),
                 "tags": tags}
        obj_type = str(hint.get("objectType") or (hint.get("attributes") or {})
                       .get("objectType") or "").lower()
        t0 = time.monotonic()
        raw = clip_embed.image_embed(o["img"])
        _prof_add("clipemb", t0)
        veto = clip_veto_threshold()
        if obj_type == "animal" and raw and veto > 0:
            t0 = time.monotonic()
            p = not_animal_prob(raw)
            _prof_add("veto", t0)
            if p is not None and p >= veto:
                animal_words = SPECIES | {COCO_TO_RAM.get(lbl, lbl) for lbl in FAMILY["animal"]} \
                    | {"animal"}
                dropped = [t["tag"] for t in entry["tags"] if t["tag"] in animal_words]
                entry["tags"] = [t for t in entry["tags"] if t["tag"] not in animal_words]
                supported = False
                log.info("tracker %s: CLIP says not an animal (%.2f >= DIYKEY_CLIP_VETO=%.2f); "
                         "dropped %s", entry["trackerID"], p, veto, ", ".join(dropped) or "nothing")
        # Blend before the umbrella tag: DIYKEY_CLIP_TAG_BLEND was tuned on "a photo of car,
        # night", not "car, night, vehicle".
        emb = clip_embed.blend(raw, [t["tag"] for t in entry["tags"]])
        if emb:
            entry["imgEmbed"] = emb
        reid = None
        if obj_type == "person":
            t0 = time.monotonic()
            reid = reid_model.embed(o["img"])
            _prof_add("reid", t0)
        if reid:
            entry["reidEmbed"] = reid
        umbrella = UMBRELLA.get(obj_type)
        if umbrella and not supported:
            log.info("tracker %s: Protect says %s but our models don't back it "
                     "(tags %s); not tagged %s", entry["trackerID"], obj_type,
                     ", ".join(t["tag"] for t in entry["tags"]) or "none", umbrella)
        elif umbrella and all(t["tag"] != umbrella for t in entry["tags"]):
            entry["tags"].append({"tag": umbrella,
                                  "confScore": round((hint.get("confidence") or 50) / 100, 3)})
        out.append(entry)
    return out




def _roi_list(roi_meta, ms):
    """The roi entries of roiMeta items with ts == ms; a roi is one dict or a list of them (the
    firmware's jq `select(.ts == $km) | .roi` passes either through, SW:2737-2760)."""
    out = []
    for item in roi_meta or ():
        if item.get("ts") != ms:
            continue
        roi = item.get("roi")
        out.extend(r for r in (roi if isinstance(roi, list) else [roi]) if isinstance(r, dict))
    return out


def build_result(tagger, camera_id, event_id, frames, pre_ms=0, hints=None, objects=None,
                 roi_meta=None):
    """recognizeKeyFrames result (ramInferenceSchema) from {moment_ms: jpeg_path}; hints maps
    moment_ms -> the image entry when the frame is a crop of one object (image tasks).

    With roiMeta each key moment also carries tagObjs, that moment's roi list verbatim
    (SW:2737-2760; not in ramInferenceSchema M/315148:75, so firmware parity only).

    Never sends keyMomentsTags[].searchSnapshots. ubnt_ai_infer_agent 0.0.24.4 (fw 2.2.8)
    RunRecognizeAnything sets tags and tagObjs on each key moment but never that field; its
    only writer is the JSON loader, and /tmp/snapshots crops are face/LPR/pose only, so the
    upload loop at SW:4062-4070 has nothing to send. Protect turns every entry into a NEW
    smartDetectObject (M/12387:244-262): live 2026-11-08 07:55 on Backyard one walker became
    two person detections with one trackerId, and 16:18 on Doorbell a parked car became three
    tiles. Tracked objects are tagged through thumbnailTags only."""
    t0 = time.monotonic()
    key_moments, merged = [], {}
    for ms in sorted(frames):
        img = cv2.imread(frames[ms])
        if img is None:
            log.warning("could not read frame %s", frames[ms])
            continue
        _prof_add("ramframes", None)
        hint = (hints or {}).get(ms)
        tags = tag_crop(tagger, img, hint) if hint else tagger.tag_frame(img)
        km = {"keyMomentMs": int(ms), "tags": tags}
        rois = _roi_list(roi_meta, ms)
        if rois:
            km["tagObjs"] = rois
        key_moments.append(km)
        for t in tags:
            merged[t["tag"]] = max(merged.get(t["tag"], 0), t["confScore"])
    # Whole-frame tags vouch for a crop's species (agreed_species); image tasks have no whole frame.
    frame_tags = set(merged) if key_moments and not hints else None
    thumbs = thumbnail_tags(tagger, objects or [], frame_tags)
    infer_ms = int((time.monotonic() - t0) * 1000)
    all_tags = [{"tag": t, "confScore": s} for t, s in sorted(merged.items(), key=lambda kv: -kv[1])]
    return {
        "cameraId": camera_id,
        "eventId": event_id,
        "status": "success",
        "description": describe(all_tags),
        "keyMomentsTags": key_moments,
        **({"thumbnailTags": thumbs} if thumbs else {}),
        "inferBoxMs": infer_ms,
        "inferTagMs": 0,
        "inferTxtMs": 0,
        "preProcessMs": int(pre_ms),
        "timeElapsedMs": int(pre_ms) + infer_ms,
    }
