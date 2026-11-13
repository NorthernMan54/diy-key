# SPDX-License-Identifier: AGPL-3.0-or-later
"""Find Anything over the AI Key's nl-search connection.

A real AI Key opens a second WebSocket, wss://<console>:7443/wss/nl-search/v1
(x-ident, subprotocol ucp4, its pinned client cert). While one is open, Protect
sends every Find Anything query there instead of to a console AI Controller:

  NL_PARSE     {querySentence, model}  -> {keyTags: [{matchedWord, tags}], objectTypes,
                                           startTime?, endTime?, timeTag?, txtEmbed,
                                           model, dim, exact_match}
  IMAGE_SEARCH {imgUri}                -> {imgEmbed}

and then queries the AI Key's ramDetections: tag ids for keyTags, cosine distance
for txtEmbed/imgEmbed. Tags come from Protect's RAM vocabulary (DIYKEY_RAM_TAGS:
path to Protect's app/fixtures/ram_tags.csv; FALLBACK_VOCAB, names checked against it,
otherwise).
Embeddings are clip_embed's CLIP ViT-L/14, the space DIY-Key stores images in.
"""
import datetime
import json
import logging
import os
import re

import clip_embed
import ram_tagger

log = logging.getLogger("diykey.nlsearch")

PATH = "/wss/nl-search/v1"
PORT = 7443

OBJECT_WORDS = {
    "person": {"person", "people", "man", "men", "woman", "women", "kid", "kids", "child",
               "children", "boy", "girl", "someone", "pedestrian", "human"},
    "vehicle": {"car", "cars", "truck", "trucks", "van", "suv", "vehicle", "vehicles", "bus",
                "motorcycle", "motorbike", "bike", "bicycle", "pickup", "trailer"},
    "animal": {"dog", "dogs", "cat", "cats", "animal", "animals", "deer", "bird", "birds", "fox",
               "raccoon", "bear", "horse", "squirrel"},
}
# Every species ram_tagger can put on an animal crop is an animal word for search too, so a
# query like "raccoons" or "coyote" carries objectTypes ['animal'] and the umbrella tag (live
# 2026-11-12: "raccoons" parsed to tag raccoon but no object type).
OBJECT_WORDS["animal"] |= {w for w in ram_tagger.KIND_WORDS["animal"] if " " not in w}
# Same for vehicle kinds ("ambulance", "taxi", "fire truck"): phrases match through aliases and
# key tags, so multi-word kinds are kept too.
OBJECT_WORDS["vehicle"] |= set(ram_tagger.KIND_WORDS["vehicle"])
STOP = {"a", "an", "the", "of", "in", "on", "at", "with", "and", "or", "to", "by", "near",
        "show", "me", "find", "any", "some", "is", "are", "was", "who", "that", "wearing"}
# Query word -> the RAM tag Protect knows it by. Protect turns keyTags[].tags into ids with
# store.ramTagBidirectionalMap.get(name) and drops every miss (M/253677_aiprocessors.js:26-29);
# that map holds only the fixture's enabled=true rows, by exact lowercase name
# (M/499374_store.js:56-60, `SELECT id,name FROM "ramTags" WHERE enabled=true`). Fixture
# (up/app/fixtures/ram_tags.csv) has no "people", "kid", "bike", "pickup", "human" or "grey",
# but has person/child/bicycle/truck/gray.
ALIASES = {
    "people": "person", "someone": "person", "human": "person", "humans": "person",
    "men": "man", "women": "woman", "kid": "child", "kids": "child", "children": "child",
    "boys": "boy", "girls": "girl", "pedestrians": "pedestrian",
    "cars": "car", "trucks": "truck", "vehicles": "vehicle", "pickup": "truck", "bike": "bicycle",
    "bikes": "bicycle", "bicycles": "bicycle", "motorcycles": "motorcycle", "vans": "van",
    "buses": "bus", "trailers": "trailer",
    "dogs": "dog", "cats": "cat", "animals": "animal", "birds": "bird", "horses": "horse",
    "squirrels": "squirrel", "foxes": "fox", "bears": "bear",
    "geese": "goose", "mice": "mouse", "sheep": "sheep",
    "grey": "gray",
    # Emergency vehicles (2026-11-13): one-word and colloquial spellings of the RAM tags
    # "fire truck" / "police car"; without these "firetruck" got no tag and "fire engine"
    # split into the tags "fire" and "engine".
    "firetruck": "fire truck", "firetrucks": "fire truck", "fire engine": "fire truck",
    "fire engines": "fire truck", "cop car": "police car", "cop cars": "police car",
    "squad car": "police car",
}
# Without DIYKEY_RAM_TAGS: only names checked against the fixture's enabled=true rows
# (2026-11-07, ram_tags.csv from Protect 7.3.70; tests.test_nl_search re-checks when the
# fixture is present). The old fallback also offered "grey", which Protect drops: a rule
# "grey car" then still fired on "car", but "grey" alone stored ramTagIds=[] -- and once
# embedding and ramTagIds are set (even []) the rule is never re-parsed (M/253677:19-22), so
# it could never fire until edited.
FALLBACK_VOCAB = frozenset({
    # Protect object types and what ram_tagger puts on their crops (ram_tagger.FAMILY,
    # COCO_TO_RAM, the camera's vehicleType values in lpr.py)
    "person", "man", "woman", "child", "boy", "girl", "pedestrian",
    "vehicle", "car", "truck", "bus", "van", "suv", "sedan", "trailer", "motorcycle", "motorbike",
    "bicycle", "train", "boat", "plane", "police car",
    "animal", "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "deer", "fox", "raccoon", "squirrel",
    "backpack", "umbrella", "handbag", "luggage", "package",
    # scene
    "night", "driveway", "yard",
    # colours (ram_tagger.COLOURS minus "grey")
    "white", "black", "gray", "silver", "red", "blue", "green", "yellow", "orange", "brown",
    "purple", "pink", "beige", "gold",
})
# Object type -> the umbrella RAM tag ram_tagger.thumbnail_tags puts on every crop of it, so a
# query naming any object of that type shares a tag id with the object row; Protect's alarm
# matching returns false without that overlap before it looks at the embedding
# (M/718745_aiprocessors.js:23,33).
UMBRELLA = {"person": "person", "vehicle": "vehicle", "animal": "animal"}
_vocab = None


def vocabulary():
    """RAM tag names Protect accepts (lowercase)."""
    global _vocab
    if _vocab is None:
        names = set()
        path = os.path.expanduser(os.environ.get("DIYKEY_RAM_TAGS") or "")
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 3 and parts[2].strip().lower() == "true":
                        names.add(parts[1].strip().lower())
        except OSError:
            pass
        if not names:
            names = set(FALLBACK_VOCAB)
        log.info("nl-search vocabulary: %d tags%s", len(names), f" from {path}" if path else
                 " (built-in; set DIYKEY_RAM_TAGS to Protect's ram_tags.csv)")
        _vocab = names
    return _vocab


# Firmware 2.2.8 maps query words to RAM tags with ui-slm-runtime-engine's
# ui_slm_word_linker.json ({word: {tag: weight}}, 35,184 words; e.g. "pickup" {truck: 0.8,
# jeep: 0.4}, "12 a.m." {midnight: 1.0}). It is vendor data, so it stays outside the repo:
# DIYKEY_WORD_LINKER names a local copy. Only links at or above LINK_MIN are used, so weak
# associations ("cars" -> "train interior" 0.1) don't become exact tags.
LINK_MIN = 0.8
_linker = None


def word_linker():
    """{word: {tag: weight}} from DIYKEY_WORD_LINKER, or {} when unset or unreadable."""
    global _linker
    if _linker is None:
        _linker = {}
        path = os.path.expanduser(os.environ.get("DIYKEY_WORD_LINKER") or "")
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                _linker = {str(k).strip().lower(): v for k, v in data.items()
                           if isinstance(v, dict)}
                log.info("nl-search word linker: %d words from %s", len(_linker), path)
            except (OSError, ValueError, AttributeError) as e:
                log.warning("nl-search word linker %s unreadable: %r", path, e)
    return _linker


_PLURAL_ENDINGS = (("ies", "y"), ("ves", "f"), ("es", ""), ("s", ""))


def _canonical(word, vocab):
    """The vocabulary tag for one query word: a hand alias, the word itself, a regular
    singular ("boxes" -> "box", "trees" -> "tree", "puppies" -> "puppy"), then the firmware
    word linker's strongest link. Falls back to the word unchanged."""
    if word in ALIASES:
        return ALIASES[word]
    if word in vocab:
        return word
    for cut, add in _PLURAL_ENDINGS:
        if word.endswith(cut) and len(word) > len(cut) + 2:
            single = word[:-len(cut)] + add
            if single in vocab:
                return single
    links = word_linker().get(word) or {}
    best = max(((w, t) for t, w in links.items()
                if isinstance(w, (int, float)) and t in vocab), default=None)
    if best and best[0] >= LINK_MIN:
        return best[1]
    return word


def _object_type(word):
    """person / vehicle / animal for a query word, including regular plurals ("raccoons")."""
    forms = [word, ALIASES.get(word, word)]
    forms += [word[:-len(cut)] + add for cut, add in _PLURAL_ENDINGS
              if word.endswith(cut) and len(word) > len(cut) + 2]
    for t, ws in OBJECT_WORDS.items():
        if any(f in ws for f in forms):
            return t
    return None


def _detectable(object_type):
    """Tags ram_tagger can put on a crop of this object type (besides the umbrella)."""
    labels = ram_tagger.FAMILY.get(object_type, ())
    # Detector labels only: "suv"/"van" come from the camera's vehicleType, which most crops
    # lack (the detector calls them "car"), so those queries also carry "vehicle".
    tags = {ram_tagger.COCO_TO_RAM.get(lbl, lbl) for lbl in labels} - {None}
    return tags | {UMBRELLA[object_type]}


def _time_range(text, now=None):
    """(start_ms, end_ms, phrase) for a time phrase in text, or (None, None, None)."""
    now = now or datetime.datetime.now()
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    ms = lambda d: int(d.timestamp() * 1000)  # noqa: E731
    m = re.search(r"\b(?:in the )?(?:last|past) (\d+) (minute|hour|day|week)s?\b", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"minute": datetime.timedelta(minutes=n), "hour": datetime.timedelta(hours=n),
                 "day": datetime.timedelta(days=n), "week": datetime.timedelta(weeks=n)}[unit]
        return ms(now - delta), ms(now), m.group(0)
    fixed = [
        (r"\blast night\b", day - datetime.timedelta(hours=6), day + datetime.timedelta(hours=6)),
        (r"\bthis morning\b", day + datetime.timedelta(hours=5), day + datetime.timedelta(hours=12)),
        (r"\btonight\b", day + datetime.timedelta(hours=18), now),
        (r"\byesterday\b", day - datetime.timedelta(days=1), day),
        (r"\btoday\b", day, now),
        (r"\bthis week\b", day - datetime.timedelta(days=day.weekday()), now),
    ]
    for pattern, start, end in fixed:
        m = re.search(pattern, text)
        if m:
            return ms(start), ms(end), m.group(0)
    return None, None, None


def parse(query, now=None):
    """NL_PARSE reply for a query sentence. Every name in keyTags[].tags is in vocabulary(), and
    a query that names an object (person, car, dog, ...) always carries a tag ram_tagger puts on
    such objects: an alarm rule stores these as ramTagIds once and for all (M/253677:19-45), and
    matches an object only when its tag ids overlap them (M/718745:23,33)."""
    text = " ".join(query.lower().split())
    vocab = vocabulary()
    start, end, phrase = _time_range(text, now)
    retrieval = " ".join(text.replace(phrase, " ").split()) if phrase else text
    words = [w.strip(".,!?\"'") for w in retrieval.split()]
    key_tags, seen = [], set()
    for size in (3, 2, 1):  # longest vocabulary phrase first ("police car" over "car")
        for i in range(len(words) - size + 1):
            gram = " ".join(words[i:i + size])
            tag = _canonical(gram, vocab)  # phrases too: "fire trucks", "fire engine"
            if gram in seen or (size == 1 and gram in STOP) or tag not in vocab:
                continue
            if any(f" {gram} " in f" {s} " for s in seen):  # inside a longer match
                continue
            seen.add(gram)
            key_tags.append({"matchedWord": gram, "tags": [tag]})
    phrase_tags = [t for k in key_tags for t in k["tags"]]
    object_types = [t for t in OBJECT_WORDS
                    if any(_object_type(w) == t for w in words + phrase_tags)]
    bare = retrieval.strip()
    exact = (len(words) == 1 and _object_type(bare) is not None
             and _canonical(bare, vocab) in vocab
             and " " not in _canonical(bare, vocab)  # "firetruck" acts like "fire truck"
             and _canonical(bare, vocab) not in ram_tagger.PLUS_VEHICLE_KINDS)
    # Exact mode orders by the label's confidence, not similarity (M/718745:18-28). Live
    # 2026-11-13: "ambulance" (exact) listed RAM++'s misread red vans (ambulance ~1.0) above the
    # real ambulance (0.999, 80.7 % similarity, rank 1 by vector). The special vehicle kinds only
    # RAM++ guesses (ambulance, fire truck, police car, school bus, tow truck) therefore go
    # through the vector search; other object words keep exact mode.
    for w in words:
        t = _object_type(w)
        if t is None:
            continue
        canon = _canonical(w, vocab)
        want = [canon] if canon in vocab else []
        # Exact mode fires on its one label's confidence (M/718745:18-28): keep "car" from also
        # matching trucks via "vehicle" unless ram_tagger never says "car"-level words for it
        # ("man", "deer", "suv" crops are tagged person/animal/vehicle).
        if not exact or canon not in _detectable(t):
            want.append(UMBRELLA[t])
        want = [x for x in want if x in vocab]
        entry = next((k for k in key_tags if w in k["matchedWord"].split()), None)
        if entry is None:
            if want:
                key_tags.append({"matchedWord": w, "tags": want})
        else:
            entry["tags"] += [x for x in want if x not in entry["tags"]]
    # A phrase whose words carry no type on their own ("fire engine" -> fire truck) still gets
    # its umbrella, like "fire truck" does through "truck".
    for k in key_tags:
        t = next((_object_type(x) for x in k["tags"] if _object_type(x)), None)
        if t and UMBRELLA[t] in vocab and UMBRELLA[t] not in k["tags"] and not exact:
            k["tags"].append(UMBRELLA[t])
    reply = {"keyTags": key_tags, "objectTypes": object_types,
             "txtEmbed": clip_embed.text_embed(retrieval or text) or [],
             "model": clip_embed.MODEL_NAME, "dim": clip_embed.DIM}
    if start is not None:
        reply.update(startTime=start, endTime=end, timeTag=phrase)
    if not key_tags:
        reply["exact_match"] = start is not None  # a pure time query lists everything in it
        log.info("NL_PARSE %r: no RAM tags -- fine for search, but an alarm rule with this "
                 "sentence stores ramTagIds=[] and can never fire", query)
    elif len(key_tags) == 1:
        # Protect's exact mode searches by object label and returns nothing without one, so only
        # a bare object word ("car", "person") is exact; "night", "grey car" go to vector/hybrid.
        reply["exact_match"] = exact
    return reply


def image_search(img_bgr):
    """IMAGE_SEARCH reply for the fetched image."""
    return {"imgEmbed": clip_embed.image_embed(img_bgr) or []}
