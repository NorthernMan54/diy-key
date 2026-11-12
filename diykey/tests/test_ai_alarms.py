# SPDX-License-Identifier: AGPL-3.0-or-later
"""AI Key Advanced alarms: the tags nl_search gives a rule sentence and the tags ram_tagger
gives an object must be names Protect keeps and must overlap, and the stored embedding is
scored with Protect's own formula (M/718745_aiprocessors.js)."""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import clip_embed  # noqa: E402
import nl_search  # noqa: E402
import ram_tagger  # noqa: E402
import ucp4_client  # noqa: E402

# Protect's fixture, when this checkout sits next to the extracted console (dev Macs).
FIXTURE = os.path.expanduser("~/Code/up/app/fixtures/ram_tags.csv")


def _enabled_fixture_names():
    names = set()
    with open(FIXTURE, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 3 and parts[2] == "true":
                names.add(parts[1])
    return names


def _rule_tags(reply):
    """What Protect keeps of an NL_PARSE reply: keyTags flattened (M/253677:26-29)."""
    return {t for k in reply["keyTags"] for t in k["tags"]}


class _FakeTagger:
    """ram_tagger.RamTagger stand-in: tag_frame returns fixed detector tags."""
    def __init__(self, tags):
        self.tags = tags

    def tag_frame(self, _img):
        return [{"tag": t, "confScore": 0.8} for t in self.tags]


def _object_tags(detector_tags, hint):
    img = np.zeros((64, 64, 3), np.uint8)
    with mock.patch.object(clip_embed, "image_embed", return_value=None):
        out = ram_tagger.thumbnail_tags(_FakeTagger(detector_tags),
                                        [{"trackerID": 1, "keyMomentMs": 5, "img": img,
                                          "hint": hint}])
    return {t["tag"] for t in out[0]["tags"]}


class _VocabCase(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {"DIYKEY_CLIP": "0"}, clear=False)
        self._env.start()
        os.environ.pop("DIYKEY_RAM_TAGS", None)
        nl_search._vocab = None

    def tearDown(self):
        self._env.stop()
        nl_search._vocab = None


class TestFallbackVocabulary(_VocabCase):
    def test_fallback_is_the_checked_list(self):
        self.assertEqual(nl_search.vocabulary(), set(nl_search.FALLBACK_VOCAB))
        self.assertNotIn("grey", nl_search.vocabulary())

    @unittest.skipUnless(os.path.exists(FIXTURE), "Protect fixture not present")
    def test_fallback_and_tagger_names_exist_in_protect_fixture(self):
        enabled = _enabled_fixture_names()
        self.assertEqual(set(nl_search.FALLBACK_VOCAB) - enabled, set())
        self.assertEqual(set(nl_search.ALIASES.values()) - enabled, set())
        self.assertEqual(set(ram_tagger.ATTR_TO_RAM.values()) - enabled, set())
        self.assertEqual(set(ram_tagger.UMBRELLA.values()) - enabled, set())
        family = {ram_tagger.COCO_TO_RAM.get(lbl, lbl)
                  for labels in ram_tagger.FAMILY.values() for lbl in labels}
        self.assertEqual(family - {None} - enabled, set())

    def test_grey_maps_to_gray(self):
        reply = nl_search.parse("grey car")
        self.assertIn("gray", _rule_tags(reply))
        self.assertNotIn("grey", _rule_tags(reply))

    def test_csv_vocabulary_filters_every_tag(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("1\tcar\ttrue\n2\tvehicle\ttrue\n3\tperson\ttrue\n4\tred\tfalse\n")
        try:
            os.environ["DIYKEY_RAM_TAGS"] = f.name
            nl_search._vocab = None
            for q in ("red car", "a man", "people near a truck", "grey suv"):
                tags = _rule_tags(nl_search.parse(q))
                self.assertTrue(tags <= {"car", "vehicle", "person"}, (q, tags))
                self.assertTrue(tags, q)
        finally:
            os.unlink(f.name)


class TestObjectTagOverlap(_VocabCase):
    # (query, detector tags on the crop, hint): the rule's tags must overlap the object's.
    CASES = [
        ("person", ["person"], {"objectType": "person", "confidence": 90}),
        ("people", [], {"objectType": "person", "confidence": 90}),
        ("a man in a red shirt", ["person"], {"objectType": "person", "confidence": 90}),
        ("kids", [], {"objectType": "person", "confidence": 70}),
        ("car", ["car"], {"objectType": "vehicle", "confidence": 80}),
        ("red car", [], {"objectType": "vehicle", "confidence": 80}),
        ("suv", ["car"], {"objectType": "vehicle", "confidence": 80}),
        ("someone on a bike", ["bicycle"], {"objectType": "vehicle", "confidence": 80}),
        ("truck", [], {"objectType": "vehicle", "confidence": 80,
                       "attributes": {"vehicleType": {"val": "truck", "confidence": 70}}}),
        ("dog", ["dog"], {"objectType": "animal", "confidence": 80}),
        ("deer in the yard", [], {"objectType": "animal", "confidence": 80}),
        ("vehicle", ["truck"], {"objectType": "vehicle", "confidence": 80}),
    ]

    def test_query_tags_overlap_object_tags(self):
        for query, det, hint in self.CASES:
            with self.subTest(query=query):
                rule = _rule_tags(nl_search.parse(query))
                obj = _object_tags(det, hint)
                self.assertTrue(rule, query)
                self.assertTrue(rule <= nl_search.vocabulary(), rule)
                self.assertTrue(rule & obj, (query, rule, obj))

    def test_object_words_map_both_ways(self):
        """Every object word gets a tag the tagger emits for its type, and every tag the
        tagger can emit for a type is reachable from that type's umbrella."""
        for obj_type, words in nl_search.OBJECT_WORDS.items():
            emitted = nl_search._detectable(obj_type) | {ram_tagger.UMBRELLA[obj_type]}
            for w in words:
                with self.subTest(word=w):
                    self.assertTrue(_rule_tags(nl_search.parse(w)) & emitted, w)
            for det in nl_search._detectable(obj_type):
                tags = _object_tags([det], {"objectType": obj_type, "confidence": 80})
                self.assertIn(ram_tagger.UMBRELLA[obj_type], tags)

    def test_exact_car_does_not_widen_to_vehicle(self):
        reply = nl_search.parse("car")
        self.assertTrue(reply["exact_match"])
        self.assertEqual(_rule_tags(reply), {"car"})

    def test_exact_man_keeps_person(self):
        reply = nl_search.parse("man")
        self.assertTrue(reply["exact_match"])
        self.assertIn("person", _rule_tags(reply))

    def test_bike_attribute_becomes_bicycle(self):
        tags = _object_tags([], {"objectType": "vehicle", "confidence": 80,
                                 "attributes": {"vehicleType": {"val": "bike", "confidence": 70},
                                                "color": {"val": "Grey", "confidence": 60}}})
        self.assertTrue({"bicycle", "gray", "vehicle"} <= tags, tags)

    def test_no_tags_for_nonsense(self):
        self.assertEqual(nl_search.parse("blah blah")["keyTags"], [])


class TestProtectScore(unittest.TestCase):
    def test_formula_on_known_pair(self):
        # cos 0.40 -> 100 %, cos 0.08 -> 0 %, cos 0.24 -> 50 % (M/718745:45-48)
        q = [1.0, 0.0]
        for cos, pct in ((0.40, 100), (0.08, 0), (0.24, 50), (0.20, 37.5)):
            o = [cos, float(np.sqrt(1 - cos * cos))]
            self.assertAlmostEqual(clip_embed.protect_similarity(o, q), pct, places=6)

    def test_blend_lifts_matching_query_score(self):
        rng = np.random.default_rng(7)
        text = rng.standard_normal(768)
        text /= np.linalg.norm(text)
        noise = rng.standard_normal(768)
        noise -= (noise @ text) * text
        noise /= np.linalg.norm(noise)
        image = 0.2 * text + np.sqrt(1 - 0.04) * noise  # stock-CLIP crop: cos 0.2 = 37.5 %
        self.assertAlmostEqual(clip_embed.protect_similarity(image, text), 37.5, places=4)
        with mock.patch.object(clip_embed, "text_embed", return_value=text.tolist()):
            off = clip_embed.blend(image.tolist(), ["car"], weight=0)
            on = clip_embed.blend(image.tolist(), ["car"], weight=0.1)
            with mock.patch.dict(os.environ, {"DIYKEY_CLIP_TAG_BLEND": "0"}):
                env_off = clip_embed.blend(image.tolist(), ["car"])
            with mock.patch.dict(os.environ, {"DIYKEY_CLIP_TAG_BLEND": "bogus"}):
                env_bad = clip_embed.blend(image.tolist(), ["car"])
        self.assertEqual(off, image.tolist())
        self.assertEqual(env_off, image.tolist())
        lifted = clip_embed.protect_similarity(on, text)
        self.assertGreater(lifted, 50)   # clears the UI's default 50 % threshold
        self.assertLess(lifted, 70)
        self.assertAlmostEqual(clip_embed.protect_similarity(env_bad, text), lifted, places=4)
        # a query unrelated to both the crop and its tags doesn't move
        other = rng.standard_normal(768)
        for v in (text, noise):
            other -= (other @ v) * v
        other /= np.linalg.norm(other)
        self.assertLess(abs(clip_embed.protect_similarity(on, other)
                            - clip_embed.protect_similarity(image, other)), 5)


class TestAdvancedAlarmGaps(unittest.TestCase):
    def test_all_set(self):
        env = {"DIYKEY_CLIP": "1", "DIYKEY_RAM_MODEL": "/m.onnx", "DIYKEY_RAM_TAGS": "/t.csv"}
        self.assertEqual(ucp4_client.advanced_alarm_gaps(env), [])

    def test_names_missing(self):
        self.assertEqual(ucp4_client.advanced_alarm_gaps({"DIYKEY_NL_SEARCH": "0"}),
                         ["DIYKEY_CLIP=1", "DIYKEY_RAM_MODEL", "DIYKEY_RAM_TAGS",
                          "DIYKEY_NL_SEARCH (set to 0)"])


if __name__ == "__main__":
    unittest.main()


class TestCanonicalWords(_VocabCase):
    """Plural fallback and the firmware word linker (DIYKEY_WORD_LINKER)."""

    def setUp(self):
        super().setUp()
        nl_search._linker = None
        self.vocab = {"box", "tree", "puppy", "leaf", "truck", "jeep", "person", "car", "bus"}

    def tearDown(self):
        nl_search._linker = None
        super().tearDown()

    def test_regular_plurals(self):
        c = nl_search._canonical
        self.assertEqual(c("boxes", self.vocab), "box")
        self.assertEqual(c("trees", self.vocab), "tree")
        self.assertEqual(c("puppies", self.vocab), "puppy")
        self.assertEqual(c("leaves", self.vocab), "leaf")

    def test_aliases_and_vocab_words_win(self):
        self.assertEqual(nl_search._canonical("people", self.vocab), "person")
        self.assertEqual(nl_search._canonical("bus", self.vocab), "bus")  # not "bu"

    def test_unknown_word_unchanged(self):
        self.assertEqual(nl_search._canonical("rogers", self.vocab), "rogers")

    def test_word_linker_strong_links_only(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('{"lorry": {"truck": 1.0}, "wagon": {"jeep": 0.4},'
                    ' "auto": {"car": 0.8, "jeep": 0.9}}')
        self.addCleanup(os.unlink, f.name)
        with mock.patch.dict(os.environ, {"DIYKEY_WORD_LINKER": f.name}):
            nl_search._linker = None
            self.assertEqual(nl_search._canonical("lorry", self.vocab), "truck")
            self.assertEqual(nl_search._canonical("wagon", self.vocab), "wagon")  # 0.4 < 0.8
            self.assertEqual(nl_search._canonical("auto", self.vocab), "jeep")  # strongest

    def test_missing_linker_is_empty(self):
        with mock.patch.dict(os.environ, {"DIYKEY_WORD_LINKER": "/nonexistent/linker.json"}):
            nl_search._linker = None
            self.assertEqual(nl_search.word_linker(), {})

    def test_parse_tags_a_plural(self):
        nl_search._vocab = set(nl_search.FALLBACK_VOCAB) | {"tree"}
        tags = [t for k in nl_search.parse("trees")["keyTags"] for t in k["tags"]]
        self.assertIn("tree", tags)
