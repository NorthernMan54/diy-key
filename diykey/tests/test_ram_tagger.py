# SPDX-License-Identifier: AGPL-3.0-or-later
"""build_result: tagObjs from roiMeta (SW:2737-2760), and never searchSnapshots."""
import os
import sys
import tempfile
import unittest
import unittest.mock

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ram_tagger  # noqa: E402
from protect_labels import convert_ram_tags_to_labels  # noqa: E402


class FakeTagger:
    def tag_frame(self, img):
        return [{"tag": "person", "confScore": 0.9}]


class RamImagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "f.jpg")
        cv2.imwrite(self.path, np.full((200, 400, 3), 120, np.uint8))

    def test_tagobjs_without_snapshots(self):
        roi = [{"coord": [100, 100, 300, 600], "objectType": "person", "trackerId": 5},
               {"coord": [0, 0, 0, 0], "objectType": "vehicle", "trackerId": 9}]
        meta = [{"ts": 1000, "roi": roi}, {"ts": 2000, "roi": roi[0]}]
        res = ram_tagger.build_result(FakeTagger(), "cam", "ev", {1000: self.path}, roi_meta=meta)
        km, = res["keyMomentsTags"]
        self.assertEqual(km["tagObjs"], roi)
        # Infer agent 0.0.24.4 never sets it; Protect makes a new detection per entry (live
        # 07:55 Backyard duplicate person, 16:18 Doorbell three tiles of a parked car).
        self.assertNotIn("searchSnapshots", km)

    def test_tracked_objects_use_thumbnail_tags(self):
        roi = {"coord": [100, 100, 300, 600], "objectType": "person", "trackerId": 5}
        objects = [{"trackerID": 5, "keyMomentMs": 1000, "img": cv2.imread(self.path), "hint": roi}]
        res = ram_tagger.build_result(FakeTagger(), "cam", "ev", {1000: self.path},
                                      objects=objects, roi_meta=[{"ts": 1000, "roi": [roi]}])
        self.assertEqual([t["trackerID"] for t in res["thumbnailTags"]], [5])

    def test_profile_counts_frames_crops_and_models(self):
        roi = {"coord": [100, 100, 300, 600], "objectType": "person", "trackerId": 5}
        objects = [{"trackerID": 5, "keyMomentMs": 1000, "img": cv2.imread(self.path), "hint": roi}]
        ram_tagger.prof_reset()
        ram_tagger.build_result(FakeTagger(), "cam", "ev", {1000: self.path}, objects=objects)
        stats = ram_tagger.prof_take()
        self.assertEqual({k: v[1] for k, v in stats.items()},
                         {"ramframes": 1, "crops": 1, "clipemb": 1, "reid": 1})
        self.assertRegex(ram_tagger.prof_format(stats),
                         r"^ramframes=1 crops=1 clipemb=\d+/1 reid=\d+/1$")

    def test_profile_off_outside_a_task(self):
        ram_tagger.prof_take()
        ram_tagger.build_result(FakeTagger(), "cam", "ev", {1000: self.path})
        self.assertEqual(ram_tagger.prof_take(), {})

    def test_ram_plus_call_is_timed_even_when_disabled(self):
        plus = ram_tagger.RamPlus("/nonexistent/ram_plus.onnx")
        plus._failed = True
        ram_tagger.prof_reset()
        self.assertEqual(plus.tags(cv2.imread(self.path)), {})
        self.assertEqual(ram_tagger.prof_take()["ramplus"][1], 1)

    def test_colour_goes_on_the_vehicle(self):
        tags = lambda *n: [{"tag": t, "confScore": 0.9} for t in n]
        # Live 2026-11-10 08:44 Driveway gave "Red road, car, drive, ...".
        self.assertEqual(ram_tagger.describe(tags("road", "red", "car", "curb")),
                         "Road, red car and curb detected.")
        # 08:43: scene memory dropped the parked car, leaving "Red woman, road, ...".
        self.assertEqual(ram_tagger.describe(tags("red", "woman", "road")),
                         "Woman and road detected.")
        self.assertEqual(ram_tagger.describe(tags("white", "pickup truck")),
                         "White pickup truck detected.")

    def test_build_result_without_roi(self):
        res = ram_tagger.build_result(FakeTagger(), "cam", "ev", {1000: self.path})
        self.assertIsInstance(res, dict)
        self.assertNotIn("tagObjs", res["keyMomentsTags"][0])


class FakeSession:
    """Stands in for the RAM++ onnxruntime session: fixed probabilities over 5 tags."""
    def __init__(self, probs):
        self.probs, self.seen = np.array(probs, np.float32), []

    def get_inputs(self):
        return [type("I", (), {"name": "image"})()]

    def run(self, _, feed):
        self.seen.append(feed["image"].shape)
        return [self.probs[None]]


def make_plus(tmp, probs, vocab, names=None, thresholds=None, **kw):
    plus = ram_tagger.RamPlus(os.path.join(tmp, "ram_plus.onnx"), vocab=vocab, **kw)
    plus._session = FakeSession(probs)
    plus.names = names or ["man", "sofa", "shirt", "sedan", "glasses"]
    plus.thresholds = np.array(thresholds or [0.7, 0.7, 0.5, 0.6, 0.9], np.float32)
    plus.keep = np.array([n in vocab for n in plus.names])
    return plus


class RamPlusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_per_class_thresholds_and_vocab(self):
        # sofa passes its threshold but is not an enabled Protect tag; glasses (0.8 < 0.9) and
        # sedan (0.55 < 0.6) fall short of their own thresholds; shirt passes at 0.55 because its threshold is 0.5.
        plus = make_plus(self.tmp.name, [0.8, 0.9, 0.55, 0.55, 0.8],
                         {"man", "shirt", "sedan", "glasses"})
        tags = plus.tags(np.zeros((50, 30, 3), np.uint8))
        self.assertEqual(list(tags), ["man", "shirt"])
        self.assertEqual(plus._session.seen, [(1, 3, 384, 384)])

    def test_max_tags_keeps_best(self):
        plus = make_plus(self.tmp.name, [0.8, 0.9, 0.95, 0.99, 0.97],
                         {"man", "shirt", "sedan", "glasses"}, max_tags=2)
        self.assertEqual(list(plus.tags(np.zeros((8, 8, 3), np.uint8))), ["sedan", "glasses"])

    def test_failure_disables_and_returns_empty(self):
        plus = ram_tagger.RamPlus(os.path.join(self.tmp.name, "missing.onnx"))
        self.assertEqual(plus.tags(np.zeros((8, 8, 3), np.uint8)), {})
        self.assertTrue(plus._failed)

    def test_filter_by_object_type(self):
        tags = {"man": 0.9, "sedan": 0.8, "shirt": 0.7}
        self.assertEqual(set(ram_tagger.plus_filter(tags, "person")), {"man", "shirt"})
        self.assertEqual(set(ram_tagger.plus_filter(tags, "vehicle")), {"sedan", "shirt"})
        self.assertEqual(set(ram_tagger.plus_filter(tags, "animal")), {"shirt"})
        self.assertEqual(set(ram_tagger.plus_filter(tags, None)), set(tags))

    def test_crop_keeps_only_label_tags(self):
        tags = {"man": 0.9, "oven": 0.86, "food": 0.8, "black shirt": 0.7, "glasses": 0.7,
                "person": 0.8, "tray": 0.6}
        self.assertEqual(set(ram_tagger.plus_filter(tags, "person")),
                         {"man", "black shirt", "glasses", "person"})
        self.assertEqual(set(ram_tagger.plus_filter(tags, None)), set(tags))  # whole frame
        for t in ("man", "oven", "blue sedan", "bus", "hat", "tray", "short", "grey coat"):
            self.assertEqual(ram_tagger.label_tag(t),
                             bool(convert_ram_tags_to_labels("vehicle", [t]) or
                                  convert_ram_tags_to_labels("person", [t])), t)

    def test_unset_keeps_current_behaviour(self):
        os.environ.pop("DIYKEY_RAM_PLUS_MODEL", None)
        self.assertIsNone(getattr(FakeTagger(), "plus", None))
        res = ram_tagger.tag_crop(FakeTagger(), np.zeros((20, 20, 3), np.uint8),
                                  {"objectType": "person", "confidence": 90})
        self.assertEqual([t["tag"] for t in res], ["person"])

    def test_special_vehicles_survive_only_on_a_vehicle_crop(self):
        tags = {"police car": 0.95, "school bus": 1.0, "ambulance": 0.7, "fire truck": 0.7,
                "oven": 0.9}
        self.assertEqual(set(ram_tagger.plus_filter(tags, "vehicle")), {"school bus"})
        self.assertEqual(set(ram_tagger.plus_filter({"ambulance": 0.66, "sedan": 0.8}, "vehicle")),
                         {"ambulance", "sedan"})
        self.assertEqual(set(ram_tagger.plus_filter({"tow truck": 0.8}, "vehicle")), {"tow truck"})
        self.assertEqual(ram_tagger.plus_filter(tags, "person"), {})
        self.assertEqual(ram_tagger.plus_filter(tags, "animal"), {})


class PlusTagger(FakeTagger):
    def __init__(self, plus):
        self.plus = plus

    def tag_frame(self, img, plus=True):
        out = [{"tag": "person", "confScore": 0.9}, {"tag": "car", "confScore": 0.8}]
        if plus:
            out.append({"tag": "sofa", "confScore": 0.6})
        return out


class PlusMergeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.plus = make_plus(self.tmp.name, [0.8, 0.9, 0.9, 0.9, 0.95],
                              {"man", "shirt", "sedan", "glasses"})
        self.img = np.full((60, 40, 3), 90, np.uint8)

    def test_person_crop_merges_plus_tags(self):
        tags = ram_tagger.tag_crop(PlusTagger(self.plus), self.img,
                                   {"objectType": "person", "confidence": 90})
        got = {t["tag"]: t["confScore"] for t in tags}
        self.assertEqual(set(got), {"person", "man", "shirt", "glasses"})  # no sedan, no car
        self.assertEqual(got["glasses"], 0.95)
        self.assertEqual([t["tag"] for t in tags][0], "glasses")  # sorted by score

    def test_vehicle_crop_drops_person_tags_and_defers_to_camera(self):
        tagger = PlusTagger(self.plus)
        tags = ram_tagger.tag_crop(tagger, self.img, {"objectType": "vehicle"})
        self.assertEqual({t["tag"] for t in tags}, {"car", "sedan", "shirt", "glasses"})
        hint = {"objectType": "vehicle",
                "attributes": {"vehicleType": {"val": "suv", "confidence": 80}}}
        tags = ram_tagger.tag_crop(tagger, self.img, hint)
        self.assertEqual({t["tag"] for t in tags}, {"suv", "shirt", "glasses"})

    def test_whole_frame_and_thumbnails_get_labels(self):
        path = os.path.join(self.tmp.name, "f.jpg")
        cv2.imwrite(path, self.img)
        tagger = PlusTagger(self.plus)
        tagger.plus = self.plus
        # The whole-frame path: PlusTagger.tag_frame(plus=True) stands for RamTagger's merge.
        res = ram_tagger.build_result(tagger, "cam", "ev", {1000: path},
                                      objects=[{"trackerID": 3, "keyMomentMs": 1000,
                                                "img": self.img,
                                                "hint": {"objectType": "person", "confidence": 90}}])
        thumb, = res["thumbnailTags"]
        names = [t["tag"] for t in thumb["tags"]]
        self.assertEqual(sorted(convert_ram_tags_to_labels("person", names)),
                         ["accessory:glasses", "clothingType:shirt", "gender:male"])
        self.assertIn("sofa", [t["tag"] for t in res["keyMomentsTags"][0]["tags"]])

    def test_vehicle_umbrella_stays_when_the_camera_typed_it(self):
        # RAM++ sees no vehicle word here (sedan is below its threshold), COCO only person/car is
        # dropped for the camera's type: the camera's own classification still backs "vehicle".
        plus = make_plus(self.tmp.name, [0.8, 0.0, 0.9, 0.1, 0.0], {"man", "shirt", "sedan"})
        hint = {"objectType": "vehicle", "confidence": 70,
                "attributes": {"vehicleType": {"val": "suv", "confidence": 80}}}
        thumb, = ram_tagger.thumbnail_tags(PlusTagger(plus), [{"trackerID": 4, "keyMomentMs": 1,
                                                               "img": self.img, "hint": hint}])
        self.assertIn("vehicle", [t["tag"] for t in thumb["tags"]])

    def test_labels_port(self):
        self.assertEqual(convert_ram_tags_to_labels("vehicle", ["blue sedan", "bus"]),
                         ["vehicleType:car", "color:blue", "vehicleType:bus"])
        self.assertEqual(convert_ram_tags_to_labels("person", ["black shirt", "woman", "glass"]),
                         ["upperClothingColor:black", "clothingType:shirt", "gender:female",
                          "accessory:glasses"])


class CocoTagger:
    """COCO stand-in returning fixed labels; RAM++ is whatever plus is (None = not loaded)."""
    def __init__(self, coco, plus=None):
        self.coco, self.plus = coco, plus

    def tag_frame(self, img, plus=True):
        return [{"tag": t, "confScore": 0.7} for t in self.coco]


class ClaimedTypeTest(unittest.TestCase):
    """Live 2026-11-09 18:35:45 Driveway: a crack diy-port called "animal" went out tagged
    animal 0.82 though RAM++ saw crack/pavement and no animal on the crop."""
    NAMES = ["crack", "pavement", "dog", "man", "night"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = unittest.mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("DIYKEY_TRUST_CLAIMED_TYPE", None)
        os.environ.pop("DIYKEY_CLIP", None)
        os.environ.pop("DIYKEY_REID_MODEL", None)
        self.img = np.full((61, 67, 3), 110, np.uint8)
        self.hint = {"objectType": "animal", "confidence": 82, "trackerId": 2818}

    def plus(self, probs):
        return make_plus(self.tmp.name, probs, set(self.NAMES), names=self.NAMES,
                         thresholds=[0.5] * 5)

    def thumb(self, tagger, hint=None, frame_tags=None):
        thumb, = ram_tagger.thumbnail_tags(tagger, [{"trackerID": 2818, "keyMomentMs": 1,
                                                     "img": self.img, "hint": hint or self.hint}],
                                           frame_tags)
        return [t["tag"] for t in thumb["tags"]]

    def test_crack_called_animal_is_not_tagged_animal(self):
        tagger = CocoTagger([], self.plus([0.88, 0.85, 0.1, 0.1, 0.1]))
        self.assertNotIn("animal", [t["tag"] for t in ram_tagger.tag_crop(tagger, self.img, self.hint)])
        self.assertNotIn("animal", self.thumb(tagger))

    def test_ram_plus_animal_word_keeps_it_when_the_frame_agrees(self):
        tagger = CocoTagger([], self.plus([0.88, 0.1, 0.9, 0.1, 0.1]))
        self.assertIn("animal", self.thumb(tagger, frame_tags={"dog", "night"}))
        # The crack's 0 s frame: RAM++ saw a "bird" on the crop that the whole frame never named.
        self.assertNotIn("animal", self.thumb(tagger, frame_tags={"crack", "pavement"}))

    def test_coco_family_label_keeps_it(self):
        tagger = CocoTagger(["dog"], self.plus([0.88, 0.85, 0.1, 0.1, 0.1]))
        tags = self.thumb(tagger)
        self.assertIn("dog", tags)
        self.assertIn("animal", tags)

    def test_coco_label_of_another_family_does_not(self):
        tagger = CocoTagger(["car"], self.plus([0.88, 0.85, 0.1, 0.1, 0.1]))
        self.assertNotIn("animal", self.thumb(tagger))

    def test_person_needs_a_person_word(self):
        hint = {"objectType": "person", "confidence": 75}
        self.assertNotIn("person", self.thumb(CocoTagger([], self.plus([0.9, 0.9, 0, 0, 0])), hint))
        self.assertIn("person", self.thumb(CocoTagger([], self.plus([0.9, 0, 0, 0.8, 0])), hint))

    def test_without_ram_plus_protect_is_trusted(self):
        self.assertIn("animal", self.thumb(CocoTagger([])))

    def test_failed_ram_plus_is_no_second_opinion(self):
        plus = ram_tagger.RamPlus(os.path.join(self.tmp.name, "missing.onnx"))
        self.assertIn("animal", self.thumb(CocoTagger([], plus)))
        self.assertTrue(plus._failed)

    def test_env_restores_old_behaviour(self):
        os.environ["DIYKEY_TRUST_CLAIMED_TYPE"] = "1"
        self.assertIn("animal", self.thumb(CocoTagger([], self.plus([0.88, 0.85, 0.1, 0.1, 0.1]))))

    def test_image_task_crop_follows_the_same_rule(self):
        path = os.path.join(self.tmp.name, "crop.jpg")
        cv2.imwrite(path, self.img)
        tagger = CocoTagger([], self.plus([0.88, 0.85, 0.1, 0.1, 0.1]))
        res = ram_tagger.build_result(tagger, "cam", "ev", {1000: path}, hints={1000: self.hint})
        self.assertNotIn("animal", [t["tag"] for t in res["keyMomentsTags"][0]["tags"]])

    def test_types_without_a_word_list_are_trusted(self):
        self.assertTrue(ram_tagger.claimed_type_supported("package", [], {}, plus_ran=True))


class SpeciesTest(unittest.TestCase):
    """Live 2026-11-09 18:27:07 Doorbell: summary "A cat walking through a gate", whole
    frame cat 0.90 / dog 0.95, but the object went out animal/night/bear 0.62, so a search for
    "cat" found nothing."""
    NAMES = ["cat", "dog", "bird", "fence", "man"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = unittest.mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for k in ("DIYKEY_TRUST_CLAIMED_TYPE", "DIYKEY_CLIP", "DIYKEY_REID_MODEL",
                  "DIYKEY_SPECIES_MIN_PX"):
            os.environ.pop(k, None)
        self.small = np.full((110, 400, 3), 100, np.uint8)
        self.hint = {"objectType": "animal", "confidence": 91, "trackerId": 74537}

    def tags(self, coco, probs, frame_tags, img=None):
        plus = make_plus(self.tmp.name, probs, set(self.NAMES), names=self.NAMES,
                         thresholds=[0.5] * 5)
        tags, _ = ram_tagger.crop_tags(CocoTagger(coco, plus),
                                       self.small if img is None else img, self.hint, frame_tags)
        return {t["tag"]: t["confScore"] for t in tags}

    def test_doorbell_cat_replaces_coco_bear(self):
        got = self.tags(["bear"], [0.8, 0.1, 0.1, 0.7, 0.1], {"cat", "dog", "fence", "night"})
        self.assertEqual(got.get("cat"), 0.8)
        self.assertNotIn("bear", got)
        self.assertIn("animal", got)
        self.assertNotIn("fence", got)  # still not a label tag on a crop

    def test_one_species_per_animal(self):
        got = self.tags([], [0.8, 0.9, 0.1, 0.1, 0.1], {"cat", "dog"})
        self.assertIn("dog", got)
        self.assertNotIn("cat", got)

    def test_species_the_frame_never_named_is_dropped(self):
        got = self.tags(["bear"], [0.1, 0.1, 0.83, 0.1, 0.1], {"crack", "pavement"})
        self.assertNotIn("bird", got)
        self.assertIn("bear", got)  # COCO's family label stays when no species is agreed

    def test_coco_and_ram_plus_agreeing_give_one_tag(self):
        got = self.tags(["dog"], [0.1, 0.9, 0.1, 0.1, 0.1], {"dog"})
        self.assertEqual([t for t in got if t in ram_tagger.SPECIES], ["dog"])

    def test_without_whole_frames_the_crop_must_be_big_enough(self):
        self.assertNotIn("cat", self.tags([], [0.8, 0, 0, 0, 0], None))
        self.assertIn("cat", self.tags([], [0.8, 0, 0, 0, 0], None,
                                       img=np.full((200, 300, 3), 100, np.uint8)))
        os.environ["DIYKEY_SPECIES_MIN_PX"] = "100"
        self.assertIn("cat", self.tags([], [0.8, 0, 0, 0, 0], None))

    def test_species_only_on_animal_crops(self):
        plus = make_plus(self.tmp.name, [0.8, 0, 0, 0, 0.9], set(self.NAMES), names=self.NAMES,
                         thresholds=[0.5] * 5)
        tags = ram_tagger.tag_crop(CocoTagger([], plus), self.small,
                                   {"objectType": "person", "confidence": 80}, {"cat"})
        self.assertNotIn("cat", [t["tag"] for t in tags])

    def test_build_result_passes_whole_frame_tags(self):
        path = os.path.join(self.tmp.name, "frame.jpg")
        cv2.imwrite(path, np.full((360, 640, 3), 100, np.uint8))
        plus = make_plus(self.tmp.name, [0.8, 0.1, 0.1, 0.1, 0.1], set(self.NAMES),
                         names=self.NAMES, thresholds=[0.5] * 5)

        class FrameAndCrop(CocoTagger):
            def tag_frame(self, img, plus=True):
                out = [{"tag": "bear", "confScore": 0.62}]
                if plus:
                    out += [{"tag": t, "confScore": s} for t, s in self.plus.tags(img).items()]
                return out

        res = ram_tagger.build_result(FrameAndCrop([], plus), "cam", "ev", {1000: path},
                                      objects=[{"trackerID": 74537, "keyMomentMs": 1000,
                                                "img": self.small, "hint": self.hint}])
        self.assertIn("cat", [t["tag"] for t in res["keyMomentsTags"][0]["tags"]])
        thumb, = res["thumbnailTags"]
        names = [t["tag"] for t in thumb["tags"]]
        self.assertIn("cat", names)
        self.assertNotIn("bear", names)


class ClipVetoTest(unittest.TestCase):
    """Real crops 2026-11-09: the Driveway crack read "crack in the pavement" 0.99 under CLIP in
    every frame; raccoons never above 0.30 for any non-animal label."""
    NAMES = ["cat", "dog", "bird", "fence", "man"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = unittest.mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for k in ("DIYKEY_TRUST_CLAIMED_TYPE", "DIYKEY_REID_MODEL", "DIYKEY_CLIP_VETO"):
            os.environ.pop(k, None)
        for target, value in (("clip_embed.image_embed", lambda img: [1.0, 0.0]),
                              ("clip_embed.blend", lambda vec, tags: vec)):
            p = unittest.mock.patch(target, value)
            p.start()
            self.addCleanup(p.stop)
        self.img = np.full((110, 400, 3), 100, np.uint8)

    def thumb(self, p_not, hint=None, coco=("bear",), probs=(0.8, 0.1, 0.1, 0.1, 0.1)):
        plus = make_plus(self.tmp.name, list(probs), set(self.NAMES), names=self.NAMES,
                         thresholds=[0.5] * 5)
        hint = hint or {"objectType": "animal", "confidence": 91}
        with unittest.mock.patch("ram_tagger.not_animal_prob", lambda vec: p_not):
            thumb, = ram_tagger.thumbnail_tags(CocoTagger(list(coco), plus),
                                               [{"trackerID": 7, "keyMomentMs": 1,
                                                 "img": self.img, "hint": hint}], {"cat"})
        return [t["tag"] for t in thumb["tags"]], thumb

    def test_confident_not_animal_drops_every_animal_tag(self):
        tags, thumb = self.thumb(0.99)
        for t in ("animal", "cat", "bear"):
            self.assertNotIn(t, tags)
        self.assertEqual(thumb["imgEmbed"], [1.0, 0.0])  # the crop still gets a search row

    def test_below_threshold_keeps_them(self):
        tags, _ = self.thumb(0.55)  # raccoon 02:30 -1 s: shadow/stump/stain added to ~0.5
        self.assertIn("animal", tags)
        self.assertIn("cat", tags)

    def test_threshold_from_env_and_zero_turns_it_off(self):
        os.environ["DIYKEY_CLIP_VETO"] = "0.5"
        self.assertNotIn("animal", self.thumb(0.55)[0])
        os.environ["DIYKEY_CLIP_VETO"] = "0"
        self.assertIn("animal", self.thumb(0.99)[0])

    def test_only_claimed_animals_are_checked(self):
        tags, _ = self.thumb(0.99, hint={"objectType": "person", "confidence": 80},
                             probs=(0.1, 0.1, 0.1, 0.1, 0.9))
        self.assertIn("person", tags)

    def test_without_clip_nothing_is_vetoed(self):
        with unittest.mock.patch("clip_embed.image_embed", lambda img: None):
            tags, thumb = self.thumb(0.99)
        self.assertIn("animal", tags)
        self.assertNotIn("imgEmbed", thumb)


class LabelProbsTest(unittest.TestCase):
    def test_softmax_over_cached_label_vectors(self):
        import clip_embed
        axes = {"crack in the pavement": [1.0, 0.0], "cat": [0.0, 1.0]}

        def fake_text(text):
            return next(v for k, v in axes.items() if k in text)

        clip_embed._label_cache.clear()
        self.addCleanup(clip_embed._label_cache.clear)
        with unittest.mock.patch("clip_embed.text_embed", side_effect=fake_text) as te:
            probs = clip_embed.label_probs([0.995, 0.0999], ["crack in the pavement", "cat"])
            clip_embed.label_probs([0.0, 1.0], ["crack in the pavement", "cat"])
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertGreater(probs["crack in the pavement"], 0.99)
        self.assertEqual(te.call_count, 2 * len(clip_embed.LABEL_TEMPLATES))  # cached after one set
        self.assertIsNone(clip_embed.label_probs(None, ["cat"]))


if __name__ == "__main__":
    unittest.main()
