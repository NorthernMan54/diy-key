# SPDX-License-Identifier: AGPL-3.0-or-later
"""ram_tagger.describe_labels: deep-mode describe labels must be key:value, or Protect's
sanitizeSessionLabels (M/594367:333) drops them (Saul 2026-11-16: every bare tag was dropped)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3_wss"))
sys.path.insert(0, os.path.dirname(__file__))
import ram_tagger  # noqa: E402
from protect_labels import convert_ram_tags_to_labels  # noqa: E402


class DescribeLabelsTest(unittest.TestCase):
    def test_person_matches_protect_converter(self):
        tags = ["man", "red shirt", "blue pant", "glass", "backpack", "hat", "woman", "person"]
        self.assertEqual(sorted(ram_tagger.describe_labels(tags, {"person"})),
                         sorted(set(convert_ram_tags_to_labels("person", tags))))

    def test_vehicle_matches_protect_converter(self):
        tags = ["white sedan", "truck", "car"]
        self.assertEqual(sorted(ram_tagger.describe_labels(tags, {"vehicle"})),
                         sorted(set(convert_ram_tags_to_labels("vehicle", tags))))

    def test_every_label_is_key_value(self):
        labels = ram_tagger.describe_labels(["man", "night", "driveway", "black coat"],
                                            {"person"})
        self.assertTrue(labels)
        self.assertTrue(all(":" in x and not x.endswith(":") for x in labels))
        self.assertNotIn("night", labels)

    def test_vehicle_and_species_need_their_family(self):
        self.assertEqual(ram_tagger.describe_labels(["van", "dog"], {"person"}), [])
        self.assertEqual(ram_tagger.describe_labels(["dog", "raccoon"], {"animal"}),
                         ["species:dog", "species:raccoon"])

    def test_bicycle_and_grey(self):
        self.assertEqual(ram_tagger.describe_labels(["grey bicycle"], {"vehicle"}),
                         ["vehicleType:bike", "color:gray"])

    def test_no_duplicates(self):
        self.assertEqual(ram_tagger.describe_labels(["man", "boy"], {"person"}),
                         ["gender:male"])


if __name__ == "__main__":
    unittest.main()
