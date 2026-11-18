# SPDX-License-Identifier: AGPL-3.0-or-later
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3_wss"))
import e5_embed  # noqa: E402


class TestE5Embed(unittest.TestCase):
    def test_wants_e5_with_or_without_version(self):
        self.assertTrue(e5_embed.wants("multilingual-e5-small"))
        self.assertTrue(e5_embed.wants("multilingual-e5-small@2"))
        self.assertFalse(e5_embed.wants("clip-ViT-L-14"))
        self.assertFalse(e5_embed.wants(None))

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {"DIYKEY_E5": ""}), \
                mock.patch.object(e5_embed, "_load") as load:
            self.assertIsNone(e5_embed.query_embed("bird"))
        load.assert_not_called()

    def test_prefixes_mean_pool_and_unit_length(self):
        import torch
        tok = mock.Mock(return_value={"input_ids": torch.zeros((1, 2), dtype=torch.long),
                                      "attention_mask": torch.tensor([[1, 0]])})
        hidden = torch.zeros((1, 2, 384))
        hidden[0, 0, 0] = 3.0
        hidden[0, 1, 1] = 9.0  # padding: must not count
        mdl = mock.Mock(return_value=mock.Mock(last_hidden_state=hidden))
        with mock.patch.dict(os.environ, {"DIYKEY_E5": "1"}), \
                mock.patch.object(e5_embed, "_load", return_value=(tok, mdl)):
            q = e5_embed.query_embed(" a bird ")
            p = e5_embed.passage_embed("A bird stands.")
        self.assertEqual(tok.call_args_list[0].args[0], ["query: a bird"])
        self.assertEqual(tok.call_args_list[1].args[0], ["passage: A bird stands."])
        self.assertEqual(len(q), 384)
        self.assertAlmostEqual(q[0], 1.0)
        self.assertEqual(q[1], 0.0)
        self.assertEqual(p, q)


if __name__ == "__main__":
    unittest.main()
