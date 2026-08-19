# FW-CDL:
# Forward word-level complementary learning
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from model.criterion import Criterion
from model.text_encoder import CLIPTextEncoder, GloveTextEncoder
from runner import build_fw_cdl_resources
from utils.config import BaseOptions
from utils.word_pos import POS_TO_ID, word_to_pos


# FW-CDL:
# Forward word-level complementary learning
def make_criterion(losses=None, weight_dict=None, tau=1.0):
    return Criterion(
        matcher=lambda outputs, targets: torch.zeros(
            outputs["pred_logits"].size(0), 1, dtype=torch.long
        ),
        weight_dict=weight_dict or {},
        losses=losses or ["fw_cdl"],
        eos_coef=0.1,
        span_loss_type="l1",
        max_video_l=75,
        rank_coef=12.0,
        use_triplet=False,
        fw_cdl_tau=tau,
        fw_cdl_class_pos=torch.tensor([
            POS_TO_ID["verb"],
            POS_TO_ID["verb"],
            POS_TO_ID["verb"],
            POS_TO_ID["noun"],
            POS_TO_ID["pronoun"],
            POS_TO_ID["noun"],
        ]),
        fw_cdl_class_features=torch.tensor([
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
        ]),
    )


# FW-CDL:
# Forward word-level complementary learning
class FWCDLLossTest(unittest.TestCase):
    def test_offline_pos_categories_are_independent(self):
        self.assertEqual(word_to_pos["open"], "verb")
        self.assertEqual(word_to_pos["door"], "noun")
        self.assertEqual(word_to_pos["he"], "pronoun")
        self.assertEqual(word_to_pos["arrange"], "verb")
        self.assertEqual(word_to_pos["awaiting"], "verb")
        self.assertEqual(word_to_pos["dispose"], "verb")
        self.assertNotIn("snowy", word_to_pos)
        self.assertNotIn("stainless", word_to_pos)
        self.assertNotEqual(POS_TO_ID["pronoun"], POS_TO_ID["noun"])
        self.assertNotEqual(POS_TO_ID["pronoun"], POS_TO_ID["verb"])

    def test_only_incorrect_masked_words_participate(self):
        criterion = make_criterion()
        logits = torch.full((1, 2, 6), -4.0)
        logits[0, 0, 0] = 4.0
        logits[0, 1, 3] = 4.0
        logits.requires_grad_()
        outputs = {
            "recfw_words_logit": logits,
            "words_mask": torch.tensor([[True, True]]),
            "recfw_masked_words_loc": torch.tensor([[True, False]]),
        }
        targets = {"words_label": torch.tensor([[0, 1]])}

        loss = criterion.loss_fw_cdl(outputs, targets)["loss_fw_cdl"]
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_batch_candidates_match_exact_kl_and_exclude_other_pos(self):
        criterion = make_criterion(tau=1.0)
        logits = torch.full((1, 4, 6), -4.0)
        logits[0, 0, 0] = 0.2
        logits[0, 0, 1] = 0.4
        logits[0, 0, 3] = 2.0
        logits[0, 1, 1] = 3.0
        logits[0, 2, 3] = 3.0
        logits[0, 3, 4] = 3.0
        logits.requires_grad_()
        outputs = {
            "recfw_words_logit": logits,
            "words_mask": torch.ones(1, 4, dtype=torch.bool),
            "recfw_masked_words_loc": torch.ones(1, 4, dtype=torch.bool),
        }
        targets = {"words_label": torch.tensor([[0, 1, 3, 4]])}

        loss = criterion.loss_fw_cdl(outputs, targets)["loss_fw_cdl"]
        similarity = torch.tensor([1.0, 0.8])
        teacher = F.softmax(-(1.0 - similarity), dim=0)
        log_student = F.log_softmax(torch.tensor([0.2, 0.4]), dim=0)
        expected = F.kl_div(log_student, teacher, reduction="sum")
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

        loss.backward()
        gradient = logits.grad[0, 0]
        self.assertGreater(float(gradient[0].abs()), 0.0)
        self.assertGreater(float(gradient[1].abs()), 0.0)
        self.assertTrue(torch.equal(gradient[2:], torch.zeros_like(gradient[2:])))

    def test_global_same_pos_fallback_uses_supported_classes(self):
        criterion = make_criterion(tau=1.0)
        logits = torch.full((1, 1, 6), -4.0)
        logits[0, 0, 0] = 0.2
        logits[0, 0, 1] = 0.4
        logits[0, 0, 2] = 0.6
        logits[0, 0, 3] = 2.0
        logits.requires_grad_()
        outputs = {
            "recfw_words_logit": logits,
            "words_mask": torch.tensor([[True]]),
            "recfw_masked_words_loc": torch.tensor([[True]]),
        }
        targets = {"words_label": torch.tensor([[0]])}

        loss = criterion.loss_fw_cdl(outputs, targets)["loss_fw_cdl"]
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        gradient = logits.grad[0, 0]
        self.assertTrue(torch.all(gradient[:3].abs() > 0))
        self.assertTrue(torch.equal(gradient[3:], torch.zeros_like(gradient[3:])))

    def test_original_fw_loss_is_unchanged(self):
        criterion = make_criterion(losses=["rec_fw"])
        logits = torch.tensor([
            [[1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
             [0.0, 1.0, -1.0, -2.0, -3.0, -4.0]]
        ])
        labels = torch.tensor([[0, 1]])
        mask = torch.tensor([[True, True]])
        outputs = {"recfw_words_logit": logits, "words_mask": mask}
        targets = {"words_label": labels}

        expected, expected_acc = criterion.cal_nll_loss(logits, labels, mask)
        actual = criterion.loss_rec_fw(outputs, targets, None)
        self.assertTrue(torch.allclose(actual["loss_rec_fw"], expected.mean()))
        self.assertTrue(torch.allclose(actual["rec_fw_acc"], expected_acc))

    def test_weight_is_applied_once_and_eval_skips_fw_cdl(self):
        criterion = make_criterion(
            losses=["fw_cdl"], weight_dict={"loss_fw_cdl": 0.1}, tau=1.0
        )
        logits = torch.full((1, 1, 6), -4.0)
        logits[0, 0, 3] = 2.0
        outputs = {
            "pred_logits": torch.zeros(1, 1, 2),
            "recfw_words_logit": logits,
            "words_mask": torch.tensor([[True]]),
            "recfw_masked_words_loc": torch.tensor([[True]]),
        }
        targets = {"words_label": torch.tensor([[0]])}

        loss_dict, total = criterion(outputs, targets, is_training=True)
        self.assertTrue(
            torch.allclose(total, loss_dict["loss_fw_cdl"] * 0.1)
        )
        eval_losses, eval_total = criterion(
            {"pred_logits": torch.zeros(1, 1, 2)}, targets, is_training=False
        )
        self.assertEqual(eval_losses, {})
        self.assertEqual(eval_total, 0)

    def test_parser_defaults_keep_old_configs_compatible(self):
        options = BaseOptions()
        options.initialize()
        self.assertEqual(options.parser.get_default("fw_cdl_tau"), 0.07)
        self.assertEqual(options.parser.get_default("fw_cdl_coef"), 0.1)

    # FW-CDL:
    # Forward word-level complementary learning
    def test_clip_resources_align_source_tokens_to_classifier_classes(self):
        encoder = object.__new__(CLIPTextEncoder)
        nn.Module.__init__(encoder)
        encoder.token_embedding = nn.Embedding.from_pretrained(
            torch.arange(16, dtype=torch.float32).reshape(8, 2), freeze=True
        )
        model = SimpleNamespace(
            rec_fw=True,
            text_encoder=encoder,
            output_txt_proj=[nn.Linear(2, 6)],
        )
        tokenizer = SimpleNamespace(
            id2label={2: 1, 3: 0, 4: 2, "<unknown>": 3},
            decoder={2: "open</w>", 3: "door</w>", 4: "walk"},
            decode=lambda ids: {2: "open ", 3: "door ", 4: "walk"}[ids[0]],
        )

        class_pos, class_features = build_fw_cdl_resources(model, tokenizer)
        self.assertEqual(int(class_pos[0]), POS_TO_ID["noun"])
        self.assertEqual(int(class_pos[1]), POS_TO_ID["verb"])
        self.assertEqual(int(class_pos[2]), -1)  # BPE fragment, not a whole word.
        self.assertTrue(torch.equal(class_features[0], encoder.token_embedding.weight[3]))
        self.assertTrue(torch.equal(class_features[1], encoder.token_embedding.weight[2]))

    # FW-CDL:
    # Forward word-level complementary learning
    def test_glove_resources_align_vocab_ids_to_classifier_classes(self):
        encoder = object.__new__(GloveTextEncoder)
        nn.Module.__init__(encoder)
        encoder.emb = nn.Embedding.from_pretrained(
            torch.arange(10, dtype=torch.float32).reshape(5, 2), freeze=True
        )
        model = SimpleNamespace(
            rec_fw=True,
            text_encoder=encoder,
            output_txt_proj=[nn.Linear(2, 4)],
        )
        tokenizer = SimpleNamespace(
            id2label={1: 2, 2: 0, "<unknown>": 3},
            vocab=SimpleNamespace(itow={1: "run", 2: "door"}),
        )

        class_pos, class_features = build_fw_cdl_resources(model, tokenizer)
        self.assertEqual(int(class_pos[0]), POS_TO_ID["noun"])
        self.assertEqual(int(class_pos[2]), POS_TO_ID["verb"])
        self.assertTrue(torch.equal(class_features[0], encoder.emb.weight[2]))
        self.assertTrue(torch.equal(class_features[2], encoder.emb.weight[1]))

    # FW-CDL:
    # Forward word-level complementary learning
    def test_pickle_resources_follow_tokenizer_class_order(self):
        model = SimpleNamespace(
            rec_fw=True,
            text_encoder=None,
            output_txt_proj=[nn.Linear(2, 3)],
        )
        tokenizer = SimpleNamespace(
            id2label={"he": 1, "run": 0, "<unknown>": 2},
            vocab={
                "w2id": {"he": 0, "run": 1},
                "id2vec": [
                    torch.tensor([1.0, 2.0]),
                    torch.tensor([3.0, 4.0]),
                ],
            },
        )

        class_pos, class_features = build_fw_cdl_resources(model, tokenizer)
        self.assertEqual(int(class_pos[0]), POS_TO_ID["verb"])
        self.assertEqual(int(class_pos[1]), POS_TO_ID["pronoun"])
        self.assertTrue(torch.equal(class_features[0], torch.tensor([3.0, 4.0])))
        self.assertTrue(torch.equal(class_features[1], torch.tensor([1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
