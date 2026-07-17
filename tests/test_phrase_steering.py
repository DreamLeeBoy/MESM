import torch

from phrase_steering.criterion import PhraseCriterion
from phrase_steering.dataset import phrase_collate
from phrase_steering.model import ActionObjectPhraseEncoder, PhraseQuerySteering


def test_phrase_encoder_invalid_rows_are_zero():
    encoder = ActionObjectPhraseEncoder(hidden_dim=8, dropout=0.0)
    words = torch.randn(2, 5, 8)
    words_mask = torch.ones(2, 5, dtype=torch.bool)
    action_mask = torch.tensor([
        [False, True, False, False, False],
        [False, False, False, False, False],
    ])
    object_mask = torch.tensor([
        [False, False, False, True, False],
        [False, False, True, False, False],
    ])
    phrase_valid = torch.tensor([True, True])

    phrase, effective_valid = encoder(
        words,
        words_mask,
        action_mask,
        object_mask,
        phrase_valid,
    )

    assert effective_valid.tolist() == [True, False]
    assert torch.allclose(phrase[1], torch.zeros_like(phrase[1]))
    assert torch.allclose(phrase[0].norm(), torch.tensor(1.0), atol=1e-5)


def test_query_steering_invalid_sample_is_exact_identity():
    steering = PhraseQuerySteering(init_logit=-3.0)
    video = torch.randn(2, 4, 8)
    phrase = torch.randn(2, 8)
    valid = torch.tensor([False, True])
    padding = torch.tensor([
        [False, False, False, False],
        [False, False, True, True],
    ])

    output = steering(video, phrase, valid, padding)

    assert torch.equal(output[0], video[0])
    assert torch.equal(output[1, 2:], video[1, 2:])


def test_query_steering_preserves_frame_norms():
    steering = PhraseQuerySteering(init_logit=-1.0)
    video = torch.randn(3, 6, 16)
    phrase = torch.randn(3, 16)
    valid = torch.ones(3, dtype=torch.bool)

    output = steering(video, phrase, valid)

    assert torch.allclose(
        output.norm(dim=-1),
        video.norm(dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )


def _synthetic_item(num_queries, competitor_local_idx, ann_offset):
    text_length = 4
    video_length = 3
    return {
        "num_clips": num_queries,
        "video_feat": torch.randn(video_length, 6),
        "video_id": f"video-{ann_offset}",
        "duration": 10.0,
        "moment": [[0.0, 1.0] for _ in range(num_queries)],
        "sentence": ["person opens the door" for _ in range(num_queries)],
        "words_id": [torch.tensor([[49406, 1, 2, 49407]]) for _ in range(num_queries)],
        "words_weight": [torch.ones(1, text_length) for _ in range(num_queries)],
        "unknown_mask": [None for _ in range(num_queries)],
        "words_label": [None for _ in range(num_queries)],
        "start_idx": [0 for _ in range(num_queries)],
        "end_idx": [1 for _ in range(num_queries)],
        "clip_mask": [torch.tensor([True, True, False]) for _ in range(num_queries)],
        "pos_idx": [None for _ in range(num_queries)],
        "neg_idx": [None for _ in range(num_queries)],
        "qid": [None for _ in range(num_queries)],
        "ann_idx": list(range(ann_offset, ann_offset + num_queries)),
        "phrase_action_mask": [
            torch.tensor([False, True, False, False])
            for _ in range(num_queries)
        ],
        "phrase_object_mask": [
            torch.tensor([False, False, True, False])
            for _ in range(num_queries)
        ],
        "phrase_valid": [True for _ in range(num_queries)],
        "competitor_local_idx": competitor_local_idx,
    }


def test_phrase_collate_converts_local_competitor_indices():
    first = _synthetic_item(2, [1, 0], 10)
    second = _synthetic_item(3, [2, -1, 0], 20)

    batch = phrase_collate([first, second])

    assert batch["competitor_index"].tolist() == [1, 0, 4, -1, 2]
    assert batch["phrase_action_mask"].shape == (5, 4)
    assert batch["phrase_object_mask"].shape == (5, 4)
    assert batch["phrase_valid"].tolist() == [True] * 5


def test_competitor_loss_uses_only_valid_rows():
    criterion = PhraseCriterion(
        matcher=None,
        weight_dict={"loss_competitor": 1.0},
        losses=["competitor"],
        eos_coef=0.1,
        span_loss_type="l1",
        max_video_l=4,
        rank_coef=12.0,
        use_triplet=False,
        competitor_margin=0.2,
    )

    outputs = {
        "pred_logits": torch.zeros(3, 2, 2, requires_grad=True),
        "phrase_token": torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]),
        "phrase_valid": torch.tensor([True, True, False]),
        "encoded_video_feat": torch.tensor([
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]),
    }
    targets = {
        "clip_mask": torch.ones(3, 2, dtype=torch.bool),
        "competitor_index": torch.tensor([1, 0, -1]),
    }

    losses = criterion.loss_competitor(outputs, targets, indices=None)

    assert losses["competitor_valid_count"].item() == 2
    assert losses["loss_competitor"].item() == 0.0
