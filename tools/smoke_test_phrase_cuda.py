#!/usr/bin/env python3
"""One-step CUDA forward/backward smoke test for phrase steering.

MESM's original cross-video saliency-negative path requires at least two
video groups. This script deliberately selects two different Charades videos:
one containing a valid Level-1 competitor and one small auxiliary video group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from phrase_steering.integration import install_phrase_steering


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./config/charades/C+SF_C_phrase_steering_v1_2.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2019)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    install_phrase_steering()

    import runner
    from dataset import prepare_batch_input
    from phrase_steering.dataset import phrase_collate

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config.update({
        "num_workers": 0,
        "pin_memory": False,
        "batch_size": 2,
        "eval_batch_size": 1,
        "is_inference": False,
        "device": torch.device(args.device),
    })
    # BaseOptions.parse() normally applies this adjustment.
    if config.get("use_tef", False):
        config["v_feat_dim"] += 2

    opt = SimpleNamespace(**config)
    train_loader, _, _ = runner.build_dataloader(opt, vocab=None)
    dataset = train_loader.dataset

    competitor_candidates = [
        index
        for index, meta in enumerate(dataset.merged_data)
        if any(len(candidate_ids) > 0 for candidate_ids in meta["level1_candidate_line_ids"])
    ]
    if not competitor_candidates:
        raise RuntimeError("No Level-1 competitor video group was found in the training sidecar.")

    # Minimize memory while retaining a real competition pair.
    primary_index = min(
        competitor_candidates,
        key=lambda index: len(dataset.merged_data[index]["video_id"]),
    )
    primary_video_id = dataset.merged_data[primary_index]["video_id"][0]

    secondary_candidates = [
        index
        for index, meta in enumerate(dataset.merged_data)
        if meta["video_id"][0] != primary_video_id
    ]
    secondary_index = min(
        secondary_candidates,
        key=lambda index: len(dataset.merged_data[index]["video_id"]),
    )

    batch = phrase_collate([
        dataset[primary_index],
        dataset[secondary_index],
    ])
    if batch["num_clips"].numel() < 2:
        raise AssertionError("Smoke test must contain at least two video groups.")
    if int((batch["competitor_index"] >= 0).sum()) == 0:
        raise AssertionError("Selected smoke batch has no valid Level-1 competitor.")

    prepare_batch_input(batch, opt.device, non_blocking=False)

    model = runner.build_model(opt, vocab=None)
    criterion = runner.build_criterion(opt)
    model.train()
    criterion.train()

    outputs = model(
        **batch,
        dataset_name=opt.dataset_name,
        is_training=True,
    )
    loss_dict, total_loss = criterion(outputs, batch, is_training=True)

    if not torch.isfinite(total_loss):
        raise FloatingPointError(f"Non-finite total loss: {total_loss}")
    total_loss.backward()

    steering_grad = model.t2v_encoder.phrase_steering.steering_logit.grad
    phrase_parameter = next(model.phrase_encoder.parameters())
    phrase_grad = phrase_parameter.grad

    print("CUDA:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    print("video groups:", batch["num_clips"].tolist())
    print("flattened queries:", int(batch["words_id"].shape[0]))
    print("phrase valid:", int(batch["phrase_valid"].sum()))
    print("competitor valid:", int((batch["competitor_index"] >= 0).sum()))
    print("total loss:", float(total_loss.detach()))

    for key in [
        "loss_span",
        "loss_giou",
        "loss_label",
        "loss_saliency",
        "loss_competitor",
        "competitor_valid_count",
        "competitor_active_ratio",
        "competitor_pos_sim",
        "competitor_neg_sim",
    ]:
        if key in loss_dict:
            print(f"{key}:", float(loss_dict[key].detach()))

    print("steering scale:", float(outputs["phrase_steering_scale"].detach()))
    print(
        "steering gradient:",
        None if steering_grad is None else float(steering_grad.detach()),
    )
    print(
        "phrase MLP gradient norm:",
        None if phrase_grad is None else float(phrase_grad.detach().norm()),
    )

    if steering_grad is None or not torch.isfinite(steering_grad):
        raise AssertionError("Steering scale did not receive a finite gradient.")
    if phrase_grad is None or not torch.isfinite(phrase_grad).all():
        raise AssertionError("Phrase MLP did not receive finite gradients.")

    print("ONE-BATCH CUDA SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
