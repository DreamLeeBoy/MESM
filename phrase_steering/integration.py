"""Runtime integration that leaves the original MESM-v2 files untouched."""

from __future__ import annotations

import logging

import torch

from .criterion import PhraseCriterion
from .dataset import (
    PhraseCharadesDataset,
    configure_phrase_dataset,
    phrase_collate,
)
from .model import PhraseMESM, PhraseSteeredT2VEncoder


LOGGER = logging.getLogger(__name__)
_INSTALLED = False
_ORIGINAL_BUILD_DATALOADER = None


def _set_default_options(opt) -> None:
    defaults = {
        "semantic_sidecar_dir": None,
        "semantic_sidecar_train_file": "charades_sta_train_semantic_final_v1_2.jsonl",
        "semantic_sidecar_test_file": "charades_sta_test_semantic_final_v1_2.jsonl",
        "require_semantic_sidecar": True,
        "enable_phrase_steering": False,
        "phrase_steering_layer": 0,
        "phrase_steering_init_logit": -3.0,
        "phrase_mlp_dropout": 0.1,
        "loss_competitor_coef": 0.0,
        "competitor_margin": 0.2,
    }
    for key, value in defaults.items():
        if not hasattr(opt, key):
            setattr(opt, key, value)


def _build_dataloader(opt, vocab=None):
    import runner

    _set_default_options(opt)
    configure_phrase_dataset(opt)

    if opt.dataset_name != "charades":
        return _ORIGINAL_BUILD_DATALOADER(opt, vocab)

    original_dataset = runner.CharadesDataset
    original_collate = runner.collate
    runner.CharadesDataset = PhraseCharadesDataset
    runner.collate = phrase_collate
    try:
        return _ORIGINAL_BUILD_DATALOADER(opt, vocab)
    finally:
        runner.CharadesDataset = original_dataset
        runner.collate = original_collate


def _build_model(args, vocab=None):
    import runner

    _set_default_options(args)
    if args.phrase_steering_layer != 0:
        raise ValueError(
            "The v1 implementation supports phrase_steering_layer=0 only."
        )

    if args.tokenizer_type == "GloVeSimple":
        text_encoder = runner.build_GloVe_text_encoder(args.text_model_path, vocab)
    elif args.tokenizer_type == "CLIP":
        text_encoder = runner.build_CLIP_text_encoder(args.text_model_path)
    elif args.tokenizer_type == "GloVeNLTK":
        if args.load_vocab_pkl:
            text_encoder = None
        else:
            text_encoder = runner.build_GloVe_text_encoder(args.text_model_path, vocab)
    else:
        raise NotImplementedError

    enhance_encoder = runner.build_enhance_encoder(args)
    base_t2v_encoder = runner.build_t2v_encoder(args)
    t2v_encoder = PhraseSteeredT2VEncoder(
        encoder=base_t2v_encoder,
        enable_phrase_steering=args.enable_phrase_steering,
        init_logit=args.phrase_steering_init_logit,
    )
    transformer = runner.build_transformer(args)
    vid_position_embedding, txt_position_embedding = runner.build_position_encoding(args)

    enable_competitor_loss = args.loss_competitor_coef > 0
    model = PhraseMESM(
        text_encoder=text_encoder,
        t2v_encoder=t2v_encoder,
        enhance_encoder=enhance_encoder,
        transformer=transformer,
        vid_position_embed=vid_position_embedding,
        txt_position_embed=txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        max_video_l=args.max_video_l,
        max_words_l=args.max_words_l,
        normalize_txt=args.normalize_txt,
        use_txt_pos=args.use_txt_pos,
        span_loss_type=args.span_loss_type,
        n_input_proj=args.n_input_proj,
        rec_fw=args.rec_fw,
        vocab_size=args.vocab_size,
        rec_ss=args.rec_ss,
        num_recss_layers=args.num_recss_layers,
        enable_phrase_steering=args.enable_phrase_steering,
        enable_competitor_loss=enable_competitor_loss,
        phrase_mlp_dropout=args.phrase_mlp_dropout,
    )
    model.to(args.device)
    return model


def _build_criterion(args):
    import runner

    _set_default_options(args)
    matcher = runner.build_matcher(args)
    losses = ["span", "label", "saliency"]
    weight_dict = {
        "loss_span": args.loss_span_coef,
        "loss_giou": args.loss_giou_coef,
        "loss_label": args.loss_label_coef,
        "loss_saliency": args.loss_saliency_coef,
    }

    if args.use_path_encoder and args.loss_path_balance_coef > 0:
        losses.append("path_balance")
        weight_dict["loss_path_balance"] = 1.0

    if args.aux_loss:
        auxiliary_weights = {}
        for layer_index in range(args.dec_layers - 1):
            auxiliary_weights.update({
                f"{key}_{layer_index}": value
                for key, value in weight_dict.items()
                if key != "loss_saliency"
            })
        weight_dict.update(auxiliary_weights)

    if args.rec_fw:
        losses.append("rec_fw")
        weight_dict["loss_rec_fw"] = args.loss_recfw_coef
    if args.rec_ss:
        losses.append("rec_ss")
        weight_dict["loss_rec_ss"] = args.loss_recss_coef
    if args.loss_competitor_coef > 0:
        losses.append("competitor")
        weight_dict["loss_competitor"] = args.loss_competitor_coef

    criterion = PhraseCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        eos_coef=args.eos_coef,
        span_loss_type=args.span_loss_type,
        max_video_l=args.max_video_l,
        rank_coef=args.rank_coef,
        use_triplet=args.use_triplet,
        saliency_margin=args.saliency_margin,
        multi_clip=args.dataset_name in ["qvhighlights"],
        gamma=args.iou_gamma,
        recss_tau=args.recss_tau,
        competitor_margin=args.competitor_margin,
    )
    criterion.to(args.device)
    return criterion


def install_phrase_steering() -> None:
    """Patch runner factories before importing train.py or eval.py."""
    global _INSTALLED, _ORIGINAL_BUILD_DATALOADER
    if _INSTALLED:
        return

    import runner

    _ORIGINAL_BUILD_DATALOADER = runner.build_dataloader
    runner.build_dataloader = _build_dataloader
    runner.build_model = _build_model
    runner.build_criterion = _build_criterion
    _INSTALLED = True
    LOGGER.info("Installed phrase steering v1.2 runtime integration.")
