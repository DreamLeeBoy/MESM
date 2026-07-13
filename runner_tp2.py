"""Builder functions for the TP²-MESM variant.

The original ``runner.py`` remains the source of datasets, text encoders,
MESM alignment modules, optimizer construction, and the original criterion.
"""
from __future__ import annotations

import logging

import runner as mesm_runner
from model.deformable_transformer_1d import DeformableTransformer1D
from model.tp2_model import MESMTP2, TP2Criterion

logger = logging.getLogger(__name__)

# Re-export unchanged official MESM utilities.
build_vocab = mesm_runner.build_vocab
build_vocab_from_pkl = mesm_runner.build_vocab_from_pkl
build_dataloader = mesm_runner.build_dataloader
build_optimizer = mesm_runner.build_optimizer


def build_transformer(args):
    return DeformableTransformer1D(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_queries=args.num_queries,
        num_encoder_layers=getattr(args, "tp2_enc_layers", args.enc_layers),
        num_decoder_layers=getattr(args, "tp2_dec_layers", args.dec_layers),
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_feature_levels=getattr(args, "num_feature_levels", 4),
        enc_n_points=getattr(args, "enc_n_points", 4),
        dec_n_points=getattr(args, "dec_n_points", 4),
        fpn_stem_layers=getattr(args, "fpn_stem_layers", 1),
        fpn_branch_layers=getattr(args, "fpn_branch_layers", 3),
        fpn_downsample_rate=getattr(args, "fpn_downsample_rate", 2),
        fpn_window_size=getattr(args, "fpn_window_size", 9),
    )


def build_model(args, vocab=None):
    logger.info("Building TP²-MESM model from the official MESM components...")
    if args.tokenizer_type == "GloVeSimple":
        text_encoder = mesm_runner.build_GloVe_text_encoder(args.text_model_path, vocab)
    elif args.tokenizer_type == "CLIP":
        text_encoder = mesm_runner.build_CLIP_text_encoder(args.text_model_path)
    elif args.tokenizer_type == "GloVeNLTK":
        if args.load_vocab_pkl:
            text_encoder = None
        else:
            text_encoder = mesm_runner.build_GloVe_text_encoder(args.text_model_path, vocab)
    else:
        raise NotImplementedError

    enhance_encoder = mesm_runner.build_enhance_encoder(args)
    t2v_encoder = mesm_runner.build_t2v_encoder(args)
    transformer = build_transformer(args)
    vid_position_embedding, txt_position_embedding = mesm_runner.build_position_encoding(args)

    model = MESMTP2(
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
        num_feature_levels=getattr(args, "num_feature_levels", 4),
        salient_upsample_type=getattr(args, "salient_upsample_type", "nearest"),
        separate_predict_head=getattr(args, "separate_predict_head", True),
    )
    model.to(args.device)
    return model


def build_criterion(args):
    base_criterion = mesm_runner.build_criterion(args)
    criterion = TP2Criterion(
        base_criterion,
        loss_coef=getattr(args, "loss_tp2_saliency_coef", 3.0),
        focal_alpha=getattr(args, "tp2_focal_alpha", 0.25),
        focal_gamma=getattr(args, "tp2_focal_gamma", 2.0),
        encoder_weight=getattr(args, "tp2_encoder_saliency_weight", 1.5),
        fpn_weight=getattr(args, "tp2_fpn_saliency_weight", 0.5),
    )
    criterion.to(args.device)
    return criterion
