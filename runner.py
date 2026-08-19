import math
import os
import pickle
import torch
import logging
from torch.utils.data import DataLoader

from dataset import Vocabulary
from dataset import CharadesDataset, CharadesCGDataset, CharadesCDDataset
from dataset import TACoSDataset, QVHighlightsDataset
from dataset import SplitGatherBatchSampler
from dataset import collate, collate_qvh
from model import GloVe, GloveTextEncoder
from model import CLIPTextEncoder, T2VEncoder, T2VEncoder_TwoMLP, Transformer
from model import PositionEmbeddingSine, TrainablePositionalEncoding
from model import MESM, HungarianMatcher, Criterion
from model import convert_weights

# FW-CDL:
# Forward word-level complementary learning
from utils.word_pos import POS_TO_ID, normalize_pos_word, word_to_pos


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def build_vocab(opt):
    vocab_file = os.path.join(opt.ann_path, "GloVe_tokenized_count.txt")
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
        words_set = set()
        for line in lines:
            word = line.split(' ')[0]
            words_set.add(word)
    vocab = Vocabulary(words_set)
    return vocab


def build_vocab_from_pkl(opt):
    vocab_file = os.path.join(opt.ann_path, "glove.pkl")
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def build_dataloader(opt, vocab=None):
    logger.info("Building dataset...")
    name2dataset = {
        "charades": CharadesDataset,
        "charades-cg": CharadesCGDataset,
        "charades-cd": CharadesCDDataset,
        "tacos": TACoSDataset,
        "qvhighlights": QVHighlightsDataset,
    }
    dataset_config = dict(
        ann_path = opt.ann_path,
        feat_files = opt.feat_files,
        use_tef = opt.use_tef,
        clip_len = opt.clip_len, 
        max_words_l = opt.max_words_l, 
        max_video_l = opt.max_video_l,
        tokenizer_type=opt.tokenizer_type,
        load_vocab_pkl=opt.load_vocab_pkl,
        bpe_path = opt.bpe_path, 
        vocab=vocab,
        normalize_video = opt.normalize_video,
        contra_samples=opt.contra_samples,
        vocab_size=opt.vocab_size,
        max_gather_size=opt.max_gather_size,
    )
    collate_fn = collate
    if opt.dataset_name == "charades":
        val_splits = ["test"]
    elif opt.dataset_name == "charades-cg":
        val_splits = ["novel_composition", "novel_word"]
        # val_splits = ["test_trivial"]
    elif opt.dataset_name == "charades-cd":
        val_splits = ["test_ood"]
    elif opt.dataset_name == "tacos":
        val_splits = ["test"]
    elif opt.dataset_name == "qvhighlights":
        val_splits = ["val"]
        dataset_config["max_windows"] = opt.max_windows
        collate_fn = collate_qvh
        
    if not opt.is_inference:
        train_dataset = name2dataset[opt.dataset_name](
            **dataset_config, recfw=opt.rec_fw, split="train"
        )
        if opt.max_gather_size > 0:
            train_loader = DataLoader(
                train_dataset, collate_fn=collate_fn,
                batch_sampler=SplitGatherBatchSampler(train_dataset, opt.batch_size, shuffle=True),
                num_workers=opt.num_workers, pin_memory=opt.pin_memory
            )
        else:
            train_loader = DataLoader(
                train_dataset, collate_fn=collate_fn, batch_size=opt.batch_size,
                num_workers=opt.num_workers, shuffle=True, pin_memory=opt.pin_memory,
            )
        val_loaders = {}
        for split in val_splits:
            val_dataset = name2dataset[opt.dataset_name](
                **dataset_config, recfw=False, split=split
            )
            val_loader = DataLoader(
                val_dataset, collate_fn=collate_fn, batch_size=opt.eval_batch_size,
                num_workers=opt.num_workers, shuffle=True, pin_memory=opt.pin_memory,
            )
            val_loaders[split] = val_loader
        test_loaders = None

        # vocab_dict = train_dataset.tokenize_id_dict
        # val_vocab_dicts = [val_loaders[split].dataset.tokenize_id_dict for split in val_loaders]
        # for val_vocab_dict in val_vocab_dicts:
        #     for key, value in val_vocab_dict.items():
        #         if key in vocab_dict:
        #             vocab_dict[key] += value
        #         else:
        #             vocab_dict[key] = value
        # vocab_count = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
        # import os
        # with open(os.path.join(opt.ann_path, "CLIP_tokenized_count.txt"), 'w') as f:
        #     for vocab in vocab_count:
        #         f.write(f"{vocab[0]} {vocab[1]}\n")
        
        # vocab_dict = train_dataset.data
        # val_vocab_dicts = [val_loaders[split].dataset.data for split in val_loaders]
        # for val_vocab_dict in val_vocab_dicts:
        #     for key, value in val_vocab_dict.items():
        #         if key in vocab_dict:
        #             vocab_dict[key] += value
        #         else:
        #             vocab_dict[key] = value
        # vocabs = set(vocab_dict.keys())
        # from dataset import Vocabulary
        # vocab = Vocabulary(vocabs)
        # vocab_count = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
        # import os
        # with open(os.path.join(opt.ann_path, "GloVe_tokenized_count.txt"), 'w') as f:
        #     for v in vocab_count:
        #         f.write(f"{v[0]} {vocab.wtoi[v[0]]} {v[1]}\n")
        
    else:
        train_loader = None
        val_loaders = None
        test_loaders = {}

        for split in val_splits:
            test_dataset = name2dataset[opt.dataset_name](
                **dataset_config, recfw=False, split=split
            )
            test_loader = DataLoader(
                test_dataset, collate_fn=collate_fn, batch_size=opt.eval_batch_size,
                num_workers=opt.num_workers, shuffle=True, pin_memory=opt.pin_memory,
            )
            test_loaders[split] = test_loader

    return train_loader, val_loaders, test_loaders


def build_GloVe_text_encoder(glove_path, vocab):
    glove = GloVe(glove_path)
    # vocab = build_vocab(opt)
    glove_text_encoder = GloveTextEncoder(vocab, glove)
    return glove_text_encoder


def build_CLIP_text_encoder(CLIP_text_encoder_path):
    state_dict = torch.load(CLIP_text_encoder_path)
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIPTextEncoder(
        embed_dim,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


def build_enhance_encoder(args):
    if args.share_MLP:
        return T2VEncoder(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.num_recfw_layers,
            normalize_before=args.pre_norm,
            activation='prelu',
        )
    else:
        return T2VEncoder_TwoMLP(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.num_recfw_layers,
            normalize_before=args.pre_norm,
            activation='prelu',
        )


def build_t2v_encoder(args):
    return T2VEncoder(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.t2v_layers,
        normalize_before=args.pre_norm,
        activation='prelu',
    )


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        activation='prelu',
        use_path_encoder=args.use_path_encoder,
        path_input_size=args.max_video_l,
        path_patch_sizes=args.path_patch_sizes,
        path_top_k=args.path_top_k,
        path_noisy_gating=not args.disable_path_noisy_gating,
        path_balance_loss_coef=args.loss_path_balance_coef,
    )


def build_position_encoding(args):
    N_steps = args.hidden_dim
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    # elif args.position_embedding in ('v3', 'learned'):
    #     position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    txt_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=args.max_words_l + 1 if args.rec_ss else args.max_words_l,
            hidden_size=args.hidden_dim, dropout=args.input_dropout)
    return position_embedding, txt_pos_embed


def build_model(args, vocab=None):
    logger.info("Building model...")
    if args.tokenizer_type == "GloVeSimple":
        text_encoder = build_GloVe_text_encoder(args.text_model_path, vocab)
    elif args.tokenizer_type == "CLIP":
        text_encoder = build_CLIP_text_encoder(args.text_model_path)
    elif args.tokenizer_type == "GloVeNLTK":
        if args.load_vocab_pkl:
            text_encoder = None
        else:
            text_encoder = build_GloVe_text_encoder(args.text_model_path, vocab)
    else:
        raise NotImplementedError
    enhance_encoder = build_enhance_encoder(args) # if args.rec_fw else None
    t2v_encoder = build_t2v_encoder(args)
    transformer = build_transformer(args)
    vid_position_embedding, txt_position_embedding = build_position_encoding(args)
    model = MESM(
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
            # num_recfw_layers=args.num_recfw_layers,
            # rec_fw_contra=args.rec_fw_contra,
            vocab_size=args.vocab_size,
            rec_ss=args.rec_ss,
            num_recss_layers=args.num_recss_layers,
    )
    model.to(args.device)
    return model


# FW-CDL:
# Forward word-level complementary learning
def build_fw_cdl_resources(model, tokenizer):
    """Align offline POS labels and frozen text features to FW class IDs."""
    if not model.rec_fw:
        return None, None

    num_classes = model.output_txt_proj[-1].out_features
    id2label = tokenizer.id2label

    if isinstance(model.text_encoder, CLIPTextEncoder):
        frozen_features = model.text_encoder.token_embedding.weight.detach()
        feature_dim = frozen_features.size(-1)
        class_features = torch.zeros(
            num_classes,
            feature_dim,
            device=frozen_features.device,
            dtype=torch.float32,
        )

        # FW-CDL:
        # Forward word-level complementary learning
        def resolve_word_feature(source_id):
            if not isinstance(source_id, int):
                return None, None
            token_surface = tokenizer.decoder.get(source_id, "")
            if not token_surface.endswith("</w>"):
                return None, None
            return tokenizer.decode([source_id]).strip(), frozen_features[source_id]

    elif isinstance(model.text_encoder, GloveTextEncoder):
        frozen_features = model.text_encoder.emb.weight.detach()
        feature_dim = frozen_features.size(-1)
        class_features = torch.zeros(
            num_classes,
            feature_dim,
            device=frozen_features.device,
            dtype=torch.float32,
        )

        # FW-CDL:
        # Forward word-level complementary learning
        def resolve_word_feature(source_id):
            if not isinstance(source_id, int):
                return None, None
            return tokenizer.vocab.itow.get(source_id), frozen_features[source_id]

    else:
        vocab = tokenizer.vocab
        feature_dim = int(torch.as_tensor(vocab["id2vec"][0]).numel())
        class_features = torch.zeros(num_classes, feature_dim, dtype=torch.float32)

        # FW-CDL:
        # Forward word-level complementary learning
        def resolve_word_feature(source_word):
            if not isinstance(source_word, str) or source_word not in vocab["w2id"]:
                return None, None
            feature = vocab["id2vec"][vocab["w2id"][source_word]]
            return source_word, torch.as_tensor(feature, dtype=torch.float32)

    class_pos = torch.full((num_classes,), -1, dtype=torch.long)
    for source_id, class_id in id2label.items():
        if not isinstance(class_id, int) or not 0 <= class_id < num_classes:
            continue
        surface, feature = resolve_word_feature(source_id)
        if surface is None or feature is None:
            continue
        pos_name = word_to_pos.get(normalize_pos_word(surface))
        if pos_name is None:
            continue
        feature = feature.detach().to(device=class_features.device, dtype=torch.float32)
        class_pos[class_id] = POS_TO_ID[pos_name]
        class_features[class_id] = feature

    # FW-CDL:
    # Forward word-level complementary learning
    valid_features = (
        torch.isfinite(class_features).all(dim=-1)
        & class_features.norm(dim=-1).gt(0)
    ).cpu()
    class_pos[~valid_features] = -1
    mapped_classes = int((class_pos >= 0).sum())
    logger.info(
        "FW-CDL loaded offline POS/frozen-feature metadata for %d/%d FW classes",
        mapped_classes,
        num_classes,
    )
    return class_pos, class_features.detach()


def build_matcher(args):
    return HungarianMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, span_loss_type=args.span_loss_type, max_v_l=args.max_video_l,
        multi_clip=args.dataset_name in ["qvhighlights"]
    )


def build_criterion(args, model=None, tokenizer=None):
    logger.info("Building criterion...")
    matcher = build_matcher(args)

    # FW-CDL:
    # Forward word-level complementary learning
    fw_cdl_coef = float(getattr(args, "fw_cdl_coef", 0.1))
    fw_cdl_tau = float(getattr(args, "fw_cdl_tau", 0.07))
    if not math.isfinite(fw_cdl_coef) or fw_cdl_coef < 0:
        raise ValueError("fw_cdl_coef must be finite and non-negative")
    if not math.isfinite(fw_cdl_tau) or fw_cdl_tau <= 0:
        raise ValueError("fw_cdl_tau must be finite and positive")
    fw_cdl_class_pos = None
    fw_cdl_class_features = None
    if args.rec_fw and fw_cdl_coef > 0 and model is not None and tokenizer is not None:
        fw_cdl_class_pos, fw_cdl_class_features = build_fw_cdl_resources(
            model, tokenizer
        )
    elif args.rec_fw and fw_cdl_coef > 0:
        logger.warning(
            "FW-CDL is disabled because frozen class metadata requires both "
            "the model and training tokenizer"
        )

    losses = ['span', 'label', 'saliency']
    weight_dict = {"loss_span": args.loss_span_coef,
                   "loss_giou": args.loss_giou_coef,
                   "loss_label": args.loss_label_coef,
                   "loss_saliency": args.loss_saliency_coef,}

    if args.use_path_encoder and args.loss_path_balance_coef > 0:
        losses.append("path_balance")
        weight_dict["loss_path_balance"] = 1.0
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)
    
    if args.rec_fw:
        losses.append("rec_fw")
        weight_dict["loss_rec_fw"] = args.loss_recfw_coef

        # FW-CDL:
        # Forward word-level complementary learning
        if fw_cdl_coef > 0 and fw_cdl_class_pos is not None:
            losses.append("fw_cdl")
            weight_dict["loss_fw_cdl"] = fw_cdl_coef
    
    if args.rec_ss:
        losses.append("rec_ss")
        weight_dict["loss_rec_ss"] = args.loss_recss_coef

    criterion = Criterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, span_loss_type=args.span_loss_type,
        max_video_l=args.max_video_l,
        rank_coef=args.rank_coef,
        use_triplet=args.use_triplet,
        saliency_margin=args.saliency_margin,
        # recfw_margin=args.recfw_margin,
        multi_clip=args.dataset_name in ["qvhighlights"],
        gamma=args.iou_gamma,
        recss_tau=args.recss_tau,
        # FW-CDL:
        # Forward word-level complementary learning
        fw_cdl_tau=fw_cdl_tau,
        fw_cdl_class_pos=fw_cdl_class_pos,
        fw_cdl_class_features=fw_cdl_class_features,
    )
    criterion.to(args.device)
    return criterion


def build_optimizer(opt, model):
    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop, gamma=opt.gamma)
    return optimizer, lr_scheduler
