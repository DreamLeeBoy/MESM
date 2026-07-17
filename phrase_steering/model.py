"""Phrase construction and Query Steering for the original MESM model."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.model import MESM
from model.text_encoder import CLIPTextEncoder, GloveTextEncoder
from utils import inverse_sigmoid, sample_outclass_neg, split_and_pad, split_expand_and_pad


class ActionObjectPhraseEncoder(nn.Module):
    """Compose contextualized action and object BPE features into one token."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _masked_mean(features: Tensor, mask: Tensor) -> Tensor:
        mask_float = mask.to(dtype=features.dtype).unsqueeze(-1)
        numerator = (features * mask_float).sum(dim=1)
        denominator = mask_float.sum(dim=1).clamp_min(1.0)
        return numerator / denominator

    def forward(
        self,
        word_features: Tensor,
        words_mask: Tensor,
        action_mask: Optional[Tensor],
        object_mask: Optional[Tensor],
        phrase_valid: Optional[Tensor],
    ):
        batch_size, text_length, hidden_dim = word_features.shape
        device = word_features.device

        if action_mask is None or object_mask is None or phrase_valid is None:
            return (
                word_features.new_zeros(batch_size, hidden_dim),
                torch.zeros(batch_size, dtype=torch.bool, device=device),
            )

        action_mask = action_mask[:, :text_length].to(device=device, dtype=torch.bool)
        object_mask = object_mask[:, :text_length].to(device=device, dtype=torch.bool)
        words_mask = words_mask[:, :text_length].to(device=device, dtype=torch.bool)
        phrase_valid = phrase_valid.to(device=device, dtype=torch.bool)

        action_mask = action_mask & words_mask
        object_mask = object_mask & words_mask
        effective_valid = phrase_valid & action_mask.any(dim=1) & object_mask.any(dim=1)

        action_feature = self._masked_mean(word_features, action_mask)
        object_feature = self._masked_mean(word_features, object_mask)
        phrase_delta = self.mlp(torch.cat([action_feature, object_feature], dim=-1))
        phrase_token = F.normalize(
            self.norm(action_feature + phrase_delta),
            dim=-1,
            eps=1e-6,
        )
        phrase_token = phrase_token * effective_valid.to(phrase_token.dtype).unsqueeze(-1)
        return phrase_token, effective_valid


class PhraseQuerySteering(nn.Module):
    """Rank-1 target phrase projection over the visual content query."""

    def __init__(self, init_logit: float = -3.0, eps: float = 1e-6):
        super().__init__()
        self.steering_logit = nn.Parameter(torch.tensor(float(init_logit)))
        self.eps = eps

    @property
    def scale(self) -> Tensor:
        return torch.sigmoid(self.steering_logit)

    def forward(
        self,
        video_content: Tensor,
        phrase_token: Optional[Tensor],
        phrase_valid: Optional[Tensor],
        video_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if phrase_token is None or phrase_valid is None:
            return video_content

        phrase_valid = phrase_valid.to(device=video_content.device, dtype=torch.bool)
        if not bool(phrase_valid.any()):
            return video_content

        unit_phrase = F.normalize(
            phrase_token.to(dtype=video_content.dtype),
            dim=-1,
            eps=self.eps,
        )
        coefficient = (video_content * unit_phrase.unsqueeze(1)).sum(dim=-1, keepdim=True)
        projection = coefficient * unit_phrase.unsqueeze(1)
        steered = video_content + self.scale.to(video_content.dtype) * projection

        original_norm = video_content.norm(dim=-1, keepdim=True)
        steered_norm = steered.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        restored = steered * (original_norm / steered_norm)

        gate = phrase_valid[:, None, None]
        if video_padding_mask is not None:
            gate = gate & (~video_padding_mask.to(torch.bool)).unsqueeze(-1)
        return torch.where(gate, restored, video_content)


class PhraseSteeredT2VEncoder(nn.Module):
    """Apply Query Steering once before the original MESM T2V encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        enable_phrase_steering: bool,
        init_logit: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.enable_phrase_steering = bool(enable_phrase_steering)
        self.phrase_steering = PhraseQuerySteering(init_logit=init_logit)
        self.d_model = getattr(encoder, "d_model", None)
        self.nhead = getattr(encoder, "nhead", None)

    def forward(
        self,
        src_txt: Tensor,
        src_vid: Tensor,
        src_txt_mask: Optional[Tensor] = None,
        src_txt_key_padding_mask: Optional[Tensor] = None,
        pos_txt: Optional[Tensor] = None,
        src_vid_mask: Optional[Tensor] = None,
        src_vid_key_padding_mask: Optional[Tensor] = None,
        pos_vid: Optional[Tensor] = None,
        phrase_token: Optional[Tensor] = None,
        phrase_valid: Optional[Tensor] = None,
        **kwargs,
    ):
        if self.enable_phrase_steering:
            src_vid = self.phrase_steering(
                video_content=src_vid,
                phrase_token=phrase_token,
                phrase_valid=phrase_valid,
                video_padding_mask=src_vid_key_padding_mask,
            )
        return self.encoder(
            src_txt=src_txt,
            src_vid=src_vid,
            src_txt_mask=src_txt_mask,
            src_txt_key_padding_mask=src_txt_key_padding_mask,
            pos_txt=pos_txt,
            src_vid_mask=src_vid_mask,
            src_vid_key_padding_mask=src_vid_key_padding_mask,
            pos_vid=pos_vid,
            **kwargs,
        )


class PhraseMESM(MESM):
    """Original MESM plus phrase construction and positive-path steering."""

    def __init__(
        self,
        *args,
        enable_phrase_steering: bool = False,
        enable_competitor_loss: bool = False,
        phrase_mlp_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_phrase_steering = bool(enable_phrase_steering)
        self.enable_competitor_loss = bool(enable_competitor_loss)
        self.phrase_encoder = ActionObjectPhraseEncoder(
            hidden_dim=self.hidden_dim,
            dropout=phrase_mlp_dropout,
        )

    def _encode_phrase(self, projected_words: Tensor, words_mask: Tensor, kwargs):
        if not (self.enable_phrase_steering or self.enable_competitor_loss):
            batch_size = projected_words.shape[0]
            return (
                projected_words.new_zeros(batch_size, self.hidden_dim),
                torch.zeros(
                    batch_size,
                    dtype=torch.bool,
                    device=projected_words.device,
                ),
            )
        return self.phrase_encoder(
            word_features=projected_words,
            words_mask=words_mask,
            action_mask=kwargs.get("phrase_action_mask"),
            object_mask=kwargs.get("phrase_object_mask"),
            phrase_valid=kwargs.get("phrase_valid"),
        )

    def forward(
        self,
        video_feat,
        video_mask,
        words_id,
        words_mask,
        words_weight,
        num_clips,
        **kwargs,
    ):
        if isinstance(self.text_encoder, CLIPTextEncoder):
            words_feat, sentence_feat, words_id, words_mask = self.CLIP_encode_text(
                words_id,
                words_mask,
                device=words_id.device,
            )
        elif isinstance(self.text_encoder, GloveTextEncoder):
            words_feat, sentence_feat = self.GloVe_encode_text(words_id, words_mask)
        elif self.text_encoder is None:
            words_feat, words_mask, sentence_feat = self.post_process_text(words_id)
        else:
            raise NotImplementedError

        batch_size = video_feat.shape[0]
        projed_video_feat = self.input_vid_proj(video_feat)
        projed_words_feat = self.input_txt_proj(words_feat)
        phrase_token, effective_phrase_valid = self._encode_phrase(
            projed_words_feat,
            words_mask,
            kwargs,
        )

        vid_position = self.vid_position_embed(projed_video_feat, video_mask)
        if self.use_txt_pos:
            txt_position = self.txt_position_embed(projed_words_feat)
        else:
            txt_position = torch.zeros_like(projed_words_feat)

        if self.rec_fw:
            enhanced_video_feat = self.enhance_encoder(
                projed_words_feat,
                projed_video_feat,
                src_txt_key_padding_mask=~words_mask,
                pos_txt=txt_position,
                src_vid_key_padding_mask=~video_mask,
                pos_vid=vid_position,
            )
        else:
            enhanced_video_feat = projed_video_feat

        if self.rec_ss:
            if kwargs["dataset_name"] in ["charades", "charades-cg", "charades-cd", "tacos"]:
                batched_vid = video_feat
                batched_vid_mask = video_mask
                batched_vid_position = vid_position
            elif kwargs["dataset_name"] in ["qvhighlights"]:
                video_length = torch.stack(
                    [item.sum() for item in torch.split(video_mask, num_clips.tolist())]
                ).long()
                unpadded_video_feat = video_feat[video_mask]
                batched_vid, batched_vid_mask = split_expand_and_pad(
                    video_length,
                    num_clips,
                    unpadded_video_feat,
                )
                batched_vid_position = self.vid_position_embed(
                    batched_vid,
                    batched_vid_mask,
                )
            else:
                raise NotImplementedError

            batched_sent, batched_sent_mask = split_expand_and_pad(
                num_clips,
                num_clips,
                sentence_feat,
            )
            batched_vid = self.input_vid_proj(batched_vid)
            batched_sent = self.input_txt_proj(batched_sent)
            recon_feat, projed_recon_feat = self.ss_reconstructor(
                batched_vid,
                batched_vid_mask,
                batched_sent,
                batched_sent_mask,
                num_clips,
                batched_vid_position,
            )
            expanded_words_feat = torch.cat(
                [recon_feat.unsqueeze(1), projed_words_feat],
                dim=1,
            )
            recon_mask = torch.ones(
                [batch_size, 1],
                dtype=torch.bool,
                device=recon_feat.device,
            )
            expanded_words_mask = torch.cat([recon_mask, words_mask], dim=1)
        else:
            expanded_words_feat = projed_words_feat
            expanded_words_mask = words_mask

        if self.use_txt_pos:
            expanded_txt_position = self.txt_position_embed(expanded_words_feat)
        else:
            expanded_txt_position = torch.zeros_like(expanded_words_feat)

        encoded_video_feat = self.t2v_encoder(
            expanded_words_feat,
            enhanced_video_feat,
            src_txt_key_padding_mask=~expanded_words_mask,
            pos_txt=expanded_txt_position,
            src_vid_key_padding_mask=~video_mask,
            pos_vid=vid_position,
            phrase_token=phrase_token if self.enable_phrase_steering else None,
            phrase_valid=effective_phrase_valid if self.enable_phrase_steering else None,
        )

        global_token = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(
            batch_size,
            1,
            1,
        )
        global_token_pos = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(
            batch_size,
            1,
            1,
        )
        hs, reference, memory, memory_global = self.transformer(
            encoded_video_feat,
            ~video_mask,
            self.query_embed.weight,
            vid_position,
            global_token,
            global_token_pos,
        )

        outputs_class = self.class_embed(hs)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()

        neg_index = sample_outclass_neg(num_clips)
        neg_expanded_words_feat = expanded_words_feat[neg_index]
        neg_expanded_words_mask = expanded_words_mask[neg_index]
        neg_expanded_txt_position = expanded_txt_position[neg_index]
        if self.rec_ss:
            neg_words_feat = neg_expanded_words_feat[:, 1:, :]
            neg_words_mask = neg_expanded_words_mask[:, 1:]
            neg_txt_position = neg_expanded_txt_position[:, 1:, :]
        else:
            neg_words_feat = neg_expanded_words_feat
            neg_words_mask = neg_expanded_words_mask
            neg_txt_position = neg_expanded_txt_position

        neg_vid_position = vid_position.clone()
        if self.rec_fw:
            neg_enhanced_video_feat = self.enhance_encoder(
                neg_words_feat,
                projed_video_feat,
                src_txt_key_padding_mask=~neg_words_mask,
                pos_txt=neg_txt_position,
                src_vid_key_padding_mask=~video_mask,
                pos_vid=neg_vid_position,
            )
        else:
            neg_enhanced_video_feat = projed_video_feat

        neg_encoded_video_feat = self.t2v_encoder(
            neg_expanded_words_feat,
            neg_enhanced_video_feat,
            src_txt_key_padding_mask=~neg_expanded_words_mask,
            pos_txt=neg_expanded_txt_position,
            src_vid_key_padding_mask=~video_mask,
            pos_vid=neg_vid_position,
        )
        _, _, neg_memory, neg_memory_global = self.transformer(
            neg_encoded_video_feat,
            ~video_mask,
            self.query_embed.weight,
            neg_vid_position,
            global_token,
            global_token_pos,
        )

        saliency_scores = torch.sum(
            self.saliency_proj1(memory)
            * self.saliency_proj2(memory_global).unsqueeze(1),
            dim=-1,
        ) / np.sqrt(self.hidden_dim)
        neg_saliency_scores = torch.sum(
            self.saliency_proj1(neg_memory)
            * self.saliency_proj2(neg_memory_global).unsqueeze(1),
            dim=-1,
        ) / np.sqrt(self.hidden_dim)

        if self.aux_loss:
            aux_outputs = [
                {"pred_logits": class_output, "pred_spans": span_output}
                for class_output, span_output in zip(
                    outputs_class[:-1],
                    outputs_coord[:-1],
                )
            ]

        if self.rec_fw and kwargs["is_training"]:
            unknown_mask = kwargs["unknown_mask"]
            unknowned_words_feat = self._replace_unknown(
                projed_words_feat,
                unknown_mask,
                self.unknown_token,
                proj=True,
            )
            clip_mask = kwargs["clip_mask"]
            selected_video_feat = projed_video_feat[clip_mask]
            selected_length = clip_mask.sum(dim=1)
            merged_clip_feat, merged_clip_mask = split_and_pad(
                selected_length,
                selected_video_feat,
            )
            masked_words_feat, _ = self._mask_words(
                unknowned_words_feat,
                words_mask,
                self.masked_token,
                proj=True,
                weight=words_weight,
            )
            selected_vid_position = vid_position[clip_mask]
            merged_clip_position, _ = split_and_pad(
                selected_length,
                selected_vid_position,
            )
            recfw_out = self.enhance_encoder(
                merged_clip_feat,
                masked_words_feat,
                src_txt_key_padding_mask=~merged_clip_mask,
                pos_txt=merged_clip_position,
                src_vid_key_padding_mask=~words_mask,
                pos_vid=txt_position,
                is_MLM=True,
            )
            recfw_words_logit = self.output_txt_proj(recfw_out)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_spans": outputs_coord[-1],
            "saliency_scores": saliency_scores,
            "neg_saliency_scores": neg_saliency_scores,
        }
        if self.enable_phrase_steering or self.enable_competitor_loss:
            out.update(
                {
                    "phrase_token": phrase_token,
                    "phrase_valid": effective_phrase_valid,
                    "encoded_video_feat": encoded_video_feat,
                }
            )
            if hasattr(self.t2v_encoder, "phrase_steering"):
                out["phrase_steering_scale"] = self.t2v_encoder.phrase_steering.scale

        if self.aux_loss:
            out["aux_outputs"] = aux_outputs
        if self.rec_ss:
            out.update(
                {
                    "projed_video_feat": projed_video_feat,
                    "recon_feat": recon_feat,
                    "projed_recon_feat": projed_recon_feat,
                    "expanded_words_feat": expanded_words_feat,
                    "expanded_words_mask": expanded_words_mask,
                    "enhanced_video_feat": enhanced_video_feat,
                    "projed_words_feat": projed_words_feat,
                }
            )
        if self.rec_fw and kwargs["is_training"]:
            out.update(
                {
                    "words_mask": words_mask,
                    "recfw_words_logit": recfw_words_logit,
                }
            )
        return out
