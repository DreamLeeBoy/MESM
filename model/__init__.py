# from .clip_text_encoder import build_text_encoder
# from .transformer import build_t2v_encoder, build_transformer
# from .position_encoding import build_position_encoding
# from .model import build_DETR
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .text_encoder import GloveTextEncoder, GloVe
from .text_encoder import CLIPTextEncoder, convert_weights
from .transformer import T2VEncoder, T2VEncoder_TwoMLP, Transformer
from .position_encoding import PositionEmbeddingSine, TrainablePositionalEncoding
from .model import MESM as _BaseMESM
# from .model_mid import DETR
from .matcher import HungarianMatcher
from .criterion import Criterion as _BaseCriterion


_PHRASE_CONTRASTIVE_CONFIG = {
    "enable_phrase_contrastive": False,
    "phrase_mlp_dropout": 0.1,
    "phrase_nce_temperature": 0.1,
    "action_nce_temperature": 0.1,
    "loss_phrase_nce_coef": 0.0,
    "loss_action_nce_coef": 0.0,
}


def configure_phrase_contrastive(opt) -> None:
    """Configure the original MESM model and criterion from parsed options."""
    for key, default in tuple(_PHRASE_CONTRASTIVE_CONFIG.items()):
        _PHRASE_CONTRASTIVE_CONFIG[key] = getattr(opt, key, default)


class PhraseEncoding(NamedTuple):
    action_token: Tensor
    object_token: Tensor
    phrase_token: Tensor
    valid: Tensor


class ActionObjectPhraseEncoder(nn.Module):
    """Build action, object, and action-object phrase vectors from word tokens."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_norm = nn.LayerNorm(hidden_dim)
        self.object_norm = nn.LayerNorm(hidden_dim)
        self.phrase_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _masked_mean(features: Tensor, mask: Tensor) -> Tensor:
        mask_float = mask.to(dtype=features.dtype).unsqueeze(-1)
        numerator = (features * mask_float).sum(dim=1)
        denominator = mask_float.sum(dim=1).clamp_min(1.0)
        return numerator / denominator

    def forward(
        self,
        word_features: Tensor,
        words_mask: Optional[Tensor],
        action_mask: Optional[Tensor],
        object_mask: Optional[Tensor],
        phrase_valid: Optional[Tensor],
    ) -> PhraseEncoding:
        batch_size, text_length, hidden_dim = word_features.shape
        device = word_features.device
        empty_token = word_features.new_zeros(batch_size, hidden_dim)
        empty_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if (
            words_mask is None
            or action_mask is None
            or object_mask is None
            or phrase_valid is None
        ):
            return PhraseEncoding(empty_token, empty_token.clone(), empty_token.clone(), empty_valid)

        words_mask = words_mask[:, :text_length].to(device=device, dtype=torch.bool)
        action_mask = action_mask[:, :text_length].to(device=device, dtype=torch.bool)
        object_mask = object_mask[:, :text_length].to(device=device, dtype=torch.bool)
        phrase_valid = phrase_valid.to(device=device, dtype=torch.bool)

        action_mask = action_mask & words_mask
        object_mask = object_mask & words_mask
        effective_valid = phrase_valid & action_mask.any(dim=1) & object_mask.any(dim=1)

        action_raw = self._masked_mean(word_features, action_mask)
        object_raw = self._masked_mean(word_features, object_mask)

        action_token = F.normalize(self.action_norm(action_raw), dim=-1, eps=1e-6)
        object_token = F.normalize(self.object_norm(object_raw), dim=-1, eps=1e-6)
        phrase_delta = self.mlp(torch.cat([action_raw, object_raw], dim=-1))
        phrase_token = F.normalize(
            self.phrase_norm(action_raw + phrase_delta),
            dim=-1,
            eps=1e-6,
        )

        valid_float = effective_valid.to(word_features.dtype).unsqueeze(-1)
        return PhraseEncoding(
            action_token * valid_float,
            object_token * valid_float,
            phrase_token * valid_float,
            effective_valid,
        )


class PhraseContrastiveMESM(_BaseMESM):
    """Original MESM with training-only action/phrase contrastive outputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = _PHRASE_CONTRASTIVE_CONFIG
        self.enable_phrase_contrastive = bool(config["enable_phrase_contrastive"])
        self.phrase_encoder = ActionObjectPhraseEncoder(
            hidden_dim=self.hidden_dim,
            dropout=float(config["phrase_mlp_dropout"]),
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
        is_training = bool(kwargs.get("is_training", False))
        if not (self.enable_phrase_contrastive and is_training):
            return super().forward(
                video_feat,
                video_mask,
                words_id,
                words_mask,
                words_weight,
                num_clips,
                **kwargs,
            )

        captured = {}

        def capture_projected_words(_module, _inputs, output):
            if "projected_words" not in captured:
                captured["projected_words"] = output

        def capture_positive_t2v(_module, _inputs, output):
            if "encoded_video_feat" not in captured:
                captured["encoded_video_feat"] = output

        word_hook = self.input_txt_proj.register_forward_hook(capture_projected_words)
        video_hook = self.t2v_encoder.register_forward_hook(capture_positive_t2v)
        try:
            outputs = super().forward(
                video_feat,
                video_mask,
                words_id,
                words_mask,
                words_weight,
                num_clips,
                **kwargs,
            )
        finally:
            word_hook.remove()
            video_hook.remove()

        if "projected_words" not in captured or "encoded_video_feat" not in captured:
            raise RuntimeError("Failed to capture original MESM text or T2V features.")

        phrase_encoding = self.phrase_encoder(
            word_features=captured["projected_words"],
            words_mask=words_mask,
            action_mask=kwargs.get("phrase_action_mask"),
            object_mask=kwargs.get("phrase_object_mask"),
            phrase_valid=kwargs.get("phrase_valid"),
        )
        outputs.update(
            {
                "action_token": phrase_encoding.action_token,
                "object_token": phrase_encoding.object_token,
                "phrase_token": phrase_encoding.phrase_token,
                "phrase_valid": phrase_encoding.valid,
                "encoded_video_feat": captured["encoded_video_feat"],
            }
        )
        return outputs


class PhraseContrastiveCriterion(_BaseCriterion):
    """Original MESM criterion plus masked phrase/action InfoNCE."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = _PHRASE_CONTRASTIVE_CONFIG
        self.enable_phrase_contrastive = bool(config["enable_phrase_contrastive"])
        self.phrase_nce_temperature = float(config["phrase_nce_temperature"])
        self.action_nce_temperature = float(config["action_nce_temperature"])
        self.weight_dict["loss_phrase_nce"] = float(config["loss_phrase_nce_coef"])
        self.weight_dict["loss_action_nce"] = float(config["loss_action_nce_coef"])

    @staticmethod
    def _zero(outputs):
        return outputs["pred_logits"].sum() * 0.0

    @staticmethod
    def _masked_info_nce(
        video_token: Tensor,
        text_token: Tensor,
        negative_indices: Tensor,
        negative_mask: Tensor,
        anchor_valid: Tensor,
        temperature: float,
    ):
        safe_indices = negative_indices.clamp(min=0, max=text_token.shape[0] - 1)
        negative_token = text_token[safe_indices]
        positive_logit = (video_token * text_token).sum(dim=-1, keepdim=True)
        negative_logits = (video_token.unsqueeze(1) * negative_token).sum(dim=-1)
        negative_logits = negative_logits.masked_fill(~negative_mask, -1e4)
        logits = torch.cat([positive_logit, negative_logits], dim=1) / temperature

        selected_logits = logits[anchor_valid]
        targets = torch.zeros(
            selected_logits.shape[0],
            dtype=torch.long,
            device=selected_logits.device,
        )
        loss = F.cross_entropy(selected_logits, targets)
        top1 = (selected_logits.argmax(dim=1) == 0).float().mean()
        pos_sim = positive_logit.squeeze(1)[anchor_valid].mean()
        valid_negative_sim = negative_logits[anchor_valid][negative_mask[anchor_valid]]
        neg_sim = valid_negative_sim.mean()
        return loss, top1, pos_sim, neg_sim

    def loss_phrase_contrastive(self, outputs, targets):
        required_outputs = {
            "action_token",
            "phrase_token",
            "phrase_valid",
            "encoded_video_feat",
        }
        required_targets = {
            "clip_mask",
            "strict_negative_indices",
            "strict_negative_mask",
        }
        if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
            zero = self._zero(outputs)
            return {
                "loss_phrase_nce": zero,
                "loss_action_nce": zero,
                "contrastive_valid_count": zero.detach(),
                "strict_negative_count_mean": zero.detach(),
                "phrase_top1_accuracy": zero.detach(),
                "action_top1_accuracy": zero.detach(),
                "phrase_positive_similarity": zero.detach(),
                "phrase_negative_similarity": zero.detach(),
                "action_positive_similarity": zero.detach(),
                "action_negative_similarity": zero.detach(),
            }

        encoded_video = outputs["encoded_video_feat"]
        clip_mask_bool = targets["clip_mask"].to(device=encoded_video.device, dtype=torch.bool)
        clip_mask = clip_mask_bool.to(dtype=encoded_video.dtype)
        video_token = (encoded_video * clip_mask.unsqueeze(-1)).sum(dim=1)
        video_token = video_token / clip_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        video_token = F.normalize(video_token, dim=-1, eps=1e-6)

        phrase_token = F.normalize(outputs["phrase_token"], dim=-1, eps=1e-6)
        action_token = F.normalize(outputs["action_token"], dim=-1, eps=1e-6)
        phrase_valid = outputs["phrase_valid"].to(device=encoded_video.device, dtype=torch.bool)
        negative_indices = targets["strict_negative_indices"].to(
            device=encoded_video.device,
            dtype=torch.long,
        )
        negative_mask = targets["strict_negative_mask"].to(
            device=encoded_video.device,
            dtype=torch.bool,
        )

        index_in_range = (negative_indices >= 0) & (negative_indices < phrase_token.shape[0])
        safe_indices = negative_indices.clamp(min=0, max=phrase_token.shape[0] - 1)
        negative_mask = negative_mask & index_in_range & phrase_valid[safe_indices]
        anchor_valid = phrase_valid & clip_mask_bool.any(dim=1) & negative_mask.any(dim=1)

        if not bool(anchor_valid.any()):
            zero = self._zero(outputs)
            return {
                "loss_phrase_nce": zero,
                "loss_action_nce": zero,
                "contrastive_valid_count": anchor_valid.sum().detach().to(zero.dtype),
                "strict_negative_count_mean": negative_mask.sum(dim=1).float().mean().detach(),
                "phrase_top1_accuracy": zero.detach(),
                "action_top1_accuracy": zero.detach(),
                "phrase_positive_similarity": zero.detach(),
                "phrase_negative_similarity": zero.detach(),
                "action_positive_similarity": zero.detach(),
                "action_negative_similarity": zero.detach(),
            }

        phrase_loss, phrase_acc, phrase_pos, phrase_neg = self._masked_info_nce(
            video_token,
            phrase_token,
            negative_indices,
            negative_mask,
            anchor_valid,
            self.phrase_nce_temperature,
        )
        action_loss, action_acc, action_pos, action_neg = self._masked_info_nce(
            video_token,
            action_token,
            negative_indices,
            negative_mask,
            anchor_valid,
            self.action_nce_temperature,
        )
        return {
            "loss_phrase_nce": phrase_loss,
            "loss_action_nce": action_loss,
            "contrastive_valid_count": anchor_valid.sum().detach().to(phrase_loss.dtype),
            "strict_negative_count_mean": negative_mask[anchor_valid].sum(dim=1).float().mean().detach(),
            "phrase_top1_accuracy": phrase_acc.detach(),
            "action_top1_accuracy": action_acc.detach(),
            "phrase_positive_similarity": phrase_pos.detach(),
            "phrase_negative_similarity": phrase_neg.detach(),
            "action_positive_similarity": action_pos.detach(),
            "action_negative_similarity": action_neg.detach(),
        }

    def forward(self, outputs, targets, is_training=True):
        losses, total_loss = super().forward(outputs, targets, is_training=is_training)
        if self.enable_phrase_contrastive and is_training:
            contrastive_losses = self.loss_phrase_contrastive(outputs, targets)
            losses.update(contrastive_losses)
            total_loss = total_loss + (
                contrastive_losses["loss_phrase_nce"]
                * self.weight_dict["loss_phrase_nce"]
                + contrastive_losses["loss_action_nce"]
                * self.weight_dict["loss_action_nce"]
            )
        return losses, total_loss


# Keep runner.py unchanged: it imports these names from the original model package.
MESM = PhraseContrastiveMESM
Criterion = PhraseContrastiveCriterion
