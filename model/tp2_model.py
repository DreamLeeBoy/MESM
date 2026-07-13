"""MESM wrapper and criterion for TP²-style modules."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import MESM, MLP


class LayerWiseLinear(nn.Module):
    def __init__(
        self, num_layers: int, input_dim: int, output_dim: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, output_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[0] != len(self.layers):
            raise ValueError(
                "expected [layers, batch, queries, channels]"
            )
        return torch.stack([
            layer(x[index])
            for index, layer in enumerate(self.layers)
        ])


class LayerWiseMLP(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        mlp_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MLP(
                input_dim,
                hidden_dim,
                output_dim,
                mlp_layers,
            ) for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[0] != len(self.layers):
            raise ValueError(
                "expected [layers, batch, queries, channels]"
            )
        return torch.stack([
            layer(x[index])
            for index, layer in enumerate(self.layers)
        ])


class MESMTP2(MESM):
    """Official MESM with TP² TFPN/deformable modules."""

    def __init__(
        self,
        *args,
        num_feature_levels: int = 4,
        salient_upsample_type: str = "nearest",
        separate_predict_head: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_feature_levels = num_feature_levels
        self.salient_upsample_type = salient_upsample_type
        decoder_layers = self.transformer.num_decoder_layers

        if separate_predict_head:
            self.class_embed = LayerWiseLinear(
                decoder_layers, self.hidden_dim, 2
            )
            self.span_embed = LayerWiseMLP(
                decoder_layers,
                self.hidden_dim,
                self.hidden_dim,
                2,
                3,
            )
            self.transformer.decoder.bbox_embed = (
                self.span_embed.layers
            )
        else:
            self.transformer.decoder.bbox_embed = nn.ModuleList([
                self.span_embed
                for _ in range(decoder_layers)
            ])

        self.tp2_salient_embed = MLP(
            self.hidden_dim * num_feature_levels,
            self.hidden_dim,
            1,
            2,
        )

    def multiscale_saliency(
        self,
        levels: Sequence[Tensor],
        masks: Sequence[Tensor],
        target_length: int,
    ) -> Tensor:
        aligned: List[Tensor] = []
        for feature, mask in zip(levels, masks):
            feature = feature.masked_fill(
                ~mask.unsqueeze(-1), 0.0
            )
            if feature.shape[1] != target_length:
                if self.salient_upsample_type == "nearest":
                    feature = F.interpolate(
                        feature.transpose(1, 2),
                        size=target_length,
                        mode="nearest",
                    ).transpose(1, 2)
                elif self.salient_upsample_type == "linear":
                    feature = F.interpolate(
                        feature.transpose(1, 2),
                        size=target_length,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                else:
                    raise ValueError(
                        "unsupported salient upsampling"
                    )
            aligned.append(feature)
        fused = torch.cat(aligned, dim=-1)
        return self.tp2_salient_embed(fused).squeeze(-1)

    def forward(self, *args, **kwargs) -> Dict[str, Tensor]:
        self.transformer.clear_cache()
        output = super().forward(*args, **kwargs)
        cache = self.transformer.pop_cache()
        if not cache:
            raise RuntimeError(
                "TP² transformer did not return feature cache"
            )
        positive = cache[0]
        output["tp2_salient_logits"] = (
            self.multiscale_saliency(
                positive.memory_levels,
                positive.level_masks,
                positive.original_length,
            )
        )
        output["tp2_fpn_salient_logits"] = (
            self.multiscale_saliency(
                positive.fpn_levels,
                positive.level_masks,
                positive.original_length,
            )
        )
        return output


class TP2Criterion(nn.Module):
    """Adds TP² early saliency focal loss to MESM losses."""

    def __init__(
        self,
        base_criterion: nn.Module,
        loss_coef: float = 3.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        encoder_weight: float = 1.5,
        fpn_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_criterion = base_criterion
        self.loss_coef = loss_coef
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.encoder_weight = encoder_weight
        self.fpn_weight = fpn_weight
        self.weight_dict = dict(base_criterion.weight_dict)
        self.weight_dict["loss_tp2_saliency"] = loss_coef

    def focal_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        targets = targets.to(logits.dtype)
        valid_mask = valid_mask.to(logits.dtype)
        ce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probability = logits.sigmoid()
        p_t = (
            probability * targets
            + (1 - probability) * (1 - targets)
        )
        loss = ce * ((1 - p_t) ** self.focal_gamma)
        alpha_t = (
            self.focal_alpha * targets
            + (1 - self.focal_alpha) * (1 - targets)
        )
        loss = loss * alpha_t * valid_mask
        return loss.sum() / valid_mask.sum().clamp(min=1.0)

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        is_training: bool = True,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        losses, total = self.base_criterion(
            outputs, targets, is_training=is_training
        )
        if "tp2_salient_logits" not in outputs:
            return losses, total

        salient_gt = targets["clip_mask"].float()
        valid_mask = targets["video_mask"].float()
        encoder_loss = self.focal_loss(
            outputs["tp2_salient_logits"],
            salient_gt,
            valid_mask,
        )
        fpn_loss = self.focal_loss(
            outputs["tp2_fpn_salient_logits"],
            salient_gt,
            valid_mask,
        )
        tp2_loss = (
            self.encoder_weight * encoder_loss
            + self.fpn_weight * fpn_loss
        )
        losses["loss_tp2_saliency"] = tp2_loss
        total = total + self.loss_coef * tp2_loss
        return losses, total
