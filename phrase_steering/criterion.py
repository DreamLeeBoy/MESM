"""Training-only same-video competitor loss for phrase steering."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.criterion import Criterion


class PhraseCriterion(Criterion):
    def __init__(self, *args, competitor_margin: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.competitor_margin = float(competitor_margin)

    def loss_competitor(self, outputs, targets, indices, log=True):
        required_outputs = {"phrase_token", "phrase_valid", "encoded_video_feat"}
        if not required_outputs.issubset(outputs) or "competitor_index" not in targets:
            zero = outputs["pred_logits"].sum() * 0.0
            return {
                "loss_competitor": zero,
                "competitor_valid_count": zero.detach(),
                "competitor_active_ratio": zero.detach(),
                "competitor_pos_sim": zero.detach(),
                "competitor_neg_sim": zero.detach(),
            }

        phrase_token = F.normalize(outputs["phrase_token"], dim=-1, eps=1e-6)
        encoded_video = outputs["encoded_video_feat"]
        clip_mask = targets["clip_mask"].to(
            device=encoded_video.device,
            dtype=encoded_video.dtype,
        )
        video_repr = (encoded_video * clip_mask.unsqueeze(-1)).sum(dim=1)
        video_repr = video_repr / clip_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        video_repr = F.normalize(video_repr, dim=-1, eps=1e-6)

        competitor_index = targets["competitor_index"].to(
            device=encoded_video.device,
            dtype=torch.long,
        )
        phrase_valid = outputs["phrase_valid"].to(
            device=encoded_video.device,
            dtype=torch.bool,
        )
        safe_index = competitor_index.clamp_min(0)
        valid = (
            (competitor_index >= 0)
            & phrase_valid
            & phrase_valid[safe_index]
        )

        if not bool(valid.any()):
            zero = outputs["pred_logits"].sum() * 0.0
            return {
                "loss_competitor": zero,
                "competitor_valid_count": zero.detach(),
                "competitor_active_ratio": zero.detach(),
                "competitor_pos_sim": zero.detach(),
                "competitor_neg_sim": zero.detach(),
            }

        positive_similarity = (video_repr * phrase_token).sum(dim=-1)
        negative_phrase = phrase_token[safe_index]
        negative_similarity = (video_repr * negative_phrase).sum(dim=-1)
        per_sample = F.relu(
            self.competitor_margin
            - positive_similarity
            + negative_similarity
        )
        selected_loss = per_sample[valid]
        loss = selected_loss.mean()

        return {
            "loss_competitor": loss,
            "competitor_valid_count": valid.sum().detach().to(loss.dtype),
            "competitor_active_ratio": (selected_loss > 0).float().mean().detach(),
            "competitor_pos_sim": positive_similarity[valid].mean().detach(),
            "competitor_neg_sim": negative_similarity[valid].mean().detach(),
        }

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        if loss == "competitor":
            return self.loss_competitor(outputs, targets, indices, **kwargs)
        return super().get_loss(loss, outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, is_training=True):
        outputs_without_aux = {
            key: value for key, value in outputs.items()
            if key != "aux_outputs"
        }
        indices = self.matcher(outputs_without_aux, targets)

        losses = {}
        for loss_name in self.losses:
            if loss_name == "rec_fw" and not is_training:
                continue
            if loss_name == "competitor" and not is_training:
                continue
            losses.update(
                self.get_loss(loss_name, outputs, targets, indices)
            )

        if "aux_outputs" in outputs:
            for layer_index, auxiliary_outputs in enumerate(outputs["aux_outputs"]):
                auxiliary_indices = self.matcher(auxiliary_outputs, targets)
                for loss_name in self.losses:
                    if loss_name in {
                        "saliency",
                        "bg_rank",
                        "rec_ss",
                        "rec_fw",
                        "competitor",
                        "path_balance",
                    }:
                        continue
                    layer_losses = self.get_loss(
                        loss_name,
                        auxiliary_outputs,
                        targets,
                        auxiliary_indices,
                    )
                    layer_losses = {
                        f"{key}_{layer_index}": value
                        for key, value in layer_losses.items()
                    }
                    losses.update(layer_losses)

        total_loss = sum(
            losses[key] * self.weight_dict[key]
            for key in losses
            if key in self.weight_dict
        )
        return losses, total_loss
