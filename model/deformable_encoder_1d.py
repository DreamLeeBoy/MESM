"""One-dimensional multi-scale deformable transformer encoder."""
from __future__ import annotations

import copy
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ms_deform_attn_1d import MSDeformAttn1D


def _clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


class DeformableEncoderLayer1D(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        n_levels: int,
        n_heads: int,
        n_points: int,
    ) -> None:
        super().__init__()
        self.self_attn = MSDeformAttn1D(
            d_model, n_levels, n_heads, n_points
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        reference_points: Tensor,
        temporal_shapes: Tensor,
        level_start_index: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        attended, _, _ = self.self_attn(
            src + pos,
            reference_points,
            src,
            temporal_shapes,
            level_start_index,
            padding_mask,
        )
        src = self.norm1(src + self.dropout1(attended))
        ffn = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout3(ffn))


class DeformableEncoder1D(nn.Module):
    def __init__(
        self, layer: DeformableEncoderLayer1D, num_layers: int
    ) -> None:
        super().__init__()
        self.layers = _clones(layer, num_layers)

    @staticmethod
    def get_reference_points(
        temporal_shapes: Tensor,
        valid_ratios: Tensor,
        device: torch.device,
    ) -> Tensor:
        references: List[Tensor] = []
        for level, temporal_length in enumerate(
            temporal_shapes.tolist()
        ):
            reference = torch.linspace(
                0.5,
                temporal_length - 0.5,
                temporal_length,
                dtype=torch.float32,
                device=device,
            )
            denominator = (
                valid_ratios[:, None, level, 0] * temporal_length
            )
            reference = reference[None, :] / denominator.clamp(
                min=1e-6
            )
            references.append(reference)
        reference_points = torch.cat(references, dim=1).unsqueeze(-1)
        return (
            reference_points[:, :, None, :]
            * valid_ratios[:, None, :, :]
        )

    def forward(
        self,
        src: Tensor,
        temporal_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        pos: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        reference_points = self.get_reference_points(
            temporal_shapes, valid_ratios, src.device
        )
        output = src
        for layer in self.layers:
            output = layer(
                output,
                pos,
                reference_points,
                temporal_shapes,
                level_start_index,
                padding_mask,
            )
        return output
