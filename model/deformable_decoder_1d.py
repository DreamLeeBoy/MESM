"""One-dimensional multi-scale deformable transformer decoder."""
from __future__ import annotations

import copy
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import MLP
from .ms_deform_attn_1d import MSDeformAttn1D


def _clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(0, 1)
    return torch.log(
        x.clamp(min=eps) / (1 - x).clamp(min=eps)
    )


def sine_embedding(reference: Tensor, d_model: int) -> Tensor:
    scale = 2 * math.pi
    each_dim = d_model // 2
    dim_t = torch.arange(
        each_dim, dtype=torch.float32, device=reference.device
    )
    dim_t = 10000 ** (
        2 * torch.div(dim_t, 2, rounding_mode="trunc") / each_dim
    )

    center = reference[..., 0] * scale
    pos_center = center[..., None] / dim_t
    pos_center = torch.stack(
        (pos_center[..., 0::2].sin(), pos_center[..., 1::2].cos()),
        dim=-1,
    ).flatten(-2)

    width = reference[..., 1] * scale
    pos_width = width[..., None] / dim_t
    pos_width = torch.stack(
        (pos_width[..., 0::2].sin(), pos_width[..., 1::2].cos()),
        dim=-1,
    ).flatten(-2)
    return torch.cat((pos_center, pos_width), dim=-1)


class DeformableDecoderLayer1D(nn.Module):
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
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn1D(
            d_model, n_levels, n_heads, n_points
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,
        query_pos: Tensor,
        reference_points: Tensor,
        memory: Tensor,
        temporal_shapes: Tensor,
        level_start_index: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        query = key = tgt + query_pos
        attended, _ = self.self_attn(
            query, key, tgt, need_weights=False
        )
        tgt = self.norm1(tgt + self.dropout1(attended))
        attended, _, _ = self.cross_attn(
            tgt + query_pos,
            reference_points,
            memory,
            temporal_shapes,
            level_start_index,
            padding_mask,
        )
        tgt = self.norm2(tgt + self.dropout2(attended))
        ffn = self.linear2(
            self.dropout3(F.relu(self.linear1(tgt)))
        )
        return self.norm3(tgt + self.dropout4(ffn))


class DeformableDecoder1D(nn.Module):
    def __init__(
        self,
        layer: DeformableDecoderLayer1D,
        num_layers: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.layers = _clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.ref_point_head = MLP(
            d_model, d_model, d_model, 2
        )
        self.bbox_embed: Optional[Sequence[nn.Module]] = None

    def forward(
        self,
        tgt: Tensor,
        initial_reference: Tensor,
        memory: Tensor,
        temporal_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        output = tgt
        reference_points = initial_reference
        intermediate: List[Tensor] = []
        references: List[Tensor] = []

        for layer_id, layer in enumerate(self.layers):
            references.append(reference_points)
            query_pos = self.ref_point_head(
                sine_embedding(reference_points, output.shape[-1])
            )
            reference_input = (
                reference_points[:, :, None, :]
                * valid_ratios[:, None].repeat(1, 1, 1, 2)
            )
            output = layer(
                output,
                query_pos,
                reference_input,
                memory,
                temporal_shapes,
                level_start_index,
                padding_mask,
            )
            output_norm = self.norm(output)
            intermediate.append(output_norm)

            if (
                self.bbox_embed is not None
                and layer_id < self.num_layers - 1
            ):
                delta = self.bbox_embed[layer_id](output_norm)
                reference_points = (
                    delta + inverse_sigmoid(reference_points)
                ).sigmoid().detach()

        return torch.stack(intermediate), torch.stack(references)
