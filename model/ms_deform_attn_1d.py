"""Pure-PyTorch one-dimensional multi-scale deformable attention."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def ms_deform_attn_core_pytorch_1d(
    value: Tensor,
    value_temporal_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch, _, n_heads, head_dim = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape
    split_sizes = [int(v) for v in value_temporal_shapes.tolist()]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampled_per_level: List[Tensor] = []

    for level, temporal_length in enumerate(split_sizes):
        value_l = value_list[level].permute(0, 2, 3, 1).reshape(
            batch * n_heads, head_dim, 1, temporal_length
        )
        grid_x = sampling_grids[:, :, :, level, :, 0].permute(0, 2, 1, 3)
        grid_x = grid_x.reshape(batch * n_heads, len_q, n_points)
        grid = torch.stack((grid_x, torch.zeros_like(grid_x)), dim=-1)
        sampled_per_level.append(F.grid_sample(
            value_l, grid, mode="bilinear", padding_mode="zeros",
            align_corners=False
        ))

    weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        batch * n_heads, 1, len_q, n_levels * n_points
    )
    sampled = torch.stack(sampled_per_level, dim=-2).flatten(-2)
    output = (sampled * weights).sum(-1)
    return output.view(batch, n_heads * head_dim, len_q).transpose(1, 2).contiguous()


class MSDeformAttn1D(nn.Module):
    def __init__(
        self, d_model: int, n_levels: int, n_heads: int, n_points: int
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points
        )
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        phases = torch.arange(self.n_heads, dtype=torch.float32)
        phases = phases * (2 * math.pi / self.n_heads)
        grid = phases.cos()
        grid = grid / grid.abs().max().clamp(min=1e-6)
        grid = grid.view(self.n_heads, 1, 1, 1).repeat(
            1, self.n_levels, self.n_points, 1
        )
        for point in range(self.n_points):
            grid[:, :, point, :] *= point + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.flatten())
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_temporal_shapes: Tensor,
        input_level_start_index: Tensor,
        input_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        del input_level_start_index
        batch, len_q, _ = query.shape
        _, len_in, channels = input_flatten.shape
        if int(input_temporal_shapes.sum()) != len_in:
            raise ValueError("temporal shapes do not match flattened input")

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(
            batch, len_in, self.n_heads, channels // self.n_heads
        )
        offsets = self.sampling_offsets(query).view(
            batch, len_q, self.n_heads, self.n_levels,
            self.n_points, 1
        )
        weights = self.attention_weights(query).view(
            batch, len_q, self.n_heads,
            self.n_levels * self.n_points
        )
        weights = F.softmax(weights, dim=-1).view(
            batch, len_q, self.n_heads, self.n_levels,
            self.n_points
        )

        if reference_points.shape[-1] == 1:
            normalizer = input_temporal_shapes.view(
                1, 1, 1, self.n_levels, 1, 1
            )
            locations = (
                reference_points[:, :, None, :, None, :]
                + offsets / normalizer
            )
        elif reference_points.shape[-1] == 2:
            locations = (
                reference_points[:, :, None, :, None, :1]
                + offsets / self.n_points
                * reference_points[:, :, None, :, None, 1:] * 0.5
            )
        else:
            raise ValueError("reference point dimension must be 1 or 2")

        output = ms_deform_attn_core_pytorch_1d(
            value, input_temporal_shapes, locations, weights
        )
        return self.output_proj(output), locations, weights
