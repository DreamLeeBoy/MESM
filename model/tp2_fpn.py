"""Transformer temporal feature pyramid used by the TP²-MESM variant."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LocalTemporalTransformerBlock(nn.Module):
    """Local self-attention block with optional temporal downsampling."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        window_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        self.stride = stride
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _downsample(
        x: Tensor, valid_mask: Tensor, stride: int
    ) -> Tuple[Tensor, Tensor]:
        if stride == 1:
            return x, valid_mask
        x = F.interpolate(
            x.transpose(1, 2), scale_factor=1.0 / stride, mode="nearest"
        ).transpose(1, 2)
        valid_mask = F.interpolate(
            valid_mask[:, None].float(), size=x.shape[1], mode="nearest"
        ).squeeze(1).bool()
        return x, valid_mask

    def _local_attention_mask(
        self, length: int, device: torch.device
    ) -> Tensor:
        radius = self.window_size // 2
        index = torch.arange(length, device=device)
        return (index[:, None] - index[None, :]).abs() > radius

    def forward(
        self, x: Tensor, valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        x, valid_mask = self._downsample(x, valid_mask, self.stride)
        residual = x
        q = self.norm1(x)
        attn_out, _ = self.attn(
            q,
            q,
            q,
            attn_mask=self._local_attention_mask(q.shape[1], q.device),
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return x, valid_mask


class TemporalFeaturePyramid(nn.Module):
    """Builds transformer TFPN levels: T, T/s, T/s², ..."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        stem_layers: int = 1,
        branch_layers: int = 3,
        downsample_rate: int = 2,
        window_size: int = 9,
    ) -> None:
        super().__init__()
        if stem_layers < 1 or branch_layers < 0:
            raise ValueError("invalid TFPN layer counts")
        self.stem = nn.ModuleList([
            LocalTemporalTransformerBlock(
                d_model, nhead, dim_feedforward, dropout,
                window_size, stride=1
            ) for _ in range(stem_layers)
        ])
        self.branch = nn.ModuleList([
            LocalTemporalTransformerBlock(
                d_model, nhead, dim_feedforward, dropout,
                window_size, stride=downsample_rate
            ) for _ in range(branch_layers)
        ])
        self.num_levels = stem_layers + branch_layers

    def forward(
        self, src: Tensor, valid_mask: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        levels: List[Tensor] = []
        masks: List[Tensor] = []
        x, mask = src, valid_mask
        for block in self.stem:
            x, mask = block(x, mask)
            levels.append(x)
            masks.append(mask)
        for block in self.branch:
            x, mask = block(x, mask)
            levels.append(x)
            masks.append(mask)
        return levels, masks
