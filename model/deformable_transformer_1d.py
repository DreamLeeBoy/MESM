"""MESM-compatible 1D multi-scale deformable transformer."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor, nn

from .deformable_decoder_1d import (
    DeformableDecoder1D,
    DeformableDecoderLayer1D,
)
from .deformable_encoder_1d import (
    DeformableEncoder1D,
    DeformableEncoderLayer1D,
)
from .tp2_fpn import TemporalFeaturePyramid


def sine_position(
    mask: Tensor, d_model: int, temperature: int = 10000
) -> Tensor:
    x_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = (
        x_embed / x_embed[:, -1:].clamp(min=1.0)
        * (2 * math.pi)
    )
    dim_t = torch.arange(
        d_model, dtype=torch.float32, device=mask.device
    )
    dim_t = temperature ** (
        2 * torch.div(dim_t, 2, rounding_mode="trunc")
        / d_model
    )
    pos = x_embed[:, :, None] / dim_t
    return torch.stack(
        (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()),
        dim=3,
    ).flatten(2)


@dataclass
class TP2Cache:
    fpn_levels: List[Tensor]
    memory_levels: List[Tensor]
    level_masks: List[Tensor]
    original_length: int


class DeformableTransformer1D(nn.Module):
    """Drop-in replacement for MESM's localization transformer."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_queries: int = 10,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 5,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_feature_levels: int = 4,
        enc_n_points: int = 4,
        dec_n_points: int = 4,
        fpn_stem_layers: int = 1,
        fpn_branch_layers: int = 3,
        fpn_downsample_rate: int = 2,
        fpn_window_size: int = 9,
    ) -> None:
        super().__init__()
        if (
            fpn_stem_layers + fpn_branch_layers
            != num_feature_levels
        ):
            raise ValueError(
                "TFPN layer count must equal num_feature_levels"
            )
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = "relu"
        self.normalize_before = False
        self.num_decoder_layers = num_decoder_layers
        self.num_feature_levels = num_feature_levels

        self.fpn = TemporalFeaturePyramid(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            stem_layers=fpn_stem_layers,
            branch_layers=fpn_branch_layers,
            downsample_rate=fpn_downsample_rate,
            window_size=fpn_window_size,
        )
        self.level_embed = nn.Parameter(
            torch.empty(num_feature_levels, d_model)
        )
        encoder_layer = DeformableEncoderLayer1D(
            d_model,
            dim_feedforward,
            dropout,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = DeformableEncoder1D(
            encoder_layer, num_encoder_layers
        )
        decoder_layer = DeformableDecoderLayer1D(
            d_model,
            dim_feedforward,
            dropout,
            num_feature_levels,
            nhead,
            dec_n_points,
        )
        self.decoder = DeformableDecoder1D(
            decoder_layer, num_decoder_layers, d_model
        )
        self.query_content = nn.Embedding(
            num_queries, d_model
        )

        self.global_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.global_dropout1 = nn.Dropout(dropout)
        self.global_norm1 = nn.LayerNorm(d_model)
        self.global_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.global_norm2 = nn.LayerNorm(d_model)
        self._cache: List[TP2Cache] = []
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.level_embed)

    def clear_cache(self) -> None:
        self._cache.clear()

    def pop_cache(self) -> List[TP2Cache]:
        cache = self._cache
        self._cache = []
        return cache

    @staticmethod
    def valid_ratio(padding_mask: Tensor) -> Tensor:
        valid = (~padding_mask).sum(1).float()
        return (
            valid / padding_mask.shape[1]
        ).unsqueeze(-1)

    def forward(
        self,
        src: Tensor,
        mask: Tensor,
        query_embed: Tensor,
        pos_embed: Tensor,
        global_token: Tensor,
        global_token_pos: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        del pos_embed
        fpn_levels, level_valid_masks = self.fpn(
            src, ~mask
        )

        flat_src: List[Tensor] = []
        flat_mask: List[Tensor] = []
        flat_pos: List[Tensor] = []
        temporal_shapes: List[int] = []
        for level, (feature, valid_mask) in enumerate(
            zip(fpn_levels, level_valid_masks)
        ):
            temporal_shapes.append(feature.shape[1])
            flat_src.append(feature)
            flat_mask.append(~valid_mask)
            flat_pos.append(
                sine_position(valid_mask, self.d_model)
                + self.level_embed[level]
            )

        src_flatten = torch.cat(flat_src, dim=1)
        padding_mask = torch.cat(flat_mask, dim=1)
        pos_flatten = torch.cat(flat_pos, dim=1)
        temporal_shapes_tensor = torch.as_tensor(
            temporal_shapes,
            dtype=torch.long,
            device=src.device,
        )
        level_start_index = torch.cat((
            temporal_shapes_tensor.new_zeros(1),
            temporal_shapes_tensor.cumsum(0)[:-1],
        ))
        valid_ratios = torch.stack([
            self.valid_ratio(level_padding)
            for level_padding in flat_mask
        ], dim=1)

        memory = self.encoder(
            src_flatten,
            temporal_shapes_tensor,
            level_start_index,
            valid_ratios,
            pos_flatten,
            padding_mask,
        )
        memory_levels = list(
            memory.split(temporal_shapes, dim=1)
        )
        high_resolution_memory = memory_levels[0]

        global_query = global_token + global_token_pos
        global_context, _ = self.global_attn(
            global_query,
            memory + pos_flatten,
            memory,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        global_hidden = self.global_norm1(
            global_token
            + self.global_dropout1(global_context)
        )
        memory_global = self.global_norm2(
            global_hidden + self.global_ffn(global_hidden)
        ).squeeze(1)

        batch = src.shape[0]
        if query_embed.shape[-1] != 2:
            raise ValueError(
                "MESM queries must be (center, width) logits"
            )
        initial_reference = query_embed.sigmoid().unsqueeze(0)
        initial_reference = initial_reference.expand(
            batch, -1, -1
        )
        tgt = self.query_content.weight.unsqueeze(0).expand(
            batch, -1, -1
        )
        hs, references = self.decoder(
            tgt,
            initial_reference,
            memory,
            temporal_shapes_tensor,
            level_start_index,
            valid_ratios,
            padding_mask,
        )

        self._cache.append(TP2Cache(
            fpn_levels=fpn_levels,
            memory_levels=memory_levels,
            level_masks=level_valid_masks,
            original_length=src.shape[1],
        ))
        return (
            hs,
            references,
            high_resolution_memory,
            memory_global,
        )
