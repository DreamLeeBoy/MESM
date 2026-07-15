from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


class DynamicTemporalMemory(nn.Module):
    """Bounded temporal prototype memory for video moment retrieval."""

    def __init__(
        self,
        hidden_dim: int,
        capacity: int = 2048,
        topk: int = 5,
        min_entries: int = 32,
        temperature: float = 0.07,
        momentum: float = 0.9,
        merge_threshold: float = 0.95,
        residual_scale: float = 0.2,
        prototype_topk_frames: int = 4,
        min_text_similarity: float = 0.0,
    ):
        super().__init__()
        if hidden_dim <= 0 or capacity <= 0 or topk <= 0:
            raise ValueError("hidden_dim, capacity and topk must be positive")
        if min_entries < 0 or prototype_topk_frames <= 0:
            raise ValueError("min_entries must be non-negative and prototype_topk_frames positive")
        if temperature <= 0 or residual_scale < 0:
            raise ValueError("temperature must be positive and residual_scale non-negative")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if not -1.0 <= merge_threshold <= 1.0:
            raise ValueError("merge_threshold must be in [-1, 1]")

        self.hidden_dim = hidden_dim
        self.capacity = capacity
        self.topk = topk
        self.min_entries = min_entries
        self.temperature = temperature
        self.momentum = momentum
        self.merge_threshold = merge_threshold
        self.residual_scale = residual_scale
        self.prototype_topk_frames = prototype_topk_frames
        self.min_text_similarity = min_text_similarity

        self.query_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.delta = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

        self.register_buffer("memory", torch.zeros(capacity, hidden_dim))
        self.register_buffer("memory_counts", torch.zeros(capacity))
        self.register_buffer("memory_size", torch.zeros((), dtype=torch.long))
        self.register_buffer("memory_ptr", torch.zeros((), dtype=torch.long))

    @staticmethod
    def masked_mean(features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.to(features.dtype).unsqueeze(-1)
        return (features * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def retrieve(self, query: torch.Tensor) -> Dict[str, torch.Tensor]:
        size = int(self.memory_size.item())
        if size < max(self.min_entries, 1):
            return {
                "context": query.new_zeros(query.shape),
                "max_similarity": query.new_zeros(query.shape[0]),
                "uncertainty": query.new_ones(query.shape[0]),
                "ready": query.new_zeros((), dtype=torch.bool),
            }

        bank = F.normalize(self.memory[:size], dim=-1, eps=1e-6)
        similarity = F.normalize(query, dim=-1, eps=1e-6) @ bank.t()
        values, indices = similarity.topk(min(self.topk, size), dim=1)
        weights = F.softmax(values / self.temperature, dim=1)
        context = (weights.unsqueeze(-1) * bank[indices]).sum(1)
        return {
            "context": context,
            "max_similarity": values[:, 0],
            "uncertainty": 1.0 - values[:, 0].clamp(0.0, 1.0),
            "ready": query.new_ones((), dtype=torch.bool),
        }

    def forward(
        self,
        video_features: torch.Tensor,
        video_mask: torch.Tensor,
        text_context: torch.Tensor,
    ):
        if video_features.dim() != 3 or video_features.shape[-1] != self.hidden_dim:
            raise ValueError("video_features must have shape [B, L, hidden_dim]")
        if video_mask.shape != video_features.shape[:2]:
            raise ValueError("video_mask must have shape [B, L]")
        if text_context.shape != (video_features.shape[0], self.hidden_dim):
            raise ValueError("text_context must have shape [B, hidden_dim]")

        valid_mask = video_mask.bool()
        video = video_features * valid_mask.unsqueeze(-1).to(video_features.dtype)
        query = self.query_fusion(torch.cat([
            self.masked_mean(video, valid_mask), text_context
        ], dim=-1))
        state = self.retrieve(query)

        text = text_context.unsqueeze(1).expand_as(video)
        context = state["context"].unsqueeze(1).expand_as(video)
        gate = torch.sigmoid(self.gate(torch.cat([video, text, context], dim=-1)))
        gate = gate * valid_mask.unsqueeze(-1).to(gate.dtype)
        residual = self.delta(torch.cat([video, context], dim=-1))
        enhanced = video + self.residual_scale * gate * residual
        enhanced = enhanced * valid_mask.unsqueeze(-1).to(enhanced.dtype)

        state.update({
            "gate_mean": gate.sum() / valid_mask.sum().clamp_min(1),
            "size": self.memory_size.detach().clone(),
        })
        return enhanced, state

    @torch.no_grad()
    def build_pseudo_prototypes(
        self,
        video_features: torch.Tensor,
        video_mask: torch.Tensor,
        text_context: torch.Tensor,
    ) -> torch.Tensor:
        """Select text-relevant frames and pool one prototype per video."""
        valid_mask = video_mask.bool()
        video = F.normalize(video_features.detach(), dim=-1, eps=1e-6)
        text = F.normalize(text_context.detach(), dim=-1, eps=1e-6)
        similarity = (video * text.unsqueeze(1)).sum(-1)
        similarity = similarity.masked_fill(~valid_mask, -1e4)

        prototypes = []
        for index in range(video.shape[0]):
            valid_length = int(valid_mask[index].sum().item())
            if valid_length <= 0:
                continue
            k = min(self.prototype_topk_frames, valid_length)
            values, positions = similarity[index].topk(k)
            if float(values.mean().item()) < self.min_text_similarity:
                continue
            weights = F.softmax(values / self.temperature, dim=0)
            prototype = (weights.unsqueeze(-1) * video[index, positions]).sum(0)
            prototypes.append(F.normalize(prototype, dim=0, eps=1e-6))

        if not prototypes:
            return video_features.new_empty((0, self.hidden_dim))
        return torch.stack(prototypes).detach()

    @torch.no_grad()
    def update(self, candidates: Optional[torch.Tensor]) -> None:
        if candidates is None or candidates.numel() == 0:
            return
        candidates = candidates.to(self.memory.device, self.memory.dtype)
        candidates = candidates[torch.isfinite(candidates).all(1)]
        candidates = F.normalize(candidates, dim=-1, eps=1e-6)

        for candidate in candidates:
            size = int(self.memory_size.item())
            if size:
                bank = F.normalize(self.memory[:size], dim=-1, eps=1e-6)
                value, index = (bank @ candidate).max(0)
                if float(value.item()) >= self.merge_threshold:
                    idx = int(index.item())
                    merged = self.momentum * self.memory[idx] + (1.0 - self.momentum) * candidate
                    self.memory[idx].copy_(F.normalize(merged, dim=0, eps=1e-6))
                    self.memory_counts[idx].add_(1.0)
                    continue

            if size < self.capacity:
                idx = size
                self.memory_size.add_(1)
            else:
                idx = int(self.memory_ptr.item())
                self.memory_ptr.fill_((idx + 1) % self.capacity)
            self.memory[idx].copy_(candidate)
            self.memory_counts[idx].fill_(1.0)
