"""Charades sidecar loading and batch interfaces for phrase steering."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from dataset.base import collate as base_collate
from dataset.charades import CharadesDataset


@dataclass(frozen=True)
class PhraseDatasetConfig:
    semantic_sidecar_dir: Optional[str] = None
    semantic_sidecar_train_file: str = "charades_sta_train_semantic_final_v1_2.jsonl"
    semantic_sidecar_test_file: str = "charades_sta_test_semantic_final_v1_2.jsonl"
    enable_phrase_steering: bool = False
    enable_competitor_loss: bool = False
    require_semantic_sidecar: bool = True


_RUNTIME_CONFIG = PhraseDatasetConfig()


def configure_phrase_dataset(opt) -> None:
    """Capture experiment options before the standard runner builds datasets."""
    global _RUNTIME_CONFIG
    _RUNTIME_CONFIG = PhraseDatasetConfig(
        semantic_sidecar_dir=getattr(opt, "semantic_sidecar_dir", None),
        semantic_sidecar_train_file=getattr(
            opt,
            "semantic_sidecar_train_file",
            "charades_sta_train_semantic_final_v1_2.jsonl",
        ),
        semantic_sidecar_test_file=getattr(
            opt,
            "semantic_sidecar_test_file",
            "charades_sta_test_semantic_final_v1_2.jsonl",
        ),
        enable_phrase_steering=bool(getattr(opt, "enable_phrase_steering", False)),
        enable_competitor_loss=bool(getattr(opt, "loss_competitor_coef", 0.0) > 0),
        require_semantic_sidecar=bool(getattr(opt, "require_semantic_sidecar", True)),
    )


def _normalize_sentence(sentence: str) -> str:
    return " ".join(sentence.lower().strip().split())


class PhraseCharadesDataset(CharadesDataset):
    """Original Charades-STA dataset plus phrase masks and Level-1 candidates."""

    def __init__(
        self,
        ann_path,
        feat_files,
        split,
        use_tef,
        clip_len,
        max_words_l,
        max_video_l,
        tokenizer_type,
        load_vocab_pkl,
        bpe_path,
        vocab,
        normalize_video,
        contra_samples,
        recfw,
        vocab_size,
        max_gather_size,
    ):
        config = _RUNTIME_CONFIG
        self.semantic_sidecar_dir = config.semantic_sidecar_dir
        self.semantic_sidecar_train_file = config.semantic_sidecar_train_file
        self.semantic_sidecar_test_file = config.semantic_sidecar_test_file
        self.enable_phrase_steering = config.enable_phrase_steering
        self.enable_competitor_loss = config.enable_competitor_loss and split == "train"
        self.require_semantic_sidecar = config.require_semantic_sidecar

        phrase_features_active = self.enable_phrase_steering or self.enable_competitor_loss
        if phrase_features_active and tokenizer_type != "CLIP":
            raise ValueError(
                "The v1.2 semantic sidecar stores exact CLIP BPE positions; "
                "tokenizer_type must be CLIP."
            )
        if phrase_features_active and max_words_l != 16:
            raise ValueError("The audited v1.2 sidecar was finalized for max_words_l=16.")
        if self.enable_competitor_loss and max_gather_size > 0:
            raise ValueError(
                "Level-1 competitor sampling requires max_gather_size=-1 so all "
                "annotations from one video remain in the same dataset item."
            )

        super().__init__(
            ann_path,
            feat_files,
            split,
            use_tef,
            clip_len,
            max_words_l,
            max_video_l,
            tokenizer_type,
            load_vocab_pkl,
            bpe_path,
            vocab,
            normalize_video,
            contra_samples,
            recfw,
            vocab_size,
            max_gather_size,
        )

    def _semantic_sidecar_path(self) -> Optional[Path]:
        if not self.semantic_sidecar_dir:
            return None
        filename = (
            self.semantic_sidecar_train_file
            if self.split == "train"
            else self.semantic_sidecar_test_file
        )
        return Path(self.semantic_sidecar_dir) / filename

    def _load_semantic_sidecar(self) -> Dict[int, dict]:
        sidecar_path = self._semantic_sidecar_path()
        semantic_required = self.enable_phrase_steering or self.enable_competitor_loss
        if sidecar_path is None or not sidecar_path.is_file():
            if semantic_required and self.require_semantic_sidecar:
                expected = sidecar_path or Path("<semantic_sidecar_dir is not configured>")
                raise FileNotFoundError(
                    f"Phrase features are enabled but the {self.split} sidecar is missing: {expected}"
                )
            return {}

        records: Dict[int, dict] = {}
        with sidecar_path.open("r", encoding="utf-8") as handle:
            for jsonl_line, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                line_id = int(record["line_id"])
                if line_id in records:
                    raise ValueError(
                        f"Duplicate line_id={line_id} in {sidecar_path} at JSONL line {jsonl_line}."
                    )
                records[line_id] = record
        return records

    def _empty_semantic_fields(self):
        return {
            "phrase_action_mask": torch.zeros(self.max_words_l, dtype=torch.bool),
            "phrase_object_mask": torch.zeros(self.max_words_l, dtype=torch.bool),
            "phrase_valid": False,
            "level1_candidate_line_ids": [],
        }

    def _semantic_fields(
        self,
        line_id: int,
        video_id: str,
        sentence: str,
        semantic_by_id: Dict[int, dict],
    ):
        if line_id not in semantic_by_id:
            if (
                self.enable_phrase_steering or self.enable_competitor_loss
            ) and self.require_semantic_sidecar:
                raise KeyError(
                    f"The semantic sidecar has no record for annotation line_id={line_id}."
                )
            return self._empty_semantic_fields()

        semantic = semantic_by_id[line_id]
        if semantic.get("video_id") != video_id:
            raise ValueError(
                f"Sidecar video mismatch at line_id={line_id}: "
                f"{semantic.get('video_id')} != {video_id}."
            )
        if _normalize_sentence(semantic.get("sentence", "")) != _normalize_sentence(sentence):
            raise ValueError(
                f"Sidecar sentence mismatch at line_id={line_id}. "
                "Regenerate the sidecar from the same annotation file."
            )

        alignment = semantic.get("clip_alignment", {})
        phrase_valid = bool(
            semantic.get("phrase_valid_model", False)
            and alignment.get("status") == "aligned"
        )
        action_positions = alignment.get("action_bpe_positions") or []
        object_positions = alignment.get("object_bpe_positions") or []

        action_mask = torch.zeros(self.max_words_l, dtype=torch.bool)
        object_mask = torch.zeros(self.max_words_l, dtype=torch.bool)

        for raw_position in action_positions:
            position = int(raw_position)
            if position < 0 or position >= self.max_words_l:
                phrase_valid = False
                continue
            action_mask[position] = True

        for raw_position in object_positions:
            position = int(raw_position)
            if position < 0 or position >= self.max_words_l:
                phrase_valid = False
                continue
            object_mask[position] = True

        phrase_valid = phrase_valid and bool(action_mask.any()) and bool(object_mask.any())
        candidates = semantic.get("level1_candidate_line_ids", [])
        candidates = [int(candidate) for candidate in candidates] if phrase_valid else []

        return {
            "phrase_action_mask": action_mask,
            "phrase_object_mask": object_mask,
            "phrase_valid": phrase_valid,
            "level1_candidate_line_ids": candidates,
        }

    def load_annotations(self):
        durations = self._load_durations()
        split2filename = {
            "train": "charades_sta_train.txt",
            "test": "charades_sta_test.txt",
        }
        ann_file = os.path.join(self.ann_path, split2filename[self.split])
        semantic_by_id = self._load_semantic_sidecar()

        with open(ann_file, "r", encoding="utf-8") as handle:
            lines = handle.readlines()

        annotations = []
        for line_id in tqdm(
            range(len(lines)),
            desc=f"Load Charades {self.split} annotations with phrase sidecar",
        ):
            meta = lines[line_id].split("##")
            video_id, start, end = meta[0].split()
            start, end = float(start), float(end)
            duration = durations[video_id]
            if start > duration:
                continue
            if start > end:
                start, end = end, start
            if end > duration:
                end = duration

            moment = [start, end]
            if self.clip_len == -1:
                start_idx = start / duration
                end_idx = end / duration
            else:
                start_idx = int(start / self.clip_len)
                end_idx = int(end / self.clip_len)

            sentence = meta[1].rstrip()
            words_id, words_weight, unknown_mask, words_label = self.tokenizer.tokenize(
                sentence,
                max_valid_length=self.max_words_l,
            )
            semantic_fields = self._semantic_fields(
                line_id,
                video_id,
                sentence,
                semantic_by_id,
            )

            annotations.append(
                {
                    "video_id": video_id,
                    "duration": duration,
                    "moment": moment,
                    "sentence": sentence,
                    "words_id": words_id,
                    "words_weight": words_weight,
                    "unknown_mask": unknown_mask,
                    "words_label": words_label,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "qid": None if self.split == "train" else line_id,
                    "relevant_windows": None if self.split == "train" else [moment],
                    "ann_idx": line_id,
                    **semantic_fields,
                }
            )
        return annotations

    def __getitem__(self, index):
        item = super().__getitem__(index)
        meta = self.merged_data[index]

        ann_indices: List[int] = [int(value) for value in meta["ann_idx"]]
        local_index_by_ann = {
            ann_idx: local_idx for local_idx, ann_idx in enumerate(ann_indices)
        }
        competitor_local_indices: List[int] = []

        for local_idx, candidate_line_ids in enumerate(meta["level1_candidate_line_ids"]):
            if not self.enable_competitor_loss or not meta["phrase_valid"][local_idx]:
                competitor_local_indices.append(-1)
                continue

            candidates = [
                local_index_by_ann[candidate]
                for candidate in candidate_line_ids
                if candidate in local_index_by_ann
                and meta["phrase_valid"][local_index_by_ann[candidate]]
            ]
            if not candidates:
                competitor_local_indices.append(-1)
                continue

            sampled = int(torch.randint(len(candidates), size=(1,)).item())
            competitor_local_indices.append(candidates[sampled])

        item.update(
            {
                "ann_idx": ann_indices,
                "phrase_action_mask": meta["phrase_action_mask"],
                "phrase_object_mask": meta["phrase_object_mask"],
                "phrase_valid": [bool(value) for value in meta["phrase_valid"]],
                "competitor_local_idx": competitor_local_indices,
            }
        )
        return item


def phrase_collate(batch):
    """Extend the original MESM collate with phrase metadata."""
    batched_data = base_collate(batch)

    ann_indices: List[int] = []
    action_masks: List[torch.Tensor] = []
    object_masks: List[torch.Tensor] = []
    phrase_valid: List[bool] = []
    competitor_indices: List[int] = []

    offset = 0
    for item in batch:
        num_queries = int(item["num_clips"])
        ann_indices.extend(item["ann_idx"])
        action_masks.extend(item["phrase_action_mask"])
        object_masks.extend(item["phrase_object_mask"])
        phrase_valid.extend(item["phrase_valid"])

        for local_index in item["competitor_local_idx"]:
            competitor_indices.append(offset + local_index if local_index >= 0 else -1)
        offset += num_queries

    batched_data["ann_idx"] = ann_indices
    batched_data["phrase_action_mask"] = torch.stack(action_masks, dim=0)
    batched_data["phrase_object_mask"] = torch.stack(object_masks, dim=0)
    batched_data["phrase_valid"] = torch.as_tensor(phrase_valid, dtype=torch.bool)
    batched_data["competitor_index"] = torch.as_tensor(
        competitor_indices,
        dtype=torch.long,
    )
    return batched_data
