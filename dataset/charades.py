import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseDataset, collate as base_collate


_PHRASE_DATASET_CONFIG = {
    "enable_phrase_contrastive": False,
    "semantic_sidecar_dir": None,
    "semantic_sidecar_train_file": "charades_sta_train_semantic_final_v1_2.jsonl",
    "require_semantic_sidecar": True,
    "max_strict_negatives": 2,
}


def configure_phrase_contrastive_dataset(opt) -> None:
    """Configure Charades phrase metadata from parsed experiment options."""
    for key, default in tuple(_PHRASE_DATASET_CONFIG.items()):
        _PHRASE_DATASET_CONFIG[key] = getattr(opt, key, default)


"""
Charades-STA:
- CLIP image with clip_len=1: max_video_l = 194
- Slowfast with clip_len=1: max_video_l = 195
- Train dataset
    - CLIP text tokenizer: max_words_l = 16
    - min_clip_len = 1.68
    - max_clip_len = 80.8

- Test dataset
    - CLIP text tokenizer: max_words_l = 16
    - min_clip_len = 1.82
    - max_clip_len = 24.3
"""


class CharadesDataset(BaseDataset):
    def __init__(self, ann_path, feat_files, split,
                 use_tef, clip_len, max_words_l, max_video_l,
                 tokenizer_type, load_vocab_pkl, bpe_path, vocab,
                 normalize_video, contra_samples,
                 recfw, vocab_size, max_gather_size):
        super().__init__(ann_path, feat_files, split,
                         use_tef, clip_len, max_words_l, max_video_l,
                         tokenizer_type, load_vocab_pkl, bpe_path, vocab,
                         normalize_video, contra_samples,
                         recfw, vocab_size, max_gather_size)

    def load_annotations(self):
        durations = self._load_durations()
        split2filename = {
            "train": "charades_sta_train.txt",
            "test": "charades_sta_test.txt",
        }
        ann_file = os.path.join(self.ann_path, split2filename[self.split])
        annotations = []
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            for i in tqdm(range(len(lines)), desc=f"Load Charades {self.split} annotations"):
                meta = lines[i].split("##")
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
                words_id, words_weight, unknown_mask, words_label = \
                    self.tokenizer.tokenize(sentence, max_valid_length=self.max_words_l)

                data = {
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
                    "qid": None if self.split == "train" else i,
                    "relevant_windows": None if self.split == "train" else [moment],
                }
                annotations.append(data)

        return annotations

    def _load_durations(self):
        split2filename = {
            "train": "Charades_v1_train.csv",
            "val": "Charades_v1_test.csv",
            "test": "Charades_v1_test.csv",
        }
        ann_file = os.path.join(self.ann_path, split2filename[self.split])

        with open(ann_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            first_line_flag = True
            durations = dict()
            for row in csv_reader:
                if not first_line_flag:
                    durations[row[0]] = float(row[-1])
                first_line_flag = False

        return durations

    def get_video_feat(self, video_id):
        feats = []
        for feat_file in self.feat_files:
            with h5py.File(feat_file, 'r') as f:
                feat = f[video_id][:].astype(np.float32)
                if self.normalize_video:
                    feat = F.normalize(torch.from_numpy(feat), dim=1)
                feats.append(feat)
        min_len = min([len(e) for e in feats])
        feats = [e[:min_len] for e in feats]
        return torch.cat(feats, dim=1)


class PhraseContrastiveCharadesDataset(CharadesDataset):
    """Original Charades dataset with audited action/object masks and two negatives."""

    def __init__(self, ann_path, feat_files, split,
                 use_tef, clip_len, max_words_l, max_video_l,
                 tokenizer_type, load_vocab_pkl, bpe_path, vocab,
                 normalize_video, contra_samples,
                 recfw, vocab_size, max_gather_size):
        config = _PHRASE_DATASET_CONFIG
        self.enable_phrase_contrastive = bool(config["enable_phrase_contrastive"]) and split == "train"
        self.semantic_sidecar_dir = config["semantic_sidecar_dir"]
        self.semantic_sidecar_train_file = config["semantic_sidecar_train_file"]
        self.require_semantic_sidecar = bool(config["require_semantic_sidecar"])
        self.max_strict_negatives = int(config["max_strict_negatives"])

        if self.max_strict_negatives != 2:
            raise ValueError("This implementation requires max_strict_negatives=2.")
        if self.enable_phrase_contrastive and tokenizer_type != "CLIP":
            raise ValueError("Phrase contrastive learning requires tokenizer_type='CLIP'.")
        if self.enable_phrase_contrastive and max_words_l != 16:
            raise ValueError("The audited v1.2 sidecar requires max_words_l=16.")
        if self.enable_phrase_contrastive and max_gather_size > 0:
            raise ValueError(
                "Strict same-video negatives require max_gather_size=-1 so all queries "
                "from one video stay in the same dataset item."
            )

        super().__init__(
            ann_path, feat_files, split,
            use_tef, clip_len, max_words_l, max_video_l,
            tokenizer_type, load_vocab_pkl, bpe_path, vocab,
            normalize_video, contra_samples,
            recfw, vocab_size, max_gather_size,
        )

    @staticmethod
    def _normalize_sentence(sentence: str) -> str:
        return " ".join(sentence.lower().strip().split())

    def _load_semantic_sidecar(self):
        if not self.enable_phrase_contrastive:
            return {}
        if not self.semantic_sidecar_dir:
            sidecar_path = None
        else:
            sidecar_path = Path(self.semantic_sidecar_dir) / self.semantic_sidecar_train_file

        if sidecar_path is None or not sidecar_path.is_file():
            if self.require_semantic_sidecar:
                expected = sidecar_path or Path("<semantic_sidecar_dir is not configured>")
                raise FileNotFoundError(f"Missing Charades semantic sidecar: {expected}")
            return {}

        records = {}
        with sidecar_path.open("r", encoding="utf-8") as handle:
            for jsonl_line, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                line_id = int(record["line_id"])
                if line_id in records:
                    raise ValueError(
                        f"Duplicate line_id={line_id} in {sidecar_path} at line {jsonl_line}."
                    )
                records[line_id] = record
        return records

    def _semantic_fields(self, line_id, video_id, sentence, semantic_by_id):
        empty = {
            "phrase_action_mask": torch.zeros(self.max_words_l, dtype=torch.bool),
            "phrase_object_mask": torch.zeros(self.max_words_l, dtype=torch.bool),
            "phrase_valid": False,
            "level1_candidate_line_ids": [],
        }
        if line_id not in semantic_by_id:
            if self.require_semantic_sidecar:
                raise KeyError(f"Semantic sidecar has no record for line_id={line_id}.")
            return empty

        semantic = semantic_by_id[line_id]
        if semantic.get("video_id") != video_id:
            raise ValueError(
                f"Sidecar video mismatch at line_id={line_id}: "
                f"{semantic.get('video_id')} != {video_id}."
            )
        if self._normalize_sentence(semantic.get("sentence", "")) != self._normalize_sentence(sentence):
            raise ValueError(
                f"Sidecar sentence mismatch at line_id={line_id}; regenerate from the same annotation file."
            )

        alignment = semantic.get("clip_alignment", {})
        phrase_valid = bool(
            semantic.get("phrase_valid_model", False)
            and alignment.get("status") == "aligned"
        )
        action_mask = torch.zeros(self.max_words_l, dtype=torch.bool)
        object_mask = torch.zeros(self.max_words_l, dtype=torch.bool)

        for raw_position in alignment.get("action_bpe_positions") or []:
            position = int(raw_position)
            if 0 <= position < self.max_words_l:
                action_mask[position] = True
            else:
                phrase_valid = False
        for raw_position in alignment.get("object_bpe_positions") or []:
            position = int(raw_position)
            if 0 <= position < self.max_words_l:
                object_mask[position] = True
            else:
                phrase_valid = False

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
        if not self.enable_phrase_contrastive:
            return super().load_annotations()

        durations = self._load_durations()
        ann_file = os.path.join(self.ann_path, "charades_sta_train.txt")
        semantic_by_id = self._load_semantic_sidecar()
        annotations = []

        with open(ann_file, "r", encoding="utf-8") as handle:
            lines = handle.readlines()

        for line_id in tqdm(
            range(len(lines)),
            desc="Load Charades train annotations with phrase metadata",
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
                    "qid": None,
                    "relevant_windows": None,
                    "ann_idx": line_id,
                    **semantic_fields,
                }
            )
        return annotations

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if not self.enable_phrase_contrastive:
            return item

        meta = self.merged_data[index]
        ann_indices = [int(value) for value in meta["ann_idx"]]
        local_index_by_ann = {
            ann_idx: local_idx for local_idx, ann_idx in enumerate(ann_indices)
        }
        negative_indices = torch.full(
            (len(ann_indices), self.max_strict_negatives),
            fill_value=-1,
            dtype=torch.long,
        )
        negative_mask = torch.zeros_like(negative_indices, dtype=torch.bool)

        for local_idx, candidate_line_ids in enumerate(meta["level1_candidate_line_ids"]):
            if not bool(meta["phrase_valid"][local_idx]):
                continue
            candidates = sorted(
                {
                    local_index_by_ann[int(candidate)]
                    for candidate in candidate_line_ids
                    if int(candidate) in local_index_by_ann
                    and int(candidate) != ann_indices[local_idx]
                    and bool(meta["phrase_valid"][local_index_by_ann[int(candidate)]])
                }
            )
            if not candidates:
                continue
            if len(candidates) > self.max_strict_negatives:
                order = torch.randperm(len(candidates))[:self.max_strict_negatives].tolist()
                candidates = [candidates[position] for position in order]
            count = min(len(candidates), self.max_strict_negatives)
            negative_indices[local_idx, :count] = torch.as_tensor(candidates[:count])
            negative_mask[local_idx, :count] = True

        item.update(
            {
                "ann_idx": ann_indices,
                "phrase_action_mask": meta["phrase_action_mask"],
                "phrase_object_mask": meta["phrase_object_mask"],
                "phrase_valid": [bool(value) for value in meta["phrase_valid"]],
                "strict_negative_local_indices": negative_indices,
                "strict_negative_mask": negative_mask,
            }
        )
        return item


def collate_phrase_contrastive(batch):
    """Use the original collate and add flattened phrase metadata when present."""
    batched_data = base_collate(batch)
    if not batch or "phrase_action_mask" not in batch[0]:
        return batched_data

    ann_indices = []
    action_masks = []
    object_masks = []
    phrase_valid = []
    negative_indices = []
    negative_masks = []
    offset = 0

    for item in batch:
        num_queries = int(item["num_clips"])
        ann_indices.extend(item["ann_idx"])
        action_masks.extend(item["phrase_action_mask"])
        object_masks.extend(item["phrase_object_mask"])
        phrase_valid.extend(item["phrase_valid"])

        local_indices = item["strict_negative_local_indices"]
        local_mask = item["strict_negative_mask"]
        global_indices = torch.where(
            local_mask,
            local_indices + offset,
            torch.full_like(local_indices, -1),
        )
        negative_indices.append(global_indices)
        negative_masks.append(local_mask)
        offset += num_queries

    batched_data["ann_idx"] = ann_indices
    batched_data["phrase_action_mask"] = torch.stack(action_masks, dim=0)
    batched_data["phrase_object_mask"] = torch.stack(object_masks, dim=0)
    batched_data["phrase_valid"] = torch.as_tensor(phrase_valid, dtype=torch.bool)
    batched_data["strict_negative_indices"] = torch.cat(negative_indices, dim=0)
    batched_data["strict_negative_mask"] = torch.cat(negative_masks, dim=0)
    return batched_data


# max_video_l = 0
# with h5py.File(feat_file, 'r') as f:
#     for id in tqdm(f.keys()):
#         feat = f[id][:]
#         max_video_l = max(max_video_l, feat.shape[0])
# print(max_video_l)
