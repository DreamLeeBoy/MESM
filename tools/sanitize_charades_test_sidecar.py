#!/usr/bin/env python3
"""Remove temporal/candidate fields from a text-derived Charades test sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def sanitize_record(record):
    alignment = record.get("clip_alignment", {})
    return {
        "line_id": int(record["line_id"]),
        "video_id": record["video_id"],
        "sentence": record["sentence"],
        "primary_unit": record.get("primary_unit"),
        "phrase_valid_semantic": bool(record.get("phrase_valid_semantic", False)),
        "phrase_valid_model": bool(record.get("phrase_valid_model", False)),
        "parse_confidence": record.get("parse_confidence"),
        "failure_reason": record.get("failure_reason"),
        "clip_alignment": {
            "status": alignment.get("status", "not_applicable"),
            "action_bpe_positions": alignment.get("action_bpe_positions") or [],
            "object_bpe_positions": alignment.get("object_bpe_positions") or [],
        },
        "level1_candidate_line_ids": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    valid = 0
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            record = sanitize_record(json.loads(line))
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            valid += int(record["phrase_valid_model"])

    print(f"Wrote {count} records to {args.output}")
    print(f"CLIP-aligned phrase records: {valid}")


if __name__ == "__main__":
    main()
