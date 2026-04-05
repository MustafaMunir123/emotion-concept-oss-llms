from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .config import EmotionPair


@dataclass
class PairDatasetStats:
    pair_key: str
    split: str
    side: str
    path: str
    n_records: int
    avg_chars: float
    min_chars: int
    max_chars: int


def pair_key(pair: EmotionPair) -> str:
    return f"{pair.left}_vs_{pair.right}"


def candidate_paths(
    *,
    dataset_root: Path,
    pair_key: str,
    split: str,
    side: str,
) -> list[Path]:
    """
    Supported dataset layouts:
    1) Nested: ROOT/{pair_key}/{split}/{side}.jsonl
    2) Flat:   ROOT/{pair_key}__{split}__{side}.jsonl
    """
    return [
        dataset_root / pair_key / split / f"{side}.jsonl",
        dataset_root / f"{pair_key}__{split}__{side}.jsonl",
    ]


def resolve_jsonl_path(
    *,
    dataset_root: Path,
    pair_key: str,
    split: str,
    side: str,
) -> Path | None:
    for p in candidate_paths(
        dataset_root=dataset_root,
        pair_key=pair_key,
        split=split,
        side=side,
    ):
        if p.exists():
            return p
    return None


def build_probe_spec(pairs: list[EmotionPair]) -> dict[str, Any]:
    """Build Step-2 probe design contract (math + pair metadata)."""
    pair_specs: list[dict[str, Any]] = []
    for pair in pairs:
        left = pair.left
        right = pair.right
        key = pair_key(pair)
        pair_specs.append(
            {
                "pair_key": key,
                "left": left,
                "right": right,
                "directions": {
                    f"{left}_vs_{right}": {
                        "formula": f"normalize(mean({left}) - mean({right}))"
                    },
                    f"{right}_vs_{left}": {
                        "formula": f"-( {left}_vs_{right} )",
                        "note": "Implicit opposite direction.",
                    },
                },
                "score_rule": "cosine(residual, direction)",
                "percentage_rule": {
                    f"p_{left}": "sigmoid(k * score)",
                    f"p_{right}": f"1 - p_{left}",
                },
            }
        )

    return {
        "version": "step2-probe-design-v1",
        "required_record_fields": ["id", "text"],
        "dataset_layouts_supported": [
            "ROOT/{pair_key}/{split}/{side}.jsonl",
            "ROOT/{pair_key}__{split}__{side}.jsonl",
        ],
        "splits": ["train", "val", "test"],
        "pair_specs": pair_specs,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON on line {i}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}: line {i} is not a JSON object")
        records.append(row)
    return records


def validate_records(records: list[dict[str, Any]], path: Path) -> None:
    for i, row in enumerate(records, start=1):
        if "id" not in row:
            raise ValueError(f"{path}: record {i} missing required field 'id'")
        if "text" not in row:
            raise ValueError(f"{path}: record {i} missing required field 'text'")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"{path}: record {i} has empty/non-string 'text'")


def summarize_side(
    *,
    pair: EmotionPair,
    split: str,
    side: str,
    file_path: Path,
) -> PairDatasetStats:
    records = read_jsonl(file_path)
    validate_records(records, file_path)
    lengths = [len(r["text"]) for r in records]
    return PairDatasetStats(
        pair_key=pair_key(pair),
        split=split,
        side=side,
        path=str(file_path),
        n_records=len(records),
        avg_chars=float(mean(lengths)) if lengths else 0.0,
        min_chars=min(lengths) if lengths else 0,
        max_chars=max(lengths) if lengths else 0,
    )


def validate_pair_datasets(
    *,
    dataset_root: Path,
    pairs: list[EmotionPair],
    splits: list[str] | None = None,
) -> dict[str, Any]:
    """Validate presence and schema of JSONL files for each pair/split/side."""
    if splits is None:
        splits = ["train", "val", "test"]

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "validated": False,
        "errors": [],
        "stats": [],
    }

    for pair in pairs:
        key = pair_key(pair)
        for split in splits:
            for side in [pair.left, pair.right]:
                file_path = resolve_jsonl_path(
                    dataset_root=dataset_root,
                    pair_key=key,
                    split=split,
                    side=side,
                )
                if file_path is None:
                    nested = dataset_root / key / split / f"{side}.jsonl"
                    flat = dataset_root / f"{key}__{split}__{side}.jsonl"
                    report["errors"].append(
                        f"Missing file for {key}/{split}/{side}. "
                        f"Tried: {nested} and {flat}"
                    )
                    continue

                try:
                    stats = summarize_side(
                        pair=pair,
                        split=split,
                        side=side,
                        file_path=file_path,
                    )
                    report["stats"].append(stats.__dict__)
                except Exception as exc:  # noqa: PERF203
                    report["errors"].append(str(exc))

    report["validated"] = len(report["errors"]) == 0
    return report

