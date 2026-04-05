from __future__ import annotations

from pathlib import Path
from typing import Any

import csv
import json


def load_model_evaluation(*, eval_file: Path) -> dict[str, Any]:
    if not eval_file.exists():
        raise FileNotFoundError(f"Missing evaluation file: {eval_file}")
    return json.loads(eval_file.read_text(encoding="utf-8"))


def build_pair_rows(*, model_name: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in report.get("pair_metrics", []):
        rows.append(
            {
                "model_name": model_name,
                "split": report.get("split"),
                "pair_key": r.get("pair_key"),
                "balanced_accuracy": float(r.get("balanced_accuracy", 0.0)),
                "avg_margin": float(r.get("avg_margin", 0.0)),
                "temperature": float(r.get("temperature", 0.0)),
                "layer_source": r.get("layer_source"),
                "selected_layers": ",".join(str(x) for x in r.get("selected_layers", [])),
                "left_mean_p_left": float(r.get("left_mean_p_left", 0.0)),
                "right_mean_p_left": float(r.get("right_mean_p_left", 0.0)),
            }
        )
    return rows


def build_model_summary_row(*, model_name: str, report: dict[str, Any]) -> dict[str, Any]:
    agg = report.get("aggregate_metrics", {}) or {}
    overlap = report.get("overlap_confusion", []) or []
    overlap_acc = [float(x.get("mapped_accuracy", 0.0)) for x in overlap] or [0.0]
    return {
        "model_name": model_name,
        "split": report.get("split"),
        "n_pairs": int(agg.get("n_pairs", 0)),
        "mean_pair_balanced_accuracy": float(agg.get("mean_pair_balanced_accuracy", 0.0)),
        "min_pair_balanced_accuracy": float(agg.get("min_pair_balanced_accuracy", 0.0)),
        "max_pair_balanced_accuracy": float(agg.get("max_pair_balanced_accuracy", 0.0)),
        "mean_pair_avg_margin": float(agg.get("mean_pair_avg_margin", 0.0)),
        "mean_overlap_mapped_accuracy": float(sum(overlap_acc) / len(overlap_acc)),
        "overlap_checks": int(len(overlap)),
        "warnings_count": int(len(report.get("warnings", []))),
        "errors_count": int(len(report.get("errors", []))),
    }


def write_csv(*, output_file: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
