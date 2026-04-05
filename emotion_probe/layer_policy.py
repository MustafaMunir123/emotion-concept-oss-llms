from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LayerSweepSummary:
    per_layer_accuracy: Tensor
    per_layer_separation: Tensor
    left_positive_rate: Tensor
    right_negative_rate: Tensor


def per_layer_pair_scores(
    *,
    residuals: Tensor,
    probe_direction: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute per-layer cosine projections.
    residuals: (prompt, layer, hidden)
    probe_direction: (layer, hidden)
    Returns: (prompt, layer)
    """
    if residuals.ndim != 3:
        raise ValueError("residuals must be shape (prompt, layer, hidden).")
    if probe_direction.ndim != 2:
        raise ValueError("probe_direction must be shape (layer, hidden).")
    if residuals.shape[1:] != probe_direction.shape:
        raise ValueError(
            "Residuals and probe_direction shapes do not match: "
            f"{tuple(residuals.shape)} vs {tuple(probe_direction.shape)}"
        )

    x = F.normalize(residuals.to(torch.float32), p=2, dim=-1, eps=eps)
    v = F.normalize(probe_direction.to(torch.float32), p=2, dim=-1, eps=eps)
    return (x * v.unsqueeze(0)).sum(dim=-1)


def summarize_layer_sweep(*, left_scores: Tensor, right_scores: Tensor) -> LayerSweepSummary:
    """
    left_scores/right_scores: (prompt, layer)
    Orientation assumption:
    - left prompts should produce positive scores
    - right prompts should produce negative scores
    """
    if left_scores.ndim != 2 or right_scores.ndim != 2:
        raise ValueError("left_scores and right_scores must be shape (prompt, layer).")
    if left_scores.shape[1] != right_scores.shape[1]:
        raise ValueError(
            "left_scores and right_scores must have same n_layers. "
            f"Got {tuple(left_scores.shape)} vs {tuple(right_scores.shape)}"
        )

    left_positive_rate = (left_scores > 0).float().mean(dim=0)
    right_negative_rate = (right_scores < 0).float().mean(dim=0)
    per_layer_accuracy = 0.5 * (left_positive_rate + right_negative_rate)
    per_layer_separation = left_scores.mean(dim=0) - right_scores.mean(dim=0)
    return LayerSweepSummary(
        per_layer_accuracy=per_layer_accuracy,
        per_layer_separation=per_layer_separation,
        left_positive_rate=left_positive_rate,
        right_negative_rate=right_negative_rate,
    )


def best_single_layer_index(per_layer_accuracy: Tensor) -> int:
    if per_layer_accuracy.ndim != 1:
        raise ValueError("per_layer_accuracy must be 1D.")
    return int(torch.argmax(per_layer_accuracy).item())


def best_contiguous_band(
    *,
    per_layer_accuracy: Tensor,
    band_width: int,
) -> dict[str, Any]:
    """
    Returns best contiguous layer interval [start, end] maximizing mean accuracy.
    """
    if per_layer_accuracy.ndim != 1:
        raise ValueError("per_layer_accuracy must be 1D.")
    n_layers = int(per_layer_accuracy.shape[0])
    if n_layers == 0:
        raise ValueError("per_layer_accuracy is empty.")
    if band_width <= 0:
        raise ValueError("band_width must be positive.")
    width = min(int(band_width), n_layers)

    best_start = 0
    best_mean = float("-inf")
    for start in range(0, n_layers - width + 1):
        end = start + width
        mean_acc = float(per_layer_accuracy[start:end].mean().item())
        if mean_acc > best_mean:
            best_mean = mean_acc
            best_start = start

    best_end = best_start + width - 1
    return {
        "start": best_start,
        "end": best_end,
        "width": width,
        "mean_accuracy": best_mean,
    }


def evaluate_policy_accuracy(
    *,
    left_scores: Tensor,
    right_scores: Tensor,
    selected_layers: list[int],
) -> dict[str, float]:
    if not selected_layers:
        raise ValueError("selected_layers cannot be empty.")
    idx = torch.tensor(selected_layers, dtype=torch.long)
    left_agg = left_scores.index_select(dim=1, index=idx).mean(dim=1)
    right_agg = right_scores.index_select(dim=1, index=idx).mean(dim=1)

    left_rate = float((left_agg > 0).float().mean().item())
    right_rate = float((right_agg < 0).float().mean().item())
    return {
        "left_expected_side_rate": left_rate,
        "right_expected_side_rate": right_rate,
        "balanced_accuracy": 0.5 * (left_rate + right_rate),
        "mean_separation": float(left_agg.mean().item() - right_agg.mean().item()),
    }


def choose_global_layer_with_tiebreak(
    *,
    layer_votes: dict[int, int],
    layer_stats: dict[int, dict[str, float]],
) -> dict[str, Any]:
    """
    Deterministic winner selection with tie-breaks:
    1) highest votes
    2) higher avg balanced accuracy
    3) higher avg separation
    4) lower layer index
    """
    if not layer_votes:
        raise ValueError("layer_votes is empty.")

    for layer_idx in layer_votes:
        if layer_idx not in layer_stats:
            raise ValueError(f"Missing layer_stats for layer {layer_idx}.")
        stats = layer_stats[layer_idx]
        if "avg_accuracy" not in stats or "avg_separation" not in stats:
            raise ValueError(
                f"layer_stats[{layer_idx}] must include avg_accuracy and avg_separation."
            )

    ranking = sorted(
        layer_votes.keys(),
        key=lambda layer: (
            int(layer_votes[layer]),
            float(layer_stats[layer]["avg_accuracy"]),
            float(layer_stats[layer]["avg_separation"]),
            -int(layer),
        ),
        reverse=True,
    )
    winner = int(ranking[0])
    return {
        "winner": winner,
        "ranking": ranking,
        "winner_stats": {
            "votes": int(layer_votes[winner]),
            "avg_accuracy": float(layer_stats[winner]["avg_accuracy"]),
            "avg_separation": float(layer_stats[winner]["avg_separation"]),
        },
    }
