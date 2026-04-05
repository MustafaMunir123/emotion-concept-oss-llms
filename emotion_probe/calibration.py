from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def aggregate_scores_by_layers(*, per_layer_scores: Tensor, selected_layers: list[int]) -> Tensor:
    """
    Aggregate per-layer scores to one score per prompt.
    per_layer_scores: (prompt, layer)
    """
    if per_layer_scores.ndim != 2:
        raise ValueError("per_layer_scores must be shape (prompt, layer).")
    if not selected_layers:
        raise ValueError("selected_layers cannot be empty.")
    idx = torch.tensor(selected_layers, dtype=torch.long)
    return per_layer_scores.index_select(dim=1, index=idx).mean(dim=1)


def build_labeled_pair_scores(*, left_scores: Tensor, right_scores: Tensor) -> tuple[Tensor, Tensor]:
    """
    Build binary labels for one pair:
    - left side samples -> label 1
    - right side samples -> label 0
    """
    if left_scores.ndim != 1 or right_scores.ndim != 1:
        raise ValueError("left_scores and right_scores must be 1D.")
    scores = torch.cat([left_scores, right_scores], dim=0).to(torch.float32)
    labels = torch.cat(
        [
            torch.ones_like(left_scores, dtype=torch.float32),
            torch.zeros_like(right_scores, dtype=torch.float32),
        ],
        dim=0,
    )
    return scores, labels


def binary_nll_from_scores(*, scores: Tensor, labels: Tensor, temperature: float) -> float:
    """
    Negative log-likelihood for p(left)=sigmoid(k * score), labels in {0,1}.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape.")
    logits = temperature * scores.to(torch.float32)
    y = labels.to(torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    return float(loss.item())


def expected_calibration_error(
    *,
    probs: Tensor,
    labels: Tensor,
    n_bins: int = 10,
) -> float:
    """
    Simple ECE for binary probabilities.
    """
    if probs.shape != labels.shape:
        raise ValueError("probs and labels must have the same shape.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    p = probs.to(torch.float32).clamp(0.0, 1.0)
    y = labels.to(torch.float32)
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    total = p.numel()
    ece = 0.0
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum().item())
        if count == 0:
            continue
        bin_acc = float(y[mask].mean().item())
        bin_conf = float(p[mask].mean().item())
        ece += (count / total) * abs(bin_acc - bin_conf)
    return float(ece)


def sweep_temperature_grid(
    *,
    scores: Tensor,
    labels: Tensor,
    temperatures: list[float],
    ece_bins: int = 10,
) -> dict[str, Any]:
    if not temperatures:
        raise ValueError("temperatures cannot be empty.")

    rows: list[dict[str, float]] = []
    best_temp = None
    best_nll = float("inf")
    best_ece = float("inf")
    best_brier = float("inf")

    for t in temperatures:
        nll = binary_nll_from_scores(scores=scores, labels=labels, temperature=t)
        probs = torch.sigmoid(float(t) * scores.to(torch.float32))
        y = labels.to(torch.float32)
        brier = float(torch.mean((probs - y) ** 2).item())
        ece = expected_calibration_error(probs=probs, labels=y, n_bins=ece_bins)
        rows.append(
            {
                "temperature": float(t),
                "nll": float(nll),
                "brier": float(brier),
                "ece": float(ece),
            }
        )
        if nll < best_nll or (nll == best_nll and ece < best_ece):
            best_nll = nll
            best_ece = ece
            best_brier = brier
            best_temp = float(t)

    return {
        "best_temperature": float(best_temp),
        "best_nll": float(best_nll),
        "best_brier": float(best_brier),
        "best_ece": float(best_ece),
        "grid_metrics": rows,
    }
