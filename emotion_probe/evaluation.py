from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def probabilities_from_scores(*, scores: Tensor, temperature: float) -> tuple[Tensor, Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    p_left = torch.sigmoid(float(temperature) * scores.to(torch.float32))
    p_right = 1.0 - p_left
    return p_left, p_right


def classify_from_probabilities(
    *,
    p_left: Tensor,
    p_right: Tensor,
    left_label: str,
    right_label: str,
) -> tuple[list[str], Tensor]:
    if p_left.shape != p_right.shape:
        raise ValueError("p_left and p_right must have same shape.")
    labels = [left_label if bool(x) else right_label for x in (p_left >= p_right).tolist()]
    margin = (p_left - p_right).abs()
    return labels, margin


def evaluate_pair_predictions(
    *,
    left_scores: Tensor,
    right_scores: Tensor,
    left_label: str,
    right_label: str,
    temperature: float,
) -> dict[str, Any]:
    if left_scores.ndim != 1 or right_scores.ndim != 1:
        raise ValueError("left_scores and right_scores must be 1D.")

    left_p_left, left_p_right = probabilities_from_scores(
        scores=left_scores,
        temperature=temperature,
    )
    right_p_left, right_p_right = probabilities_from_scores(
        scores=right_scores,
        temperature=temperature,
    )

    left_pred, left_margin = classify_from_probabilities(
        p_left=left_p_left,
        p_right=left_p_right,
        left_label=left_label,
        right_label=right_label,
    )
    right_pred, right_margin = classify_from_probabilities(
        p_left=right_p_left,
        p_right=right_p_right,
        left_label=left_label,
        right_label=right_label,
    )

    left_correct_rate = float(sum(1 for x in left_pred if x == left_label) / len(left_pred))
    right_correct_rate = float(sum(1 for x in right_pred if x == right_label) / len(right_pred))
    balanced_accuracy = 0.5 * (left_correct_rate + right_correct_rate)

    return {
        "left_correct_rate": left_correct_rate,
        "right_correct_rate": right_correct_rate,
        "balanced_accuracy": float(balanced_accuracy),
        "left_avg_margin": float(left_margin.mean().item()),
        "right_avg_margin": float(right_margin.mean().item()),
        "avg_margin": float(torch.cat([left_margin, right_margin]).mean().item()),
        "left_mean_p_left": float(left_p_left.mean().item()),
        "right_mean_p_left": float(right_p_left.mean().item()),
        "left_pred_counts": {
            left_label: int(sum(1 for x in left_pred if x == left_label)),
            right_label: int(sum(1 for x in left_pred if x == right_label)),
        },
        "right_pred_counts": {
            left_label: int(sum(1 for x in right_pred if x == left_label)),
            right_label: int(sum(1 for x in right_pred if x == right_label)),
        },
    }


def evaluate_mapped_cross_confusion(
    *,
    source_left_scores_on_target_probe: Tensor,
    source_right_scores_on_target_probe: Tensor,
    target_left_label: str,
    target_right_label: str,
    expected_label_for_source_left: str,
    expected_label_for_source_right: str,
    temperature: float,
) -> dict[str, Any]:
    if source_left_scores_on_target_probe.ndim != 1 or source_right_scores_on_target_probe.ndim != 1:
        raise ValueError("source_*_scores_on_target_probe must be 1D.")
    if expected_label_for_source_left not in (target_left_label, target_right_label):
        raise ValueError("expected_label_for_source_left must be one of target labels.")
    if expected_label_for_source_right not in (target_left_label, target_right_label):
        raise ValueError("expected_label_for_source_right must be one of target labels.")

    left_p_left, left_p_right = probabilities_from_scores(
        scores=source_left_scores_on_target_probe,
        temperature=temperature,
    )
    right_p_left, right_p_right = probabilities_from_scores(
        scores=source_right_scores_on_target_probe,
        temperature=temperature,
    )
    left_pred, _ = classify_from_probabilities(
        p_left=left_p_left,
        p_right=left_p_right,
        left_label=target_left_label,
        right_label=target_right_label,
    )
    right_pred, _ = classify_from_probabilities(
        p_left=right_p_left,
        p_right=right_p_right,
        left_label=target_left_label,
        right_label=target_right_label,
    )

    left_map_acc = float(
        sum(1 for x in left_pred if x == expected_label_for_source_left) / len(left_pred)
    )
    right_map_acc = float(
        sum(1 for x in right_pred if x == expected_label_for_source_right) / len(right_pred)
    )
    mapped_accuracy = 0.5 * (left_map_acc + right_map_acc)

    return {
        "target_labels": [target_left_label, target_right_label],
        "expected_map": {
            "source_left_expected_target_label": expected_label_for_source_left,
            "source_right_expected_target_label": expected_label_for_source_right,
        },
        "source_left_mapped_rate": left_map_acc,
        "source_right_mapped_rate": right_map_acc,
        "mapped_accuracy": float(mapped_accuracy),
        "source_left_mean_p_target_left": float(left_p_left.mean().item()),
        "source_right_mean_p_target_left": float(right_p_left.mean().item()),
    }
