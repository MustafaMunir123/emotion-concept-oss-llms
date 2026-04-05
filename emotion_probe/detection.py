from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor


def load_residual_artifact(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "residuals" not in payload:
        raise ValueError(f"{path} is not a valid residual artifact.")
    residuals = payload["residuals"]
    if not isinstance(residuals, Tensor) or residuals.ndim != 3:
        raise ValueError(f"{path} residuals must be a 3D tensor (prompt, layer, hidden).")
    return payload


def _normalize_last_dim(x: Tensor, eps: float) -> Tensor:
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def build_pair_probe_from_residuals(
    *,
    left_residuals: Tensor,
    right_residuals: Tensor,
    left_label: str,
    right_label: str,
    pair_key: str,
    split: str,
    model_name: str,
    model_id: str,
    dtype_name: str | None,
    token_policy: str,
    layer_policy: str,
    eps: float = 1e-8,
) -> dict[str, Any]:
    if left_residuals.ndim != 3 or right_residuals.ndim != 3:
        raise ValueError("Both residual tensors must be shape (prompt, layer, hidden).")
    if left_residuals.shape[1:] != right_residuals.shape[1:]:
        raise ValueError(
            "Left/right residual tensor shapes are incompatible: "
            f"{tuple(left_residuals.shape)} vs {tuple(right_residuals.shape)}"
        )

    left_mean = left_residuals.to(torch.float32).mean(dim=0)   # (layer, hidden)
    right_mean = right_residuals.to(torch.float32).mean(dim=0)  # (layer, hidden)
    direction_raw = left_mean - right_mean
    direction = _normalize_last_dim(direction_raw, eps=eps)
    per_layer_norm = direction_raw.norm(dim=-1)
    degenerate_layers = int((per_layer_norm <= eps).sum().item())

    return {
        "probe_direction": direction.cpu(),
        "probe_direction_raw": direction_raw.cpu(),
        "left_mean": left_mean.cpu(),
        "right_mean": right_mean.cpu(),
        "meta": {
            "pair_key": pair_key,
            "left_label": left_label,
            "right_label": right_label,
            "split": split,
            "model_name": model_name,
            "model_id": model_id,
            "dtype": dtype_name,
            "token_policy": token_policy,
            "layer_policy": layer_policy,
            "left_n": int(left_residuals.shape[0]),
            "right_n": int(right_residuals.shape[0]),
            "n_layers": int(left_residuals.shape[1]),
            "hidden_size": int(left_residuals.shape[2]),
            "degenerate_layer_count": degenerate_layers,
            "min_raw_norm": float(per_layer_norm.min().item()),
            "max_raw_norm": float(per_layer_norm.max().item()),
            "mean_raw_norm": float(per_layer_norm.mean().item()),
        },
    }


def save_probe_artifact(*, output_file: Path, artifact: dict[str, Any]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_file)


def pair_score(
    *,
    residuals: Tensor,
    probe_direction: Tensor,
    layer_reduction: Literal["mean"] = "mean",
    eps: float = 1e-8,
) -> Tensor:
    if residuals.ndim != 3:
        raise ValueError("residuals must be shape (prompt, layer, hidden).")
    if probe_direction.ndim != 2:
        raise ValueError("probe_direction must be shape (layer, hidden).")
    if residuals.shape[1:] != probe_direction.shape:
        raise ValueError(
            "Residuals and probe_direction are incompatible: "
            f"{tuple(residuals.shape)} vs {tuple(probe_direction.shape)}"
        )
    if layer_reduction != "mean":
        raise ValueError(f"Unsupported layer_reduction: {layer_reduction!r}")

    x = F.normalize(residuals.to(torch.float32), p=2, dim=-1, eps=eps)
    v = F.normalize(probe_direction.to(torch.float32), p=2, dim=-1, eps=eps)
    per_layer_scores = (x * v.unsqueeze(0)).sum(dim=-1)  # (prompt, layer)
    return per_layer_scores.mean(dim=1)


def pair_percentages(scores: Tensor, *, temperature: float = 1.0) -> tuple[Tensor, Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    p_left = torch.sigmoid(temperature * scores.to(torch.float32))
    p_right = 1.0 - p_left
    return p_left, p_right


def triggered_side(
    *,
    p_left: Tensor,
    p_right: Tensor,
    left_label: str,
    right_label: str,
) -> tuple[list[str], Tensor]:
    if p_left.shape != p_right.shape:
        raise ValueError("p_left and p_right shape mismatch.")
    mask = (p_left >= p_right).tolist()
    labels = [left_label if bool(m) else right_label for m in mask]
    margin = (p_left - p_right).abs()
    return labels, margin
