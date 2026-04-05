from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .probe_design import read_jsonl, resolve_jsonl_path, validate_records


@dataclass
class ResidualExtractionResult:
    ids: list[str]
    texts: list[str]
    residuals: Tensor  # shape: (prompt, layer, hidden_dim)


def batchify(items: list[Any], batch_size: int) -> list[list[Any]]:
    # Adapted from heretic.utils.batchify().
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def empty_cache() -> None:
    # Adapted from heretic.utils.empty_cache().
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def _build_chat_prompts(
    *,
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
) -> list[str]:
    chats = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        for text in texts
    ]
    prompts = tokenizer.apply_chat_template(
        chats,
        add_generation_prompt=True,
        tokenize=False,
    )
    return list(prompts)


@torch.no_grad()
def get_residuals(
    *,
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    system_prompt: str = "You are a helpful assistant.",
) -> Tensor:
    """
    Heretic-style residual extraction:
    - generate one token
    - read hidden states for first generated token
    - stack all layers as (prompt, layer, hidden_dim)
    """
    chat_prompts = _build_chat_prompts(
        texts=texts,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        output_hidden_states=True,
        return_dict_in_generate=True,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        raise RuntimeError(
            "Model generate() did not return hidden_states. "
            "Ensure output_hidden_states=True and return_dict_in_generate=True."
        )
    hidden_states = outputs.hidden_states[0]  # first generated token

    residuals = torch.stack(
        [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
        dim=1,
    )
    return residuals.to(torch.float32)


@torch.no_grad()
def get_residuals_batched(
    *,
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    batch_size: int,
    system_prompt: str = "You are a helpful assistant.",
) -> Tensor:
    batches = batchify(texts, batch_size)
    all_residuals: list[Tensor] = []
    for batch in batches:
        all_residuals.append(
            get_residuals(
                texts=batch,
                tokenizer=tokenizer,
                model=model,
                system_prompt=system_prompt,
            )
        )
    return torch.cat(all_residuals, dim=0) if all_residuals else torch.empty(0)


def load_side_jsonl(
    *,
    dataset_root: Path,
    pair_key: str,
    split: str,
    side: str,
    max_rows: int | None = None,
) -> tuple[list[str], list[str], Path]:
    file_path = resolve_jsonl_path(
        dataset_root=dataset_root,
        pair_key=pair_key,
        split=split,
        side=side,
    )
    if file_path is None:
        nested = dataset_root / pair_key / split / f"{side}.jsonl"
        flat = dataset_root / f"{pair_key}__{split}__{side}.jsonl"
        raise FileNotFoundError(
            f"Could not find dataset file for {pair_key}/{split}/{side}. "
            f"Tried: {nested} and {flat}"
        )
    records = read_jsonl(file_path)
    validate_records(records, file_path)
    if max_rows is not None:
        records = records[:max_rows]
    ids = [str(r["id"]) for r in records]
    texts = [str(r["text"]) for r in records]
    return ids, texts, file_path


def save_residual_artifact(
    *,
    output_file: Path,
    ids: list[str],
    texts: list[str],
    residuals: Tensor,
    meta: dict[str, Any],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ids": ids,
            "texts": texts,
            "residuals": residuals.cpu(),
            "meta": meta,
        },
        output_file,
    )

