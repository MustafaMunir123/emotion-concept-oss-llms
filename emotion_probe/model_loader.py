from __future__ import annotations

from typing import Any, Type

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import Settings


def get_model_class(
    model_id: str,
) -> Type[AutoModelForImageTextToText] | Type[AutoModelForCausalLM]:
    configs = PretrainedConfig.get_config_dict(model_id)
    if any([("vision_config" in config) for config in configs]):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def _resolve_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def load_tokenizer(model_id: str, settings: Settings) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=settings.trust_remote_code,
        use_fast=settings.use_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_with_fallback_dtypes(
    model_id: str,
    settings: Settings,
) -> tuple[PreTrainedModel, str]:
    model_cls = get_model_class(model_id)
    last_error: Exception | None = None

    for dtype_name in settings.dtypes:
        try:
            torch_dtype = _resolve_dtype(dtype_name)
            model = model_cls.from_pretrained(
                model_id,
                dtype=torch_dtype,
                device_map=settings.device_map,
                trust_remote_code=settings.trust_remote_code,
            )
            return model, dtype_name
        except Exception as error:  # noqa: PERF203 - keep clarity
            last_error = error
            continue

    raise RuntimeError(f"Failed to load {model_id}: {last_error}")


@torch.no_grad()
def smoke_test_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str = "What is 1+1?",
    max_new_tokens: int = 1,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

