from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
)


class ModelSpec(BaseModel):
    name: str = Field(description="Human-readable model label.")
    model_id: str = Field(description="Hugging Face model identifier.")
    enabled: bool = Field(default=True, description="Whether to include in runs.")


class EmotionPair(BaseModel):
    left: str
    right: str


class Settings(BaseSettings):
    project_root: str = Field(default=".")
    output_root: str = Field(default="outputs")
    data_dir: str = Field(default="data")
    results_dir: str = Field(default="results")
    plots_dir: str = Field(default="plots")
    logs_dir: str = Field(default="logs")
    seed: int = Field(default=42)
    run_id: str = Field(default="baseline")

    models: list[ModelSpec] = Field(
        default=[
            ModelSpec(name="qwen_4b", model_id="Qwen/Qwen3-4B-Instruct-2507"),
            ModelSpec(name="mistral_7b", model_id="mistralai/Mistral-7B-Instruct-v0.3"),
            ModelSpec(name="falcon_7b", model_id="tiiuae/falcon-7b-instruct"),
            ModelSpec(name="zephyr_7b", model_id="HuggingFaceH4/zephyr-7b-beta"),
            ModelSpec(name="openchat_7b", model_id="openchat/openchat_3.5"),
        ]
    )
    selected_model: str | None = Field(
        default="qwen_4b",
        description=(
            "Optional single-model selector. Can match either ModelSpec.name "
            "or ModelSpec.model_id. If set, only the matching model runs."
        ),
    )

    device_map: str = Field(default="auto")
    dtypes: list[str] = Field(default=["auto", "float16", "bfloat16", "float32"])
    trust_remote_code: bool | None = Field(default=None)
    max_new_tokens_smoke_test: int = Field(default=1)
    use_fast_tokenizer: bool = Field(default=True)

    emotion_pairs: list[EmotionPair] = Field(
        default=[
            EmotionPair(left="sad", right="happy"),
            EmotionPair(left="angry", right="calm"),
            EmotionPair(left="fear", right="confidence"),
            EmotionPair(left="love", right="hate"),
            EmotionPair(left="anxious", right="relaxed"),
        ]
    )

    execution_mode: Literal["single_notebook"] = Field(default="single_notebook")

    @property
    def enabled_models(self) -> list[ModelSpec]:
        enabled = [m for m in self.models if m.enabled]
        if self.selected_model is None:
            return enabled

        selected = self.selected_model
        filtered = [m for m in enabled if m.name == selected or m.model_id == selected]
        if filtered:
            return filtered

        choices = ", ".join([f"{m.name} ({m.model_id})" for m in enabled])
        raise ValueError(
            f"selected_model={selected!r} did not match any enabled model. "
            f"Available enabled models: {choices}"
        )

    @property
    def project_path(self) -> Path:
        return Path(self.project_root).resolve()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            EnvSettingsSource(settings_cls, env_prefix="EMOTION_PROBE_"),
            dotenv_settings,
            file_secret_settings,
        )

