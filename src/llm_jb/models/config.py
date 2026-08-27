from __future__ import annotations

from typing import Literal

from pydantic_settings import SettingsConfigDict

from llm_jb.config import YamlSettings

Backend = Literal["transformer_lens", "nnsight"]


class ModelConfig(YamlSettings):
    model_config = SettingsConfigDict(extra="forbid", env_prefix="LLM_JB_MODEL_")

    name: str
    hf_repo_id: str
    backend: Backend = "transformer_lens"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:N" — "auto" picks
    # the single GPU CUDA_VISIBLE_DEVICES exposes to this process, or CPU
    dtype: str = "float32"
