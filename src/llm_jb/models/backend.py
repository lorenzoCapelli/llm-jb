"""Model loading backends, dispatched on `ModelConfig.backend`.

TransformerLens is implemented: it's the only backend the CPU/1-GPU dev
workflow (gpt2-small, Llama-3.2-1B) needs. nnsight is a documented stub
for when a model is too large to eager-load the TransformerLens way —
implement it when that's actually needed, not before.

`HookedTransformer.from_pretrained` raises a DeprecationWarning in
transformer_lens 3.8 pointing at `TransformerBridge.boot_transformers(...)`
+ `enable_compatibility_mode()`. Not switched to yet: TransformerBridge
isn't even re-exported from the top-level package in this version (still
mid-migration, undocumented), while HookedTransformer remains what nearly
all current activation-patching/SAE tooling and tutorials target. Revisit
once TransformerBridge stabilizes.
"""

from __future__ import annotations

from typing import Any

import torch
from transformer_lens import HookedTransformer

from llm_jb.models.config import ModelConfig


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model_transformer_lens(config: ModelConfig) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        config.hf_repo_id,
        device=resolve_device(config.device),
        dtype=config.dtype,
    )
    model.eval()
    return model


def load_model_nnsight(config: ModelConfig) -> Any:
    """Not implemented. `nnsight` is intentionally not a dependency of
    this repo yet — install it only once a model actually needs it:

        pip install nnsight
    """
    raise NotImplementedError(
        "nnsight backend not implemented yet. It's reserved for models too "
        "large for HookedTransformer's eager-load approach; implement this "
        f"when '{config.name}' actually needs it."
    )


def load_model(config: ModelConfig) -> Any:
    if config.backend == "transformer_lens":
        return load_model_transformer_lens(config)
    if config.backend == "nnsight":
        return load_model_nnsight(config)
    raise ValueError(f"unknown backend: {config.backend}")
