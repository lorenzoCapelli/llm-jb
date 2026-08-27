"""Selective residual-stream capture: hook only the layers requested, and
reduce to the chosen anchor position(s) *inside* the hook — so what gets
retained is (batch, d_model) per hooked layer, never the full (batch,
seq, d_model) that `run_with_cache` would keep for every hook point.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer

from llm_jb.data.alignment import extract_anchor_activations
from llm_jb.data.types import AnchorMode, TokenizedSpan
from llm_jb.hooks.storage import ActivationPlacement, place_activation


def resid_post_hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


@dataclass(frozen=True)
class CapturedActivations:
    """Per-layer anchor activations for one batch.

    `activations[layer]` has shape (batch, d_model) — already reduced to
    the chosen anchor, with explicit, uniform dtype across layers.
    """

    layers: list[int]
    activations: dict[int, torch.Tensor]
    dtype: torch.dtype
    d_model: int
    batch_size: int


def capture_residual_stream(
    model: HookedTransformer,
    tokens: torch.Tensor,
    spans: Sequence[TokenizedSpan],
    layers: Sequence[int],
    anchor_mode: AnchorMode = AnchorMode.LAST_PROMPT_POSITION,
    k: int = 1,
    placement: ActivationPlacement = "gpu",
) -> CapturedActivations:
    if tokens.shape[0] != len(spans):
        raise ValueError(
            f"tokens batch size ({tokens.shape[0]}) must match number of spans ({len(spans)})"
        )
    if not layers:
        raise ValueError("layers must be non-empty")

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
            anchored = extract_anchor_activations(activation, spans, mode=anchor_mode, k=k)
            captured[layer] = place_activation(anchored.detach(), placement)
            return activation

        return hook_fn

    fwd_hooks = [(resid_post_hook_name(layer), make_hook(layer)) for layer in layers]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, return_type=None)

    dtype = next(iter(captured.values())).dtype

    return CapturedActivations(
        layers=list(layers),
        activations=captured,
        dtype=dtype,
        d_model=model.cfg.d_model,
        batch_size=tokens.shape[0],
    )
