"""Single place that decides which token position(s) to read activations
from for a given anchor mode, and extracts them from a batch.

The three prompt variants of a behavior have different token lengths, so
the same index does NOT mean the same content across variants — and in a
right-padded batch, the same index does not mean the same content across
rows either. Every analysis should call `extract_anchor_activations`
instead of re-deriving position logic (or worse, slicing `[:, -1, :]`).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from llm_jb.data.types import AnchorMode, TokenizedSpan


def anchor_range(span: TokenizedSpan, mode: AnchorMode, k: int = 1) -> tuple[int, int]:
    """The [start, end) token range to read for one example, given an
    anchor mode. `k` is only used by LAST_K_TOKENS."""
    if mode == AnchorMode.LAST_PROMPT_POSITION:
        return span.last_prompt_position, span.last_prompt_position + 1
    if mode == AnchorMode.LAST_K_TOKENS:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        start = max(0, span.last_prompt_position - k + 1)
        return start, span.last_prompt_position + 1
    if mode == AnchorMode.MEAN_INSTRUCTION_SPAN:
        return span.instruction_token_start, span.instruction_token_end
    raise ValueError(f"unknown anchor mode: {mode}")


def gather_positions(activations: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Extract one position per example from a batch.

    activations: (batch, seq, *dims). positions: (batch,) long tensor of
    per-example indices — e.g. the true last prompt token in a
    right-padded batch, which differs per row. Never use
    `activations[:, -1, :]` for this: with right padding, the true last
    token sits at a different index in every row.
    """
    if positions.shape[0] != activations.shape[0]:
        raise ValueError("positions must have one entry per batch example")
    batch = activations.shape[0]
    view_shape = (batch, 1) + (1,) * (activations.ndim - 2)
    expand_shape = (batch, 1) + activations.shape[2:]
    idx = positions.view(view_shape).expand(expand_shape)
    return activations.gather(1, idx).squeeze(1)


def mean_pool_range(
    activations: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    """Mean-pool activations over a per-example [start, end) token range.

    activations: (batch, seq, *dims). starts/ends: (batch,) long, end
    exclusive. Positions outside [start, end) are masked out, so this is
    correct under right padding as long as `ends` never exceeds the real
    (unpadded) sequence length for that row.
    """
    seq_len = activations.shape[1]
    positions = torch.arange(seq_len, device=activations.device).unsqueeze(0)  # (1, seq)
    mask = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))  # (batch, seq)
    extra_dims = (1,) * (activations.ndim - 2)
    mask_f = mask.view(*mask.shape, *extra_dims).to(activations.dtype)
    summed = (activations * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1)
    return summed / counts


def extract_anchor_activations(
    activations: torch.Tensor,
    spans: Sequence[TokenizedSpan],
    mode: AnchorMode = AnchorMode.LAST_PROMPT_POSITION,
    k: int = 1,
) -> torch.Tensor:
    """Extract per-example anchor activations from a batch, dispatching on
    `mode`. This is the one function analyses should call."""
    if len(spans) != activations.shape[0]:
        raise ValueError("spans must have one entry per batch example")

    ranges = [anchor_range(span, mode, k) for span in spans]
    starts = torch.tensor([r[0] for r in ranges], device=activations.device)
    ends = torch.tensor([r[1] for r in ranges], device=activations.device)

    if mode == AnchorMode.LAST_PROMPT_POSITION:
        return gather_positions(activations, starts)
    return mean_pool_range(activations, starts, ends)
