"""Deterministic (greedy) text generation on top of a loaded model.

Nothing else in the repo generates text — `hooks/capture.py` only does a
single forward pass with `return_type=None`, and `metrics/judge.py`
expects an already-generated response string. This module is the one
place that turns a tokenized prompt into a response string, so an analysis
or a notebook can feed the judge without re-implementing sampling.

TransformerLens ships `HookedTransformer.generate`; this is a thin,
greedy-only wrapper around it (`do_sample=False`), operating on the
per-example, pre-padding `input_ids` that `data/tokenize.py` already
produces so there is no left/right padding ambiguity.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformer_lens import HookedTransformer

from llm_jb.data.types import TokenizedSpan


def generate_greedy(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 40,
    stop_at_eos: bool = True,
) -> str:
    """Greedy-decode a continuation for a single prompt.

    `input_ids` is a 1-D tensor of one prompt's real tokens (no padding),
    e.g. `TokenizedSpan.input_ids`. Returns only the newly generated text,
    decoded with special tokens skipped.
    """
    if input_ids.ndim != 1:
        raise ValueError(f"expected a 1-D input_ids tensor, got shape {tuple(input_ids.shape)}")

    device = next(model.parameters()).device
    tokens = input_ids.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stop_at_eos=stop_at_eos,
            prepend_bos=False,
            return_type="tokens",
            verbose=False,
        )

    new_tokens = out[0, tokens.shape[1] :]
    return model.tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_greedy_batch(
    model: HookedTransformer,
    spans: Sequence[TokenizedSpan],
    max_new_tokens: int = 40,
    stop_at_eos: bool = True,
) -> list[str]:
    """`generate_greedy` for each span, one prompt at a time (no padding).

    One forward loop per prompt keeps this correct regardless of how the
    spans differ in length; fine for the handful of prompts an analysis or
    walkthrough runs, not meant for large-batch throughput.
    """
    return [
        generate_greedy(
            model, span.input_ids, max_new_tokens=max_new_tokens, stop_at_eos=stop_at_eos
        )
        for span in spans
    ]
