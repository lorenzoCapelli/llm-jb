"""Builds a right-padded, model-ready `Batch` from `BehaviorTriple`s — the
one place that flattens the harmful/benign/jailbroken variants of many
triples into rows and tokenizes/pads them consistently, so every analysis
consumes the same batch shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from llm_jb.data.tokenize import tokenize
from llm_jb.data.types import BehaviorTriple, TokenizedSpan

VARIANTS = ("harmful", "benign", "jailbroken")


@dataclass(frozen=True)
class Batch:
    """A forward-pass-ready batch, plus enough per-row metadata to align
    and label results afterward. `tokens` is right-padded; `spans[i]`
    still carries row `i`'s true (pre-padding) length and anchor info —
    see data/alignment.py."""

    tokens: torch.Tensor  # (batch, seq)
    spans: list[TokenizedSpan]
    behavior_ids: list[str]
    variants: list[str]  # one of VARIANTS per row


def build_batch(
    triples: Sequence[BehaviorTriple],
    tokenizer,
    variants: Sequence[str] = ("harmful",),
    pad_token_id: int | None = None,
) -> Batch:
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variant(s) {unknown}, expected a subset of {VARIANTS}")

    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has no pad_token_id or eos_token_id to pad with")

    row_ids: list[torch.Tensor] = []
    spans: list[TokenizedSpan] = []
    behavior_ids: list[str] = []
    row_variants: list[str] = []

    for triple in triples:
        tokenized = tokenize(triple, tokenizer)
        for variant in variants:
            span = getattr(tokenized, variant)
            if span is None:
                continue
            row_ids.append(span.input_ids)
            spans.append(span)
            behavior_ids.append(triple.behavior_id)
            row_variants.append(variant)

    if not row_ids:
        raise ValueError("no rows to batch: no triple had any of the requested variants")

    maxlen = max(ids.shape[0] for ids in row_ids)
    tokens = torch.stack(
        [
            torch.nn.functional.pad(ids, (0, maxlen - ids.shape[0]), value=pad_token_id)
            for ids in row_ids
        ]
    )

    return Batch(tokens=tokens, spans=spans, behavior_ids=behavior_ids, variants=row_variants)
