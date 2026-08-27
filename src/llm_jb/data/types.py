"""Core data types for behavior triples.

Character offsets are stored here, never token indices: token positions
depend on the tokenizer and the chat template, so they are only computed
on demand by `tokenize()` (see tokenize.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch


@dataclass(frozen=True)
class PromptSpan:
    """A prompt string plus, in characters, where the core instruction sits
    within it and where a response-priming suffix (if any) begins.

    `instruction_start`/`instruction_end` locate the behavior text itself
    inside any surrounding jailbreak wrapper — (0, len(text)) when the
    prompt IS the instruction (e.g. AdvBench/HarmBench goals, or JBB
    Goal/benign strings). For a jailbroken prompt, they mark where the
    original harmful instruction was found inside the adversarial wrapper.

    `response_start` marks where, within this same string, an
    attack-injected response-priming suffix begins (e.g. a prefix-injection
    jailbreak that appends "Sure, here is..." to the prompt itself); it
    defaults to len(text) when the prompt carries no such suffix, which is
    the case for every loader currently implemented. This is distinct from
    chat-template-driven generation offsets, which `tokenize()` computes
    separately per tokenizer.
    """

    text: str
    instruction_start: int
    instruction_end: int
    response_start: int

    def __post_init__(self) -> None:
        n = len(self.text)
        if not (0 <= self.instruction_start <= self.instruction_end <= n):
            raise ValueError(
                f"invalid instruction span ({self.instruction_start}, "
                f"{self.instruction_end}) for text of length {n}"
            )
        if not (self.instruction_end <= self.response_start <= n):
            raise ValueError(
                f"response_start ({self.response_start}) must be between "
                f"instruction_end ({self.instruction_end}) and len(text) ({n})"
            )

    @classmethod
    def whole(cls, text: str) -> PromptSpan:
        """The common case: the whole string is the instruction, and the
        response starts right after it."""
        return cls(
            text=text, instruction_start=0, instruction_end=len(text), response_start=len(text)
        )


@dataclass(frozen=True)
class BehaviorTriple:
    behavior_id: str
    category: str
    harmful_prompt: PromptSpan
    benign_prompt: PromptSpan | None
    jailbroken_prompt: PromptSpan | None
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenizedSpan:
    """A single tokenized prompt variant.

    `last_prompt_position` is the index of the last token of the prompt
    (before any generation) for THIS example, pre-padding. It stays valid
    as the "true last token" index after right-padding a batch, which is
    exactly why extraction must gather on these indices rather than assume
    `[:, -1, :]`.
    """

    input_ids: torch.Tensor  # (seq_len,)
    instruction_token_start: int
    instruction_token_end: int
    last_prompt_position: int


@dataclass(frozen=True)
class TokenizedTriple:
    behavior_id: str
    harmful: TokenizedSpan
    benign: TokenizedSpan | None
    jailbroken: TokenizedSpan | None


class AnchorMode(StrEnum):
    """Where, along the sequence, to read activations from. This is the
    single enum every analysis should import instead of hardcoding
    position logic — see data/alignment.py for the resolution + extraction
    functions that consume it."""

    LAST_PROMPT_POSITION = "last_prompt_position"
    LAST_K_TOKENS = "last_k_tokens"
    MEAN_INSTRUCTION_SPAN = "mean_instruction_span"
