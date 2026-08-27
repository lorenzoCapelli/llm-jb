"""Turns a BehaviorTriple (character spans) into a TokenizedTriple (token
indices), applying the tokenizer's chat template when it has one.

This is the only place that should call a tokenizer on a BehaviorTriple:
keeping char->token conversion here means every analysis sees the same
token-level spans regardless of which tokenizer produced them.
"""

from __future__ import annotations

from typing import Any, Protocol

from llm_jb.data.types import BehaviorTriple, PromptSpan, TokenizedSpan, TokenizedTriple


class ChatTokenizer(Protocol):
    chat_template: str | None

    def apply_chat_template(
        self, conversation: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> str: ...

    def __call__(self, text: str, return_offsets_mapping: bool, return_tensors: str) -> Any: ...


def _apply_template_or_raw(tokenizer: ChatTokenizer, text: str) -> tuple[str, int]:
    """Returns (full_text, prefix_len): `text` starts at `prefix_len`
    inside `full_text`. Falls back to the raw text unchanged when the
    tokenizer has no chat template (e.g. base LMs like gpt2)."""
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        return text, 0

    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_len = full_text.find(text)
    if prefix_len == -1:
        raise ValueError(
            "chat template did not preserve the prompt text verbatim; cannot "
            "map character spans onto the templated string"
        )
    return full_text, prefix_len


def _char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    """Converts a [char_start, char_end) range into a [token_start,
    token_end) range using an offset mapping from a fast tokenizer.
    Special tokens (empty (0, 0)-style spans) are skipped."""
    token_start: int | None = None
    token_end: int | None = None
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if token_start is None and e > char_start:
            token_start = i
        if s < char_end:
            token_end = i + 1
    if token_start is None:
        token_start = 0
    if token_end is None or token_end <= token_start:
        token_end = token_start + 1
    return token_start, token_end


def _tokenize_span(tokenizer: ChatTokenizer, span: PromptSpan) -> TokenizedSpan:
    full_text, prefix = _apply_template_or_raw(tokenizer, span.text)
    encoding = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
    input_ids = encoding["input_ids"][0]
    offsets = [tuple(o) for o in encoding["offset_mapping"][0].tolist()]

    instruction_token_start, instruction_token_end = _char_span_to_token_span(
        offsets, prefix + span.instruction_start, prefix + span.instruction_end
    )

    return TokenizedSpan(
        input_ids=input_ids,
        instruction_token_start=instruction_token_start,
        instruction_token_end=instruction_token_end,
        last_prompt_position=input_ids.shape[0] - 1,
    )


def tokenize(triple: BehaviorTriple, tokenizer: ChatTokenizer) -> TokenizedTriple:
    return TokenizedTriple(
        behavior_id=triple.behavior_id,
        harmful=_tokenize_span(tokenizer, triple.harmful_prompt),
        benign=_tokenize_span(tokenizer, triple.benign_prompt) if triple.benign_prompt else None,
        jailbroken=_tokenize_span(tokenizer, triple.jailbroken_prompt)
        if triple.jailbroken_prompt
        else None,
    )
