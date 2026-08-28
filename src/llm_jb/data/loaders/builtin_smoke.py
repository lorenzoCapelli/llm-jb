"""A tiny, hardcoded set of plumbing-only prompts — no benchmark content,
nothing harmful — used as the default dataset for smoke tests
(`scripts/run_analysis.py` with no `dataset=` override) so the pipeline
can be exercised without network access, HF gating, or the JBB one-off
fetch step.
"""

from __future__ import annotations

from llm_jb.data.types import BehaviorTriple, PromptSpan

_SMOKE_PAIRS = [
    (
        "Describe the steps to bake a chocolate cake.",
        "Describe the steps to plant a tomato seedling.",
    ),
    ("Explain how a bicycle derailleur works.", "Explain how a bicycle brake works."),
]


def load_builtin_smoke() -> list[BehaviorTriple]:
    return [
        BehaviorTriple(
            behavior_id=f"smoke_{i}",
            category="smoke_test",
            harmful_prompt=PromptSpan.whole(harmful_text),
            benign_prompt=PromptSpan.whole(benign_text),
            jailbroken_prompt=None,
            source="builtin_smoke",
        )
        for i, (harmful_text, benign_text) in enumerate(_SMOKE_PAIRS)
    ]
