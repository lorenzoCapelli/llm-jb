"""Loader for JailbreakBench (JBB-Behaviors + jailbreak artifacts).

Reads exclusively from the local JSON cache produced by
`scripts/fetch_jbb_artifacts.py` — the `jailbreakbench` package itself is
never imported at runtime (see that script's docstring for why).
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_jb.data.cache import default_cache_dir
from llm_jb.data.types import BehaviorTriple, PromptSpan


def _make_jailbroken_span(prompt: str, goal: str) -> tuple[PromptSpan, bool]:
    """Locates `goal` inside the adversarial `prompt` text so activations
    at the original instruction can still be compared across variants.
    Falls back to treating the whole prompt as the instruction when the
    goal text isn't found verbatim (e.g. PAIR paraphrases it; GCG appends
    a suffix to the verbatim goal and matches almost every time). Returns
    whether the match succeeded, so callers can record it instead of
    silently guessing."""
    idx = prompt.find(goal)
    if idx == -1:
        return PromptSpan.whole(prompt), False
    return (
        PromptSpan(
            text=prompt,
            instruction_start=idx,
            instruction_end=idx + len(goal),
            response_start=len(prompt),
        ),
        True,
    )


def _records_to_triples(
    harmful: dict[int, dict],
    benign: dict[int, dict],
    artifacts: dict[int, dict],
    attack_method: str,
    attack_model: str,
) -> list[BehaviorTriple]:
    triples = []
    for idx, h in sorted(harmful.items()):
        b = benign.get(idx)
        a = artifacts.get(idx)
        # some behaviors have no successful/logged jailbreak for a given
        # method (attack never produced a prompt worth recording)
        if a is not None and a.get("prompt") is None:
            a = None

        jailbroken_span = None
        span_matched = None
        if a:
            jailbroken_span, span_matched = _make_jailbroken_span(a["prompt"], h["Goal"])

        triples.append(
            BehaviorTriple(
                behavior_id=f"jbb_{idx}",
                category=h["Category"],
                harmful_prompt=PromptSpan.whole(h["Goal"]),
                benign_prompt=PromptSpan.whole(b["Goal"]) if b else None,
                jailbroken_prompt=jailbroken_span,
                source="jbb",
                metadata={
                    "behavior_name": h["Behavior"],
                    "target": h.get("Target"),
                    "attack_method": attack_method if a else None,
                    "attack_model": attack_model if a else None,
                    "jailbroken_success": a.get("jailbroken") if a else None,
                    "instruction_span_matched": span_matched,
                },
            )
        )
    return triples


def load_jbb(
    attack_method: str = "PAIR",
    attack_model: str = "vicuna-13b-v1.5",
    cache_dir: Path | None = None,
) -> list[BehaviorTriple]:
    cache_dir = cache_dir or default_cache_dir("jbb")
    harmful_path = cache_dir / "behaviors_harmful.json"
    benign_path = cache_dir / "behaviors_benign.json"
    artifact_path = cache_dir / f"artifact_{attack_method}_{attack_model}.json"

    for path in (harmful_path, benign_path, artifact_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/fetch_jbb_artifacts.py first, "
                "in an isolated venv (see its docstring), to populate the "
                "local JBB cache."
            )

    harmful = {r["index"]: r for r in json.loads(harmful_path.read_text())}
    benign = {r["index"]: r for r in json.loads(benign_path.read_text())}
    artifacts = {r["index"]: r for r in json.loads(artifact_path.read_text())}

    return _records_to_triples(harmful, benign, artifacts, attack_method, attack_model)
