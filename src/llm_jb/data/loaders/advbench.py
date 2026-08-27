"""Loader for AdvBench (harmful behaviors), via the `walledai/AdvBench`
mirror on HuggingFace. That dataset is access-gated: an `HF_TOKEN` with
the gate accepted (once, on the dataset page) is required on first
download; afterwards results are served from the local JSON cache."""

from __future__ import annotations

from pathlib import Path

from llm_jb.data.cache import cached_json, default_cache_dir
from llm_jb.data.types import BehaviorTriple, PromptSpan


def _records_to_triples(records: list[dict]) -> list[BehaviorTriple]:
    return [
        BehaviorTriple(
            behavior_id=f"advbench_{i}",
            category="uncategorized",
            harmful_prompt=PromptSpan.whole(r["prompt"]),
            benign_prompt=None,
            jailbroken_prompt=None,
            source="advbench",
            metadata={"target": r["target"]},
        )
        for i, r in enumerate(records)
    ]


def load_advbench(cache_dir: Path | None = None) -> list[BehaviorTriple]:
    cache_dir = cache_dir or default_cache_dir("advbench")
    cache_path = cache_dir / "behaviors.json"

    def _download() -> list[dict]:
        from datasets import load_dataset

        ds = load_dataset("walledai/AdvBench", split="train")
        return [{"prompt": r["prompt"], "target": r["target"]} for r in ds]

    records = cached_json(cache_path, _download)
    return _records_to_triples(records)
