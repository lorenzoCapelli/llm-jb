"""Loader for HarmBench behaviors, via the `walledai/HarmBench` mirror on
HuggingFace. Access-gated like AdvBench (see advbench.py). Defaults to the
"standard" config (plain instruction + category, 200 examples); "contextual"
and "copyright" are also available but carry extra fields this loader
folds into `metadata`."""

from __future__ import annotations

from pathlib import Path

from llm_jb.data.cache import cached_json, default_cache_dir
from llm_jb.data.types import BehaviorTriple, PromptSpan


def _records_to_triples(records: list[dict], config: str) -> list[BehaviorTriple]:
    return [
        BehaviorTriple(
            behavior_id=f"harmbench_{config}_{i}",
            category=r.get("category", "uncategorized"),
            harmful_prompt=PromptSpan.whole(r["prompt"]),
            benign_prompt=None,
            jailbroken_prompt=None,
            source="harmbench",
            metadata={k: v for k, v in r.items() if k not in ("prompt", "category")},
        )
        for i, r in enumerate(records)
    ]


def load_harmbench(config: str = "standard", cache_dir: Path | None = None) -> list[BehaviorTriple]:
    cache_dir = cache_dir or default_cache_dir("harmbench")
    cache_path = cache_dir / f"behaviors_{config}.json"

    def _download() -> list[dict]:
        from datasets import load_dataset

        ds = load_dataset("walledai/HarmBench", config, split="train")
        return [dict(r) for r in ds]

    records = cached_json(cache_path, _download)
    return _records_to_triples(records, config)
