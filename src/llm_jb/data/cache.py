"""Local on-disk JSON cache for downloaded dataset records, separate from
HuggingFace's own cache (HF_HOME): this caches already-normalized records
so a loader doesn't need network or `datasets` at all once the cache is
warm."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def repo_root() -> Path:
    # src/llm_jb/data/cache.py -> repo root is 3 parents up
    return Path(__file__).resolve().parents[3]


def default_cache_dir(*parts: str) -> Path:
    return repo_root() / "data" / "cache" / Path(*parts)


def cached_json(path: Path, compute: Callable[[], T]) -> T:
    if path.exists():
        return json.loads(path.read_text())
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result
