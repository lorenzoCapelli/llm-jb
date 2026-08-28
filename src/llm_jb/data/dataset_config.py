"""Typed, YAML-loadable selector for which benchmark (or the built-in
smoke fixture) a run pulls `BehaviorTriple`s from — kept separate from the
individual loaders so a script only imports the one it actually needs.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import SettingsConfigDict

from llm_jb.config import YamlSettings
from llm_jb.data.types import BehaviorTriple

DatasetSource = Literal["jbb", "advbench", "harmbench", "builtin_smoke"]


class DatasetConfig(YamlSettings):
    model_config = SettingsConfigDict(extra="forbid", env_prefix="LLM_JB_DATASET_")

    source: DatasetSource
    variants: list[str] = ["harmful"]
    limit: int | None = None  # cap number of triples — handy for smoke tests/sweeps
    attack_method: str = "PAIR"  # jbb only
    attack_model: str = "vicuna-13b-v1.5"  # jbb only
    harmbench_config: str = "standard"  # harmbench only


def load_dataset_triples(config: DatasetConfig) -> list[BehaviorTriple]:
    if config.source == "jbb":
        from llm_jb.data.loaders.jbb import load_jbb

        triples = load_jbb(attack_method=config.attack_method, attack_model=config.attack_model)
    elif config.source == "advbench":
        from llm_jb.data.loaders.advbench import load_advbench

        triples = load_advbench()
    elif config.source == "harmbench":
        from llm_jb.data.loaders.harmbench import load_harmbench

        triples = load_harmbench(config=config.harmbench_config)
    elif config.source == "builtin_smoke":
        from llm_jb.data.loaders.builtin_smoke import load_builtin_smoke

        triples = load_builtin_smoke()
    else:
        raise ValueError(f"unknown dataset source: {config.source}")

    if config.limit is not None:
        triples = triples[: config.limit]
    return triples
