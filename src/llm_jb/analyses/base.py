"""Common interface every analysis implements, so logit lens, activation
patching, linear probing, and SAE features are interchangeable behind one
`run(model, batch) -> AnalysisResult` call — see `REGISTRY` in
`__init__.py` for the name -> class lookup `scripts/run_analysis.py` uses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from llm_jb.analyses.batch import Batch


@dataclass(frozen=True)
class AnalysisResult:
    analysis_name: str
    behavior_ids: list[str]
    variants: list[str]
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class Analysis(ABC):
    name: str

    @abstractmethod
    def run(self, model: Any, batch: Batch) -> AnalysisResult: ...
