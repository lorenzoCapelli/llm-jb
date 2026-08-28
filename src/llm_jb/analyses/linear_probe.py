"""Stub. Linear probing on the residual stream: fit a linear classifier
on residual_capture's per-layer activations against an external label
(harmful vs. benign, jailbroken vs. not, refused vs. complied — see
metrics/judge.py) to check whether that distinction is linearly decodable
at a given layer/position, before reaching for heavier causal methods.
Not implemented yet.
"""

from __future__ import annotations

from typing import Any

from llm_jb.analyses.base import Analysis, AnalysisResult
from llm_jb.analyses.batch import Batch


class LinearProbeAnalysis(Analysis):
    name = "linear_probe"

    def run(self, model: Any, batch: Batch) -> AnalysisResult:
        raise NotImplementedError(
            "linear_probe not implemented yet. Would fit a linear classifier "
            "(e.g. logistic regression) on residual_capture's per-layer "
            "activations against an external label from metrics/judge.py, "
            "and report per-layer accuracy/AUC."
        )
