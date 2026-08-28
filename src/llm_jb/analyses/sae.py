"""Stub. Sparse autoencoder (SAE) features on the residual stream: decode
residual_capture's activations into a pretrained SAE's overcomplete,
sparse feature basis, to look for individual, more interpretable
directions associated with jailbreak behavior instead of raw (entangled)
residual-stream dimensions. Needs a pretrained SAE checkpoint per
layer/model, which isn't part of this repo; not implemented yet.
"""

from __future__ import annotations

from typing import Any

from llm_jb.analyses.base import Analysis, AnalysisResult
from llm_jb.analyses.batch import Batch


class SaeAnalysis(Analysis):
    name = "sae"

    def run(self, model: Any, batch: Batch) -> AnalysisResult:
        raise NotImplementedError(
            "sae not implemented yet. Would decode residual_capture's "
            "activations through a pretrained SAE and report top active "
            "features per example; needs an SAE checkpoint source (e.g. "
            "SAELens), not yet a dependency of this repo."
        )
