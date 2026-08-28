"""Stub. Logit lens: unembed the residual stream at each layer (via the
model's final LayerNorm + unembedding matrix) to see what token the model
would "currently" predict at intermediate depth — useful for seeing at
which layer a jailbreak's effect on the output distribution first shows
up, rather than only looking at the final logits. Not implemented yet.
"""

from __future__ import annotations

from typing import Any

from llm_jb.analyses.base import Analysis, AnalysisResult
from llm_jb.analyses.batch import Batch


class LogitLensAnalysis(Analysis):
    name = "logit_lens"

    def run(self, model: Any, batch: Batch) -> AnalysisResult:
        raise NotImplementedError(
            "logit_lens not implemented yet. Would apply model.ln_final + "
            "model.unembed to residual_capture's per-layer activations to "
            "get a per-layer vocab distribution, e.g. via "
            "model.unembed(model.ln_final(resid))."
        )
