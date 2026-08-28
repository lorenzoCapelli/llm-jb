"""Stub. Activation patching: run the model on one variant (e.g.
jailbroken) while substituting an activation captured from another
variant (e.g. harmful) at a chosen layer/position, and measure how much
of the output difference that single substitution accounts for — the
classic causal-tracing tool for localizing where a jailbreak's effect
"lives" in the network. Needs a two-pass, hook-based patch-in mechanism
beyond simple capture; not implemented yet.
"""

from __future__ import annotations

from typing import Any

from llm_jb.analyses.base import Analysis, AnalysisResult
from llm_jb.analyses.batch import Batch


class ActivationPatchingAnalysis(Analysis):
    name = "activation_patching"

    def run(self, model: Any, batch: Batch) -> AnalysisResult:
        raise NotImplementedError(
            "activation_patching not implemented yet. Would run a hooked "
            "forward pass on one variant while overwriting an activation "
            "with one captured from another variant at chosen "
            "layers/positions, then compare logits/metrics to the "
            "unpatched run."
        )
