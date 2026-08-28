"""Reference analysis: captures residual-stream activations per layer at
the configured anchor position(s). Trivial on purpose — it exists to
prove out the `Analysis` interface and the hooks/alignment machinery, not
to answer a research question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformer_lens import HookedTransformer

from llm_jb.analyses.base import Analysis, AnalysisResult
from llm_jb.analyses.batch import Batch
from llm_jb.data.types import AnchorMode
from llm_jb.hooks.capture import capture_residual_stream
from llm_jb.hooks.storage import ActivationPlacement


@dataclass(frozen=True)
class ResidualCaptureConfig:
    layers: list[int] | None = None  # None = every layer
    anchor_mode: AnchorMode = AnchorMode.LAST_PROMPT_POSITION
    k: int = 1
    placement: ActivationPlacement = "cpu"


class ResidualCaptureAnalysis(Analysis):
    name = "residual_capture"

    def __init__(self, config: ResidualCaptureConfig | None = None):
        self.config = config or ResidualCaptureConfig()

    def run(self, model: HookedTransformer, batch: Batch) -> AnalysisResult:
        layers = (
            self.config.layers
            if self.config.layers is not None
            else list(range(model.cfg.n_layers))
        )
        captured = capture_residual_stream(
            model,
            batch.tokens,
            batch.spans,
            layers=layers,
            anchor_mode=self.config.anchor_mode,
            k=self.config.k,
            placement=self.config.placement,
        )
        data: dict[str, Any] = {
            f"layer_{layer}": captured.activations[layer] for layer in captured.layers
        }

        return AnalysisResult(
            analysis_name=self.name,
            behavior_ids=batch.behavior_ids,
            variants=batch.variants,
            data=data,
            metadata={
                "d_model": captured.d_model,
                "dtype": str(captured.dtype),
                "anchor_mode": self.config.anchor_mode.value,
                "layers": captured.layers,
            },
        )
