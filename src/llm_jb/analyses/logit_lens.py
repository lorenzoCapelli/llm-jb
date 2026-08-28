"""Logit lens: read every layer's residual stream through the model's
final LayerNorm and unembedding to get the vocab distribution the model
would predict "so far" at that depth.

Useful for seeing at which layer a jailbreak's effect on the output
distribution first shows up, instead of only inspecting the final logits.
The projection reuses the model's own `ln_final` + `unembed` (with LN
folded into the unembedding, as `HookedTransformer.from_pretrained`
returns it), so the last layer's lens is by construction the model's real
next-token logits.

Caveats worth remembering when reading the output: the logit lens is not
calibrated on early layers (the residual basis has not yet been rotated
into the unembedding's frame there), so early-layer distributions are
often uninformative or misleading; and this only looks at one position
(the last prompt token) per prompt.
"""

from __future__ import annotations

import torch
from pydantic_settings import SettingsConfigDict
from transformer_lens import HookedTransformer

from llm_jb.analyses.base import Analysis, AnalysisResult
from llm_jb.analyses.batch import Batch
from llm_jb.config import YamlSettings
from llm_jb.data.types import AnchorMode
from llm_jb.hooks.capture import capture_residual_stream


class LogitLensConfig(YamlSettings):
    model_config = SettingsConfigDict(extra="forbid", env_prefix="LLM_JB_LOGIT_LENS_")

    layers: list[int] | None = None  # None = every layer
    anchor_mode: AnchorMode = AnchorMode.LAST_PROMPT_POSITION
    k: int = 1


class LogitLensAnalysis(Analysis):
    """Per-layer, per-example vocab logits at the configured anchor
    position. `data["layer_<i>"]` has shape `(batch, d_vocab)`."""

    name = "logit_lens"

    def __init__(self, config: LogitLensConfig | None = None):
        self.config = config or LogitLensConfig()

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
            placement="gpu",
        )

        data: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for layer in layers:
                resid = captured.activations[layer]  # (batch, d_model)
                logits = model.unembed(model.ln_final(resid))  # (batch, d_vocab)
                data[f"layer_{layer}"] = logits.float().cpu()

        return AnalysisResult(
            analysis_name=self.name,
            behavior_ids=batch.behavior_ids,
            variants=batch.variants,
            data=data,
            metadata={
                "layers": layers,
                "d_vocab": int(model.cfg.d_vocab),
                "anchor_mode": self.config.anchor_mode.value,
            },
        )
