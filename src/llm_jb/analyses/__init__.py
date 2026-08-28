from llm_jb.analyses.activation_patching import ActivationPatchingAnalysis
from llm_jb.analyses.base import Analysis, AnalysisResult
from llm_jb.analyses.batch import Batch, build_batch
from llm_jb.analyses.linear_probe import LinearProbeAnalysis
from llm_jb.analyses.logit_lens import LogitLensAnalysis, LogitLensConfig
from llm_jb.analyses.residual_capture import ResidualCaptureAnalysis, ResidualCaptureConfig
from llm_jb.analyses.sae import SaeAnalysis

# Name -> class lookup used by scripts/run_analysis.py (e.g.
# `analysis=residual_capture`) so analyses are selectable by config
# without every caller importing every module.
REGISTRY: dict[str, type[Analysis]] = {
    "residual_capture": ResidualCaptureAnalysis,
    "logit_lens": LogitLensAnalysis,
    "activation_patching": ActivationPatchingAnalysis,
    "linear_probe": LinearProbeAnalysis,
    "sae": SaeAnalysis,
}

__all__ = [
    "Analysis",
    "AnalysisResult",
    "Batch",
    "build_batch",
    "ResidualCaptureAnalysis",
    "ResidualCaptureConfig",
    "LogitLensAnalysis",
    "LogitLensConfig",
    "ActivationPatchingAnalysis",
    "LinearProbeAnalysis",
    "SaeAnalysis",
    "REGISTRY",
]
