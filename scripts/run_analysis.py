"""Run one analysis over one dataset with one model, end to end, and save
the result as a safetensors artifact.

Usage (key=value overrides, each resolved against configs/<key>/<value>.yaml):

    python scripts/run_analysis.py analysis=residual_capture model=gpt2-small
    python scripts/run_analysis.py analysis=residual_capture model=gpt2-small \
        dataset=jbb output=artifacts/jbb_run.safetensors

`dataset` defaults to `smoke` (configs/dataset/smoke.yaml, no network/gating
needed) and `output` defaults to artifacts/<analysis>_<model>.safetensors.
"""

from __future__ import annotations

# Load .env (HF_HOME etc.) before anything that reads it at import time.
from dotenv import load_dotenv

load_dotenv()

import dataclasses  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from llm_jb.analyses import REGISTRY, build_batch  # noqa: E402
from llm_jb.analyses.logit_lens import LogitLensConfig  # noqa: E402
from llm_jb.analyses.residual_capture import ResidualCaptureConfig  # noqa: E402
from llm_jb.data.dataset_config import DatasetConfig, load_dataset_triples  # noqa: E402
from llm_jb.hooks.storage import save_activations_safetensors  # noqa: E402
from llm_jb.models.backend import load_model  # noqa: E402
from llm_jb.models.config import ModelConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# Analyses with a typed, YAML-loadable config. The remaining stubs
# (activation_patching, ...) aren't listed here: they take no config and
# raise NotImplementedError from run() regardless.
ANALYSIS_CONFIG_TYPES = {
    "residual_capture": ResidualCaptureConfig,
    "logit_lens": LogitLensConfig,
}


def parse_overrides(argv: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"expected key=value argument, got: {arg!r}")
        key, value = arg.split("=", 1)
        overrides[key] = value
    return overrides


def main() -> None:
    overrides = parse_overrides(sys.argv[1:])

    if "analysis" not in overrides:
        raise ValueError("missing required argument: analysis=<name>")
    if "model" not in overrides:
        raise ValueError("missing required argument: model=<name>")

    analysis_name = overrides["analysis"]
    if analysis_name not in REGISTRY:
        raise ValueError(f"unknown analysis '{analysis_name}', choices: {sorted(REGISTRY)}")

    model_config = ModelConfig.load(CONFIGS_DIR / "model" / f"{overrides['model']}.yaml")

    dataset_name = overrides.get("dataset", "smoke")
    dataset_config = DatasetConfig.load(CONFIGS_DIR / "dataset" / f"{dataset_name}.yaml")

    config_cls = ANALYSIS_CONFIG_TYPES.get(analysis_name)
    analysis_config_path = CONFIGS_DIR / "analysis" / f"{analysis_name}.yaml"
    if config_cls is not None and analysis_config_path.exists():
        analysis = REGISTRY[analysis_name](config_cls.load(analysis_config_path))
    else:
        analysis = REGISTRY[analysis_name]()

    output_path = Path(
        overrides.get(
            "output", DEFAULT_ARTIFACTS_DIR / f"{analysis_name}_{model_config.name}.safetensors"
        )
    )

    print(
        f"loading '{model_config.name}' ({model_config.hf_repo_id}) via {model_config.backend}..."
    )
    t0 = time.time()
    model = load_model(model_config)
    print(f"  loaded in {time.time() - t0:.1f}s, device={next(model.parameters()).device}")

    triples = load_dataset_triples(dataset_config)
    print(f"loaded {len(triples)} behavior triple(s) from '{dataset_config.source}'")

    batch = build_batch(triples, model.tokenizer, variants=dataset_config.variants)
    batch = dataclasses.replace(batch, tokens=batch.tokens.to(next(model.parameters()).device))
    print(f"built batch: {batch.tokens.shape[0]} rows x {batch.tokens.shape[1]} tokens")

    t0 = time.time()
    result = analysis.run(model, batch)
    print(f"ran '{analysis_name}' in {time.time() - t0:.1f}s")

    tensors = {k: v for k, v in result.data.items() if hasattr(v, "shape")}
    save_activations_safetensors(tensors, output_path)
    print(f"saved artifact to {output_path}")


if __name__ == "__main__":
    main()
