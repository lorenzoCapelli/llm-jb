# PLAN — llm-jb

Modular lab for defensive mechanistic interpretability on LLM jailbreaks
(JailbreakBench, AdvBench, HarmBench). No scientific analysis at this
stage: just a reproducible repo + environment + an end-to-end smoke test.

## Decisions made (after discussion)

- **Config**: pydantic-settings + plain YAML (no Hydra). Typed configs, no
  cwd magic, sweep.py stays custom for the GPU queue regardless.
- **HF_HOME default**: `/mounts/Users/cisintern/lorenzo/.cache/huggingface`,
  overridable via `.env` (`HF_HOME=...`).
- **PyTorch**: pip from `--index-url https://download.pytorch.org/whl/cu126`,
  exact pin in `requirements.txt`. The local driver supports CUDA up to
  13.3, so any cu12x wheel is compatible; cu126 was chosen because
  `transformer_lens` requires `torch>=2.6` and `cu121` wheels stop at
  torch 2.5.1. README documents how to check/update from `nvidia-smi`.
- **Packaging**: `pyproject.toml` with `hatchling` backend, `src/` layout,
  `pip install -e .`.
- **Interp backend**: TransformerLens implemented now (the only backend
  needed for gpt2-small on CPU/1 GPU). nnsight stays a stub behind the
  same `models/` interface, optional import, NOT a hard dependency until
  it's actually needed for larger models.
- **jailbreakbench**: dependency of a one-off script only
  (`scripts/fetch_jbb_artifacts.py`) that downloads/serializes artifacts
  to local JSON. Not in runtime requirements, to avoid pin conflicts with
  transformers/TransformerLens.
- **Dev model**: `gpt2-small` (native TransformerLens support, no gating).
  `llama-3.2-1b-instruct` exists only as a config, not downloaded.

## Structure

```
src/llm_jb/
  data/       # BehaviorTriple, TokenizedTriple, JBB/AdvBench/HarmBench loaders, cache
  models/     # model configs, TransformerLens backend (+ nnsight stub)
  hooks/      # selective activation capture, CPU/disk offload
  analyses/   # Analysis base class + residual_capture (reference impl) + stubs
  metrics/    # refusal/compliance judge (substring + model-based stub)
  viz/        # empty for now, placeholder __init__
scripts/      # check_env.py, fetch_jbb_artifacts.py, run_analysis.py, sweep.py
configs/      # model/*.yaml, dataset/*.yaml, analysis/*.yaml, sample sweep
tests/
notebooks/    # placeholder .gitkeep
```

## Steps (one atomic commit per step, I stop after each for diff + test review)

1. **Scaffolding**: directory layout, `pyproject.toml`, `environment.yml`,
   `requirements.txt`, `.env.example`, `.gitignore`, pre-commit/ruff
   config, README skeleton. Working `scripts/check_env.py`.
2. **Data layer**: `BehaviorTriple`, `tokenize()` → `TokenizedTriple` with
   char→token offset mapping and anchor resolution (last prompt position /
   last k / mean over span), normalized loaders for JBB / AdvBench /
   HarmBench (from local cached JSON), on-disk cache.
   `scripts/fetch_jbb_artifacts.py` for the one-off download via
   `jailbreakbench`. Tests on tokenize/alignment.
3. **Models layer**: pydantic config per model (gpt2-small,
   llama-3.2-1b as config-only), TransformerLens backend, device selection
   via `CUDA_VISIBLE_DEVICES`, nnsight stub backend behind the same
   interface.
4. **Hooks / activation memory**: selective per-layer+position capture
   (never a full cache), CPU offload or safetensors save option, explicit
   shape/dtype. Shape/determinism test and VRAM peak test with
   configurable threshold.
5. **Analyses**: `Analysis` base class (`run(model, batch) -> AnalysisResult`),
   `residual_capture` implementation, documented stubs for logit_lens,
   activation_patching, linear_probe, sae.
6. **Metrics**: refusal/compliance judge with substring backend (baseline)
   and model-based backend (configurable stub), common interface not
   hardcoded into analyses.
7. **End-to-end scripts**: `run_analysis.py` (loads YAML config via
   pydantic-settings, runs on CPU with gpt2-small in <2min, saves an
   artifact), `sweep.py` (simple queue, one process per free GPU via
   `CUDA_VISIBLE_DEVICES`, per-run logs, Ctrl+C and failure handling), 4
   dummy example configs in `configs/sweep_example/`.
8. **Polish**: full README (setup, commands, config structure, memory
   notes, how to derive CUDA from `nvidia-smi`), final Definition-of-Done
   pass end-to-end.

## Planned tests (pytest)

- `test_alignment.py`: mixed-length batch, checks that extraction uses
  `gather` on real last-position indices, not `[:, -1, :]`.
- `test_activations.py`: explicit shape/dtype and capture determinism.
- `test_vram.py`: VRAM peak on a small batch stays under a configurable
  threshold (auto-skipped if no GPU is visible).
- `test_data.py`: loader normalization → `BehaviorTriple`, tokenize +
  offset mapping.
- `test_metrics.py`: substring judge on known refusal/compliance cases.

## Open notes (one-liners, my call unless flagged)

- pyproject build backend: `hatchling` (simpler than setuptools for a
  `src/` layout, no valid alternative worth discussing).
- Model-based judge: stays a typed `NotImplementedError` stub, no
  hardcoded provider — chosen when actually implemented.
