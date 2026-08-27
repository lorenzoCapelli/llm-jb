# llm-jb

Modular mechanistic-interpretability lab for defensive/diagnostic analysis
of transformer LLMs under jailbreak attacks, on public benchmarks
(JailbreakBench, AdvBench, HarmBench). No attack algorithm is implemented
or run here: only existing public prompts and artifacts are used.

The analysis method is not fixed: the repo exposes a common interface
(`Analysis`) to make different approaches interchangeable (logit lens,
activation patching, linear probing, SAE, ...), rather than implementing a
single paper.

## Environment setup

Requires conda (no `uv`) and Python 3.11. GPUs are optional for
development: everything also runs on CPU with the default model
(`gpt2-small`).

```bash
conda env create -f environment.yml
conda activate llm-jb
pip install -r requirements.txt
pip install -e .
python scripts/check_env.py
```

`check_env.py` should print the torch version, the CUDA version torch was
built against, and the visible GPUs (or "no GPU visible" on CPU).

If `conda activate` fails with `invalid choice: 'activate'`, conda's shell
integration hasn't been set up for your shell yet. Either run
`conda init bash` once and restart your shell (or `source ~/.bashrc`), or
for a one-off session: `source <conda_base>/etc/profile.d/conda.sh` before
`conda activate llm-jb` (find `<conda_base>` with `conda info --base`).

### PyTorch and CUDA version

`requirements.txt` installs torch from an explicit CUDA index
(`--extra-index-url https://download.pytorch.org/whl/cu126`), not from the
(deprecated) conda `pytorch` channel. The current pin
(`torch==2.13.0+cu126`) was chosen because `transformer_lens` requires
`torch>=2.6`, and the `cu121` wheel family stopped at torch 2.5.1 — so
`cu121` is no longer a valid option for this dependency combination.

If the pin stops resolving on the target hardware (older driver, or newer
torch releases dropping `cu126`):

1. `nvidia-smi` — the header shows the maximum CUDA version the driver
   supports (backward compatible: a driver supporting CUDA 13.x runs
   `cu12x` builds without issues).
2. Compare against the tags published at
   https://download.pytorch.org/whl/torch_stable.html, or list the
   versions available for a given tag with:
   ```bash
   curl -s https://download.pytorch.org/whl/cu126/torch/ | grep -oE 'torch-[0-9.]+%2Bcu126' | sort -uV
   ```
3. Update the pin in `requirements.txt` (both the `--extra-index-url` tag
   and the `torch==X.Y.Z+cuTAG` version) and check that `transformer_lens`
   still resolves (it requires `torch>=2.6`).

### Environment variables (`.env`)

```bash
cp .env.example .env
```

`HF_HOME` must **not** stay at the default (`~/.cache/huggingface`): on
this shared machine it should point explicitly to a path with dedicated
storage (see `.env.example`). `HF_TOKEN` is needed for gated models (e.g.
`llama-3.2-1b-instruct`, not downloaded by default) and for the
AdvBench/HarmBench dataset mirrors (see "Data layer" below).

## Data layer

`src/llm_jb/data/` normalizes all three benchmarks into one type,
`BehaviorTriple` (`data/types.py`): a `harmful_prompt` / `benign_prompt` /
`jailbroken_prompt`, each a `PromptSpan` carrying the prompt text plus,
**in characters**, where the core instruction sits inside it. Character
offsets are used because the three variants have different token counts
under different tokenizers — token indices are only ever computed on
demand.

- `data/tokenize.py::tokenize(triple, tokenizer)` applies the tokenizer's
  chat template (falling back to the raw prompt text for base LMs like
  gpt2, which have none), tokenizes with `return_offsets_mapping=True`,
  and converts the character spans into token spans, also recording the
  index of the prompt's last token (where generation starts).
- `data/alignment.py` is the single place that resolves an anchor mode
  (`AnchorMode.LAST_PROMPT_POSITION` (default), `LAST_K_TOKENS`,
  `MEAN_INSTRUCTION_SPAN`) into token positions and extracts them from a
  batch of activations via `torch.gather` — never `activations[:, -1, :]`,
  which silently reads padding for every row shorter than the longest one
  in a right-padded batch (see `tests/test_alignment.py`).
- `data/loaders/{jbb,advbench,harmbench}.py` each expose a `load_*()`
  function returning `list[BehaviorTriple]`, normalizing that benchmark's
  schema and caching normalized records as local JSON under
  `data/cache/<source>/` (separate from HF's own cache).

### JailbreakBench and the `jailbreakbench` package

`jailbreakbench` pins `transformers<5.0.0` (we run `5.16.1`) and its
`litellm` dependency breaks against recent `litellm` releases — so it is
**not** a runtime dependency of this repo. `scripts/fetch_jbb_artifacts.py`
downloads JBB-Behaviors and jailbreak artifacts (PAIR, GCG) once, in an
isolated throwaway venv, and caches them as JSON:

```bash
python3 -m venv /tmp/jbb-fetch-env
/tmp/jbb-fetch-env/bin/pip install jailbreakbench==1.0.0 litellm==1.44.24
/tmp/jbb-fetch-env/bin/python scripts/fetch_jbb_artifacts.py
```

`data/loaders/jbb.py` only ever reads the resulting JSON — it never
imports `jailbreakbench`. Note that PAIR paraphrases the original goal, so
the loader usually can't locate it verbatim inside the adversarial prompt
and falls back to treating the whole jailbroken prompt as the instruction
(`metadata["instruction_span_matched"] = False`); GCG appends its suffix
to the verbatim goal and matches almost every time.

### AdvBench and HarmBench

Loaded via `datasets.load_dataset` from `walledai/AdvBench` and
`walledai/HarmBench` (`config="standard"` by default). Both are
access-gated on HuggingFace: accept the terms on each dataset's page once
with the account behind your `HF_TOKEN`, then results are cached locally
after the first successful download.

## Structure

```
src/llm_jb/
  data/       # BehaviorTriple, TokenizedTriple, JBB/AdvBench/HarmBench loaders, cache
  models/     # model configs, TransformerLens backend (+ nnsight stub)
  hooks/      # selective activation capture, CPU/disk offload
  analyses/   # Analysis interface + residual_capture (reference) + stubs
  metrics/    # refusal/compliance judge (substring + model-based stub)
  viz/        # visualization utilities (empty for now)
scripts/      # CLI entrypoints: check_env.py, run_analysis.py, sweep.py, ...
configs/      # YAML configs per model / dataset / analysis
tests/
notebooks/    # exploration only, no application logic
```

_("Usage" and "Activation memory notes" sections land with the next steps
of the plan, see `PLAN.md`)_

## GPUs: single machine, 4x A100

The 4 GPUs are used to run **independent** experiments in parallel (one
per GPU via `CUDA_VISIBLE_DEVICES`), not to shard a single model. No
SLURM, no separate prefetch step: models are downloaded on first use into
`HF_HOME`.
