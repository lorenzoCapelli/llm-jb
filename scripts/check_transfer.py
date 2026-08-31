"""Checks whether jailbreak artifacts JBB recorded as successful against
one target model (`attack_model`) still elicit compliance when actually
run against a *different* model here — under this repo's own greedy
decoding and substring judge, not JBB's own (unknown) eval setup.

`jailbroken_success` in JBB's metadata only ever means "worked against
attack_model" (see data/loaders/jbb.py); this script builds the ground
truth that actually matters for a downstream analysis on `model`: does
each candidate still work against *that* model. No new package code —
reuses models.backend.load_model, models.generate.generate_greedy, and
metrics.judge.SubstringRefusalJudge as-is.

Usage (key=value, same convention as run_analysis.py):

    python scripts/check_transfer.py model=llama-2-7b-chat attack_model=vicuna-13b-v1.5
    python scripts/check_transfer.py model=llama-2-7b-chat attack_model=vicuna-13b-v1.5 \
        attack_methods=PAIR,GCG max_new_tokens=40 limit=20

Saves the full per-example verdicts as JSON to
artifacts/transfer_<model>_from_<attack_model>.json (gitignored).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from llm_jb.data.loaders.jbb import load_jbb  # noqa: E402
from llm_jb.data.tokenize import tokenize  # noqa: E402
from llm_jb.metrics.judge import SubstringRefusalJudge  # noqa: E402
from llm_jb.models.backend import load_model  # noqa: E402
from llm_jb.models.config import ModelConfig  # noqa: E402
from llm_jb.models.generate import generate_greedy  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


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
    model_name = overrides["model"]
    attack_model = overrides["attack_model"]
    attack_methods = overrides.get("attack_methods", "PAIR,GCG").split(",")
    max_new_tokens = int(overrides.get("max_new_tokens", "40"))
    limit = int(overrides["limit"]) if "limit" in overrides else None

    model_cfg = ModelConfig.load(CONFIGS_DIR / "model" / f"{model_name}.yaml")
    print(f"loading '{model_cfg.name}' ({model_cfg.hf_repo_id})...")
    model = load_model(model_cfg)
    tokenizer = model.tokenizer
    judge = SubstringRefusalJudge()

    candidates = []
    for method in attack_methods:
        triples = load_jbb(attack_method=method, attack_model=attack_model)
        for triple in triples:
            if triple.jailbroken_prompt is not None and triple.metadata.get("jailbroken_success"):
                candidates.append((method, triple))
    if limit is not None:
        candidates = candidates[:limit]

    print(
        f"{len(candidates)} candidate(s): jailbreaks JBB recorded as successful "
        f"against '{attack_model}', about to test against '{model_name}'"
    )

    results = []
    t0 = time.time()
    for i, (method, triple) in enumerate(candidates):
        tokenized = tokenize(triple, tokenizer)
        response = generate_greedy(
            model, tokenized.jailbroken.input_ids, max_new_tokens=max_new_tokens
        )
        verdict = judge.judge(response)
        transferred = verdict.verdict.value == "compliance"
        results.append(
            {
                "behavior_id": triple.behavior_id,
                "behavior_name": triple.metadata["behavior_name"],
                "category": triple.category,
                "attack_method": method,
                "transferred": transferred,
                "verdict": verdict.verdict.value,
                "judge_evidence": verdict.evidence,
                "response": response.strip(),
            }
        )
        status = "TRANSFERRED" if transferred else "refused"
        name = triple.metadata["behavior_name"]
        print(f"  [{i + 1}/{len(candidates)}] {method} {triple.behavior_id} ({name}): {status}")

    elapsed = time.time() - t0
    n_transferred = sum(r["transferred"] for r in results)
    per_example = elapsed / max(len(results), 1)
    print(f"\n{n_transferred}/{len(results)} transferred in {elapsed:.1f}s ({per_example:.1f}s/ea)")
    for method in attack_methods:
        subset = [r for r in results if r["attack_method"] == method]
        n_sub = sum(r["transferred"] for r in subset)
        print(f"  {method}: {n_sub}/{len(subset)} transferred")

    out_path = ARTIFACTS_DIR / f"transfer_{model_name}_from_{attack_model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
