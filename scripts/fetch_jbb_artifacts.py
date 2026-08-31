"""One-off download: JBB-Behaviors (harmful + benign) and jailbreak
artifacts via the `jailbreakbench` package, cached as local JSON under
data/cache/jbb/.

`jailbreakbench` pins `transformers<5.0.0`, which conflicts with this
repo's main environment (`transformers==5.16.1`, required by
`transformer_lens`). Its own `litellm` dependency also has no upper
bound and breaks against recent litellm releases (an internal import
path was removed). Do NOT install `jailbreakbench` into the `llm-jb`
conda env — run this script in an isolated, throwaway venv instead:

    python3 -m venv /tmp/jbb-fetch-env
    /tmp/jbb-fetch-env/bin/pip install jailbreakbench==1.0.0 litellm==1.44.24
    /tmp/jbb-fetch-env/bin/python scripts/fetch_jbb_artifacts.py

Re-run any time to refresh the cache; existing files are overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "jbb"

ARTIFACT_METHODS = ["PAIR", "GCG"]
# JBB's valid target-model names (jailbreakbench.config.MODEL_NAMES) are
# 'vicuna-13b-v1.5', 'llama-2-7b-chat-hf', 'gpt-3.5-turbo-1106',
# 'gpt-4-0125-preview' — note the '-hf' suffix on the Llama entry, which
# does not match the HF repo id's own short form. The loader keys its
# cache path on this exact name, so fetching several here is enough to
# switch `attack_model` later.
ARTIFACT_MODELS = ["llama-2-7b-chat-hf"]


def _dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"wrote {path}")


def fetch_behaviors() -> None:
    import jailbreakbench as jbb

    for split in ("harmful", "benign"):
        df = jbb.read_dataset(split=split).as_dataframe()
        records = df.reset_index(names="index").to_dict("records")
        _dump(CACHE_DIR / f"behaviors_{split}.json", records)


def fetch_artifacts() -> None:
    import jailbreakbench as jbb

    for model_name in ARTIFACT_MODELS:
        for method in ARTIFACT_METHODS:
            artifact = jbb.read_artifact(method=method, model_name=model_name)
            records = [j.model_dump() for j in artifact.jailbreaks]
            _dump(CACHE_DIR / f"artifact_{method}_{model_name}.json", records)


def main() -> None:
    fetch_behaviors()
    fetch_artifacts()


if __name__ == "__main__":
    main()
