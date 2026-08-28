import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_analysis import parse_overrides  # noqa: E402


class TestParseOverrides:
    def test_parses_key_value_pairs(self):
        assert parse_overrides(["analysis=residual_capture", "model=gpt2-small"]) == {
            "analysis": "residual_capture",
            "model": "gpt2-small",
        }

    def test_value_may_contain_equals(self):
        assert parse_overrides(["output=a=b.safetensors"]) == {"output": "a=b.safetensors"}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError):
            parse_overrides(["not-a-kv-pair"])


class TestRunAnalysisEndToEnd:
    """Matches the Definition-of-Done smoke test literally: forces CPU
    (even on a machine with GPUs visible) and checks it finishes well
    under 2 minutes and saves a real artifact."""

    def test_residual_capture_gpt2_small_on_cpu(self, tmp_path):
        output_path = tmp_path / "residual_capture_gpt2-small.safetensors"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU regardless of host GPUs

        start = time.time()
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_analysis.py"),
                "analysis=residual_capture",
                "model=gpt2-small",
                f"output={output_path}",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.time() - start

        assert result.returncode == 0, result.stdout + result.stderr
        assert elapsed < 120, f"took {elapsed:.1f}s, DoD budget is 120s"
        assert output_path.exists()

        tensors = load_file(str(output_path))
        assert len(tensors) == 12  # gpt2-small has 12 layers, layers=null -> all
        for tensor in tensors.values():
            assert tensor.shape[1] == 768  # gpt2-small d_model
