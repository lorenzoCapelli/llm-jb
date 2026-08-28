import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import sweep  # noqa: E402


class TestLoadSweep:
    def test_loads_example_sweep_file(self):
        runs = sweep.load_sweep(REPO_ROOT / "configs" / "sweep_example" / "sweep.yaml")
        assert len(runs) == 4
        names = [r.name for r in runs]
        assert names == ["run_0", "run_1", "run_2", "run_3"]
        assert runs[0].args["analysis"] == "residual_capture"

    def test_run_spec_cli_args_format(self):
        spec = sweep.RunSpec(name="x", args={"model": "gpt2-small", "analysis": "residual_capture"})
        assert set(spec.cli_args()) == {"model=gpt2-small", "analysis=residual_capture"}


class TestAvailableGpuSlots:
    def test_no_gpu_returns_single_cpu_slot(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert sweep.available_gpu_slots() == [-1]

    def test_gpus_return_index_list(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
        assert sweep.available_gpu_slots() == [0, 1, 2]


class TestSweepEndToEnd:
    """Real subprocess orchestration on forced CPU-only slots (monkeypatched),
    so this runs the same on a GPU box or a laptop: two dummy smoke runs,
    one of them made to fail deliberately, distributed over 2 CPU slots."""

    def test_runs_queue_and_collects_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sweep, "available_gpu_slots", lambda: [-1, -1])
        monkeypatch.setattr(sweep, "LOG_DIR", tmp_path / "logs")

        sweep_file = tmp_path / "sweep.yaml"
        sweep_file.write_text(f"""
runs:
  - name: ok_run
    args:
      analysis: residual_capture
      model: gpt2-small
      dataset: smoke
      output: {tmp_path / "ok.safetensors"}
  - name: bad_run
    args:
      analysis: residual_capture
      model: does-not-exist
      dataset: smoke
""")

        monkeypatch.setattr(sys, "argv", ["sweep.py", str(sweep_file)])
        env_backup = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            with __import__("pytest").raises(SystemExit) as exc_info:
                sweep.main()
            assert exc_info.value.code == 1  # bad_run failed -> nonzero exit
        finally:
            if env_backup is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = env_backup

        assert (tmp_path / "ok.safetensors").exists()
        assert (tmp_path / "logs" / "ok_run.log").exists()
        assert (tmp_path / "logs" / "bad_run.log").exists()
        assert "unknown" in (tmp_path / "logs" / "bad_run.log").read_text().lower() or (
            "error" in (tmp_path / "logs" / "bad_run.log").read_text().lower()
        )
