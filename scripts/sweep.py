"""Distributes a list of run configs across free GPUs — one
`run_analysis.py` subprocess per GPU (via `CUDA_VISIBLE_DEVICES`), a
simple queue, per-run log files, and clean handling of failures and
Ctrl+C.

Usage:
    python scripts/sweep.py [path/to/sweep.yaml]

Defaults to configs/sweep_example/sweep.yaml — 4 dummy runs proving the
parallel-dispatch mechanics work, not a real research sweep.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SWEEP_FILE = REPO_ROOT / "configs" / "sweep_example" / "sweep.yaml"
LOG_DIR = REPO_ROOT / "logs" / "sweep"
RUN_ANALYSIS = REPO_ROOT / "scripts" / "run_analysis.py"
POLL_INTERVAL_SECONDS = 1.0
TERMINATE_GRACE_SECONDS = 10.0


@dataclass
class RunSpec:
    name: str
    args: dict[str, str]

    def cli_args(self) -> list[str]:
        return [f"{key}={value}" for key, value in self.args.items()]


@dataclass
class RunHandle:
    spec: RunSpec
    gpu: int
    process: subprocess.Popen
    log_path: Path
    log_file: object
    start_time: float


def load_sweep(path: Path) -> list[RunSpec]:
    data = yaml.safe_load(path.read_text())
    return [RunSpec(name=r["name"], args=r["args"]) for r in data["runs"]]


def available_gpu_slots() -> list[int]:
    """GPU indices to distribute runs over. -1 means "no GPU" (forces CPU
    via CUDA_VISIBLE_DEVICES=-1) and is used as the sole slot when no GPU
    is visible, so the sweep mechanics still work on a CPU-only machine."""
    if not torch.cuda.is_available():
        return [-1]
    return list(range(torch.cuda.device_count()))


def launch(spec: RunSpec, gpu: int) -> RunHandle:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{spec.name}.log"
    log_file = open(log_path, "w")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    process = subprocess.Popen(
        [sys.executable, str(RUN_ANALYSIS), *spec.cli_args()],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return RunHandle(
        spec=spec,
        gpu=gpu,
        process=process,
        log_path=log_path,
        log_file=log_file,
        start_time=time.time(),
    )


def main() -> None:
    sweep_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SWEEP_FILE
    queue = load_sweep(sweep_file)
    gpu_slots = available_gpu_slots()

    free_gpus = list(gpu_slots)
    running: list[RunHandle] = []
    results: dict[str, str] = {}
    interrupted = False

    def handle_sigint(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handle_sigint)

    print(f"sweep: {len(queue)} run(s) queued across {len(gpu_slots)} GPU slot(s) {gpu_slots}")
    print(f"logs: {LOG_DIR}")

    try:
        while (queue or running) and not interrupted:
            while queue and free_gpus:
                spec = queue.pop(0)
                gpu = free_gpus.pop(0)
                handle = launch(spec, gpu)
                running.append(handle)
                pid = handle.process.pid
                print(f"[{spec.name}] gpu={gpu} pid={pid} -> {handle.log_path.name}")

            still_running = []
            for handle in running:
                ret = handle.process.poll()
                if ret is None:
                    still_running.append(handle)
                    continue
                handle.log_file.close()
                elapsed = time.time() - handle.start_time
                status = "ok" if ret == 0 else f"failed (exit {ret})"
                results[handle.spec.name] = status
                print(f"[{handle.spec.name}] finished in {elapsed:.1f}s: {status}")
                free_gpus.append(handle.gpu)
            running = still_running

            if running or queue:
                time.sleep(POLL_INTERVAL_SECONDS)
    finally:
        if interrupted and running:
            print("\nCtrl+C received, terminating running run(s)...")
            for handle in running:
                handle.process.terminate()
            deadline = time.time() + TERMINATE_GRACE_SECONDS
            for handle in running:
                remaining = max(0.0, deadline - time.time())
                try:
                    handle.process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    handle.process.kill()
                    handle.process.wait()
                handle.log_file.close()
                results[handle.spec.name] = "interrupted"

    print("\nsweep summary:")
    for name, status in results.items():
        print(f"  {name}: {status}")

    if any(status != "ok" for status in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
