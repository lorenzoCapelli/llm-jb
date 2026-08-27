"""Where captured activations live during a run, and how a finished
capture gets persisted as an artifact on disk.

Two disk formats are supported, matching different needs: safetensors
preserves dtype exactly (including bf16/fp16) and is simplest for a single
one-shot dict of tensors; zarr round-trips through numpy (no bfloat16
support, so bf16 tensors are upcast to float32) but suits a chunked,
incrementally-written on-disk store better. Pick one per use case — this
module doesn't force either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import zarr
from safetensors.torch import load_file, save_file

ActivationPlacement = Literal["gpu", "cpu"]


def place_activation(tensor: torch.Tensor, placement: ActivationPlacement) -> torch.Tensor:
    if placement == "cpu":
        return tensor.cpu()
    if placement == "gpu":
        return tensor
    raise ValueError(f"unknown placement: {placement}")


def save_activations_safetensors(activations: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in activations.items()}, str(path))


def load_activations_safetensors(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


def save_activations_zarr(activations: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w")
    for name, tensor in activations.items():
        tensor = tensor.cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        group[name] = tensor.numpy()


def load_activations_zarr(path: Path) -> dict[str, torch.Tensor]:
    group = zarr.open_group(str(path), mode="r")
    return {name: torch.from_numpy(group[name][:]) for name in group.array_keys()}
