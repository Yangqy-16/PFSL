"""CPU-only, atomic persistence helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from pfsl_process.sae.sae_checkpointing import (
    write_json_atomic,
    write_torch_atomic,
)


def _to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {_to_cpu(key): _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, set):
        return {_to_cpu(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_to_cpu(item) for item in value)
    return value


def save_cpu_torch(path: Path, payload: object) -> None:
    """Recursively detach tensors to CPU and atomically persist the tree."""

    write_torch_atomic(Path(path), _to_cpu(payload))


def save_json(path: Path, payload: object) -> None:
    """Atomically persist one canonical, sorted JSON representation."""

    write_json_atomic(Path(path), payload)
