"""Immutable configuration for the approved ProTrek TopK SAE pilot."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SaePilotConfig:
    input_dim: int = 1024
    width: int = 8192
    k: int = 32
    aux_k: int = 256
    aux_coefficient: float = 1.0 / 32.0
    activation_threshold: float = 1e-3
    aspect_alpha: float = 0.75
    batch_size: int = 512
    seed: int = 42
    max_epochs: int = 20
    patience: int = 3
    min_delta: float = 1e-4
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 6.25e-10
    weight_decay: float = 0.0
    max_train_units: int | None = None
    max_validation_units: int | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "input_dim",
            "width",
            "k",
            "aux_k",
            "batch_size",
            "seed",
            "max_epochs",
            "patience",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        for name in ("input_dim", "width", "batch_size", "max_epochs", "patience"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not 0 < self.k <= self.width:
            raise ValueError("k must be positive and no greater than width")
        if not 0 < self.aux_k <= self.width:
            raise ValueError("aux_k must be positive and no greater than width")
        if not 0 < self.patience <= self.max_epochs:
            raise ValueError("patience must be positive and no greater than max_epochs")
        for name in ("max_train_units", "max_validation_units"):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise TypeError(f"{name} must be an integer when provided")
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")

        float_fields = (
            "aux_coefficient",
            "activation_threshold",
            "aspect_alpha",
            "min_delta",
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "weight_decay",
        )
        for name in float_fields:
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a float")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("aux_coefficient", "learning_rate", "adam_epsilon"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("activation_threshold", "min_delta", "weight_decay"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 <= self.aspect_alpha <= 1.0:
            raise ValueError("aspect_alpha must be between zero and one")
        for name in ("adam_beta1", "adam_beta2"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be at least zero and less than one")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
