"""Streaming evaluation metrics and baselines for the ProTrek SAE pilot."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Iterable

import numpy as np
import sklearn
import torch
from sklearn.decomposition import IncrementalPCA

from .sae_training_data import ActivationBatch
from .topk_sae import normalized_squared_error


_L0_QUANTILES = (
    ("p0", 0.0),
    ("p25", 0.25),
    ("p50", 0.5),
    ("p75", 0.75),
    ("p100", 1.0),
)


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_float32_matrix(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype float32")
    if value.ndim != 2:
        raise ValueError(f"{name} must be a rank-two tensor")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must have nonzero dimensions")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _require_aspect_indices(
    aspect_indices: torch.Tensor,
    *,
    batch_size: int,
    num_aspects: int,
) -> None:
    if not isinstance(aspect_indices, torch.Tensor):
        raise TypeError("aspect_indices must be a tensor")
    if aspect_indices.dtype != torch.long:
        raise TypeError("aspect_indices must have dtype int64")
    if tuple(aspect_indices.shape) != (batch_size,):
        raise ValueError("aspect_indices must have shape [B]")
    if torch.any(aspect_indices < 0).item() or torch.any(
        aspect_indices >= num_aspects
    ).item():
        raise ValueError("aspect_indices contains an out-of-range index")


def _linear_quantiles(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    last = len(ordered) - 1
    result: dict[str, float] = {}
    for name, quantile in _L0_QUANTILES:
        position = quantile * last
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        result[name] = float(
            ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
        )
    return result


@dataclass
class _ReconstructionContribution:
    count: int
    sum_nmse: float
    sum_squared_error: float
    sum_input_vector: torch.Tensor
    sum_input_element_squared: float
    sum_cosine: float
    l0_values: list[int]


@dataclass
class _ReconstructionTotals:
    count: int
    sum_nmse: float
    sum_squared_error: float
    sum_input_vector: torch.Tensor
    sum_input_element_squared: float
    sum_cosine: float
    new_l0_values: list[int]


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains a non-finite value")


def _require_finite_scalar(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} is non-finite")


@dataclass
class _ReconstructionState:
    count: int = 0
    sum_nmse: float = 0.0
    sum_squared_error: float = 0.0
    sum_input_vector: torch.Tensor | None = None
    sum_input_element_squared: float = 0.0
    sum_cosine: float = 0.0
    l0_values: list[int] = field(default_factory=list)

    @staticmethod
    def prepare(
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        activations: torch.Tensor,
    ) -> _ReconstructionContribution:
        if len(inputs) == 0:
            raise ValueError("Cannot prepare an empty reconstruction batch")
        inputs64 = inputs.detach().to(device="cpu", dtype=torch.float64)
        reconstructions64 = reconstructions.detach().to(
            device="cpu", dtype=torch.float64
        )
        _require_finite_tensor("inputs64", inputs64)
        _require_finite_tensor("reconstructions64", reconstructions64)
        squared_errors = (inputs64 - reconstructions64).pow(2).sum(dim=-1)
        input_energy = inputs64.pow(2).sum(dim=-1)
        errors = normalized_squared_error(inputs64, reconstructions64)
        input_norm = inputs64.norm(dim=-1)
        reconstruction_norm = reconstructions64.norm(dim=-1)
        cosine_denominator = (input_norm * reconstruction_norm).clamp_min(1e-8)
        cosine_numerator = (inputs64 * reconstructions64).sum(dim=-1)
        cosine = cosine_numerator / cosine_denominator
        input_sum = inputs64.sum(dim=0)
        input_element_squared = inputs64.pow(2).sum()
        l0_tensor = (activations.detach() > 0).sum(
            dim=-1, dtype=torch.int64
        ).to(device="cpu")
        for name, value in (
            ("squared errors", squared_errors),
            ("input energy", input_energy),
            ("normalized squared errors", errors),
            ("input norms", input_norm),
            ("reconstruction norms", reconstruction_norm),
            ("cosine denominators", cosine_denominator),
            ("cosine numerators", cosine_numerator),
            ("cosine similarities", cosine),
            ("input vector sum", input_sum),
            ("input element squared sum", input_element_squared),
        ):
            _require_finite_tensor(name, value)
        l0_values = [int(value) for value in l0_tensor.tolist()]
        if len(l0_values) != len(inputs) or any(value < 0 for value in l0_values):
            raise ValueError("realized L0 values must be nonnegative integers")
        contribution = _ReconstructionContribution(
            count=len(inputs),
            sum_nmse=float(errors.sum().item()),
            sum_squared_error=float(squared_errors.sum().item()),
            sum_input_vector=input_sum,
            sum_input_element_squared=float(input_element_squared.item()),
            sum_cosine=float(cosine.sum().item()),
            l0_values=l0_values,
        )
        for name, value in (
            ("NMSE sum", contribution.sum_nmse),
            ("squared-error sum", contribution.sum_squared_error),
            ("input-square sum", contribution.sum_input_element_squared),
            ("cosine sum", contribution.sum_cosine),
        ):
            _require_finite_scalar(name, value)
        return contribution

    def validate(
        self, contribution: _ReconstructionContribution
    ) -> _ReconstructionTotals:
        input_vector = (
            contribution.sum_input_vector.clone()
            if self.sum_input_vector is None
            else self.sum_input_vector + contribution.sum_input_vector
        )
        totals = _ReconstructionTotals(
            count=self.count + contribution.count,
            sum_nmse=self.sum_nmse + contribution.sum_nmse,
            sum_squared_error=self.sum_squared_error
            + contribution.sum_squared_error,
            sum_input_vector=input_vector,
            sum_input_element_squared=self.sum_input_element_squared
            + contribution.sum_input_element_squared,
            sum_cosine=self.sum_cosine + contribution.sum_cosine,
            new_l0_values=contribution.l0_values,
        )
        _require_finite_tensor("cumulative input vector sum", input_vector)
        for name, value in (
            ("cumulative NMSE sum", totals.sum_nmse),
            ("cumulative squared-error sum", totals.sum_squared_error),
            ("cumulative input-square sum", totals.sum_input_element_squared),
            ("cumulative cosine sum", totals.sum_cosine),
        ):
            _require_finite_scalar(name, value)
        if (
            totals.count <= 0
            or len(self.l0_values) + len(contribution.l0_values) != totals.count
        ):
            raise ValueError("invalid cumulative reconstruction count")
        return totals

    def commit(self, totals: _ReconstructionTotals) -> None:
        self.count = totals.count
        self.sum_nmse = totals.sum_nmse
        self.sum_squared_error = totals.sum_squared_error
        self.sum_input_vector = totals.sum_input_vector
        self.sum_input_element_squared = totals.sum_input_element_squared
        self.sum_cosine = totals.sum_cosine
        self.l0_values.extend(totals.new_l0_values)

    def add(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        activations: torch.Tensor,
    ) -> None:
        contribution = self.prepare(inputs, reconstructions, activations)
        self.commit(self.validate(contribution))

    def summary(self) -> dict[str, object]:
        if self.count == 0:
            return {
                "count": 0,
                "nmse": None,
                "explained_variance": None,
                "cosine_similarity": None,
                "mean_l0": None,
                "l0_quantiles": None,
            }
        if self.sum_input_vector is None:
            raise AssertionError("Missing input-vector sum")
        squared_input_sum = float(self.sum_input_vector.pow(2).sum().item())
        _require_finite_scalar("squared input-vector sum", squared_input_sum)
        sst = self.sum_input_element_squared - squared_input_sum / self.count
        result = {
            "count": self.count,
            "nmse": self.sum_nmse / self.count,
            "explained_variance": 1.0
            - self.sum_squared_error / max(sst, 1e-8),
            "cosine_similarity": self.sum_cosine / self.count,
            "mean_l0": sum(self.l0_values) / self.count,
            "l0_quantiles": _linear_quantiles(self.l0_values),
        }
        for name in ("nmse", "explained_variance", "cosine_similarity", "mean_l0"):
            _require_finite_scalar(name, result[name])  # type: ignore[arg-type]
        return result


class ReconstructionMetricsAccumulator:
    """Accumulate reconstruction metrics without retaining full input batches."""

    def __init__(self, num_aspects: int) -> None:
        _require_positive_integer("num_aspects", num_aspects)
        self.num_aspects = num_aspects
        self._overall = _ReconstructionState()
        self._per_aspect = [_ReconstructionState() for _ in range(num_aspects)]
        self._input_width: int | None = None
        self._activation_width: int | None = None
        self._finalized = False

    def update(
        self,
        *,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        activations: torch.Tensor,
        aspect_indices: torch.Tensor,
    ) -> None:
        if self._finalized:
            raise RuntimeError("Cannot update after finalization")
        _require_float32_matrix("inputs", inputs)
        _require_float32_matrix("reconstructions", reconstructions)
        _require_float32_matrix("activations", activations)
        if reconstructions.shape != inputs.shape:
            raise ValueError("reconstructions must have the same shape as inputs")
        if activations.shape[0] != inputs.shape[0]:
            raise ValueError("activations must have the same batch size as inputs")
        _require_aspect_indices(
            aspect_indices,
            batch_size=len(inputs),
            num_aspects=self.num_aspects,
        )
        if self._input_width is not None and inputs.shape[1] != self._input_width:
            raise ValueError("input width changed between updates")
        if (
            self._activation_width is not None
            and activations.shape[1] != self._activation_width
        ):
            raise ValueError("activation width changed between updates")
        pending: list[
            tuple[_ReconstructionState, _ReconstructionContribution]
        ] = [
            (
                self._overall,
                self._overall.prepare(inputs, reconstructions, activations),
            )
        ]
        aspects_cpu = aspect_indices.detach().to(device="cpu")
        for aspect in range(self.num_aspects):
            mask_cpu = aspects_cpu == aspect
            if not bool(mask_cpu.any().item()):
                continue
            indices = mask_cpu.nonzero(as_tuple=False).flatten().to(inputs.device)
            state = self._per_aspect[aspect]
            pending.append(
                (
                    state,
                    state.prepare(
                        inputs.index_select(0, indices),
                        reconstructions.index_select(
                            0, indices.to(reconstructions.device)
                        ),
                        activations.index_select(0, indices.to(activations.device)),
                    ),
                )
            )
        validated = [(state, state.validate(value)) for state, value in pending]
        for state, totals in validated:
            state.commit(totals)
        self._input_width = inputs.shape[1]
        self._activation_width = activations.shape[1]

    def finalize(self) -> dict[str, object]:
        if self._finalized:
            raise RuntimeError("Metrics have already been finalized")
        if self._overall.count == 0:
            raise ValueError("Cannot finalize metrics without samples")
        result = self._overall.summary()
        result["per_aspect"] = {
            str(aspect): state.summary()
            for aspect, state in enumerate(self._per_aspect)
        }
        self._finalized = True
        return result


class FeatureStatsAccumulator:
    """Accumulate fixed-size CPU feature and aspect firing statistics."""

    def __init__(
        self,
        width: int,
        num_aspects: int,
        activation_threshold: float = 1e-3,
    ) -> None:
        _require_positive_integer("width", width)
        _require_positive_integer("num_aspects", num_aspects)
        if isinstance(activation_threshold, bool) or not isinstance(
            activation_threshold, Real
        ):
            raise TypeError("activation_threshold must be a real number")
        activation_threshold = float(activation_threshold)
        if not math.isfinite(activation_threshold):
            raise ValueError("activation_threshold must be finite")
        if activation_threshold != 1e-3:
            raise ValueError("activation_threshold is fixed at exactly 1e-3")
        self.width = width
        self.num_aspects = num_aspects
        self.activation_threshold = activation_threshold
        self.activation_count = torch.zeros(width, dtype=torch.int64)
        self.activation_sum = torch.zeros(width, dtype=torch.float64)
        self.activation_sum_squared = torch.zeros(width, dtype=torch.float64)
        self.activation_max = torch.zeros(width, dtype=torch.float32)
        self.aspect_activation_count = torch.zeros(
            width, num_aspects, dtype=torch.int64
        )
        self.aspect_unit_count = torch.zeros(num_aspects, dtype=torch.int64)
        self._sample_count = 0
        self._finalized = False

    def update(
        self,
        activations: torch.Tensor,
        aspect_indices: torch.Tensor,
    ) -> None:
        if self._finalized:
            raise RuntimeError("Cannot update after finalization")
        _require_float32_matrix("activations", activations)
        if activations.shape[1] != self.width:
            raise ValueError(f"activations must have shape [B, {self.width}]")
        _require_aspect_indices(
            aspect_indices,
            batch_size=len(activations),
            num_aspects=self.num_aspects,
        )
        values = activations.detach().to(device="cpu")
        aspects = aspect_indices.detach().to(device="cpu")
        firing = values > self.activation_threshold
        firing_values = values.masked_fill(~firing, 0.0)
        self.activation_count.add_(firing.sum(dim=0, dtype=torch.int64))
        self.activation_sum.add_(firing_values.double().sum(dim=0))
        self.activation_sum_squared.add_(firing_values.double().pow(2).sum(dim=0))
        self.activation_max.copy_(
            torch.maximum(self.activation_max, firing_values.max(dim=0).values)
        )
        self.aspect_unit_count.add_(
            torch.bincount(aspects, minlength=self.num_aspects)
        )
        for aspect in range(self.num_aspects):
            mask = aspects == aspect
            if bool(mask.any().item()):
                self.aspect_activation_count[:, aspect].add_(
                    firing[mask].sum(dim=0, dtype=torch.int64)
                )
        self._sample_count += len(values)

    def finalize(self) -> dict[str, object]:
        if self._finalized:
            raise RuntimeError("Feature statistics have already been finalized")
        if self._sample_count == 0:
            raise ValueError("Cannot finalize feature statistics without samples")
        self._finalized = True

        alive = self.activation_count > 0
        safe_count = self.activation_count.clamp_min(1).double()
        means = self.activation_sum / safe_count
        variances = (
            self.activation_sum_squared / safe_count - means.pow(2)
        ).clamp_min(0.0)
        means.masked_fill_(~alive, 0.0)
        variances.masked_fill_(~alive, 0.0)
        natural_frequency = self.aspect_unit_count.double() / self._sample_count
        features: list[dict[str, object]] = []
        for feature_index in range(self.width):
            count = int(self.activation_count[feature_index].item())
            if count == 0:
                features.append(
                    {
                        "feature_index": feature_index,
                        "activation_count": 0,
                        "aspect_activation_counts": [0] * self.num_aspects,
                        "mean_on_firing": 0.0,
                        "variance_on_firing": 0.0,
                        "max_activation": 0.0,
                        "normalized_aspect_entropy": 0.0,
                        "purity": 0.0,
                        "preferred_aspect": -1,
                        "preferred_aspect_lift": 0.0,
                        "max_aspect_lift": 0.0,
                        "field_detector_warning": False,
                    }
                )
                continue
            conditional = self.aspect_activation_count[feature_index].double() / count
            positive = conditional > 0
            entropy = float(
                -(conditional[positive] * conditional[positive].log()).sum().item()
            )
            if self.num_aspects > 1:
                entropy /= math.log(self.num_aspects)
            else:
                entropy = 0.0
            preferred = int(torch.argmax(conditional).item())
            purity = float(conditional[preferred].item())
            lifts = torch.where(
                natural_frequency > 0,
                conditional / natural_frequency.clamp_min(torch.finfo(torch.float64).tiny),
                torch.zeros_like(conditional),
            )
            preferred_lift = float(lifts[preferred].item())
            maximum_lift = float(lifts.max().item())
            features.append(
                {
                    "feature_index": feature_index,
                    "activation_count": count,
                    "aspect_activation_counts": [
                        int(value)
                        for value in self.aspect_activation_count[feature_index].tolist()
                    ],
                    "mean_on_firing": float(means[feature_index].item()),
                    "variance_on_firing": float(variances[feature_index].item()),
                    "max_activation": float(self.activation_max[feature_index].item()),
                    "normalized_aspect_entropy": entropy,
                    "purity": purity,
                    "preferred_aspect": preferred,
                    "preferred_aspect_lift": preferred_lift,
                    "max_aspect_lift": maximum_lift,
                    "field_detector_warning": bool(
                        count >= 20 and purity >= 0.80 and preferred_lift >= 2.0
                    ),
                }
            )
        dead_count = int((~alive).sum().item())
        return {
            "count": self._sample_count,
            "width": self.width,
            "num_aspects": self.num_aspects,
            "activation_threshold": self.activation_threshold,
            "dead_feature_count": dead_count,
            "dead_fraction": dead_count / self.width,
            "features": features,
        }


def _validate_natural_batch(
    batch: ActivationBatch,
    *,
    expected_width: int | None,
) -> int:
    if not isinstance(batch, ActivationBatch):
        raise TypeError("streams must yield ActivationBatch values")
    _require_float32_matrix("batch.embeddings", batch.embeddings)
    rows, width = batch.embeddings.shape
    if expected_width is not None and width != expected_width:
        raise ValueError("embedding width changed between batches")
    if not isinstance(batch.unit_ids, tuple) or len(batch.unit_ids) != rows:
        raise ValueError("batch.unit_ids must be a tuple with one ID per row")
    if any(not isinstance(unit_id, str) or not unit_id for unit_id in batch.unit_ids):
        raise ValueError("batch.unit_ids must contain nonempty strings")
    if not isinstance(batch.aspect_indices, torch.Tensor):
        raise TypeError("batch.aspect_indices must be a tensor")
    if batch.aspect_indices.dtype != torch.long:
        raise TypeError("batch.aspect_indices must have dtype int64")
    if tuple(batch.aspect_indices.shape) != (rows,):
        raise ValueError("batch.aspect_indices must have shape [B]")
    if torch.any(batch.aspect_indices < 0).item():
        raise ValueError("batch.aspect_indices must be nonnegative")
    if batch.loss_weights is not None:
        raise ValueError("baseline streams must be natural and unweighted")
    return width


def _streaming_training_mean(
    training_batches: Iterable[ActivationBatch],
) -> tuple[torch.Tensor, int, int]:
    count = 0
    width: int | None = None
    total: torch.Tensor | None = None
    for batch in training_batches:
        width = _validate_natural_batch(batch, expected_width=width)
        values = batch.embeddings.detach().to(device="cpu", dtype=torch.float64)
        if total is None:
            total = values.sum(dim=0)
        else:
            total.add_(values.sum(dim=0))
        count += len(values)
    if count == 0 or total is None or width is None:
        raise ValueError("training stream must contain samples")
    return total / count, count, width


def evaluate_mean_baseline(
    training_batches: Iterable[ActivationBatch],
    validation_batches: Iterable[ActivationBatch],
    *,
    training_mean: torch.Tensor | None = None,
) -> dict[str, object]:
    """Evaluate the natural validation stream against the natural training mean."""

    if training_mean is None:
        mean64, _, width = _streaming_training_mean(training_batches)
    else:
        if not isinstance(training_mean, torch.Tensor):
            raise TypeError("training_mean must be a tensor")
        if training_mean.device.type != "cpu":
            raise ValueError("training_mean must be on CPU")
        if training_mean.dtype != torch.float32:
            raise TypeError("training_mean must have dtype float32")
        if training_mean.ndim != 1 or training_mean.numel() == 0:
            raise ValueError("training_mean must be a nonempty rank-one tensor")
        if not bool(torch.isfinite(training_mean).all().item()):
            raise ValueError("training_mean must contain only finite values")
        width = int(training_mean.shape[0])
        mean64 = training_mean.detach().to(dtype=torch.float64)
    count = 0
    sum_nmse = 0.0
    for batch in validation_batches:
        _validate_natural_batch(batch, expected_width=width)
        values = batch.embeddings.detach().to(device="cpu", dtype=torch.float64)
        errors = (values - mean64).pow(2).sum(dim=-1) / values.pow(2).sum(
            dim=-1
        ).clamp_min(1e-8)
        sum_nmse += float(errors.sum().item())
        count += len(values)
    if count == 0:
        raise ValueError("validation stream must contain samples")
    return {"baseline": "training_mean", "count": count, "nmse": sum_nmse / count}


def _array_sha256(values: np.ndarray | torch.Tensor) -> str:
    """Hash the stable, contiguous raw bytes of an ndarray or CPU tensor."""

    if isinstance(values, torch.Tensor):
        array = values.detach().to(device="cpu").contiguous().numpy()
    elif isinstance(values, np.ndarray):
        array = values
    else:
        raise TypeError("values must be an ndarray or tensor")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def fit_pca32_baseline(
    training_batches: Iterable[ActivationBatch],
    validation_batches: Iterable[ActivationBatch],
    *,
    batch_size: int,
) -> dict[str, object]:
    """Fit IncrementalPCA-32 on natural training data and stream evaluation."""

    _require_positive_integer("batch_size", batch_size)
    component_count = 32
    fit_chunk_size = max(component_count, batch_size)
    model = IncrementalPCA(
        n_components=component_count,
        batch_size=max(4096, batch_size),
    )
    width: int | None = None
    carry: np.ndarray | None = None
    fitted_count = 0
    for batch in training_batches:
        width = _validate_natural_batch(batch, expected_width=width)
        if width < component_count:
            raise ValueError("PCA-32 requires at least 32 embedding dimensions")
        values = np.ascontiguousarray(
            batch.embeddings.detach().to(device="cpu").numpy()
        )
        carry = values if carry is None else np.concatenate((carry, values), axis=0)
        while len(carry) >= fit_chunk_size + component_count:
            fitting = carry[:fit_chunk_size]
            model.partial_fit(fitting)
            fitted_count += len(fitting)
            carry = carry[fit_chunk_size:]
    if width is None or carry is None:
        raise ValueError("training stream must contain samples")
    if len(carry) < component_count:
        raise ValueError("PCA-32 requires at least 32 training samples")
    model.partial_fit(carry)
    fitted_count += len(carry)

    validation_count = 0
    sum_nmse = 0.0
    for batch in validation_batches:
        _validate_natural_batch(batch, expected_width=width)
        values = np.ascontiguousarray(
            batch.embeddings.detach().to(device="cpu").numpy()
        )
        reconstructions = model.inverse_transform(model.transform(values))
        values64 = values.astype(np.float64, copy=False)
        reconstructions64 = np.asarray(reconstructions, dtype=np.float64)
        errors = np.square(values64 - reconstructions64).sum(axis=-1) / np.maximum(
            np.square(values64).sum(axis=-1), 1e-8
        )
        if not np.isfinite(errors).all():
            raise ValueError("PCA validation metrics contain non-finite values")
        sum_nmse += float(errors.astype(np.float64).sum())
        validation_count += len(values)
    if validation_count == 0:
        raise ValueError("validation stream must contain samples")
    if fitted_count < component_count:
        raise AssertionError("PCA was fitted with too few samples")
    return {
        "baseline": "incremental_pca32",
        "count": validation_count,
        "nmse": sum_nmse / validation_count,
        "n_components": component_count,
        "components_sha256": _array_sha256(model.components_),
        "sklearn_version": sklearn.__version__,
    }


def evaluate_pilot_gates(
    *,
    validation_nmse: float,
    mean_baseline_nmse: float,
    explained_variance: float,
    mean_l0: float,
    train_dead_fraction: float,
    split_valid: bool,
    resume_valid: bool,
    artifacts_valid: bool,
    all_finite: bool,
) -> dict[str, object]:
    """Evaluate every pilot acceptance gate without short-circuiting failures."""

    def finite_float(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    validation_nmse = finite_float("validation_nmse", validation_nmse)
    mean_baseline_nmse = finite_float(
        "mean_baseline_nmse", mean_baseline_nmse
    )
    explained_variance = finite_float("explained_variance", explained_variance)
    mean_l0 = finite_float("mean_l0", mean_l0)
    train_dead_fraction = finite_float(
        "train_dead_fraction", train_dead_fraction
    )
    invariants = {
        "split_valid": split_valid,
        "resume_valid": resume_valid,
        "artifacts_valid": artifacts_valid,
        "all_finite": all_finite,
    }
    for name, value in invariants.items():
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
    gates = {
        "beats_mean_baseline": bool(validation_nmse < mean_baseline_nmse),
        "positive_explained_variance": bool(explained_variance > 0.0),
        "l0_in_range": bool(31.0 <= mean_l0 <= 32.0),
        "dead_fraction_below_limit": bool(train_dead_fraction < 0.20),
        **invariants,
    }
    return {"passed": bool(all(gates.values())), "gates": gates}
