#!/usr/bin/env python3
"""Deterministic end-to-end trainer for the ProTrek TopK SAE pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pfsl_process.sae import sae_checkpointing as checkpointing
    from pfsl_process.sae.extract_protrek_activations import (
        DEFAULT_OUTPUT_DIR as DEFAULT_ACTIVATION_DIR,
        DEFAULT_UNITS_PATH,
    )
    from pfsl_process.sae.sae_metrics import (
        FeatureStatsAccumulator,
        ReconstructionMetricsAccumulator,
        evaluate_mean_baseline,
        evaluate_pilot_gates,
        fit_pca32_baseline,
    )
    from pfsl_process.sae.sae_pilot_config import SaePilotConfig
    from pfsl_process.sae.sae_training_data import (
        ActivationBatch,
        SaeDataIndex,
        build_data_index,
        build_split_manifest,
        compute_aspect_weights,
        iter_split_batches,
        limit_data_index,
        stream_train_mean,
    )
    from pfsl_process.sae.text_templates import ASPECT_NAMES
    from pfsl_process.sae.topk_sae import (
        TopKSAE,
        assert_finite_parameters_and_gradients,
        compute_sae_loss,
        normalize_decoder_,
        project_decoder_gradients_,
    )
else:
    from . import sae_checkpointing as checkpointing
    from .extract_protrek_activations import (
        DEFAULT_OUTPUT_DIR as DEFAULT_ACTIVATION_DIR,
        DEFAULT_UNITS_PATH,
    )
    from .sae_metrics import (
        FeatureStatsAccumulator,
        ReconstructionMetricsAccumulator,
        evaluate_mean_baseline,
        evaluate_pilot_gates,
        fit_pca32_baseline,
    )
    from .sae_pilot_config import SaePilotConfig
    from .sae_training_data import (
        ActivationBatch,
        SaeDataIndex,
        build_data_index,
        build_split_manifest,
        compute_aspect_weights,
        iter_split_batches,
        limit_data_index,
        stream_train_mean,
    )
    from .text_templates import ASPECT_NAMES
    from .topk_sae import (
        TopKSAE,
        assert_finite_parameters_and_gradients,
        compute_sae_loss,
        normalize_decoder_,
        project_decoder_gradients_,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT_MANIFEST_PATH = DEFAULT_UNITS_PATH.parent / "text_corpus_manifest.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/sae_runs/protrek_final_1024/topk_w8_k32_seed42"
)
_CUBLAS_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})
SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "config",
        "config_fingerprint",
        "provenance",
        "source_paths",
        "source_provenance",
        "source_split_counts",
        "effective_split_counts",
        "dead_steps_threshold",
        "baselines",
        "best_epoch",
        "final_epoch",
        "stopping_reason",
        "best_validation_metrics",
        "resume_self_check",
        "gates",
        "passed",
        "determinism",
        "files",
    }
)
RUN_CONTEXT_KEYS = frozenset(
    {"baselines", "resume_self_check", "determinism", "dead_steps_threshold"}
)
SELF_CHECK_KEYS = frozenset(
    {
        "passed",
        "uninterrupted_model_state_sha256",
        "resumed_model_state_sha256",
        "uninterrupted_optimizer_state_sha256",
        "resumed_optimizer_state_sha256",
        "uninterrupted_random_trace_sha256",
        "resumed_random_trace_sha256",
    }
)
DETERMINISM_KEYS = frozenset(
    {
        "seed",
        "numpy_seed",
        "torch_seed",
        "device",
        "torch_deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cublas_workspace_config",
    }
)
GATE_KEYS = frozenset(
    {
        "beats_mean_baseline",
        "positive_explained_variance",
        "l0_in_range",
        "dead_fraction_below_limit",
        "split_valid",
        "resume_valid",
        "artifacts_valid",
        "all_finite",
    }
)
RECONSTRUCTION_METRIC_KEYS = frozenset(
    {
        "count",
        "nmse",
        "explained_variance",
        "cosine_similarity",
        "mean_l0",
        "l0_quantiles",
        "per_aspect",
    }
)
TRAINING_METRIC_KEYS = RECONSTRUCTION_METRIC_KEYS | frozenset(
    {
        "loss",
        "main_loss",
        "auxiliary_loss",
        "optimizer_steps",
        "activation_magnitude",
        "dead_feature_fraction",
        "gradient_norm",
        "decoder_norm_max_error",
    }
)
VALIDATION_METRIC_KEYS = RECONSTRUCTION_METRIC_KEYS | frozenset(
    {"activation_magnitude", "dead_feature_fraction"}
)


class TrainerError(ValueError):
    """Raised when a pilot run cannot be trained or safely resumed."""


def _configure_cuda_determinism_environment(device: torch.device) -> None:
    if device.type != "cuda":
        return
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif configured not in _CUBLAS_WORKSPACE_CONFIGS:
        raise TrainerError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for CUDA"
        )


def _expected_determinism_record(
    seed: int, device: torch.device
) -> dict[str, object]:
    """Build the post-setup determinism record without touching RNG or Torch state."""

    if type(seed) is not int or seed < 0:
        raise TrainerError("seed must be a nonnegative integer")
    if not isinstance(device, torch.device):
        raise TypeError("device must be torch.device")
    return {
        "seed": seed,
        "numpy_seed": seed % (2**32),
        "torch_seed": seed % (2**63),
        "device": str(device),
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def setup_determinism(seed: int, device: torch.device) -> dict[str, object]:
    """Reset every training RNG and enable deterministic Torch execution."""

    if type(seed) is not int or seed < 0:
        raise TrainerError("seed must be a nonnegative integer")
    if not isinstance(device, torch.device):
        raise TypeError("device must be torch.device")
    _configure_cuda_determinism_environment(device)
    expected = _expected_determinism_record(seed, device)
    numpy_seed = int(expected["numpy_seed"])
    torch_seed = int(expected["torch_seed"])
    random.seed(seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise TrainerError("CUDA was requested but is unavailable")
        torch.cuda.manual_seed_all(torch_seed)
    observed = dict(expected)
    observed["torch_deterministic_algorithms"] = bool(
        torch.are_deterministic_algorithms_enabled()
    )
    observed["cudnn_deterministic"] = bool(
        getattr(torch.backends.cudnn, "deterministic", False)
    )
    observed["cudnn_benchmark"] = bool(
        getattr(torch.backends.cudnn, "benchmark", False)
    )
    if observed != expected:
        raise TrainerError("Torch determinism setup did not reach the expected state")
    return observed


def _natural_batches(
    data_index: SaeDataIndex,
    split: str,
    batch_size: int,
) -> Iterable[ActivationBatch]:
    return iter_split_batches(
        data_index,
        split=split,  # type: ignore[arg-type]
        batch_size=batch_size,
        aspect_weights=None,
        seed=0,
        epoch=0,
        shuffle=False,
        pin_memory=False,
    )


class _ActivationHealthAccumulator:
    def __init__(self, width: int, threshold: float) -> None:
        self.width = width
        self.threshold = threshold
        self.firing_count = 0
        self.firing_sum = 0.0
        self.maximum = 0.0
        self.active_features = torch.zeros(width, dtype=torch.bool)
        self.sample_count = 0

    def update(self, activations: torch.Tensor) -> None:
        values = activations.detach()
        if (
            values.dtype != torch.float32
            or values.ndim != 2
            or values.shape[1] != self.width
            or not bool(torch.isfinite(values).all().item())
        ):
            raise TrainerError("Activation health received invalid activations")
        firing = values > self.threshold
        count = int(firing.sum().item())
        self.firing_count += count
        if count:
            firing_values = values[firing]
            self.firing_sum += float(firing_values.double().sum().item())
            self.maximum = max(self.maximum, float(firing_values.max().item()))
        active_features = firing.any(dim=0).to(device="cpu")
        self.active_features.logical_or_(active_features)
        self.sample_count += int(values.shape[0])

    def finalize(self) -> dict[str, object]:
        if self.sample_count == 0:
            raise TrainerError("Activation health cannot finalize without samples")
        mean = self.firing_sum / self.firing_count if self.firing_count else 0.0
        dead_fraction = float((~self.active_features).sum().item()) / self.width
        values = (mean, self.maximum, dead_fraction)
        if not all(math.isfinite(value) for value in values):
            raise TrainerError("Activation health contains non-finite values")
        return {
            "activation_magnitude": {
                "firing_count": self.firing_count,
                "mean_on_firing": mean,
                "max": self.maximum,
            },
            "dead_feature_fraction": dead_fraction,
        }


def train_one_epoch(
    *,
    model: TopKSAE,
    optimizer: torch.optim.Optimizer,
    data_index: SaeDataIndex,
    config: SaePilotConfig,
    aspect_weights: torch.Tensor,
    device: torch.device,
    epoch: int,
) -> dict[str, object]:
    """Train one deterministic epoch and return exact-count streaming metrics."""

    if type(epoch) is not int or epoch < 0:
        raise TrainerError("epoch must be a nonnegative integer")
    model.train()
    accumulator = ReconstructionMetricsAccumulator(num_aspects=len(ASPECT_NAMES))
    observed_count = 0
    optimizer_steps = 0
    total_loss = 0.0
    total_main = 0.0
    total_auxiliary = 0.0
    activation_health = _ActivationHealthAccumulator(
        config.width, config.activation_threshold
    )
    gradient_norms: list[float] = []
    decoder_norm_errors: list[float] = []
    for batch in iter_split_batches(
        data_index,
        split="train",
        batch_size=config.batch_size,
        aspect_weights=aspect_weights,
        seed=config.seed,
        epoch=epoch,
        shuffle=True,
        pin_memory=device.type == "cuda",
    ):
        if batch.loss_weights is None:
            raise TrainerError("Training batches must contain global loss weights")
        inputs = batch.embeddings.to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        loss_weights = batch.loss_weights.to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        output = model(inputs, update_dead_stats=True)
        losses = compute_sae_loss(
            output,
            inputs,
            loss_weights,
            config.aux_coefficient,
        )
        if not bool(torch.isfinite(losses.total).item()):
            raise TrainerError("Training loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        project_decoder_gradients_(model)
        assert_finite_parameters_and_gradients(model)
        squared_gradient_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                squared_gradient_norm += float(
                    parameter.grad.detach().double().pow(2).sum().item()
                )
        gradient_norm = math.sqrt(squared_gradient_norm)
        if not math.isfinite(gradient_norm):
            raise TrainerError("Gradient norm is non-finite")
        gradient_norms.append(gradient_norm)
        optimizer.step()
        normalize_decoder_(model)
        assert_finite_parameters_and_gradients(model)
        decoder_error = float(
            (model.decoder_weight.detach().norm(dim=0) - 1.0).abs().max().item()
        )
        if not math.isfinite(decoder_error):
            raise TrainerError("Decoder norm error is non-finite")
        decoder_norm_errors.append(decoder_error)

        activation_health.update(output.activations)
        accumulator.update(
            inputs=inputs.detach(),
            reconstructions=output.reconstruction.detach(),
            activations=output.activations.detach(),
            aspect_indices=batch.aspect_indices,
        )
        rows = int(inputs.shape[0])
        observed_count += rows
        optimizer_steps += 1
        total_loss += float(losses.total.detach().cpu().item()) * rows
        total_main += float(losses.main.detach().cpu().item()) * rows
        total_auxiliary += float(losses.auxiliary.detach().cpu().item()) * rows
    if observed_count != data_index.train_unit_count:
        raise TrainerError("Training epoch did not observe the exact effective count")
    metrics = accumulator.finalize()
    if metrics.get("count") != observed_count:
        raise TrainerError("Training metric count does not match observed units")
    metrics.update(
        {
            "loss": total_loss / observed_count,
            "main_loss": total_main / observed_count,
            "auxiliary_loss": total_auxiliary / observed_count,
            "optimizer_steps": optimizer_steps,
            **activation_health.finalize(),
            "gradient_norm": {
                "mean": sum(gradient_norms) / len(gradient_norms),
                "max": max(gradient_norms),
            },
            "decoder_norm_max_error": max(decoder_norm_errors),
        }
    )
    return metrics


def evaluate_split(
    *,
    model: TopKSAE,
    data_index: SaeDataIndex,
    config: SaePilotConfig,
    device: torch.device,
    split: str,
) -> dict[str, object]:
    """Evaluate one effective split in natural source order without dead updates."""

    if split not in ("train", "validation"):
        raise TrainerError("split must be train or validation")
    expected_count = (
        data_index.train_unit_count
        if split == "train"
        else data_index.validation_unit_count
    )
    model.eval()
    accumulator = ReconstructionMetricsAccumulator(num_aspects=len(ASPECT_NAMES))
    activation_health = _ActivationHealthAccumulator(
        config.width, config.activation_threshold
    )
    observed_count = 0
    with torch.inference_mode():
        for batch in _natural_batches(data_index, split, config.batch_size):
            inputs = batch.embeddings.to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            output = model(inputs, update_dead_stats=False)
            activation_health.update(output.activations)
            accumulator.update(
                inputs=inputs,
                reconstructions=output.reconstruction,
                activations=output.activations,
                aspect_indices=batch.aspect_indices,
            )
            observed_count += int(inputs.shape[0])
    if observed_count != expected_count:
        raise TrainerError(f"{split} evaluation did not observe the exact effective count")
    metrics = accumulator.finalize()
    if metrics.get("count") != expected_count:
        raise TrainerError(f"{split} metric count does not match effective count")
    metrics.update(activation_health.finalize())
    return metrics


def _make_optimizer(
    model: TopKSAE,
    config: SaePilotConfig,
) -> torch.optim.Adam:
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )


def _cpu_model_state(model: TopKSAE) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu").clone()
        for name, tensor in model.state_dict().items()
    }


def _sha256_tree(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().to(device="cpu").contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
        elif isinstance(item, tuple):
            digest.update(b"tuple\0")
            for child in item:
                update(child)
        elif isinstance(item, list):
            digest.update(b"list\0")
            for child in item:
                update(child)
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def _self_check_index() -> SaeDataIndex:
    return SaeDataIndex(
        units=(),
        shards=(),
        train_accessions=frozenset(),
        validation_accessions=frozenset(),
        train_unit_count=4,
        validation_unit_count=1,
        source_train_unit_count=4,
        source_validation_unit_count=1,
        train_aspect_counts=(1,) * len(ASPECT_NAMES),
        validation_aspect_counts=(1,) + (0,) * (len(ASPECT_NAMES) - 1),
        assignment_sha256="a" * 64,
        source_assignment_sha256="d" * 64,
        text_manifest_sha256="b" * 64,
        activation_manifest_sha256="c" * 64,
    )


def _self_check_step(
    model: TopKSAE,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    config: SaePilotConfig,
) -> tuple[float, float, float]:
    python_value = random.random()
    numpy_value = float(np.random.random())
    torch_value = float(torch.rand((), dtype=torch.float32).item())
    perturbation = torch.tensor(
        [
            python_value,
            numpy_value,
            torch_value,
            python_value + numpy_value + torch_value,
        ],
        dtype=torch.float32,
    ).mul_(1e-3)
    perturbed_inputs = torch.nn.functional.normalize(
        inputs + perturbation.unsqueeze(0), dim=-1
    )
    output = model(perturbed_inputs, update_dead_stats=True)
    losses = compute_sae_loss(
        output,
        perturbed_inputs,
        torch.ones(len(perturbed_inputs), dtype=torch.float32),
        config.aux_coefficient,
    )
    optimizer.zero_grad(set_to_none=True)
    losses.total.backward()
    project_decoder_gradients_(model)
    assert_finite_parameters_and_gradients(model)
    optimizer.step()
    normalize_decoder_(model)
    assert_finite_parameters_and_gradients(model)
    return (python_value, numpy_value, torch_value)


def run_resume_self_check() -> dict[str, object]:
    """Exercise an exact stop/resume trajectory through production APIs."""

    outer_rng = checkpointing.capture_rng_state()
    outer_deterministic_mode = (
        torch.get_deterministic_debug_mode()
        if hasattr(torch, "get_deterministic_debug_mode")
        else None
    )
    outer_deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    outer_cudnn_deterministic = torch.backends.cudnn.deterministic
    outer_cudnn_benchmark = torch.backends.cudnn.benchmark
    config = SaePilotConfig(
        input_dim=4,
        width=12,
        k=3,
        aux_k=5,
        batch_size=4,
        seed=8675309,
        max_epochs=3,
        patience=3,
    )
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, -1.0, 2.0, -2.0],
            [-3.0, 1.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    inputs = torch.nn.functional.normalize(values, dim=-1)
    try:
        setup_determinism(config.seed, torch.device("cpu"))
        uninterrupted_model = TopKSAE(
            input_dim=4,
            width=12,
            k=3,
            aux_k=5,
            activation_threshold=config.activation_threshold,
            dead_steps_threshold=2,
        )
        uninterrupted_optimizer = _make_optimizer(uninterrupted_model, config)
        uninterrupted_trace = []
        for _ in range(3):
            uninterrupted_trace.append(
                _self_check_step(
                    uninterrupted_model, uninterrupted_optimizer, inputs, config
                )
            )
        uninterrupted_model_state = _cpu_model_state(uninterrupted_model)
        uninterrupted_optimizer_state = uninterrupted_optimizer.state_dict()

        setup_determinism(config.seed, torch.device("cpu"))
        initial_model = TopKSAE(
            input_dim=4,
            width=12,
            k=3,
            aux_k=5,
            activation_threshold=config.activation_threshold,
            dead_steps_threshold=2,
        )
        initial_optimizer = _make_optimizer(initial_model, config)
        initial_trace = _self_check_step(
            initial_model, initial_optimizer, inputs, config
        )
        payload = checkpointing.build_checkpoint_payload(
            config=config,
            data_index=_self_check_index(),
            epoch=0,
            global_step=1,
            best_validation_nmse=1.0,
            patience_counter=0,
            epoch_metrics={"epoch": 0, "self_check": True},
            model_state=initial_model.state_dict(),
            optimizer_state=initial_optimizer.state_dict(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "last.pt"
            checkpointing.save_checkpoint_atomic(checkpoint_path, payload)
            loaded = checkpointing.load_checkpoint(
                checkpoint_path,
                expected_config=config,
                expected_assignment_sha256="a" * 64,
                expected_text_manifest_sha256="b" * 64,
                expected_activation_manifest_sha256="c" * 64,
            )

        resumed_model = TopKSAE(
            input_dim=4,
            width=12,
            k=3,
            aux_k=5,
            activation_threshold=config.activation_threshold,
            dead_steps_threshold=2,
        )
        resumed_optimizer = _make_optimizer(resumed_model, config)
        resumed_model.load_state_dict(loaded["model_state"], strict=True)
        resumed_optimizer.load_state_dict(loaded["optimizer_state"])
        checkpointing.restore_rng_state(loaded["rng_state"])
        resumed_trace = [initial_trace]
        for _ in range(2):
            resumed_trace.append(
                _self_check_step(resumed_model, resumed_optimizer, inputs, config)
            )
        resumed_model_state = _cpu_model_state(resumed_model)
        resumed_optimizer_state = resumed_optimizer.state_dict()

        uninterrupted_model_sha = _sha256_tree(uninterrupted_model_state)
        resumed_model_sha = _sha256_tree(resumed_model_state)
        uninterrupted_optimizer_sha = _sha256_tree(uninterrupted_optimizer_state)
        resumed_optimizer_sha = _sha256_tree(resumed_optimizer_state)
        uninterrupted_trace_sha = _sha256_tree(uninterrupted_trace)
        resumed_trace_sha = _sha256_tree(resumed_trace)
        passed = bool(
            uninterrupted_model_sha == resumed_model_sha
            and uninterrupted_optimizer_sha == resumed_optimizer_sha
            and uninterrupted_trace_sha == resumed_trace_sha
        )
        return {
            "passed": passed,
            "uninterrupted_model_state_sha256": uninterrupted_model_sha,
            "resumed_model_state_sha256": resumed_model_sha,
            "uninterrupted_optimizer_state_sha256": uninterrupted_optimizer_sha,
            "resumed_optimizer_state_sha256": resumed_optimizer_sha,
            "uninterrupted_random_trace_sha256": uninterrupted_trace_sha,
            "resumed_random_trace_sha256": resumed_trace_sha,
        }
    finally:
        checkpointing.restore_rng_state(outer_rng)
        if outer_deterministic_mode is not None:
            torch.set_deterministic_debug_mode(outer_deterministic_mode)
        else:
            torch.use_deterministic_algorithms(outer_deterministic_enabled)
        torch.backends.cudnn.deterministic = outer_cudnn_deterministic
        torch.backends.cudnn.benchmark = outer_cudnn_benchmark


def _strict_json_object(path: Path, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise TrainerError(f"Invalid existing {label}: {path}") from error
    if not isinstance(payload, dict):
        raise TrainerError(f"Existing {label} must be a JSON object")
    return payload


def _require_resume_directory(output_dir: Path) -> None:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise TrainerError("Resume output directory must be a real directory")
    if output_dir.resolve(strict=True) != output_dir.absolute():
        raise TrainerError("Resume output directory must not traverse symlinks")
    required = ("config.json", "split_manifest.json", "best.pt", "last.pt")
    for name in required:
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise TrainerError(f"Resume artifact is missing or invalid: {name}")
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and (metrics_path.is_symlink() or not metrics_path.is_file()):
        raise TrainerError("Resume metrics artifact is invalid")
    summary_path = output_dir / "summary.json"
    if summary_path.exists() or summary_path.is_symlink():
        raise TrainerError("Completed runs cannot be resumed")


def _fit_baselines(
    data_index: SaeDataIndex,
    config: SaePilotConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    train_mean = stream_train_mean(data_index, config.batch_size)
    mean_baseline = evaluate_mean_baseline(
        (),
        _natural_batches(data_index, "validation", config.batch_size),
        training_mean=train_mean,
    )
    mean_baseline["mean_sha256"] = _sha256_tree(train_mean)
    pca_baseline = fit_pca32_baseline(
        _natural_batches(data_index, "train", config.batch_size),
        _natural_batches(data_index, "validation", config.batch_size),
        batch_size=config.batch_size,
    )
    if mean_baseline.get("count") != data_index.validation_unit_count:
        raise TrainerError("Mean baseline did not use the effective validation split")
    if pca_baseline.get("count") != data_index.validation_unit_count:
        raise TrainerError("PCA baseline did not use the effective validation split")
    return train_mean, {
        "training_mean": mean_baseline,
        "incremental_pca32": pca_baseline,
    }


def _feature_statistics(
    *,
    model: TopKSAE,
    data_index: SaeDataIndex,
    config: SaePilotConfig,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    model.eval()
    with torch.inference_mode():
        for split in ("train", "validation"):
            accumulator = FeatureStatsAccumulator(
                width=config.width,
                num_aspects=len(ASPECT_NAMES),
                activation_threshold=config.activation_threshold,
            )
            observed = 0
            for batch in _natural_batches(data_index, split, config.batch_size):
                inputs = batch.embeddings.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                output = model(inputs, update_dead_stats=False)
                accumulator.update(output.activations, batch.aspect_indices)
                observed += int(inputs.shape[0])
            expected = (
                data_index.train_unit_count
                if split == "train"
                else data_index.validation_unit_count
            )
            if observed != expected:
                raise TrainerError(f"Feature statistics did not cover {split}")
            results[split] = accumulator.finalize()
    return results["train"], results["validation"]


def _provenance(data_index: SaeDataIndex) -> dict[str, str]:
    return {
        "assignment_sha256": data_index.assignment_sha256,
        "text_manifest_sha256": data_index.text_manifest_sha256,
        "activation_manifest_sha256": data_index.activation_manifest_sha256,
    }


def _file_metadata(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def run_pilot(
    *,
    config: SaePilotConfig,
    units_path: Path,
    text_manifest_path: Path,
    activation_dir: Path,
    output_dir: Path,
    device: str | torch.device,
    resume: bool = False,
    overwrite: bool = False,
    stop_after_epoch: int | None = None,
) -> dict[str, object]:
    """Train, resume, or finalize one deterministic SAE pilot run."""

    if not isinstance(config, SaePilotConfig):
        raise TypeError("config must be SaePilotConfig")
    if type(resume) is not bool or type(overwrite) is not bool:
        raise TypeError("resume and overwrite must be bool")
    if resume and overwrite:
        raise TrainerError("resume and overwrite are mutually exclusive")
    if stop_after_epoch is not None and (
        type(stop_after_epoch) is not int
        or stop_after_epoch < 0
        or stop_after_epoch >= config.max_epochs
    ):
        raise TrainerError("stop_after_epoch must be within the configured epochs")
    resolved_device = torch.device(device)
    if resolved_device.type not in ("cpu", "cuda"):
        raise TrainerError("device must be CPU or CUDA")
    _configure_cuda_determinism_environment(resolved_device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise TrainerError("CUDA was requested but is unavailable")

    units_path = Path(units_path).resolve()
    text_manifest_path = Path(text_manifest_path).resolve()
    activation_dir = Path(activation_dir).resolve()
    output_dir = Path(output_dir)

    full_index = build_data_index(units_path, text_manifest_path, activation_dir)
    data_index = limit_data_index(
        full_index,
        config.max_train_units,
        config.max_validation_units,
    )
    aspect_weights = compute_aspect_weights(data_index, config.aspect_alpha)
    split_manifest = build_split_manifest(
        data_index,
        units_path,
        text_manifest_path,
        activation_dir,
        config.aspect_alpha,
        aspect_weights,
    )
    resume_report = run_resume_self_check()
    if resume_report.get("passed") is not True:
        raise TrainerError("Production resume self-check failed")

    last_checkpoint: dict[str, object] | None = None
    best_checkpoint: dict[str, object] | None = None
    if resume:
        _require_resume_directory(output_dir)
        existing_config = _strict_json_object(output_dir / "config.json", "config")
        if existing_config != config.to_dict():
            raise TrainerError("Existing config does not match requested config")
        existing_split = _strict_json_object(
            output_dir / "split_manifest.json", "split manifest"
        )
        if existing_split != split_manifest:
            raise TrainerError("Existing split manifest does not match current sources")
        last_checkpoint = checkpointing.load_checkpoint(
            output_dir / "last.pt",
            expected_config=config,
            expected_assignment_sha256=data_index.assignment_sha256,
            expected_text_manifest_sha256=data_index.text_manifest_sha256,
            expected_activation_manifest_sha256=data_index.activation_manifest_sha256,
        )
        best_checkpoint = checkpointing.load_checkpoint(
            output_dir / "best.pt",
            expected_config=config,
            expected_assignment_sha256=data_index.assignment_sha256,
            expected_text_manifest_sha256=data_index.text_manifest_sha256,
            expected_activation_manifest_sha256=data_index.activation_manifest_sha256,
        )
        if int(best_checkpoint["epoch"]) > int(last_checkpoint["epoch"]):
            raise TrainerError("Best checkpoint epoch is after the last checkpoint")
        if (
            float(best_checkpoint["best_validation_nmse"])
            != float(last_checkpoint["best_validation_nmse"])
        ):
            raise TrainerError("Best and last checkpoints disagree on the best score")
        best_metrics = best_checkpoint["epoch_metrics"]
        if (
            not isinstance(best_metrics, dict)
            or not isinstance(best_metrics.get("validation"), dict)
            or float(best_metrics["validation"].get("nmse", -1.0))
            != float(last_checkpoint["best_validation_nmse"])
        ):
            raise TrainerError("Best checkpoint metrics do not match the best score")
    else:
        checkpointing.prepare_run_directory(output_dir, overwrite=overwrite)
        checkpointing.write_json_atomic(output_dir / "config.json", config.to_dict())
        checkpointing.write_json_atomic(
            output_dir / "split_manifest.json", split_manifest
        )

    train_mean, baselines = _fit_baselines(data_index, config)
    determinism = _expected_determinism_record(config.seed, resolved_device)
    dead_steps_threshold = math.ceil(
        data_index.train_unit_count / config.batch_size
    )
    run_context: dict[str, object] = {
        "baselines": baselines,
        "resume_self_check": resume_report,
        "determinism": determinism,
        "dead_steps_threshold": dead_steps_threshold,
    }

    if last_checkpoint is not None:
        metrics_path = output_dir / "metrics.jsonl"
        if metrics_path.is_file():
            historical_rows = _load_metrics(metrics_path)
        elif int(last_checkpoint["epoch"]) == 0:
            historical_rows = [dict(last_checkpoint["epoch_metrics"])]
        else:
            raise TrainerError("Historical metrics are missing before resume")
        for historical_metrics in historical_rows:
            if historical_metrics.get("run_context") != run_context:
                raise TrainerError(
                    "Historical run context does not match recomputed run context"
                )
        for label, stored_checkpoint in (
            ("best", best_checkpoint),
            ("last", last_checkpoint),
        ):
            if stored_checkpoint is None:
                raise AssertionError(f"Missing validated {label} checkpoint")
            stored_metrics = stored_checkpoint.get("epoch_metrics")
            if (
                not isinstance(stored_metrics, dict)
                or stored_metrics.get("run_context") != run_context
            ):
                raise TrainerError(
                    f"Recomputed run context does not match {label} checkpoint"
                )
        checkpointing.reconcile_metrics_jsonl(metrics_path, last_checkpoint)

    observed_determinism = setup_determinism(config.seed, resolved_device)
    if observed_determinism != determinism:
        raise TrainerError("Observed determinism state does not match run context")
    model = TopKSAE(
        input_dim=config.input_dim,
        width=config.width,
        k=config.k,
        aux_k=config.aux_k,
        activation_threshold=config.activation_threshold,
        dead_steps_threshold=dead_steps_threshold,
    ).to(resolved_device)
    optimizer = _make_optimizer(model, config)

    if last_checkpoint is None:
        model.initialize_pre_bias_(train_mean.to(resolved_device))
        start_epoch = 0
        global_step = 0
        best_validation_nmse: float | None = None
        best_epoch: int | None = None
        patience_counter = 0
        final_epoch = -1
    else:
        model.load_state_dict(last_checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(last_checkpoint["optimizer_state"])
        start_epoch = int(last_checkpoint["epoch"]) + 1
        global_step = int(last_checkpoint["global_step"])
        best_validation_nmse = float(last_checkpoint["best_validation_nmse"])
        patience_counter = int(last_checkpoint["patience_counter"])
        if best_checkpoint is None:
            raise AssertionError("Missing validated best checkpoint")
        best_epoch = int(best_checkpoint["epoch"])
        final_epoch = int(last_checkpoint["epoch"])
        checkpointing.restore_rng_state(last_checkpoint["rng_state"])

    stopping_reason = "max_epochs"
    if patience_counter >= config.patience:
        stopping_reason = "early_stopping"
    else:
        for epoch in range(start_epoch, config.max_epochs):
            training_metrics = train_one_epoch(
                model=model,
                optimizer=optimizer,
                data_index=data_index,
                config=config,
                aspect_weights=aspect_weights,
                device=resolved_device,
                epoch=epoch,
            )
            global_step += int(training_metrics["optimizer_steps"])
            validation_metrics = evaluate_split(
                model=model,
                data_index=data_index,
                config=config,
                device=resolved_device,
                split="validation",
            )
            validation_nmse = float(validation_metrics["nmse"])
            improved = bool(
                best_validation_nmse is None
                or best_validation_nmse - validation_nmse >= config.min_delta
            )
            if improved:
                best_validation_nmse = validation_nmse
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            epoch_metrics: dict[str, object] = {
                "epoch": epoch,
                "global_step": global_step,
                "training": training_metrics,
                "validation": validation_metrics,
                "run_context": run_context,
            }
            if best_validation_nmse is None:
                raise AssertionError("First epoch did not establish a finite best score")
            payload = checkpointing.build_checkpoint_payload(
                config=config,
                data_index=data_index,
                epoch=epoch,
                global_step=global_step,
                best_validation_nmse=float(best_validation_nmse),
                patience_counter=patience_counter,
                epoch_metrics=epoch_metrics,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
            )
            if improved:
                checkpointing.save_checkpoint_atomic(output_dir / "best.pt", payload)
            checkpointing.save_checkpoint_atomic(output_dir / "last.pt", payload)
            checkpointing.reconcile_metrics_jsonl(
                output_dir / "metrics.jsonl", payload
            )
            final_epoch = epoch

            if stop_after_epoch == epoch:
                return {
                    "interrupted": True,
                    "final_epoch": epoch,
                    "global_step": global_step,
                    "resume_self_check": resume_report,
                }
            if patience_counter >= config.patience:
                stopping_reason = "early_stopping"
                break

    if final_epoch < 0 or best_epoch is None or best_validation_nmse is None:
        raise TrainerError("Training produced no checkpointed epoch")
    best_checkpoint = checkpointing.load_checkpoint(
        output_dir / "best.pt",
        expected_config=config,
        expected_assignment_sha256=data_index.assignment_sha256,
        expected_text_manifest_sha256=data_index.text_manifest_sha256,
        expected_activation_manifest_sha256=data_index.activation_manifest_sha256,
    )
    model.load_state_dict(best_checkpoint["model_state"], strict=True)
    model.to(resolved_device)
    train_statistics, validation_statistics = _feature_statistics(
        model=model,
        data_index=data_index,
        config=config,
        device=resolved_device,
    )
    provenance = _provenance(data_index)
    common = {
        "schema_version": checkpointing.SCHEMA_VERSION,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "provenance": provenance,
    }
    inference = {
        **common,
        "model_state": _cpu_model_state(model),
    }
    feature_stats = {
        **common,
        "width": config.width,
        "train": train_statistics,
        "validation": validation_statistics,
    }
    checkpointing.write_torch_atomic(output_dir / "inference.pt", inference)
    checkpointing.write_torch_atomic(
        output_dir / "feature_stats.pt", feature_stats
    )

    best_validation_metrics = dict(best_checkpoint["epoch_metrics"]["validation"])
    gate_report = evaluate_pilot_gates(
        validation_nmse=float(best_validation_metrics["nmse"]),
        mean_baseline_nmse=float(baselines["training_mean"]["nmse"]),
        explained_variance=float(best_validation_metrics["explained_variance"]),
        mean_l0=float(best_validation_metrics["mean_l0"]),
        train_dead_fraction=float(train_statistics["dead_fraction"]),
        split_valid=True,
        resume_valid=bool(resume_report["passed"]),
        artifacts_valid=False,
        all_finite=True,
    )
    files = {
        name: _file_metadata(output_dir / name)
        for name in checkpointing.SUMMARY_FILE_NAMES
    }
    summary: dict[str, object] = {
        "schema_version": checkpointing.SCHEMA_VERSION,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "provenance": provenance,
        "source_paths": {
            "units_path": str(units_path),
            "text_manifest_path": str(text_manifest_path),
            "activation_dir": str(activation_dir),
        },
        "source_provenance": {
            "source_assignment_sha256": full_index.source_assignment_sha256,
            "text_manifest_sha256": full_index.text_manifest_sha256,
            "activation_manifest_sha256": full_index.activation_manifest_sha256,
        },
        "source_split_counts": {
            "train": full_index.source_train_unit_count,
            "validation": full_index.source_validation_unit_count,
        },
        "effective_split_counts": {
            "train": data_index.train_unit_count,
            "validation": data_index.validation_unit_count,
        },
        "dead_steps_threshold": dead_steps_threshold,
        "baselines": baselines,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "stopping_reason": stopping_reason,
        "best_validation_metrics": best_validation_metrics,
        "resume_self_check": resume_report,
        "gates": dict(gate_report["gates"]),
        "passed": bool(gate_report["passed"]),
        "determinism": determinism,
        "files": files,
    }
    checkpointing.validate_pre_summary_artifacts(output_dir, summary)
    summary_gates = summary["gates"]
    if not isinstance(summary_gates, dict):
        raise AssertionError("Summary gates must be a mapping")
    summary_gates["artifacts_valid"] = True
    summary["passed"] = bool(all(summary_gates.values()))
    checkpointing.write_json_atomic(output_dir / "summary.json", summary)
    return validate_pilot_run(output_dir)


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainerError(f"{label} must be a lowercase SHA-256")
    return value


def _require_finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TrainerError(f"{label} must be a finite float")
    return value


def _validate_activation_health(value: Mapping[str, object], label: str) -> None:
    magnitude = value.get("activation_magnitude")
    if not isinstance(magnitude, dict) or set(magnitude) != {
        "firing_count",
        "mean_on_firing",
        "max",
    }:
        raise TrainerError(f"{label} activation magnitude schema is invalid")
    firing_count = magnitude["firing_count"]
    if type(firing_count) is not int or firing_count < 0:
        raise TrainerError(f"{label} firing_count must be nonnegative int")
    mean = _require_finite_float(
        magnitude["mean_on_firing"], f"{label} mean_on_firing"
    )
    maximum = _require_finite_float(magnitude["max"], f"{label} max")
    dead_fraction = _require_finite_float(
        value.get("dead_feature_fraction"), f"{label} dead fraction"
    )
    if mean < 0.0 or maximum < mean or not 0.0 <= dead_fraction <= 1.0:
        raise TrainerError(f"{label} activation health value is out of range")


def _validate_reconstruction_metrics(
    value: object,
    *,
    label: str,
    expected_count: int,
    training: bool,
) -> dict[str, object]:
    expected_keys = TRAINING_METRIC_KEYS if training else VALIDATION_METRIC_KEYS
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise TrainerError(f"{label} metrics must use the exact schema")
    if type(value["count"]) is not int or value["count"] != expected_count:
        raise TrainerError(f"{label} metrics count is invalid")
    for name in (
        "nmse",
        "explained_variance",
        "cosine_similarity",
        "mean_l0",
    ):
        _require_finite_float(value[name], f"{label} {name}")
    quantiles = value["l0_quantiles"]
    if not isinstance(quantiles, dict) or set(quantiles) != {
        "p0",
        "p25",
        "p50",
        "p75",
        "p100",
    }:
        raise TrainerError(f"{label} L0 quantile schema is invalid")
    for name, quantile in quantiles.items():
        _require_finite_float(quantile, f"{label} L0 {name}")
    per_aspect = value["per_aspect"]
    if not isinstance(per_aspect, dict) or set(per_aspect) != {
        str(index) for index in range(len(ASPECT_NAMES))
    }:
        raise TrainerError(f"{label} per-aspect schema is invalid")
    aspect_keys = {
        "count",
        "nmse",
        "explained_variance",
        "cosine_similarity",
        "mean_l0",
        "l0_quantiles",
    }
    for aspect, metrics in per_aspect.items():
        if not isinstance(metrics, dict) or set(metrics) != aspect_keys:
            raise TrainerError(f"{label} aspect {aspect} schema is invalid")
        if type(metrics["count"]) is not int or metrics["count"] < 0:
            raise TrainerError(f"{label} aspect {aspect} count is invalid")
    _validate_activation_health(value, label)
    if training:
        for name in ("loss", "main_loss", "auxiliary_loss"):
            _require_finite_float(value[name], f"{label} {name}")
        if type(value["optimizer_steps"]) is not int or value["optimizer_steps"] <= 0:
            raise TrainerError(f"{label} optimizer_steps is invalid")
        gradient = value["gradient_norm"]
        if not isinstance(gradient, dict) or set(gradient) != {"mean", "max"}:
            raise TrainerError(f"{label} gradient norm schema is invalid")
        gradient_mean = _require_finite_float(
            gradient["mean"], f"{label} gradient mean"
        )
        gradient_max = _require_finite_float(
            gradient["max"], f"{label} gradient max"
        )
        decoder_error = _require_finite_float(
            value["decoder_norm_max_error"], f"{label} decoder norm error"
        )
        if (
            gradient_mean < 0.0
            or gradient_max < gradient_mean
            or not 0.0 <= decoder_error < 1e-5
        ):
            raise TrainerError(f"{label} training health value is out of range")
    return value


def _validate_baselines(
    value: object, expected_validation_count: int
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "training_mean",
        "incremental_pca32",
    }:
        raise TrainerError("Summary baselines schema is invalid")
    mean = value["training_mean"]
    if not isinstance(mean, dict) or set(mean) != {
        "baseline",
        "count",
        "nmse",
        "mean_sha256",
    }:
        raise TrainerError("Training mean baseline schema is invalid")
    if (
        mean["baseline"] != "training_mean"
        or type(mean["count"]) is not int
        or mean["count"] != expected_validation_count
    ):
        raise TrainerError("Training mean baseline metadata is invalid")
    _require_finite_float(mean["nmse"], "training mean baseline NMSE")
    _require_sha256(mean["mean_sha256"], "training mean checksum")
    pca = value["incremental_pca32"]
    if not isinstance(pca, dict) or set(pca) != {
        "baseline",
        "count",
        "nmse",
        "n_components",
        "components_sha256",
        "sklearn_version",
    }:
        raise TrainerError("PCA baseline schema is invalid")
    if (
        pca["baseline"] != "incremental_pca32"
        or type(pca["count"]) is not int
        or pca["count"] != expected_validation_count
        or type(pca["n_components"]) is not int
        or pca["n_components"] != 32
        or type(pca["sklearn_version"]) is not str
        or not pca["sklearn_version"]
    ):
        raise TrainerError("PCA baseline metadata is invalid")
    _require_finite_float(pca["nmse"], "PCA baseline NMSE")
    _require_sha256(pca["components_sha256"], "PCA components checksum")
    return value


def _validate_self_check(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != SELF_CHECK_KEYS:
        raise TrainerError("Resume self-check schema is invalid")
    if type(value["passed"]) is not bool or value["passed"] is not True:
        raise TrainerError("Resume self-check must have passed")
    for name in SELF_CHECK_KEYS - {"passed"}:
        _require_sha256(value[name], f"resume self-check {name}")
    for prefix in ("model_state", "optimizer_state", "random_trace"):
        if (
            value[f"uninterrupted_{prefix}_sha256"]
            != value[f"resumed_{prefix}_sha256"]
        ):
            raise TrainerError(f"Resume self-check {prefix} checksums differ")
    return value


def _validate_determinism(
    value: object, config: SaePilotConfig
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != DETERMINISM_KEYS:
        raise TrainerError("Determinism schema is invalid")
    expected_seeds = {
        "seed": config.seed,
        "numpy_seed": config.seed % (2**32),
        "torch_seed": config.seed % (2**63),
    }
    for name, expected in expected_seeds.items():
        if type(value[name]) is not int or value[name] != expected:
            raise TrainerError(f"Determinism {name} is invalid")
    if type(value["device"]) is not str or not value["device"]:
        raise TrainerError("Determinism device is invalid")
    for name in (
        "torch_deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
    ):
        if type(value[name]) is not bool:
            raise TrainerError(f"Determinism {name} must be bool")
    if value["torch_deterministic_algorithms"] is not True:
        raise TrainerError("Torch deterministic algorithms must be enabled")
    workspace = value["cublas_workspace_config"]
    if workspace is not None and workspace not in _CUBLAS_WORKSPACE_CONFIGS:
        raise TrainerError("Determinism cuBLAS workspace setting is invalid")
    return value


def _load_torch_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            value = torch.load(path, map_location="cpu")
    except Exception as error:
        raise TrainerError(f"Cannot load {label}: {path}") from error
    if not isinstance(value, dict):
        raise TrainerError(f"{label} must be a mapping")
    return value


def _load_metrics(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise TrainerError("Cannot read metrics history") from error
    return [
        _strict_json_object_from_text(line, f"metrics line {index}")
        for index, line in enumerate(lines, 1)
        if line.strip()
    ]


def _strict_json_object_from_text(text: str, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise TrainerError(f"Invalid {label}") from error
    if not isinstance(value, dict):
        raise TrainerError(f"{label} must be a JSON object")
    return value


def _replay_training_trajectory(
    metrics_rows: Sequence[Mapping[str, object]],
    config: SaePilotConfig,
    expected_optimizer_steps: int | None = None,
) -> dict[str, object]:
    if not 1 <= len(metrics_rows) <= config.max_epochs:
        raise TrainerError("Training replay length must be within 1..max_epochs")
    if expected_optimizer_steps is None:
        first_training = metrics_rows[0].get("training")
        if not isinstance(first_training, Mapping):
            raise TrainerError("Training replay metrics are invalid")
        expected_optimizer_steps = first_training.get("optimizer_steps")  # type: ignore[assignment]
    if type(expected_optimizer_steps) is not int or expected_optimizer_steps <= 0:
        raise TrainerError("Training replay expected optimizer steps are invalid")
    best_epoch: int | None = None
    best_validation_nmse: float | None = None
    patience_counter = 0
    for epoch, row in enumerate(metrics_rows):
        training = row["training"]
        validation = row["validation"]
        if not isinstance(training, Mapping) or not isinstance(validation, Mapping):
            raise TrainerError("Training replay metrics are invalid")
        optimizer_steps = training["optimizer_steps"]
        if type(optimizer_steps) is not int or optimizer_steps != expected_optimizer_steps:
            raise TrainerError("Training replay optimizer steps are inconsistent")
        expected_global_step = (epoch + 1) * expected_optimizer_steps
        if (
            type(row["global_step"]) is not int
            or row["global_step"] != expected_global_step
        ):
            raise TrainerError("Training replay global step is inconsistent")
        validation_nmse = _require_finite_float(
            validation["nmse"], f"training replay epoch {epoch} validation NMSE"
        )
        improved = bool(
            best_validation_nmse is None
            or best_validation_nmse - validation_nmse >= config.min_delta
        )
        if improved:
            best_epoch = epoch
            best_validation_nmse = validation_nmse
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.patience and epoch != len(metrics_rows) - 1:
            raise TrainerError("Training replay continues after patience exhaustion")
    if best_epoch is None or best_validation_nmse is None or not metrics_rows:
        raise TrainerError("Training replay produced no best epoch")
    final_epoch = len(metrics_rows) - 1
    if patience_counter >= config.patience:
        stopping_reason = "early_stopping"
    elif final_epoch == config.max_epochs - 1:
        stopping_reason = "max_epochs"
    else:
        raise TrainerError("Training replay has no valid terminal condition")
    return {
        "best_epoch": best_epoch,
        "best_validation_nmse": best_validation_nmse,
        "final_epoch": final_epoch,
        "global_step": len(metrics_rows) * expected_optimizer_steps,
        "patience_counter": patience_counter,
        "stopping_reason": stopping_reason,
    }


def validate_pilot_run(output_dir: Path) -> dict[str, object]:
    """Validate artifacts and rebind them to current full and limited sources."""

    output_dir = Path(output_dir)
    summary = checkpointing.validate_run_artifacts(output_dir)
    if set(summary) != SUMMARY_KEYS:
        raise TrainerError("Summary must contain the exact Task 6 keys")
    paths = summary.get("source_paths")
    if not isinstance(paths, dict) or set(paths) != {
        "units_path",
        "text_manifest_path",
        "activation_dir",
    }:
        raise TrainerError("Summary source_paths has invalid keys")
    if any(not isinstance(value, str) or not value for value in paths.values()):
        raise TrainerError("Summary source_paths must contain nonempty strings")
    units_path = Path(paths["units_path"]).resolve()
    text_manifest_path = Path(paths["text_manifest_path"]).resolve()
    activation_dir = Path(paths["activation_dir"]).resolve()
    try:
        config = SaePilotConfig(**summary["config"])
    except (TypeError, ValueError) as error:
        raise TrainerError("Summary config is invalid") from error
    full_index = build_data_index(units_path, text_manifest_path, activation_dir)
    data_index = limit_data_index(
        full_index,
        config.max_train_units,
        config.max_validation_units,
    )
    for label, counts in (
        ("source", summary.get("source_split_counts")),
        ("effective", summary.get("effective_split_counts")),
    ):
        if (
            not isinstance(counts, dict)
            or set(counts) != {"train", "validation"}
            or any(type(counts[name]) is not int for name in counts)
        ):
            raise TrainerError(f"Summary {label} split count schema is invalid")
    provenance = summary.get("provenance")
    if provenance != _provenance(data_index):
        raise TrainerError("Current effective source provenance does not match summary")
    source_provenance = summary.get("source_provenance")
    expected_source_provenance = {
        "source_assignment_sha256": full_index.source_assignment_sha256,
        "text_manifest_sha256": full_index.text_manifest_sha256,
        "activation_manifest_sha256": full_index.activation_manifest_sha256,
    }
    if source_provenance != expected_source_provenance:
        raise TrainerError("Current full source provenance does not match summary")
    if summary.get("source_split_counts") != {
        "train": full_index.source_train_unit_count,
        "validation": full_index.source_validation_unit_count,
    }:
        raise TrainerError("Summary source split counts do not match current sources")
    if summary.get("effective_split_counts") != {
        "train": data_index.train_unit_count,
        "validation": data_index.validation_unit_count,
    }:
        raise TrainerError("Summary effective split counts do not match config limits")
    split_manifest = _strict_json_object(
        output_dir / "split_manifest.json", "split manifest"
    )
    effective_weights = compute_aspect_weights(data_index, config.aspect_alpha)
    expected_split_manifest = build_split_manifest(
        data_index,
        units_path,
        text_manifest_path,
        activation_dir,
        config.aspect_alpha,
        effective_weights,
    )
    if split_manifest != expected_split_manifest:
        raise TrainerError("Stored split manifest does not match recomputed split manifest")

    dead_steps_threshold = summary.get("dead_steps_threshold")
    expected_dead_threshold = math.ceil(
        data_index.train_unit_count / config.batch_size
    )
    if type(dead_steps_threshold) is not int or dead_steps_threshold != expected_dead_threshold:
        raise TrainerError("Summary dead steps threshold is invalid")
    baselines = _validate_baselines(
        summary.get("baselines"), data_index.validation_unit_count
    )
    self_check = _validate_self_check(summary.get("resume_self_check"))
    determinism = _validate_determinism(summary.get("determinism"), config)
    expected_run_context = {
        "baselines": baselines,
        "resume_self_check": self_check,
        "determinism": determinism,
        "dead_steps_threshold": dead_steps_threshold,
    }

    best_checkpoint = checkpointing.load_checkpoint(
        output_dir / "best.pt",
        expected_config=config,
        expected_assignment_sha256=data_index.assignment_sha256,
        expected_text_manifest_sha256=data_index.text_manifest_sha256,
        expected_activation_manifest_sha256=data_index.activation_manifest_sha256,
    )
    last_checkpoint = checkpointing.load_checkpoint(
        output_dir / "last.pt",
        expected_config=config,
        expected_assignment_sha256=data_index.assignment_sha256,
        expected_text_manifest_sha256=data_index.text_manifest_sha256,
        expected_activation_manifest_sha256=data_index.activation_manifest_sha256,
    )
    metrics_rows = _load_metrics(output_dir / "metrics.jsonl")
    if len(metrics_rows) != int(summary["final_epoch"]) + 1:
        raise TrainerError("Metrics history length is invalid")
    for expected_epoch, row in enumerate(metrics_rows):
        if not isinstance(row, dict) or set(row) != {
            "epoch",
            "global_step",
            "training",
            "validation",
            "run_context",
        }:
            raise TrainerError("Epoch metrics must use the exact schema")
        if type(row["epoch"]) is not int or row["epoch"] != expected_epoch:
            raise TrainerError("Epoch metric index is invalid")
        if type(row["global_step"]) is not int or row["global_step"] <= 0:
            raise TrainerError("Epoch global step is invalid")
        _validate_reconstruction_metrics(
            row["training"],
            label=f"epoch {expected_epoch} training",
            expected_count=data_index.train_unit_count,
            training=True,
        )
        _validate_reconstruction_metrics(
            row["validation"],
            label=f"epoch {expected_epoch} validation",
            expected_count=data_index.validation_unit_count,
            training=False,
        )
        if (
            not isinstance(row["run_context"], dict)
            or set(row["run_context"]) != RUN_CONTEXT_KEYS
            or row["run_context"] != expected_run_context
        ):
            raise TrainerError("Epoch protected run context is invalid")
    replay = _replay_training_trajectory(
        metrics_rows,
        config,
        expected_optimizer_steps=expected_dead_threshold,
    )
    best_epoch = int(replay["best_epoch"])
    final_epoch = int(replay["final_epoch"])
    if summary["best_epoch"] != best_epoch or summary["final_epoch"] != final_epoch:
        raise TrainerError("Summary epochs disagree with training replay")
    if int(best_checkpoint["epoch"]) != best_epoch:
        raise TrainerError("Best checkpoint epoch disagrees with training replay")
    if int(last_checkpoint["epoch"]) != final_epoch:
        raise TrainerError("Last checkpoint epoch disagrees with training replay")
    if int(best_checkpoint["global_step"]) != int(metrics_rows[best_epoch]["global_step"]):
        raise TrainerError("Best checkpoint global step disagrees with training replay")
    if int(last_checkpoint["global_step"]) != int(replay["global_step"]):
        raise TrainerError("Last checkpoint global step disagrees with training replay")
    if int(best_checkpoint["patience_counter"]) != 0:
        raise TrainerError("Best checkpoint patience disagrees with training replay")
    if int(last_checkpoint["patience_counter"]) != int(replay["patience_counter"]):
        raise TrainerError("Last checkpoint patience disagrees with training replay")
    replay_best_nmse = float(replay["best_validation_nmse"])
    if (
        float(best_checkpoint["best_validation_nmse"]) != replay_best_nmse
        or float(last_checkpoint["best_validation_nmse"]) != replay_best_nmse
    ):
        raise TrainerError("Checkpoint best NMSE disagrees with training replay")
    if best_checkpoint["epoch_metrics"] != metrics_rows[best_epoch]:
        raise TrainerError("Best checkpoint metrics do not match protected history")
    if last_checkpoint["epoch_metrics"] != metrics_rows[final_epoch]:
        raise TrainerError("Last checkpoint metrics do not match protected history")
    best_validation = metrics_rows[best_epoch]["validation"]
    if summary.get("best_validation_metrics") != best_validation:
        raise TrainerError("Summary best validation metrics are not protected")

    expected_reason = replay["stopping_reason"]
    if summary.get("stopping_reason") != expected_reason:
        raise TrainerError("Summary stopping reason is inconsistent")

    inference = _load_torch_mapping(output_dir / "inference.pt", "inference")
    inference_state = inference.get("model_state")
    best_model_state = best_checkpoint.get("model_state")
    if (
        not isinstance(inference_state, Mapping)
        or not isinstance(best_model_state, Mapping)
        or set(inference_state) != set(best_model_state)
    ):
        raise TrainerError("inference model state schema does not match best checkpoint")
    for name in best_model_state:
        inference_tensor = inference_state[name]
        best_tensor = best_model_state[name]
        if (
            not isinstance(inference_tensor, torch.Tensor)
            or not isinstance(best_tensor, torch.Tensor)
            or not torch.equal(inference_tensor, best_tensor)
        ):
            raise TrainerError(
                f"inference tensor {name} does not equal best checkpoint"
            )

    feature_stats = _load_torch_mapping(
        output_dir / "feature_stats.pt", "feature statistics"
    )
    train_feature_stats = feature_stats.get("train")
    if not isinstance(train_feature_stats, dict):
        raise TrainerError("Training feature statistics are invalid")
    train_dead_fraction = _require_finite_float(
        train_feature_stats.get("dead_fraction"), "training dead fraction"
    )
    expected_gate_report = evaluate_pilot_gates(
        validation_nmse=float(best_validation["nmse"]),
        mean_baseline_nmse=float(baselines["training_mean"]["nmse"]),
        explained_variance=float(best_validation["explained_variance"]),
        mean_l0=float(best_validation["mean_l0"]),
        train_dead_fraction=train_dead_fraction,
        split_valid=True,
        resume_valid=True,
        artifacts_valid=True,
        all_finite=True,
    )
    gates = summary.get("gates")
    if (
        not isinstance(gates, dict)
        or set(gates) != GATE_KEYS
        or any(type(value) is not bool for value in gates.values())
        or gates != expected_gate_report["gates"]
    ):
        raise TrainerError("Summary gates are inconsistent")
    if type(summary.get("passed")) is not bool or summary["passed"] != expected_gate_report["passed"]:
        raise TrainerError("Summary passed value is inconsistent")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--units-path", type=Path, default=DEFAULT_UNITS_PATH)
    parser.add_argument(
        "--text-manifest-path", type=Path, default=DEFAULT_TEXT_MANIFEST_PATH
    )
    parser.add_argument(
        "--activation-dir", type=Path, default=DEFAULT_ACTIVATION_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-train-units", type=int)
    parser.add_argument("--max-validation-units", type=int)
    parser.add_argument("--max-epochs", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.validate_only:
        summary = validate_pilot_run(args.output_dir)
        print(json.dumps(summary, allow_nan=False, sort_keys=True))
        return 0
    if (
        args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
        and (args.max_train_units is not None or args.max_validation_units is not None)
    ):
        raise TrainerError("Smoke limits are forbidden for the production output directory")
    defaults = SaePilotConfig()
    max_epochs = defaults.max_epochs if args.max_epochs is None else args.max_epochs
    if type(max_epochs) is not int:
        raise TrainerError("max_epochs must be an integer")
    config = SaePilotConfig(
        max_epochs=max_epochs,
        patience=min(defaults.patience, max_epochs),
        max_train_units=args.max_train_units,
        max_validation_units=args.max_validation_units,
    )
    summary = run_pilot(
        config=config,
        units_path=args.units_path,
        text_manifest_path=args.text_manifest_path,
        activation_dir=args.activation_dir,
        output_dir=args.output_dir,
        device=args.device,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
