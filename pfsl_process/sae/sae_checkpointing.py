"""Atomic persistence and strict validation for the ProTrek SAE pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import copy
import stat
import tempfile
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from .sae_pilot_config import SaePilotConfig
from .sae_training_data import SaeDataIndex


ARTIFACT_NAMES = (
    "config.json",
    "split_manifest.json",
    "metrics.jsonl",
    "summary.json",
    "best.pt",
    "last.pt",
    "inference.pt",
    "feature_stats.pt",
)
SUMMARY_FILE_NAMES = tuple(name for name in ARTIFACT_NAMES if name != "summary.json")
CHECKPOINT_KEYS = (
    "schema_version",
    "config",
    "config_fingerprint",
    "assignment_sha256",
    "text_manifest_sha256",
    "activation_manifest_sha256",
    "epoch",
    "global_step",
    "best_validation_nmse",
    "patience_counter",
    "epoch_metrics",
    "model_state",
    "optimizer_state",
    "rng_state",
)
PROVENANCE_KEYS = (
    "assignment_sha256",
    "text_manifest_sha256",
    "activation_manifest_sha256",
)
MODEL_STATE_KEYS = (
    "encoder_weight",
    "decoder_weight",
    "encoder_bias",
    "pre_bias",
    "steps_since_active",
)
INFERENCE_KEYS = (
    "schema_version",
    "config",
    "config_fingerprint",
    "provenance",
    "model_state",
)
FEATURE_STATS_ARTIFACT_KEYS = (
    "schema_version",
    "config",
    "config_fingerprint",
    "provenance",
    "width",
    "train",
    "validation",
)
FEATURE_STATS_KEYS = (
    "count",
    "width",
    "num_aspects",
    "activation_threshold",
    "dead_feature_count",
    "dead_fraction",
    "features",
)
FEATURE_KEYS = (
    "feature_index",
    "activation_count",
    "aspect_activation_counts",
    "mean_on_firing",
    "variance_on_firing",
    "max_activation",
    "normalized_aspect_entropy",
    "purity",
    "preferred_aspect",
    "preferred_aspect_lift",
    "max_aspect_lift",
    "field_detector_warning",
)
SCHEMA_VERSION = 1


class CheckpointError(ValueError):
    """Raised when a checkpoint is malformed or has incompatible provenance."""


class ArtifactError(ValueError):
    """Raised when a run directory or completed artifact set is invalid."""


class MetricsError(ValueError):
    """Raised when metrics history cannot be reconciled safely."""


def _is_int(value: object) -> bool:
    return type(value) is int


def _strict_json_dumps(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write(path: Path, writer: Callable[[Any], None], mode: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, mode) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_json_atomic(path: Path, payload: object) -> None:
    """Write canonical strict JSON through a sibling temporary file."""

    encoded = (_strict_json_dumps(payload) + "\n").encode("utf-8")
    _atomic_write(Path(path), lambda handle: handle.write(encoded), "wb")


def write_torch_atomic(path: Path, payload: object) -> None:
    """Write a torch payload atomically through a sibling temporary file."""

    _atomic_write(Path(path), lambda handle: torch.save(payload, handle), "wb")


def _path_has_parent_reference(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def _reject_existing_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as error:
            raise ArtifactError(f"Cannot inspect run directory ancestor: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactError(f"Run directory ancestor must not be a symbolic link: {current}")


def prepare_run_directory(run_dir: Path, overwrite: bool = False) -> Path:
    """Create a run directory and enforce the fixed-artifact overwrite policy."""

    run_dir = Path(run_dir)
    if _path_has_parent_reference(run_dir):
        raise ArtifactError("Run directory path contains a parent reference")
    _reject_existing_symlink_components(run_dir)
    if run_dir.is_symlink():
        raise ArtifactError("Run directory must not be a symbolic link")
    if run_dir.exists() and not run_dir.is_dir():
        raise ArtifactError("Run path must be a directory")
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
    absolute = run_dir.absolute()
    try:
        resolved = run_dir.resolve(strict=True)
    except OSError as error:
        raise ArtifactError(f"Cannot resolve run directory: {run_dir}") from error
    if resolved != absolute:
        raise ArtifactError("Run directory path must not traverse symbolic links")

    present: list[Path] = []
    for name in ARTIFACT_NAMES:
        artifact = run_dir / name
        if artifact.parent != run_dir:
            raise ArtifactError("Fixed artifact path escapes the run directory")
        if artifact.is_symlink():
            raise ArtifactError(f"Fixed artifact must not be a symbolic link: {name}")
        if artifact.exists():
            if not artifact.is_file():
                raise ArtifactError(f"Fixed artifact must be a regular file: {name}")
            present.append(artifact)
    if present and not overwrite:
        raise ArtifactError(f"Fixed artifact already exists: {present[0].name}")
    if overwrite:
        for artifact in present:
            artifact.unlink()
    return run_dir


def _validate_sha256(value: object, label: str, error_type: type[ValueError]) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise error_type(f"{label} must be a lowercase SHA-256 checksum")
    return value


def _validate_config(
    value: object,
    fingerprint: object,
    error_type: type[ValueError],
) -> SaePilotConfig:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise error_type("Stored configuration must be a mapping with string keys")
    try:
        config = SaePilotConfig(**value)
        canonical = config.to_dict()
        computed_fingerprint = config.fingerprint()
    except (TypeError, ValueError, OverflowError) as error:
        raise error_type("Stored configuration is invalid") from error
    if value != canonical:
        raise error_type("Stored configuration does not use the canonical schema")
    if type(fingerprint) is not str or fingerprint != computed_fingerprint:
        raise error_type("Stored configuration fingerprint is invalid")
    return config


def _ensure_finite_tree(
    value: object,
    label: str,
    error_type: type[ValueError],
    *,
    require_string_keys: bool = False,
) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all().item()):
                raise error_type(f"{label} must contain only finite tensor values")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact) and not bool(np.isfinite(value).all()):
            raise error_type(f"{label} must contain only finite array values")
        return
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.inexact) and not bool(np.isfinite(value)):
            raise error_type(f"{label} must contain only finite numeric values")
        return
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise error_type(f"{label} must contain only finite numeric values")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if require_string_keys and not isinstance(key, str):
                raise error_type(f"{label} must use string mapping keys")
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise error_type(f"{label} contains an unsupported mapping key")
            _ensure_finite_tree(
                item,
                f"{label}.{key}",
                error_type,
                require_string_keys=require_string_keys,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _ensure_finite_tree(
                item,
                f"{label}[{index}]",
                error_type,
                require_string_keys=require_string_keys,
            )
        return
    raise error_type(f"{label} contains unsupported metadata type {type(value).__name__}")


def _validate_model_state(
    value: object,
    config: SaePilotConfig,
    error_type: type[ValueError],
) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != set(MODEL_STATE_KEYS):
        raise error_type("model_state has invalid keys")
    expected = {
        "encoder_weight": (config.width, config.input_dim),
        "decoder_weight": (config.input_dim, config.width),
        "encoder_bias": (config.width,),
        "pre_bias": (config.input_dim,),
        "steps_since_active": (config.width,),
    }
    for name, shape in expected.items():
        tensor = value[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise error_type(f"model_state tensor {name} has the wrong shape")
        expected_dtype = torch.long if name == "steps_since_active" else torch.float32
        if tensor.dtype != expected_dtype:
            raise error_type(f"model_state tensor {name} has the wrong dtype")
        _ensure_finite_tree(tensor, f"model_state.{name}", error_type)
    return value  # type: ignore[return-value]


def _validated_rng_state(
    state: object,
    *,
    require_cuda_available: bool,
) -> Mapping[str, object]:
    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise CheckpointError("RNG state has invalid keys")
    _ensure_finite_tree(state, "RNG state", CheckpointError)
    python_state = state["python"]
    try:
        validator = random.Random()
        validator.setstate(python_state)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise CheckpointError("Python RNG state is invalid") from error
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise CheckpointError("NumPy RNG state has invalid schema")
    try:
        np.random.RandomState().set_state(numpy_state)
    except (TypeError, ValueError) as error:
        raise CheckpointError("NumPy RNG state is invalid") from error
    cpu_state = state["torch_cpu"]
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.device.type != "cpu"
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
    ):
        raise CheckpointError("Torch CPU RNG state is invalid")
    try:
        torch.Generator(device="cpu").set_state(cpu_state.clone())
    except RuntimeError as error:
        raise CheckpointError("Torch CPU RNG state has an invalid length") from error
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list) or any(
        not isinstance(item, torch.Tensor)
        or item.device.type != "cpu"
        or item.dtype != torch.uint8
        or item.ndim != 1
        for item in cuda_states
    ):
        raise CheckpointError("Torch CUDA RNG state is invalid")
    if require_cuda_available:
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            if cuda_states:
                raise CheckpointError("CUDA RNG state must be empty without CUDA")
        else:
            device_count = torch.cuda.device_count()
            current_states = torch.cuda.get_rng_state_all()
            if len(current_states) != device_count:
                raise CheckpointError("CUDA RNG current device state count is inconsistent")
            if len(cuda_states) != device_count:
                raise CheckpointError("CUDA RNG state count does not match device count")
            for index, (stored, current) in enumerate(zip(cuda_states, current_states)):
                if stored.shape != current.shape or stored.numel() != current.numel():
                    raise CheckpointError(
                        f"CUDA RNG state shape does not match device {index}"
                    )
    return state


def capture_rng_state() -> dict[str, object]:
    """Capture Python, NumPy, Torch CPU, and every available CUDA RNG state."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _clone_rng_state(state: Mapping[str, object]) -> dict[str, object]:
    validated = _validated_rng_state(state, require_cuda_available=False)
    return {
        "python": copy.deepcopy(validated["python"]),
        "numpy": copy.deepcopy(validated["numpy"]),
        "torch_cpu": validated["torch_cpu"].clone(),  # type: ignore[union-attr]
        "torch_cuda": [
            tensor.clone() for tensor in validated["torch_cuda"]  # type: ignore[union-attr]
        ],
    }


def restore_rng_state(state: object) -> None:
    """Validate an entire RNG payload, then restore it without dropping devices."""

    validated = _validated_rng_state(state, require_cuda_available=True)
    random.setstate(validated["python"])  # type: ignore[arg-type]
    np.random.set_state(validated["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(validated["torch_cpu"])  # type: ignore[arg-type]
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(validated["torch_cuda"])  # type: ignore[arg-type]


def build_checkpoint_payload(
    *,
    config: SaePilotConfig,
    data_index: SaeDataIndex,
    epoch: int,
    global_step: int,
    best_validation_nmse: float,
    patience_counter: int,
    epoch_metrics: Mapping[str, object],
    model_state: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, object],
    rng_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build and validate the exact checkpoint schema consumed by resume."""

    if not isinstance(config, SaePilotConfig):
        raise TypeError("config must be SaePilotConfig")
    if not isinstance(data_index, SaeDataIndex):
        raise TypeError("data_index must be SaeDataIndex")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "assignment_sha256": data_index.assignment_sha256,
        "text_manifest_sha256": data_index.text_manifest_sha256,
        "activation_manifest_sha256": data_index.activation_manifest_sha256,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_nmse": best_validation_nmse,
        "patience_counter": patience_counter,
        "epoch_metrics": dict(epoch_metrics),
        "model_state": dict(model_state),
        "optimizer_state": dict(optimizer_state),
        "rng_state": capture_rng_state()
        if rng_state is None
        else _clone_rng_state(rng_state),
    }
    _validate_checkpoint_payload(payload)
    return payload


def _validate_checkpoint_payload(
    payload: object,
    *,
    expected_config: SaePilotConfig | None = None,
    expected_assignment_sha256: str | None = None,
    expected_text_manifest_sha256: str | None = None,
    expected_activation_manifest_sha256: str | None = None,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint payload must be a mapping")
    if set(payload) != set(CHECKPOINT_KEYS):
        raise CheckpointError("Checkpoint keys do not match the exact schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise CheckpointError("Checkpoint schema version is unsupported")
    config = _validate_config(
        payload["config"], payload["config_fingerprint"], CheckpointError
    )
    if expected_config is not None and (
        config.to_dict() != expected_config.to_dict()
        or payload["config_fingerprint"] != expected_config.fingerprint()
    ):
        raise CheckpointError("Checkpoint configuration does not match")
    observed_provenance = {
        key: _validate_sha256(payload[key], key, CheckpointError)
        for key in PROVENANCE_KEYS
    }
    expected_provenance = {
        "assignment_sha256": expected_assignment_sha256,
        "text_manifest_sha256": expected_text_manifest_sha256,
        "activation_manifest_sha256": expected_activation_manifest_sha256,
    }
    for key, expected in expected_provenance.items():
        if expected is not None and observed_provenance[key] != expected:
            raise CheckpointError(f"Checkpoint provenance mismatch for {key}")
    for key in ("epoch", "global_step", "patience_counter"):
        value = payload[key]
        if not _is_int(value) or int(value) < 0:
            raise CheckpointError(f"Checkpoint {key} must be a nonnegative integer")
    score = payload["best_validation_nmse"]
    if type(score) is not float or not math.isfinite(score) or score < 0:
        raise CheckpointError("Checkpoint best_validation_nmse must be finite and nonnegative")
    metrics = payload["epoch_metrics"]
    if not isinstance(metrics, dict):
        raise CheckpointError("Checkpoint epoch_metrics must be a mapping")
    metric_epoch = metrics.get("epoch")
    if not _is_int(metric_epoch) or int(metric_epoch) != int(payload["epoch"]):
        raise CheckpointError("Checkpoint epoch_metrics epoch does not match checkpoint epoch")
    _ensure_finite_tree(
        metrics,
        "checkpoint epoch_metrics",
        CheckpointError,
        require_string_keys=True,
    )
    _validate_model_state(payload["model_state"], config, CheckpointError)
    optimizer = payload["optimizer_state"]
    if not isinstance(optimizer, Mapping):
        raise CheckpointError("Checkpoint optimizer_state must be a mapping")
    _ensure_finite_tree(optimizer, "checkpoint optimizer_state", CheckpointError)
    _validated_rng_state(payload["rng_state"], require_cuda_available=False)
    _ensure_finite_tree(payload, "checkpoint", CheckpointError)
    return payload


def save_checkpoint_atomic(path: Path, payload: object) -> None:
    """Validate and atomically save one exact-schema training checkpoint."""

    validated = _validate_checkpoint_payload(payload)
    write_torch_atomic(Path(path), validated)


def _torch_load(path: Path, error_type: type[ValueError], label: str) -> object:
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as error:
        raise error_type(f"Cannot load {label}: {path}") from error


def load_checkpoint(
    path: Path,
    *,
    expected_config: SaePilotConfig,
    expected_assignment_sha256: str,
    expected_text_manifest_sha256: str,
    expected_activation_manifest_sha256: str,
) -> dict[str, object]:
    """Load on CPU, fully validate schema/provenance, and only then return."""

    path = Path(path)
    payload = _torch_load(path, CheckpointError, "checkpoint")
    return _validate_checkpoint_payload(
        payload,
        expected_config=expected_config,
        expected_assignment_sha256=expected_assignment_sha256,
        expected_text_manifest_sha256=expected_text_manifest_sha256,
        expected_activation_manifest_sha256=expected_activation_manifest_sha256,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _parse_json_object(
    text: str,
    label: str,
    error_type: type[ValueError],
) -> dict[str, object]:
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise error_type(f"Invalid JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise error_type(f"{label} must be a JSON object")
    _ensure_finite_tree(value, label, error_type, require_string_keys=True)
    return value


def _read_metrics_rows(path: Path, error_type: type[ValueError]) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise error_type(f"Cannot read metrics JSONL: {path}") from error
    rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = _parse_json_object(line, f"metrics line {line_number}", error_type)
        epoch = row.get("epoch")
        if not _is_int(epoch) or int(epoch) < 0:
            raise error_type(f"metrics line {line_number} has invalid epoch")
        integer_epoch = int(epoch)
        if integer_epoch in seen:
            raise error_type(f"metrics contains duplicate epoch {integer_epoch}")
        seen.add(integer_epoch)
        rows.append(row)
    return rows


def _write_metrics_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    encoded = "".join(_strict_json_dumps(row) + "\n" for row in rows).encode("utf-8")
    _atomic_write(path, lambda handle: handle.write(encoded), "wb")


def reconcile_metrics_jsonl(
    path: Path,
    checkpoint: Mapping[str, object],
) -> list[dict[str, object]]:
    """Reconcile history to a checkpoint and require a continuous 0..epoch prefix."""

    validated = _validate_checkpoint_payload(dict(checkpoint))
    checkpoint_epoch = int(validated["epoch"])
    rows = _read_metrics_rows(Path(path), MetricsError)
    before = {int(row["epoch"]): row for row in rows if int(row["epoch"]) < checkpoint_epoch}
    expected_before = set(range(checkpoint_epoch))
    if set(before) != expected_before:
        raise MetricsError("Metrics history before checkpoint must be continuous from epoch 0")
    reconciled = [before[epoch] for epoch in range(checkpoint_epoch)]
    reconciled.append(dict(validated["epoch_metrics"]))
    _write_metrics_atomic(Path(path), reconciled)
    return reconciled


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_file(path: Path, label: str) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactError(f"Cannot read {label}: {path}") from error
    return _parse_json_object(text, label, ArtifactError)


def _validate_provenance(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(PROVENANCE_KEYS):
        raise ArtifactError(f"{label} provenance must contain the exact source checksums")
    return {
        key: _validate_sha256(value[key], f"{label} provenance {key}", ArtifactError)
        for key in PROVENANCE_KEYS
    }


def _validate_artifact_common(
    payload: object,
    *,
    label: str,
    config: SaePilotConfig,
    provenance: Mapping[str, str],
    exact_keys: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ArtifactError(f"{label} must be a mapping")
    if set(payload) != set(exact_keys):
        raise ArtifactError(f"{label} must contain the exact top-level keys")
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"{label} schema version is unsupported")
    observed_config = _validate_config(
        payload.get("config"), payload.get("config_fingerprint"), ArtifactError
    )
    if observed_config.to_dict() != config.to_dict():
        raise ArtifactError(f"{label} configuration does not match summary")
    if _validate_provenance(payload.get("provenance"), label) != dict(provenance):
        raise ArtifactError(f"{label} provenance does not match summary")
    _ensure_finite_tree(payload, label, ArtifactError)
    return payload


def _require_exact_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ArtifactError(f"{label} must be a finite float")
    return value


def _validate_feature_statistics(
    value: object,
    *,
    label: str,
    expected_width: int,
) -> None:
    if not isinstance(value, dict) or set(value) != set(FEATURE_STATS_KEYS):
        raise ArtifactError(f"{label} must contain the exact feature-statistics keys")
    count = value["count"]
    width = value["width"]
    num_aspects = value["num_aspects"]
    dead_count = value["dead_feature_count"]
    if not _is_int(count) or count < 0:
        raise ArtifactError(f"{label} count must be a nonnegative integer")
    if not _is_int(width) or width != expected_width:
        raise ArtifactError(f"{label} width does not match configuration")
    if not _is_int(num_aspects) or num_aspects != 24:
        raise ArtifactError(f"{label} num_aspects must be exactly 24")
    threshold = _require_exact_float(
        value["activation_threshold"], f"{label} activation_threshold"
    )
    if threshold != 1e-3:
        raise ArtifactError(f"{label} activation_threshold must be exactly 1e-3")
    if not _is_int(dead_count) or not 0 <= dead_count <= width:
        raise ArtifactError(f"{label} dead_feature_count is out of range")
    dead_fraction = _require_exact_float(
        value["dead_fraction"], f"{label} dead_fraction"
    )
    if not 0.0 <= dead_fraction <= 1.0 or not math.isclose(
        dead_fraction, dead_count / width, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ArtifactError(f"{label} dead_fraction is inconsistent")
    features = value["features"]
    if not isinstance(features, list) or len(features) != width:
        raise ArtifactError(f"{label} features must contain exactly width entries")

    observed_dead = 0
    for expected_index, feature in enumerate(features):
        feature_label = f"{label} feature {expected_index}"
        if not isinstance(feature, dict) or set(feature) != set(FEATURE_KEYS):
            raise ArtifactError(f"{feature_label} must contain the exact feature keys")
        feature_index = feature["feature_index"]
        activation_count = feature["activation_count"]
        preferred_aspect = feature["preferred_aspect"]
        if not _is_int(feature_index) or feature_index != expected_index:
            raise ArtifactError(f"{feature_label} index must be continuous")
        if not _is_int(activation_count) or not 0 <= activation_count <= count:
            raise ArtifactError(f"{feature_label} activation_count is out of range")
        if not _is_int(preferred_aspect):
            raise ArtifactError(f"{feature_label} preferred_aspect must be an integer")
        aspect_counts = feature["aspect_activation_counts"]
        if (
            not isinstance(aspect_counts, list)
            or len(aspect_counts) != 24
            or any(not _is_int(item) or item < 0 for item in aspect_counts)
            or sum(aspect_counts) != activation_count
        ):
            raise ArtifactError(f"{feature_label} aspect activation counts are invalid")
        mean = _require_exact_float(feature["mean_on_firing"], f"{feature_label} mean")
        variance = _require_exact_float(
            feature["variance_on_firing"], f"{feature_label} variance"
        )
        maximum = _require_exact_float(
            feature["max_activation"], f"{feature_label} maximum"
        )
        entropy = _require_exact_float(
            feature["normalized_aspect_entropy"], f"{feature_label} entropy"
        )
        purity = _require_exact_float(feature["purity"], f"{feature_label} purity")
        preferred_lift = _require_exact_float(
            feature["preferred_aspect_lift"], f"{feature_label} preferred lift"
        )
        maximum_lift = _require_exact_float(
            feature["max_aspect_lift"], f"{feature_label} maximum lift"
        )
        warning = feature["field_detector_warning"]
        if type(warning) is not bool:
            raise ArtifactError(f"{feature_label} warning must be boolean")
        if (
            mean < 0.0
            or variance < 0.0
            or maximum < mean
            or not 0.0 <= entropy <= 1.0
            or not 0.0 <= purity <= 1.0
            or preferred_lift < 0.0
            or maximum_lift < preferred_lift
        ):
            raise ArtifactError(f"{feature_label} numeric value is out of range")

        if activation_count == 0:
            observed_dead += 1
            if (
                mean != 0.0
                or variance != 0.0
                or maximum != 0.0
                or entropy != 0.0
                or purity != 0.0
                or preferred_aspect != -1
                or preferred_lift != 0.0
                or maximum_lift != 0.0
                or warning
            ):
                raise ArtifactError(f"{feature_label} dead-feature defaults are invalid")
            continue
        if not 0 <= preferred_aspect < 24:
            raise ArtifactError(f"{feature_label} preferred_aspect is out of range")
        expected_preferred = aspect_counts.index(max(aspect_counts))
        expected_purity = aspect_counts[expected_preferred] / activation_count
        probabilities = [item / activation_count for item in aspect_counts if item]
        expected_entropy = -sum(item * math.log(item) for item in probabilities) / math.log(24)
        if preferred_aspect != expected_preferred:
            raise ArtifactError(f"{feature_label} preferred_aspect is inconsistent")
        if not math.isclose(purity, expected_purity, rel_tol=0.0, abs_tol=1e-12):
            raise ArtifactError(f"{feature_label} purity is inconsistent")
        if not math.isclose(entropy, expected_entropy, rel_tol=0.0, abs_tol=1e-12):
            raise ArtifactError(f"{feature_label} entropy is inconsistent")
        expected_warning = activation_count >= 20 and purity >= 0.80 and preferred_lift >= 2.0
        if warning is not expected_warning:
            raise ArtifactError(f"{feature_label} warning is inconsistent")
    if observed_dead != dead_count:
        raise ArtifactError(f"{label} dead_feature_count is inconsistent")


def validate_pre_summary_artifacts(
    run_dir: Path,
    summary_core: Mapping[str, object],
) -> dict[str, object]:
    """Validate the seven non-summary artifacts against an in-memory summary core."""

    run_dir = Path(run_dir)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ArtifactError("Run directory must be a real directory")
    paths = {name: run_dir / name for name in SUMMARY_FILE_NAMES}
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ArtifactError(f"Artifact must be a regular file: {name}")

    if not isinstance(summary_core, Mapping):
        raise ArtifactError("summary core must be a mapping")
    summary = dict(summary_core)
    required_summary_keys = {
        "schema_version",
        "config",
        "config_fingerprint",
        "provenance",
        "best_epoch",
        "final_epoch",
        "files",
    }
    if type(summary.get("schema_version")) is not int or summary.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError("summary schema version is unsupported")
    if not required_summary_keys <= set(summary):
        raise ArtifactError("summary core is missing required keys")
    _ensure_finite_tree(
        summary, "summary core", ArtifactError, require_string_keys=True
    )
    config = _validate_config(
        summary.get("config"), summary.get("config_fingerprint"), ArtifactError
    )
    provenance = _validate_provenance(summary.get("provenance"), "summary")
    best_epoch = summary.get("best_epoch")
    final_epoch = summary.get("final_epoch")
    if not _is_int(final_epoch) or int(final_epoch) < 0:
        raise ArtifactError("summary final_epoch must be a nonnegative integer")
    if (
        not _is_int(best_epoch)
        or int(best_epoch) < 0
        or int(best_epoch) > int(final_epoch)
    ):
        raise ArtifactError("summary best_epoch must be within 0..final_epoch")

    files = summary.get("files")
    if not isinstance(files, dict) or set(files) != set(SUMMARY_FILE_NAMES):
        raise ArtifactError("summary files must contain exactly the seven fixed basenames")
    for name in SUMMARY_FILE_NAMES:
        metadata = files[name]
        if not isinstance(metadata, dict) or set(metadata) != {"bytes", "sha256"}:
            raise ArtifactError(f"summary files metadata is invalid for {name}")
        size = metadata["bytes"]
        if not _is_int(size) or int(size) < 0:
            raise ArtifactError(f"summary file size is invalid for {name}")
        checksum = _validate_sha256(metadata["sha256"], f"summary checksum {name}", ArtifactError)
        if paths[name].stat().st_size != int(size):
            raise ArtifactError(f"Artifact size mismatch for {name}")
        if _sha256(paths[name]) != checksum:
            raise ArtifactError(f"Artifact checksum mismatch for {name}")

    config_file = _load_json_file(paths["config.json"], "config JSON")
    if config_file != config.to_dict():
        raise ArtifactError("config.json does not match summary configuration")
    split_manifest = _load_json_file(
        paths["split_manifest.json"], "split manifest JSON"
    )
    split_provenance = {key: split_manifest.get(key) for key in PROVENANCE_KEYS}
    if split_provenance != provenance:
        raise ArtifactError("split manifest provenance does not match summary")
    metrics = _read_metrics_rows(paths["metrics.jsonl"], ArtifactError)
    observed_epochs = [int(row["epoch"]) for row in metrics]
    if observed_epochs != list(range(int(final_epoch) + 1)):
        raise ArtifactError("Metrics epochs must be strictly ordered from 0 through final_epoch")

    loaded_checkpoints: dict[str, dict[str, object]] = {}
    for name in ("best.pt", "last.pt"):
        try:
            loaded_checkpoints[name] = _validate_checkpoint_payload(
                _torch_load(paths[name], ArtifactError, name),
                expected_config=config,
                expected_assignment_sha256=provenance["assignment_sha256"],
                expected_text_manifest_sha256=provenance["text_manifest_sha256"],
                expected_activation_manifest_sha256=provenance["activation_manifest_sha256"],
            )
        except CheckpointError as error:
            raise ArtifactError(f"Invalid {name}: {error}") from error
    if int(loaded_checkpoints["best.pt"]["epoch"]) != int(best_epoch):
        raise ArtifactError("best.pt epoch does not match summary best_epoch")
    if int(loaded_checkpoints["last.pt"]["epoch"]) != int(final_epoch):
        raise ArtifactError("last.pt epoch does not match summary final_epoch")
    if loaded_checkpoints["best.pt"]["epoch_metrics"] != metrics[int(best_epoch)]:
        raise ArtifactError("best.pt epoch metrics do not match metrics.jsonl")
    if loaded_checkpoints["last.pt"]["epoch_metrics"] != metrics[int(final_epoch)]:
        raise ArtifactError("last.pt epoch metrics do not match metrics.jsonl")

    inference = _validate_artifact_common(
        _torch_load(paths["inference.pt"], ArtifactError, "inference artifact"),
        label="inference artifact",
        config=config,
        provenance=provenance,
        exact_keys=INFERENCE_KEYS,
    )
    _validate_model_state(inference.get("model_state"), config, ArtifactError)

    feature_stats = _validate_artifact_common(
        _torch_load(paths["feature_stats.pt"], ArtifactError, "feature statistics"),
        label="feature statistics",
        config=config,
        provenance=provenance,
        exact_keys=FEATURE_STATS_ARTIFACT_KEYS,
    )
    width = feature_stats.get("width")
    if not _is_int(width) or int(width) != config.width:
        raise ArtifactError("feature statistics width does not match configuration")
    for split in ("train", "validation"):
        _validate_feature_statistics(
            feature_stats[split],
            label=f"feature statistics {split}",
            expected_width=config.width,
        )
    return summary


def validate_run_artifacts(run_dir: Path) -> dict[str, object]:
    """Load summary JSON and reuse the seven-artifact core validator."""

    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ArtifactError("summary artifact must be a regular file")
    summary = _load_json_file(summary_path, "summary JSON")
    return validate_pre_summary_artifacts(run_dir, summary)
