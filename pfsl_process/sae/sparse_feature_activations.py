#!/usr/bin/env python3
"""Extract and validate frozen-SAE post-TopK activations as mirrored CSR shards."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pfsl_process.sae import sae_checkpointing as checkpointing
    from pfsl_process.sae.feature_atlas_config import (
        AtlasPaths,
        AtlasSourceBinding,
        DEFAULT_ACTIVATION_DIR,
        DEFAULT_CORPUS_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_RUN_DIR,
        build_source_binding,
        validate_source_binding,
    )
    from pfsl_process.sae.feature_atlas_io import (
        AtlasIOError,
        sha256_file,
        strict_json_load,
        write_json_atomic,
        write_torch_atomic,
    )
    from pfsl_process.sae.sae_pilot_config import SaePilotConfig
    from pfsl_process.sae.sae_training_data import (
        SaeDataIndex,
        ShardMetadata,
        build_data_index,
    )
    from pfsl_process.sae.topk_sae import TopKSAE
else:
    from . import sae_checkpointing as checkpointing
    from .feature_atlas_config import (
        AtlasPaths,
        AtlasSourceBinding,
        DEFAULT_ACTIVATION_DIR,
        DEFAULT_CORPUS_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_RUN_DIR,
        build_source_binding,
        validate_source_binding,
    )
    from .feature_atlas_io import (
        AtlasIOError,
        sha256_file,
        strict_json_load,
        write_json_atomic,
        write_torch_atomic,
    )
    from .sae_pilot_config import SaePilotConfig
    from .sae_training_data import SaeDataIndex, ShardMetadata, build_data_index
    from .topk_sae import TopKSAE


SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
SPOT_CHECK_ROWS = 64
SPOT_CHECK_ATOL = 1e-6
SUM_ATOL = 1e-7
SUM_RTOL = 1e-6


class SparseActivationError(ValueError):
    """Raised when sparse extraction inputs or outputs violate their contract."""


_BINDING_FIELDS = tuple(AtlasSourceBinding.__dataclass_fields__)


def _binding_dict(binding: AtlasSourceBinding) -> dict[str, str]:
    return {name: getattr(binding, name) for name in _BINDING_FIELDS}


def _validate_atlas_source_context(
    source_index: SaeDataIndex,
    inference_path: Path,
    output_dir: Path,
    *,
    feature_stats_path: Path | None,
    atlas_paths: AtlasPaths | None,
    source_binding: AtlasSourceBinding | None,
    allow_test_fixture_unbound: bool,
) -> tuple[dict[str, str] | None, bool]:
    if type(allow_test_fixture_unbound) is not bool:
        raise TypeError("allow_test_fixture_unbound must be a bool")
    if (atlas_paths is None) != (source_binding is None):
        raise SparseActivationError(
            "atlas_paths and source_binding must be provided together"
        )
    if atlas_paths is None or source_binding is None:
        if not allow_test_fixture_unbound:
            raise SparseActivationError(
                "unbound sparse activation operation is forbidden outside test fixtures"
            )
        return None, True
    if allow_test_fixture_unbound:
        raise SparseActivationError(
            "bound operations cannot enable allow_test_fixture_unbound"
        )
    if not isinstance(atlas_paths, AtlasPaths):
        raise TypeError("atlas_paths must be AtlasPaths")
    if not isinstance(source_binding, AtlasSourceBinding):
        raise TypeError("source_binding must be AtlasSourceBinding")
    expected_inference = atlas_paths.run_dir / "inference.pt"
    if Path(inference_path).resolve() != expected_inference.resolve():
        raise SparseActivationError("inference_path does not match AtlasPaths run_dir")
    expected_output = atlas_paths.output_dir / "sparse_activations"
    if Path(output_dir).resolve() != expected_output.resolve():
        raise SparseActivationError(
            "sparse output_dir does not match AtlasPaths output_dir"
        )
    if feature_stats_path is not None and (
        Path(feature_stats_path).resolve()
        != (atlas_paths.run_dir / "feature_stats.pt").resolve()
    ):
        raise SparseActivationError(
            "feature_stats_path does not match AtlasPaths run_dir"
        )
    bound_files = (
        ("inference_sha256", "inference.pt", expected_inference),
        (
            "feature_stats_sha256",
            "feature_stats.pt",
            atlas_paths.run_dir / "feature_stats.pt",
        ),
        (
            "split_manifest_sha256",
            "split_manifest.json",
            atlas_paths.run_dir / "split_manifest.json",
        ),
        (
            "summary_sha256",
            "summary.json",
            atlas_paths.run_dir / "summary.json",
        ),
        ("units_sha256", "sae_units.jsonl", atlas_paths.units_path),
        (
            "activation_manifest_sha256",
            "activation manifest.json",
            atlas_paths.activation_dir / "manifest.json",
        ),
        (
            "text_manifest_sha256",
            "text_corpus_manifest.json",
            atlas_paths.text_manifest_path,
        ),
        (
            "filtered_uniprot_sha256",
            atlas_paths.filtered_uniprot_tsv.name,
            atlas_paths.filtered_uniprot_tsv,
        ),
    )
    for field_name, label, path in bound_files:
        try:
            observed = sha256_file(path)
        except (AtlasIOError, OSError, ValueError) as error:
            raise SparseActivationError(f"Cannot validate bound {label}: {path}") from error
        if observed != getattr(source_binding, field_name):
            raise SparseActivationError(f"bound {label} checksum mismatch")
    if source_binding.assignment_sha256 != source_index.assignment_sha256:
        raise SparseActivationError("source binding assignment mismatch")
    try:
        validate_source_binding(source_binding, atlas_paths)
    except Exception as error:
        raise SparseActivationError(f"Task 1 source binding validation failed: {error}") from error
    return _binding_dict(source_binding), False


def _resolve_device_identity(
    requested: str | torch.device,
) -> tuple[torch.device, dict[str, object]]:
    try:
        parsed = torch.device(requested)
    except (TypeError, RuntimeError, ValueError) as error:
        raise SparseActivationError(f"Invalid device: {requested!r}") from error
    if parsed.type == "cpu":
        return torch.device("cpu"), {"kind": "cpu", "logical_device": "cpu"}
    if parsed.type != "cuda":
        raise SparseActivationError("Only cpu and cuda devices are supported")
    if not torch.cuda.is_available():
        raise SparseActivationError("CUDA was requested but is unavailable")
    current_index = int(torch.cuda.current_device())
    logical_index = current_index if parsed.index is None else parsed.index
    device_count = int(torch.cuda.device_count())
    if logical_index < 0 or logical_index >= device_count:
        raise SparseActivationError("CUDA logical device index is out of range")
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visible is None:
        physical_token = str(logical_index)
        physical_index: int | None = logical_index
    else:
        tokens = tuple(token.strip() for token in raw_visible.split(",") if token.strip())
        if logical_index >= len(tokens):
            raise SparseActivationError(
                "CUDA_VISIBLE_DEVICES mapping does not contain the logical device"
            )
        physical_token = tokens[logical_index]
        physical_index = int(physical_token) if physical_token.isdecimal() else None
    device_name = str(torch.cuda.get_device_name(logical_index))
    logical_device = f"cuda:{logical_index}"
    identity: dict[str, object] = {
        "cuda_visible_devices": raw_visible,
        "current_logical_index": current_index,
        "device_identity": f"{physical_token}|{device_name}",
        "device_name": device_name,
        "kind": "cuda",
        "logical_device": logical_device,
        "physical_index": physical_index,
        "physical_token": physical_token,
    }
    return torch.device(logical_device), identity


def _validate_device_identity(recorded: object) -> torch.device:
    if not isinstance(recorded, dict):
        raise SparseActivationError("Manifest device identity is invalid")
    if recorded.get("kind") == "cpu":
        if recorded != {"kind": "cpu", "logical_device": "cpu"}:
            raise SparseActivationError("Manifest CPU identity is invalid")
        return torch.device("cpu")
    expected_keys = {
        "cuda_visible_devices",
        "current_logical_index",
        "device_identity",
        "device_name",
        "kind",
        "logical_device",
        "physical_index",
        "physical_token",
    }
    if set(recorded) != expected_keys or recorded.get("kind") != "cuda":
        raise SparseActivationError("Manifest CUDA device identity is invalid")
    current_device, current = _resolve_device_identity(
        str(recorded.get("logical_device"))
    )
    if current.get("current_logical_index") != recorded.get("current_logical_index"):
        raise SparseActivationError("CUDA current device changed since extraction")
    mapping_keys = (
        "cuda_visible_devices", "physical_token", "physical_index"
    )
    if any(current.get(key) != recorded.get(key) for key in mapping_keys):
        raise SparseActivationError("CUDA visible-to-physical mapping changed")
    identity_keys = ("logical_device", "device_name", "device_identity")
    if any(current.get(key) != recorded.get(key) for key in identity_keys):
        raise SparseActivationError("CUDA device identity changed")
    return current_device


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SparseActivationError(f"{label} must be a built-in integer >= {minimum}")
    return value


def _torch_payload(path: Path, label: str) -> dict[str, object]:
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as error:
        raise SparseActivationError(f"Cannot load {label}: {path}") from error
    if not isinstance(payload, dict):
        raise SparseActivationError(f"{label} must be a mapping: {path}")
    return payload


def _load_inference(
    inference_path: Path,
    device: str | torch.device,
) -> tuple[TopKSAE, SaePilotConfig, dict[str, str], str]:
    inference_path = Path(inference_path)
    try:
        inference_sha256 = sha256_file(inference_path)
        payload = _torch_payload(inference_path, "inference artifact")
        if set(payload) != set(checkpointing.INFERENCE_KEYS):
            raise SparseActivationError(
                "Inference artifact must contain the exact inference schema"
            )
        config = checkpointing._validate_config(
            payload.get("config"),
            payload.get("config_fingerprint"),
            checkpointing.ArtifactError,
        )
        provenance = checkpointing._validate_provenance(
            payload.get("provenance"), "inference artifact"
        )
        checkpointing._validate_artifact_common(
            payload,
            label="inference artifact",
            config=config,
            provenance=provenance,
            exact_keys=checkpointing.INFERENCE_KEYS,
        )
        state = checkpointing._validate_model_state(
            payload.get("model_state"), config, checkpointing.ArtifactError
        )
        resolved_device = torch.device(device)
        model = TopKSAE(
            input_dim=config.input_dim,
            width=config.width,
            k=config.k,
            aux_k=config.aux_k,
            activation_threshold=config.activation_threshold,
            dead_steps_threshold=1,
        )
        model.load_state_dict(dict(state), strict=True)
        model.eval().requires_grad_(False)
        model.to(resolved_device)
    except SparseActivationError:
        raise
    except (AtlasIOError, checkpointing.ArtifactError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SparseActivationError(f"Invalid frozen inference artifact: {error}") from error
    return model, config, provenance, inference_sha256


def load_frozen_sae(
    inference_path: Path,
    device: str | torch.device,
) -> TopKSAE:
    """Load the exact inference artifact as an eval-only, gradient-free SAE."""

    model, _, _, _ = _load_inference(inference_path, device)
    return model


def dense_to_csr(
    activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a float32 activation matrix to deterministic positive-only CSR."""

    if not isinstance(activations, torch.Tensor):
        raise SparseActivationError("activations must be a tensor")
    if activations.dtype != torch.float32:
        raise SparseActivationError("activations must have dtype float32")
    if activations.ndim != 2:
        raise SparseActivationError("activations must have shape [rows, features]")
    if not bool(torch.isfinite(activations).all().item()):
        raise SparseActivationError("activations must contain only finite values")
    positive = activations > 0
    counts = positive.sum(dim=1, dtype=torch.int64)
    indptr = torch.cat(
        (torch.zeros(1, dtype=torch.int64), counts.detach().cpu().cumsum(0))
    )
    row, column = positive.nonzero(as_tuple=True)
    if len(row):
        order = torch.argsort(
            row.to(torch.int64) * activations.shape[1] + column.to(torch.int64)
        )
        row = row.index_select(0, order)
        column = column.index_select(0, order)
        values = activations[row, column]
    else:
        values = activations.new_empty((0,))
    return (
        indptr,
        column.detach().to(device="cpu", dtype=torch.int32),
        values.detach().to(device="cpu", dtype=torch.float32),
    )


def _validate_index_layout(index: SaeDataIndex) -> None:
    if not isinstance(index, SaeDataIndex):
        raise TypeError("source_index must be SaeDataIndex")
    if not index.units or not index.shards:
        raise SparseActivationError("Source index must contain units and shards")
    for row_index, unit in enumerate(index.units):
        if unit.row_index != row_index:
            raise SparseActivationError("Source index rows are not in exact source order")
        if not unit.unit_id or not 0 <= unit.aspect_index < 24:
            raise SparseActivationError("Source index contains invalid unit metadata")
    next_row = 0
    for shard_number, shard in enumerate(index.shards):
        if (
            shard.start_row != next_row
            or shard.end_row <= shard.start_row
            or shard.end_row > len(index.units)
        ):
            raise SparseActivationError(
                f"Source shard {shard_number} does not mirror contiguous source ranges"
            )
        next_row = shard.end_row
    if next_row != len(index.units):
        raise SparseActivationError("Source shards do not cover every source row")
    train_count = sum(unit.split == "train" for unit in index.units)
    validation_count = len(index.units) - train_count
    if (train_count, validation_count) != (
        index.train_unit_count,
        index.validation_unit_count,
    ):
        raise SparseActivationError("Source split counts do not match the data index")


def _check_provenance(index: SaeDataIndex, provenance: Mapping[str, str]) -> None:
    expected = {
        "assignment_sha256": index.assignment_sha256,
        "text_manifest_sha256": index.text_manifest_sha256,
        "activation_manifest_sha256": index.activation_manifest_sha256,
    }
    if dict(provenance) != expected:
        raise SparseActivationError("Inference provenance does not match source index")


def _load_source_shard(
    index: SaeDataIndex,
    shard: ShardMetadata,
    input_dim: int,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    try:
        observed = sha256_file(shard.path)
    except (AtlasIOError, OSError, ValueError) as error:
        raise SparseActivationError(f"Cannot hash source shard: {shard.path}") from error
    if observed != shard.sha256:
        raise SparseActivationError(f"source shard checksum mismatch: {shard.path}")
    payload = _torch_payload(shard.path, "source activation shard")
    rows = shard.end_row - shard.start_row
    stored_ids = payload.get("unit_ids")
    if (
        not isinstance(stored_ids, (list, tuple))
        or len(stored_ids) != rows
        or any(type(unit_id) is not str or not unit_id for unit_id in stored_ids)
    ):
        raise SparseActivationError(f"Invalid unit IDs in source shard: {shard.path}")
    expected_ids = tuple(
        unit.unit_id for unit in index.units[shard.start_row : shard.end_row]
    )
    if tuple(stored_ids) != expected_ids:
        raise SparseActivationError(
            f"Source shard unit IDs violate exact source order: {shard.path}"
        )
    embeddings = payload.get("embeddings")
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.dtype != torch.float32
        or tuple(embeddings.shape) != (rows, input_dim)
        or not bool(torch.isfinite(embeddings).all().item())
    ):
        raise SparseActivationError(f"Invalid embeddings in source shard: {shard.path}")
    norms = embeddings.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0):
        raise SparseActivationError(f"Source shard embeddings are not normalized: {shard.path}")
    return embeddings, expected_ids


def _unit_ids_sha256(unit_ids: tuple[str, ...]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for unit_id in unit_ids:
        digest.update(unit_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


_SHARD_KEYS = {
    "schema_version",
    "start_row",
    "end_row",
    "unit_ids",
    "indptr",
    "feature_indices",
    "activation_values",
}


def _validate_sparse_shard(
    payload: dict[str, object],
    *,
    start_row: int,
    end_row: int,
    expected_ids: tuple[str, ...],
    width: int,
    k: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if set(payload) != _SHARD_KEYS or payload.get("schema_version") != SCHEMA_VERSION:
        raise SparseActivationError(f"{label} has invalid sparse shard schema")
    if payload.get("start_row") != start_row or payload.get("end_row") != end_row:
        raise SparseActivationError(f"{label} does not mirror its source row range")
    unit_ids = payload.get("unit_ids")
    if not isinstance(unit_ids, tuple) or unit_ids != expected_ids:
        raise SparseActivationError(f"{label} unit IDs violate exact source order")
    indptr = payload.get("indptr")
    indices = payload.get("feature_indices")
    values = payload.get("activation_values")
    rows = end_row - start_row
    if (
        not isinstance(indptr, torch.Tensor)
        or indptr.dtype != torch.int64
        or indptr.device.type != "cpu"
        or tuple(indptr.shape) != (rows + 1,)
    ):
        raise SparseActivationError(f"{label} indptr must be CPU int64 [N+1]")
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.int32
        or indices.device.type != "cpu"
        or indices.ndim != 1
    ):
        raise SparseActivationError(f"{label} feature_indices must be CPU int32")
    if (
        not isinstance(values, torch.Tensor)
        or values.dtype != torch.float32
        or values.device.type != "cpu"
        or values.ndim != 1
        or len(values) != len(indices)
    ):
        raise SparseActivationError(f"{label} activation_values must be CPU float32")
    if not bool(torch.isfinite(values).all().item()) or bool(torch.any(values <= 0).item()):
        raise SparseActivationError(f"{label} values must be finite and strictly positive")
    if (
        int(indptr[0].item()) != 0
        or int(indptr[-1].item()) != len(indices)
        or bool(torch.any(indptr[1:] < indptr[:-1]).item())
    ):
        raise SparseActivationError(f"{label} indptr is invalid")
    counts = indptr[1:] - indptr[:-1]
    if bool(torch.any(counts > k).item()):
        raise SparseActivationError(f"{label} contains a row with more than TopK entries")
    if len(indices) and (
        int(indices.min().item()) < 0 or int(indices.max().item()) >= width
    ):
        raise SparseActivationError(f"{label} feature index is out of range")
    for row in range(rows):
        left = int(indptr[row].item())
        right = int(indptr[row + 1].item())
        row_indices = indices[left:right].to(torch.int64)
        if len(row_indices) > 1 and bool(torch.any(row_indices[1:] <= row_indices[:-1]).item()):
            raise SparseActivationError(
                f"{label} feature IDs must be unique and ascending within each row"
            )
    return indptr, indices, values


def _forward_csr(
    model: TopKSAE,
    embeddings: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indptr_parts = [torch.zeros(1, dtype=torch.int64)]
    index_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    nnz = 0
    dead_before = model.steps_since_active.detach().cpu().clone()
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            batch = embeddings[start : start + batch_size].to(device)
            activations = model(batch, update_dead_stats=False).activations
            indptr, indices, values = dense_to_csr(activations)
            indptr_parts.append(indptr[1:] + nnz)
            index_parts.append(indices)
            value_parts.append(values)
            nnz += len(values)
    if not torch.equal(dead_before, model.steps_since_active.detach().cpu()):
        raise SparseActivationError("Frozen inference mutated dead-feature statistics")
    return (
        torch.cat(indptr_parts),
        torch.cat(index_parts) if index_parts else torch.empty(0, dtype=torch.int32),
        torch.cat(value_parts) if value_parts else torch.empty(0, dtype=torch.float32),
    )


def _metadata(
    path: Path,
    source: ShardMetadata,
    expected_ids: tuple[str, ...],
    nnz: int,
) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "end_row": source.end_row,
        "nnz": nnz,
        "path": path.name,
        "rows": source.end_row - source.start_row,
        "sha256": sha256_file(path),
        "source_path": str(source.path.resolve()),
        "source_sha256": source.sha256,
        "start_row": source.start_row,
        "unit_ids_sha256": _unit_ids_sha256(expected_ids),
    }


def _manifest_template(
    index: SaeDataIndex,
    config: SaePilotConfig,
    inference_sha256: str,
    device_identity: dict[str, object],
    batch_size: int,
    atlas_source_binding: dict[str, str] | None,
    test_fixture_unbound: bool,
) -> dict[str, object]:
    return {
        "atlas_source_binding": atlas_source_binding,
        "batch_size": batch_size,
        "complete": False,
        "device": device_identity["logical_device"],
        "device_identity": device_identity,
        "inference_sha256": inference_sha256,
        "input_dim": config.input_dim,
        "k": config.k,
        "model_config_fingerprint": config.fingerprint(),
        "row_count": len(index.units),
        "schema_version": SCHEMA_VERSION,
        "shards": [],
        "source_provenance": {
            "activation_manifest_sha256": index.activation_manifest_sha256,
            "assignment_sha256": index.assignment_sha256,
            "text_manifest_sha256": index.text_manifest_sha256,
        },
        "test_fixture_unbound": test_fixture_unbound,
        "width": config.width,
    }


def _load_manifest(output_dir: Path) -> dict[str, object]:
    try:
        return strict_json_load(output_dir / MANIFEST_FILE)
    except (AtlasIOError, OSError, ValueError) as error:
        raise SparseActivationError(f"Invalid sparse activation manifest: {error}") from error


def _validate_manifest_binding(
    manifest: dict[str, object],
    index: SaeDataIndex,
    config: SaePilotConfig,
    inference_sha256: str,
    device_identity: dict[str, object],
    atlas_source_binding: dict[str, str] | None,
    test_fixture_unbound: bool,
) -> list[dict[str, object]]:
    required = set(
        _manifest_template(
            index,
            config,
            inference_sha256,
            device_identity,
            1,
            atlas_source_binding,
            test_fixture_unbound,
        )
    )
    if set(manifest) != required or manifest.get("schema_version") != SCHEMA_VERSION:
        raise SparseActivationError("Sparse activation manifest has invalid schema")
    if manifest.get("inference_sha256") != inference_sha256:
        raise SparseActivationError("Sparse manifest inference checksum mismatch")
    expected = _manifest_template(
        index,
        config,
        inference_sha256,
        device_identity,
        int(manifest.get("batch_size")) if type(manifest.get("batch_size")) is int else 0,
        atlas_source_binding,
        test_fixture_unbound,
    )
    for key in (
        "input_dim", "k", "model_config_fingerprint", "row_count",
        "source_provenance", "width", "atlas_source_binding",
        "test_fixture_unbound", "device", "device_identity",
    ):
        if manifest.get(key) != expected[key]:
            raise SparseActivationError(f"Sparse manifest binding mismatch for {key}")
    _exact_int(manifest.get("batch_size"), "manifest batch_size", minimum=1)
    if type(manifest.get("test_fixture_unbound")) is not bool:
        raise SparseActivationError("Sparse manifest fixture-binding flag is invalid")
    if type(manifest.get("complete")) is not bool:
        raise SparseActivationError("Sparse manifest complete flag is invalid")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or any(not isinstance(item, dict) for item in shards):
        raise SparseActivationError("Sparse manifest shards are invalid")
    if len(shards) > len(index.shards):
        raise SparseActivationError("Sparse manifest registers too many shards")
    return shards  # type: ignore[return-value]


def _registered_output_path(output_dir: Path, value: object, shard_number: int) -> Path:
    expected_name = f"shard_{shard_number:05d}.pt"
    if type(value) is not str or value != expected_name:
        raise SparseActivationError("Sparse manifest shard path is invalid")
    return output_dir / expected_name


def _validate_registered(
    manifest: dict[str, object],
    output_dir: Path,
    index: SaeDataIndex,
    config: SaePilotConfig,
) -> list[dict[str, object]]:
    shards = manifest["shards"]
    assert isinstance(shards, list)
    for shard_number, raw_metadata in enumerate(shards):
        if not isinstance(raw_metadata, dict):
            raise SparseActivationError("Sparse manifest shard metadata is invalid")
        source = index.shards[shard_number]
        expected_ids = tuple(
            unit.unit_id for unit in index.units[source.start_row : source.end_row]
        )
        try:
            current_source_sha256 = sha256_file(source.path)
        except (AtlasIOError, OSError, ValueError) as error:
            raise SparseActivationError(
                f"Cannot hash source shard during resume: {source.path}"
            ) from error
        if current_source_sha256 != source.sha256:
            raise SparseActivationError(
                f"source shard checksum mismatch during resume: {source.path}"
            )
        path = _registered_output_path(output_dir, raw_metadata.get("path"), shard_number)
        if raw_metadata.get("start_row") != source.start_row or raw_metadata.get("end_row") != source.end_row:
            raise SparseActivationError("Sparse manifest does not mirror source row ranges")
        if raw_metadata.get("rows") != source.end_row - source.start_row:
            raise SparseActivationError("Sparse manifest row count is invalid")
        if raw_metadata.get("source_path") != str(source.path.resolve()) or raw_metadata.get("source_sha256") != source.sha256:
            raise SparseActivationError("Sparse manifest source shard binding is invalid")
        if raw_metadata.get("unit_ids_sha256") != _unit_ids_sha256(expected_ids):
            raise SparseActivationError("Sparse manifest ordered unit-ID binding is invalid")
        try:
            observed_sha = sha256_file(path)
        except (AtlasIOError, OSError, ValueError) as error:
            raise SparseActivationError(f"Cannot read registered sparse shard: {path}") from error
        if raw_metadata.get("sha256") != observed_sha:
            raise SparseActivationError(f"Registered sparse shard checksum mismatch: {path}")
        if raw_metadata.get("bytes") != path.stat().st_size:
            raise SparseActivationError(f"Registered sparse shard byte size mismatch: {path}")
        payload = _torch_payload(path, "sparse activation shard")
        _, indices, _ = _validate_sparse_shard(
            payload,
            start_row=source.start_row,
            end_row=source.end_row,
            expected_ids=expected_ids,
            width=config.width,
            k=config.k,
            label=str(path),
        )
        if raw_metadata.get("nnz") != len(indices):
            raise SparseActivationError(f"Sparse manifest nnz mismatch: {path}")
    if manifest.get("complete") and len(shards) != len(index.shards):
        raise SparseActivationError("Complete sparse manifest does not cover every source shard")
    return shards  # type: ignore[return-value]


def _compare_all_rows(
    model: TopKSAE,
    source_embeddings: torch.Tensor,
    sparse: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    label: str,
) -> None:
    expected = _forward_csr(
        model, source_embeddings, device=device, batch_size=batch_size
    )
    observed_indptr, observed_indices, observed_values = sparse
    expected_indptr, expected_indices, expected_values = expected
    if not torch.equal(observed_indptr, expected_indptr) or not torch.equal(
        observed_indices, expected_indices
    ):
        raise SparseActivationError(f"{label} direct-forward support mismatch")
    if len(expected_values):
        error = float((observed_values - expected_values).abs().max().item())
        if error > SPOT_CHECK_ATOL:
            raise SparseActivationError(
                f"{label} direct-forward activation error {error} exceeds {SPOT_CHECK_ATOL}"
            )


def _remove_registered_outputs(
    manifest: dict[str, object], output_dir: Path, index: SaeDataIndex
) -> None:
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise SparseActivationError("Cannot overwrite an invalid sparse manifest")
    paths = []
    for number, metadata in enumerate(shards):
        if not isinstance(metadata, dict) or number >= len(index.shards):
            raise SparseActivationError("Cannot overwrite invalid registered outputs")
        paths.append(_registered_output_path(output_dir, metadata.get("path"), number))
    for path in paths:
        if path.is_symlink():
            raise SparseActivationError("Refusing to remove a symlinked registered output")
        path.unlink(missing_ok=True)
    (output_dir / MANIFEST_FILE).unlink(missing_ok=True)


def extract_sparse_feature_activations(
    source_index: SaeDataIndex,
    inference_path: Path,
    output_dir: Path,
    *,
    device: str,
    batch_size: int = 512,
    overwrite: bool = False,
    atlas_paths: AtlasPaths | None = None,
    source_binding: AtlasSourceBinding | None = None,
    allow_test_fixture_unbound: bool = False,
) -> dict[str, object]:
    """Mirror source activation shards through a frozen SAE into CSR shards."""

    _validate_index_layout(source_index)
    _exact_int(batch_size, "batch_size", minimum=1)
    if type(device) is not str or not device:
        raise TypeError("device must be a nonempty string")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    output_dir = Path(output_dir)
    binding_payload, test_fixture_unbound = _validate_atlas_source_context(
        source_index,
        inference_path,
        output_dir,
        feature_stats_path=None,
        atlas_paths=atlas_paths,
        source_binding=source_binding,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
    )
    resolved_device, device_identity = _resolve_device_identity(device)
    device_name = str(resolved_device)
    model, config, provenance, inference_sha256 = _load_inference(
        inference_path, resolved_device
    )
    _check_provenance(source_index, provenance)
    manifest_path = output_dir / MANIFEST_FILE
    manifest: dict[str, object]
    if manifest_path.exists():
        manifest = _load_manifest(output_dir)
        _validate_manifest_binding(
            manifest,
            source_index,
            config,
            inference_sha256,
            device_identity,
            binding_payload,
            test_fixture_unbound,
        )
        if overwrite:
            _remove_registered_outputs(manifest, output_dir, source_index)
        else:
            if manifest.get("device") != device_name or manifest.get("batch_size") != batch_size:
                raise SparseActivationError("Resume device or batch_size does not match manifest")
            _validate_registered(manifest, output_dir, source_index, config)
            if manifest.get("complete"):
                return manifest
    elif overwrite:
        manifest = {}

    if not manifest_path.exists():
        manifest = _manifest_template(
            source_index,
            config,
            inference_sha256,
            device_identity,
            batch_size,
            binding_payload,
            test_fixture_unbound,
        )
        write_json_atomic(manifest_path, manifest)

    shards = manifest["shards"]
    if not isinstance(shards, list):
        raise SparseActivationError("Sparse manifest shards are invalid")
    for shard_number in range(len(shards), len(source_index.shards)):
        source = source_index.shards[shard_number]
        embeddings, expected_ids = _load_source_shard(
            source_index, source, config.input_dim
        )
        shard_path = output_dir / f"shard_{shard_number:05d}.pt"
        if shard_path.exists():
            payload = _torch_payload(shard_path, "unregistered sparse shard")
            sparse = _validate_sparse_shard(
                payload,
                start_row=source.start_row,
                end_row=source.end_row,
                expected_ids=expected_ids,
                width=config.width,
                k=config.k,
                label=str(shard_path),
            )
            _compare_all_rows(
                model,
                embeddings,
                sparse,
                device=resolved_device,
                batch_size=batch_size,
                label=str(shard_path),
            )
            _, indices, _ = sparse
        else:
            indptr, indices, values = _forward_csr(
                model,
                embeddings,
                device=resolved_device,
                batch_size=batch_size,
            )
            payload = {
                "schema_version": SCHEMA_VERSION,
                "start_row": source.start_row,
                "end_row": source.end_row,
                "unit_ids": expected_ids,
                "indptr": indptr,
                "feature_indices": indices,
                "activation_values": values,
            }
            write_torch_atomic(shard_path, payload)
        metadata = _metadata(shard_path, source, expected_ids, len(indices))
        shards.append(metadata)
        write_json_atomic(manifest_path, manifest)
    _validate_registered(manifest, output_dir, source_index, config)
    manifest["complete"] = True
    write_json_atomic(manifest_path, manifest)
    return manifest


def _sample_spot_check_rows(
    row_count: int, seed: int, shard_number: int
) -> tuple[int, ...]:
    _exact_int(row_count, "row_count", minimum=0)
    _exact_int(seed, "seed", minimum=0)
    _exact_int(shard_number, "shard_number", minimum=0)
    generator = random.Random(seed + shard_number)
    return tuple(
        sorted(generator.sample(range(row_count), min(SPOT_CHECK_ROWS, row_count)))
    )


def _spot_check_shard(
    model: TopKSAE,
    embeddings: torch.Tensor,
    sparse: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    seed: int,
    shard_number: int,
) -> None:
    indptr, indices, values = sparse
    rows = len(embeddings)
    selected = _sample_spot_check_rows(rows, seed, shard_number)
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    dead_before = model.steps_since_active.detach().cpu().clone()
    with torch.inference_mode():
        direct = model(
            embeddings.index_select(0, selected_tensor).to(device),
            update_dead_stats=False,
        ).activations.detach().cpu()
    if not torch.equal(dead_before, model.steps_since_active.detach().cpu()):
        raise SparseActivationError("Direct-forward validation mutated dead statistics")
    for direct_row, source_row in enumerate(selected):
        left = int(indptr[source_row].item())
        right = int(indptr[source_row + 1].item())
        expected_support = torch.nonzero(direct[direct_row] > 0, as_tuple=False).flatten().to(torch.int32)
        observed_support = indices[left:right]
        if not torch.equal(observed_support, expected_support):
            raise SparseActivationError("direct-forward support spot check failed")
        expected_values = direct[direct_row].index_select(
            0, expected_support.to(torch.long)
        )
        if len(expected_values):
            error = float((values[left:right] - expected_values).abs().max().item())
            if error > SPOT_CHECK_ATOL:
                raise SparseActivationError(
                    f"direct-forward activation value error {error} exceeds {SPOT_CHECK_ATOL}"
                )


def _load_feature_stats(
    feature_stats_path: Path,
    config: SaePilotConfig,
    provenance: Mapping[str, str],
) -> dict[str, object]:
    try:
        payload = _torch_payload(feature_stats_path, "feature statistics")
        checkpointing._validate_artifact_common(
            payload,
            label="feature statistics",
            config=config,
            provenance=provenance,
            exact_keys=checkpointing.FEATURE_STATS_ARTIFACT_KEYS,
        )
        if payload.get("width") != config.width:
            raise checkpointing.ArtifactError("feature statistics width mismatch")
        for split in ("train", "validation"):
            checkpointing._validate_feature_statistics(
                payload.get(split),
                label=f"feature statistics {split}",
                expected_width=config.width,
            )
        return payload
    except SparseActivationError:
        raise
    except (checkpointing.ArtifactError, TypeError, ValueError) as error:
        raise SparseActivationError(f"Invalid feature statistics artifact: {error}") from error


def _empty_aggregates(width: int) -> dict[str, torch.Tensor | int]:
    return {
        "count": 0,
        "activation_count": torch.zeros(width, dtype=torch.int64),
        "activation_sum": torch.zeros(width, dtype=torch.float64),
        "activation_max": torch.zeros(width, dtype=torch.float32),
        "aspect_counts": torch.zeros((width, 24), dtype=torch.int64),
    }


def _accumulate_sparse(
    aggregate: dict[str, torch.Tensor | int],
    *,
    sample_count: int,
    row_indices: torch.Tensor,
    aspect_indices: torch.Tensor,
    feature_indices: torch.Tensor,
    values: torch.Tensor,
    width: int,
) -> None:
    aggregate["count"] = int(aggregate["count"]) + sample_count
    if not len(feature_indices):
        return
    firing = values > 1e-3
    if not bool(firing.any().item()):
        return
    fired_features = feature_indices[firing].to(torch.long)
    fired_values = values[firing]
    activation_count = aggregate["activation_count"]
    activation_sum = aggregate["activation_sum"]
    activation_max = aggregate["activation_max"]
    aspect_counts = aggregate["aspect_counts"]
    assert isinstance(activation_count, torch.Tensor)
    assert isinstance(activation_sum, torch.Tensor)
    assert isinstance(activation_max, torch.Tensor)
    assert isinstance(aspect_counts, torch.Tensor)
    activation_count.add_(torch.bincount(fired_features, minlength=width))
    activation_sum.scatter_add_(0, fired_features, fired_values.to(torch.float64))
    activation_max.scatter_reduce_(0, fired_features, fired_values, reduce="amax", include_self=True)
    fired_rows = row_indices[firing]
    fired_aspects = aspect_indices.index_select(0, fired_rows)
    flat = fired_features * 24 + fired_aspects
    aspect_counts.view(-1).add_(torch.bincount(flat, minlength=width * 24))


def _reconcile_split(
    split: str,
    aggregate: dict[str, torch.Tensor | int],
    reference: object,
    width: int,
) -> None:
    if not isinstance(reference, dict):
        raise SparseActivationError(f"Reference {split} feature statistics are invalid")
    if aggregate["count"] != reference.get("count"):
        raise SparseActivationError(f"{split} feature statistics row count mismatch")
    features = reference.get("features")
    if not isinstance(features, list) or len(features) != width:
        raise SparseActivationError(f"Reference {split} features are invalid")
    counts = aggregate["activation_count"]
    sums = aggregate["activation_sum"]
    maxima = aggregate["activation_max"]
    aspects = aggregate["aspect_counts"]
    assert isinstance(counts, torch.Tensor)
    assert isinstance(sums, torch.Tensor)
    assert isinstance(maxima, torch.Tensor)
    assert isinstance(aspects, torch.Tensor)
    expected_counts = torch.tensor(
        [feature["activation_count"] for feature in features], dtype=torch.int64
    )
    if not torch.equal(counts, expected_counts):
        raise SparseActivationError(f"{split} feature statistics firing counts mismatch")
    expected_aspects = torch.tensor(
        [feature["aspect_activation_counts"] for feature in features], dtype=torch.int64
    )
    if not torch.equal(aspects, expected_aspects):
        raise SparseActivationError(f"{split} feature statistics aspect counts mismatch")
    expected_sums = torch.tensor(
        [feature["mean_on_firing"] * feature["activation_count"] for feature in features],
        dtype=torch.float64,
    )
    if not torch.allclose(sums, expected_sums, rtol=SUM_RTOL, atol=SUM_ATOL):
        raise SparseActivationError(f"{split} feature statistics activation sum mismatch")
    expected_maxima = torch.tensor(
        [feature["max_activation"] for feature in features], dtype=torch.float32
    )
    if not torch.allclose(maxima, expected_maxima, rtol=0.0, atol=SPOT_CHECK_ATOL):
        raise SparseActivationError(f"{split} feature statistics maximum mismatch")


def validate_sparse_feature_activations(
    source_index: SaeDataIndex,
    inference_path: Path,
    feature_stats_path: Path,
    output_dir: Path,
    *,
    atlas_paths: AtlasPaths | None = None,
    source_binding: AtlasSourceBinding | None = None,
    allow_test_fixture_unbound: bool = False,
) -> dict[str, object]:
    """Validate source mirroring, direct forwards, and feature-stat reconciliation."""

    _validate_index_layout(source_index)
    output_dir = Path(output_dir)
    binding_payload, test_fixture_unbound = _validate_atlas_source_context(
        source_index,
        inference_path,
        output_dir,
        feature_stats_path=feature_stats_path,
        atlas_paths=atlas_paths,
        source_binding=source_binding,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
    )
    manifest = _load_manifest(output_dir)
    device_identity = manifest.get("device_identity")
    device = _validate_device_identity(device_identity)
    model, config, provenance, inference_sha256 = _load_inference(
        inference_path, device
    )
    _check_provenance(source_index, provenance)
    _validate_manifest_binding(
        manifest,
        source_index,
        config,
        inference_sha256,
        device_identity,  # type: ignore[arg-type]
        binding_payload,
        test_fixture_unbound,
    )
    if not manifest.get("complete"):
        raise SparseActivationError("Sparse activation cache is incomplete")
    _validate_registered(manifest, output_dir, source_index, config)
    reference = _load_feature_stats(feature_stats_path, config, provenance)
    aggregates = {
        "train": _empty_aggregates(config.width),
        "validation": _empty_aggregates(config.width),
    }
    units = source_index.units
    for shard_number, source in enumerate(source_index.shards):
        embeddings, expected_ids = _load_source_shard(
            source_index, source, config.input_dim
        )
        path = output_dir / f"shard_{shard_number:05d}.pt"
        sparse = _validate_sparse_shard(
            _torch_payload(path, "sparse activation shard"),
            start_row=source.start_row,
            end_row=source.end_row,
            expected_ids=expected_ids,
            width=config.width,
            k=config.k,
            label=str(path),
        )
        _spot_check_shard(
            model,
            embeddings,
            sparse,
            device=device,
            seed=config.seed,
            shard_number=shard_number,
        )
        indptr, feature_indices, values = sparse
        local_rows = torch.repeat_interleave(
            torch.arange(source.end_row - source.start_row, dtype=torch.long),
            indptr[1:] - indptr[:-1],
        )
        local_aspects = torch.tensor(
            [unit.aspect_index for unit in units[source.start_row : source.end_row]],
            dtype=torch.long,
        )
        for split in ("train", "validation"):
            selected_rows = torch.tensor(
                [
                    local
                    for local, unit in enumerate(units[source.start_row : source.end_row])
                    if unit.split == split
                ],
                dtype=torch.long,
            )
            if not len(selected_rows):
                continue
            row_mask = torch.zeros(source.end_row - source.start_row, dtype=torch.bool)
            row_mask[selected_rows] = True
            entry_mask = row_mask.index_select(0, local_rows)
            split_local_rows = local_rows[entry_mask]
            remap = torch.full(
                (source.end_row - source.start_row,), -1, dtype=torch.long
            )
            remap[selected_rows] = torch.arange(len(selected_rows))
            _accumulate_sparse(
                aggregates[split],
                sample_count=len(selected_rows),
                row_indices=remap.index_select(0, split_local_rows),
                aspect_indices=local_aspects.index_select(0, selected_rows),
                feature_indices=feature_indices[entry_mask],
                values=values[entry_mask],
                width=config.width,
            )
    _reconcile_split("train", aggregates["train"], reference.get("train"), config.width)
    _reconcile_split(
        "validation", aggregates["validation"], reference.get("validation"), config.width
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--units-path", type=Path, default=DEFAULT_CORPUS_DIR / "sae_units.jsonl")
    parser.add_argument(
        "--text-manifest-path",
        type=Path,
        default=DEFAULT_CORPUS_DIR / "text_corpus_manifest.json",
    )
    parser.add_argument("--activation-dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument(
        "--filtered-uniprot-tsv",
        type=Path,
        default=AtlasPaths().filtered_uniprot_tsv,
    )
    parser.add_argument("--inference-path", type=Path, default=DEFAULT_RUN_DIR / "inference.pt")
    parser.add_argument(
        "--feature-stats-path", type=Path, default=DEFAULT_RUN_DIR / "feature_stats.pt"
    )
    parser.add_argument(
        "--atlas-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    atlas_paths = AtlasPaths(
        run_dir=args.run_dir,
        units_path=args.units_path,
        activation_dir=args.activation_dir,
        text_manifest_path=args.text_manifest_path,
        filtered_uniprot_tsv=args.filtered_uniprot_tsv,
        output_dir=args.atlas_output_dir,
    )
    source_binding = build_source_binding(atlas_paths)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else atlas_paths.output_dir / "sparse_activations"
    )
    source_index = build_data_index(
        args.units_path, args.text_manifest_path, args.activation_dir
    )
    if args.validate_only:
        manifest = validate_sparse_feature_activations(
            source_index,
            args.inference_path,
            args.feature_stats_path,
            output_dir,
            atlas_paths=atlas_paths,
            source_binding=source_binding,
        )
    else:
        manifest = extract_sparse_feature_activations(
            source_index,
            args.inference_path,
            output_dir,
            device=args.device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            atlas_paths=atlas_paths,
            source_binding=source_binding,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
