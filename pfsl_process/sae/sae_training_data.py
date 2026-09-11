"""Validated source indexing and deterministic splits for SAE training."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Literal

import torch

from .extract_protrek_activations import (
    ActivationCacheError,
    MANIFEST_FILE as ACTIVATION_MANIFEST_FILE,
    validate_activation_cache,
)
from .text_templates import ASPECT_NAMES


SplitName = Literal["train", "validation"]
SPLIT_ALGORITHM = "sha256(accession) modulo 10; validation when remainder is 0"


class SaeDataError(ValueError):
    """Raised when SAE training inputs violate their recorded schemas."""


@dataclass(frozen=True)
class SaeUnitMetadata:
    row_index: int
    unit_id: str
    accession: str
    aspect_index: int
    split: SplitName


@dataclass(frozen=True)
class ShardMetadata:
    path: Path
    start_row: int
    end_row: int
    sha256: str


@dataclass(frozen=True)
class SaeDataIndex:
    units: tuple[SaeUnitMetadata, ...]
    shards: tuple[ShardMetadata, ...]
    train_accessions: frozenset[str]
    validation_accessions: frozenset[str]
    train_unit_count: int
    validation_unit_count: int
    source_train_unit_count: int
    source_validation_unit_count: int
    train_aspect_counts: tuple[int, ...]
    validation_aspect_counts: tuple[int, ...]
    assignment_sha256: str
    source_assignment_sha256: str
    text_manifest_sha256: str
    activation_manifest_sha256: str


@dataclass(frozen=True)
class ActivationBatch:
    embeddings: torch.Tensor
    unit_ids: tuple[str, ...]
    aspect_indices: torch.Tensor
    loss_weights: torch.Tensor | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SaeDataError(f"Invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise SaeDataError(f"{label} must be a JSON object: {path}")
    return payload


def split_for_accession(accession: str) -> SplitName:
    value = int(hashlib.sha256(accession.encode("utf-8")).hexdigest(), 16)
    return "validation" if value % 10 == 0 else "train"


def validate_text_units_manifest(
    units_path: Path,
    text_manifest_path: Path,
) -> dict[str, object]:
    units_path = Path(units_path)
    text_manifest_path = Path(text_manifest_path)
    if not units_path.is_file():
        raise SaeDataError(f"SAE units file does not exist: {units_path}")
    manifest = _load_json_object(text_manifest_path, "text corpus manifest")
    if manifest.get("aspect_names") != list(ASPECT_NAMES):
        raise SaeDataError("Text manifest aspect names do not match the exact schema")
    files = manifest.get("files")
    units_metadata = files.get("sae_units") if isinstance(files, dict) else None
    if not isinstance(units_metadata, dict):
        raise SaeDataError("Text manifest SAE units metadata is invalid")
    if units_metadata.get("path") != str(units_path.resolve()):
        raise SaeDataError("SAE units path does not match text manifest")
    if units_metadata.get("sha256") != _sha256(units_path):
        raise SaeDataError("SAE units checksum does not match text manifest")
    if units_metadata.get("bytes") != units_path.stat().st_size:
        raise SaeDataError("SAE units byte size does not match text manifest")
    counts = manifest.get("counts")
    unit_count = counts.get("units") if isinstance(counts, dict) else None
    if not isinstance(unit_count, int) or isinstance(unit_count, bool) or unit_count <= 0:
        raise SaeDataError("Text manifest unit count is invalid")
    return manifest


def _assignment_sha256(units: tuple[SaeUnitMetadata, ...]) -> str:
    digest = hashlib.sha256()
    for unit in units:
        digest.update(f"{unit.unit_id}\t{unit.split}\n".encode("utf-8"))
    return digest.hexdigest()


def _aspect_counts(
    units: tuple[SaeUnitMetadata, ...], split: SplitName
) -> tuple[int, ...]:
    counts = [0] * len(ASPECT_NAMES)
    for unit in units:
        if unit.split == split:
            counts[unit.aspect_index] += 1
    return tuple(counts)


def _validate_effective_splits(
    train_count: int,
    validation_count: int,
    train_aspect_counts: tuple[int, ...],
) -> None:
    if train_count == 0 or validation_count == 0:
        raise SaeDataError("Training and validation splits must both be nonempty")
    if any(count <= 0 for count in train_aspect_counts):
        raise SaeDataError("Every aspect must have a training unit")


def build_data_index(
    units_path: Path,
    text_manifest_path: Path,
    activation_dir: Path,
) -> SaeDataIndex:
    units_path = Path(units_path)
    text_manifest_path = Path(text_manifest_path)
    activation_dir = Path(activation_dir)
    text_manifest = validate_text_units_manifest(units_path, text_manifest_path)
    try:
        activation_manifest = validate_activation_cache(units_path, activation_dir)
    except (ActivationCacheError, FileNotFoundError) as error:
        raise SaeDataError(str(error)) from error

    units: list[SaeUnitMetadata] = []
    seen_unit_ids: set[str] = set()
    try:
        with units_path.open("r", encoding="utf-8") as handle:
            for row_index, raw_line in enumerate(handle):
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise SaeDataError(
                        f"Invalid SAE unit JSON at row {row_index}"
                    ) from error
                if not isinstance(payload, dict):
                    raise SaeDataError(f"SAE unit row {row_index} is not an object")
                unit_id = payload.get("unit_id")
                accession = payload.get("accession")
                aspect_index = payload.get("aspect_index")
                if not isinstance(unit_id, str) or not unit_id:
                    raise SaeDataError(f"SAE unit row {row_index} has invalid unit_id")
                if unit_id in seen_unit_ids:
                    raise SaeDataError(f"Duplicate SAE unit ID: {unit_id!r}")
                if not isinstance(accession, str) or not accession:
                    raise SaeDataError(
                        f"SAE unit row {row_index} has invalid accession"
                    )
                if (
                    not isinstance(aspect_index, int)
                    or isinstance(aspect_index, bool)
                    or not 0 <= aspect_index < len(ASPECT_NAMES)
                ):
                    raise SaeDataError(
                        f"SAE unit row {row_index} has invalid aspect_index"
                    )
                units.append(
                    SaeUnitMetadata(
                        row_index=row_index,
                        unit_id=unit_id,
                        accession=accession,
                        aspect_index=aspect_index,
                        split=split_for_accession(accession),
                    )
                )
                seen_unit_ids.add(unit_id)
    except OSError as error:
        raise SaeDataError(f"Cannot read SAE units: {units_path}") from error

    counts = text_manifest["counts"]
    expected_count = counts["units"]  # type: ignore[index]
    activation_input = activation_manifest.get("input")
    activation_count = (
        activation_input.get("rows") if isinstance(activation_input, dict) else None
    )
    if len(units) != expected_count or len(units) != activation_count:
        raise SaeDataError("SAE unit row count does not match source manifests")

    unit_tuple = tuple(units)
    train_count = sum(unit.split == "train" for unit in unit_tuple)
    validation_count = len(unit_tuple) - train_count
    train_aspect_counts = _aspect_counts(unit_tuple, "train")
    validation_aspect_counts = _aspect_counts(unit_tuple, "validation")
    _validate_effective_splits(train_count, validation_count, train_aspect_counts)

    raw_shards = activation_manifest.get("shards")
    if not isinstance(raw_shards, list):
        raise SaeDataError("Activation manifest shards are invalid")
    shards = tuple(
        ShardMetadata(
            path=(activation_dir / str(metadata["path"])).resolve(),
            start_row=int(metadata["start_row"]),
            end_row=int(metadata["end_row"]),
            sha256=str(metadata["sha256"]),
        )
        for metadata in raw_shards
        if isinstance(metadata, dict)
    )
    if len(shards) != len(raw_shards):
        raise SaeDataError("Activation manifest shard metadata is invalid")

    assignment_sha256 = _assignment_sha256(unit_tuple)
    train_accessions = frozenset(
        unit.accession for unit in unit_tuple if unit.split == "train"
    )
    validation_accessions = frozenset(
        unit.accession for unit in unit_tuple if unit.split == "validation"
    )
    if train_accessions & validation_accessions:
        raise SaeDataError("Accession split is not disjoint")
    return SaeDataIndex(
        units=unit_tuple,
        shards=shards,
        train_accessions=train_accessions,
        validation_accessions=validation_accessions,
        train_unit_count=train_count,
        validation_unit_count=validation_count,
        source_train_unit_count=train_count,
        source_validation_unit_count=validation_count,
        train_aspect_counts=train_aspect_counts,
        validation_aspect_counts=validation_aspect_counts,
        assignment_sha256=assignment_sha256,
        source_assignment_sha256=assignment_sha256,
        text_manifest_sha256=_sha256(text_manifest_path),
        activation_manifest_sha256=_sha256(
            activation_dir / ACTIVATION_MANIFEST_FILE
        ),
    )


def limit_data_index(
    index: SaeDataIndex,
    train_limit: int | None,
    validation_limit: int | None,
) -> SaeDataIndex:
    if train_limit is None and validation_limit is None:
        return index
    for name, value in (
        ("train_limit", train_limit),
        ("validation_limit", validation_limit),
    ):
        if value is not None and value <= 0:
            raise SaeDataError(f"{name} must be positive when provided")

    kept: list[SaeUnitMetadata] = []
    used = {"train": 0, "validation": 0}
    limits = {"train": train_limit, "validation": validation_limit}
    for unit in index.units:
        limit = limits[unit.split]
        if limit is None or used[unit.split] < limit:
            kept.append(unit)
            used[unit.split] += 1
    units = tuple(kept)
    train_count = used["train"]
    validation_count = used["validation"]
    train_aspect_counts = _aspect_counts(units, "train")
    validation_aspect_counts = _aspect_counts(units, "validation")
    _validate_effective_splits(train_count, validation_count, train_aspect_counts)
    return replace(
        index,
        units=units,
        train_accessions=frozenset(
            unit.accession for unit in units if unit.split == "train"
        ),
        validation_accessions=frozenset(
            unit.accession for unit in units if unit.split == "validation"
        ),
        train_unit_count=train_count,
        validation_unit_count=validation_count,
        train_aspect_counts=train_aspect_counts,
        validation_aspect_counts=validation_aspect_counts,
        assignment_sha256=_assignment_sha256(units),
    )


def compute_aspect_weights(index: SaeDataIndex, alpha: float) -> torch.Tensor:
    counts = torch.tensor(index.train_aspect_counts, dtype=torch.float64)
    if torch.any(counts <= 0):
        raise SaeDataError("Every aspect must have a training unit")
    raw = counts.pow(alpha - 1.0)
    train_total = counts.sum()
    normalization = (raw * counts).sum() / train_total
    weights = raw / normalization
    weights = weights.to(torch.float32)
    weighted_mean = (weights.double() * counts).sum() / train_total
    if abs(float(weighted_mean) - 1.0) > torch.finfo(torch.float32).eps:
        raise AssertionError("Aspect weights do not have unit weighted mean")
    return weights


def _load_indexed_shard(
    index: SaeDataIndex,
    shard: ShardMetadata,
    units_by_row: dict[int, SaeUnitMetadata],
) -> torch.Tensor:
    try:
        observed_sha256 = _sha256(shard.path)
    except OSError as error:
        raise SaeDataError(f"Cannot read activation shard: {shard.path}") from error
    if observed_sha256 != shard.sha256:
        raise SaeDataError(
            f"Activation shard checksum does not match data index: {shard.path}"
        )
    try:
        try:
            payload = torch.load(
                shard.path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            payload = torch.load(shard.path, map_location="cpu")
    except Exception as error:
        raise SaeDataError(f"Cannot load activation shard: {shard.path}") from error
    if not isinstance(payload, dict):
        raise SaeDataError(f"Activation shard is not a mapping: {shard.path}")
    expected_rows = shard.end_row - shard.start_row
    stored_unit_ids = payload.get("unit_ids")
    if (
        not isinstance(stored_unit_ids, list)
        or len(stored_unit_ids) != expected_rows
        or any(not isinstance(unit_id, str) or not unit_id for unit_id in stored_unit_ids)
    ):
        raise SaeDataError(f"Invalid unit IDs in activation shard: {shard.path}")
    for row_index in range(shard.start_row, shard.end_row):
        indexed_unit = units_by_row.get(row_index)
        if (
            indexed_unit is not None
            and stored_unit_ids[row_index - shard.start_row] != indexed_unit.unit_id
        ):
            raise SaeDataError(
                f"Activation shard unit IDs do not match data index: {shard.path}"
            )
    embeddings = payload.get("embeddings")
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.dtype != torch.float32
        or embeddings.ndim != 2
        or tuple(embeddings.shape) != (expected_rows, 1024)
        or not torch.isfinite(embeddings).all()
    ):
        raise SaeDataError(f"Invalid embeddings in activation shard: {shard.path}")
    return embeddings


def _make_activation_batch(
    embeddings: torch.Tensor,
    unit_ids: tuple[str, ...],
    aspect_indices: torch.Tensor,
    aspect_weights: torch.Tensor | None,
    pin_memory: bool,
) -> ActivationBatch:
    loss_weights = (
        None
        if aspect_weights is None
        else aspect_weights.index_select(0, aspect_indices)
    )
    if pin_memory and torch.cuda.is_available():
        embeddings = embeddings.pin_memory()
        aspect_indices = aspect_indices.pin_memory()
        if loss_weights is not None:
            loss_weights = loss_weights.pin_memory()
    return ActivationBatch(
        embeddings=embeddings,
        unit_ids=unit_ids,
        aspect_indices=aspect_indices,
        loss_weights=loss_weights,
    )


def iter_split_batches(
    index: SaeDataIndex,
    split: SplitName,
    batch_size: int,
    aspect_weights: torch.Tensor | None,
    seed: int,
    epoch: int,
    shuffle: bool,
    pin_memory: bool,
) -> Iterator[ActivationBatch]:
    if split not in ("train", "validation"):
        raise SaeDataError(f"Unknown split: {split!r}")
    if batch_size <= 0:
        raise SaeDataError("batch_size must be positive")
    if not isinstance(shuffle, bool):
        raise SaeDataError("shuffle must be a bool")
    if not isinstance(pin_memory, bool):
        raise SaeDataError("pin_memory must be a bool")
    if split == "validation" and shuffle:
        raise SaeDataError("Validation iteration requires shuffle=False")
    if aspect_weights is not None:
        if not isinstance(aspect_weights, torch.Tensor):
            raise SaeDataError("Aspect weights must be a tensor")
        if aspect_weights.dtype != torch.float32:
            raise SaeDataError("Aspect weights must be float32")
        if tuple(aspect_weights.shape) != (len(ASPECT_NAMES),):
            raise SaeDataError("Aspect weights must contain exactly 24 values")
        if not bool(torch.isfinite(aspect_weights).all()):
            raise SaeDataError("Aspect weights must contain only finite values")
        aspect_weights = aspect_weights.detach().to(device="cpu")
    if split == "validation":
        aspect_weights = None

    units_by_row = {unit.row_index: unit for unit in index.units}
    if len(units_by_row) != len(index.units):
        raise SaeDataError("Data index contains duplicate source rows")
    shard_indices = list(range(len(index.shards)))
    local_seeds: dict[int, int] = {}
    if split == "train" and shuffle:
        generator = random.Random(seed + 1_000_003 * epoch)
        generator.shuffle(shard_indices)
        local_seeds = {
            shard_index: generator.randrange(0, 2**63)
            for shard_index in shard_indices
        }

    remainder_embeddings: torch.Tensor | None = None
    remainder_unit_ids: tuple[str, ...] = ()
    remainder_aspects: torch.Tensor | None = None
    observed_count = 0
    for shard_index in shard_indices:
        shard = index.shards[shard_index]
        shard_units = [
            unit
            for row_index in range(shard.start_row, shard.end_row)
            if (unit := units_by_row.get(row_index)) is not None
            and unit.split == split
        ]
        if split == "train" and shuffle:
            random.Random(local_seeds[shard_index]).shuffle(shard_units)

        if not shard_units:
            continue
        embeddings = _load_indexed_shard(index, shard, units_by_row)
        observed_count += len(shard_units)
        cursor = 0

        if remainder_embeddings is not None:
            if remainder_aspects is None:
                raise AssertionError("Missing aspect remainder")
            needed = batch_size - len(remainder_unit_ids)
            taken = min(needed, len(shard_units))
            taken_units = shard_units[:taken]
            taken_local_rows = torch.tensor(
                [unit.row_index - shard.start_row for unit in taken_units],
                dtype=torch.long,
            )
            taken_embeddings = embeddings.index_select(0, taken_local_rows)
            remainder_embeddings = torch.cat(
                (remainder_embeddings, taken_embeddings),
                dim=0,
            )
            del taken_embeddings, taken_local_rows
            remainder_unit_ids += tuple(unit.unit_id for unit in taken_units)
            remainder_aspects = torch.cat(
                (
                    remainder_aspects,
                    torch.tensor(
                        [unit.aspect_index for unit in taken_units],
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            cursor = taken
            if len(remainder_unit_ids) == batch_size:
                yield _make_activation_batch(
                    remainder_embeddings,
                    remainder_unit_ids,
                    remainder_aspects,
                    aspect_weights,
                    pin_memory,
                )
                remainder_embeddings = None
                remainder_unit_ids = ()
                remainder_aspects = None

        while cursor + batch_size <= len(shard_units):
            end = cursor + batch_size
            batch_units = shard_units[cursor:end]
            batch_local_rows = torch.tensor(
                [unit.row_index - shard.start_row for unit in batch_units],
                dtype=torch.long,
            )
            yield _make_activation_batch(
                embeddings.index_select(0, batch_local_rows),
                tuple(unit.unit_id for unit in batch_units),
                torch.tensor(
                    [unit.aspect_index for unit in batch_units],
                    dtype=torch.long,
                ),
                aspect_weights,
                pin_memory,
            )
            cursor = end

        if cursor < len(shard_units):
            remainder_units = shard_units[cursor:]
            remainder_local_rows = torch.tensor(
                [unit.row_index - shard.start_row for unit in remainder_units],
                dtype=torch.long,
            )
            remainder_embeddings = embeddings.index_select(
                0,
                remainder_local_rows,
            )
            remainder_unit_ids = tuple(unit.unit_id for unit in remainder_units)
            remainder_aspects = torch.tensor(
                [unit.aspect_index for unit in remainder_units],
                dtype=torch.long,
            )
        del embeddings

    expected_count = (
        index.train_unit_count if split == "train" else index.validation_unit_count
    )
    if observed_count != expected_count:
        raise SaeDataError(f"Batch iteration did not cover the full {split} split")
    if remainder_embeddings is not None:
        if remainder_aspects is None:
            raise AssertionError("Missing aspect remainder")
        yield _make_activation_batch(
            remainder_embeddings,
            remainder_unit_ids,
            remainder_aspects,
            aspect_weights,
            pin_memory,
        )


def stream_train_mean(index: SaeDataIndex, batch_size: int) -> torch.Tensor:
    total = torch.zeros(1024, dtype=torch.float64)
    count = 0
    for batch in iter_split_batches(
        index,
        split="train",
        batch_size=batch_size,
        aspect_weights=None,
        seed=0,
        epoch=0,
        shuffle=False,
        pin_memory=False,
    ):
        total += batch.embeddings.double().sum(dim=0)
        count += batch.embeddings.shape[0]
    if count != index.train_unit_count:
        raise SaeDataError("Training mean did not cover the full split")
    return (total / count).to(torch.float32)


def _accession_list_sha256(accessions: frozenset[str]) -> str:
    payload = "".join(f"{accession}\n" for accession in sorted(accessions))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_split_manifest(
    index: SaeDataIndex,
    units_path: Path,
    text_manifest_path: Path,
    activation_dir: Path,
    alpha: float,
    weights: torch.Tensor,
) -> dict[str, object]:
    units_path = Path(units_path)
    text_manifest_path = Path(text_manifest_path)
    activation_manifest_path = Path(activation_dir) / ACTIVATION_MANIFEST_FILE
    validate_text_units_manifest(units_path, text_manifest_path)
    try:
        text_manifest_sha256 = _sha256(text_manifest_path)
        activation_manifest_sha256 = _sha256(activation_manifest_path)
    except OSError as error:
        raise SaeDataError("Cannot read source manifest") from error
    if text_manifest_sha256 != index.text_manifest_sha256:
        raise SaeDataError("Current text manifest does not match the data index")
    if activation_manifest_sha256 != index.activation_manifest_sha256:
        raise SaeDataError("Current activation manifest does not match the data index")
    if weights.ndim != 1 or len(weights) != len(ASPECT_NAMES):
        raise SaeDataError("Aspect weights must contain exactly 24 values")
    manifest: dict[str, object] = {
        "activation_manifest_path": str(activation_manifest_path.resolve()),
        "activation_manifest_sha256": activation_manifest_sha256,
        "aspect_alpha": alpha,
        "aspect_names": list(ASPECT_NAMES),
        "aspect_weights": [float(value) for value in weights.tolist()],
        "assignment_sha256": index.assignment_sha256,
        "source_assignment_sha256": index.source_assignment_sha256,
        "source_train_unit_count": index.source_train_unit_count,
        "source_validation_unit_count": index.source_validation_unit_count,
        "split_algorithm": SPLIT_ALGORITHM,
        "text_manifest_path": str(text_manifest_path.resolve()),
        "text_manifest_sha256": text_manifest_sha256,
        "train_accession_count": len(index.train_accessions),
        "train_accessions": sorted(index.train_accessions),
        "train_accessions_sha256": _accession_list_sha256(index.train_accessions),
        "train_aspect_counts": list(index.train_aspect_counts),
        "train_unit_count": index.train_unit_count,
        "units_path": str(units_path.resolve()),
        "units_sha256": _sha256(units_path),
        "validation_accession_count": len(index.validation_accessions),
        "validation_accessions": sorted(index.validation_accessions),
        "validation_accessions_sha256": _accession_list_sha256(
            index.validation_accessions
        ),
        "validation_aspect_counts": list(index.validation_aspect_counts),
        "validation_unit_count": index.validation_unit_count,
    }
    return dict(sorted(manifest.items()))
