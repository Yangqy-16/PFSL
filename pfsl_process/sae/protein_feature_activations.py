"""Protein-level sparse SAE maxima and annotation covariates.

The aggregation contract deliberately keeps absence distinct from a zero
activation.  A source-excluded view exists only when at least one text unit is
eligible after applying that source's exact exclusion mask.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from itertools import zip_longest
from pathlib import Path
from types import MappingProxyType

import torch

from .concept_parsers import ConceptParseError, ConceptRecord, validate_concept_index
from .feature_atlas_config import (
    AtlasConfigError,
    AtlasPaths,
    AtlasSourceBinding,
    EXPECTED_ACTIVATION_MANIFEST_SHA256,
    EXPECTED_FEATURE_STATS_SHA256,
    EXPECTED_INFERENCE_SHA256,
    EXPECTED_SPLIT_MANIFEST_SHA256,
    EXPECTED_UNITS_SHA256,
    validate_source_binding,
)

from .feature_atlas_io import (
    AtlasIOError,
    sha256_file,
    strict_json_lines,
    strict_json_load,
    write_json_atomic,
    write_jsonl_atomic,
    write_torch_atomic,
)
from .sae_pilot_config import SaePilotConfig
from . import sae_training_data


SCHEMA_VERSION = 1
SOURCE_EXCLUDED_ASPECTS = MappingProxyType({
    "go": frozenset({5, 6, 7}),
    "reactome": frozenset({11}),
    "interpro": frozenset({10, 21}),
    "corum": frozenset({22}),
})
_SOURCES = tuple(sorted(SOURCE_EXCLUDED_ASPECTS))
_SHA256_CHARS = frozenset("0123456789abcdef")
_TASK4_MANIFEST_KEYS = {
    "atlas_source_binding", "batch_size", "complete", "device", "device_identity",
    "inference_sha256", "input_dim", "k", "model_config_fingerprint", "row_count",
    "schema_version", "shards", "source_provenance", "test_fixture_unbound", "width",
}
_PROTEIN_MANIFEST_KEYS = {
    "accession_count", "complete", "concept_binding", "filtered_uniprot_sha256",
    "proteins_per_shard", "schema_version", "shards", "source_provenance",
    "split_counts", "width",
}
_PROTEIN_SHARD_KEYS = {
    "bytes", "end_protein", "path", "proteins", "sha256", "start_protein",
}


class ProteinActivationError(ValueError):
    """Raised when protein aggregation inputs or outputs violate the contract."""


@dataclass(frozen=True)
class ProteinActivationRow:
    """One immutable protein-level activation row."""

    accession: str
    split: str
    aspect_unit_counts: tuple[int, ...]
    all_sources: Mapping[int, float]
    source_excluded: Mapping[str, Mapping[int, float]]
    eligibility: Mapping[str, bool]


def _fail(label: str, error: Exception) -> ProteinActivationError:
    return ProteinActivationError(f"Invalid {label}: {error}")


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        return strict_json_load(Path(path))
    except (AtlasIOError, OSError, TypeError, ValueError) as error:
        raise _fail(label, error) from error


def _load_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    try:
        return list(strict_json_lines(Path(path)))
    except (AtlasIOError, OSError, TypeError, ValueError) as error:
        raise _fail(label, error) from error


def _load_torch(path: Path, label: str) -> dict[str, object]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise _fail(label, error) from error
    if not isinstance(payload, dict):
        raise ProteinActivationError(f"Invalid {label}: expected a dictionary")
    return payload


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _binding_payload(binding: AtlasSourceBinding) -> dict[str, str]:
    return {
        name: getattr(binding, name)
        for name in AtlasSourceBinding.__dataclass_fields__
    }


def _unit_ids_sha256(unit_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for unit_id in unit_ids:
        digest.update(unit_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_task4_binding(
    manifest: dict[str, object],
    *,
    atlas_paths: AtlasPaths | None,
    source_binding: AtlasSourceBinding | None,
    allow_test_fixture_unbound: bool,
    expected_width: int | None,
    expected_k: int | None,
) -> None:
    if type(allow_test_fixture_unbound) is not bool:
        raise TypeError("allow_test_fixture_unbound must be a bool")
    if set(manifest) != _TASK4_MANIFEST_KEYS:
        raise ProteinActivationError("Sparse activation manifest has invalid exact schema")
    if not _is_sha256(manifest.get("inference_sha256")):
        raise ProteinActivationError("Sparse activation inference checksum is invalid")
    if not _is_sha256(manifest.get("model_config_fingerprint")):
        raise ProteinActivationError("Sparse activation model config fingerprint is invalid")
    fixture = manifest.get("test_fixture_unbound")
    if type(fixture) is not bool:
        raise ProteinActivationError("Sparse activation fixture binding flag is invalid")
    if fixture and not allow_test_fixture_unbound:
        raise ProteinActivationError("Task4 test fixture unbound cache is forbidden")
    if fixture and (expected_width is None or expected_k is None):
        raise ProteinActivationError(
            "Task4 fixture gate requires explicit expected width and k"
        )
    if expected_width is None:
        expected_width = 8192
    if expected_k is None:
        expected_k = 32
    if type(expected_width) is not int or expected_width <= 0:
        raise TypeError("expected_width must be a positive integer")
    if type(expected_k) is not int or expected_k <= 0:
        raise TypeError("expected_k must be a positive integer")
    if manifest.get("width") != expected_width:
        raise ProteinActivationError("Sparse activation width does not match expected width")
    if manifest.get("k") != expected_k:
        raise ProteinActivationError("Sparse activation TopK does not match expected k")
    if fixture:
        if manifest.get("atlas_source_binding") is not None:
            raise ProteinActivationError("Unbound Task4 fixture has invalid source binding")
        if atlas_paths is not None or source_binding is not None:
            raise ProteinActivationError("Unbound fixture cannot use production source binding")
        return
    if allow_test_fixture_unbound:
        raise ProteinActivationError("Bound Task4 input cannot enable fixture gate")
    if not isinstance(atlas_paths, AtlasPaths) or not isinstance(
        source_binding, AtlasSourceBinding
    ):
        raise ProteinActivationError("Production Task4 input requires AtlasSourceBinding")
    if expected_width != 8192 or expected_k != 32:
        raise ProteinActivationError("Production Task4 input requires width 8192 and TopK 32")
    expected_config = SaePilotConfig().fingerprint()
    if manifest.get("input_dim") != 1024 or manifest.get(
        "model_config_fingerprint"
    ) != expected_config:
        raise ProteinActivationError("Sparse activation production model config mismatch")
    approved = {
        "inference_sha256": EXPECTED_INFERENCE_SHA256,
        "feature_stats_sha256": EXPECTED_FEATURE_STATS_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
        "units_sha256": EXPECTED_UNITS_SHA256,
        "activation_manifest_sha256": EXPECTED_ACTIVATION_MANIFEST_SHA256,
    }
    for name, checksum in approved.items():
        if getattr(source_binding, name) != checksum:
            raise ProteinActivationError(f"Task4 source binding has unapproved {name}")
    if manifest.get("inference_sha256") != source_binding.inference_sha256:
        raise ProteinActivationError("Sparse activation inference binding mismatch")
    if manifest.get("atlas_source_binding") != _binding_payload(source_binding):
        raise ProteinActivationError("Sparse activation AtlasSourceBinding mismatch")
    try:
        validate_source_binding(source_binding, atlas_paths)
    except (AtlasConfigError, OSError, TypeError, ValueError) as error:
        raise _fail("Task4 AtlasSourceBinding", error) from error


def _validate_task4_paths(
    units_path: Path,
    sparse_dir: Path,
    split_manifest_path: Path,
    filtered_uniprot_tsv: Path,
    *,
    atlas_paths: AtlasPaths | None,
    allow_test_fixture_unbound: bool,
) -> None:
    if allow_test_fixture_unbound:
        return
    if not isinstance(atlas_paths, AtlasPaths):
        raise ProteinActivationError("Production Task4 input requires AtlasPaths")
    expected = (
        (units_path, atlas_paths.units_path, "units"),
        (sparse_dir, atlas_paths.output_dir / "sparse_activations", "sparse output"),
        (split_manifest_path, atlas_paths.run_dir / "split_manifest.json", "split manifest"),
        (filtered_uniprot_tsv, atlas_paths.filtered_uniprot_tsv, "filtered UniProt"),
    )
    for observed, wanted, label in expected:
        if observed.resolve() != wanted.resolve():
            raise ProteinActivationError(f"Task4 {label} path is not AtlasPaths-bound")


def _validate_concept_input(
    concept_index_path: Path | None,
    units_path: Path,
    universe: set[str],
    *,
    concept_snapshot_dir: Path | None,
    allow_test_fixture_concept_unbound: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if concept_index_path is None:
        if concept_snapshot_dir is not None or allow_test_fixture_concept_unbound:
            raise ProteinActivationError("Concept context was provided without a concept index")
        return [], {
            "available": False, "index_sha256": None, "manifest_sha256": None,
            "test_fixture_unbound": False,
        }
    path = Path(concept_index_path)
    if concept_snapshot_dir is None:
        if not allow_test_fixture_concept_unbound:
            raise ProteinActivationError("Concept index requires validated Task3 context")
        manifest_sha: str | None = None
        fixture = True
    else:
        if allow_test_fixture_concept_unbound:
            raise ProteinActivationError("Bound concept index cannot enable fixture gate")
        if path.name != "concept_index.jsonl":
            raise ProteinActivationError("Task3 concept index path must be concept_index.jsonl")
        try:
            validate_concept_index(Path(concept_snapshot_dir), units_path, path.parent)
        except (ConceptParseError, OSError, TypeError, ValueError) as error:
            raise _fail("Task3 concept index", error) from error
        manifest_sha = sha256_file(path.parent / "manifest.json")
        fixture = False
    rows = _load_jsonl(path, "concept index")
    required = {field.name for field in fields(ConceptRecord)}
    identities: list[tuple[str, str]] = []
    for number, row in enumerate(rows):
        if set(row) != required:
            raise ProteinActivationError(f"Concept row {number} schema is invalid")
        database, concept_id = row.get("database"), row.get("concept_id")
        members = row.get("protein_accessions")
        if database not in SOURCE_EXCLUDED_ASPECTS or type(concept_id) is not str:
            raise ProteinActivationError(f"Concept row {number} identity is invalid")
        if not isinstance(members, list) or members != sorted(set(members)):
            raise ProteinActivationError(f"Concept row {number} members are not sorted unique")
        if not set(members).issubset(universe):
            raise ProteinActivationError(f"Concept row {number} members violate protein universe")
        identities.append((database, concept_id))
    if identities != sorted(set(identities)):
        raise ProteinActivationError("Concept identities are not sorted unique")
    return rows, {
        "available": True, "index_sha256": sha256_file(path),
        "manifest_sha256": manifest_sha, "test_fixture_unbound": fixture,
    }


def _read_uniprot(path: Path) -> list[tuple[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or not {"Entry", "Sequence"}.issubset(reader.fieldnames):
                raise ProteinActivationError(
                    "Filtered UniProt TSV must contain Entry and Sequence columns"
                )
            rows: list[tuple[str, str]] = []
            seen: set[str] = set()
            for number, row in enumerate(reader, start=2):
                accession = row.get("Entry")
                sequence = row.get("Sequence")
                if not accession or not sequence:
                    raise ProteinActivationError(
                        f"Filtered UniProt row {number} has an empty accession or sequence"
                    )
                if accession in seen:
                    raise ProteinActivationError(
                        f"Duplicate filtered UniProt accession: {accession}"
                    )
                seen.add(accession)
                rows.append((accession, sequence))
    except ProteinActivationError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise _fail("filtered UniProt TSV", error) from error
    if not rows:
        raise ProteinActivationError("Filtered UniProt TSV is empty")
    return rows


def _validate_units(
    units_path: Path,
) -> tuple[list[dict[str, object]], list[str], dict[str, list[dict[str, object]]]]:
    rows = _load_jsonl(units_path, "SAE units")
    accessions: list[str] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    seen_units: set[str] = set()
    closed: set[str] = set()
    previous: str | None = None
    for number, row in enumerate(rows):
        unit_id = row.get("unit_id")
        accession = row.get("accession")
        aspect = row.get("aspect_index")
        if type(unit_id) is not str or not unit_id:
            raise ProteinActivationError(f"Unit row {number} has an invalid unit ID")
        if unit_id in seen_units:
            raise ProteinActivationError(f"Duplicate unit ID: {unit_id}")
        seen_units.add(unit_id)
        if type(accession) is not str or not accession:
            raise ProteinActivationError(f"Unit row {number} has an invalid accession")
        if type(aspect) is not int or not 0 <= aspect < 24:
            raise ProteinActivationError(f"Unit row {number} has an invalid aspect index")
        if accession != previous:
            if previous is not None:
                closed.add(previous)
            if accession in closed:
                raise ProteinActivationError(
                    f"Protein units must be contiguous by accession: {accession}"
                )
            accessions.append(accession)
            grouped[accession] = []
            previous = accession
        grouped[accession].append(row)
    if not rows:
        raise ProteinActivationError("SAE units are empty")
    return rows, accessions, grouped


def _validate_split_manifest(
    path: Path,
    units_path: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    manifest = _load_json(path, "split manifest")
    if manifest.get("units_sha256") != sha256_file(Path(units_path)):
        raise ProteinActivationError("Split manifest units checksum mismatch")
    train = manifest.get("train_accessions")
    validation = manifest.get("validation_accessions")
    if (
        not isinstance(train, list)
        or not isinstance(validation, list)
        or any(type(item) is not str or not item for item in train + validation)
        or len(train) != len(set(train))
        or len(validation) != len(set(validation))
    ):
        raise ProteinActivationError("Split manifest accession lists are invalid")
    overlap = set(train).intersection(validation)
    if overlap:
        raise ProteinActivationError(
            f"Split manifest train/validation overlap: {sorted(overlap)[0]}"
        )
    if manifest.get("train_accession_count") != len(train) or manifest.get(
        "validation_accession_count"
    ) != len(validation):
        raise ProteinActivationError("Split manifest accession counts are invalid")
    splits = {accession: "train" for accession in train}
    splits.update({accession: "validation" for accession in validation})
    return manifest, splits


def _validate_sparse_inputs(
    sparse_dir: Path,
    unit_rows: list[dict[str, object]],
    split_manifest: dict[str, object],
    *,
    atlas_paths: AtlasPaths | None,
    source_binding: AtlasSourceBinding | None,
    allow_test_fixture_unbound: bool,
    expected_width: int | None,
    expected_k: int | None,
) -> tuple[dict[str, object], Iterator[tuple[tuple[int, ...], tuple[float, ...]]]]:
    sparse_dir = Path(sparse_dir)
    manifest = _load_json(sparse_dir / "manifest.json", "sparse activation manifest")
    _validate_task4_binding(
        manifest, atlas_paths=atlas_paths, source_binding=source_binding,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
        expected_width=expected_width, expected_k=expected_k,
    )
    width = manifest.get("width")
    k = manifest.get("k")
    shards = manifest.get("shards")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("complete") is not True
        or type(width) is not int
        or width <= 0
        or type(k) is not int
        or k <= 0
        or manifest.get("row_count") != len(unit_rows)
        or not isinstance(shards, list)
        or not shards
    ):
        raise ProteinActivationError("Sparse activation manifest has invalid schema")
    provenance = manifest.get("source_provenance")
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {
            "activation_manifest_sha256", "assignment_sha256", "text_manifest_sha256"
        }
        or any(not _is_sha256(value) for value in provenance.values())
        or provenance.get("assignment_sha256") != split_manifest.get("assignment_sha256")
    ):
        raise ProteinActivationError("Sparse activation assignment provenance mismatch")
    if source_binding is not None and (
        provenance.get("activation_manifest_sha256")
        != source_binding.activation_manifest_sha256
        or provenance.get("text_manifest_sha256") != source_binding.text_manifest_sha256
        or provenance.get("assignment_sha256") != source_binding.assignment_sha256
    ):
        raise ProteinActivationError("Sparse activation source provenance binding mismatch")

    bound_index: object | None = None
    if not allow_test_fixture_unbound:
        if not isinstance(atlas_paths, AtlasPaths):
            raise ProteinActivationError("Production Task4 input requires AtlasPaths")
        try:
            bound_index = sae_training_data.build_data_index(
                atlas_paths.units_path,
                atlas_paths.text_manifest_path,
                atlas_paths.activation_dir,
            )
        except (sae_training_data.SaeDataError, OSError, TypeError, ValueError) as error:
            raise _fail("bound Task4 source data index", error) from error
        bound_units = getattr(bound_index, "units", ())
        bound_shards = getattr(bound_index, "shards", ())
        if (
            tuple(getattr(unit, "unit_id", None) for unit in bound_units)
            != tuple(str(row["unit_id"]) for row in unit_rows)
            or len(bound_shards) != len(shards)
        ):
            raise ProteinActivationError(
                "Task4 sources do not match registered activation index layout"
            )

    validated: list[tuple[Path, int, int]] = []
    next_row = 0
    for number, metadata in enumerate(shards):
        if not isinstance(metadata, dict) or set(metadata) != {
            "bytes", "end_row", "nnz", "path", "rows", "sha256",
            "source_path", "source_sha256", "start_row", "unit_ids_sha256",
        }:
            raise ProteinActivationError("Sparse shard metadata is invalid")
        expected_name = f"shard_{number:05d}.pt"
        start = metadata.get("start_row")
        end = metadata.get("end_row")
        path = sparse_dir / expected_name
        if (
            metadata.get("path") != expected_name
            or type(start) is not int
            or type(end) is not int
            or start != next_row
            or end <= start
            or end > len(unit_rows)
            or metadata.get("rows") != end - start
        ):
            raise ProteinActivationError("Sparse shards do not cover contiguous row ranges")
        try:
            observed_sha = sha256_file(path)
            observed_bytes = path.stat().st_size
        except (AtlasIOError, OSError, ValueError) as error:
            raise _fail("registered sparse shard", error) from error
        if metadata.get("sha256") != observed_sha:
            raise ProteinActivationError(f"Sparse shard checksum mismatch: {path}")
        if metadata.get("bytes") != observed_bytes:
            raise ProteinActivationError(f"Sparse shard byte size mismatch: {path}")
        payload = _load_torch(path, "sparse activation shard")
        expected_keys = {
            "activation_values",
            "end_row",
            "feature_indices",
            "indptr",
            "schema_version",
            "start_row",
            "unit_ids",
        }
        if set(payload) != expected_keys or payload.get("schema_version") != 1:
            raise ProteinActivationError("Sparse activation shard has invalid schema")
        expected_ids = tuple(
            str(row["unit_id"]) for row in unit_rows[start:end]
        )
        if metadata.get("unit_ids_sha256") != _unit_ids_sha256(expected_ids):
            raise ProteinActivationError("Sparse shard ordered unit-ID binding is invalid")
        raw_source_path = metadata.get("source_path")
        if type(raw_source_path) is not str or not _is_sha256(
            metadata.get("source_sha256")
        ):
            raise ProteinActivationError("Sparse shard source binding is invalid")
        source_path = Path(raw_source_path)
        if bound_index is not None:
            registered = getattr(bound_index, "shards")[number]
            registered_ids = tuple(
                getattr(unit, "unit_id")
                for unit in getattr(bound_index, "units")[start:end]
            )
            if (
                source_path.resolve() != Path(getattr(registered, "path")).resolve()
                or metadata.get("source_sha256") != getattr(registered, "sha256")
                or start != getattr(registered, "start_row")
                or end != getattr(registered, "end_row")
                or registered_ids != expected_ids
            ):
                raise ProteinActivationError(
                    "Task4 source shard does not match registered activation shard"
                )
        try:
            if sha256_file(source_path) != metadata.get("source_sha256"):
                raise ProteinActivationError("Sparse shard source checksum mismatch")
        except (AtlasIOError, OSError, ValueError) as error:
            raise _fail("Task4 source shard", error) from error
        if payload.get("unit_ids") != expected_ids:
            raise ProteinActivationError("Sparse shard unit IDs violate exact source order")
        indptr = payload.get("indptr")
        indices = payload.get("feature_indices")
        values = payload.get("activation_values")
        rows = end - start
        if (
            payload.get("start_row") != start
            or payload.get("end_row") != end
            or not isinstance(indptr, torch.Tensor)
            or indptr.dtype != torch.int64
            or indptr.device.type != "cpu"
            or tuple(indptr.shape) != (rows + 1,)
            or not isinstance(indices, torch.Tensor)
            or indices.dtype != torch.int32
            or indices.device.type != "cpu"
            or indices.ndim != 1
            or not isinstance(values, torch.Tensor)
            or values.dtype != torch.float32
            or values.device.type != "cpu"
            or values.ndim != 1
            or len(indices) != len(values)
            or int(indptr[0].item()) != 0
            or int(indptr[-1].item()) != len(indices)
            or bool(torch.any(indptr[1:] < indptr[:-1]).item())
            or bool(torch.any(indptr[1:] - indptr[:-1] > k).item())
            or (len(values) and not bool(torch.isfinite(values).all().item()))
            or (len(values) and bool(torch.any(values <= 0).item()))
            or (len(indices) and int(indices.min().item()) < 0)
            or (len(indices) and int(indices.max().item()) >= width)
        ):
            raise ProteinActivationError("Sparse activation shard CSR data are invalid")
        for row in range(rows):
            left = int(indptr[row].item())
            right = int(indptr[row + 1].item())
            selected = indices[left:right]
            if len(selected) > 1 and bool(torch.any(selected[1:] <= selected[:-1]).item()):
                raise ProteinActivationError(
                    "Sparse feature indices must be unique and ascending within each row"
                )
        if metadata.get("nnz") != len(indices):
            raise ProteinActivationError("Sparse shard nnz does not match its manifest")
        validated.append((path, start, end))
        next_row = end
    if next_row != len(unit_rows):
        raise ProteinActivationError("Sparse shards do not cover every unit row")

    def iterator() -> Iterator[tuple[tuple[int, ...], tuple[float, ...]]]:
        for path, start, end in validated:
            payload = _load_torch(path, "sparse activation shard")
            indptr = payload["indptr"]
            indices = payload["feature_indices"]
            values = payload["activation_values"]
            assert isinstance(indptr, torch.Tensor)
            assert isinstance(indices, torch.Tensor)
            assert isinstance(values, torch.Tensor)
            for row in range(end - start):
                left = int(indptr[row].item())
                right = int(indptr[row + 1].item())
                yield (
                    tuple(int(value) for value in indices[left:right].tolist()),
                    tuple(float(value) for value in values[left:right].tolist()),
                )

    return manifest, iterator()


def _freeze_row(
    accession: str,
    split: str,
    counts: list[int] | tuple[int, ...],
    all_sources: Mapping[int, float],
    source_excluded: Mapping[str, Mapping[int, float]],
    eligibility: Mapping[str, bool],
) -> ProteinActivationRow:
    frozen_excluded = MappingProxyType(
        {
            source: MappingProxyType(dict(sorted(values.items())))
            for source, values in sorted(source_excluded.items())
        }
    )
    return ProteinActivationRow(
        accession=accession,
        split=split,
        aspect_unit_counts=tuple(counts),
        all_sources=MappingProxyType(dict(sorted(all_sources.items()))),
        source_excluded=frozen_excluded,
        eligibility=MappingProxyType(dict(sorted(eligibility.items()))),
    )


def _iter_computed_rows(
    units_path: Path,
    sparse_dir: Path,
    split_manifest_path: Path,
    filtered_uniprot_tsv: Path,
    *,
    expected_accession_count: int,
    expected_split_counts: tuple[int, int],
    atlas_paths: AtlasPaths | None,
    source_binding: AtlasSourceBinding | None,
    allow_test_fixture_unbound: bool,
    expected_width: int | None,
    expected_k: int | None,
) -> tuple[Iterator[ProteinActivationRow], dict[str, object], dict[str, object]]:
    if type(expected_accession_count) is not int or expected_accession_count <= 0:
        raise TypeError("expected_accession_count must be a positive integer")
    if (
        not isinstance(expected_split_counts, tuple)
        or len(expected_split_counts) != 2
        or any(type(value) is not int or value < 0 for value in expected_split_counts)
    ):
        raise TypeError("expected_split_counts must be a pair of nonnegative integers")
    unit_rows, accession_order, grouped = _validate_units(Path(units_path))
    split_manifest, splits = _validate_split_manifest(
        Path(split_manifest_path), Path(units_path)
    )
    uniprot_rows = _read_uniprot(Path(filtered_uniprot_tsv))
    uniprot_accessions = [accession for accession, _ in uniprot_rows]
    if len(uniprot_accessions) != expected_accession_count:
        raise ProteinActivationError(
            f"Observed {len(uniprot_accessions)} proteins, expected accession count "
            f"{expected_accession_count}"
        )
    if set(uniprot_accessions) != set(accession_order):
        raise ProteinActivationError("SAE units do not cover the exact filtered UniProt accessions")
    if set(splits) != set(uniprot_accessions):
        raise ProteinActivationError("Split manifest does not cover the exact protein accessions")
    observed_split_counts = (
        sum(splits[item] == "train" for item in uniprot_accessions),
        sum(splits[item] == "validation" for item in uniprot_accessions),
    )
    if observed_split_counts != expected_split_counts:
        raise ProteinActivationError(
            f"Observed split counts {observed_split_counts}, expected {expected_split_counts}"
        )
    sparse_manifest, sparse_rows = _validate_sparse_inputs(
        Path(sparse_dir), unit_rows, split_manifest,
        atlas_paths=atlas_paths, source_binding=source_binding,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
        expected_width=expected_width, expected_k=expected_k,
    )
    def iterator() -> Iterator[ProteinActivationRow]:
        row_activations = iter(sparse_rows)
        for accession in accession_order:
            all_maxima: dict[int, float] = {}
            excluded_maxima: dict[str, dict[int, float]] = {
                source: {} for source in _SOURCES
            }
            counts = [0] * 24
            eligibility = {source: False for source in _SOURCES}
            for unit in grouped[accession]:
                aspect = int(unit["aspect_index"])
                counts[aspect] += 1
                try:
                    indices, values = next(row_activations)
                except StopIteration as error:
                    raise ProteinActivationError(
                        "Sparse activations ended before SAE units"
                    ) from error
                for feature, value in zip(indices, values, strict=True):
                    all_maxima[feature] = max(all_maxima.get(feature, value), value)
                for source in _SOURCES:
                    if aspect in SOURCE_EXCLUDED_ASPECTS[source]:
                        continue
                    eligibility[source] = True
                    maxima = excluded_maxima[source]
                    for feature, value in zip(indices, values, strict=True):
                        maxima[feature] = max(maxima.get(feature, value), value)
            yield _freeze_row(
                accession, splits[accession], counts, all_maxima,
                {
                    source: {
                        feature: value for feature, value in excluded_maxima[source].items()
                        if value == all_maxima[feature]
                    }
                    for source in _SOURCES if eligibility[source]
                }, eligibility,
            )
        try:
            next(row_activations)
        except StopIteration:
            return
        raise ProteinActivationError("Sparse activations contain extra unit rows")

    return iterator(), sparse_manifest, split_manifest


def _csr(mappings: list[Mapping[int, float]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indptr = [0]
    indices: list[int] = []
    values: list[float] = []
    for mapping in mappings:
        for feature, value in sorted(mapping.items()):
            indices.append(feature)
            values.append(value)
        indptr.append(len(indices))
    return (
        torch.tensor(indptr, dtype=torch.int64),
        torch.tensor(indices, dtype=torch.int32),
        torch.tensor(values, dtype=torch.float32),
    )


def _encode_shard(rows: list[ProteinActivationRow], start: int) -> dict[str, object]:
    all_indptr, all_indices, all_values = _csr([row.all_sources for row in rows])
    excluded: dict[str, dict[str, torch.Tensor]] = {}
    eligibility: dict[str, torch.Tensor] = {}
    for source in _SOURCES:
        eligibility[source] = torch.tensor(
            [row.eligibility[source] for row in rows], dtype=torch.bool
        )
        indptr, indices, values = _csr(
            [row.source_excluded.get(source, {}) for row in rows]
        )
        excluded[source] = {"indices": indices, "indptr": indptr, "values": values}
    return {
        "accessions": tuple(row.accession for row in rows),
        "all_source_indices": all_indices,
        "all_source_indptr": all_indptr,
        "all_source_values": all_values,
        "aspect_unit_counts": torch.tensor(
            [row.aspect_unit_counts for row in rows], dtype=torch.int64
        ),
        "eligibility": eligibility,
        "end_protein": start + len(rows),
        "schema_version": SCHEMA_VERSION,
        "source_excluded": excluded,
        "splits": tuple(row.split for row in rows),
        "start_protein": start,
    }


def _manifest_template(
    *,
    accession_count: int,
    split_counts: tuple[int, int],
    units_path: Path,
    sparse_dir: Path,
    split_manifest_path: Path,
    filtered_uniprot_tsv: Path,
    concept_binding: dict[str, object],
    proteins_per_shard: int,
    width: int,
) -> dict[str, object]:
    return {
        "accession_count": accession_count,
        "complete": False,
        "concept_binding": concept_binding,
        "filtered_uniprot_sha256": sha256_file(filtered_uniprot_tsv),
        "proteins_per_shard": proteins_per_shard,
        "schema_version": SCHEMA_VERSION,
        "shards": {"train": [], "validation": []},
        "source_provenance": {
            "sparse_manifest_sha256": sha256_file(sparse_dir / "manifest.json"),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "units_sha256": sha256_file(units_path),
        },
        "split_counts": {
            "train": split_counts[0],
            "validation": split_counts[1],
        },
        "width": width,
    }


def _registered_output_paths(manifest: dict[str, object], output_dir: Path) -> list[Path]:
    """Return the explicit-overwrite allowlist for Task 5 shard names.

    Resume validation remains exact and fail closed.  Explicit overwrite must
    additionally be able to recover from a damaged registration prefix, so it
    owns only the fixed Task 5 shard namespace and never trusts a manifest path
    as an arbitrary filesystem path.
    """

    def owned_name(name: str) -> bool:
        if not name.endswith(".pt"):
            return False
        stem = name[:-3]
        prefixes = ("shard_", "train_shard_", "validation_shard_")
        return any(
            stem.startswith(prefix)
            and len(stem) == len(prefix) + 5
            and stem[len(prefix):].isdigit()
            for prefix in prefixes
        )

    shards = manifest.get("shards")
    if isinstance(shards, list):
        records = shards
    elif isinstance(shards, dict) and set(shards) == {"train", "validation"}:
        if not all(isinstance(shards[split], list) for split in ("train", "validation")):
            raise ProteinActivationError("Protein activation split shards are invalid")
        records = [
            metadata
            for split in ("train", "validation")
            for metadata in shards[split]
        ]
    else:
        raise ProteinActivationError("Protein activation manifest shards are invalid")
    for metadata in records:
        path = metadata.get("path") if isinstance(metadata, dict) else None
        if type(path) is not str or not owned_name(path):
            raise ProteinActivationError("Protein activation manifest shard path is invalid")
    try:
        return sorted(
            (path for path in output_dir.iterdir() if owned_name(path.name)),
            key=lambda path: path.name,
        )
    except OSError as error:
        raise _fail("protein activation overwrite allowlist", error) from error


def _output_metadata(path: Path, start: int, end: int) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "end_protein": end,
        "path": path.name,
        "proteins": end - start,
        "sha256": sha256_file(path),
        "start_protein": start,
    }


def _validate_split_registration_schema(manifest: dict[str, object]) -> None:
    if set(manifest) != _PROTEIN_MANIFEST_KEYS:
        raise ProteinActivationError("Protein activation manifest has invalid exact schema")
    shards = manifest.get("shards")
    counts = manifest.get("split_counts")
    size = manifest.get("proteins_per_shard")
    if (
        not isinstance(shards, dict) or set(shards) != {"train", "validation"}
        or not isinstance(counts, dict) or set(counts) != {"train", "validation"}
        or any(type(value) is not int or value < 0 for value in counts.values())
        or type(size) is not int or size <= 0
    ):
        raise ProteinActivationError("Protein activation manifest split schema is invalid")
    for split in ("train", "validation"):
        records = shards[split]
        if not isinstance(records, list):
            raise ProteinActivationError("Protein activation split shard list is invalid")
        maximum = (counts[split] + size - 1) // size
        if len(records) > maximum:
            raise ProteinActivationError("Protein activation manifest registers too many split shards")
        for number, metadata in enumerate(records):
            start = number * size
            end = min(start + size, counts[split])
            if (
                not isinstance(metadata, dict)
                or set(metadata) != _PROTEIN_SHARD_KEYS
                or metadata.get("path") != f"{split}_shard_{number:05d}.pt"
                or metadata.get("start_protein") != start
                or metadata.get("end_protein") != end
                or metadata.get("proteins") != end - start
                or type(metadata.get("bytes")) is not int
                or metadata["bytes"] <= 0
                or not _is_sha256(metadata.get("sha256"))
            ):
                raise ProteinActivationError("Protein activation registered prefix metadata is invalid")


def _validate_registered_prefixes(
    manifest: dict[str, object],
    output_dir: Path,
    expected_rows: Iterator[ProteinActivationRow],
    *,
    width: int,
) -> None:
    _validate_split_registration_schema(manifest)
    shards = manifest["shards"]
    size = manifest["proteins_per_shard"]
    assert isinstance(shards, dict) and isinstance(size, int)
    buffers: dict[str, list[ProteinActivationRow]] = {"train": [], "validation": []}
    starts = {"train": 0, "validation": 0}

    def validate_buffer(split: str) -> None:
        rows = buffers[split]
        number = starts[split] // size
        records = shards[split]
        assert isinstance(records, list)
        if number < len(records):
            path = output_dir / f"{split}_shard_{number:05d}.pt"
            metadata = records[number]
            if metadata != _output_metadata(path, starts[split], starts[split] + len(rows)):
                raise ProteinActivationError("Registered protein activation prefix checksum changed")
            observed = _decode_output_shard(
                _load_torch(path, "registered protein activation prefix"),
                start=starts[split], end=starts[split] + len(rows), width=width,
            )
            if [_row_signature(row) for row in observed] != [
                _row_signature(row) for row in rows
            ]:
                raise ProteinActivationError("Registered protein activation prefix content mismatch")
        starts[split] += len(rows)
        buffers[split] = []

    for row in expected_rows:
        buffers[row.split].append(row)
        if len(buffers[row.split]) == size:
            validate_buffer(row.split)
    for split in ("train", "validation"):
        if buffers[split]:
            validate_buffer(split)


def _validate_resume_shards(
    manifest: dict[str, object],
    output_dir: Path,
    expected_rows: list[ProteinActivationRow],
    *,
    proteins_per_shard: int,
    width: int,
) -> int:
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ProteinActivationError("Protein activation manifest shards are invalid")
    total_shards = (len(expected_rows) + proteins_per_shard - 1) // proteins_per_shard
    if len(shards) > total_shards:
        raise ProteinActivationError("Protein activation manifest registers too many shards")
    for number, metadata in enumerate(shards):
        start = number * proteins_per_shard
        end = min(start + proteins_per_shard, len(expected_rows))
        path = output_dir / f"shard_{number:05d}.pt"
        if not isinstance(metadata, dict) or set(metadata) != {
            "bytes", "end_protein", "path", "proteins", "sha256", "start_protein"
        } or any(
            (
                metadata.get("path") != path.name,
                metadata.get("start_protein") != start,
                metadata.get("end_protein") != end,
                metadata.get("proteins") != end - start,
            )
        ):
            raise ProteinActivationError("Registered protein activation shard metadata mismatch")
        try:
            observed_sha = sha256_file(path)
            observed_bytes = path.stat().st_size
        except (AtlasIOError, OSError, ValueError) as error:
            raise _fail("registered protein activation shard", error) from error
        if metadata.get("sha256") != observed_sha:
            raise ProteinActivationError(
                f"Registered protein activation shard checksum mismatch: {path}"
            )
        if metadata.get("bytes") != observed_bytes:
            raise ProteinActivationError(
                f"Registered protein activation shard byte size mismatch: {path}"
            )
        observed = _decode_output_shard(
            _load_torch(path, "protein activation shard"),
            start=start,
            end=end,
            width=width,
        )
        if [_row_signature(row) for row in observed] != [
            _row_signature(row) for row in expected_rows[start:end]
        ]:
            raise ProteinActivationError(
                "Registered protein activation shard does not match recomputed maxima"
            )
    return len(shards)


def aggregate_protein_activations(
    units_path: Path,
    sparse_dir: Path,
    split_manifest_path: Path,
    filtered_uniprot_tsv: Path,
    output_dir: Path,
    *,
    concept_index_path: Path | None = None,
    proteins_per_shard: int = 1024,
    expected_accession_count: int = 10_618,
    expected_split_counts: tuple[int, int] = (9_539, 1_079),
    overwrite: bool = False,
    atlas_paths: AtlasPaths | None = None,
    source_binding: AtlasSourceBinding | None = None,
    allow_test_fixture_unbound: bool = False,
    expected_width: int | None = None,
    expected_k: int | None = None,
    concept_snapshot_dir: Path | None = None,
    allow_test_fixture_concept_unbound: bool = False,
) -> dict[str, object]:
    """Stream unit-level sparse rows into deterministic protein-level maxima."""

    if type(proteins_per_shard) is not int or proteins_per_shard <= 0:
        raise TypeError("proteins_per_shard must be a positive integer")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    paths = tuple(
        Path(path)
        for path in (units_path, sparse_dir, split_manifest_path, filtered_uniprot_tsv, output_dir)
    )
    units_path, sparse_dir, split_manifest_path, filtered_uniprot_tsv, output_dir = paths
    concept = Path(concept_index_path) if concept_index_path is not None else None
    universe = {accession for accession, _ in _read_uniprot(filtered_uniprot_tsv)}
    _, concept_binding = _validate_concept_input(
        concept, units_path, universe, concept_snapshot_dir=concept_snapshot_dir,
        allow_test_fixture_concept_unbound=allow_test_fixture_concept_unbound,
    )
    rows, sparse_manifest, _ = _iter_computed_rows(
        units_path,
        sparse_dir,
        split_manifest_path,
        filtered_uniprot_tsv,
        expected_accession_count=expected_accession_count,
        expected_split_counts=expected_split_counts,
        atlas_paths=atlas_paths, source_binding=source_binding,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
        expected_width=expected_width, expected_k=expected_k,
    )
    _validate_task4_paths(
        units_path, sparse_dir, split_manifest_path, filtered_uniprot_tsv,
        atlas_paths=atlas_paths, allow_test_fixture_unbound=allow_test_fixture_unbound,
    )
    template = _manifest_template(
        accession_count=expected_accession_count,
        split_counts=expected_split_counts,
        units_path=units_path,
        sparse_dir=sparse_dir,
        split_manifest_path=split_manifest_path,
        filtered_uniprot_tsv=filtered_uniprot_tsv,
        concept_binding=concept_binding,
        proteins_per_shard=proteins_per_shard,
        width=int(sparse_manifest["width"]),
    )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = _load_json(manifest_path, "protein activation manifest")
        if overwrite:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(
                prefix=".protein-stage-", dir=output_dir.parent
            ))
            backup = Path(tempfile.mkdtemp(
                prefix=".protein-backup-", dir=output_dir.parent
            ))
            old_paths: list[Path] = []
            committed: list[Path] = []
            try:
                old_paths = _registered_output_paths(existing, output_dir)
                staged_manifest = aggregate_protein_activations(
                    units_path,
                    sparse_dir,
                    split_manifest_path,
                    filtered_uniprot_tsv,
                    stage,
                    concept_index_path=concept,
                    proteins_per_shard=proteins_per_shard,
                    expected_accession_count=expected_accession_count,
                    expected_split_counts=expected_split_counts,
                    overwrite=False,
                    atlas_paths=atlas_paths,
                    source_binding=source_binding,
                    allow_test_fixture_unbound=allow_test_fixture_unbound,
                    expected_width=expected_width,
                    expected_k=expected_k,
                    concept_snapshot_dir=concept_snapshot_dir,
                    allow_test_fixture_concept_unbound=allow_test_fixture_concept_unbound,
                )
                staged_paths = _registered_output_paths(staged_manifest, stage)
                old_names = {path.name for path in old_paths}
                for staged_path in staged_paths:
                    destination = output_dir / staged_path.name
                    if destination.exists() and destination.name not in old_names:
                        raise ProteinActivationError(
                            f"Refusing to overwrite unregistered split shard: {destination}"
                        )
                for path in [*old_paths, manifest_path]:
                    if path.is_symlink():
                        raise ProteinActivationError(
                            "Refusing to overwrite a symlinked registered output"
                        )
                    os.replace(path, backup / path.name)
                for staged_path in staged_paths:
                    destination = output_dir / staged_path.name
                    os.replace(staged_path, destination)
                    committed.append(destination)
                os.replace(stage / "manifest.json", manifest_path)
                committed.append(manifest_path)
                result = validate_protein_activations(
                    units_path,
                    sparse_dir,
                    split_manifest_path,
                    filtered_uniprot_tsv,
                    output_dir,
                    concept_index_path=concept,
                    expected_accession_count=expected_accession_count,
                    expected_split_counts=expected_split_counts,
                    atlas_paths=atlas_paths,
                    source_binding=source_binding,
                    allow_test_fixture_unbound=allow_test_fixture_unbound,
                    expected_width=expected_width,
                    expected_k=expected_k,
                    concept_snapshot_dir=concept_snapshot_dir,
                    allow_test_fixture_concept_unbound=allow_test_fixture_concept_unbound,
                )
            except BaseException as error:
                rollback_errors: list[tuple[str, BaseException]] = []
                # Preserve every partially committed replacement in the named
                # stage before restoring old bytes.  The follow-up unlink is a
                # fail-closed guard against a destination recreated by a race.
                for path in committed:
                    if path.exists():
                        recovery = stage / f"partial-new-{path.name}"
                        try:
                            os.replace(path, recovery)
                            path.unlink(missing_ok=True)
                        except BaseException as failure:
                            rollback_errors.append((f"cleanup {path}", failure))
                # Copy through the retained stage and replace unconditionally;
                # this keeps the backup's only old copy available until the
                # whole rollback has succeeded.
                for old_path in [*old_paths, manifest_path]:
                    saved = backup / old_path.name
                    if saved.exists():
                        restore_copy = stage / f"restore-old-{old_path.name}"
                        try:
                            shutil.copy2(saved, restore_copy)
                            os.replace(restore_copy, old_path)
                        except BaseException as failure:
                            rollback_errors.append((f"restore {old_path}", failure))
                if rollback_errors:
                    details = "; ".join(
                        f"{label}: {failure}" for label, failure in rollback_errors
                    )
                    raise ProteinActivationError(
                        f"Protein overwrite rollback requires recovery; backup: {backup}; "
                        f"staging: {stage}; original error: {error}; {details}"
                    ) from rollback_errors[0][1]
                shutil.rmtree(stage, ignore_errors=True)
                shutil.rmtree(backup, ignore_errors=True)
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                if isinstance(error, ProteinActivationError):
                    raise
                raise ProteinActivationError(
                    f"Cannot atomically overwrite protein activation split views: {error}"
                ) from error
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
            return result
        else:
            _validate_split_registration_schema(existing)
            if set(existing) != set(template) or {
                key: value for key, value in existing.items() if key not in {"complete", "shards"}
            } != {
                key: value for key, value in template.items() if key not in {"complete", "shards"}
            }:
                raise ProteinActivationError("Protein activation manifest input binding mismatch")
            if existing.get("complete") is True:
                return validate_protein_activations(
                    units_path,
                    sparse_dir,
                    split_manifest_path,
                    filtered_uniprot_tsv,
                    output_dir,
                    concept_index_path=concept,
                    expected_accession_count=expected_accession_count,
                    expected_split_counts=expected_split_counts,
                    atlas_paths=atlas_paths, source_binding=source_binding,
                    allow_test_fixture_unbound=allow_test_fixture_unbound,
                    expected_width=expected_width, expected_k=expected_k,
                    concept_snapshot_dir=concept_snapshot_dir,
                    allow_test_fixture_concept_unbound=allow_test_fixture_concept_unbound,
                )
            if existing.get("complete") is not False:
                raise ProteinActivationError("Protein activation completion flag is invalid")
            _validate_registered_prefixes(
                existing, output_dir, rows, width=int(sparse_manifest["width"])
            )
            rows, sparse_manifest, _ = _iter_computed_rows(
                units_path,
                sparse_dir,
                split_manifest_path,
                filtered_uniprot_tsv,
                expected_accession_count=expected_accession_count,
                expected_split_counts=expected_split_counts,
                atlas_paths=atlas_paths, source_binding=source_binding,
                allow_test_fixture_unbound=allow_test_fixture_unbound,
                expected_width=expected_width, expected_k=expected_k,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = existing
    else:
        manifest = template
        write_json_atomic(manifest_path, manifest)
    shards = manifest["shards"]
    assert isinstance(shards, dict)
    registered = {
        split: len(shards[split]) for split in ("train", "validation")
    }
    selected: dict[str, list[ProteinActivationRow]] = {
        "train": [], "validation": []
    }
    starts = {"train": 0, "validation": 0}

    def publish(split: str, selected_rows: list[ProteinActivationRow], shard_start: int) -> None:
        split_shards = shards[split]
        assert isinstance(split_shards, list)
        number = shard_start // proteins_per_shard
        path = output_dir / f"{split}_shard_{number:05d}.pt"
        end = shard_start + len(selected_rows)
        if number < registered[split]:
            metadata = split_shards[number]
            if not isinstance(metadata, dict) or metadata != _output_metadata(
                path, shard_start, end
            ):
                raise ProteinActivationError("Registered protein activation shard mismatch")
            observed = _decode_output_shard(
                _load_torch(path, "registered protein activation shard"),
                start=shard_start,
                end=end,
                width=int(sparse_manifest["width"]),
            )
            if [_row_signature(row) for row in observed] != [
                _row_signature(row) for row in selected_rows
            ]:
                raise ProteinActivationError(
                    "Registered protein activation shard does not match recomputed maxima"
                )
            return
        if path.exists():
            observed = _decode_output_shard(
                _load_torch(path, "unregistered protein activation shard"),
                start=shard_start, end=end, width=int(sparse_manifest["width"]),
            )
            if [_row_signature(row) for row in observed] != [
                _row_signature(row) for row in selected_rows
            ]:
                raise ProteinActivationError(
                    "Unregistered protein activation shard does not match recomputed maxima"
                )
        else:
            write_torch_atomic(path, _encode_shard(selected_rows, shard_start))
        split_shards.append(_output_metadata(path, shard_start, end))
        write_json_atomic(manifest_path, manifest)

    for row in rows:
        split = row.split
        selected[split].append(row)
        if len(selected[split]) == proteins_per_shard:
            publish(split, selected[split], starts[split])
            starts[split] += len(selected[split])
            selected[split] = []
    for split in ("train", "validation"):
        if selected[split]:
            publish(split, selected[split], starts[split])
            starts[split] += len(selected[split])
    if (
        starts != {"train": expected_split_counts[0], "validation": expected_split_counts[1]}
        or any(registered[split] > len(shards[split]) for split in registered)
    ):
        raise ProteinActivationError("Protein activation shard coverage mismatch")
    manifest["complete"] = True
    write_json_atomic(manifest_path, manifest)
    return validate_protein_activations(
        units_path,
        sparse_dir,
        split_manifest_path,
        filtered_uniprot_tsv,
        output_dir,
        concept_index_path=concept,
        expected_accession_count=expected_accession_count,
        expected_split_counts=expected_split_counts,
        atlas_paths=atlas_paths, source_binding=source_binding,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
        expected_width=expected_width, expected_k=expected_k,
        concept_snapshot_dir=concept_snapshot_dir,
        allow_test_fixture_concept_unbound=allow_test_fixture_concept_unbound,
    )


def _decode_csr(
    indptr: object,
    indices: object,
    values: object,
    rows: int,
    width: int,
    label: str,
) -> list[dict[int, float]]:
    if (
        not isinstance(indptr, torch.Tensor)
        or indptr.dtype != torch.int64
        or tuple(indptr.shape) != (rows + 1,)
        or not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.int32
        or indices.ndim != 1
        or not isinstance(values, torch.Tensor)
        or values.dtype != torch.float32
        or values.ndim != 1
        or len(indices) != len(values)
        or int(indptr[0].item()) != 0
        or int(indptr[-1].item()) != len(indices)
        or bool(torch.any(indptr[1:] < indptr[:-1]).item())
        or (len(indices) and (int(indices.min().item()) < 0 or int(indices.max().item()) >= width))
        or (len(values) and (not bool(torch.isfinite(values).all().item()) or bool(torch.any(values <= 0).item())))
    ):
        raise ProteinActivationError(f"Invalid {label} CSR data")
    result: list[dict[int, float]] = []
    for row in range(rows):
        left = int(indptr[row].item())
        right = int(indptr[row + 1].item())
        row_indices = indices[left:right]
        if len(row_indices) > 1 and bool(torch.any(row_indices[1:] <= row_indices[:-1]).item()):
            raise ProteinActivationError(f"Invalid {label}: feature indices are not sorted")
        result.append(
            {
                int(feature): float(value)
                for feature, value in zip(
                    row_indices.tolist(), values[left:right].tolist(), strict=True
                )
            }
        )
    return result


def _decode_output_shard(
    payload: dict[str, object], *, start: int, end: int, width: int
) -> list[ProteinActivationRow]:
    expected_keys = {
        "accessions", "all_source_indices", "all_source_indptr", "all_source_values",
        "aspect_unit_counts", "eligibility", "end_protein", "schema_version",
        "source_excluded", "splits", "start_protein",
    }
    rows = end - start
    accessions = payload.get("accessions")
    splits = payload.get("splits")
    counts = payload.get("aspect_unit_counts")
    eligibility = payload.get("eligibility")
    excluded = payload.get("source_excluded")
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("start_protein") != start
        or payload.get("end_protein") != end
        or not isinstance(accessions, tuple)
        or len(accessions) != rows
        or any(type(value) is not str or not value for value in accessions)
        or not isinstance(splits, tuple)
        or len(splits) != rows
        or any(value not in {"train", "validation"} for value in splits)
        or not isinstance(counts, torch.Tensor)
        or counts.dtype != torch.int64
        or tuple(counts.shape) != (rows, 24)
        or bool(torch.any(counts < 0).item())
        or not isinstance(eligibility, dict)
        or set(eligibility) != set(_SOURCES)
        or not isinstance(excluded, dict)
        or set(excluded) != set(_SOURCES)
    ):
        raise ProteinActivationError("Protein activation shard has invalid schema")
    all_rows = _decode_csr(
        payload["all_source_indptr"], payload["all_source_indices"],
        payload["all_source_values"], rows, width, "all-source",
    )
    source_rows: dict[str, list[dict[int, float]]] = {}
    for source in _SOURCES:
        flags = eligibility[source]
        fields = excluded[source]
        if (
            not isinstance(flags, torch.Tensor)
            or flags.dtype != torch.bool
            or tuple(flags.shape) != (rows,)
            or not isinstance(fields, dict)
            or set(fields) != {"indices", "indptr", "values"}
        ):
            raise ProteinActivationError("Protein activation source-exclusion schema is invalid")
        source_rows[source] = _decode_csr(
            fields["indptr"], fields["indices"], fields["values"],
            rows, width, f"{source}-excluded",
        )
        for row in range(rows):
            if not bool(flags[row].item()) and source_rows[source][row]:
                raise ProteinActivationError("Ineligible source-excluded row contains values")
    result = []
    for row in range(rows):
        flags = {source: bool(eligibility[source][row].item()) for source in _SOURCES}
        result.append(
            _freeze_row(
                accessions[row],
                splits[row],
                [int(value) for value in counts[row].tolist()],
                all_rows[row],
                {source: source_rows[source][row] for source in _SOURCES if flags[source]},
                flags,
            )
        )
    return result


def _iter_registered_split(
    output_dir: Path,
    manifest: dict[str, object],
    split: str,
) -> Iterator[ProteinActivationRow]:
    width = manifest["width"]
    split_counts = manifest["split_counts"]
    shards = manifest["shards"]
    assert isinstance(width, int) and isinstance(split_counts, dict) and isinstance(shards, dict)
    count = split_counts[split]
    records = shards[split]
    if type(count) is not int or count < 0 or not isinstance(records, list):
        raise ProteinActivationError("Protein activation split manifest is invalid")
    next_protein = 0
    seen_accessions: set[str] = set()
    for number, metadata in enumerate(records):
        if not isinstance(metadata, dict):
            raise ProteinActivationError("Protein activation shard metadata is invalid")
        path = output_dir / f"{split}_shard_{number:05d}.pt"
        start = metadata.get("start_protein")
        end = metadata.get("end_protein")
        if (
            metadata.get("path") != path.name
            or type(start) is not int
            or type(end) is not int
            or start != next_protein
            or end <= start
            or end > count
            or metadata.get("proteins") != end - start
        ):
            raise ProteinActivationError("Protein activation shards are not contiguous")
        try:
            observed_sha = sha256_file(path)
            observed_bytes = path.stat().st_size
        except (AtlasIOError, OSError, ValueError) as error:
            raise _fail("registered protein activation shard", error) from error
        if metadata.get("sha256") != observed_sha:
            raise ProteinActivationError(f"Protein activation shard checksum mismatch: {path}")
        if metadata.get("bytes") != observed_bytes:
            raise ProteinActivationError(f"Protein activation shard byte size mismatch: {path}")
        decoded = _decode_output_shard(
            _load_torch(path, "protein activation shard"), start=start, end=end, width=width
        )
        for row in decoded:
            if row.split != split:
                raise ProteinActivationError("Protein activation row is registered in wrong split")
            if row.accession in seen_accessions:
                raise ProteinActivationError("Protein activation split has duplicate accessions")
            seen_accessions.add(row.accession)
            yield row
        next_protein = end
    if next_protein != count:
        raise ProteinActivationError("Protein activation split shards do not cover every protein")


def iter_protein_activation_rows(
    output_dir: Path,
    *,
    split: str | None = None,
) -> Iterator[ProteinActivationRow]:
    """Yield validated rows, opening only registered files for a selected split."""

    output_dir = Path(output_dir)
    if split is not None and (type(split) is not str or split not in {"train", "validation"}):
        raise TypeError("split must be None, 'train', or 'validation'")
    manifest = _load_json(output_dir / "manifest.json", "protein activation manifest")
    _validate_split_registration_schema(manifest)
    shards = manifest.get("shards")
    split_counts = manifest.get("split_counts")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or type(manifest.get("width")) is not int
        or manifest["width"] <= 0
        or type(manifest.get("accession_count")) is not int
        or manifest["accession_count"] <= 0
        or not isinstance(shards, dict)
        or set(shards) != {"train", "validation"}
        or not isinstance(split_counts, dict)
        or set(split_counts) != {"train", "validation"}
        or any(type(value) is not int or value < 0 for value in split_counts.values())
        or sum(split_counts.values()) != manifest["accession_count"]
    ):
        raise ProteinActivationError("Protein activation manifest has invalid split schema")
    if split is not None:
        yield from _iter_registered_split(output_dir, manifest, split)
        return
    train_rows = _iter_registered_split(output_dir, manifest, "train")
    validation_rows = _iter_registered_split(output_dir, manifest, "validation")
    seen_accessions: set[str] = set()
    observed = 0
    for rows in (train_rows, validation_rows):
        for row in rows:
            if row.accession in seen_accessions:
                raise ProteinActivationError("Protein activation views have duplicate accessions")
            seen_accessions.add(row.accession)
            observed += 1
            yield row
    if observed != manifest["accession_count"]:
        raise ProteinActivationError("Protein activation views do not cover every accession")


def _row_signature(row: ProteinActivationRow) -> tuple[object, ...]:
    return (
        row.accession,
        row.split,
        row.aspect_unit_counts,
        tuple(row.all_sources.items()),
        tuple((source, tuple(values.items())) for source, values in row.source_excluded.items()),
        tuple(row.eligibility.items()),
    )


def validate_protein_activations(
    units_path: Path,
    sparse_dir: Path,
    split_manifest_path: Path,
    filtered_uniprot_tsv: Path,
    output_dir: Path,
    *,
    concept_index_path: Path | None = None,
    expected_accession_count: int = 10_618,
    expected_split_counts: tuple[int, int] = (9_539, 1_079),
    atlas_paths: AtlasPaths | None = None,
    source_binding: AtlasSourceBinding | None = None,
    allow_test_fixture_unbound: bool = False,
    expected_width: int | None = None,
    expected_k: int | None = None,
    concept_snapshot_dir: Path | None = None,
    allow_test_fixture_concept_unbound: bool = False,
) -> dict[str, object]:
    """Fully validate provenance, sparse outputs, and recomputed protein maxima."""

    units_path = Path(units_path)
    sparse_dir = Path(sparse_dir)
    split_manifest_path = Path(split_manifest_path)
    filtered_uniprot_tsv = Path(filtered_uniprot_tsv)
    output_dir = Path(output_dir)
    concept = Path(concept_index_path) if concept_index_path is not None else None
    universe = {accession for accession, _ in _read_uniprot(filtered_uniprot_tsv)}
    _, concept_binding = _validate_concept_input(
        concept, units_path, universe, concept_snapshot_dir=concept_snapshot_dir,
        allow_test_fixture_concept_unbound=allow_test_fixture_concept_unbound,
    )
    expected, sparse_manifest, _ = _iter_computed_rows(
        units_path,
        sparse_dir,
        split_manifest_path,
        filtered_uniprot_tsv,
        expected_accession_count=expected_accession_count,
        expected_split_counts=expected_split_counts,
        atlas_paths=atlas_paths, source_binding=source_binding,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
        expected_width=expected_width, expected_k=expected_k,
    )
    _validate_task4_paths(
        units_path, sparse_dir, split_manifest_path, filtered_uniprot_tsv,
        atlas_paths=atlas_paths, allow_test_fixture_unbound=allow_test_fixture_unbound,
    )
    manifest = _load_json(output_dir / "manifest.json", "protein activation manifest")
    proteins_per_shard = manifest.get("proteins_per_shard")
    if type(proteins_per_shard) is not int or proteins_per_shard <= 0:
        raise ProteinActivationError("Protein activation manifest shard size is invalid")
    template = _manifest_template(
        accession_count=expected_accession_count,
        split_counts=expected_split_counts,
        units_path=units_path,
        sparse_dir=sparse_dir,
        split_manifest_path=split_manifest_path,
        filtered_uniprot_tsv=filtered_uniprot_tsv,
        concept_binding=concept_binding,
        proteins_per_shard=proteins_per_shard,
        width=int(sparse_manifest["width"]),
    )
    if manifest.get("complete") is not True:
        raise ProteinActivationError("Protein activation output is incomplete")
    _validate_split_registration_schema(manifest)
    if set(manifest) != set(template):
        raise ProteinActivationError("Protein activation manifest has invalid exact schema")
    for key, value in template.items():
        if key not in {"complete", "shards"} and manifest.get(key) != value:
            raise ProteinActivationError(f"Protein activation manifest mismatch for {key}")
    sentinel = object()
    for split in ("train", "validation"):
        if split == "validation":
            expected, _, _ = _iter_computed_rows(
                units_path,
                sparse_dir,
                split_manifest_path,
                filtered_uniprot_tsv,
                expected_accession_count=expected_accession_count,
                expected_split_counts=expected_split_counts,
                atlas_paths=atlas_paths, source_binding=source_binding,
                allow_test_fixture_unbound=allow_test_fixture_unbound,
                expected_width=expected_width, expected_k=expected_k,
            )
        wanted_rows = (row for row in expected if row.split == split)
        for observed, wanted in zip_longest(
            iter_protein_activation_rows(output_dir, split=split),
            wanted_rows,
            fillvalue=sentinel,
        ):
            if observed is sentinel or wanted is sentinel or _row_signature(
                observed  # type: ignore[arg-type]
            ) != _row_signature(wanted):  # type: ignore[arg-type]
                raise ProteinActivationError(
                    "Protein activation rows do not match recomputed maxima"
                )
    return manifest


def _covariate_rows_and_binding(
    filtered_uniprot_tsv: Path,
    units_path: Path,
    concept_index_path: Path | None,
    *,
    expected_accession_count: int,
    concept_snapshot_dir: Path | None,
    allow_test_fixture_concept_unbound: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if type(expected_accession_count) is not int or expected_accession_count <= 0:
        raise TypeError("expected_accession_count must be a positive integer")
    filtered_uniprot_tsv = Path(filtered_uniprot_tsv)
    units_path = Path(units_path)
    proteins = _read_uniprot(filtered_uniprot_tsv)
    if len(proteins) != expected_accession_count:
        raise ProteinActivationError(
            f"Observed {len(proteins)} proteins, expected accession count {expected_accession_count}"
        )
    _, accession_order, grouped = _validate_units(units_path)
    if set(accession_order) != {accession for accession, _ in proteins}:
        raise ProteinActivationError("SAE units do not cover exact covariate accessions")
    universe = {accession for accession, _ in proteins}
    concepts, concept_binding = _validate_concept_input(
        Path(concept_index_path) if concept_index_path is not None else None,
        units_path, universe, concept_snapshot_dir=concept_snapshot_dir,
        allow_test_fixture_concept_unbound=allow_test_fixture_concept_unbound,
    )
    annotations: dict[str, list[dict[str, object]]] = {
        accession: [] for accession, _ in proteins
    }
    for number, concept in enumerate(concepts):
        database = concept.get("database")
        concept_id = concept.get("concept_id")
        members = concept.get("protein_accessions")
        assert isinstance(database, str) and isinstance(concept_id, str)
        assert isinstance(members, list)
        normalized = concept
        for member in members:
            annotations[member].append(normalized)
    output_rows: list[dict[str, object]] = []
    for accession, sequence in proteins:
        counts = [0] * 24
        for unit in grouped[accession]:
            counts[int(unit["aspect_index"])] += 1
        source_availability = {
            source: any(
                count > 0 and aspect not in SOURCE_EXCLUDED_ASPECTS[source]
                for aspect, count in enumerate(counts)
            )
            for source in _SOURCES
        }
        protein_annotations = annotations[accession]
        database_counts = {
            source: sum(item["database"] == source for item in protein_annotations)
            for source in _SOURCES
        }
        interpro_families = sorted(
            str(item["concept_id"])
            for item in protein_annotations
            if item["database"] == "interpro" and item.get("concept_type") == "family"
        )
        output_rows.append(
            {
                "accession": accession,
                "annotation_covariates_available": concept_binding["available"],
                "aspect_present": [count > 0 for count in counts],
                "aspect_unit_counts": counts,
                "database_annotation_counts": database_counts,
                "interpro_family_indicators": interpro_families,
                "sequence_length": len(sequence),
                "source_availability": source_availability,
                "total_annotation_count": len(protein_annotations),
                "total_text_unit_count": sum(counts),
            }
        )
    return output_rows, concept_binding


def _covariate_manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(".manifest.json")


def _covariate_manifest_template(
    output_path: Path,
    filtered_uniprot_tsv: Path,
    units_path: Path,
    concept_binding: dict[str, object],
    accession_count: int,
    *,
    registered_output_path: Path | None = None,
) -> dict[str, object]:
    registered_path = output_path if registered_output_path is None else registered_output_path
    return {
        "accession_count": accession_count,
        "complete": True,
        "concept_binding": concept_binding,
        "concept_index_sha256": concept_binding["index_sha256"],
        "filtered_uniprot_sha256": sha256_file(filtered_uniprot_tsv),
        "output": {
            "bytes": output_path.stat().st_size,
            "path": str(registered_path.resolve()),
            "sha256": sha256_file(output_path),
        },
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
        "schema_version": SCHEMA_VERSION,
        "units_sha256": sha256_file(units_path),
    }


def validate_protein_covariates(
    filtered_uniprot_tsv: Path,
    units_path: Path,
    concept_index_path: Path | None,
    output_path: Path,
    *,
    expected_accession_count: int = 10_618,
    concept_snapshot_dir: Path | None = None,
    allow_test_fixture_concept_unbound: bool = False,
) -> dict[str, object]:
    """Validate the registered covariate stage and recompute every row."""

    filtered_uniprot_tsv, units_path, output_path = map(
        Path, (filtered_uniprot_tsv, units_path, output_path)
    )
    rows, concept_binding = _covariate_rows_and_binding(
        filtered_uniprot_tsv, units_path, concept_index_path,
        expected_accession_count=expected_accession_count,
        concept_snapshot_dir=concept_snapshot_dir,
        allow_test_fixture_concept_unbound=allow_test_fixture_concept_unbound,
    )
    manifest = _load_json(_covariate_manifest_path(output_path), "covariate manifest")
    try:
        expected = _covariate_manifest_template(
            output_path, filtered_uniprot_tsv, units_path, concept_binding, len(rows)
        )
    except (AtlasIOError, OSError, ValueError) as error:
        raise _fail("registered covariate output checksum", error) from error
    if manifest != expected:
        raise ProteinActivationError("Covariate manifest input binding or checksum mismatch")
    observed = _load_jsonl(output_path, "covariate output")
    if observed != rows:
        raise ProteinActivationError("Covariate rows do not match recomputed values")
    return manifest


def build_protein_covariates(
    filtered_uniprot_tsv: Path,
    units_path: Path,
    concept_index_path: Path | None,
    output_path: Path,
    *,
    expected_accession_count: int = 10_618,
    concept_snapshot_dir: Path | None = None,
    allow_test_fixture_concept_unbound: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build an atomic, registered per-protein covariate stage."""

    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    filtered_uniprot_tsv, units_path, output_path = map(
        Path, (filtered_uniprot_tsv, units_path, output_path)
    )
    manifest_path = _covariate_manifest_path(output_path)
    if output_path.exists() or manifest_path.exists():
        if not overwrite:
            return validate_protein_covariates(
                filtered_uniprot_tsv, units_path, concept_index_path, output_path,
                expected_accession_count=expected_accession_count,
                concept_snapshot_dir=concept_snapshot_dir,
                allow_test_fixture_concept_unbound=allow_test_fixture_concept_unbound,
            )
        if output_path.is_symlink() or manifest_path.is_symlink():
            raise ProteinActivationError("Refusing to overwrite symlinked covariate output")
    output_rows, concept_binding = _covariate_rows_and_binding(
        filtered_uniprot_tsv, units_path, concept_index_path,
        expected_accession_count=expected_accession_count,
        concept_snapshot_dir=concept_snapshot_dir,
        allow_test_fixture_concept_unbound=allow_test_fixture_concept_unbound,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output_path.name}.covariate-stage-", dir=output_path.parent
    ))
    staged_output = stage / output_path.name
    staged_manifest = stage / manifest_path.name
    backup: Path | None = None
    old_exists = (output_path.exists(), manifest_path.exists())
    try:
        write_jsonl_atomic(staged_output, output_rows)
        manifest = _covariate_manifest_template(
            staged_output, filtered_uniprot_tsv, units_path,
            concept_binding, len(output_rows),
            registered_output_path=output_path,
        )
        write_json_atomic(staged_manifest, manifest)
        if (
            _load_json(staged_manifest, "staged covariate manifest") != manifest
            or _load_jsonl(staged_output, "staged covariate output") != output_rows
        ):
            raise ProteinActivationError("Staged covariate replacement validation failed")

        backup = Path(tempfile.mkdtemp(
            prefix=f".{output_path.name}.covariate-backup-", dir=output_path.parent
        ))
        if old_exists[0]:
            os.replace(output_path, backup / output_path.name)
        if old_exists[1]:
            os.replace(manifest_path, backup / manifest_path.name)
        os.replace(staged_output, output_path)
        # The manifest is the completion marker and is always committed last.
        os.replace(staged_manifest, manifest_path)
        validated = validate_protein_covariates(
            filtered_uniprot_tsv, units_path, concept_index_path, output_path,
            expected_accession_count=expected_accession_count,
            concept_snapshot_dir=concept_snapshot_dir,
            allow_test_fixture_concept_unbound=allow_test_fixture_concept_unbound,
        )
    except BaseException as error:
        rollback_errors: list[str] = []
        if backup is not None:
            for destination, existed in (
                (output_path, old_exists[0]), (manifest_path, old_exists[1])
            ):
                saved = backup / destination.name
                try:
                    if saved.exists():
                        if destination.exists():
                            os.replace(destination, stage / f"failed-new-{destination.name}")
                        os.replace(saved, destination)
                    elif not existed and destination.exists():
                        os.replace(destination, stage / f"failed-new-{destination.name}")
                except BaseException as rollback_error:
                    rollback_errors.append(f"cannot restore {destination.name}: {rollback_error}")
        if rollback_errors:
            raise ProteinActivationError(
                "Covariate publication recovery required; "
                f"backup recovery path: {backup}; staging recovery path: {stage}; "
                + "; ".join(rollback_errors)
            ) from error
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(error, ProteinActivationError):
            raise
        raise _fail("protein covariate output", error) from error
    shutil.rmtree(stage, ignore_errors=True)
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
    return validated
