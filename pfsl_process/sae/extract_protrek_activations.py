#!/usr/bin/env python3
"""Extract resumable, validated ProTrek activation shards from SAE units."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Protocol, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pfsl_process.sae.protrek_inference import (
        DEFAULT_PROTREK_REPO,
        DEFAULT_PROTREK_WEIGHTS,
        NORM_TOLERANCE,
        REPRESENTATION_DIM,
        load_text_embedder,
    )
else:
    from .protrek_inference import (
        DEFAULT_PROTREK_REPO,
        DEFAULT_PROTREK_WEIGHTS,
        NORM_TOLERANCE,
        REPRESENTATION_DIM,
        load_text_embedder,
    )


SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
DEFAULT_BATCH_SIZE = 64
DEFAULT_SHARD_SIZE = 10_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNITS_PATH = (
    PROJECT_ROOT
    / "data/sae_corpus/uniprot_human_reviewed/processed/sae_units.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/sae_corpus/uniprot_human_reviewed/activations/protrek_final_1024"
)


class TextEmbedder(Protocol):
    def encode(self, texts: Sequence[str], batch_size: int) -> torch.Tensor:
        ...


class ActivationCacheError(ValueError):
    """Raised when activation inputs, outputs, or metadata are invalid."""


def _embedder_provenance(embedder: TextEmbedder) -> dict[str, object]:
    provenance = getattr(embedder, "provenance", None)
    if provenance is None:
        provenance = {
            "protrek": {
                "checkpoint": "unknown",
                "protein_config": "unknown",
                "text_config": "unknown",
                "structure_config": "unknown",
            },
            "representation": {
                "method": "ProTrekTrimodalModel.get_text_repr",
                "dimension": REPRESENTATION_DIM,
            },
            "environment": {
                "python_executable": str(Path(sys.executable).resolve()),
                "python_prefix": str(Path(sys.prefix).resolve()),
                "python_version": sys.version.split()[0],
                "torch_version": str(torch.__version__),
            },
            "device": str(getattr(embedder, "device", "unknown")),
        }
    if not isinstance(provenance, dict):
        raise ActivationCacheError("Embedder provenance must be a mapping")
    try:
        snapshot = json.loads(json.dumps(provenance, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ActivationCacheError("Embedder provenance must be JSON serializable") from error
    return snapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_metadata(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            rows += 1
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "rows": rows,
    }


def _temporary_path(parent: Path, name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{name}.",
        suffix=".tmp",
        dir=parent,
    )
    os.close(descriptor)
    return Path(raw_path)


def _write_manifest_atomic(path: Path, manifest: dict[str, object]) -> None:
    temporary = _temporary_path(path.parent, path.name)
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ActivationCacheError(f"Invalid activation manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise ActivationCacheError(f"Activation manifest is not an object: {path}")
    return manifest


def _registered_path(output_dir: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ActivationCacheError("Manifest shard path must be a non-empty string")
    candidate = Path(relative_path)
    if candidate.is_absolute() or candidate.name != relative_path:
        raise ActivationCacheError(
            f"Manifest shard path must be a file name: {relative_path!r}"
        )
    return output_dir / candidate


def _load_shard(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception as error:
        raise ActivationCacheError(f"Cannot load activation shard: {path}") from error
    if not isinstance(payload, dict):
        raise ActivationCacheError(f"Activation shard is not a mapping: {path}")
    return payload


def _validate_embeddings(
    embeddings: object,
    expected_rows: int,
    label: str,
) -> torch.Tensor:
    if not isinstance(embeddings, torch.Tensor):
        raise ActivationCacheError(f"{label} embeddings are not a tensor")
    expected_shape = (expected_rows, REPRESENTATION_DIM)
    if embeddings.ndim != 2 or tuple(embeddings.shape) != expected_shape:
        raise ActivationCacheError(
            f"{label} embeddings must have shape {expected_shape}, "
            f"got {tuple(embeddings.shape)}"
        )
    if embeddings.dtype != torch.float32:
        raise ActivationCacheError(
            f"{label} embeddings must be float32, got {embeddings.dtype}"
        )
    if not torch.isfinite(embeddings).all():
        raise ActivationCacheError(f"{label} embeddings contain non-finite values")
    norms = embeddings.norm(dim=-1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=NORM_TOLERANCE,
        rtol=0.0,
    ):
        raise ActivationCacheError(f"{label} embeddings are not L2 normalized")
    return embeddings


def _validate_input_metadata(
    expected: object,
    observed: dict[str, object],
) -> None:
    if not isinstance(expected, dict):
        raise ActivationCacheError("Manifest input metadata is invalid")
    if expected.get("sha256") != observed["sha256"]:
        raise ActivationCacheError("Input checksum does not match manifest")
    if expected.get("bytes") != observed["bytes"]:
        raise ActivationCacheError("Input byte size does not match manifest")
    if expected.get("rows") != observed["rows"]:
        raise ActivationCacheError("Input row count does not match manifest")


def _validate_manifest_binding(manifest: dict[str, object]) -> None:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ActivationCacheError("Manifest provenance is invalid")
    protrek = provenance.get("protrek")
    required_paths = (
        "checkpoint",
        "protein_config",
        "text_config",
        "structure_config",
    )
    if not isinstance(protrek, dict) or any(
        not isinstance(protrek.get(key), str) or not protrek[key]
        for key in required_paths
    ):
        raise ActivationCacheError("Manifest ProTrek provenance is invalid")
    representation = provenance.get("representation")
    if not isinstance(representation, dict) or representation != {
        "method": "ProTrekTrimodalModel.get_text_repr",
        "dimension": REPRESENTATION_DIM,
    }:
        raise ActivationCacheError("Manifest representation provenance is invalid")
    environment = provenance.get("environment")
    required_environment = (
        "python_executable",
        "python_prefix",
        "python_version",
        "torch_version",
    )
    if not isinstance(environment, dict) or any(
        not isinstance(environment.get(key), str) or not environment[key]
        for key in required_environment
    ):
        raise ActivationCacheError("Manifest environment provenance is invalid")
    if not isinstance(provenance.get("device"), str) or not provenance["device"]:
        raise ActivationCacheError("Manifest device provenance is invalid")
    if not isinstance(manifest.get("batch_size"), int) or manifest["batch_size"] <= 0:
        raise ActivationCacheError("Manifest batch_size is invalid")
    if not isinstance(manifest.get("shard_size"), int) or manifest["shard_size"] <= 0:
        raise ActivationCacheError("Manifest shard_size is invalid")


def _validate_registered_shards(
    manifest: dict[str, object],
    output_dir: Path,
    units_path: Path,
) -> int:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ActivationCacheError("Unsupported activation manifest schema")
    if manifest.get("embedding_dim") != REPRESENTATION_DIM:
        raise ActivationCacheError("Manifest embedding dimension is invalid")
    if manifest.get("dtype") != "float32":
        raise ActivationCacheError("Manifest activation dtype is invalid")
    _validate_manifest_binding(manifest)
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ActivationCacheError("Manifest shards must be a list")
    input_metadata = manifest.get("input")
    if not isinstance(input_metadata, dict) or not isinstance(
        input_metadata.get("rows"), int
    ):
        raise ActivationCacheError("Manifest input row count is invalid")
    total_rows = input_metadata["rows"]
    next_row = 0
    with units_path.open("r", encoding="utf-8") as units_handle:
        for shard_index, metadata in enumerate(shards):
            if not isinstance(metadata, dict):
                raise ActivationCacheError("Manifest shard metadata is invalid")
            start_row = metadata.get("start_row")
            end_row = metadata.get("end_row")
            if (
                not isinstance(start_row, int)
                or not isinstance(end_row, int)
                or start_row != next_row
                or end_row <= start_row
                or end_row > total_rows
            ):
                raise ActivationCacheError(
                    f"Manifest shard {shard_index} has invalid row range"
                )
            expected_rows = end_row - start_row
            if metadata.get("shape") != [expected_rows, REPRESENTATION_DIM]:
                raise ActivationCacheError(
                    f"Manifest shard {shard_index} shape is invalid"
                )
            if metadata.get("dtype") != "float32":
                raise ActivationCacheError(
                    f"Manifest shard {shard_index} dtype is invalid"
                )
            shard_path = _registered_path(output_dir, metadata.get("path"))
            if not shard_path.is_file():
                raise ActivationCacheError(
                    f"Activation shard does not exist: {shard_path}"
                )
            if metadata.get("sha256") != _sha256(shard_path):
                raise ActivationCacheError(
                    f"Activation shard checksum does not match manifest: {shard_path}"
                )
            if metadata.get("bytes") != shard_path.stat().st_size:
                raise ActivationCacheError(
                    f"Activation shard byte size does not match manifest: {shard_path}"
                )
            payload = _load_shard(shard_path)
            unit_ids = payload.get("unit_ids")
            if (
                not isinstance(unit_ids, list)
                or len(unit_ids) != expected_rows
                or any(
                    not isinstance(unit_id, str) or not unit_id
                    for unit_id in unit_ids
                )
            ):
                raise ActivationCacheError(f"Invalid unit IDs in shard: {shard_path}")
            expected_unit_ids: list[str] = []
            for row_index in range(start_row, end_row):
                raw_line = units_handle.readline()
                if not raw_line:
                    raise ActivationCacheError(
                        f"SAE units source ended before row {row_index}"
                    )
                try:
                    source_row = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ActivationCacheError(
                        f"Invalid JSON at input row {row_index}: {units_path}"
                    ) from error
                source_unit_id = (
                    source_row.get("unit_id")
                    if isinstance(source_row, dict)
                    else None
                )
                if not isinstance(source_unit_id, str) or not source_unit_id:
                    raise ActivationCacheError(
                        f"Input row {row_index} has an invalid unit_id"
                    )
                expected_unit_ids.append(source_unit_id)
            if unit_ids != expected_unit_ids:
                raise ActivationCacheError(
                    f"Shard unit IDs do not match SAE source order: {shard_path}"
                )
            _validate_embeddings(
                payload.get("embeddings"), expected_rows, str(shard_path)
            )
            next_row = end_row
    if manifest.get("complete") and next_row != total_rows:
        raise ActivationCacheError("Complete manifest does not cover every input row")
    return next_row


def _read_unit_groups(
    units_path: Path,
    start_row: int,
    shard_size: int,
) -> Iterator[tuple[int, list[str], list[str]]]:
    group_start = start_row
    unit_ids: list[str] = []
    texts: list[str] = []
    with units_path.open("r", encoding="utf-8") as handle:
        for row_index, raw_line in enumerate(handle):
            if row_index < start_row:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ActivationCacheError(
                    f"Invalid JSON at input row {row_index}: {units_path}"
                ) from error
            if not isinstance(payload, dict):
                raise ActivationCacheError(
                    f"Input row {row_index} is not a JSON object"
                )
            unit_id = payload.get("unit_id")
            text = payload.get("text")
            if not isinstance(unit_id, str) or not unit_id:
                raise ActivationCacheError(
                    f"Input row {row_index} has an invalid unit_id"
                )
            if not isinstance(text, str) or not text.strip():
                raise ActivationCacheError(f"Input row {row_index} has empty text")
            unit_ids.append(unit_id)
            texts.append(text)
            if len(unit_ids) == shard_size:
                yield group_start, unit_ids, texts
                group_start += len(unit_ids)
                unit_ids = []
                texts = []
    if unit_ids:
        yield group_start, unit_ids, texts


def _encode_group(
    embedder: TextEmbedder,
    texts: list[str],
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        output = embedder.encode(batch, batch_size=batch_size)
        output = _validate_embeddings(output, len(batch), "Embedder output")
        outputs.append(output.detach().cpu())
    embeddings = torch.cat(outputs, dim=0)
    return _validate_embeddings(embeddings, len(texts), "Shard")


def _remove_registered_outputs(
    manifest: dict[str, object],
    output_dir: Path,
    manifest_path: Path,
) -> None:
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ActivationCacheError("Manifest shards must be a list")
    registered: list[Path] = []
    for metadata in shards:
        if not isinstance(metadata, dict):
            raise ActivationCacheError("Manifest shard metadata is invalid")
        registered.append(_registered_path(output_dir, metadata.get("path")))
    for path in registered:
        path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)


def validate_activation_cache(
    units_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    units_path = Path(units_path)
    output_dir = Path(output_dir)
    if not units_path.is_file():
        raise FileNotFoundError(f"SAE units file does not exist: {units_path}")
    manifest_path = output_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Activation manifest does not exist: {manifest_path}")
    observed_input = _input_metadata(units_path)
    manifest = _load_manifest(manifest_path)
    _validate_input_metadata(manifest.get("input"), observed_input)
    if not manifest.get("complete"):
        raise ActivationCacheError("Activation cache is incomplete")
    _validate_registered_shards(manifest, output_dir, units_path)
    return manifest


def extract_activations(
    units_path: Path,
    output_dir: Path,
    embedder: TextEmbedder,
    batch_size: int,
    shard_size: int,
    overwrite: bool = False,
) -> dict[str, object]:
    units_path = Path(units_path)
    output_dir = Path(output_dir)
    if not units_path.is_file():
        raise FileNotFoundError(f"SAE units file does not exist: {units_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    provenance = _embedder_provenance(embedder)
    observed_input = _input_metadata(units_path)
    if observed_input["rows"] == 0:
        raise ActivationCacheError("SAE units file is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILE

    manifest: dict[str, object]
    start_row = 0
    resuming = False
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        if overwrite:
            _remove_registered_outputs(manifest, output_dir, manifest_path)
        else:
            _validate_input_metadata(manifest.get("input"), observed_input)
            _validate_manifest_binding(manifest)
            if manifest.get("provenance") != provenance:
                raise ActivationCacheError("Embedder provenance does not match manifest")
            if manifest.get("batch_size") != batch_size:
                raise ActivationCacheError("batch_size does not match manifest")
            if manifest.get("shard_size") != shard_size:
                raise ActivationCacheError("shard_size does not match manifest")
            start_row = _validate_registered_shards(manifest, output_dir, units_path)
            if manifest.get("complete"):
                raise FileExistsError(
                    f"Activation cache already exists; use --overwrite: {output_dir}"
                )
            resuming = True
    elif overwrite:
        manifest = {}

    if not manifest_path.exists():
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "complete": False,
            "input": observed_input,
            "provenance": provenance,
            "embedding_dim": REPRESENTATION_DIM,
            "dtype": "float32",
            "batch_size": batch_size,
            "shard_size": shard_size,
            "shards": [],
        }
        _write_manifest_atomic(manifest_path, manifest)

    shards = manifest["shards"]
    if not isinstance(shards, list):
        raise ActivationCacheError("Manifest shards must be a list")
    for group_start, unit_ids, texts in _read_unit_groups(
        units_path,
        start_row,
        shard_size,
    ):
        shard_index = len(shards)
        shard_name = f"shard_{shard_index:05d}.pt"
        shard_path = output_dir / shard_name
        if shard_path.exists():
            if not resuming:
                raise FileExistsError(
                    f"Refusing to overwrite unregistered file: {shard_path}"
                )
            payload = _load_shard(shard_path)
            if payload.get("unit_ids") != unit_ids:
                raise ActivationCacheError(
                    f"Unregistered shard unit IDs do not match input: {shard_path}"
                )
            _validate_embeddings(
                payload.get("embeddings"),
                len(unit_ids),
                str(shard_path),
            )
        else:
            embeddings = _encode_group(embedder, texts, batch_size)
            temporary = _temporary_path(output_dir, shard_name)
            try:
                torch.save(
                    {"embeddings": embeddings, "unit_ids": unit_ids},
                    temporary,
                )
                os.replace(temporary, shard_path)
            finally:
                temporary.unlink(missing_ok=True)
        metadata = {
            "start_row": group_start,
            "end_row": group_start + len(unit_ids),
            "shape": [len(unit_ids), REPRESENTATION_DIM],
            "dtype": "float32",
            "path": shard_name,
            "bytes": shard_path.stat().st_size,
            "sha256": _sha256(shard_path),
        }
        shards.append(metadata)
        _write_manifest_atomic(manifest_path, manifest)

    covered_rows = _validate_registered_shards(manifest, output_dir, units_path)
    if covered_rows != observed_input["rows"]:
        raise ActivationCacheError("Extraction did not cover every input row")
    manifest["complete"] = True
    _write_manifest_atomic(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units-path", type=Path, default=DEFAULT_UNITS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_PROTREK_REPO)
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_PROTREK_WEIGHTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.validate_only:
        manifest = validate_activation_cache(args.units_path, args.output_dir)
    else:
        embedder, _ = load_text_embedder(
            device=args.device,
            repo_root=args.repo_root,
            weights_root=args.weights_root,
        )
        manifest = extract_activations(
            args.units_path,
            args.output_dir,
            embedder,
            args.batch_size,
            args.shard_size,
            overwrite=args.overwrite,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
