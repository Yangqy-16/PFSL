#!/usr/bin/env python3
"""Build deterministic 24-aspect and fine-grained SAE text corpora."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pfsl_process.sae.download_uniprot_human import read_excluded_accessions
    from pfsl_process.sae.text_templates import (
        ASPECT_NAMES,
        build_aspect_texts,
        make_unit_id,
    )
else:
    from .download_uniprot_human import read_excluded_accessions
    from .text_templates import ASPECT_NAMES, build_aspect_texts, make_unit_id


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = "1.0.0"
ASPECTS_FILE = "protein_aspects.jsonl"
UNITS_FILE = "sae_units.jsonl"
MANIFEST_FILE = "text_corpus_manifest.json"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_TSV = (
    PROJECT_ROOT
    / "data/sae_corpus/uniprot_human_reviewed/processed/"
    "uniprotkb_human_reviewed_non_sl.tsv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_TSV.parent
DEFAULT_EXCLUSION_TABLE = (
    PROJECT_ROOT / "data/prot_func/idmapping_reviewed_true_2026_01_09.tsv"
)
DEFAULT_EC_MAPPING = Path(
    "/data/qingyuyang/prot_design/ProTrek/faiss_index/SwissProt/"
    "ProTrek_650M_UniRef50/text/subsections/Enzyme_commission_number_ids.tsv"
)


class CorpusBuildError(ValueError):
    """Raised when source or generated corpus data violates the schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _temporary_path(parent: Path, name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, path = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=parent)
    os.close(descriptor)
    return Path(path)


def _json_line(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_ec_descriptions(path: Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"EC mapping does not exist: {path}")
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            key, separator, value = line.partition(": ")
            if not separator or not key.startswith("EC ") or not value.strip():
                raise CorpusBuildError(
                    f"Invalid EC mapping at line {line_number}: {path}"
                )
            result[key.strip()] = value.strip()
    if not result:
        raise CorpusBuildError(f"EC mapping is empty: {path}")
    return result


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = _temporary_path(path.parent, path.name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_corpus(
    input_tsv: Path,
    output_dir: Path,
    exclusion_table: Path,
    ec_mapping: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    input_tsv = Path(input_tsv)
    output_dir = Path(output_dir)
    exclusion_table = Path(exclusion_table)
    ec_mapping = Path(ec_mapping)
    if not input_tsv.is_file():
        raise FileNotFoundError(f"Input TSV does not exist: {input_tsv}")

    aspects_path = output_dir / ASPECTS_FILE
    units_path = output_dir / UNITS_FILE
    manifest_path = output_dir / MANIFEST_FILE
    existing = [path for path in (aspects_path, units_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Corpus output already exists; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    excluded = read_excluded_accessions(exclusion_table)
    ec_descriptions = load_ec_descriptions(ec_mapping)
    aspects_temporary = _temporary_path(output_dir, ASPECTS_FILE)
    units_temporary = _temporary_path(output_dir, UNITS_FILE)
    accessions: set[str] = set()
    unit_ids: set[str] = set()
    protein_count = 0
    unit_count = 0
    try:
        with input_tsv.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if not reader.fieldnames or "Entry" not in reader.fieldnames:
                raise CorpusBuildError("Input TSV is missing required 'Entry' column")
            with aspects_temporary.open("w", encoding="utf-8", newline="") as aspects_out:
                with units_temporary.open("w", encoding="utf-8", newline="") as units_out:
                    for line_number, row in enumerate(reader, start=2):
                        accession = (row.get("Entry") or "").strip()
                        if not accession:
                            raise CorpusBuildError(
                                f"Empty accession at input line {line_number}"
                            )
                        if accession in excluded:
                            raise CorpusBuildError(
                                f"Input contains excluded accession {accession!r}"
                            )
                        if accession in accessions:
                            raise CorpusBuildError(
                                f"Duplicate accession {accession!r} at input line {line_number}"
                            )

                        aspects = build_aspect_texts(row, ec_descriptions)
                        if len(aspects) != len(ASPECT_NAMES):
                            raise CorpusBuildError(
                                f"Accession {accession!r} did not produce 24 aspects"
                            )
                        aspect_payload = {
                            "accession": accession,
                            "aspects": [
                                {
                                    "aspect_index": aspect.index,
                                    "aspect_name": aspect.name,
                                    "text": aspect.text,
                                }
                                for aspect in aspects
                            ],
                        }
                        aspects_out.write(_json_line(aspect_payload) + "\n")

                        for aspect in aspects:
                            for unit_index, unit in enumerate(aspect.units):
                                text = unit.text.strip()
                                if not text:
                                    raise CorpusBuildError(
                                        f"Empty unit for accession {accession!r}, aspect {aspect.index}"
                                    )
                                unit_id = make_unit_id(
                                    accession,
                                    aspect.index,
                                    unit.unit_type,
                                    unit_index,
                                )
                                if unit_id in unit_ids:
                                    raise CorpusBuildError(f"Duplicate unit ID {unit_id!r}")
                                unit_payload = {
                                    "unit_id": unit_id,
                                    "accession": accession,
                                    "aspect_index": aspect.index,
                                    "aspect_name": aspect.name,
                                    "unit_type": unit.unit_type,
                                    "unit_index": unit_index,
                                    "text": text,
                                    "parent_aspect_text": aspect.text,
                                }
                                units_out.write(_json_line(unit_payload) + "\n")
                                unit_ids.add(unit_id)
                                unit_count += 1
                        accessions.add(accession)
                        protein_count += 1

        if protein_count == 0:
            raise CorpusBuildError("Input TSV contains no proteins")
        if unit_count == 0:
            raise CorpusBuildError("Text transformation produced no SAE units")

        os.replace(aspects_temporary, aspects_path)
        os.replace(units_temporary, units_path)
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "aspect_names": list(ASPECT_NAMES),
            "counts": {
                "proteins": protein_count,
                "aspects": protein_count * len(ASPECT_NAMES),
                "units": unit_count,
                "excluded_accessions": len(excluded),
            },
            "sources": {
                "input_tsv": _file_metadata(input_tsv),
                "exclusion_table": _file_metadata(exclusion_table),
                "ec_mapping": _file_metadata(ec_mapping),
            },
            "files": {
                "protein_aspects": _file_metadata(aspects_path),
                "sae_units": _file_metadata(units_path),
            },
        }
        _write_json_atomic(manifest_path, manifest)
        return manifest
    finally:
        aspects_temporary.unlink(missing_ok=True)
        units_temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise CorpusBuildError(
                    f"Invalid JSON at line {line_number}: {path}"
                ) from error
            if not isinstance(payload, dict):
                raise CorpusBuildError(f"JSONL row is not an object at line {line_number}")
            yield payload


def validate_corpus(
    output_dir: Path,
    input_tsv: Path,
    exclusion_table: Path,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    input_tsv = Path(input_tsv)
    exclusion_table = Path(exclusion_table)
    aspects_path = output_dir / ASPECTS_FILE
    units_path = output_dir / UNITS_FILE
    manifest_path = output_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Corpus manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_input = manifest["sources"]["input_tsv"]
    expected_exclusion = manifest["sources"]["exclusion_table"]
    if expected_input["sha256"] != _sha256(input_tsv):
        raise CorpusBuildError("Input TSV checksum does not match manifest")
    if expected_exclusion["sha256"] != _sha256(exclusion_table):
        raise CorpusBuildError("Exclusion table checksum does not match manifest")
    for label, path in (("protein_aspects", aspects_path), ("sae_units", units_path)):
        expected = manifest["files"][label]
        if expected["sha256"] != _sha256(path):
            raise CorpusBuildError(f"{label} checksum does not match manifest")
        if expected["bytes"] != path.stat().st_size:
            raise CorpusBuildError(f"{label} byte size does not match manifest")

    excluded = read_excluded_accessions(exclusion_table)
    accessions: set[str] = set()
    aspect_rows = 0
    for payload in _read_jsonl(aspects_path):
        accession = str(payload.get("accession", ""))
        if accession in excluded:
            raise CorpusBuildError(f"Corpus contains excluded accession {accession!r}")
        if accession in accessions:
            raise CorpusBuildError(f"Duplicate aspect accession {accession!r}")
        aspects = payload.get("aspects")
        if not isinstance(aspects, list) or len(aspects) != len(ASPECT_NAMES):
            raise CorpusBuildError(f"Accession {accession!r} does not have 24 aspects")
        accessions.add(accession)
        aspect_rows += 1

    unit_ids: set[str] = set()
    unit_rows = 0
    for payload in _read_jsonl(units_path):
        unit_id = str(payload.get("unit_id", ""))
        accession = str(payload.get("accession", ""))
        text = str(payload.get("text", "")).strip()
        if accession in excluded:
            raise CorpusBuildError(f"Unit contains excluded accession {accession!r}")
        if not text:
            raise CorpusBuildError(f"Empty unit text for {unit_id!r}")
        if unit_id in unit_ids:
            raise CorpusBuildError(f"Duplicate unit ID {unit_id!r}")
        unit_ids.add(unit_id)
        unit_rows += 1

    counts = manifest["counts"]
    if aspect_rows != counts["proteins"]:
        raise CorpusBuildError("Protein count does not match manifest")
    if unit_rows != counts["units"]:
        raise CorpusBuildError("Unit count does not match manifest")
    if counts["aspects"] != aspect_rows * len(ASPECT_NAMES):
        raise CorpusBuildError("Aspect count does not match manifest")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, default=DEFAULT_INPUT_TSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exclusion-table", type=Path, default=DEFAULT_EXCLUSION_TABLE
    )
    parser.add_argument("--ec-mapping", type=Path, default=DEFAULT_EC_MAPPING)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.validate_only:
        manifest = validate_corpus(
            args.output_dir,
            args.input_tsv,
            args.exclusion_table,
        )
    else:
        manifest = build_corpus(
            args.input_tsv,
            args.output_dir,
            args.exclusion_table,
            args.ec_mapping,
            overwrite=args.overwrite,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
