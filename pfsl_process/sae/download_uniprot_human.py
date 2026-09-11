#!/usr/bin/env python3
"""Download reviewed human UniProtKB records for leakage-free SAE training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = "1.0.0"
UNIPROT_STREAM_ENDPOINT = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_QUERY = "(reviewed:true) AND (organism_id:9606)"
UNIPROT_FIELDS = (
    "accession",
    "id",
    "reviewed",
    "protein_name",
    "gene_names",
    "xref_geneid",
    "organism_name",
    "sequence",
    "cc_activity_regulation",
    "cc_developmental_stage",
    "cc_induction",
    "go_p",
    "go_c",
    "go_f",
    "cc_disease",
    "cc_subcellular_location",
    "cc_domain",
    "cc_pathway",
    "cc_cofactor",
    "cc_catalytic_activity",
    "ec",
    "cc_caution",
    "cc_function",
    "cc_miscellaneous",
    "cc_polymorphism",
    "cc_ptm",
    "xref_proteomes",
    "cc_similarity",
    "cc_subunit",
    "cc_tissue_specificity",
)
EXPECTED_HEADERS = (
    "Entry",
    "Entry Name",
    "Reviewed",
    "Protein names",
    "Gene Names",
    "GeneID",
    "Organism",
    "Sequence",
    "Activity regulation",
    "Developmental stage",
    "Induction",
    "Gene Ontology (biological process)",
    "Gene Ontology (cellular component)",
    "Gene Ontology (molecular function)",
    "Involvement in disease",
    "Subcellular location [CC]",
    "Domain [CC]",
    "Pathway",
    "Cofactor",
    "Catalytic activity",
    "EC number",
    "Caution",
    "Function [CC]",
    "Miscellaneous [CC]",
    "Polymorphism",
    "Post-translational modification",
    "Proteomes",
    "Sequence similarities",
    "Subunit structure",
    "Tissue specificity",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/sae_corpus/uniprot_human_reviewed"
DEFAULT_EXCLUSION_TABLE = (
    PROJECT_ROOT / "data/prot_func/idmapping_reviewed_true_2026_01_09.tsv"
)
RAW_RELATIVE_PATH = Path("raw/uniprotkb_human_reviewed.tsv")
PROCESSED_RELATIVE_PATH = Path(
    "processed/uniprotkb_human_reviewed_non_sl.tsv"
)
MANIFEST_NAME = "manifest.json"


class CorpusValidationError(ValueError):
    """Raised when a downloaded or processed corpus violates its contract."""


@dataclass(frozen=True)
class CorpusStats:
    rows: int
    unique_accessions: int


@dataclass(frozen=True)
class FilterStats:
    raw_rows: int
    excluded_rows: int
    retained_rows: int


def build_request_url(endpoint: str = UNIPROT_STREAM_ENDPOINT) -> str:
    """Build the complete UniProt streaming request URL."""
    query = urlencode(
        {
            "query": UNIPROT_QUERY,
            "format": "tsv",
            "fields": ",".join(UNIPROT_FIELDS),
        }
    )
    separator = "&" if "?" in endpoint else "?"
    return f"{endpoint}{separator}{query}"


def read_excluded_accessions(path: Path) -> set[str]:
    """Read the conservative exclusion set from the existing mapping table."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Exclusion table does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "Entry" not in reader.fieldnames:
            raise CorpusValidationError(
                f"Exclusion table is missing required 'Entry' column: {path}"
            )
        accessions = {
            row["Entry"].strip()
            for row in reader
            if row.get("Entry") and row["Entry"].strip()
        }

    if not accessions:
        raise CorpusValidationError(f"Exclusion table contains no accessions: {path}")
    return accessions


def _validate_headers(fieldnames: Iterable[str] | None, path: Path) -> tuple[str, ...]:
    if fieldnames is None:
        raise CorpusValidationError(f"TSV has no header: {path}")
    headers = tuple(fieldnames)
    missing = [header for header in EXPECTED_HEADERS if header not in headers]
    if missing:
        raise CorpusValidationError(
            f"TSV is missing required columns {missing}: {path}"
        )
    return headers


def validate_raw_tsv(path: Path) -> CorpusStats:
    """Validate schema, provenance constraints, and accession uniqueness."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise CorpusValidationError(f"TSV is missing or empty: {path}")

    accessions: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _validate_headers(reader.fieldnames, path)
        for line_number, row in enumerate(reader, start=2):
            accession = (row.get("Entry") or "").strip()
            if not accession:
                raise CorpusValidationError(
                    f"empty accession at line {line_number}: {path}"
                )
            if accession in accessions:
                raise CorpusValidationError(
                    f"duplicate accession {accession!r} at line {line_number}: {path}"
                )
            if (row.get("Reviewed") or "").strip().lower() != "reviewed":
                raise CorpusValidationError(
                    f"unreviewed accession {accession!r} at line {line_number}: {path}"
                )
            if (row.get("Organism") or "").strip() != "Homo sapiens (Human)":
                raise CorpusValidationError(
                    f"non-human accession {accession!r} at line {line_number}: {path}"
                )
            accessions.add(accession)
            rows += 1

    if rows == 0:
        raise CorpusValidationError(f"TSV contains no protein records: {path}")
    return CorpusStats(rows=rows, unique_accessions=len(accessions))


def filter_excluded_accessions(
    raw_path: Path,
    output_path: Path,
    excluded: set[str],
) -> FilterStats:
    """Write rows not present in the exclusion set, preserving the TSV schema."""
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = 0
    excluded_rows = 0
    retained_rows = 0
    with raw_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        headers = _validate_headers(reader.fieldnames, raw_path)
        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=headers,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in reader:
                raw_rows += 1
                accession = (row.get("Entry") or "").strip()
                if accession in excluded:
                    excluded_rows += 1
                    continue
                writer.writerow(row)
                retained_rows += 1

    if raw_rows == 0:
        raise CorpusValidationError(f"Cannot filter an empty corpus: {raw_path}")
    if retained_rows == 0:
        raise CorpusValidationError("Exclusion removed every downloaded protein")
    return FilterStats(
        raw_rows=raw_rows,
        excluded_rows=excluded_rows,
        retained_rows=retained_rows,
    )


def _temporary_path(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=parent)
    os.close(descriptor)
    return Path(name)


def _open_url(url: str, timeout: float) -> BinaryIO:
    request = Request(url, headers={"User-Agent": "ESM4SL-SAE-corpus/1.0"})
    return urlopen(request, timeout=timeout)


def _download(url: str, destination: Path, timeout: float, retries: int) -> None:
    for attempt in range(retries + 1):
        try:
            with _open_url(url, timeout) as response:
                with destination.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if destination.stat().st_size == 0:
                raise CorpusValidationError(f"UniProt returned an empty response: {url}")
            return
        except HTTPError as error:
            retryable = error.code == 429 or error.code >= 500
            if not retryable or attempt == retries:
                raise RuntimeError(
                    f"UniProt request failed with HTTP {error.code}: {error.reason}"
                ) from error
        except (URLError, TimeoutError, OSError) as error:
            if attempt == retries:
                raise RuntimeError(f"UniProt request failed: {error}") from error
        time.sleep(min(2**attempt, 8))


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


def _processed_accessions(path: Path, excluded: set[str]) -> set[str]:
    accessions: set[str] = set()
    leaked: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _validate_headers(reader.fieldnames, path)
        for row in reader:
            accession = (row.get("Entry") or "").strip()
            accessions.add(accession)
            if accession in excluded:
                leaked.add(accession)
    if leaked:
        examples = ", ".join(sorted(leaked)[:5])
        raise CorpusValidationError(
            f"processed corpus contains excluded accession(s): {examples}"
        )
    return accessions


def _manifest(
    *,
    request_url: str,
    raw_path: Path,
    processed_path: Path,
    exclusion_path: Path,
    exclusion_count: int,
    raw_stats: CorpusStats,
    processed_stats: CorpusStats,
    filter_stats: FilterStats,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "endpoint": UNIPROT_STREAM_ENDPOINT,
            "query": UNIPROT_QUERY,
            "fields": list(UNIPROT_FIELDS),
            "resolved_url": request_url,
        },
        "counts": {
            "raw_rows": raw_stats.rows,
            "raw_unique_accessions": raw_stats.unique_accessions,
            "excluded_rows": filter_stats.excluded_rows,
            "retained_rows": processed_stats.rows,
            "retained_unique_accessions": processed_stats.unique_accessions,
            "exclusion_accessions": exclusion_count,
        },
        "exclusion_source": str(exclusion_path.resolve()),
        "files": {
            "raw": _file_metadata(raw_path),
            "processed": _file_metadata(processed_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": str(Path(__file__).resolve()),
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = _temporary_path(path.parent, f".{path.name}.")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_existing(
    output_root: Path,
    exclusion_path: Path,
) -> dict[str, object]:
    raw_path = output_root / RAW_RELATIVE_PATH
    processed_path = output_root / PROCESSED_RELATIVE_PATH
    manifest_path = output_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    excluded = read_excluded_accessions(exclusion_path)
    raw_stats = validate_raw_tsv(raw_path)
    processed_stats = validate_raw_tsv(processed_path)
    _processed_accessions(processed_path, excluded)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_raw = manifest.get("files", {}).get("raw", {})
    expected_processed = manifest.get("files", {}).get("processed", {})
    for label, path, expected in (
        ("raw", raw_path, expected_raw),
        ("processed", processed_path, expected_processed),
    ):
        if expected.get("sha256") != _sha256(path):
            raise CorpusValidationError(f"{label} checksum does not match manifest")
        if expected.get("bytes") != path.stat().st_size:
            raise CorpusValidationError(f"{label} byte size does not match manifest")

    counts = manifest.get("counts", {})
    if counts.get("raw_rows") != raw_stats.rows:
        raise CorpusValidationError("raw row count does not match manifest")
    if counts.get("retained_rows") != processed_stats.rows:
        raise CorpusValidationError("retained row count does not match manifest")
    if counts.get("raw_rows") != counts.get("excluded_rows", 0) + counts.get(
        "retained_rows", 0
    ):
        raise CorpusValidationError("raw count is not excluded plus retained")
    return manifest


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run a download/filter job or validate an existing job."""
    output_root = Path(args.output_root)
    exclusion_path = Path(args.exclusion_table)
    if args.validate_only:
        return _validate_existing(output_root, exclusion_path)

    raw_path = output_root / RAW_RELATIVE_PATH
    processed_path = output_root / PROCESSED_RELATIVE_PATH
    manifest_path = output_root / MANIFEST_NAME
    existing = [path for path in (raw_path, processed_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists; use --overwrite: {names}")

    excluded = read_excluded_accessions(exclusion_path)
    request_url = args.source_url or build_request_url()
    raw_temporary = _temporary_path(raw_path.parent, f".{raw_path.name}.")
    processed_temporary = _temporary_path(
        processed_path.parent, f".{processed_path.name}."
    )
    try:
        _download(request_url, raw_temporary, args.timeout, args.retries)
        raw_stats = validate_raw_tsv(raw_temporary)
        filter_stats = filter_excluded_accessions(
            raw_temporary,
            processed_temporary,
            excluded,
        )
        processed_stats = validate_raw_tsv(processed_temporary)
        _processed_accessions(processed_temporary, excluded)
        if raw_stats.rows != filter_stats.raw_rows:
            raise CorpusValidationError("filter did not consume every raw row")
        if raw_stats.rows != filter_stats.excluded_rows + processed_stats.rows:
            raise CorpusValidationError("raw count is not excluded plus retained")

        os.replace(raw_temporary, raw_path)
        os.replace(processed_temporary, processed_path)
        manifest = _manifest(
            request_url=request_url,
            raw_path=raw_path,
            processed_path=processed_path,
            exclusion_path=exclusion_path,
            exclusion_count=len(excluded),
            raw_stats=raw_stats,
            processed_stats=processed_stats,
            filter_stats=filter_stats,
        )
        _write_json_atomic(manifest_path, manifest)
        return manifest
    finally:
        raw_temporary.unlink(missing_ok=True)
        processed_temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--exclusion-table",
        type=Path,
        default=DEFAULT_EXCLUSION_TABLE,
    )
    parser.add_argument(
        "--source-url",
        help="Use a complete alternate source URL instead of the live UniProt query.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")
    manifest = run(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
