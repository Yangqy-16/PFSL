"""Leakage-resistant train-only concept discovery for protein SAE features.

The stage consumes sparse protein maxima without densifying feature width.  A
source-excluded score is considered only for proteins whose corresponding view
is eligible; absent sparse values in an eligible view are genuine zeros.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .concept_parsers import ConceptRecord
from .feature_atlas_config import FeatureAtlasConfig
from .feature_atlas_io import (
    AtlasIOError,
    sha256_file,
    strict_json_lines,
    strict_json_load,
    write_json_atomic,
)
from .protein_feature_activations import (
    ProteinActivationError,
    ProteinActivationRow,
    iter_protein_activation_rows,
)


SCHEMA_VERSION = 1
DATABASES = ("go", "reactome", "interpro", "corum")
_DATABASE_SET = frozenset(DATABASES)
_OUTPUT_NAME = "train_candidates.jsonl"
_MANIFEST_NAME = "discovery_manifest.json"
_CONCEPT_FIELDS = frozenset(field.name for field in fields(ConceptRecord))


class DiscoveryError(ValueError):
    """Raised when discovery inputs or registered outputs violate the contract."""


@dataclass(frozen=True)
class FisherResult:
    odds_ratio: float
    p_value: float


@dataclass
class _TrainingData:
    """Sparse single-pass training accumulator; never retains source rows."""

    accessions: set[str]
    eligible_accessions: dict[str, set[str]]
    excluded_values: dict[str, dict[int, dict[str, float]]]
    all_source_values: dict[str, dict[int, dict[str, float]]]
    row_signatures: dict[str, dict[str, object]]


def fisher_overrepresentation(
    overlap: int,
    firing_only: int,
    annotated_only: int,
    neither: int,
) -> FisherResult:
    """Return the exact one-sided Fisher overrepresentation result."""

    counts = (overlap, firing_only, annotated_only, neither)
    if any(type(value) is not int for value in counts):
        raise TypeError("Fisher table counts must be built-in integers")
    if any(value < 0 for value in counts):
        raise ValueError("Fisher table counts must be nonnegative")
    if sum(counts) == 0:
        raise ValueError("Fisher table must contain at least one observation")
    odds, p_value = stats.fisher_exact(
        [[overlap, firing_only], [annotated_only, neither]],
        alternative="greater",
    )
    odds_float, p_float = float(odds), float(p_value)
    if math.isnan(odds_float) or not math.isfinite(p_float):
        raise ValueError("Fisher result is undefined for the supplied margins")
    return FisherResult(odds_ratio=odds_float, p_value=p_float)


def rank_biserial_effect(
    positive: np.ndarray,
    unlabeled: np.ndarray,
) -> tuple[float, float]:
    """Return rank-biserial effect and one-sided Mann-Whitney p-value."""

    if not isinstance(positive, np.ndarray) or not isinstance(unlabeled, np.ndarray):
        raise TypeError("rank inputs must be NumPy arrays")
    if positive.ndim != 1 or unlabeled.ndim != 1:
        raise ValueError("rank inputs must be one-dimensional")
    if positive.size == 0 or unlabeled.size == 0:
        raise ValueError("rank inputs must both be nonempty")
    if positive.dtype == np.bool_ or unlabeled.dtype == np.bool_:
        raise TypeError("rank inputs must be numeric, not boolean")
    try:
        finite = bool(np.isfinite(positive).all() and np.isfinite(unlabeled).all())
    except TypeError as error:
        raise TypeError("rank inputs must be numeric") from error
    if not finite:
        raise ValueError("rank inputs must be finite")
    result = stats.mannwhitneyu(positive, unlabeled, alternative="greater")
    denominator = int(positive.size) * int(unlabeled.size)
    effect = 2.0 * float(result.statistic) / denominator - 1.0
    p_value = float(result.pvalue)
    if not math.isfinite(effect) or not math.isfinite(p_value):
        raise ValueError("rank result is non-finite")
    return effect, p_value


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Apply deterministic BH correction with stable tie ordering."""

    if isinstance(p_values, (str, bytes)) or not isinstance(p_values, Sequence):
        raise TypeError("p_values must be a sequence")
    parsed: list[float] = []
    for value in p_values:
        if type(value) not in (int, float):
            raise TypeError("p-values must be built-in numeric scalars")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("p-values must be finite and between zero and one")
        parsed.append(number)
    count = len(parsed)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: (parsed[index], index))
    adjusted_sorted = [0.0] * count
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, parsed[index] * count / rank)
        adjusted_sorted[reverse_rank] = min(1.0, running)
    adjusted = [0.0] * count
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted


def _canonical_sha(payload: object) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _write_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
) -> tuple[int, str]:
    """Atomically stream deterministic JSONL, including a valid empty file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_path)
    count = 0
    pair_digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                if not isinstance(row, Mapping):
                    raise DiscoveryError("JSONL spill row is not a mapping")
                encoded = json.dumps(
                    row, allow_nan=False, sort_keys=True, separators=(",", ":")
                )
                handle.write(encoded)
                handle.write("\n")
                pair_digest.update(
                    f'{row["feature_index"]}\t{row["database"]}\t{row["concept_id"]}\n'.encode(
                        "utf-8"
                    )
                )
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count, pair_digest.hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if any(type(key) is not str for key in keys) or len(keys) != len(set(keys)):
        raise DiscoveryError("JSONL spill object has duplicate or invalid keys")
    if keys != sorted(keys):
        raise DiscoveryError("JSONL spill object keys are not deterministic")
    return dict(pairs)


def _reject_json_constant(value: str) -> object:
    raise DiscoveryError(f"JSONL spill contains non-finite constant {value}")


def _iter_rows(path: Path) -> Iterator[dict[str, object]]:
    """Read one strict JSONL object at a time without whole-file readback."""

    if path.is_symlink() or not path.is_file():
        raise DiscoveryError(f"JSONL spill is not a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line == "\n" or not line.endswith("\n") or line.endswith("\r\n"):
                    raise DiscoveryError(
                        f"JSONL spill line {line_number} has invalid framing: {path}"
                    )
                row = json.loads(
                    line[:-1],
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
                if not isinstance(row, dict):
                    raise DiscoveryError(
                        f"JSONL spill line {line_number} is not an object: {path}"
                    )
                yield row
    except DiscoveryError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise DiscoveryError(f"Cannot stream strict JSONL rows from {path}: {error}") from error


def _read_rows(path: Path) -> list[dict[str, object]]:
    try:
        if path.stat().st_size == 0:
            return []
    except OSError as error:
        raise DiscoveryError(f"Cannot inspect JSONL output: {path}") from error
    return list(strict_json_lines(path))


def _load_concepts(
    source: Path | Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, object]], str, str | None]:
    if isinstance(source, Path):
        try:
            rows = list(strict_json_lines(source))
            checksum = sha256_file(source)
            manifest_path = source.parent / "manifest.json"
            manifest = strict_json_load(manifest_path)
            manifest_checksum = sha256_file(manifest_path)
        except (AtlasIOError, OSError, ValueError) as error:
            raise DiscoveryError(f"Invalid validated Task 3 concept index: {error}") from error
        files = manifest.get("files")
        metadata = files.get("concept_index") if isinstance(files, dict) else None
        if (
            source.name != "concept_index.jsonl"
            or not isinstance(metadata, dict)
            or metadata.get("sha256") != checksum
            or metadata.get("bytes") != source.stat().st_size
        ):
            raise DiscoveryError("Task 3 concept index is not registered by its manifest")
    else:
        try:
            rows = [dict(row) for row in source]
        except (TypeError, ValueError) as error:
            raise DiscoveryError("Invalid in-memory concept fixture") from error
        manifest_checksum = None
        checksum = ""

    normalized: list[dict[str, object]] = []
    identities: set[tuple[str, str]] = set()
    for number, row in enumerate(rows):
        if frozenset(row) != _CONCEPT_FIELDS:
            raise DiscoveryError(f"Concept row {number} has invalid exact schema")
        database = row.get("database")
        concept_id = row.get("concept_id")
        members = row.get("protein_accessions")
        if database not in _DATABASE_SET or type(concept_id) is not str or not concept_id:
            raise DiscoveryError(f"Concept row {number} has invalid identity")
        if not isinstance(members, (list, tuple)) or any(
            type(member) is not str or not member for member in members
        ):
            raise DiscoveryError(f"Concept row {number} has invalid protein members")
        member_list = list(members)
        if member_list != sorted(set(member_list)):
            raise DiscoveryError(f"Concept row {number} members are not sorted unique")
        identity = (database, concept_id)
        if identity in identities:
            raise DiscoveryError(f"Concept index contains duplicate identity {identity}")
        identities.add(identity)
        copied = dict(row)
        copied["protein_accessions"] = member_list
        normalized.append(copied)
    normalized.sort(key=lambda row: (str(row["database"]), str(row["concept_id"])))
    if not checksum:
        checksum = _canonical_sha(normalized)
    return normalized, checksum, manifest_checksum


def _validate_row(row: object, config: FeatureAtlasConfig) -> ProteinActivationRow:
    if not isinstance(row, ProteinActivationRow):
        raise DiscoveryError("Protein activation source yielded an invalid row type")
    if row.split != "train":
        raise DiscoveryError("Validation row was exposed to train-only discovery")
    if type(row.accession) is not str or not row.accession:
        raise DiscoveryError("Protein activation row has invalid accession")
    if set(row.eligibility) != _DATABASE_SET or any(
        type(value) is not bool for value in row.eligibility.values()
    ):
        raise DiscoveryError("Protein activation row has invalid eligibility schema")
    mappings: list[tuple[str, Mapping[int, float]]] = [("all_sources", row.all_sources)]
    for database in DATABASES:
        values = row.source_excluded.get(database, {})
        if not row.eligibility[database] and values:
            raise DiscoveryError("Ineligible source-excluded view contains activations")
        mappings.append((database, values))
    for label, values in mappings:
        if not isinstance(values, Mapping):
            raise DiscoveryError(f"{label} activation view is not a mapping")
        for feature, activation in values.items():
            if type(feature) is not int or not 0 <= feature < config.width:
                raise DiscoveryError(f"{label} feature index is out of range")
            if type(activation) is not float or not math.isfinite(activation):
                raise DiscoveryError(f"{label} activations must be finite built-in floats")
            if activation <= 0.0:
                raise DiscoveryError(f"{label} sparse activations must be positive")
    return row


def _load_train_rows(
    source: Any,
    config: FeatureAtlasConfig,
) -> tuple[_TrainingData, str, dict[str, object] | None]:
    upstream: dict[str, object] | None
    if isinstance(source, Path):
        manifest_path = source / "manifest.json"
        try:
            manifest = strict_json_load(manifest_path)
            manifest_checksum = sha256_file(manifest_path)
            iterator = iter_protein_activation_rows(source, split="train")
        except (AtlasIOError, ProteinActivationError, OSError, ValueError) as error:
            raise DiscoveryError(f"Invalid validated Task 5 protein activations: {error}") from error
        if manifest.get("complete") is not True or manifest.get("width") != config.width:
            raise DiscoveryError("Task 5 manifest does not match discovery configuration")
        upstream = {
            "concept_binding": manifest.get("concept_binding"),
            "source_provenance": manifest.get("source_provenance"),
        }
    elif hasattr(source, "iter_train_rows") and callable(source.iter_train_rows):
        iterator = source.iter_train_rows()
        manifest_checksum = ""
        upstream = None
    else:
        raise TypeError(
            "protein_activation_dir must be a Path or an explicit train-row provider"
        )
    data = _TrainingData(
        accessions=set(),
        eligible_accessions={database: set() for database in DATABASES},
        excluded_values={database: {} for database in DATABASES},
        all_source_values={database: {} for database in DATABASES},
        row_signatures={},
    )
    for raw_row in iterator:
        row = _validate_row(raw_row, config)
        if row.accession in data.accessions:
            raise DiscoveryError("Protein activation source contains duplicate accessions")
        data.accessions.add(row.accession)
        signature = {
            "accession": row.accession,
            "all_sources": sorted(row.all_sources.items()),
            "eligibility": dict(row.eligibility),
            "source_excluded": {
                database: sorted(row.source_excluded.get(database, {}).items())
                for database in DATABASES
            },
        }
        data.row_signatures[row.accession] = signature
        for database in DATABASES:
            if not row.eligibility[database]:
                continue
            data.eligible_accessions[database].add(row.accession)
            for feature, value in row.source_excluded.get(database, {}).items():
                data.excluded_values[database].setdefault(feature, {})[row.accession] = value
            for feature, value in row.all_sources.items():
                data.all_source_values[database].setdefault(feature, {})[row.accession] = value
    if not data.accessions:
        raise DiscoveryError("Train-only discovery received no training accessions")
    if not manifest_checksum:
        manifest_checksum = _canonical_sha([
            data.row_signatures[accession] for accession in sorted(data.accessions)
        ])
    return data, manifest_checksum, upstream


def _metrics(
    values: np.ndarray,
    annotated: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    firing = values > threshold
    overlap = int(np.count_nonzero(firing & annotated))
    firing_only = int(np.count_nonzero(firing & ~annotated))
    annotated_only = int(np.count_nonzero(~firing & annotated))
    neither = int(np.count_nonzero(~firing & ~annotated))
    try:
        fisher = fisher_overrepresentation(overlap, firing_only, annotated_only, neither)
        fisher_odds = _finite_or_none(fisher.odds_ratio)
        fisher_p = fisher.p_value
    except ValueError:
        # A constant firing diagnostic has a zero table margin.  Its odds are
        # undefined and it provides no overrepresentation evidence.
        fisher_odds = None
        fisher_p = 1.0
    effect, rank_p = rank_biserial_effect(values[annotated], values[~annotated])
    return {
        "annotated_count": int(np.count_nonzero(annotated)),
        "annotated_prevalence": float(np.mean(annotated)),
        "firing_count": int(np.count_nonzero(firing)),
        "overlap_count": overlap,
        "firing_only_count": firing_only,
        "annotated_only_count": annotated_only,
        "neither_count": neither,
        "fisher_odds_ratio": fisher_odds,
        "fisher_p_value": fisher_p,
        "rank_biserial_effect": effect,
        "rank_p_value": rank_p,
        "direction": "positive" if effect > 0.0 else "negative" if effect < 0.0 else "null",
    }


def _empty_filter_counts(concept_count: int) -> dict[str, int]:
    return {
        "below_min_annotated": 0,
        "below_min_firing": 0,
        "concept_count": concept_count,
        "eligible_tests": 0,
        "fisher_or_rank_fdr": 0,
        "insufficient_overlap": 0,
        "no_unlabeled_comparison": 0,
        "nonpositive_direction": 0,
    }


def _spill_eligible_tests(
    data: _TrainingData,
    concepts: list[dict[str, object]],
    config: FeatureAtlasConfig,
    directory: Path,
) -> dict[str, dict[str, int]]:
    grouped = {
        database: [row for row in concepts if row["database"] == database]
        for database in DATABASES
    }
    counts = {
        database: _empty_filter_counts(len(grouped[database]))
        for database in DATABASES
    }
    for database in DATABASES:
        accessions = sorted(data.eligible_accessions[database])
        accession_set = set(accessions)
        features = sorted(data.excluded_values[database])
        prepared: list[tuple[dict[str, object], set[str]]] = []
        for concept_row in grouped[database]:
            member_set = set(concept_row["protein_accessions"]) & accession_set
            annotated_count = len(member_set)
            if annotated_count < config.train_min_annotated:
                counts[database]["below_min_annotated"] += 1
                continue
            if annotated_count == len(accessions):
                counts[database]["no_unlabeled_comparison"] += len(features)
                continue
            prepared.append((concept_row, member_set))

        def eligible_rows() -> Iterator[dict[str, object]]:
            for feature in features:
                sparse_excluded = data.excluded_values[database][feature]
                firing_accessions = {
                    accession for accession, value in sparse_excluded.items()
                    if value > config.activation_threshold
                }
                firing_count = len(firing_accessions)
                if firing_count < config.train_min_firing:
                    counts[database]["below_min_firing"] += len(prepared)
                    continue
                for concept_row, member_set in prepared:
                    overlap = len(firing_accessions & member_set)
                    if overlap < config.train_min_overlap:
                        counts[database]["insufficient_overlap"] += 1
                        continue
                    annotated = np.fromiter(
                        (accession in member_set for accession in accessions),
                        dtype=np.bool_, count=len(accessions),
                    )
                    source_values = np.fromiter(
                        (sparse_excluded.get(accession, 0.0) for accession in accessions),
                        dtype=np.float64, count=len(accessions),
                    )
                    source_metrics = _metrics(
                        source_values, annotated, config.activation_threshold
                    )
                    del source_values
                    sparse_all = data.all_source_values[database].get(feature, {})
                    all_values = np.fromiter(
                        (sparse_all.get(accession, 0.0) for accession in accessions),
                        dtype=np.float64, count=len(accessions),
                    )
                    all_metrics = _metrics(
                        all_values, annotated, config.activation_threshold
                    )
                    del all_values, annotated
                    yield {
                        "all_source": all_metrics,
                        "concept_id": concept_row["concept_id"],
                        "concept_name": concept_row["concept_name"],
                        "concept_type": concept_row["concept_type"],
                        "database": database,
                        "eligible_accession_count": len(accessions),
                        "feature_index": feature,
                        "schema_version": SCHEMA_VERSION,
                        "source_excluded": source_metrics,
                        "source_exclusion_change": {
                            "rank_biserial_delta": float(
                                source_metrics["rank_biserial_effect"]
                                - all_metrics["rank_biserial_effect"]
                            ),
                            "rank_biserial_retention": float(
                                source_metrics["rank_biserial_effect"]
                                / max(abs(all_metrics["rank_biserial_effect"]), 1e-8)
                            ),
                        },
                    }

        count, _ = _write_rows(directory / f"{database}.jsonl", eligible_rows())
        counts[database]["eligible_tests"] = count
    return counts


def _bh_memmap(p_values: np.ndarray, q_values: np.ndarray) -> None:
    """Correct one on-disk p-value vector without Python-object families."""

    count = len(p_values)
    if count == 0:
        return
    if not bool(np.isfinite(p_values).all()) or bool(
        np.any((p_values < 0.0) | (p_values > 1.0))
    ):
        raise DiscoveryError("Eligible-test spill contains invalid p-values")
    order = np.argsort(p_values, kind="stable")
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        original_index = int(order[reverse_rank])
        running = min(
            running,
            float(p_values[original_index]) * count / (reverse_rank + 1),
        )
        q_values[original_index] = min(1.0, running)


def _correct_and_spill_candidates(
    family_path: Path,
    selected_path: Path,
    binary_path: Path,
    count: int,
    counts: dict[str, int],
    config: FeatureAtlasConfig,
) -> int:
    if count == 0:
        selected_count, _ = _write_rows(selected_path, ())
        return selected_count
    values = np.memmap(binary_path, dtype=np.float64, mode="w+", shape=(count, 4))
    observed = 0
    for observed, row in enumerate(_iter_rows(family_path), start=1):
        if observed > count:
            raise DiscoveryError("Eligible-test spill count grew during correction")
        source = row.get("source_excluded")
        if not isinstance(source, dict):
            raise DiscoveryError("Eligible-test spill has invalid statistical view")
        values[observed - 1, 0] = float(source["fisher_p_value"])
        values[observed - 1, 1] = float(source["rank_p_value"])
    if observed != count:
        raise DiscoveryError("Eligible-test spill count changed during correction")
    _bh_memmap(values[:, 0], values[:, 2])
    _bh_memmap(values[:, 1], values[:, 3])
    values.flush()

    def selected_rows() -> Iterator[dict[str, object]]:
        for index, row in enumerate(_iter_rows(family_path)):
            source = row["source_excluded"]
            if not isinstance(source, dict):
                raise DiscoveryError("Eligible-test spill has invalid statistical view")
            fisher_value = float(values[index, 2])
            rank_value = float(values[index, 3])
            source["fisher_q_value"] = fisher_value
            source["rank_q_value"] = rank_value
            if source["direction"] != "positive":
                counts["nonpositive_direction"] += 1
                continue
            if not (
                fisher_value < config.train_candidate_fdr
                or rank_value < config.train_candidate_fdr
            ):
                counts["fisher_or_rank_fdr"] += 1
                continue
            yield row

    selected_count, _ = _write_rows(selected_path, selected_rows())
    del values
    return selected_count


def _merge_candidate_spills(directory: Path) -> Iterator[dict[str, object]]:
    yield from heapq.merge(
        *(
            _iter_rows(directory / f"{database}.selected.jsonl")
            for database in DATABASES
        ),
        key=lambda row: (
            int(row["feature_index"]),
            str(row["database"]),
            str(row["concept_id"]),
        ),
    )


def _pair_list_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        line = f'{row["feature_index"]}\t{row["database"]}\t{row["concept_id"]}\n'
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


_VIEW_KEYS = frozenset({
    "annotated_count", "annotated_prevalence", "firing_count",
    "overlap_count", "firing_only_count", "annotated_only_count",
    "neither_count", "fisher_odds_ratio", "fisher_p_value",
    "rank_biserial_effect", "rank_p_value", "direction",
})
_SOURCE_VIEW_KEYS = _VIEW_KEYS | {"fisher_q_value", "rank_q_value"}
_CANDIDATE_KEYS = frozenset({
    "all_source", "concept_id", "concept_name", "concept_type", "database",
    "eligible_accession_count", "feature_index", "schema_version",
    "source_excluded", "source_exclusion_change",
})
_MANIFEST_KEYS = frozenset({
    "candidate_count", "complete", "config_fingerprint", "eligible_test_counts",
    "filter_counts", "input_provenance", "output", "pair_list_sha256",
    "schema_version", "train_accession_count", "train_candidate_fdr",
    "upstream_bindings", "width",
})


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise DiscoveryError(f"Candidate {label} must be a finite built-in float")
    return value


def _probability(value: object, label: str) -> float:
    number = _finite_float(value, label)
    if not 0.0 <= number <= 1.0:
        raise DiscoveryError(f"Candidate {label} must be between zero and one")
    return number


def _validate_statistical_view(
    view: object,
    *,
    eligible_count: int,
    source_excluded: bool,
) -> dict[str, object]:
    expected = _SOURCE_VIEW_KEYS if source_excluded else _VIEW_KEYS
    if not isinstance(view, dict) or frozenset(view) != expected:
        raise DiscoveryError("Candidate statistical view has invalid exact schema")
    count_names = (
        "annotated_count", "firing_count", "overlap_count", "firing_only_count",
        "annotated_only_count", "neither_count",
    )
    for name in count_names:
        if type(view[name]) is not int or view[name] < 0:
            raise DiscoveryError(f"Candidate {name} must be a nonnegative built-in int")
    overlap = view["overlap_count"]
    firing_only = view["firing_only_count"]
    annotated_only = view["annotated_only_count"]
    neither = view["neither_count"]
    if (
        overlap + firing_only + annotated_only + neither != eligible_count
        or view["annotated_count"] != overlap + annotated_only
        or view["firing_count"] != overlap + firing_only
    ):
        raise DiscoveryError("Candidate contingency counts do not reconcile")
    prevalence = _probability(view["annotated_prevalence"], "annotated_prevalence")
    if not math.isclose(
        prevalence, view["annotated_count"] / eligible_count,
        rel_tol=1e-15, abs_tol=0.0,
    ):
        raise DiscoveryError("Candidate annotation prevalence does not match counts")
    odds = view["fisher_odds_ratio"]
    if odds is not None and (_finite_float(odds, "fisher_odds_ratio") < 0.0):
        raise DiscoveryError("Candidate Fisher odds ratio must be nonnegative or null")
    _probability(view["fisher_p_value"], "fisher_p_value")
    _probability(view["rank_p_value"], "rank_p_value")
    effect = _finite_float(view["rank_biserial_effect"], "rank_biserial_effect")
    if not -1.0 <= effect <= 1.0:
        raise DiscoveryError("Candidate rank-biserial effect is out of range")
    expected_direction = "positive" if effect > 0.0 else "negative" if effect < 0.0 else "null"
    if view["direction"] != expected_direction:
        raise DiscoveryError("Candidate direction does not match rank-biserial effect")
    if source_excluded:
        _probability(view["fisher_q_value"], "fisher_q_value")
        _probability(view["rank_q_value"], "rank_q_value")
    return view


def _validate_candidate_row(
    row: object,
    *,
    width: int,
    train_candidate_fdr: float,
) -> tuple[int, str, str]:
    if not isinstance(row, dict) or frozenset(row) != _CANDIDATE_KEYS:
        raise DiscoveryError("Candidate row has invalid exact schema")
    if row["schema_version"] != SCHEMA_VERSION or type(row["schema_version"]) is not int:
        raise DiscoveryError("Candidate schema_version is invalid")
    feature = row["feature_index"]
    database = row["database"]
    concept_id = row["concept_id"]
    if type(feature) is not int or not 0 <= feature < width:
        raise DiscoveryError("Candidate feature index is invalid")
    if type(database) is not str or database not in _DATABASE_SET:
        raise DiscoveryError("Candidate database is invalid")
    if type(concept_id) is not str or not concept_id:
        raise DiscoveryError("Candidate concept ID is invalid")
    for name in ("concept_name", "concept_type"):
        if type(row[name]) is not str or not row[name]:
            raise DiscoveryError(f"Candidate {name} is invalid")
    eligible = row["eligible_accession_count"]
    if type(eligible) is not int or eligible <= 0:
        raise DiscoveryError("Candidate eligible accession count is invalid")
    all_view = _validate_statistical_view(
        row["all_source"], eligible_count=eligible, source_excluded=False
    )
    source_view = _validate_statistical_view(
        row["source_excluded"], eligible_count=eligible, source_excluded=True
    )
    if all_view["annotated_count"] != source_view["annotated_count"]:
        raise DiscoveryError("Candidate views disagree on annotation count")
    if source_view["direction"] != "positive" or not (
        source_view["fisher_q_value"] < train_candidate_fdr
        or source_view["rank_q_value"] < train_candidate_fdr
    ):
        raise DiscoveryError("Candidate does not satisfy strict positive Fisher-or-rank FDR")
    change = row["source_exclusion_change"]
    if not isinstance(change, dict) or set(change) != {
        "rank_biserial_delta", "rank_biserial_retention"
    }:
        raise DiscoveryError("Candidate source-exclusion change schema is invalid")
    delta = _finite_float(change["rank_biserial_delta"], "rank_biserial_delta")
    retention = _finite_float(
        change["rank_biserial_retention"], "rank_biserial_retention"
    )
    expected_delta = source_view["rank_biserial_effect"] - all_view["rank_biserial_effect"]
    expected_retention = source_view["rank_biserial_effect"] / max(
        abs(all_view["rank_biserial_effect"]), 1e-8
    )
    if delta != expected_delta or retention != expected_retention:
        raise DiscoveryError("Candidate source-exclusion change values do not reconcile")
    return feature, database, concept_id


def validate_train_discovery(
    output_dir: Path,
    *,
    config_fingerprint: str | None = None,
    input_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate the completed registered Task 6 output and exact pair checksum."""

    output_dir = Path(output_dir)
    try:
        manifest = strict_json_load(output_dir / _MANIFEST_NAME)
        if frozenset(manifest) != _MANIFEST_KEYS:
            raise DiscoveryError("Discovery manifest has invalid exact schema")
        output = manifest.get("output")
        candidate_path = output_dir / _OUTPUT_NAME
        if (
            not isinstance(output, dict)
            or set(output) != {"bytes", "path", "sha256"}
            or output.get("path") != _OUTPUT_NAME
            or type(output.get("bytes")) is not int
            or output["bytes"] < 0
            or type(output.get("sha256")) is not str
            or len(output["sha256"]) != 64
        ):
            raise DiscoveryError("Discovery manifest output registration is invalid")
        if output.get("sha256") != sha256_file(candidate_path):
            raise DiscoveryError("Registered discovery output checksum mismatch")
        if output.get("bytes") != candidate_path.stat().st_size:
            raise DiscoveryError("Registered discovery output byte size mismatch")
    except DiscoveryError:
        raise
    except (AtlasIOError, OSError, ValueError) as error:
        raise DiscoveryError(f"Invalid discovery manifest or output: {error}") from error
    width = manifest.get("width")
    threshold = manifest.get("train_candidate_fdr")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or type(manifest.get("schema_version")) is not int
        or manifest.get("complete") is not True
        or type(width) is not int or width <= 0
        or type(threshold) is not float or not 0.0 < threshold <= 1.0
        or type(manifest.get("train_accession_count")) is not int
        or manifest["train_accession_count"] <= 0
        or type(manifest.get("candidate_count")) is not int
        or manifest["candidate_count"] < 0
    ):
        raise DiscoveryError("Discovery manifest is incomplete or has invalid schema")
    if config_fingerprint is not None and manifest.get("config_fingerprint") != config_fingerprint:
        raise DiscoveryError("Discovery configuration fingerprint changed")
    if input_provenance is not None and manifest.get("input_provenance") != dict(input_provenance):
        raise DiscoveryError("Discovery upstream input binding changed")
    eligible_counts = manifest.get("eligible_test_counts")
    filter_counts = manifest.get("filter_counts")
    expected_filters = set(_empty_filter_counts(0))
    if (
        not isinstance(eligible_counts, dict)
        or set(eligible_counts) != _DATABASE_SET
        or any(type(value) is not int or value < 0 for value in eligible_counts.values())
        or not isinstance(filter_counts, dict)
        or set(filter_counts) != _DATABASE_SET
    ):
        raise DiscoveryError("Discovery manifest database counts are invalid")
    for database, values in filter_counts.items():
        if (
            not isinstance(values, dict)
            or set(values) != expected_filters
            or any(type(value) is not int or value < 0 for value in values.values())
            or values["eligible_tests"] != eligible_counts[database]
        ):
            raise DiscoveryError("Discovery manifest filter counts are invalid")
    previous: tuple[int, str, str] | None = None
    observed_count = 0
    pair_digest = hashlib.sha256()
    for row in _iter_rows(candidate_path):
        identity = _validate_candidate_row(
            row, width=width, train_candidate_fdr=threshold
        )
        if previous is not None and identity <= previous:
            raise DiscoveryError("Candidate rows are not deterministically sorted and unique")
        previous = identity
        pair_digest.update(
            f"{identity[0]}\t{identity[1]}\t{identity[2]}\n".encode("utf-8")
        )
        observed_count += 1
    if manifest.get("candidate_count") != observed_count:
        raise DiscoveryError("Discovery candidate count mismatch")
    if manifest.get("pair_list_sha256") != pair_digest.hexdigest():
        raise DiscoveryError("Frozen feature-concept pair-list checksum mismatch")
    return manifest


def _publish(
    stage: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    config_fingerprint: str,
    input_provenance: Mapping[str, object],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / _OUTPUT_NAME, output_dir / _MANIFEST_NAME]
    if any(path.exists() for path in targets) and not overwrite:
        if not all(path.exists() for path in targets):
            raise DiscoveryError("Incomplete discovery output requires explicit overwrite")
        return validate_train_discovery(
            output_dir,
            config_fingerprint=config_fingerprint,
            input_provenance=input_provenance,
        )
    backup = Path(tempfile.mkdtemp(prefix=".discovery-backup-", dir=output_dir.parent))
    old = [path.exists() for path in targets]
    try:
        for exists, target in zip(old, targets, strict=True):
            if exists:
                os.replace(target, backup / target.name)
        os.replace(stage / _OUTPUT_NAME, targets[0])
        os.replace(stage / _MANIFEST_NAME, targets[1])
        result = validate_train_discovery(
            output_dir,
            config_fingerprint=config_fingerprint,
            input_provenance=input_provenance,
        )
    except BaseException as error:
        for target in targets:
            if target.exists():
                target.unlink()
        for exists, target in zip(old, targets, strict=True):
            saved = backup / target.name
            if exists and saved.exists():
                os.replace(saved, target)
        if isinstance(error, DiscoveryError):
            raise
        raise DiscoveryError(f"Cannot atomically publish discovery output: {error}") from error
    finally:
        shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
    return result


def discover_train_candidates(
    protein_activation_dir: Path | Any,
    concept_index_path: Path | Iterable[Mapping[str, object]],
    config: FeatureAtlasConfig,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Discover and atomically freeze train-only feature-concept candidates."""

    if not isinstance(config, FeatureAtlasConfig):
        raise TypeError("config must be a FeatureAtlasConfig")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    concepts, concept_sha, concept_manifest_sha = _load_concepts(concept_index_path)
    training, protein_manifest_sha, upstream = _load_train_rows(
        protein_activation_dir, config
    )
    if isinstance(protein_activation_dir, Path):
        binding = upstream.get("concept_binding") if isinstance(upstream, dict) else None
        if not isinstance(binding, dict) or (
            binding.get("index_sha256") != concept_sha
            or binding.get("manifest_sha256") != concept_manifest_sha
        ):
            raise DiscoveryError("Task 5 concept binding does not match validated Task 3 input")

    input_provenance = {
        "concept_index_sha256": concept_sha,
        "protein_activation_manifest_sha256": protein_manifest_sha,
    }
    fingerprint = config.fingerprint()
    existing = output_dir / _MANIFEST_NAME
    candidate = output_dir / _OUTPUT_NAME
    if not overwrite and existing.exists() and candidate.exists():
        return validate_train_discovery(
            output_dir,
            config_fingerprint=fingerprint,
            input_provenance=input_provenance,
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    test_stage = Path(tempfile.mkdtemp(prefix=".eligible-tests-", dir=output_dir.parent))
    stage: Path | None = None
    try:
        filter_counts = _spill_eligible_tests(
            training, concepts, config, test_stage
        )
        for database in DATABASES:
            _correct_and_spill_candidates(
                test_stage / f"{database}.jsonl",
                test_stage / f"{database}.selected.jsonl",
                test_stage / f"{database}.pvalues.bin",
                filter_counts[database]["eligible_tests"],
                filter_counts[database],
                config,
            )
        stage = Path(tempfile.mkdtemp(
            prefix=".discovery-stage-", dir=output_dir.parent
        ))
        candidate_path = stage / _OUTPUT_NAME
        candidate_count, pair_checksum = _write_rows(
            candidate_path, _merge_candidate_spills(test_stage)
        )
        manifest = {
            "candidate_count": candidate_count,
            "complete": True,
            "config_fingerprint": fingerprint,
            "eligible_test_counts": {
                database: filter_counts[database]["eligible_tests"]
                for database in DATABASES
            },
            "filter_counts": filter_counts,
            "input_provenance": input_provenance,
            "output": {
                "bytes": candidate_path.stat().st_size,
                "path": _OUTPUT_NAME,
                "sha256": sha256_file(candidate_path),
            },
            "pair_list_sha256": pair_checksum,
            "schema_version": SCHEMA_VERSION,
            "train_accession_count": len(training.accessions),
            "train_candidate_fdr": config.train_candidate_fdr,
            "upstream_bindings": upstream,
            "width": config.width,
        }
        write_json_atomic(stage / _MANIFEST_NAME, manifest)
    except (AtlasIOError, OSError, ValueError) as error:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        raise DiscoveryError(f"Cannot build streamed discovery outputs: {error}") from error
    finally:
        shutil.rmtree(test_stage, ignore_errors=True)
    assert stage is not None
    return _publish(
        stage,
        output_dir,
        overwrite=overwrite,
        config_fingerprint=fingerprint,
        input_provenance=input_provenance,
    )
