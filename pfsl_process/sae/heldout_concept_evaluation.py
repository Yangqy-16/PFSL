"""Leakage-resistant matched positive-unlabeled held-out evaluation.

Candidate identities are frozen on train, while every score, label, propensity
model, match, and inferential result in this module is computed exclusively on
the registered validation split.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
import tempfile
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .concept_parsers import ConceptParseError, validate_concept_index
from .feature_atlas_config import FeatureAtlasConfig
from .feature_atlas_io import (
    AtlasIOError,
    sha256_file,
    strict_json_lines,
    strict_json_load,
    write_json_atomic,
    write_jsonl_atomic,
)
from .feature_concept_discovery import benjamini_hochberg, validate_train_discovery
from .protein_feature_activations import iter_protein_activation_rows


SCHEMA_VERSION = 1
DATABASES = ("go", "reactome", "interpro", "corum")
_DATABASE_SET = frozenset(DATABASES)
_RESULT_NAME = "heldout_results.jsonl"
_MANIFEST_NAME = "heldout_manifest.json"
_SHA256_CHARS = frozenset("0123456789abcdef")
_COVARIATE_KEYS = frozenset({
    "accession", "annotation_covariates_available", "aspect_present",
    "aspect_unit_counts", "database_annotation_counts",
    "interpro_family_indicators", "sequence_length", "source_availability",
    "total_annotation_count", "total_text_unit_count",
})


class HeldoutEvaluationError(ValueError):
    """Raised when held-out inputs or outputs violate the frozen contract."""


@dataclass(frozen=True)
class MatchGroup:
    positive: str
    controls: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.positive) is not str or not self.positive:
            raise TypeError("MatchGroup positive must be a nonempty string")
        if (
            type(self.controls) is not tuple
            or not self.controls
            or any(type(item) is not str or not item for item in self.controls)
            or len(self.controls) != len(set(self.controls))
            or self.positive in self.controls
        ):
            raise ValueError("MatchGroup controls must be nonempty unique strings")


@dataclass(frozen=True)
class MatchedPUSet:
    groups: tuple[MatchGroup, ...]
    covariate_names: tuple[str, ...]
    unmatched_positives: tuple[str, ...]
    propensity_sha256: str

    def __post_init__(self) -> None:
        if type(self.groups) is not tuple or any(
            not isinstance(group, MatchGroup) for group in self.groups
        ):
            raise TypeError("MatchedPUSet groups must be a tuple of MatchGroup values")
        positives = tuple(group.positive for group in self.groups)
        controls = tuple(item for group in self.groups for item in group.controls)
        if (
            positives != tuple(sorted(set(positives)))
            or len(controls) != len(set(controls))
            or set(positives).intersection(controls)
        ):
            raise ValueError("MatchedPUSet groups reuse or disorder accessions")
        if (
            type(self.covariate_names) is not tuple
            or any(type(item) is not str or not item for item in self.covariate_names)
            or self.covariate_names != tuple(sorted(set(self.covariate_names)))
        ):
            raise ValueError("MatchedPUSet covariate names must be sorted unique strings")
        if (
            type(self.unmatched_positives) is not tuple
            or self.unmatched_positives != tuple(sorted(set(self.unmatched_positives)))
            or set(self.unmatched_positives).intersection(positives)
        ):
            raise ValueError("MatchedPUSet unmatched positives are invalid")
        if not _is_sha256(self.propensity_sha256):
            raise ValueError("MatchedPUSet propensity checksum is invalid")


@dataclass(frozen=True)
class MatchingMatrix:
    accessions: tuple[str, ...]
    covariate_names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if (
            type(self.accessions) is not tuple
            or not self.accessions
            or self.accessions != tuple(sorted(set(self.accessions)))
        ):
            raise ValueError("Matching accessions must be nonempty, sorted, and unique")
        if (
            type(self.covariate_names) is not tuple
            or not self.covariate_names
            or self.covariate_names != tuple(sorted(set(self.covariate_names)))
        ):
            raise ValueError("Matching covariates must be nonempty, sorted, and unique")
        if (
            not isinstance(self.values, np.ndarray)
            or self.values.dtype != np.float64
            or self.values.shape != (len(self.accessions), len(self.covariate_names))
            or not bool(np.isfinite(self.values).all())
        ):
            raise ValueError("Matching matrix must be a finite float64 matrix")
        self.values.setflags(write=False)


@dataclass(frozen=True)
class _HeldoutConceptContext:
    eligible: tuple[str, ...]
    positives: tuple[str, ...]
    matched: MatchedPUSet
    tier: str


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value).issubset(_SHA256_CHARS)


def _finite_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(
        value, (bool, np.bool_)
    ):
        raise HeldoutEvaluationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HeldoutEvaluationError(f"{label} must be finite")
    return result


def _validate_covariate_row(row: object) -> dict[str, object]:
    if not isinstance(row, dict) or frozenset(row) != _COVARIATE_KEYS:
        raise HeldoutEvaluationError("Covariate row has invalid exact schema")
    accession = row["accession"]
    counts, present = row["aspect_unit_counts"], row["aspect_present"]
    database_counts, availability = (
        row["database_annotation_counts"], row["source_availability"]
    )
    families = row["interpro_family_indicators"]
    if type(accession) is not str or not accession:
        raise HeldoutEvaluationError("Covariate accession is invalid")
    if (
        type(row["annotation_covariates_available"]) is not bool
        or row["annotation_covariates_available"] is not True
    ):
        raise HeldoutEvaluationError("Annotation covariates are unavailable")
    if (
        type(counts) is not list or len(counts) != 24
        or any(type(value) is not int or value < 0 for value in counts)
        or type(present) is not list or len(present) != 24
        or any(type(value) is not bool for value in present)
        or present != [value > 0 for value in counts]
    ):
        raise HeldoutEvaluationError("Covariate aspect schema is invalid")
    if (
        not isinstance(database_counts, dict) or set(database_counts) != _DATABASE_SET
        or any(type(value) is not int or value < 0 for value in database_counts.values())
        or not isinstance(availability, dict) or set(availability) != _DATABASE_SET
        or any(type(value) is not bool for value in availability.values())
    ):
        raise HeldoutEvaluationError("Covariate database schema is invalid")
    if (
        type(families) is not list
        or families != sorted(set(families))
        or any(type(item) is not str or not item for item in families)
    ):
        raise HeldoutEvaluationError("InterPro family covariates are invalid")
    for name in ("sequence_length", "total_annotation_count", "total_text_unit_count"):
        value = row[name]
        if type(value) is not int or value < 0:
            if type(value) is float and not math.isfinite(value):
                raise HeldoutEvaluationError(f"Covariate {name} must be finite")
            raise HeldoutEvaluationError(f"Covariate {name} is invalid")
    if row["sequence_length"] <= 0 or row["total_text_unit_count"] != sum(counts):
        raise HeldoutEvaluationError("Covariate count values do not reconcile")
    return row


def build_matching_matrix(
    covariates: Iterable[Mapping[str, object]],
    eligible_accessions: Sequence[str],
    *,
    database: str,
    concept_id: str,
    interpro_ancestor_ids: Sequence[str] = (),
    interpro_excluded_memberships: Mapping[str, Sequence[str]] | None = None,
) -> MatchingMatrix:
    """Build an activation-free matching matrix on eligible validation rows only."""

    if type(database) is not str or database not in _DATABASE_SET:
        raise HeldoutEvaluationError("Matching database is invalid")
    if type(concept_id) is not str or not concept_id:
        raise HeldoutEvaluationError("Matching concept ID is invalid")
    if isinstance(eligible_accessions, (str, bytes)) or not isinstance(
        eligible_accessions, Sequence
    ):
        raise TypeError("eligible_accessions must be a sequence")
    requested = tuple(sorted(eligible_accessions))
    if (
        any(type(item) is not str or not item for item in requested)
        or requested != tuple(sorted(set(requested)))
    ):
        raise HeldoutEvaluationError("Eligible accessions must be unique strings")
    ancestors = tuple(interpro_ancestor_ids)
    if any(type(item) is not str or not item for item in ancestors):
        raise HeldoutEvaluationError("InterPro ancestor IDs are invalid")
    excluded_memberships: dict[str, frozenset[str]] = {}
    if database == "interpro":
        expected_ids = {concept_id, *ancestors}
        if not isinstance(interpro_excluded_memberships, Mapping) or set(
            interpro_excluded_memberships
        ) != expected_ids:
            raise HeldoutEvaluationError(
                "InterPro matching requires exact target and ancestor memberships"
            )
        for excluded_id, members in interpro_excluded_memberships.items():
            if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
                raise HeldoutEvaluationError("InterPro excluded memberships are invalid")
            parsed = tuple(members)
            if (
                any(type(item) is not str or not item for item in parsed)
                or parsed != tuple(sorted(set(parsed)))
            ):
                raise HeldoutEvaluationError("InterPro excluded memberships are invalid")
            excluded_memberships[excluded_id] = frozenset(parsed)
    elif interpro_excluded_memberships not in (None, {}):
        raise HeldoutEvaluationError("InterPro memberships are invalid for this database")
    rows: dict[str, dict[str, object]] = {}
    for raw in covariates:
        row = _validate_covariate_row(dict(raw) if isinstance(raw, Mapping) else raw)
        accession = str(row["accession"])
        if accession in rows:
            raise HeldoutEvaluationError(f"Duplicate covariate accession: {accession}")
        rows[accession] = row
    missing = sorted(set(requested).difference(rows))
    if missing:
        raise HeldoutEvaluationError(f"Eligible accession has missing covariates: {missing[0]}")
    selected = tuple(
        accession for accession in requested
        if bool(rows[accession]["source_availability"][database])  # type: ignore[index]
    )
    if not selected:
        raise HeldoutEvaluationError("No eligible held-out rows have source availability")
    excluded_families = {concept_id, *ancestors} if database == "interpro" else set()
    families = sorted({
        family
        for accession in selected
        for family in rows[accession]["interpro_family_indicators"]  # type: ignore[union-attr]
        if family not in excluded_families
    })
    names = sorted(
        ["sequence_length_log2_bin", "total_annotation_count",
         "total_text_unit_count_log2_bin"]
        + [f"aspect_count:{index:02d}" for index in range(24)]
        + [f"aspect_present:{index:02d}" for index in range(24)]
        + [f"database_annotation_count:{source}" for source in DATABASES]
        + [f"source_available:{source}" for source in DATABASES]
        + [f"interpro_family:{family}" for family in families]
    )
    values: list[list[float]] = []
    for accession in selected:
        row = rows[accession]
        counts = row["aspect_unit_counts"]
        present = row["aspect_present"]
        database_counts = row["database_annotation_counts"]
        availability = row["source_availability"]
        indicators = set(row["interpro_family_indicators"])  # type: ignore[arg-type]
        excluded_contribution = sum(
            accession in members for members in excluded_memberships.values()
        )
        adjusted_interpro = int(database_counts["interpro"]) - excluded_contribution  # type: ignore[index]
        adjusted_total = int(row["total_annotation_count"]) - excluded_contribution
        if adjusted_interpro < 0 or adjusted_total < 0:
            raise HeldoutEvaluationError(
                "InterPro excluded annotation counts do not reconcile"
            )
        mapping: dict[str, float] = {
            "sequence_length_log2_bin": float(
                math.floor(math.log2(1 + int(row["sequence_length"])))
            ),
            "total_annotation_count": float(adjusted_total),
            "total_text_unit_count_log2_bin": float(
                math.floor(math.log2(1 + int(row["total_text_unit_count"])))
            ),
        }
        mapping.update({f"aspect_count:{i:02d}": float(counts[i]) for i in range(24)})  # type: ignore[index]
        mapping.update({f"aspect_present:{i:02d}": float(present[i]) for i in range(24)})  # type: ignore[index]
        mapping.update({
            f"database_annotation_count:{source}": float(
                adjusted_interpro if source == "interpro" else database_counts[source]  # type: ignore[index]
            )
            for source in DATABASES
        })
        mapping.update({
            f"source_available:{source}": float(availability[source])  # type: ignore[index]
            for source in DATABASES
        })
        mapping.update({
            f"interpro_family:{family}": float(family in indicators) for family in families
        })
        values.append([mapping[name] for name in names])
    matrix = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(matrix).all()):
        raise HeldoutEvaluationError("Matching covariates must be finite")
    return MatchingMatrix(selected, tuple(names), matrix)


def _cached_matching_matrix(
    cache: dict[tuple[str, str | None], MatchingMatrix],
    covariates: Iterable[Mapping[str, object]],
    eligible_accessions: Sequence[str],
    *,
    database: str,
    concept_id: str,
    interpro_ancestor_ids: Sequence[str] = (),
    interpro_excluded_memberships: Mapping[str, Sequence[str]] | None = None,
) -> MatchingMatrix:
    key = (database, concept_id if database == "interpro" else None)
    cached = cache.get(key)
    if cached is not None:
        return cached
    matrix = build_matching_matrix(
        covariates,
        eligible_accessions,
        database=database,
        concept_id=concept_id,
        interpro_ancestor_ids=interpro_ancestor_ids,
        interpro_excluded_memberships=interpro_excluded_memberships,
    )
    cache[key] = matrix
    return matrix


def _propensity_sha256(logits: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for accession in sorted(logits):
        value = _finite_number(logits[accession], "propensity logit")
        digest.update(f"{accession}\t{format(value, '.17g')}\n".encode("utf-8"))
    return digest.hexdigest()


def _nearest_groups(
    propensity_logits: Mapping[str, object],
    positive_accessions: Sequence[str],
    *,
    caliper: float,
    controls_per_positive: int,
) -> tuple[MatchGroup, ...]:
    if type(caliper) is not float or not math.isfinite(caliper) or caliper < 0.0:
        raise TypeError("caliper must be a finite nonnegative built-in float")
    if type(controls_per_positive) is not int or controls_per_positive <= 0:
        raise TypeError("controls_per_positive must be a positive built-in int")
    logits = {
        accession: _finite_number(value, "propensity logit")
        for accession, value in propensity_logits.items()
    }
    positives = tuple(sorted(positive_accessions))
    if positives != tuple(sorted(set(positives))) or any(item not in logits for item in positives):
        raise HeldoutEvaluationError("Positive accessions are invalid for propensity logits")
    positive_set = set(positives)
    available = set(logits).difference(positive_set)
    groups: list[MatchGroup] = []
    for positive in positives:
        ranked = sorted(
            (
                (abs(logits[control] - logits[positive]), control)
                for control in available
                if abs(logits[control] - logits[positive]) <= caliper
            ),
            key=lambda item: (item[0], item[1]),
        )
        controls = tuple(control for _, control in ranked[:controls_per_positive])
        if controls:
            groups.append(MatchGroup(positive, controls))
            available.difference_update(controls)
    return tuple(groups)


def match_positive_unlabeled(
    matching_matrix: MatchingMatrix,
    positive_accessions: Sequence[str],
    config: FeatureAtlasConfig,
) -> MatchedPUSet:
    """Fit the exact fixed propensity model and greedily match without reuse."""

    if not isinstance(matching_matrix, MatchingMatrix):
        raise TypeError("matching_matrix must be a MatchingMatrix")
    if not isinstance(config, FeatureAtlasConfig):
        raise TypeError("config must be a FeatureAtlasConfig")
    positives = tuple(sorted(positive_accessions))
    if positives != tuple(sorted(set(positives))) or any(
        item not in set(matching_matrix.accessions) for item in positives
    ):
        raise HeldoutEvaluationError("Positive accessions must be unique eligible accessions")
    if not positives or len(positives) == len(matching_matrix.accessions):
        raise HeldoutEvaluationError("Propensity fitting requires positive and unlabeled rows")
    labels = np.asarray(
        [int(accession in set(positives)) for accession in matching_matrix.accessions],
        dtype=np.int64,
    )
    values = matching_matrix.values.copy()
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    values -= means
    nonconstant = scales > 0.0
    values[:, nonconstant] /= scales[nonconstant]
    values[:, ~nonconstant] = 0.0
    model = LogisticRegression(penalty="l2", solver="liblinear", random_state=42)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(values, labels)
        decision = np.asarray(model.decision_function(values), dtype=np.float64)
    except (ConvergenceWarning, ValueError, FloatingPointError) as error:
        raise HeldoutEvaluationError(f"Propensity model did not converge: {error}") from error
    if decision.shape != (len(labels),) or not bool(np.isfinite(decision).all()):
        raise HeldoutEvaluationError("Propensity model produced nonfinite logits")
    logits = dict(zip(matching_matrix.accessions, decision.tolist(), strict=True))
    positive_values = decision[labels == 1]
    control_values = decision[labels == 0]
    denominator = len(decision) - 2
    if denominator > 0:
        numerator = float(
            np.square(positive_values - positive_values.mean()).sum()
            + np.square(control_values - control_values.mean()).sum()
        )
        pooled_sd = math.sqrt(max(0.0, numerator / denominator))
    else:
        pooled_sd = 0.0
    caliper = float(config.propensity_caliper_sd * pooled_sd)
    groups = _nearest_groups(
        logits, positives, caliper=caliper,
        controls_per_positive=config.controls_per_positive,
    )
    matched = {group.positive for group in groups}
    return MatchedPUSet(
        groups=groups,
        covariate_names=matching_matrix.covariate_names,
        unmatched_positives=tuple(sorted(set(positives).difference(matched))),
        propensity_sha256=_propensity_sha256(logits),
    )


def pair_seed(base_seed: int, feature_index: int, database: str, concept_id: str) -> int:
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative built-in int")
    if type(feature_index) is not int or feature_index < 0:
        raise ValueError("feature_index must be a nonnegative built-in int")
    if type(database) is not str or database not in _DATABASE_SET:
        raise ValueError("database is invalid")
    if type(concept_id) is not str or not concept_id:
        raise ValueError("concept_id is invalid")
    payload = f"{base_seed}\t{database}\t{concept_id}\t{feature_index}\n".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def evidence_tier(matched_positive_count: int, config: FeatureAtlasConfig) -> str:
    if type(matched_positive_count) is not int or matched_positive_count < 0:
        raise TypeError("matched_positive_count must be a nonnegative built-in int")
    if not isinstance(config, FeatureAtlasConfig):
        raise TypeError("config must be a FeatureAtlasConfig")
    if matched_positive_count < config.validation_descriptive_min:
        return "insufficient"
    if matched_positive_count < config.validation_confirmatory_min:
        return "descriptive"
    return "confirmatory"


def _should_report_progress(processed: int, total: int) -> bool:
    return total >= 100_000 and (processed % 100_000 == 0 or processed == total)


def _rank_effect(positive: np.ndarray, controls: np.ndarray) -> float:
    result = stats.mannwhitneyu(positive, controls, alternative="greater")
    return 2.0 * float(result.statistic) / (len(positive) * len(controls)) - 1.0


def _vectorized_permutation_extreme_count(
    scores: Mapping[str, float],
    matched: MatchedPUSet,
    permutations: int,
    seed: int,
    observed_effect: float,
) -> tuple[int, int, str]:
    members = tuple(
        (group.positive, *group.controls) for group in matched.groups
    )
    group_sizes = np.asarray([len(group) for group in members], dtype=np.int64)
    offsets = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(group_sizes[:-1], dtype=np.int64),
    ))
    flattened_scores = np.asarray(
        [scores[accession] for group in members for accession in group],
        dtype=np.float64,
    )
    ranks = np.asarray(
        stats.rankdata(flattened_scores, method="average"), dtype=np.float64
    )
    assignment_count = math.prod(int(size) for size in group_sizes)
    if assignment_count <= permutations:
        mode = "exact"
        choices = np.asarray(tuple(itertools.product(
            *(range(int(size)) for size in group_sizes)
        )), dtype=np.int64)
        count = assignment_count
    else:
        mode = "sampled"
        generator = np.random.default_rng(seed)
        choices = generator.integers(
            0,
            group_sizes,
            size=(permutations, len(group_sizes)),
        )
        count = permutations
    selected_rank_sums = ranks[choices + offsets].sum(axis=1)
    positive_count = len(matched.groups)
    control_count = len(flattened_scores) - positive_count
    u_statistics = (
        selected_rank_sums - positive_count * (positive_count + 1) / 2.0
    )
    effects = 2.0 * u_statistics / (positive_count * control_count) - 1.0
    extreme = int(np.count_nonzero(effects >= observed_effect))
    return extreme, count, mode


def evaluate_matched_pair(
    feature_scores: Mapping[str, object],
    matched: MatchedPUSet,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    """Compute matched metrics and an exact-or-sampled grouped permutation test."""

    if not isinstance(matched, MatchedPUSet):
        raise TypeError("matched must be a MatchedPUSet")
    if type(permutations) is not int or permutations <= 0:
        raise TypeError("permutations must be a positive built-in int")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32-compatible built-in int")
    if not matched.groups:
        raise HeldoutEvaluationError("Matched evaluation requires at least one group")
    accessions = tuple(
        item for group in matched.groups for item in (group.positive, *group.controls)
    )
    missing = sorted(set(accessions).difference(feature_scores))
    if missing:
        raise HeldoutEvaluationError(f"Matched feature score is missing: {missing[0]}")
    scores = {item: _finite_number(feature_scores[item], "feature score") for item in accessions}
    positive = np.asarray([scores[group.positive] for group in matched.groups], dtype=np.float64)
    controls = np.asarray(
        [scores[item] for group in matched.groups for item in group.controls], dtype=np.float64
    )
    labels = np.asarray(
        [1] * len(positive) + [0] * len(controls), dtype=np.int64
    )
    ordered_scores = np.concatenate([positive, controls])
    auroc = float(roc_auc_score(labels, ordered_scores))
    auprc = float(average_precision_score(labels, ordered_scores))
    prevalence = float(len(positive) / len(ordered_scores))
    effect = _rank_effect(positive, controls)
    extreme, count, mode = _vectorized_permutation_extreme_count(
        scores, matched, permutations, seed, effect
    )
    p_value = float((extreme + 1) / (count + 1))
    result = {
        "auprc": auprc,
        "auprc_lift": float(auprc / prevalence),
        "auroc": auroc,
        "direction": "positive" if effect > 0 else "negative" if effect < 0 else "null",
        "matched_control_count": int(len(controls)),
        "matched_positive_count": int(len(positive)),
        "permutation_count": int(count),
        "permutation_mode": mode,
        "permutation_p_value": p_value,
        "prevalence": prevalence,
        "rank_biserial_effect": float(effect),
    }
    if any(type(value) is float and not math.isfinite(value) for value in result.values()):
        raise HeldoutEvaluationError("Matched metric is nonfinite")
    return result


def _apply_confirmatory_decisions(
    rows: Sequence[Mapping[str, object]], config: FeatureAtlasConfig
) -> list[dict[str, object]]:
    if not isinstance(config, FeatureAtlasConfig):
        raise TypeError("config must be a FeatureAtlasConfig")
    output = [dict(row) for row in rows]
    for database in DATABASES:
        indices = [index for index, row in enumerate(output) if row.get("database") == database]
        if not indices:
            continue
        p_values = [
            _finite_number(output[index].get("permutation_p_value"), "permutation p-value")
            for index in indices
        ]
        if any(not 0.0 <= value <= 1.0 for value in p_values):
            raise HeldoutEvaluationError("Permutation p-value is outside [0, 1]")
        for index, q_value in zip(indices, benjamini_hochberg(p_values), strict=True):
            row = output[index]
            row["permutation_q_value"] = q_value
            confirmed = bool(
                row.get("tier") == "confirmatory"
                and q_value < config.heldout_fdr
                and _finite_number(row.get("auroc"), "AUROC") >= config.heldout_min_auroc
                and _finite_number(row.get("auprc_lift"), "AUPRC lift")
                >= config.heldout_min_auprc_lift
                and row.get("direction") == row.get("training_direction")
            )
            row["confirmed"] = confirmed
            row["status"] = (
                "validated" if confirmed else "not_validated"
                if row.get("tier") == "confirmatory" else str(row.get("tier"))
            )
    unknown = [row.get("database") for row in output if row.get("database") not in _DATABASE_SET]
    if unknown:
        raise HeldoutEvaluationError("Decision row database is invalid")
    return output


_VIEW_RESULT_KEYS = frozenset({
    "auprc", "auprc_lift", "auroc", "direction", "permutation_count",
    "permutation_mode", "permutation_p_value", "permutation_q_value",
    "prevalence", "rank_biserial_effect",
})
_RESULT_KEYS = frozenset({
    "all_source", "annotated_positive_count", "confirmed", "concept_id",
    "concept_name", "concept_type", "covariate_names", "database",
    "eligible_accession_count", "evidence_tier", "feature_index",
    "matched_control_count", "matched_positive_count", "primary_view",
    "propensity_sha256", "schema_version", "seed", "source_excluded",
    "status", "training_direction", "unmatched_positives",
})
_MANIFEST_KEYS = frozenset({
    "candidate_count", "complete", "config_fingerprint", "input_provenance",
    "output", "pair_list_sha256", "parameters", "schema_version",
    "upstream_manifests", "validation_accession_count",
})
_PARAMETER_KEYS = frozenset({
    "controls_per_positive", "heldout_fdr", "heldout_min_auprc_lift",
    "heldout_min_auroc", "permutations", "propensity_caliper_sd",
    "propensity_model", "seed", "validation_confirmatory_min",
    "validation_descriptive_min", "width",
})
_INPUT_PROVENANCE_KEYS = frozenset({
    "concept_index_sha256", "concept_manifest_sha256",
    "covariates_manifest_sha256", "covariates_sha256",
    "discovery_manifest_sha256", "protein_activation_manifest_sha256",
    "train_candidates_sha256",
})
_UPSTREAM_KEYS = frozenset({
    "concept", "covariates", "discovery", "protein_activations",
})


def _load_registered_concepts(
    concept_index_path: Path,
    *,
    snapshot_dir: Path | None,
    units_path: Path | None,
    allow_test_fixture_unbound: bool,
) -> tuple[dict[tuple[str, str], dict[str, object]], dict[str, object], str]:
    concept_index_path = Path(concept_index_path)
    manifest_path = concept_index_path.parent / "manifest.json"
    if type(allow_test_fixture_unbound) is not bool:
        raise TypeError("allow_test_fixture_unbound must be a built-in bool")
    if (snapshot_dir is None) != (units_path is None):
        raise HeldoutEvaluationError("snapshot_dir and units_path must be supplied together")
    if allow_test_fixture_unbound:
        if snapshot_dir is not None or units_path is not None:
            raise HeldoutEvaluationError(
                "Fixture-unbound Task 3 input cannot supply production validation context"
            )
    else:
        if not isinstance(snapshot_dir, Path) or not isinstance(units_path, Path):
            raise HeldoutEvaluationError(
                "Production Task 3 validation requires snapshot_dir and units_path"
            )
        try:
            validate_concept_index(snapshot_dir, units_path, concept_index_path.parent)
        except (ConceptParseError, OSError, TypeError, ValueError) as error:
            raise HeldoutEvaluationError(f"Task 3 full validation failed: {error}") from error
    try:
        manifest = strict_json_load(manifest_path)
        checksum = sha256_file(concept_index_path)
        metadata = manifest.get("files")
        metadata = metadata.get("concept_index") if isinstance(metadata, dict) else None
        if (
            concept_index_path.name != "concept_index.jsonl"
            or not isinstance(metadata, dict)
            or metadata.get("path") != str(concept_index_path.resolve())
            or metadata.get("sha256") != checksum
            or metadata.get("bytes") != concept_index_path.stat().st_size
        ):
            raise HeldoutEvaluationError("Task 3 concept index is not registered")
        rows = list(strict_json_lines(concept_index_path))
    except HeldoutEvaluationError:
        raise
    except (AtlasIOError, OSError, ValueError) as error:
        raise HeldoutEvaluationError(f"Invalid Task 3 concept index: {error}") from error
    required = {
        "concept_id", "concept_name", "concept_type", "database", "evidence_tier",
        "membership_tracks", "original_accessions", "parent_ids",
        "protein_accessions", "redundancy_cluster_id", "source_release", "source_sha256",
    }
    concepts: dict[tuple[str, str], dict[str, object]] = {}
    previous: tuple[str, str] | None = None
    for number, row in enumerate(rows, 1):
        if set(row) != required:
            raise HeldoutEvaluationError(f"Concept row {number} has invalid exact schema")
        identity = (row.get("database"), row.get("concept_id"))
        if (
            type(identity[0]) is not str or identity[0] not in _DATABASE_SET
            or type(identity[1]) is not str or not identity[1]
            or previous is not None and identity <= previous
        ):
            raise HeldoutEvaluationError("Concept identities are invalid or unsorted")
        members, parents = row.get("protein_accessions"), row.get("parent_ids")
        if (
            type(members) is not list or members != sorted(set(members))
            or any(type(item) is not str or not item for item in members)
            or type(parents) is not list or parents != sorted(set(parents))
            or any(type(item) is not str or not item for item in parents)
        ):
            raise HeldoutEvaluationError("Concept membership or parent schema is invalid")
        concepts[identity] = row
        previous = identity
    identities = set(concepts)
    if any(
        (database, parent) not in identities
        for (database, _), row in concepts.items()
        for parent in row["parent_ids"]  # type: ignore[union-attr]
    ):
        raise HeldoutEvaluationError("Concept index has dangling parents")
    return concepts, manifest, sha256_file(manifest_path)


def _interpro_ancestors(
    concepts: Mapping[tuple[str, str], Mapping[str, object]], concept_id: str
) -> tuple[str, ...]:
    seen: set[str] = set()
    pending = list(concepts[("interpro", concept_id)]["parent_ids"])  # type: ignore[arg-type]
    while pending:
        parent = pending.pop()
        if parent in seen:
            continue
        seen.add(parent)
        pending.extend(concepts[("interpro", parent)]["parent_ids"])  # type: ignore[arg-type]
    return tuple(sorted(seen))


def _build_concept_context(
    concepts: Mapping[tuple[str, str], Mapping[str, object]],
    concept: Mapping[str, object],
    covariates: Sequence[Mapping[str, object]],
    eligible: tuple[str, ...],
    config: FeatureAtlasConfig,
    matrix_cache: dict[tuple[str, str | None], MatchingMatrix],
    *,
    database: str,
    concept_id: str,
) -> _HeldoutConceptContext:
    positives = tuple(sorted(
        set(concept["protein_accessions"]).intersection(eligible)  # type: ignore[arg-type]
    ))
    if not eligible:
        matched = MatchedPUSet(
            groups=(),
            covariate_names=(),
            unmatched_positives=(),
            propensity_sha256=hashlib.sha256(b"").hexdigest(),
        )
    else:
        ancestors = (
            _interpro_ancestors(concepts, concept_id)
            if database == "interpro" else ()
        )
        excluded_memberships = (
            {
                excluded_id: tuple(
                    concepts[("interpro", excluded_id)]["protein_accessions"]  # type: ignore[arg-type]
                )
                for excluded_id in (concept_id, *ancestors)
            }
            if database == "interpro" else None
        )
        matrix = _cached_matching_matrix(
            matrix_cache,
            covariates,
            eligible,
            database=database,
            concept_id=concept_id,
            interpro_ancestor_ids=ancestors,
            interpro_excluded_memberships=excluded_memberships,
        )
        if not positives or len(positives) == len(matrix.accessions):
            matched = MatchedPUSet(
                groups=(),
                covariate_names=matrix.covariate_names,
                unmatched_positives=positives,
                propensity_sha256=hashlib.sha256(b"").hexdigest(),
            )
        else:
            matched = match_positive_unlabeled(matrix, positives, config)
    return _HeldoutConceptContext(
        eligible=eligible,
        positives=positives,
        matched=matched,
        tier=evidence_tier(len(matched.groups), config),
    )


def _load_registered_covariates(
    covariates_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object], str]:
    covariates_path = Path(covariates_path)
    manifest_path = covariates_path.with_suffix(".manifest.json")
    expected_keys = {
        "accession_count", "complete", "concept_binding", "concept_index_sha256",
        "filtered_uniprot_sha256", "output", "output_bytes", "output_sha256",
        "schema_version", "units_sha256",
    }
    try:
        manifest = strict_json_load(manifest_path)
        output = manifest.get("output")
        observed_sha = sha256_file(covariates_path)
        observed_bytes = covariates_path.stat().st_size
        if (
            set(manifest) != expected_keys
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("complete") is not True
            or not isinstance(output, dict)
            or set(output) != {"bytes", "path", "sha256"}
            or output.get("path") != str(covariates_path.resolve())
            or output.get("sha256") != observed_sha
            or output.get("bytes") != observed_bytes
            or manifest.get("output_sha256") != observed_sha
            or manifest.get("output_bytes") != observed_bytes
        ):
            raise HeldoutEvaluationError("Registered covariate checksum or schema is invalid")
        rows = [
            _validate_covariate_row(row) for row in strict_json_lines(covariates_path)
        ]
    except HeldoutEvaluationError:
        raise
    except (AtlasIOError, OSError, ValueError) as error:
        raise HeldoutEvaluationError(f"Invalid registered covariates: {error}") from error
    accessions = [str(row["accession"]) for row in rows]
    if (
        manifest.get("accession_count") != len(rows)
        or len(accessions) != len(set(accessions))
    ):
        raise HeldoutEvaluationError("Registered covariate accessions are invalid")
    return rows, manifest, sha256_file(manifest_path)


def _canonical_pair_sha(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f'{row["feature_index"]}\t{row["database"]}\t{row["concept_id"]}\n'.encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _frozen_candidate_mapping(
    candidates: Sequence[Mapping[str, object]],
) -> dict[tuple[int, str, str], dict[str, str]]:
    frozen: dict[tuple[int, str, str], dict[str, str]] = {}
    for candidate in candidates:
        identity = (
            candidate.get("feature_index"), candidate.get("database"),
            candidate.get("concept_id"),
        )
        source_view = candidate.get("source_excluded")
        if (
            type(identity[0]) is not int
            or type(identity[1]) is not str
            or identity[1] not in _DATABASE_SET
            or type(identity[2]) is not str or not identity[2]
            or identity in frozen
            or type(candidate.get("concept_name")) is not str
            or not candidate["concept_name"]
            or type(candidate.get("concept_type")) is not str
            or not candidate["concept_type"]
            or not isinstance(source_view, Mapping)
            or source_view.get("direction") not in {"positive", "negative", "null"}
        ):
            raise HeldoutEvaluationError("Frozen Task 6 candidate mapping is invalid")
        frozen[identity] = {  # type: ignore[index]
            "concept_name": candidate["concept_name"],  # type: ignore[dict-item]
            "concept_type": candidate["concept_type"],  # type: ignore[dict-item]
            "training_direction": source_view["direction"],  # type: ignore[index]
        }
    return frozen


def _view_for_output(metrics: Mapping[str, object], q_value: float | None) -> dict[str, object]:
    return {
        "auprc": metrics["auprc"],
        "auprc_lift": metrics["auprc_lift"],
        "auroc": metrics["auroc"],
        "direction": metrics["direction"],
        "permutation_count": metrics["permutation_count"],
        "permutation_mode": metrics["permutation_mode"],
        "permutation_p_value": metrics["permutation_p_value"],
        "permutation_q_value": q_value,
        "prevalence": metrics["prevalence"],
        "rank_biserial_effect": metrics["rank_biserial_effect"],
    }


def _not_evaluated_view() -> dict[str, object]:
    return {
        "auprc": None,
        "auprc_lift": None,
        "auroc": None,
        "direction": "not_evaluated",
        "permutation_count": 0,
        "permutation_mode": "not_evaluated",
        "permutation_p_value": None,
        "permutation_q_value": None,
        "prevalence": None,
        "rank_biserial_effect": None,
    }


def _validate_view(view: object, *, primary: bool, evaluated: bool) -> None:
    if not isinstance(view, dict) or frozenset(view) != _VIEW_RESULT_KEYS:
        raise HeldoutEvaluationError("Held-out statistical view has invalid exact schema")
    if not evaluated:
        if view != _not_evaluated_view():
            raise HeldoutEvaluationError("Insufficient held-out view must be exactly unevaluated")
        return
    for name in ("auprc", "auprc_lift", "auroc", "permutation_p_value", "prevalence",
                 "rank_biserial_effect"):
        if type(view[name]) is not float:
            raise HeldoutEvaluationError(f"Held-out {name} must be a built-in float")
        _finite_number(view[name], f"held-out {name}")
    for name in ("auprc", "auroc", "permutation_p_value", "prevalence"):
        if not 0.0 <= float(view[name]) <= 1.0:
            raise HeldoutEvaluationError(f"Held-out {name} is outside [0, 1]")
    if float(view["auprc_lift"]) < 0.0:
        raise HeldoutEvaluationError("Held-out AUPRC lift is negative")
    effect = float(view["rank_biserial_effect"])
    if not -1.0 <= effect <= 1.0:
        raise HeldoutEvaluationError("Held-out rank effect is outside [-1, 1]")
    direction = "positive" if effect > 0 else "negative" if effect < 0 else "null"
    if view["direction"] != direction:
        raise HeldoutEvaluationError("Held-out direction does not match rank effect")
    q_value = view["permutation_q_value"]
    if primary:
        if type(q_value) is not float or not math.isfinite(q_value) or not 0 <= q_value <= 1:
            raise HeldoutEvaluationError("Held-out q-value is invalid")
    elif q_value is not None:
        raise HeldoutEvaluationError("All-source diagnostic q-value must be null")
    if (
        view["direction"] not in {"positive", "negative", "null"}
        or view["permutation_mode"] not in {"exact", "sampled"}
        or type(view["permutation_count"]) is not int or view["permutation_count"] <= 0
    ):
        raise HeldoutEvaluationError("Held-out statistical categorical values are invalid")


def _validate_result_row(row: object, *, width: int) -> tuple[int, str, str]:
    if not isinstance(row, dict) or frozenset(row) != _RESULT_KEYS:
        raise HeldoutEvaluationError("Held-out row has invalid exact schema")
    identity = (row["feature_index"], row["database"], row["concept_id"])
    if (
        row["schema_version"] != SCHEMA_VERSION
        or type(identity[0]) is not int or not 0 <= identity[0] < width
        or type(identity[1]) is not str or identity[1] not in _DATABASE_SET
        or type(identity[2]) is not str or not identity[2]
        or row["primary_view"] != "source_excluded"
        or row["evidence_tier"] not in {"insufficient", "descriptive", "confirmatory"}
        or row["status"] not in {
            "insufficient_validation_positives", "descriptive", "validated",
            "not_validated"
        }
        or type(row["confirmed"]) is not bool
        or type(row["seed"]) is not int or not 0 <= row["seed"] < 2**32
        or not _is_sha256(row["propensity_sha256"])
    ):
        raise HeldoutEvaluationError("Held-out row identity or status is invalid")
    if row["training_direction"] not in {"positive", "negative", "null"}:
        raise HeldoutEvaluationError("Held-out training direction is invalid")
    for name in ("concept_name", "concept_type"):
        if type(row[name]) is not str or not row[name]:
            raise HeldoutEvaluationError(f"Held-out {name} is invalid")
    for name in (
        "eligible_accession_count", "annotated_positive_count",
        "matched_positive_count", "matched_control_count",
    ):
        if type(row[name]) is not int or row[name] < 0:
            raise HeldoutEvaluationError("Held-out row count is invalid")
    if (
        row["matched_positive_count"] > row["annotated_positive_count"]
        or row["annotated_positive_count"] > row["eligible_accession_count"]
        or row["matched_positive_count"] + row["matched_control_count"]
        > row["eligible_accession_count"]
        or type(row["unmatched_positives"]) is not list
        or any(type(item) is not str or not item for item in row["unmatched_positives"])
        or row["unmatched_positives"] != sorted(set(row["unmatched_positives"]))
        or len(row["unmatched_positives"])
        != row["annotated_positive_count"] - row["matched_positive_count"]
        or type(row["covariate_names"]) is not list
        or any(type(item) is not str or not item for item in row["covariate_names"])
        or row["covariate_names"] != sorted(set(row["covariate_names"]))
    ):
        raise HeldoutEvaluationError("Held-out matching counts do not reconcile")
    evaluated = row["evidence_tier"] != "insufficient"
    _validate_view(row["source_excluded"], primary=True, evaluated=evaluated)
    _validate_view(row["all_source"], primary=False, evaluated=evaluated)
    return identity  # type: ignore[return-value]


def validate_heldout_evaluation(
    output_dir: Path,
    *,
    config_fingerprint: str | None = None,
    input_provenance: Mapping[str, object] | None = None,
    config: FeatureAtlasConfig | None = None,
    upstream_manifests: Mapping[str, object] | None = None,
    frozen_candidates: Mapping[tuple[int, str, str], Mapping[str, str]] | None = None,
) -> dict[str, object]:
    """Validate exact registered Task 7 schemas, checksums, and decisions."""

    output_dir = Path(output_dir)
    result_path = output_dir / _RESULT_NAME
    try:
        manifest = strict_json_load(output_dir / _MANIFEST_NAME)
        output = manifest.get("output")
        if (
            frozenset(manifest) != _MANIFEST_KEYS
            or not isinstance(output, dict)
            or set(output) != {"bytes", "path", "sha256"}
            or output.get("path") != _RESULT_NAME
            or output.get("sha256") != sha256_file(result_path)
            or output.get("bytes") != result_path.stat().st_size
        ):
            raise HeldoutEvaluationError("Held-out output checksum or exact schema is invalid")
    except HeldoutEvaluationError:
        raise
    except (AtlasIOError, OSError, ValueError) as error:
        raise HeldoutEvaluationError(f"Invalid held-out manifest or output: {error}") from error
    parameters = manifest.get("parameters")
    provenance = manifest.get("input_provenance")
    upstream = manifest.get("upstream_manifests")
    if (
        not isinstance(parameters, dict) or frozenset(parameters) != _PARAMETER_KEYS
        or parameters.get("propensity_model")
        != {"penalty": "l2", "random_state": 42, "solver": "liblinear"}
        or not isinstance(provenance, dict)
        or frozenset(provenance) != _INPUT_PROVENANCE_KEYS
        or any(not _is_sha256(value) for value in provenance.values())
        or not isinstance(upstream, dict) or frozenset(upstream) != _UPSTREAM_KEYS
        or any(not isinstance(value, dict) for value in upstream.values())
    ):
        raise HeldoutEvaluationError(
            "Held-out parameters, input_provenance, or upstream_manifests schema is invalid"
        )
    integer_parameters = (
        "controls_per_positive", "permutations", "seed",
        "validation_confirmatory_min", "validation_descriptive_min", "width",
    )
    float_parameters = (
        "heldout_fdr", "heldout_min_auprc_lift", "heldout_min_auroc",
        "propensity_caliper_sd",
    )
    if (
        any(type(parameters[name]) is not int or parameters[name] < 0 for name in integer_parameters)
        or parameters["controls_per_positive"] <= 0
        or parameters["permutations"] <= 0
        or parameters["validation_descriptive_min"] <= 0
        or parameters["validation_confirmatory_min"] < parameters["validation_descriptive_min"]
        or parameters["width"] <= 0
        or any(
            type(parameters[name]) is not float or not math.isfinite(parameters[name])
            for name in float_parameters
        )
        or not 0.0 < parameters["heldout_fdr"] <= 1.0
        or not 0.0 <= parameters["heldout_min_auroc"] <= 1.0
        or parameters["heldout_min_auprc_lift"] <= 0.0
        or parameters["propensity_caliper_sd"] <= 0.0
    ):
        raise HeldoutEvaluationError("Held-out parameters have invalid values")
    width = parameters["width"]
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or type(manifest.get("candidate_count")) is not int
        or manifest["candidate_count"] < 0
        or type(manifest.get("validation_accession_count")) is not int
        or manifest["validation_accession_count"] <= 0
        or type(width) is not int or width <= 0
    ):
        raise HeldoutEvaluationError("Held-out manifest completion schema is invalid")
    if config_fingerprint is not None and manifest.get("config_fingerprint") != config_fingerprint:
        raise HeldoutEvaluationError("Held-out configuration fingerprint changed")
    if input_provenance is not None and manifest.get("input_provenance") != dict(input_provenance):
        raise HeldoutEvaluationError("Held-out upstream input provenance changed")
    if upstream_manifests is not None and upstream != dict(upstream_manifests):
        raise HeldoutEvaluationError("Held-out copied upstream manifests changed")
    if config is not None:
        if not isinstance(config, FeatureAtlasConfig):
            raise TypeError("config must be a FeatureAtlasConfig")
        expected_parameters = {
            "controls_per_positive": config.controls_per_positive,
            "heldout_fdr": config.heldout_fdr,
            "heldout_min_auprc_lift": config.heldout_min_auprc_lift,
            "heldout_min_auroc": config.heldout_min_auroc,
            "permutations": config.permutations,
            "propensity_caliper_sd": config.propensity_caliper_sd,
            "propensity_model": {
                "penalty": "l2", "random_state": 42, "solver": "liblinear"
            },
            "seed": config.seed,
            "validation_confirmatory_min": config.validation_confirmatory_min,
            "validation_descriptive_min": config.validation_descriptive_min,
            "width": config.width,
        }
        if parameters != expected_parameters:
            raise HeldoutEvaluationError("Held-out parameters changed from configuration")
    if not isinstance(frozen_candidates, Mapping):
        raise HeldoutEvaluationError("Frozen Task 6 candidate mapping is required")
    normalized_candidates: dict[tuple[int, str, str], dict[str, str]] = {}
    for identity, metadata in frozen_candidates.items():
        if (
            type(identity) is not tuple or len(identity) != 3
            or type(identity[0]) is not int
            or type(identity[1]) is not str or identity[1] not in _DATABASE_SET
            or type(identity[2]) is not str or not identity[2]
            or not isinstance(metadata, Mapping)
            or set(metadata) != {"concept_name", "concept_type", "training_direction"}
            or type(metadata["concept_name"]) is not str or not metadata["concept_name"]
            or type(metadata["concept_type"]) is not str or not metadata["concept_type"]
            or metadata["training_direction"] not in {"positive", "negative", "null"}
        ):
            raise HeldoutEvaluationError("Frozen Task 6 candidate mapping is invalid")
        normalized_candidates[identity] = dict(metadata)
    previous: tuple[int, str, str] | None = None
    row_count = 0
    seen: set[tuple[int, str, str]] = set()
    pair_digest = hashlib.sha256()
    database_rows: dict[str, list[dict[str, object]]] = {database: [] for database in DATABASES}
    for row in strict_json_lines(result_path):
        row_count += 1
        identity = _validate_result_row(row, width=width)
        if previous is not None and identity <= previous:
            raise HeldoutEvaluationError("Held-out rows are not deterministically sorted and unique")
        previous = identity
        seen.add(identity)
        pair_digest.update(
            f'{row["feature_index"]}\t{row["database"]}\t{row["concept_id"]}\n'.encode("utf-8")
        )
        candidate_metadata = normalized_candidates.get(identity)
        if candidate_metadata != {
            "concept_name": row["concept_name"],
            "concept_type": row["concept_type"],
            "training_direction": row["training_direction"],
        }:
            raise HeldoutEvaluationError(
                "Held-out row disagrees with frozen Task 6 candidate"
            )
        matched_positive = row["matched_positive_count"]
        matched_control = row["matched_control_count"]
        expected_tier = (
            "insufficient" if matched_positive < parameters["validation_descriptive_min"]
            else "descriptive" if matched_positive < parameters["validation_confirmatory_min"]
            else "confirmatory"
        )
        if row["evidence_tier"] != expected_tier:
            raise HeldoutEvaluationError("Held-out evidence tier does not match counts")
        if (
            matched_control > matched_positive * parameters["controls_per_positive"]
            or (matched_positive > 0 and expected_tier != "insufficient"
                and matched_control < matched_positive)
            or row["seed"] != pair_seed(
                parameters["seed"], row["feature_index"], row["database"], row["concept_id"]
            )
        ):
            raise HeldoutEvaluationError("Held-out matching or seed reconciliation failed")
        if expected_tier == "insufficient":
            if row["confirmed"] is not False or row["status"] != "insufficient_validation_positives":
                raise HeldoutEvaluationError("Insufficient held-out decision is invalid")
        else:
            for view_name in ("source_excluded", "all_source"):
                view = row[view_name]
                prevalence = matched_positive / (matched_positive + matched_control)
                if (
                    not math.isclose(view["prevalence"], prevalence, rel_tol=1e-15, abs_tol=0.0)
                    or not math.isclose(
                        view["auprc_lift"], view["auprc"] / prevalence,
                        rel_tol=1e-15, abs_tol=0.0,
                    )
                ):
                    raise HeldoutEvaluationError("Held-out prevalence or lift does not reconcile")
            database_rows[row["database"]].append({
                "evidence_tier": row["evidence_tier"],
                "training_direction": row["training_direction"],
                "confirmed": row["confirmed"],
                "status": row["status"],
                "source_excluded": row["source_excluded"],
            })
    if row_count != manifest["candidate_count"]:
        raise HeldoutEvaluationError("Held-out candidate count mismatch")
    if set(normalized_candidates) != seen:
        raise HeldoutEvaluationError(
            "Held-out rows do not completely cover frozen Task 6 candidates"
        )
    if manifest.get("pair_list_sha256") != pair_digest.hexdigest():
        raise HeldoutEvaluationError("Held-out pair-list checksum mismatch")
    for database in DATABASES:
        family = database_rows[database]
        q_values = benjamini_hochberg([
            row["source_excluded"]["permutation_p_value"] for row in family
        ])
        for row, expected_q in zip(family, q_values, strict=True):
            if row["source_excluded"]["permutation_q_value"] != expected_q:
                raise HeldoutEvaluationError("Held-out per-database BH q-value changed")
            confirmed = bool(
                row["evidence_tier"] == "confirmatory"
                and expected_q < parameters["heldout_fdr"]
                and row["source_excluded"]["auroc"] >= parameters["heldout_min_auroc"]
                and row["source_excluded"]["auprc_lift"]
                >= parameters["heldout_min_auprc_lift"]
                and row["source_excluded"]["direction"] == row["training_direction"]
            )
            status = (
                "validated" if confirmed else "not_validated"
                if row["evidence_tier"] == "confirmatory" else "descriptive"
            )
            if row["confirmed"] is not confirmed or row["status"] != status:
                raise HeldoutEvaluationError("Held-out confirmation decision changed")
    return manifest


def _publish_heldout(
    stage: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    config_fingerprint: str,
    input_provenance: Mapping[str, object],
    frozen_candidates: Mapping[tuple[int, str, str], Mapping[str, str]],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / _RESULT_NAME, output_dir / _MANIFEST_NAME]
    if any(path.exists() for path in targets) and not overwrite:
        if not all(path.exists() for path in targets):
            raise HeldoutEvaluationError("Incomplete held-out output requires explicit overwrite")
        shutil.rmtree(stage, ignore_errors=True)
        return validate_heldout_evaluation(
            output_dir, config_fingerprint=config_fingerprint,
            input_provenance=input_provenance,
            frozen_candidates=frozen_candidates,
        )
    backup = Path(tempfile.mkdtemp(prefix=".heldout-backup-", dir=output_dir.parent))
    existed = [path.exists() for path in targets]
    moved_old = [False, False]
    committed_new = [False, False]
    preserve_recovery = False
    try:
        for index, (present, target) in enumerate(zip(existed, targets, strict=True)):
            if present:
                os.replace(target, backup / target.name)
                moved_old[index] = True
        for index, name in enumerate((_RESULT_NAME, _MANIFEST_NAME)):
            os.replace(stage / name, targets[index])
            committed_new[index] = True
        validated = validate_heldout_evaluation(
            output_dir, config_fingerprint=config_fingerprint,
            input_provenance=input_provenance,
            frozen_candidates=frozen_candidates,
        )
    except BaseException as error:
        rollback_errors: list[str] = []
        for index, target in enumerate(targets):
            try:
                if committed_new[index] and target.exists():
                    target.unlink()
            except BaseException as rollback_error:
                rollback_errors.append(f"cannot remove replacement {target.name}: {rollback_error}")
        for index, target in enumerate(targets):
            saved = backup / target.name
            try:
                if moved_old[index] and saved.exists():
                    os.replace(saved, target)
            except BaseException as rollback_error:
                rollback_errors.append(f"cannot restore {target.name}: {rollback_error}")
        if rollback_errors:
            preserve_recovery = True
            raise HeldoutEvaluationError(
                "Held-out publication recovery required; "
                f"backup recovery path: {backup}; staging recovery path: {stage}; "
                + "; ".join(rollback_errors)
            ) from error
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(error, HeldoutEvaluationError):
            raise
        raise HeldoutEvaluationError(f"Cannot atomically publish held-out output: {error}") from error
    finally:
        if not preserve_recovery:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
    return validated


def evaluate_heldout_candidates(
    train_candidates_path: Path,
    protein_activation_dir: Path,
    covariates_path: Path,
    concept_index_path: Path,
    config: FeatureAtlasConfig,
    output_dir: Path,
    overwrite: bool = False,
    *,
    snapshot_dir: Path | None = None,
    units_path: Path | None = None,
    allow_test_fixture_unbound: bool = False,
) -> dict[str, object]:
    """Evaluate frozen train candidates on registered validation data and publish atomically."""

    if not all(isinstance(path, Path) for path in (
        train_candidates_path, protein_activation_dir, covariates_path,
        concept_index_path, output_dir,
    )):
        raise TypeError("Task 7 paths must be pathlib.Path values")
    if not isinstance(config, FeatureAtlasConfig):
        raise TypeError("config must be a FeatureAtlasConfig")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a built-in bool")
    if type(allow_test_fixture_unbound) is not bool:
        raise TypeError("allow_test_fixture_unbound must be a built-in bool")
    if train_candidates_path.name != "train_candidates.jsonl":
        raise HeldoutEvaluationError("Task 6 candidate path has an invalid filename")
    try:
        discovery_manifest = validate_train_discovery(train_candidates_path.parent)
        candidates = list(strict_json_lines(train_candidates_path))
        concepts, concept_manifest, concept_manifest_sha = _load_registered_concepts(
            concept_index_path,
            snapshot_dir=snapshot_dir,
            units_path=units_path,
            allow_test_fixture_unbound=allow_test_fixture_unbound,
        )
        covariates, covariate_manifest, covariate_manifest_sha = _load_registered_covariates(
            covariates_path
        )
        protein_manifest_path = protein_activation_dir / "manifest.json"
        protein_manifest = strict_json_load(protein_manifest_path)
        validation_rows = list(
            iter_protein_activation_rows(protein_activation_dir, split="validation")
        )
    except HeldoutEvaluationError:
        raise
    except (AtlasIOError, OSError, ValueError) as error:
        raise HeldoutEvaluationError(f"Invalid registered held-out input: {error}") from error
    if (
        discovery_manifest.get("width") != config.width
        or protein_manifest.get("width") != config.width
        or discovery_manifest.get("candidate_count") != len(candidates)
    ):
        raise HeldoutEvaluationError("Held-out input width or candidate count mismatch")
    frozen_candidates = _frozen_candidate_mapping(candidates)
    protein_manifest_sha = sha256_file(protein_manifest_path)
    discovery_input = discovery_manifest.get("input_provenance")
    discovery_upstream = discovery_manifest.get("upstream_bindings")
    discovery_sources = (
        discovery_upstream.get("source_provenance")
        if isinstance(discovery_upstream, dict) else None
    )
    if (
        not isinstance(discovery_input, dict)
        or discovery_input.get("protein_activation_manifest_sha256")
        != protein_manifest_sha
        or discovery_sources != protein_manifest.get("source_provenance")
    ):
        raise HeldoutEvaluationError(
            "Task 6 candidates and current Task 5 activations do not share one registered source"
        )
    protein_binding = protein_manifest.get("concept_binding")
    covariate_binding = covariate_manifest.get("concept_binding")
    expected_binding = {
        "index_sha256": sha256_file(concept_index_path),
        "manifest_sha256": concept_manifest_sha,
    }
    for label, binding in (("protein activations", protein_binding),
                           ("covariates", covariate_binding)):
        if not isinstance(binding, dict) or any(
            binding.get(name) != value for name, value in expected_binding.items()
        ):
            raise HeldoutEvaluationError(f"Task 3 binding mismatch in {label}")
        if binding.get("test_fixture_unbound") is not allow_test_fixture_unbound:
            raise HeldoutEvaluationError(f"Task 3 fixture binding mismatch in {label}")
    if not allow_test_fixture_unbound:
        discovery_provenance = discovery_manifest.get("input_provenance")
        discovery_upstream = discovery_manifest.get("upstream_bindings")
        discovery_concept_binding = (
            discovery_upstream.get("concept_binding")
            if isinstance(discovery_upstream, dict) else None
        )
        if (
            not isinstance(discovery_provenance, dict)
            or discovery_provenance.get("concept_index_sha256")
            != expected_binding["index_sha256"]
            or not isinstance(discovery_concept_binding, dict)
            or any(
                discovery_concept_binding.get(name) != value
                for name, value in expected_binding.items()
            )
        ):
            raise HeldoutEvaluationError("Task 6 and Task 3 concept bindings disagree")
    validation_accessions = tuple(row.accession for row in validation_rows)
    if len(validation_accessions) != len(set(validation_accessions)):
        raise HeldoutEvaluationError("Validation activation accessions are invalid")
    covariate_accessions = {str(row["accession"]) for row in covariates}
    if not set(validation_accessions).issubset(covariate_accessions):
        raise HeldoutEvaluationError("Validation activation has missing covariates")
    activation_by_accession = {row.accession: row for row in validation_rows}
    eligible_by_database: dict[str, tuple[str, ...]] = {}
    matrix_cache: dict[tuple[str, str | None], MatchingMatrix] = {}
    context_cache: dict[tuple[str, str], _HeldoutConceptContext] = {}
    provisional: list[dict[str, object]] = []
    evaluation_started = time.monotonic()
    total_candidates = len(candidates)
    for processed, candidate in enumerate(candidates, start=1):
        feature = candidate["feature_index"]
        database = candidate["database"]
        concept_id = candidate["concept_id"]
        identity = (database, concept_id)
        concept = concepts.get(identity)
        if concept is None:
            raise HeldoutEvaluationError("Frozen candidate is absent from Task 3 concept index")
        if (
            candidate.get("concept_name") != concept.get("concept_name")
            or candidate.get("concept_type") != concept.get("concept_type")
        ):
            raise HeldoutEvaluationError("Frozen candidate concept metadata changed")
        eligible = eligible_by_database.get(database)
        if eligible is None:
            eligible = tuple(
                accession for accession in validation_accessions
                if activation_by_accession[accession].eligibility.get(database) is True
            )
            if any(
                database not in activation_by_accession[item].source_excluded
                for item in eligible
            ):
                raise HeldoutEvaluationError(
                    "Eligible held-out source-excluded score is missing"
                )
            eligible_by_database[database] = eligible
        context = context_cache.get(identity)
        if context is None:
            context = _build_concept_context(
                concepts,
                concept,
                covariates,
                eligible,
                config,
                matrix_cache,
                database=database,
                concept_id=concept_id,
            )
            context_cache[identity] = context
        seed = pair_seed(config.seed, feature, database, concept_id)
        if context.tier == "insufficient":
            source_metrics = None
            all_metrics = None
        else:
            matched_accessions = tuple(
                accession
                for group in context.matched.groups
                for accession in (group.positive, *group.controls)
            )
            source_scores = {
                accession: float(
                    activation_by_accession[accession]
                    .source_excluded[database]
                    .get(feature, 0.0)
                )
                for accession in matched_accessions
            }
            all_scores = {
                accession: float(
                    activation_by_accession[accession].all_sources.get(feature, 0.0)
                )
                for accession in matched_accessions
            }
            source_metrics = evaluate_matched_pair(
                source_scores, context.matched, config.permutations, seed
            )
            all_metrics = evaluate_matched_pair(
                all_scores, context.matched, config.permutations, seed
            )
        provisional.append({
            "candidate": candidate,
            "eligible_count": len(context.eligible),
            "matched": context.matched,
            "positives": context.positives,
            "seed": seed,
            "source_metrics": source_metrics,
            "all_metrics": all_metrics,
            "tier": context.tier,
            "training_direction": candidate["source_excluded"]["direction"],
        })
        if _should_report_progress(processed, total_candidates):
            print(
                "heldout progress "
                f"processed={processed}/{total_candidates} "
                f"contexts={len(context_cache)} matrices={len(matrix_cache)} "
                f"elapsed_seconds={time.monotonic() - evaluation_started:.1f}",
                flush=True,
            )
    evaluated = [item for item in provisional if item["source_metrics"] is not None]
    decision_inputs = [{
        "database": item["candidate"]["database"],  # type: ignore[index]
        "permutation_p_value": item["source_metrics"]["permutation_p_value"],  # type: ignore[index]
        "tier": item["tier"],
        "auroc": item["source_metrics"]["auroc"],  # type: ignore[index]
        "auprc_lift": item["source_metrics"]["auprc_lift"],  # type: ignore[index]
        "direction": item["source_metrics"]["direction"],  # type: ignore[index]
        "training_direction": item["training_direction"],
    } for item in evaluated]
    decisions = _apply_confirmatory_decisions(decision_inputs, config)
    decision_by_identity = {
        (item["candidate"]["feature_index"], item["candidate"]["database"],  # type: ignore[index]
         item["candidate"]["concept_id"]): decision  # type: ignore[index]
        for item, decision in zip(evaluated, decisions, strict=True)
    }
    output_rows: list[dict[str, object]] = []
    for item in provisional:
        candidate = item["candidate"]
        matched = item["matched"]
        positives = item["positives"]
        source_metrics = item["source_metrics"]
        all_metrics = item["all_metrics"]
        assert isinstance(candidate, dict) and isinstance(matched, MatchedPUSet)
        assert isinstance(positives, tuple)
        identity = (candidate["feature_index"], candidate["database"], candidate["concept_id"])
        decision = decision_by_identity.get(identity)
        if source_metrics is None:
            source_view = _not_evaluated_view()
            all_view = _not_evaluated_view()
            confirmed = False
            status = "insufficient_validation_positives"
        else:
            assert isinstance(source_metrics, dict) and isinstance(all_metrics, dict)
            assert decision is not None
            source_view = _view_for_output(
                source_metrics, float(decision["permutation_q_value"])
            )
            all_view = _view_for_output(all_metrics, None)
            confirmed = decision["confirmed"]
            status = decision["status"]
        output_rows.append({
            "all_source": all_view,
            "annotated_positive_count": len(positives),
            "confirmed": confirmed,
            "concept_id": candidate["concept_id"],
            "concept_name": candidate["concept_name"],
            "concept_type": candidate["concept_type"],
            "covariate_names": list(matched.covariate_names),
            "database": candidate["database"],
            "eligible_accession_count": item["eligible_count"],
            "evidence_tier": item["tier"],
            "feature_index": candidate["feature_index"],
            "matched_control_count": sum(len(group.controls) for group in matched.groups),
            "matched_positive_count": len(matched.groups),
            "primary_view": "source_excluded",
            "propensity_sha256": matched.propensity_sha256,
            "schema_version": SCHEMA_VERSION,
            "seed": item["seed"],
            "source_excluded": source_view,
            "status": status,
            "training_direction": item["training_direction"],
            "unmatched_positives": list(matched.unmatched_positives),
        })
    output_rows.sort(key=lambda row: (row["feature_index"], row["database"], row["concept_id"]))
    input_provenance = {
        "concept_index_sha256": sha256_file(concept_index_path),
        "concept_manifest_sha256": concept_manifest_sha,
        "covariates_manifest_sha256": covariate_manifest_sha,
        "covariates_sha256": sha256_file(covariates_path),
        "discovery_manifest_sha256": sha256_file(train_candidates_path.parent / "discovery_manifest.json"),
        "protein_activation_manifest_sha256": sha256_file(protein_activation_dir / "manifest.json"),
        "train_candidates_sha256": sha256_file(train_candidates_path),
    }
    upstream_payload = {
        "concept": concept_manifest,
        "covariates": covariate_manifest,
        "discovery": discovery_manifest,
        "protein_activations": protein_manifest,
    }
    fingerprint = config.fingerprint()
    existing = output_dir / _MANIFEST_NAME
    result_path = output_dir / _RESULT_NAME
    if not overwrite and existing.exists() and result_path.exists():
        return validate_heldout_evaluation(
            output_dir, config_fingerprint=fingerprint,
            input_provenance=input_provenance, config=config,
            upstream_manifests=upstream_payload,
            frozen_candidates=frozen_candidates,
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".heldout-stage-", dir=output_dir.parent))
    try:
        staged_result = stage / _RESULT_NAME
        write_jsonl_atomic(staged_result, output_rows)
        manifest = {
            "candidate_count": len(output_rows),
            "complete": True,
            "config_fingerprint": fingerprint,
            "input_provenance": input_provenance,
            "output": {
                "bytes": staged_result.stat().st_size,
                "path": _RESULT_NAME,
                "sha256": sha256_file(staged_result),
            },
            "pair_list_sha256": _canonical_pair_sha(output_rows),
            "parameters": {
                "controls_per_positive": config.controls_per_positive,
                "heldout_fdr": config.heldout_fdr,
                "heldout_min_auprc_lift": config.heldout_min_auprc_lift,
                "heldout_min_auroc": config.heldout_min_auroc,
                "permutations": config.permutations,
                "propensity_caliper_sd": config.propensity_caliper_sd,
                "propensity_model": {
                    "penalty": "l2", "random_state": 42, "solver": "liblinear"
                },
                "seed": config.seed,
                "validation_confirmatory_min": config.validation_confirmatory_min,
                "validation_descriptive_min": config.validation_descriptive_min,
                "width": config.width,
            },
            "schema_version": SCHEMA_VERSION,
            "upstream_manifests": upstream_payload,
            "validation_accession_count": len(validation_rows),
        }
        write_json_atomic(stage / _MANIFEST_NAME, manifest)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return _publish_heldout(
        stage, output_dir, overwrite=overwrite, config_fingerprint=fingerprint,
        input_provenance=input_provenance, frozen_candidates=frozen_candidates,
    )
