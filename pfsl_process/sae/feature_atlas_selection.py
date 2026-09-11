"""Bounded full-feature atlas assembly and immutable deep-feature selection.

Same-source results are retained only as field diagnostics.  Biological
eligibility is derived exclusively from validated source-excluded results.
Splitting and absorption outputs are deliberately named candidate diagnostics;
they are not biological conclusions.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .feature_atlas_config import (
    AtlasPaths,
    AtlasSourceBinding,
    EXPECTED_FEATURE_STATS_SHA256,
    EXPECTED_INFERENCE_SHA256,
    EXPECTED_SPLIT_MANIFEST_SHA256,
    EXPECTED_UNITS_SHA256,
    EXPECTED_ACTIVATION_MANIFEST_SHA256,
    FeatureAtlasConfig,
    validate_source_binding,
)
from .feature_atlas_io import (
    AtlasIOError,
    file_metadata,
    sha256_file,
    strict_json_lines,
    strict_json_load,
    write_json_atomic,
    write_jsonl_atomic,
)
from . import sae_training_data
from .concept_parsers import validate_concept_index
from .feature_concept_discovery import (
    DiscoveryError,
    _validate_candidate_row,
    validate_train_discovery,
)
from .heldout_concept_evaluation import (
    HeldoutEvaluationError,
    _validate_result_row,
    validate_heldout_evaluation,
)
from .protein_feature_activations import (
    iter_protein_activation_rows,
    validate_protein_activations,
    validate_protein_covariates,
)
from .sparse_feature_activations import (
    load_frozen_sae,
    validate_sparse_feature_activations,
)


SCHEMA_VERSION = 1
DATABASES = ("go", "reactome", "interpro", "corum")
CLASSIFICATIONS = frozenset(
    {
        "field_detector",
        "validated_biological_concept",
        "mixed",
        "exploratory",
        "dead_or_too_sparse",
        "unresolved",
    }
)
SELECTION_CATEGORIES = frozenset({"biological", "field_detector", "control"})
_SHA_CHARS = frozenset("0123456789abcdef")
_ATLAS_INPUT_KEYS = frozenset({
    "train_concept_results", "contrasts", "dead", "decoder_norm", "feature_index",
    "firing_accessions", "preferred_aspect", "source_effect_loss",
    "heldout_concept_results", "source_lift", "source_purity",
    "split_reproducibility", "splitting_cluster", "too_sparse", "top_protein_candidates",
    "top_text_candidates", "train_activation_count", "train_concept_ids",
    "train_max_activation", "train_mean_activation", "train_split_coverage",
    "validation_activation_count", "validation_concept_ids",
    "validation_max_activation", "validation_mean_activation",
    "validation_split_coverage",
})
_ATLAS_ROW_KEYS = frozenset(
    (_ATLAS_INPUT_KEYS - {"top_protein_candidates", "top_text_candidates"})
    | {
        "absorption_diagnostics", "classification", "schema_version",
        "selection_category", "selection_fingerprint", "selection_reason", "source_provenance",
        "splitting_cluster", "splitting_diagnostics", "splitting_parent",
        "top_proteins", "top_texts",
        "upstream_artifacts",
    }
)
_TASK6_KEYS = frozenset({
    "all_source", "concept_id", "concept_name", "concept_type", "database",
    "eligible_accession_count", "feature_index", "schema_version",
    "source_excluded", "source_exclusion_change",
})
_TASK7_KEYS = frozenset({
    "all_source", "annotated_positive_count", "confirmed", "concept_id",
    "concept_name", "concept_type", "covariate_names", "database",
    "eligible_accession_count", "evidence_tier", "feature_index",
    "matched_control_count", "matched_positive_count", "primary_view",
    "propensity_sha256", "schema_version", "seed", "source_excluded",
    "status", "training_direction", "unmatched_positives",
})
_TASK7_COMPACT_KEYS = frozenset({
    "all_source", "annotated_positive_count", "confirmed", "concept_id",
    "concept_name", "concept_type", "database", "eligible_accession_count",
    "evidence_tier", "feature_index", "matched_control_count",
    "matched_positive_count", "primary_view", "schema_version",
    "source_excluded", "status", "training_direction",
})
_TOP_TEXT_KEYS = frozenset({
    "accession", "activation", "aspect_index", "row_index", "split", "text", "unit_id",
})
_TOP_PROTEIN_KEYS = frozenset({"accession", "activation", "split"})
_CONTRAST_KEYS = frozenset({
    "aspect_index", "covariate_distance", "high_accession", "high_activation",
    "low_accession", "low_activation", "split",
})
_TASK6_VIEW_KEYS = frozenset({
    "annotated_count", "annotated_prevalence", "firing_count", "overlap_count",
    "firing_only_count", "annotated_only_count", "neither_count",
    "fisher_odds_ratio", "fisher_p_value", "rank_biserial_effect",
    "rank_p_value", "direction",
})
_TASK7_VIEW_KEYS = frozenset({
    "auprc", "auprc_lift", "auroc", "direction", "permutation_count",
    "permutation_mode", "permutation_p_value", "permutation_q_value",
    "prevalence", "rank_biserial_effect",
})


def _compact_heldout_result(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "all_source": row["all_source"],
        "annotated_positive_count": row["annotated_positive_count"],
        "confirmed": row["confirmed"],
        "concept_id": row["concept_id"],
        "concept_name": row["concept_name"],
        "concept_type": row["concept_type"],
        "database": row["database"],
        "eligible_accession_count": row["eligible_accession_count"],
        "evidence_tier": row["evidence_tier"],
        "feature_index": row["feature_index"],
        "matched_control_count": row["matched_control_count"],
        "matched_positive_count": row["matched_positive_count"],
        "primary_view": row["primary_view"],
        "schema_version": row["schema_version"],
        "source_excluded": row["source_excluded"],
        "status": row["status"],
        "training_direction": row["training_direction"],
    }
_SPLIT_DIAGNOSTIC_KEYS = frozenset({
    "decoder_cosine", "firing_jaccard", "kind", "other_feature_index", "shared_concepts",
})
_ABSORPTION_DIAGNOSTIC_KEYS = frozenset({
    "ancestor_related", "concept_identities", "concept_jaccard", "kind",
})


class FeatureAtlasSelectionError(ValueError):
    """Raised when atlas evidence or a frozen selection violates its contract."""


def _validate_atlas_row_schema(
    row: Mapping[str, object],
    config: FeatureAtlasConfig,
    *,
    frozen: bool,
    allow_test_fixture_unbound: bool,
    expected_source_provenance: Mapping[str, object],
    expected_decoder: torch.Tensor | None,
) -> None:
    if type(allow_test_fixture_unbound) is not bool:
        raise TypeError("allow_test_fixture_unbound must be a built-in bool")
    if set(row) != _ATLAS_ROW_KEYS:
        raise FeatureAtlasSelectionError("atlas row has invalid exact schema")
    feature = row.get("feature_index")
    if type(feature) is not int or not 0 <= feature < config.width:
        raise FeatureAtlasSelectionError("atlas feature index is invalid")
    if row.get("schema_version") != SCHEMA_VERSION:
        raise FeatureAtlasSelectionError("atlas schema version is invalid")
    if row.get("classification") not in CLASSIFICATIONS:
        raise FeatureAtlasSelectionError("atlas classification is invalid")
    if any(type(row.get(name)) is not bool for name in ("dead", "too_sparse")):
        raise FeatureAtlasSelectionError("atlas dead/sparse flags are invalid")
    for name in (
        "decoder_norm", "source_effect_loss", "source_lift", "source_purity",
        "split_reproducibility", "train_max_activation", "train_mean_activation",
        "train_split_coverage", "validation_max_activation",
        "validation_mean_activation", "validation_split_coverage",
    ):
        _finite_float(row.get(name), f"atlas {name}")
    for name in ("train_activation_count", "validation_activation_count"):
        if type(row.get(name)) is not int or row[name] < 0:
            raise FeatureAtlasSelectionError(f"atlas {name} is invalid")
    if row["decoder_norm"] <= 0.0:
        raise FeatureAtlasSelectionError("atlas decoder_norm must be positive")
    for name in (
        "source_lift", "train_mean_activation",
        "train_max_activation", "validation_mean_activation",
        "validation_max_activation",
    ):
        if row[name] < 0.0:
            raise FeatureAtlasSelectionError(f"atlas {name} cannot be negative")
    for name in (
        "source_effect_loss", "source_purity", "split_reproducibility",
        "train_split_coverage", "validation_split_coverage",
    ):
        if not 0.0 <= row[name] <= 1.0:
            raise FeatureAtlasSelectionError(f"atlas {name} must be in [0, 1]")
    for split in ("train", "validation"):
        count = row[f"{split}_activation_count"]
        mean = row[f"{split}_mean_activation"]
        maximum = row[f"{split}_max_activation"]
        coverage = row[f"{split}_split_coverage"]
        if mean > maximum:
            raise FeatureAtlasSelectionError(f"atlas {split} mean exceeds maximum")
        if count == 0 and (mean != 0.0 or maximum != 0.0 or coverage != 0.0):
            raise FeatureAtlasSelectionError(f"atlas {split} empty summary is inconsistent")
        if count > 0 and coverage <= 0.0:
            raise FeatureAtlasSelectionError(f"atlas {split} nonempty coverage is invalid")
    expected_reproducibility = min(
        row["train_activation_count"], row["validation_activation_count"]
    ) / max(row["train_activation_count"], row["validation_activation_count"], 1)
    if row["split_reproducibility"] != expected_reproducibility:
        raise FeatureAtlasSelectionError("atlas split reproducibility does not reconcile")
    if expected_decoder is None:
        if not allow_test_fixture_unbound:
            raise FeatureAtlasSelectionError("production atlas validation requires decoder")
    elif (
        not isinstance(expected_decoder, torch.Tensor)
        or expected_decoder.dtype != torch.float32
        or expected_decoder.ndim != 2
        or expected_decoder.shape[1] != config.width
        or (not allow_test_fixture_unbound and expected_decoder.shape[0] != 1024)
    ):
        raise FeatureAtlasSelectionError("expected decoder shape or dtype is invalid")
    provenance = row.get("source_provenance")
    if not isinstance(provenance, dict):
        raise FeatureAtlasSelectionError("atlas source provenance is invalid")
    if provenance != dict(expected_source_provenance):
        raise FeatureAtlasSelectionError("atlas source provenance differs from caller binding")
    if not allow_test_fixture_unbound:
        if (
            "test_fixture_unbound" in provenance
            or
            set(provenance) != set(AtlasSourceBinding.__dataclass_fields__)
            or any(not _is_sha256(value) for value in provenance.values())
        ):
            raise FeatureAtlasSelectionError("atlas production provenance has invalid exact schema")
    elif (
        provenance.get("test_fixture_unbound") is not True
        or any(
            key != "test_fixture_unbound" and not _is_sha256(value)
            for key, value in provenance.items()
        )
    ):
        raise FeatureAtlasSelectionError("atlas fixture provenance is invalid")
    upstream = row.get("upstream_artifacts")
    if not isinstance(upstream, dict):
        raise FeatureAtlasSelectionError("atlas upstream artifact binding is invalid")
    if not allow_test_fixture_unbound:
        expected_upstreams = {
            "concept_manifest", "covariate_manifest", "discovery_manifest",
            "heldout_manifest", "protein_manifest", "sparse_manifest",
        }
        if set(upstream) != expected_upstreams or any(
            not isinstance(value, dict)
            or set(value) != {"bytes", "path", "sha256"}
            or not _is_sha256(value.get("sha256"))
            or type(value.get("bytes")) is not int or value["bytes"] <= 0
            or type(value.get("path")) is not str or not value["path"]
            or not Path(value["path"]).is_absolute()
            for value in upstream.values()
        ):
            raise FeatureAtlasSelectionError("atlas upstream artifacts have invalid exact schema")
    train_results, heldout_results = row.get("train_concept_results"), row.get("heldout_concept_results")
    if not isinstance(train_results, list) or not isinstance(heldout_results, list):
        raise FeatureAtlasSelectionError("atlas concept result lists are invalid")
    if not allow_test_fixture_unbound:
        for result in train_results:
            if not isinstance(result, dict) or frozenset(result) != _TASK6_KEYS or result.get("feature_index") != feature:
                raise FeatureAtlasSelectionError("atlas Task 6 result has invalid exact schema or link")
            all_view, excluded_view = result.get("all_source"), result.get("source_excluded")
            if (
                not isinstance(all_view, dict) or frozenset(all_view) != _TASK6_VIEW_KEYS
                or not isinstance(excluded_view, dict)
                or frozenset(excluded_view) != _TASK6_VIEW_KEYS | {"fisher_q_value", "rank_q_value"}
            ):
                raise FeatureAtlasSelectionError("atlas Task 6 nested statistical schema is invalid")
            try:
                identity = _validate_candidate_row(
                    result, width=config.width,
                    train_candidate_fdr=config.train_candidate_fdr,
                )
            except (DiscoveryError, TypeError, ValueError) as error:
                raise FeatureAtlasSelectionError(
                    f"atlas Task 6 result failed public validation: {error}"
                ) from error
            if identity[0] != feature:
                raise FeatureAtlasSelectionError("atlas Task 6 outer feature link changed")
        train_pairs = {(item["database"], item["concept_id"]) for item in train_results}
        for result in heldout_results:
            if (
                not isinstance(result, dict)
                or frozenset(result) not in {_TASK7_KEYS, _TASK7_COMPACT_KEYS}
                or result.get("feature_index") != feature
            ):
                raise FeatureAtlasSelectionError("atlas Task 7 result has invalid exact schema or link")
            if (result["database"], result["concept_id"]) not in train_pairs:
                raise FeatureAtlasSelectionError("atlas Task 7 result is not linked to Task 6")
            if any(
                not isinstance(result.get(name), dict)
                or frozenset(result[name]) != _TASK7_VIEW_KEYS
                for name in ("all_source", "source_excluded")
            ):
                raise FeatureAtlasSelectionError("atlas Task 7 nested statistical schema is invalid")
            if frozenset(result) == _TASK7_KEYS:
                try:
                    identity = _validate_result_row(result, width=config.width)
                except (HeldoutEvaluationError, TypeError, ValueError) as error:
                    raise FeatureAtlasSelectionError(
                        f"atlas Task 7 result failed public validation: {error}"
                    ) from error
            else:
                identity = (
                    result.get("feature_index"), result.get("database"),
                    result.get("concept_id"),
                )
                if (
                    type(identity[0]) is not int or not 0 <= identity[0] < config.width
                    or identity[1] not in DATABASES
                    or type(identity[2]) is not str or not identity[2]
                    or type(result.get("concept_name")) is not str or not result["concept_name"]
                    or type(result.get("concept_type")) is not str or not result["concept_type"]
                    or result.get("training_direction") not in {"positive", "negative", "null"}
                    or result.get("evidence_tier") not in {"insufficient", "descriptive", "confirmatory"}
                    or type(result.get("confirmed")) is not bool
                    or result.get("status") not in {
                        "descriptive", "insufficient_validation_positives",
                        "not_validated", "validated",
                    }
                    or result.get("primary_view") != "source_excluded"
                    or result.get("schema_version") != SCHEMA_VERSION
                    or any(
                        type(result.get(name)) is not int or result[name] < 0
                        for name in (
                            "annotated_positive_count", "eligible_accession_count",
                            "matched_control_count", "matched_positive_count",
                        )
                    )
                ):
                    raise FeatureAtlasSelectionError("atlas compact Task 7 result is invalid")
            if identity[0] != feature:
                raise FeatureAtlasSelectionError("atlas Task 7 outer feature link changed")
    for field, keys, limit in (
        ("top_texts", _TOP_TEXT_KEYS, 50),
        ("top_proteins", _TOP_PROTEIN_KEYS, 20),
    ):
        split_map = row.get(field)
        if not isinstance(split_map, dict) or set(split_map) != {"train", "validation"}:
            raise FeatureAtlasSelectionError(f"atlas {field} split schema is invalid")
        for split, values in split_map.items():
            if not isinstance(values, list) or len(values) > limit:
                raise FeatureAtlasSelectionError(f"atlas {field} is not bounded")
            for value in values:
                if not isinstance(value, dict) or (
                    frozenset(value) != keys or value.get("split") != split
                ):
                    raise FeatureAtlasSelectionError(f"atlas {field} nested schema is invalid")
                _finite_float(value.get("activation"), f"atlas {field} activation")
                activation = float(value["activation"])
                if activation <= 0.0 or value.get("split") not in {"train", "validation"}:
                    raise FeatureAtlasSelectionError(f"atlas {field} activation/split domain is invalid")
                if type(value.get("accession")) is not str or not value["accession"]:
                    raise FeatureAtlasSelectionError(f"atlas {field} accession is invalid")
                if field == "top_texts" and (
                    type(value.get("unit_id")) is not str or not value["unit_id"]
                    or type(value.get("text")) is not str or not value["text"]
                    or type(value.get("row_index")) is not int or value["row_index"] < 0
                    or type(value.get("aspect_index")) is not int
                    or not 0 <= value["aspect_index"] < 24
                ):
                    raise FeatureAtlasSelectionError("atlas top text identity/domain is invalid")
            identities = [
                value.get("unit_id") if field == "top_texts" else value.get("accession")
                for value in values
            ]
            if len(identities) != len(set(identities)):
                raise FeatureAtlasSelectionError(f"atlas {field} identities are not unique")
            if field == "top_texts":
                expected_order = sorted(values, key=lambda value: (-value["activation"], value["row_index"], value["unit_id"]))
            else:
                expected_order = sorted(
                    values,
                    key=lambda value: (value["activation"], value["accession"]),
                    reverse=True,
                )
            if values != expected_order:
                raise FeatureAtlasSelectionError(f"atlas {field} ordering is not deterministic")
    contrasts = row.get("contrasts")
    if not isinstance(contrasts, list) or len(contrasts) > 20:
        raise FeatureAtlasSelectionError("atlas contrast list is invalid")
    for contrast in contrasts:
        if not isinstance(contrast, dict) or (
            frozenset(contrast) != _CONTRAST_KEYS
        ):
            raise FeatureAtlasSelectionError("atlas contrast has invalid exact schema")
        for name in ("covariate_distance", "high_activation", "low_activation"):
            _finite_float(contrast.get(name), f"contrast {name}")
        if (
            type(contrast.get("high_accession")) is not str or not contrast["high_accession"]
            or type(contrast.get("low_accession")) is not str or not contrast["low_accession"]
            or contrast["high_accession"] == contrast["low_accession"]
            or contrast.get("split") not in {"train", "validation"}
            or type(contrast.get("aspect_index")) is not int
            or not 0 <= contrast["aspect_index"] < 24
            or contrast["covariate_distance"] < 0.0
            or contrast["high_activation"] < contrast["low_activation"]
            or contrast["low_activation"] < 0.0
        ):
            raise FeatureAtlasSelectionError("atlas contrast identity/domain is invalid")
    if type(row.get("preferred_aspect")) is not int or not -1 <= row["preferred_aspect"] < 24:
        raise FeatureAtlasSelectionError("atlas preferred aspect is invalid")
    for name in ("train_concept_ids", "validation_concept_ids", "firing_accessions"):
        values = row.get(name)
        if not isinstance(values, list) or any(type(value) is not str or not value for value in values):
            raise FeatureAtlasSelectionError(f"atlas {name} is invalid")
        if name != "firing_accessions" and len(values) > 5:
            raise FeatureAtlasSelectionError(f"atlas {name} exceeds top-five")
        if name == "firing_accessions" and values != sorted(set(values)):
            raise FeatureAtlasSelectionError("atlas firing accessions are not canonical")
    split_diag, absorb = row.get("splitting_diagnostics"), row.get("absorption_diagnostics")
    if (
        not isinstance(split_diag, dict) or set(split_diag) != {"top_neighbors", "total_count"}
        or type(split_diag["total_count"]) is not int or split_diag["total_count"] < 0
        or not isinstance(split_diag["top_neighbors"], list) or len(split_diag["top_neighbors"]) > 20
        or not isinstance(absorb, dict) or set(absorb) != {"top_pairs", "total_count"}
        or type(absorb["total_count"]) is not int or absorb["total_count"] < 0
        or not isinstance(absorb["top_pairs"], list) or len(absorb["top_pairs"]) > 20
    ):
        raise FeatureAtlasSelectionError("atlas diagnostic summary schema is invalid")
    def validate_split_edge(item: object, label: str) -> int:
        if (
            not isinstance(item, dict) or frozenset(item) != _SPLIT_DIAGNOSTIC_KEYS
            or item.get("kind") != "candidate_diagnostic"
            or type(item.get("other_feature_index")) is not int
            or not isinstance(item.get("shared_concepts"), list)
        ):
            raise FeatureAtlasSelectionError(f"atlas {label} entry is invalid")
        cosine, jaccard, other = item.get("decoder_cosine"), item.get("firing_jaccard"), item.get("other_feature_index")
        if cosine is not None:
            cosine = _finite_float(cosine, "decoder cosine")
        jaccard = _finite_float(jaccard, "firing jaccard")
        if (
            type(other) is not int or not 0 <= other < config.width or other == feature
            or cosine is not None and not 0.0 <= cosine <= 1.0
            or not 0.0 <= jaccard <= 1.0
            or not ((cosine is not None and cosine >= 0.80) or jaccard >= 0.50)
            or any(
                not isinstance(identity, dict)
                or set(identity) != {"concept_id", "database"}
                or type(identity["concept_id"]) is not str or not identity["concept_id"]
                or identity["database"] not in DATABASES
                for identity in item["shared_concepts"]
            )
            or not item["shared_concepts"]
            or item["shared_concepts"] != sorted(
                item["shared_concepts"],
                key=lambda value: (value["database"], value["concept_id"]),
            )
            or len({
                (value["database"], value["concept_id"])
                for value in item["shared_concepts"]
            }) != len(item["shared_concepts"])
        ):
            raise FeatureAtlasSelectionError(f"atlas {label} domain is invalid")
        if (expected_decoder is None) != (cosine is None):
            raise FeatureAtlasSelectionError(f"atlas {label} decoder evidence is unavailable")
        return other

    for item in split_diag["top_neighbors"]:
        validate_split_edge(item, "splitting diagnostic")
    for item in absorb["top_pairs"]:
        if (
            not isinstance(item, dict) or frozenset(item) != _ABSORPTION_DIAGNOSTIC_KEYS
            or item.get("kind") != "candidate_diagnostic"
            or item.get("ancestor_related") is not False
            or not isinstance(item.get("concept_identities"), list)
            or len(item["concept_identities"]) != 2
            or any(
                not isinstance(identity, dict)
                or set(identity) != {"concept_id", "concept_name", "database"}
                for identity in item["concept_identities"]
            )
        ):
            raise FeatureAtlasSelectionError("atlas absorption diagnostic entry is invalid")
        jaccard = _finite_float(item.get("concept_jaccard"), "absorption concept jaccard")
        identities = item["concept_identities"]
        if (
            not 0.0 <= jaccard < 0.10
            or identities != sorted(identities, key=lambda value: (value["database"], value["concept_id"]))
            or len({(value["database"], value["concept_id"]) for value in identities}) != 2
            or any(
                type(value.get("concept_name")) is not str or not value["concept_name"]
                or type(value.get("concept_id")) is not str or not value["concept_id"]
                or value.get("database") not in DATABASES
                for value in identities
            )
        ):
            raise FeatureAtlasSelectionError("atlas absorption diagnostic domain is invalid")
    supported_concepts = len(_validated_identities(row))
    max_absorption_pairs = supported_concepts * (supported_concepts - 1) // 2
    if (
        split_diag["total_count"] < len(split_diag["top_neighbors"])
        or split_diag["total_count"] > config.width - 1
        or absorb["total_count"] < len(absorb["top_pairs"])
        or absorb["total_count"] > max_absorption_pairs
    ):
        raise FeatureAtlasSelectionError("atlas diagnostic total count is inconsistent")
    cluster = row.get("splitting_cluster")
    cluster_suffix = cluster[6:] if type(cluster) is str and cluster.startswith("split-") else ""
    if cluster is not None and (
        type(cluster) is not str
        or not cluster.startswith("split-")
        or not cluster_suffix.isdigit()
        or len(cluster_suffix) != max(4, len(str(int(cluster_suffix or "0"))))
        or int(cluster_suffix or "0") >= config.width
    ):
        raise FeatureAtlasSelectionError("atlas splitting cluster is invalid")
    if (split_diag["total_count"] == 0) != (cluster is None):
        raise FeatureAtlasSelectionError("atlas splitting cluster disagrees with diagnostics")
    parent = row.get("splitting_parent")
    cluster_root = int(cluster_suffix or "0") if cluster is not None else None
    if cluster is None or feature == cluster_root:
        if parent is not None:
            raise FeatureAtlasSelectionError("atlas splitting root must not have a parent")
    else:
        validate_split_edge(parent, "splitting parent")
    category, reason = row.get("selection_category"), row.get("selection_reason")
    fingerprint = row.get("selection_fingerprint")
    if frozen:
        if fingerprint is None:
            if category is not None or reason is not None:
                raise FeatureAtlasSelectionError("atlas frozen selection fields are invalid")
        elif (
            not _is_sha256(fingerprint)
            or category not in SELECTION_CATEGORIES | {None}
            or type(reason) is not str or not reason
        ):
            raise FeatureAtlasSelectionError("atlas frozen selection fields are invalid")
    elif category is not None or reason is not None or fingerprint is not None:
        raise FeatureAtlasSelectionError("unfrozen atlas contains selection fields")


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureAtlasSelectionError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise FeatureAtlasSelectionError(f"{label} must be finite")
    return result


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise FeatureAtlasSelectionError(f"{label} must be a finite built-in float")
    return value


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _SHA_CHARS


def _metric_value(metrics: Mapping[str, object], *names: str) -> object:
    for name in names:
        if name in metrics:
            return metrics[name]
    raise ValueError(f"missing ranking metric: {names[0]}")


def _concept_ids(metrics: Mapping[str, object], prefix: str) -> list[str]:
    names = (
        f"{prefix}_concept_ids",
        f"top_{prefix}_concept_ids",
        f"{prefix}_ranked_concept_ids",
    )
    raw: object = []
    for name in names:
        if name in metrics:
            raw = metrics[name]
            break
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError(f"{prefix} concept IDs must be a sequence")
    result: list[str] = []
    for value in raw:
        if type(value) is not str or not value:
            raise TypeError(f"{prefix} concept IDs must be nonempty strings")
        if value not in result:
            result.append(value)
        if len(result) == 5:
            break
    return result


def biological_ranking_score(metrics: Mapping[str, float]) -> float:
    """Return the literal approved six-term biological ranking score."""

    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    auroc = _finite(_metric_value(metrics, "auroc"), "auroc")
    lift = _finite(_metric_value(metrics, "auprc_lift"), "auprc_lift")
    effect = _finite(
        _metric_value(metrics, "rank_biserial_effect"), "rank_biserial_effect"
    )
    excluded = _finite(
        _metric_value(metrics, "source_excluded_effect", "excluded_effect"),
        "source_excluded_effect",
    )
    all_source = _finite(
        _metric_value(metrics, "all_source_effect"), "all_source_effect"
    )
    fdr = _finite(
        _metric_value(metrics, "fdr", "permutation_q_value", "heldout_fdr"),
        "fdr",
    )
    if lift <= 0.0:
        lift_term = 0.0
    else:
        lift_term = _clip(math.log2(lift) / 3.0)
    if fdr < 0.0 or fdr > 1.0:
        raise ValueError("fdr must be between zero and one")
    fdr_term = 1.0 if fdr == 0.0 else _clip(-math.log10(fdr) / 10.0)
    train = set(_concept_ids(metrics, "train"))
    validation = set(_concept_ids(metrics, "validation"))
    consistency = (
        len(train & validation) / len(train | validation)
        if train and validation
        else 0.0
    )
    return float(
        0.30 * _clip((auroc - 0.50) / 0.50)
        + 0.20 * lift_term
        + 0.15 * abs(effect)
        + 0.15 * _clip(excluded / max(all_source, 1e-8))
        + 0.10 * consistency
        + 0.10 * fdr_term
    )


def _source_results(record: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = record.get("heldout_concept_results", [])
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise FeatureAtlasSelectionError("source-excluded concept results must be a list")
    if any(not isinstance(item, Mapping) for item in raw):
        raise FeatureAtlasSelectionError("source-excluded concept result is invalid")
    return list(raw)  # type: ignore[return-value]


def _is_validated(result: Mapping[str, object]) -> bool:
    confirmed = result.get("confirmed") is True or result.get("validated") is True
    status = result.get("status")
    if status is not None:
        confirmed = confirmed and status == "validated"
    training = result.get("training_direction", result.get("train_direction", "positive"))
    heldout = result.get("direction")
    if heldout is None and isinstance(result.get("source_excluded"), Mapping):
        heldout = result["source_excluded"].get("direction")  # type: ignore[index]
    if heldout is None:
        heldout = "positive"
    return confirmed and training == heldout and heldout == "positive"


def _field_values(record: Mapping[str, object]) -> tuple[float, float, float, float]:
    purity = _finite(record.get("source_purity", record.get("purity", 0.0)), "source purity")
    lift = _finite(
        record.get("source_lift", record.get("preferred_aspect_lift", 0.0)),
        "source lift",
    )
    loss = _finite(record.get("source_effect_loss", 0.0), "source effect loss")
    reproducibility = _finite(
        record.get("split_reproducibility", 0.0), "split reproducibility"
    )
    return purity, lift, loss, reproducibility


def classify_feature(
    record: Mapping[str, object], config: FeatureAtlasConfig
) -> str:
    """Classify one feature while excluding same-source evidence from biology."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if not isinstance(config, FeatureAtlasConfig):
        raise TypeError("config must be FeatureAtlasConfig")
    if record.get("dead") is True or record.get("too_sparse") is True:
        return "dead_or_too_sparse"
    validated = any(_is_validated(item) for item in _source_results(record))
    purity, lift, loss, reproducibility = _field_values(record)
    field_preference = purity >= 0.80 and lift >= 2.0 and reproducibility >= 0.50
    pure_field = field_preference and loss >= 0.80 and not validated
    if validated and field_preference:
        return "mixed"
    if validated:
        return "validated_biological_concept"
    if pure_field:
        return "field_detector"
    train_results = record.get("train_concept_results", [])
    if _source_results(record) or (isinstance(train_results, Sequence) and len(train_results)):
        return "exploratory"
    return "unresolved"


def _provenance_payload(
    provenance: Mapping[str, object] | AtlasSourceBinding | None,
    *,
    allow_test_fixture_unbound: bool,
) -> dict[str, object]:
    if isinstance(provenance, AtlasSourceBinding):
        result: dict[str, object] = asdict(provenance)
    elif isinstance(provenance, Mapping):
        result = dict(provenance)
    elif provenance is None:
        result = {}
    else:
        raise TypeError("provenance must be a mapping or AtlasSourceBinding")
    if not result:
        if allow_test_fixture_unbound:
            return {"test_fixture_unbound": True}
        raise FeatureAtlasSelectionError("complete production source binding is required")
    if any(not _is_sha256(value) for value in result.values()):
        raise FeatureAtlasSelectionError("provenance values must be SHA-256 strings")
    if allow_test_fixture_unbound:
        return result
    expected_keys = set(AtlasSourceBinding.__dataclass_fields__)
    if set(result) != expected_keys:
        raise FeatureAtlasSelectionError("complete production source binding is required")
    required = {
        "inference_sha256": EXPECTED_INFERENCE_SHA256,
        "feature_stats_sha256": EXPECTED_FEATURE_STATS_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
        "units_sha256": EXPECTED_UNITS_SHA256,
        "activation_manifest_sha256": EXPECTED_ACTIVATION_MANIFEST_SHA256,
    }
    if any(result[name] != value for name, value in required.items()):
        raise FeatureAtlasSelectionError("production provenance does not match approved hashes")
    return result


def _canonical_sha(payload: object) -> str:
    data = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _bounded_top(
    raw: object, limit: int, *, text: bool
) -> list[dict[str, object]]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        raise FeatureAtlasSelectionError("top evidence candidates must be iterable")
    heap: list[tuple[float, int, str, dict[str, object]]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise FeatureAtlasSelectionError("top evidence candidate must be a mapping")
        activation = _finite(item.get("activation"), "top evidence activation")
        copied = dict(item)
        if text:
            row_index = copied.get("row_index")
            if type(row_index) is not int or row_index < 0:
                raise FeatureAtlasSelectionError("top text row_index is invalid")
            tie = -row_index
            identity = str(row_index)
        else:
            accession = copied.get("accession")
            if type(accession) is not str or not accession:
                raise FeatureAtlasSelectionError("top protein accession is invalid")
            tie = 0
            identity = accession
        if identity in seen:
            raise FeatureAtlasSelectionError(f"duplicate top evidence identity: {identity}")
        seen.add(identity)
        key = (activation, tie, identity)
        entry = (activation, tie, identity, copied)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif key > heap[0][:3]:
            heapq.heapreplace(heap, entry)
    return [item for *_, item in sorted(heap, key=lambda value: value[:3], reverse=True)]


def _validated_identities(record: Mapping[str, object]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for metric in _source_results(record):
        database, concept = metric.get("database"), metric.get("concept_id")
        if _is_validated(metric) and type(database) is str and type(concept) is str:
            result.add((database, concept))
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


class _UnionFind:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b:
            return False
        low, high = sorted((a, b))
        self.parent[high] = low
        return True


def _splitting_diagnostics(
    records: list[dict[str, object]], decoder: torch.Tensor | None
) -> tuple[
    dict[int, dict[str, object]],
    dict[int, str | None],
    dict[int, dict[str, object] | None],
]:
    width = len(records)
    identities = [_validated_identities(record) for record in records]
    firing = [set(record.get("firing_accessions", [])) for record in records]
    concept_features: dict[tuple[str, str], list[int]] = defaultdict(list)
    for feature, values in enumerate(identities):
        for identity in values:
            concept_features[identity].append(feature)
    totals = [0] * width
    heaps: list[list[tuple[float, int, dict[str, object]]]] = [[] for _ in range(width)]
    union = _UnionFind(range(width))
    forest_edges: list[tuple[int, int, dict[str, object]]] = []

    def retain(feature: int, other: int, item: dict[str, object]) -> None:
        totals[feature] += 1
        score = max(
            float(item["firing_jaccard"]),
            float(item["decoder_cosine"] if item["decoder_cosine"] is not None else -1.0),
        )
        entry = (score, -other, {**item, "other_feature_index": other})
        heap = heaps[feature]
        if len(heap) < 20:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)

    def emit(left: int, right: int, cosine: float | None, jac: float) -> None:
        shared = [
            {"concept_id": concept, "database": database}
            for database, concept in sorted(identities[left] & identities[right])
        ]
        item = {
            "decoder_cosine": cosine,
            "firing_jaccard": jac,
            "kind": "candidate_diagnostic",
            "shared_concepts": shared,
        }
        retain(left, right, item)
        retain(right, left, item)
        if union.union(left, right):
            forest_edges.append((left, right, item))

    if decoder is not None:
        if (
            not isinstance(decoder, torch.Tensor)
            or decoder.dtype != torch.float32
            or decoder.ndim != 2
            or decoder.shape[1] != width
            or not bool(torch.isfinite(decoder).all().item())
        ):
            raise FeatureAtlasSelectionError("decoder must be finite float32 [input,width]")
        norms = decoder.norm(dim=0)
        if bool(torch.any(norms <= 0).item()):
            raise FeatureAtlasSelectionError("decoder columns must have positive norm")
        normalized = decoder / norms.unsqueeze(0)
        block = min(256, width)
        for left_start in range(0, width, block):
            left_end = min(width, left_start + block)
            left_block = normalized[:, left_start:left_end].T
            for right_start in range(left_start, width, block):
                right_end = min(width, right_start + block)
                similarity = left_block @ normalized[:, right_start:right_end]
                hits = (similarity >= 0.80).nonzero(as_tuple=False)
                for hit in hits.tolist():
                    left, right = left_start + hit[0], right_start + hit[1]
                    if left >= right or not identities[left] & identities[right]:
                        continue
                    jac = _jaccard(firing[left], firing[right])
                    emit(
                        left, right,
                        min(1.0, max(0.0, float(similarity[hit[0], hit[1]].item()))), jac,
                    )
                del similarity
    # A pair shared by several concepts is handled only for its lexicographically
    # first shared identity.  This avoids retaining an O(width**2) pair set.
    for identity in sorted(concept_features):
        features = concept_features[identity]
        for left_position, left in enumerate(features):
            for right in features[left_position + 1 :]:
                shared_identities = identities[left] & identities[right]
                if identity != min(shared_identities):
                    continue
                jac = _jaccard(firing[left], firing[right])
                if jac < 0.50:
                    continue
                cosine = None
                if decoder is not None:
                    cosine = float(torch.dot(
                        decoder[:, left] / decoder[:, left].norm(),
                        decoder[:, right] / decoder[:, right].norm(),
                    ).item())
                    if cosine >= 0.80:
                        continue  # already emitted by the block scan
                    cosine = min(1.0, max(0.0, cosine))
                emit(left, right, cosine, jac)
    members: dict[int, list[int]] = defaultdict(list)
    for feature in range(width):
        members[union.find(feature)].append(feature)
    clusters = {
        feature: (f"split-{min(group):04d}" if len(group) > 1 else None)
        for group in members.values()
        for feature in group
    }
    adjacency: dict[int, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    for left, right, item in forest_edges:
        adjacency[left].append((right, item))
        adjacency[right].append((left, item))
    parents: dict[int, dict[str, object] | None] = {}
    for group in members.values():
        root = min(group)
        parents[root] = None
        pending = [root]
        seen = {root}
        while pending:
            current = pending.pop()
            for other, item in sorted(adjacency[current], key=lambda value: value[0], reverse=True):
                if other in seen:
                    continue
                seen.add(other)
                parents[other] = {**item, "other_feature_index": current}
                pending.append(other)
    diagnostics = {
        feature: {
            "top_neighbors": [
                item for *_, item in sorted(
                    heaps[feature], key=lambda entry: entry[:2], reverse=True
                )
            ],
            "total_count": totals[feature],
        }
        for feature in range(width)
    }
    return diagnostics, clusters, parents


def _ancestor_closure(
    concept: tuple[str, str],
    parents: Mapping[object, Sequence[object]],
) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    raw = parents.get(concept, parents.get(concept[1], ()))
    pending = [
        value if isinstance(value, tuple) else (concept[0], str(value))
        for value in raw
    ]
    while pending:
        parent = pending.pop()
        if parent in seen:
            continue
        seen.add(parent)
        raw_parents = parents.get(parent, parents.get(parent[1], ()))
        pending.extend(
            value if isinstance(value, tuple) else (parent[0], str(value))
            for value in raw_parents
        )
    return seen


def _absorption_diagnostics(
    record: Mapping[str, object], concept_parents: Mapping[object, Sequence[object]],
    concept_memberships: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> dict[str, object]:
    concepts = [item for item in _source_results(record) if _is_validated(item)]
    heap: list[tuple[float, str, str, dict[str, object]]] = []
    total = 0
    for left_index, left in enumerate(concepts):
        for right in concepts[left_index + 1 :]:
            left_id, right_id = left.get("concept_id"), right.get("concept_id")
            left_db, right_db = left.get("database"), right.get("database")
            if any(type(value) is not str for value in (left_id, right_id, left_db, right_db)):
                continue
            left_identity = (left_db, left_id)
            right_identity = (right_db, right_id)
            if (
                left_identity in _ancestor_closure(right_identity, concept_parents)
                or right_identity in _ancestor_closure(left_identity, concept_parents)
            ):
                continue
            memberships = concept_memberships or {}
            left_members = set(memberships.get(left_identity, left.get("protein_accessions", [])))
            right_members = set(memberships.get(right_identity, right.get("protein_accessions", [])))
            similarity = _jaccard(left_members, right_members)
            if similarity < 0.10:
                total += 1
                first, second = sorted((left_identity, right_identity))
                names = {
                    left_identity: left.get("concept_name", left_id),
                    right_identity: right.get("concept_name", right_id),
                }
                item = {
                    "concept_identities": [
                        {
                            "concept_id": concept,
                            "concept_name": names[(database, concept)],
                            "database": database,
                        }
                        for database, concept in (first, second)
                    ],
                    "concept_jaccard": similarity,
                    "ancestor_related": False,
                    "kind": "candidate_diagnostic",
                }
                entry = (-similarity, f"{first[0]}:{first[1]}", f"{second[0]}:{second[1]}", item)
                if len(heap) < 20:
                    heapq.heappush(heap, entry)
                elif entry[:3] > heap[0][:3]:
                    heapq.heapreplace(heap, entry)
    return {
        "top_pairs": [item for *_, item in sorted(heap, reverse=True)],
        "total_count": total,
    }


def _build_feature_atlas_fixture(
    feature_records: Iterable[Mapping[str, object]],
    config: FeatureAtlasConfig,
    output_path: Path,
    *,
    decoder: torch.Tensor | None = None,
    concept_parents: Mapping[str, Sequence[str]] | None = None,
    concept_memberships: Mapping[tuple[str, str], Sequence[str]] | None = None,
    provenance: Mapping[str, object] | AtlasSourceBinding | None = None,
    inference_path: Path | None = None,
    feature_stats_path: Path | None = None,
    upstream_artifacts: Mapping[str, Path] | None = None,
    overwrite: bool = False,
    allow_test_fixture_unbound: bool = False,
) -> dict[str, object]:
    """Build exactly ``config.width`` sorted records with bounded evidence.

    The iterable is the validated, already joined Task 1/3/4/5/6/7 public
    record stream.  Production callers must supply the complete source binding
    and exact inference/feature-stat artifacts; fixture callers must opt in.
    """

    if not isinstance(config, FeatureAtlasConfig):
        raise TypeError("config must be FeatureAtlasConfig")
    output_path = Path(output_path)
    if type(overwrite) is not bool or type(allow_test_fixture_unbound) is not bool:
        raise TypeError("overwrite flags must be built-in bool values")
    source = _provenance_payload(
        provenance, allow_test_fixture_unbound=allow_test_fixture_unbound
    )
    if allow_test_fixture_unbound:
        source = {**source, "test_fixture_unbound": True}
    if not allow_test_fixture_unbound:
        if not isinstance(inference_path, Path) or not isinstance(feature_stats_path, Path):
            raise FeatureAtlasSelectionError("production inference and feature_stats paths are required")
        if sha256_file(inference_path) != source["inference_sha256"]:
            raise FeatureAtlasSelectionError("inference checksum does not match source binding")
        if sha256_file(feature_stats_path) != source["feature_stats_sha256"]:
            raise FeatureAtlasSelectionError("feature_stats checksum does not match source binding")
    upstream: dict[str, object] = {}
    for label, path in sorted((upstream_artifacts or {}).items()):
        if type(label) is not str or not isinstance(path, Path):
            raise TypeError("upstream_artifacts must map strings to Paths")
        upstream[label] = file_metadata(path)
    records = [dict(record) for record in feature_records]
    for record in records:
        record.setdefault("contrasts", [])
        record.setdefault("firing_accessions", [])
        record.setdefault("splitting_cluster", None)
        if set(record) != _ATLAS_INPUT_KEYS:
            raise FeatureAtlasSelectionError("feature input has invalid exact schema")
    if output_path.exists() and not overwrite:
        existing_rows = list(strict_json_lines(output_path))
        if any(row.get("selection_fingerprint") is not None for row in existing_rows):
            selection_path = output_path.parent / "selection_manifest.json"
            if not selection_path.is_file():
                raise FeatureAtlasSelectionError(
                    "joint-frozen atlas is missing sibling selection manifest"
                )
            stage = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.joint-resume-", dir=output_path.parent))
            expected_path = stage / output_path.name
            try:
                _build_feature_atlas_fixture(
                    records, config, expected_path, decoder=decoder,
                    concept_parents=concept_parents, concept_memberships=concept_memberships,
                    provenance=provenance,
                    inference_path=inference_path, feature_stats_path=feature_stats_path,
                    upstream_artifacts=upstream_artifacts,
                    allow_test_fixture_unbound=allow_test_fixture_unbound,
                )
                expected_rows = list(strict_json_lines(expected_path))
                unfrozen_existing = json.loads(json.dumps(existing_rows, sort_keys=True))
                for row in unfrozen_existing:
                    row["selection_category"] = None
                    row["selection_fingerprint"] = None
                    row["selection_reason"] = None
                if unfrozen_existing != expected_rows:
                    raise FeatureAtlasSelectionError(
                        "joint-frozen atlas evidence differs from canonical rebuild"
                    )
                validate_selection_manifest(
                    selection_path, output_path, inference_path, config,
                    provenance=provenance,
                    allow_test_fixture_unbound=allow_test_fixture_unbound,
                )
                return {
                    "bytes": output_path.stat().st_size,
                    "config_fingerprint": config.fingerprint(),
                    "path": str(output_path.resolve()),
                    "record_count": len(existing_rows),
                    "sha256": sha256_file(output_path),
                }
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.recompute-", dir=output_path.parent))
        expected_path = stage / output_path.name
        try:
            expected = _build_feature_atlas_fixture(
                records, config, expected_path, decoder=decoder,
                concept_parents=concept_parents, provenance=provenance,
                concept_memberships=concept_memberships,
                inference_path=inference_path, feature_stats_path=feature_stats_path,
                upstream_artifacts=upstream_artifacts,
                allow_test_fixture_unbound=allow_test_fixture_unbound,
            )
            if output_path.read_bytes() != expected_path.read_bytes():
                raise FeatureAtlasSelectionError(
                    "immutable atlas semantic content differs from canonical recomputation"
                )
            return {
                **expected,
                "path": str(output_path.resolve()),
                "sha256": sha256_file(output_path),
                "bytes": output_path.stat().st_size,
            }
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    if len(records) != config.width:
        raise FeatureAtlasSelectionError(
            f"atlas must contain exactly {config.width} feature records"
        )
    try:
        records.sort(key=lambda item: item["feature_index"])
    except (KeyError, TypeError) as error:
        raise FeatureAtlasSelectionError("feature record index is invalid") from error
    if [record.get("feature_index") for record in records] != list(range(config.width)):
        raise FeatureAtlasSelectionError("feature records must uniquely cover every feature index")
    splitting, clusters, splitting_parents = _splitting_diagnostics(records, decoder)
    parents = concept_parents or {}
    output: list[dict[str, object]] = []
    for record in records:
        feature = int(record["feature_index"])
        texts = record.pop("top_text_candidates", {"train": [], "validation": []})
        proteins = record.pop("top_protein_candidates", {"train": [], "validation": []})
        if not isinstance(texts, Mapping) or not isinstance(proteins, Mapping):
            raise FeatureAtlasSelectionError("top candidate split maps are invalid")
        record["schema_version"] = SCHEMA_VERSION
        record["source_provenance"] = source
        record["upstream_artifacts"] = upstream
        record["classification"] = classify_feature(record, config)
        record["top_texts"] = {
            split: _bounded_top(texts.get(split, []), 50, text=True)
            for split in ("train", "validation")
        }
        record["top_proteins"] = {
            split: _bounded_top(proteins.get(split, []), 20, text=False)
            for split in ("train", "validation")
        }
        contrasts = record.get("contrasts", [])
        if not isinstance(contrasts, list):
            raise FeatureAtlasSelectionError("contrasts must be a list")
        record["contrasts"] = contrasts[:20]
        record["splitting_diagnostics"] = splitting[feature]
        record["splitting_cluster"] = record.get("splitting_cluster") or clusters[feature]
        record["splitting_parent"] = splitting_parents[feature]
        record["absorption_diagnostics"] = _absorption_diagnostics(
            record, parents, concept_memberships
        )
        record["selection_category"] = None
        record["selection_fingerprint"] = None
        record["selection_reason"] = None
        _validate_atlas_row_schema(
            record, config, frozen=False,
            allow_test_fixture_unbound=allow_test_fixture_unbound,
            expected_source_provenance=source,
            expected_decoder=decoder,
        )
        output.append(record)
    write_jsonl_atomic(output_path, output)
    return {
        "bytes": output_path.stat().st_size,
        "config_fingerprint": config.fingerprint(),
        "path": str(output_path.resolve()),
        "record_count": len(output),
        "sha256": sha256_file(output_path),
    }


def _load_torch_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise FeatureAtlasSelectionError(f"cannot load validated {label}: {path}") from error
    if not isinstance(payload, dict):
        raise FeatureAtlasSelectionError(f"validated {label} must be a mapping")
    return payload


def _stream_task4_top_texts(
    sparse_dir: Path,
    units: list[dict[str, object]],
    width: int,
) -> list[dict[str, list[dict[str, object]]]]:
    manifest = strict_json_load(sparse_dir / "manifest.json")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise FeatureAtlasSelectionError("validated Task 4 shard registration is invalid")
    heaps: list[dict[str, list[tuple[float, int, str, dict[str, object]]]]] = [
        {"train": [], "validation": []} for _ in range(width)
    ]
    for metadata in shards:
        if not isinstance(metadata, dict) or type(metadata.get("path")) is not str:
            raise FeatureAtlasSelectionError("validated Task 4 shard metadata is invalid")
        payload = _load_torch_mapping(sparse_dir / metadata["path"], "Task 4 shard")
        start = int(payload["start_row"])
        indptr = payload["indptr"]
        indices = payload["feature_indices"]
        values = payload["activation_values"]
        assert isinstance(indptr, torch.Tensor)
        assert isinstance(indices, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        for local in range(len(indptr) - 1):
            row_index = start + local
            unit = units[row_index]
            accession = unit.get("accession")
            unit_id = unit.get("unit_id")
            aspect = unit.get("aspect_index")
            text = unit.get("text")
            split = sae_training_data.split_for_accession(str(accession))
            left, right = int(indptr[local]), int(indptr[local + 1])
            for offset in range(left, right):
                feature = int(indices[offset])
                activation = float(values[offset])
                evidence = {
                    "accession": accession,
                    "activation": activation,
                    "aspect_index": aspect,
                    "row_index": row_index,
                    "split": split,
                    "text": text,
                    "unit_id": unit_id,
                }
                entry = (activation, -row_index, str(unit_id), evidence)
                heap = heaps[feature][split]
                if len(heap) < 50:
                    heapq.heappush(heap, entry)
                elif entry[:3] > heap[0][:3]:
                    heapq.heapreplace(heap, entry)
    return [
        {
            split: [item for *_, item in sorted(item[split], key=lambda x: x[:3], reverse=True)]
            for split in ("train", "validation")
        }
        for item in heaps
    ]


def _rank_train_ids(rows: Sequence[Mapping[str, object]]) -> list[str]:
    ordered = sorted(
        rows,
        key=lambda row: (
            min(float(row["source_excluded"]["fisher_q_value"]),
                float(row["source_excluded"]["rank_q_value"])),  # type: ignore[index]
            -float(row["source_excluded"]["rank_biserial_effect"]),  # type: ignore[index]
            str(row["database"]), str(row["concept_id"]),
        ),
    )
    result: list[str] = []
    for row in ordered:
        concept = str(row["concept_id"])
        if concept not in result:
            result.append(concept)
        if len(result) == 5:
            break
    return result


def _rank_validation_ids(rows: Sequence[Mapping[str, object]]) -> list[str]:
    supported = [
        row for row in rows
        if isinstance(row.get("source_excluded"), Mapping)
        and row["source_excluded"].get("direction") == row.get("training_direction")  # type: ignore[index]
        and row["source_excluded"].get("direction") == "positive"  # type: ignore[index]
        and row.get("status") != "insufficient_validation_positives"
    ]
    ordered = sorted(
        supported,
        key=lambda row: (
            float(row["source_excluded"].get("permutation_q_value") or 1.0),  # type: ignore[index]
            -float(row["source_excluded"].get("auroc") or 0.0),  # type: ignore[index]
            str(row["database"]), str(row["concept_id"]),
        ),
    )
    result: list[str] = []
    for row in ordered:
        concept = str(row["concept_id"])
        if concept not in result:
            result.append(concept)
        if len(result) == 5:
            break
    return result


def _deterministic_contrasts(
    feature: int,
    preferred_aspect: int,
    protein_rows: Sequence[object],
    covariates: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    ordered = sorted(
        protein_rows,
        key=lambda row: (
            -float(getattr(row, "all_sources").get(feature, 0.0)),
            str(getattr(row, "accession")),
        ),
    )
    highs = ordered[:20]
    used: set[str] = set()
    contrasts: list[dict[str, object]] = []
    for high in highs:
        high_accession = str(getattr(high, "accession"))
        high_activation = float(getattr(high, "all_sources").get(feature, 0.0))
        high_cov = covariates[high_accession]
        candidates = []
        for low in reversed(ordered):
            low_accession = str(getattr(low, "accession"))
            if low_accession == high_accession or low_accession in used:
                continue
            low_activation = float(getattr(low, "all_sources").get(feature, 0.0))
            low_cov = covariates[low_accession]
            if (
                getattr(low, "split") != getattr(high, "split")
                or bool(low_cov["aspect_present"][preferred_aspect])  # type: ignore[index]
                != bool(high_cov["aspect_present"][preferred_aspect])  # type: ignore[index]
                or low_activation > high_activation
            ):
                continue
            distance = (
                abs(int(low_cov["sequence_length"]) - int(high_cov["sequence_length"]))
                + abs(int(low_cov["total_text_unit_count"]) - int(high_cov["total_text_unit_count"]))
            )
            candidates.append((distance, low_accession, low_activation, low))
        if not candidates:
            continue
        distance, low_accession, low_activation, low = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        used.add(low_accession)
        contrasts.append({
            "aspect_index": preferred_aspect,
            "covariate_distance": float(distance),
            "high_accession": high_accession,
            "high_activation": high_activation,
            "low_accession": low_accession,
            "low_activation": low_activation,
            "split": getattr(high, "split"),
        })
    return contrasts


def _validated_path_records(
    atlas_paths: AtlasPaths,
    source_binding: AtlasSourceBinding,
    sparse_activation_dir: Path,
    protein_activation_dir: Path,
    covariates_path: Path,
    concept_index_path: Path,
    concept_snapshot_dir: Path,
    train_candidates_path: Path,
    heldout_results_path: Path,
    inference_path: Path,
    feature_stats_path: Path,
    config: FeatureAtlasConfig,
) -> tuple[
    list[dict[str, object]], torch.Tensor, dict[object, Sequence[object]],
    dict[tuple[str, str], Sequence[str]], dict[str, Path],
]:
    validate_source_binding(source_binding, atlas_paths)
    index = sae_training_data.build_data_index(
        atlas_paths.units_path, atlas_paths.text_manifest_path, atlas_paths.activation_dir
    )
    validate_concept_index(concept_snapshot_dir, atlas_paths.units_path, concept_index_path.parent)
    sparse_manifest = validate_sparse_feature_activations(
        index, inference_path, feature_stats_path, sparse_activation_dir,
        atlas_paths=atlas_paths, source_binding=source_binding,
    )
    protein_manifest = validate_protein_activations(
        atlas_paths.units_path, sparse_activation_dir,
        atlas_paths.run_dir / "split_manifest.json", atlas_paths.filtered_uniprot_tsv,
        protein_activation_dir, concept_index_path=concept_index_path,
        atlas_paths=atlas_paths, source_binding=source_binding,
        concept_snapshot_dir=concept_snapshot_dir,
    )
    validate_protein_covariates(
        atlas_paths.filtered_uniprot_tsv, atlas_paths.units_path, concept_index_path,
        covariates_path, concept_snapshot_dir=concept_snapshot_dir,
    )
    discovery_manifest = validate_train_discovery(
        train_candidates_path.parent, config_fingerprint=config.fingerprint()
    )
    candidate_rows = list(strict_json_lines(train_candidates_path))
    frozen = {
        (int(row["feature_index"]), str(row["database"]), str(row["concept_id"])): {
            "concept_name": str(row["concept_name"]),
            "concept_type": str(row["concept_type"]),
            "training_direction": str(row["source_excluded"]["direction"]),  # type: ignore[index]
        }
        for row in candidate_rows
    }
    heldout_manifest = validate_heldout_evaluation(
        heldout_results_path.parent, config_fingerprint=config.fingerprint(),
        config=config, frozen_candidates=frozen,
    )
    if (
        sparse_manifest.get("width") != config.width
        or protein_manifest.get("width") != config.width
        or discovery_manifest.get("width") != config.width
        or heldout_manifest.get("candidate_count") != len(candidate_rows)
        or protein_manifest.get("source_provenance", {}).get("sparse_manifest_sha256")
        != sha256_file(sparse_activation_dir / "manifest.json")  # type: ignore[union-attr]
    ):
        raise FeatureAtlasSelectionError("validated upstream manifest bindings disagree")
    model = load_frozen_sae(inference_path, "cpu")
    decoder = model.decoder_weight.detach().to(device="cpu", dtype=torch.float32)
    if decoder.shape != (1024, config.width):
        raise FeatureAtlasSelectionError("production decoder must have exact shape [1024,width]")
    stats = _load_torch_mapping(feature_stats_path, "feature_stats")
    units = list(strict_json_lines(atlas_paths.units_path))
    top_texts = _stream_task4_top_texts(sparse_activation_dir, units, config.width)
    protein_rows = list(iter_protein_activation_rows(protein_activation_dir))
    covariate_rows = list(strict_json_lines(covariates_path))
    covariates = {str(row["accession"]): row for row in covariate_rows}
    concepts = list(strict_json_lines(concept_index_path))
    concept_map = {(str(row["database"]), str(row["concept_id"])): row for row in concepts}
    memberships = {
        identity: list(row["protein_accessions"])
        for identity, row in concept_map.items()
    }
    parents = {
        (str(row["database"]), str(row["concept_id"])): [
            (str(row["database"]), str(parent)) for parent in row["parent_ids"]
        ]
        for row in concepts
    }
    candidates_by_feature: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_feature[int(row["feature_index"])].append(row)
    heldout_by_feature: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in strict_json_lines(heldout_results_path):
        heldout_by_feature[int(row["feature_index"])].append(
            _compact_heldout_result(row)
        )
    protein_top: list[dict[str, list[dict[str, object]]]] = [
        {"train": [], "validation": []} for _ in range(config.width)
    ]
    firing: list[set[str]] = [set() for _ in range(config.width)]
    protein_heaps: list[dict[str, list[tuple[float, str, dict[str, object]]]]] = [
        {"train": [], "validation": []} for _ in range(config.width)
    ]
    for row in protein_rows:
        for feature, activation in row.all_sources.items():
            if activation > config.activation_threshold:
                firing[feature].add(row.accession)
            item = {"accession": row.accession, "activation": float(activation), "split": row.split}
            entry = (float(activation), row.accession, item)
            heap = protein_heaps[feature][row.split]
            if len(heap) < 20:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
    for feature in range(config.width):
        for split in ("train", "validation"):
            protein_top[feature][split] = [
                item for *_, item in sorted(protein_heaps[feature][split], reverse=True)
            ]
    train_stats = stats["train"]["features"]  # type: ignore[index]
    validation_stats = stats["validation"]["features"]  # type: ignore[index]
    records: list[dict[str, object]] = []
    for feature in range(config.width):
        train = train_stats[feature]
        validation = validation_stats[feature]
        heldout = heldout_by_feature[feature]
        all_effects, excluded_effects = [], []
        for row in heldout:
            if isinstance(row.get("all_source"), Mapping) and row["all_source"].get("rank_biserial_effect") is not None:  # type: ignore[index]
                all_effects.append(abs(float(row["all_source"]["rank_biserial_effect"])))  # type: ignore[index]
                excluded_effects.append(abs(float(row["source_excluded"]["rank_biserial_effect"])))  # type: ignore[index]
        loss = max(
            (1.0 - excluded / max(all_value, 1e-8) for all_value, excluded in zip(all_effects, excluded_effects)),
            default=0.0,
        )
        train_count, validation_count = int(train["activation_count"]), int(validation["activation_count"])
        reproducibility = min(train_count, validation_count) / max(train_count, validation_count, 1)
        preferred = int(train["preferred_aspect"])
        contrasts = _deterministic_contrasts(
            feature, max(preferred, 0), protein_rows, covariates
        )
        if len(contrasts) != 20:
            raise FeatureAtlasSelectionError(
                f"production feature {feature} does not have exactly 20 matched contrasts"
            )
        records.append({
            "train_concept_results": candidates_by_feature[feature],
            "contrasts": contrasts,
            "dead": train_count == 0 and validation_count == 0,
            "decoder_norm": float(decoder[:, feature].norm().item()),
            "feature_index": feature,
            "firing_accessions": sorted(firing[feature]),
            "preferred_aspect": preferred,
            "source_effect_loss": float(_clip(loss)),
            "heldout_concept_results": heldout,
            "source_lift": float(train["preferred_aspect_lift"]),
            "source_purity": float(train["purity"]),
            "split_reproducibility": float(reproducibility),
            "too_sparse": train_count < config.train_min_firing or validation_count == 0,
            "top_protein_candidates": protein_top[feature],
            "top_text_candidates": top_texts[feature],
            "train_activation_count": train_count,
            "train_concept_ids": _rank_train_ids(candidates_by_feature[feature]),
            "train_max_activation": float(train["max_activation"]),
            "train_mean_activation": float(train["mean_on_firing"]),
            "train_split_coverage": train_count / max(int(stats["train"]["count"]), 1),  # type: ignore[index]
            "validation_activation_count": validation_count,
            "validation_concept_ids": _rank_validation_ids(heldout),
            "validation_max_activation": float(validation["max_activation"]),
            "validation_mean_activation": float(validation["mean_on_firing"]),
            "validation_split_coverage": validation_count / max(int(stats["validation"]["count"]), 1),  # type: ignore[index]
        })
    upstream = {
        "concept_manifest": concept_index_path.parent / "manifest.json",
        "covariate_manifest": covariates_path.with_suffix(".manifest.json"),
        "discovery_manifest": train_candidates_path.parent / "discovery_manifest.json",
        "heldout_manifest": heldout_results_path.parent / "heldout_manifest.json",
        "protein_manifest": protein_activation_dir / "manifest.json",
        "sparse_manifest": sparse_activation_dir / "manifest.json",
    }
    return records, decoder, parents, memberships, upstream


def build_feature_atlas(
    atlas_paths: AtlasPaths | Iterable[Mapping[str, object]],
    source_binding: AtlasSourceBinding | FeatureAtlasConfig,
    sparse_activation_dir: Path,
    protein_activation_dir: Path | None = None,
    covariates_path: Path | None = None,
    concept_index_path: Path | None = None,
    concept_snapshot_dir: Path | None = None,
    train_candidates_path: Path | None = None,
    heldout_results_path: Path | None = None,
    output_path: Path | None = None,
    config: FeatureAtlasConfig | None = None,
    *,
    inference_path: Path | None = None,
    feature_stats_path: Path | None = None,
    overwrite: bool = False,
    allow_test_fixture_unbound: bool = False,
    decoder: torch.Tensor | None = None,
    concept_parents: Mapping[str, Sequence[str]] | None = None,
    concept_memberships: Mapping[tuple[str, str], Sequence[str]] | None = None,
    provenance: Mapping[str, object] | None = None,
    upstream_artifacts: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Build from fully validated paths; prejoined records are fixture-only."""

    if not isinstance(atlas_paths, AtlasPaths):
        if not allow_test_fixture_unbound:
            raise FeatureAtlasSelectionError("prejoined feature records are fixture-only")
        if not isinstance(source_binding, FeatureAtlasConfig):
            raise TypeError("fixture branch requires FeatureAtlasConfig as second argument")
        return _build_feature_atlas_fixture(
            atlas_paths, source_binding, Path(sparse_activation_dir), decoder=decoder,
            concept_parents=concept_parents, provenance=provenance,
            concept_memberships=concept_memberships,
            inference_path=inference_path, feature_stats_path=feature_stats_path,
            upstream_artifacts=upstream_artifacts, overwrite=overwrite,
            allow_test_fixture_unbound=True,
        )
    if allow_test_fixture_unbound:
        raise FeatureAtlasSelectionError("path-based production branch cannot be fixture-unbound")
    if not isinstance(source_binding, AtlasSourceBinding) or not isinstance(config, FeatureAtlasConfig):
        raise TypeError("production branch requires AtlasSourceBinding and FeatureAtlasConfig")
    required_paths = (
        protein_activation_dir, covariates_path, concept_index_path, concept_snapshot_dir,
        train_candidates_path, heldout_results_path, output_path, inference_path,
        feature_stats_path,
    )
    if any(not isinstance(path, Path) for path in required_paths):
        raise TypeError("all production artifact inputs must be explicit Paths")
    records, loaded_decoder, parents, memberships, upstream = _validated_path_records(
        atlas_paths, source_binding, Path(sparse_activation_dir),
        protein_activation_dir, covariates_path, concept_index_path,
        concept_snapshot_dir, train_candidates_path, heldout_results_path,
        inference_path, feature_stats_path, config,
    )
    return _build_feature_atlas_fixture(
        records, config, output_path, decoder=loaded_decoder,
        concept_parents=parents, concept_memberships=memberships,
        provenance=source_binding,
        inference_path=inference_path, feature_stats_path=feature_stats_path,
        upstream_artifacts=upstream, overwrite=overwrite,
    )


def _validate_atlas_output(
    output_path: Path,
    *,
    config: FeatureAtlasConfig,
    provenance: Mapping[str, object],
    upstream: Mapping[str, object],
) -> dict[str, object]:
    try:
        rows = list(strict_json_lines(output_path))
    except (AtlasIOError, OSError, ValueError) as error:
        raise FeatureAtlasSelectionError(f"invalid immutable atlas: {error}") from error
    if (
        len(rows) != config.width
        or [row.get("feature_index") for row in rows] != list(range(config.width))
    ):
        raise FeatureAtlasSelectionError("immutable atlas width or feature order changed")
    for row in rows:
        texts, proteins = row.get("top_texts"), row.get("top_proteins")
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("source_provenance") != dict(provenance)
            or row.get("upstream_artifacts") != dict(upstream)
            or row.get("classification") not in CLASSIFICATIONS
            or not isinstance(texts, dict) or set(texts) != {"train", "validation"}
            or any(not isinstance(values, list) or len(values) > 50 for values in texts.values())
            or not isinstance(proteins, dict) or set(proteins) != {"train", "validation"}
            or any(not isinstance(values, list) or len(values) > 20 for values in proteins.values())
            or not isinstance(row.get("contrasts"), list) or len(row["contrasts"]) > 20
            or row.get("selection_category") is not None
            or row.get("selection_reason") is not None
        ):
            raise FeatureAtlasSelectionError("immutable atlas schema or provenance changed")
    return {
        "bytes": output_path.stat().st_size,
        "config_fingerprint": config.fingerprint(),
        "path": str(output_path.resolve()),
        "record_count": len(rows),
        "sha256": sha256_file(output_path),
    }


def _ranking_metrics(record: Mapping[str, object], metric: Mapping[str, object]) -> dict[str, object]:
    source_view = metric.get("source_excluded")
    all_view = metric.get("all_source")
    source_view = source_view if isinstance(source_view, Mapping) else metric
    all_view = all_view if isinstance(all_view, Mapping) else metric
    return {
        "auroc": source_view.get("auroc"),
        "auprc_lift": source_view.get("auprc_lift"),
        "rank_biserial_effect": source_view.get("rank_biserial_effect"),
        "source_excluded_effect": metric.get(
            "source_excluded_effect", source_view.get("rank_biserial_effect")
        ),
        "all_source_effect": metric.get(
            "all_source_effect", all_view.get("rank_biserial_effect")
        ),
        "fdr": metric.get("fdr", source_view.get("permutation_q_value")),
        "train_concept_ids": record.get("train_concept_ids", []),
        "validation_concept_ids": record.get("validation_concept_ids", []),
    }


def _best_biological(record: Mapping[str, object]) -> tuple[float, Mapping[str, object]] | None:
    scored: list[tuple[float, str, str, Mapping[str, object]]] = []
    for metric in _source_results(record):
        if not _is_validated(metric):
            continue
        score = biological_ranking_score(_ranking_metrics(record, metric))
        scored.append((score, str(metric.get("database", "")), str(metric.get("concept_id", "")), metric))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return scored[0][0], scored[0][3]


def _field_score(record: Mapping[str, object]) -> float:
    purity, lift, loss, reproducibility = _field_values(record)
    cross_field_failure = 0.0 if any(
        _is_validated(metric) for metric in _source_results(record)
    ) else 1.0
    return float(
        0.25 * _clip(purity)
        + 0.20 * _clip(math.log2(max(lift, 1e-8)) / 3.0)
        + 0.20 * _clip(loss)
        + 0.20 * cross_field_failure
        + 0.15 * _clip(reproducibility)
    )


def _atlas_rows(
    atlas: Path | Sequence[Mapping[str, object]],
    width: int,
    *,
    allow_test_fixture_unbound: bool,
    expected_source_provenance: Mapping[str, object],
    expected_decoder: torch.Tensor | None,
    expected_selection_fingerprint: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if isinstance(atlas, Path):
        try:
            rows = list(strict_json_lines(atlas))
        except (AtlasIOError, OSError, ValueError) as error:
            raise FeatureAtlasSelectionError(f"invalid feature atlas: {error}") from error
        metadata: dict[str, object] = file_metadata(atlas)
    elif isinstance(atlas, Sequence) and not isinstance(atlas, (str, bytes)):
        rows = [dict(row) for row in atlas]
        metadata = {"path": None, "bytes": None, "sha256": _canonical_sha(rows)}
    else:
        raise TypeError("atlas must be a Path or a sequence of records")
    if len(rows) != width or [row.get("feature_index") for row in rows] != list(range(width)):
        raise FeatureAtlasSelectionError("atlas must be sorted and contain exactly width records")
    validation_config = FeatureAtlasConfig(**{
        **FeatureAtlasConfig().to_dict(),
        "width": width,
        "k": min(FeatureAtlasConfig().k, width),
    })
    if expected_decoder is not None and (
        not isinstance(expected_decoder, torch.Tensor)
        or expected_decoder.dtype != torch.float32
        or expected_decoder.ndim != 2
        or expected_decoder.shape[1] != width
        or (not allow_test_fixture_unbound and expected_decoder.shape[0] != 1024)
        or not bool(torch.isfinite(expected_decoder).all().item())
        or bool(torch.any(expected_decoder.norm(dim=0) <= 0).item())
    ):
        raise FeatureAtlasSelectionError("expected decoder is invalid")
    if expected_decoder is None and not allow_test_fixture_unbound:
        raise FeatureAtlasSelectionError("production atlas validation requires decoder")
    for row in rows:
        _validate_atlas_row_schema(
            row, validation_config, frozen=True,
            allow_test_fixture_unbound=allow_test_fixture_unbound,
            expected_source_provenance=expected_source_provenance,
            expected_decoder=expected_decoder,
        )
    fingerprints = {row.get("selection_fingerprint") for row in rows}
    if expected_selection_fingerprint is None:
        if fingerprints != {None}:
            raise FeatureAtlasSelectionError(
                "frozen atlas requires caller selection manifest linkage"
            )
    elif (
        not _is_sha256(expected_selection_fingerprint)
        or fingerprints != {expected_selection_fingerprint}
    ):
        raise FeatureAtlasSelectionError("atlas selection fingerprint differs from caller manifest")
    clusters: dict[str, list[int]] = defaultdict(list)
    identities = [_validated_identities(row) for row in rows]
    firing = [set(row["firing_accessions"]) for row in rows]
    normalized_decoder = None
    if expected_decoder is not None:
        normalized_decoder = expected_decoder / expected_decoder.norm(dim=0).unsqueeze(0)

    def reconcile_edge(feature: int, edge: Mapping[str, object]) -> None:
        other = int(edge["other_feature_index"])
        shared = [
            {"concept_id": concept, "database": database}
            for database, concept in sorted(identities[feature] & identities[other])
        ]
        if edge["shared_concepts"] != shared:
            raise FeatureAtlasSelectionError("splitting edge shared concepts do not reconcile")
        expected_jaccard = _jaccard(firing[feature], firing[other])
        if edge["firing_jaccard"] != expected_jaccard:
            raise FeatureAtlasSelectionError("splitting edge firing Jaccard does not reconcile")
        cosine = edge["decoder_cosine"]
        if normalized_decoder is None:
            if cosine is not None or expected_jaccard < 0.50:
                raise FeatureAtlasSelectionError("splitting edge lacks qualifying evidence")
            return
        expected_cosine = min(1.0, max(0.0, float(torch.dot(
            normalized_decoder[:, feature], normalized_decoder[:, other]
        ).item())))
        if type(cosine) is not float or not math.isclose(
            cosine, expected_cosine, rel_tol=0.0, abs_tol=1e-6
        ):
            raise FeatureAtlasSelectionError("splitting edge decoder cosine does not reconcile")
        if expected_cosine < 0.80 and expected_jaccard < 0.50:
            raise FeatureAtlasSelectionError("splitting edge is not actually qualifying")

    for row in rows:
        feature = int(row["feature_index"])
        cluster = row["splitting_cluster"]
        if cluster is not None:
            clusters[cluster].append(int(row["feature_index"]))
        for neighbor in row["splitting_diagnostics"]["top_neighbors"]:
            reconcile_edge(feature, neighbor)
            other = rows[int(neighbor["other_feature_index"])]
            if cluster is None or other["splitting_cluster"] != cluster:
                raise FeatureAtlasSelectionError("splitting diagnostic crosses cluster boundary")
        parent = row["splitting_parent"]
        if parent is not None:
            reconcile_edge(feature, parent)
            other = rows[int(parent["other_feature_index"])]
            if cluster is None or other["splitting_cluster"] != cluster:
                raise FeatureAtlasSelectionError("splitting parent crosses cluster boundary")
    for cluster, members in clusters.items():
        root = min(members)
        expected_label = f"split-{root:04d}"
        if len(members) < 2 or cluster != expected_label:
            raise FeatureAtlasSelectionError("splitting cluster membership is not canonical")
        member_set = set(members)
        children: dict[int, list[int]] = defaultdict(list)
        for feature in members:
            parent = rows[feature]["splitting_parent"]
            if feature == root:
                if parent is not None:
                    raise FeatureAtlasSelectionError("splitting root has a parent")
                continue
            if parent is None:
                raise FeatureAtlasSelectionError("splitting component is disconnected")
            parent_feature = int(parent["other_feature_index"])
            if parent_feature not in member_set:
                raise FeatureAtlasSelectionError("splitting parent leaves its component")
            children[parent_feature].append(feature)
        visited: set[int] = set()
        pending = [root]
        while pending:
            current = pending.pop()
            if current in visited:
                raise FeatureAtlasSelectionError("splitting parent witness has a cycle")
            visited.add(current)
            pending.extend(children[current])
        if visited != member_set:
            raise FeatureAtlasSelectionError("splitting component witness is disconnected")
    return rows, metadata


def _select_biological(
    rows: list[dict[str, object]], config: FeatureAtlasConfig
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    decisions: dict[int, dict[str, object]] = {}
    for row in rows:
        feature = int(row["feature_index"])
        classification = classify_feature(row, config)
        best = _best_biological(row)
        if classification not in {"validated_biological_concept", "mixed"} or best is None:
            decisions[feature] = {
                "biological_score": None,
                "classification": classification,
                "feature_index": feature,
                "primary_concept_id": None,
                "primary_database": None,
                "reason": "not_biologically_eligible",
                "selected_category": None,
            }
            continue
        score, metric = best
        item = {
            "feature_index": feature,
            "score": score,
            "concept_id": metric.get("concept_id"),
            "database": metric.get("database"),
            "preferred_aspect": row.get("preferred_aspect", -1),
            "splitting_cluster": row.get("splitting_cluster"),
        }
        candidates.append(item)
        decisions[feature] = {
            "biological_score": score,
            "classification": classification,
            "feature_index": feature,
            "primary_concept_id": item["concept_id"],
            "primary_database": item["database"],
            "reason": "eligible_not_selected",
            "selected_category": None,
        }
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["feature_index"])))
    selected: list[dict[str, object]] = []
    concepts: Counter[object] = Counter()
    clusters: Counter[object] = Counter()
    aspects: Counter[object] = Counter()

    def try_add(item: dict[str, object], reason: str) -> bool:
        if len(selected) >= config.biological_slots:
            return False
        concept = item["concept_id"]
        cluster = item["splitting_cluster"]
        aspect = item["preferred_aspect"]
        rejected = None
        if concepts[concept] >= 3:
            rejected = "concept_cap"
        elif cluster is not None and clusters[cluster] >= 2:
            rejected = "cluster_cap"
        elif aspects[aspect] >= 32:
            rejected = "preferred_aspect_cap"
        if rejected is not None:
            decisions[int(item["feature_index"])]["reason"] = rejected
            return False
        selected.append(item)
        concepts[concept] += 1
        if cluster is not None:
            clusters[cluster] += 1
        aspects[aspect] += 1
        decision = decisions[int(item["feature_index"])]
        decision["selected_category"] = "biological"
        decision["reason"] = reason
        return True

    by_database = {
        database: [item for item in candidates if item["database"] == database]
        for database in DATABASES
    }
    positions = {database: 0 for database in DATABASES}
    for _round in range(20):
        for database in DATABASES:
            values = by_database[database]
            while positions[database] < len(values):
                item = values[positions[database]]
                positions[database] += 1
                if try_add(item, "database_coverage"):
                    break
            if len(selected) >= config.biological_slots:
                break
        if len(selected) >= config.biological_slots:
            break
    selected_ids = {int(item["feature_index"]) for item in selected}
    for item in candidates:
        if int(item["feature_index"]) not in selected_ids:
            try_add(item, "global_rank_fill")
            selected_ids = {int(value["feature_index"]) for value in selected}
    for item in candidates:
        decision = decisions[int(item["feature_index"])]
        if decision["selected_category"] is None and decision["reason"] == "eligible_not_selected":
            decision["reason"] = "biological_slots_full"
    return selected, decisions


def _select_fields(
    rows: list[dict[str, object]], decisions: dict[int, dict[str, object]], config: FeatureAtlasConfig
) -> list[dict[str, object]]:
    values = []
    for row in rows:
        feature = int(row["feature_index"])
        if decisions[feature]["classification"] != "field_detector":
            continue
        values.append(
            {
                "feature_index": feature,
                "preferred_aspect": row.get("preferred_aspect", -1),
                "score": _field_score(row),
            }
        )
    values.sort(key=lambda item: (-float(item["score"]), int(item["feature_index"])))
    selected: list[dict[str, object]] = []
    used_aspects: set[object] = set()
    for item in values:
        if item["preferred_aspect"] not in used_aspects:
            selected.append(item)
            used_aspects.add(item["preferred_aspect"])
            decisions[int(item["feature_index"])]["reason"] = "field_family_coverage"
        if len(selected) == config.field_slots:
            break
    selected_ids = {int(item["feature_index"]) for item in selected}
    for item in values:
        if len(selected) == config.field_slots:
            break
        if int(item["feature_index"]) not in selected_ids:
            selected.append(item)
            decisions[int(item["feature_index"])]["reason"] = "field_score_fill"
    for item in selected:
        decisions[int(item["feature_index"])]["selected_category"] = "field_detector"
    for item in values:
        decision = decisions[int(item["feature_index"])]
        if decision["selected_category"] is None:
            decision["reason"] = "field_slots_full_or_less_diverse"
    return selected


_CONTROL_NUMERIC = (
    "train_activation_count",
    "validation_activation_count",
    "train_mean_activation",
    "validation_mean_activation",
    "train_max_activation",
    "validation_max_activation",
    "train_split_coverage",
    "validation_split_coverage",
    "decoder_norm",
)


def _select_controls(
    rows: list[dict[str, object]], biological: list[dict[str, object]],
    decisions: dict[int, dict[str, object]], config: FeatureAtlasConfig,
) -> list[dict[str, object]]:
    if not biological:
        return []
    row_by_feature = {int(row["feature_index"]): row for row in rows}
    pool = [
        row for row in rows
        if decisions[int(row["feature_index"])]["classification"]
        in {"unresolved", "exploratory"}
        and decisions[int(row["feature_index"])]["selected_category"] is None
    ]
    if not pool:
        return []
    all_rows = pool + [row_by_feature[int(item["feature_index"])] for item in biological]
    scales: dict[str, float] = {}
    for name in _CONTROL_NUMERIC:
        values = [_finite(row.get(name, 0.0), name) for row in all_rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scales[name] = max(math.sqrt(variance), 1.0)
    rng = random.Random(config.seed)
    shuffled = sorted(int(row["feature_index"]) for row in pool)
    rng.shuffle(shuffled)
    tie_rank = {feature: rank for rank, feature in enumerate(shuffled)}
    matches: list[tuple[float, int, int, int, Mapping[str, object]]] = []
    for row in pool:
        feature = int(row["feature_index"])
        candidates: list[tuple[float, int]] = []
        for target in biological:
            target_feature = int(target["feature_index"])
            target_row = row_by_feature[target_feature]
            squared = sum(
                (
                    (_finite(row.get(name, 0.0), name)
                     - _finite(target_row.get(name, 0.0), name)) / scales[name]
                ) ** 2
                for name in _CONTROL_NUMERIC
            )
            if row.get("preferred_aspect") != target_row.get("preferred_aspect"):
                squared += 1.0
            candidates.append((math.sqrt(squared), target_feature))
        distance, target_feature = min(candidates, key=lambda item: (item[0], item[1]))
        decisions[feature]["control_target_feature_index"] = target_feature
        decisions[feature]["control_distance"] = float(distance)
        decisions[feature]["control_tie_rank"] = tie_rank[feature]
        matches.append((distance, tie_rank[feature], feature, target_feature, row))
    selected: list[dict[str, object]] = []
    for distance, _, feature, target_feature, row in sorted(matches):
        if len(selected) >= config.control_slots:
            decisions[feature]["reason"] = "control_rejected_by_no_reuse_slot_constraint"
            continue
        selected.append({
            "feature_index": feature,
            "matched_feature_index": target_feature,
            "matching_distance": float(distance),
            "preferred_aspect": row.get("preferred_aspect", -1),
        })
        decisions[feature]["selected_category"] = "control"
        decisions[feature]["reason"] = "seed42_nearest_matched_control"
    return selected


def _selected_entry(item: Mapping[str, object], category: str, reason: str) -> dict[str, object]:
    result = dict(item)
    result["category"] = category
    result["reason"] = reason
    return result


def _inference_metadata(
    inference_path: Path | torch.Tensor | None,
    source: Mapping[str, object],
    *,
    allow_test_fixture_unbound: bool,
) -> dict[str, object] | None:
    if isinstance(inference_path, Path):
        metadata = file_metadata(inference_path)
        if not allow_test_fixture_unbound and metadata["sha256"] != source["inference_sha256"]:
            raise FeatureAtlasSelectionError("inference checksum does not match source binding")
        return metadata
    if isinstance(inference_path, torch.Tensor):
        if not allow_test_fixture_unbound:
            raise FeatureAtlasSelectionError("production selection requires registered inference path")
        return {
            "path": None, "bytes": None,
            "sha256": hashlib.sha256(inference_path.detach().cpu().numpy().tobytes()).hexdigest(),
        }
    if inference_path is None and allow_test_fixture_unbound:
        return None
    raise FeatureAtlasSelectionError("production selection requires registered inference path")


def _selection_decoder(
    inference_path: Path | torch.Tensor | None,
    config: FeatureAtlasConfig,
    *,
    allow_test_fixture_unbound: bool,
) -> torch.Tensor | None:
    if isinstance(inference_path, Path):
        decoder = load_frozen_sae(inference_path, "cpu").decoder_weight.detach().to(
            device="cpu", dtype=torch.float32
        )
        if decoder.shape != (1024, config.width):
            raise FeatureAtlasSelectionError("selection decoder must have shape [1024,width]")
        return decoder
    if isinstance(inference_path, torch.Tensor):
        if not allow_test_fixture_unbound:
            raise FeatureAtlasSelectionError("production selection requires registered inference path")
        return inference_path.detach().to(device="cpu")
    if inference_path is None and allow_test_fixture_unbound:
        return None
    raise FeatureAtlasSelectionError("production selection requires registered inference decoder")


def _canonical_selection(
    rows: list[dict[str, object]],
    inference_metadata: dict[str, object] | None,
    config: FeatureAtlasConfig,
    source: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    evidence_rows = json.loads(json.dumps(rows, allow_nan=False, sort_keys=True))
    for row in evidence_rows:
        row["selection_category"] = None
        row["selection_fingerprint"] = None
        row["selection_reason"] = None
    biological, decisions = _select_biological(evidence_rows, config)
    fields = _select_fields(evidence_rows, decisions, config)
    controls = _select_controls(evidence_rows, biological, decisions, config)
    row_by_feature = {int(row["feature_index"]): row for row in evidence_rows}
    for feature in range(config.width):
        decision = decisions[feature]
        row = row_by_feature[feature]
        decision["field_score"] = _field_score(row)
        decision.setdefault("control_target_feature_index", None)
        decision.setdefault("control_distance", None)
        decision.setdefault("control_tie_rank", None)
        category = decision["selected_category"]
        decision["selected_reason"] = decision["reason"] if category is not None else None
        decision["rejected_reason"] = decision["reason"] if category is None else None
        decision["constraints"] = [
            {
                "name": "biological_eligibility",
                "passed": decision["classification"] in {
                    "validated_biological_concept", "mixed"
                },
            },
            {"name": "concept_cap", "passed": decision["reason"] != "concept_cap"},
            {"name": "cluster_cap", "passed": decision["reason"] != "cluster_cap"},
            {
                "name": "preferred_aspect_cap",
                "passed": decision["reason"] != "preferred_aspect_cap",
            },
            {
                "name": "field_eligibility",
                "passed": decision["classification"] == "field_detector",
            },
            {
                "name": "control_eligibility",
                "passed": decision["classification"] in {"unresolved", "exploratory"},
            },
        ]
    selected = [
        *(_selected_entry(item, "biological", decisions[int(item["feature_index"])]["reason"])
          for item in biological),
        *(_selected_entry(item, "field_detector", decisions[int(item["feature_index"])]["reason"])
          for item in fields),
        *(_selected_entry(item, "control", decisions[int(item["feature_index"])]["reason"])
          for item in controls),
    ]
    candidates = [decisions[index] for index in range(config.width)]
    final_rows = json.loads(json.dumps(evidence_rows, allow_nan=False, sort_keys=True))
    for row in final_rows:
        decision = decisions[int(row["feature_index"])]
        row["selection_category"] = decision["selected_category"]
        row["selection_reason"] = decision["reason"]
    freeze_id = _canonical_sha({
        "candidates": candidates,
        "evidence_atlas": evidence_rows,
        "selected": selected,
    })
    for row in final_rows:
        row["selection_fingerprint"] = freeze_id
    constraints = {
        "biological_slots": config.biological_slots,
        "cluster_cap": 2,
        "concept_cap": 3,
        "control_slots": config.control_slots,
        "database_coverage_cap": 20,
        "field_slots": config.field_slots,
        "preferred_aspect_cap": 32,
    }
    manifest: dict[str, object] = {
        "atlas": None,
        "candidates": candidates,
        "complete": True,
        "config_fingerprint": config.fingerprint(),
        "constraints": constraints,
        "counts": {
            "biological": len(biological), "control": len(controls),
            "field_detector": len(fields), "selected": len(selected),
        },
        "frozen": True,
        "inference": inference_metadata,
        "provenance": dict(source),
        "schema_version": SCHEMA_VERSION,
        "seed": config.seed,
        "selection_fingerprint": freeze_id,
        "selected": selected,
        "width": config.width,
    }
    return manifest, final_rows


def select_deep_features(
    atlas_path: Path | Sequence[Mapping[str, object]],
    inference_path: Path | torch.Tensor | None,
    config: FeatureAtlasConfig,
    output_path: Path | None = None,
    *,
    provenance: Mapping[str, object] | AtlasSourceBinding | None = None,
    overwrite: bool = False,
    allow_test_fixture_unbound: bool = False,
) -> dict[str, object]:
    """Freeze biological, field-detector, and matched-control selections."""

    if not isinstance(config, FeatureAtlasConfig):
        raise TypeError("config must be FeatureAtlasConfig")
    if type(overwrite) is not bool or type(allow_test_fixture_unbound) is not bool:
        raise TypeError("overwrite flags must be built-in bool values")
    source = _provenance_payload(
        provenance, allow_test_fixture_unbound=allow_test_fixture_unbound
    )
    row_source = (
        {**source, "test_fixture_unbound": True}
        if allow_test_fixture_unbound else source
    )
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and overwrite:
            raise FeatureAtlasSelectionError("complete frozen selection is immutable")
        if output_path.exists():
            return validate_selection_manifest(
                output_path, atlas_path, inference_path, config,
                provenance=provenance,
                allow_test_fixture_unbound=allow_test_fixture_unbound,
            )
    inference_metadata = _inference_metadata(
        inference_path, source,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
    )
    expected_decoder = _selection_decoder(
        inference_path, config,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
    )
    if not isinstance(atlas_path, Path) and allow_test_fixture_unbound:
        stage = Path(tempfile.mkdtemp(prefix=".fixture-selection-atlas-"))
        staged_atlas = stage / "feature_atlas.jsonl"
        try:
            _build_feature_atlas_fixture(
                atlas_path, config, staged_atlas, provenance=provenance,
                decoder=expected_decoder,
                allow_test_fixture_unbound=True,
            )
            rows, _ = _atlas_rows(
                staged_atlas, config.width,
                allow_test_fixture_unbound=True,
                expected_source_provenance=row_source,
                expected_decoder=expected_decoder,
            )
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    else:
        rows, _ = _atlas_rows(
            atlas_path, config.width,
            allow_test_fixture_unbound=allow_test_fixture_unbound,
            expected_source_provenance=row_source,
            expected_decoder=expected_decoder,
        )
    manifest, final_rows = _canonical_selection(rows, inference_metadata, config, source)
    if isinstance(atlas_path, Path) and output_path is not None:
        stage = Path(tempfile.mkdtemp(prefix=".selection-freeze-stage-", dir=output_path.parent))
        backup = stage / "original-atlas.backup"
        staged_atlas, staged_manifest = stage / atlas_path.name, stage / output_path.name
        preserve_recovery = False
        try:
            write_jsonl_atomic(staged_atlas, final_rows)
            manifest["atlas"] = {
                "bytes": staged_atlas.stat().st_size,
                "path": str(atlas_path.resolve()),
                "sha256": sha256_file(staged_atlas),
            }
            write_json_atomic(staged_manifest, manifest)
            shutil.copy2(atlas_path, backup)
            os.replace(staged_atlas, atlas_path)  # data before manifest
            try:
                os.replace(staged_manifest, output_path)
            except BaseException as publish_error:
                try:
                    os.replace(backup, atlas_path)
                except BaseException as restore_error:
                    preserve_recovery = True
                    raise FeatureAtlasSelectionError(
                        "joint freeze rollback requires recovery; "
                        f"backup: {backup}; stage: {stage}; "
                        f"publish error: {publish_error}; restore error: {restore_error}"
                    ) from restore_error
                raise
        except BaseException as error:
            if backup.exists() and not atlas_path.exists():
                os.replace(backup, atlas_path)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(error, FeatureAtlasSelectionError):
                raise
            raise FeatureAtlasSelectionError(f"joint atlas/selection freeze failed: {error}") from error
        finally:
            if not preserve_recovery:
                shutil.rmtree(stage, ignore_errors=True)
    elif output_path is not None:
        manifest["atlas"] = {"bytes": None, "path": None, "sha256": _canonical_sha(final_rows)}
        write_json_atomic(output_path, manifest)
    else:
        manifest["atlas"] = {"bytes": None, "path": None, "sha256": _canonical_sha(final_rows)}
    return manifest


def validate_selection_manifest(
    manifest_path: Path,
    atlas_path: Path | Sequence[Mapping[str, object]],
    inference_path: Path | torch.Tensor | None,
    config: FeatureAtlasConfig,
    *,
    provenance: Mapping[str, object] | AtlasSourceBinding | None = None,
    allow_test_fixture_unbound: bool = False,
) -> dict[str, object]:
    """Validate immutable contents, hashes, reasons, counts, and hard caps."""

    expected_source = _provenance_payload(
        provenance, allow_test_fixture_unbound=allow_test_fixture_unbound
    )
    try:
        manifest = strict_json_load(Path(manifest_path))
    except (AtlasIOError, OSError, ValueError) as error:
        raise FeatureAtlasSelectionError(f"invalid selection manifest: {error}") from error
    row_source = (
        {**expected_source, "test_fixture_unbound": True}
        if allow_test_fixture_unbound else expected_source
    )
    inference_metadata = _inference_metadata(
        inference_path, expected_source,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
    )
    expected_decoder = _selection_decoder(
        inference_path, config,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
    )
    rows, atlas_metadata = _atlas_rows(
        atlas_path, config.width,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
        expected_source_provenance=row_source,
        expected_decoder=expected_decoder,
        expected_selection_fingerprint=(
            manifest.get("selection_fingerprint") if isinstance(manifest, dict) else None
        ),
    )
    required = {
        "atlas", "candidates", "complete", "config_fingerprint", "constraints",
        "counts", "frozen", "inference", "provenance", "schema_version", "seed",
        "selected", "selection_fingerprint", "width",
    }
    if set(manifest) != required:
        raise FeatureAtlasSelectionError("selection manifest has invalid exact schema")
    expected, expected_rows = _canonical_selection(
        rows, inference_metadata, config, expected_source
    )
    expected["atlas"] = atlas_metadata
    if rows != expected_rows:
        raise FeatureAtlasSelectionError(
            "frozen atlas selection fields disagree with canonical decisions"
        )
    if manifest != expected:
        raise FeatureAtlasSelectionError(
            "selection manifest differs from canonical decision or constraint recomputation"
        )
    return manifest
