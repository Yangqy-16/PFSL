"""Deterministic, evidence-bounded reports for canonical Task 7 pairs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from esmc_sae.attribution import _verify_trusted_audit
from esmc_sae.calibrate import _publish_temp_no_replace, _write_json_create_only
from esmc_sae.config import ProjectConfig, load_config
from esmc_sae.heldout import _load_thresholds
from esmc_sae.intervention import PairInterventionRecord
from esmc_sae.manifest import file_sha256
from esmc_sae.metadata import (
    FEATURE_API_ROOT,
    FeatureMetadata,
    get_feature_metadata,
    load_feature_snapshot,
)
import esmc_sae.metadata as metadata_module


_EXPLANATION_STATES = frozenset({"reliable", "partial", "abstain"})
_PREDICTION_TYPES = frozenset({"TP", "FP", "FN", "TN"})
_APPLICABILITY_STATES = frozenset(
    {"consistent", "contradicted", "unresolved"}
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _as_mapping(result: PairInterventionRecord | Mapping[str, object]):
    if is_dataclass(result):
        return asdict(result)
    if not isinstance(result, Mapping):
        raise TypeError("pair result must be a dataclass or mapping")
    return dict(result)


def shortlisted_feature_ids(
    records: Sequence[PairInterventionRecord | Mapping[str, object]],
) -> tuple[int, ...]:
    """Return the exact sorted reliable/partial candidate-feature union."""

    selected: set[int] = set()
    for result in records:
        record = _as_mapping(result)
        if record.get("explanation_status") not in {"reliable", "partial"}:
            continue
        singles = record.get("singles")
        if not isinstance(singles, Sequence):
            raise ValueError("pair result singles must be a sequence")
        for single in singles:
            if not isinstance(single, Mapping):
                raise ValueError("pair result single is malformed")
            feature_id = single.get("feature_id")
            if type(feature_id) is not int or not 0 <= feature_id < 16384:
                raise ValueError("pair result contains invalid feature_id")
            selected.add(feature_id)
    return tuple(sorted(selected))


def _finite_float(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if not (-float("inf") < numeric < float("inf")):
        raise ValueError(f"{label} must be finite")
    return numeric


def positive_support_feature_ids(
    result: PairInterventionRecord | Mapping[str, object],
    *,
    single_reliable_abs_delta: float,
) -> tuple[int, ...]:
    """Select direct interventions that positively and specifically support SL."""

    threshold = _finite_float(
        single_reliable_abs_delta,
        "single_reliable_abs_delta",
    )
    if threshold < 0:
        raise ValueError("single_reliable_abs_delta must be non-negative")
    record = _as_mapping(result)
    singles = record.get("singles", ())
    controls = record.get("controls", ())
    fallback = record.get("control_abs_deltas", ())
    for value, label in (
        (singles, "pair result singles"),
        (controls, "pair result controls"),
        (fallback, "pair result control_abs_deltas"),
    ):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{label} must be a sequence")

    controls_by_side: dict[str, list[float]] = {"A": [], "B": []}
    for control in controls:
        if not isinstance(control, Mapping):
            raise ValueError("pair result control is malformed")
        side = control.get("side")
        if side not in {"A", "B"}:
            raise ValueError("pair result control side is malformed")
        controls_by_side[side].append(
            abs(_finite_float(control.get("delta_logit"), "control delta_logit"))
        )
    fallback_abs = [
        abs(_finite_float(value, "control_abs_delta")) for value in fallback
    ]

    selected: set[int] = set()
    for single in singles:
        if not isinstance(single, Mapping):
            raise ValueError("pair result single is malformed")
        feature_id = single.get("feature_id")
        side = single.get("side")
        if type(feature_id) is not int or not 0 <= feature_id < 16384:
            raise ValueError("pair result contains invalid feature_id")
        if side not in {"A", "B"}:
            raise ValueError("pair result single side is malformed")
        delta = _finite_float(single.get("delta_logit"), "single delta_logit")
        control_max = max(controls_by_side[side] or fallback_abs, default=0.0)
        if delta > max(threshold, control_max):
            selected.add(feature_id)
    return tuple(sorted(selected))


def _number(value: object, label: str) -> str:
    return repr(_finite_float(value, label))


def _applicability_for(
    applicability: Mapping[str, Mapping[str, object]] | None,
    *,
    side: str,
    feature_id: int,
) -> tuple[str, str, str | None]:
    key = f"{side}:{feature_id}"
    default = (
        "unresolved",
        "no adequate domain/location source available",
        None,
    )
    if applicability is None or key not in applicability:
        return default
    value = applicability[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"applicability entry {key} must be a mapping")
    status = value.get("status")
    evidence = value.get("evidence")
    source_url = value.get("source_url")
    if status not in _APPLICABILITY_STATES:
        raise ValueError(f"applicability entry {key} has an invalid status")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError(f"applicability entry {key} lacks evidence")
    if status != "unresolved" and (
        not isinstance(source_url, str)
        or not source_url.startswith(("https://", "http://"))
    ):
        raise ValueError(f"applicability entry {key} lacks a traceable source")
    if source_url is not None and (
        not isinstance(source_url, str)
        or not source_url.startswith(("https://", "http://"))
    ):
        raise ValueError(f"applicability entry {key} has an invalid source")
    return status, evidence.strip(), source_url


def _boundary(prediction_type: str, explanation_status: str) -> str:
    if prediction_type == "FP":
        return "model error rationale; never a real SL mechanism"
    if prediction_type == "FN":
        return "missed SL; no rationale is recovered from the original negative prediction"
    if explanation_status == "abstain":
        return "diagnostics only"
    if prediction_type == "TP" and explanation_status == "reliable":
        return "candidate model-internal rationale"
    return "diagnostic candidate evidence only"


def render_pair_report(
    result: PairInterventionRecord | Mapping[str, object],
    metadata: Mapping[int, FeatureMetadata],
    applicability: Mapping[str, Mapping[str, object]] | None = None,
    *,
    single_reliable_abs_delta: float = 0.0,
) -> str:
    """Render frozen numeric evidence without adding biological prose."""

    record = _as_mapping(result)
    row_id = record.get("row_id")
    gene_a = record.get("gene_a")
    gene_b = record.get("gene_b")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("pair result row_id must be non-empty")
    if type(gene_a) is not int or type(gene_b) is not int:
        raise ValueError("pair result gene IDs must be integers")
    prediction_type = record.get("prediction_type", "unavailable")
    explanation_status = record.get("explanation_status", "unavailable")
    if prediction_type != "unavailable" and prediction_type not in _PREDICTION_TYPES:
        raise ValueError("pair result has invalid prediction_type")
    if (
        explanation_status != "unavailable"
        and explanation_status not in _EXPLANATION_STATES
    ):
        raise ValueError("pair result has invalid explanation_status")

    lines = [
        f"# Pair report: {row_id}",
        "",
        "## Frozen identity and outcome",
        "",
        f"- row_id: {row_id}",
        f"- gene_a: {gene_a}",
        f"- gene_b: {gene_b}",
        f"- prediction_type: {prediction_type}",
        f"- explanation_status: {explanation_status}",
        f"- boundary: {_boundary(str(prediction_type), str(explanation_status))}",
        f"- original_logit: {_number(record['original_logit'], 'original_logit')}",
        f"- probability: {_number(record['probability'], 'probability')}",
    ]

    singles = record.get("singles", ())
    interactions = record.get("interactions", ())
    controls = record.get("controls", ())
    if not isinstance(singles, Sequence) or not isinstance(
        interactions, Sequence
    ):
        raise ValueError("pair result candidate fields must be sequences")
    candidate_enabled = explanation_status in {
        "reliable",
        "partial",
    } or (explanation_status == "unavailable" and bool(singles))
    mechanism_features = (
        positive_support_feature_ids(
            record,
            single_reliable_abs_delta=single_reliable_abs_delta,
        )
        if candidate_enabled
        else ()
    )
    mechanism_feature_set = set(mechanism_features)

    mechanism_contradicted = False
    if candidate_enabled:
        lines.extend(["", "## Shortlisted direct interventions", ""])
        for single in singles:
            if not isinstance(single, Mapping):
                raise ValueError("pair result single is malformed")
            feature_id = single.get("feature_id")
            side = single.get("side")
            positions = single.get("active_positions", ())
            if (
                type(feature_id) is not int
                or side not in {"A", "B"}
                or not isinstance(positions, Sequence)
            ):
                raise ValueError("pair result single identity is malformed")
            official = metadata.get(feature_id)
            if official is None or official.feature_id != feature_id:
                raise ValueError(
                    f"official metadata is missing for feature {feature_id}"
                )
            status, evidence, evidence_url = _applicability_for(
                applicability, side=side, feature_id=feature_id
            )
            positive_support = feature_id in mechanism_feature_set
            mechanism_contradicted = mechanism_contradicted or (
                positive_support and status == "contradicted"
            )
            side_controls = [
                abs(float(item["delta_logit"]))
                for item in controls
                if isinstance(item, Mapping) and item.get("side") == side
            ]
            if not side_controls:
                fallback = record.get("control_abs_deltas", ())
                if not isinstance(fallback, Sequence):
                    raise ValueError("pair result controls are malformed")
                side_controls = [abs(float(value)) for value in fallback]
            control_max = max(side_controls, default=0.0)
            rendered_positions = ", ".join(str(int(value)) for value in positions)
            lines.extend(
                [
                    f"### side {side}, feature {feature_id}",
                    "",
                    f"- residue positions: {rendered_positions or 'none'}",
                    f"- official description: {official.description}",
                    f"- metadata source: {official.source_url}",
                    f"- metadata retrieved_at: {official.retrieved_at}",
                    f"- delta_logit: {_number(single['delta_logit'], 'delta_logit')}",
                    (
                        "- matched-control max abs(delta_logit): "
                        f"{_number(control_max, 'control comparison')}"
                    ),
                    (
                        "- positive mechanism support: "
                        f"{str(positive_support).lower()}"
                    ),
                    f"- applicability: {status}",
                    f"- applicability evidence: {evidence}",
                    f"- applicability source: {evidence_url or 'none'}",
                    "",
                ]
            )

        lines.extend(["## Cross-protein intervention terms", ""])
        for interaction in interactions:
            if not isinstance(interaction, Mapping):
                raise ValueError("pair result interaction is malformed")
            feature_a = interaction.get("feature_a")
            feature_b = interaction.get("feature_b")
            lines.extend(
                [
                    f"- feature pair: A:{feature_a} + B:{feature_b}",
                    f"  - delta_a: {_number(interaction['delta_a'], 'delta_a')}",
                    f"  - delta_b: {_number(interaction['delta_b'], 'delta_b')}",
                    f"  - delta_ab: {_number(interaction['delta_ab'], 'delta_ab')}",
                    (
                        "  - interaction J: "
                        f"{_number(interaction['interaction_j'], 'interaction_j')}"
                    ),
                ]
            )
        control_interactions = record.get("control_interactions", ())
        control_js = [
            abs(float(item["interaction_j"]))
            for item in control_interactions
            if isinstance(item, Mapping)
        ]
        lines.extend(
            [
                "",
                (
                    "- matched-control max abs(J): "
                    f"{_number(max(control_js, default=0.0), 'control J')}"
                ),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Diagnostics",
                "",
                "- diagnostics only; candidate features are intentionally omitted",
                (
                    "- max_candidate_abs_delta: "
                    f"{_number(record.get('max_candidate_abs_delta', 0.0), 'max candidate')}"
                ),
                (
                    "- max_interaction_abs_j: "
                    f"{_number(record.get('max_interaction_abs_j', 0.0), 'max interaction')}"
                ),
            ]
        )

    eligible = (
        prediction_type == "TP"
        and explanation_status == "reliable"
        and bool(mechanism_features)
        and not mechanism_contradicted
    )
    rendered_mechanism_features = ", ".join(
        str(feature_id) for feature_id in mechanism_features
    )
    lines.extend(
        [
            "",
            "## Eligibility gate",
            "",
            (
                "- frozen single reliable abs(delta_logit) threshold: "
                f"{_number(single_reliable_abs_delta, 'single reliable threshold')}"
            ),
            (
                "- mechanism_feature_ids: "
                f"{rendered_mechanism_features or 'none'}"
            ),
            f"- mechanism_eligible: {str(eligible).lower()}",
            "- interpretation boundary: model-internal intervention evidence only",
            "",
        ]
    )
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"canonical Task 7 JSONL is invalid at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError("canonical Task 7 JSONL record is not an object")
            records.append(record)
    if not records:
        raise ValueError("canonical Task 7 JSONL is empty")
    return records


def _verify_sha256_manifest(
    manifest: Path,
    *,
    code_root: Path,
    required_targets: Sequence[Path],
    label: str,
) -> set[Path]:
    """Verify every manifest entry and require exact named consumers."""

    seen: set[Path] = set()
    with Path(manifest).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("  ", 1)
            if (
                len(fields) != 2
                or _SHA256_PATTERN.fullmatch(fields[0]) is None
            ):
                raise ValueError(f"{label} manifest invalid at line {line_number}")
            target = Path(code_root) / fields[1]
            relative = target.resolve(strict=False).relative_to(
                Path(code_root).resolve(strict=False)
            )
            if relative in seen:
                raise ValueError(f"{label} manifest contains duplicate targets")
            seen.add(relative)
            if not target.is_file() or file_sha256(target) != fields[0]:
                raise ValueError(
                    f"{label} manifest checksum failed for {fields[1]}"
                )
    for required in required_targets:
        relative_required = Path(required).resolve(strict=False).relative_to(
            Path(code_root).resolve(strict=False)
        )
        if relative_required not in seen:
            raise ValueError(
                f"{label} manifest does not bind {relative_required}"
            )
    return seen


def _verify_task07_manifest(project: ProjectConfig) -> tuple[Path, str]:
    manifest = project.paths.project_root / "manifests" / "task07.sha256"
    canonical = (
        project.paths.project_root
        / "experiment2"
        / "heldout"
        / "unblinded_results.jsonl"
    )
    _verify_sha256_manifest(
        manifest,
        code_root=project.paths.code_root,
        required_targets=(canonical,),
        label="Task 7",
    )
    return canonical, file_sha256(canonical)


def _validate_canonical_bindings(
    project: ProjectConfig,
    config_path: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    thresholds = _load_thresholds(project, config_path)
    _verify_trusted_audit(project)
    blind_sha = file_sha256(
        project.paths.project_root / "manifests" / "heldout_pairs_blind.csv"
    )
    thresholds_sha = file_sha256(
        project.paths.project_root
        / "configs"
        / (
            f"frozen_thresholds_{project.experiment.cell_line}_"
            f"{project.experiment.scene}_fold{project.experiment.fold}.json"
        )
    )
    seen_rows: set[str] = set()
    for record in records:
        row_id = record.get("row_id")
        if not isinstance(row_id, str) or not row_id or row_id in seen_rows:
            raise ValueError("canonical Task 7 row identity is invalid")
        seen_rows.add(row_id)
        if record.get("schema_version") != 1:
            raise ValueError("canonical Task 7 record must use schema version 1")
        if record.get("prediction_type") not in _PREDICTION_TYPES:
            raise ValueError("canonical Task 7 prediction type is invalid")
        if record.get("explanation_status") not in _EXPLANATION_STATES:
            raise ValueError("canonical Task 7 explanation status is invalid")
        bindings = record.get("artifact_bindings")
        expected = {
            "blind_manifest_sha256": blind_sha,
            "frozen_thresholds_sha256": thresholds_sha,
            "checkpoint_sha256": thresholds.checkpoint_sha256,
            "sae_provenance_fingerprint": record.get("provenance", {}).get(
                "sae_fingerprint"
            )
            if isinstance(record.get("provenance"), Mapping)
            else None,
        }
        if bindings != expected:
            raise ValueError(f"canonical Task 7 binding mismatch: {row_id}")
        checkpoint = record.get("checkpoint")
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("sha256") != thresholds.checkpoint_sha256
            or not isinstance(checkpoint.get("path"), str)
            or file_sha256(Path(checkpoint["path"]))
            != thresholds.checkpoint_sha256
        ):
            raise ValueError(f"canonical Task 7 checkpoint mismatch: {row_id}")


def _atomic_text_create_only(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temp_no_replace(Path(temporary), path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_task8_inputs(
    config_path: Path,
) -> tuple[
    ProjectConfig,
    Path,
    str,
    list[dict[str, object]],
    tuple[int, ...],
]:
    config_path = Path(config_path)
    project = load_config(config_path)
    canonical_path, task7_artifact_sha = _verify_task07_manifest(project)
    records = _read_jsonl(canonical_path)
    _validate_canonical_bindings(project, config_path, records)
    shortlist = shortlisted_feature_ids(records)
    return project, canonical_path, task7_artifact_sha, records, shortlist


def inspect_task8_inputs(config_path: Path) -> dict[str, int]:
    """Perform the full local consumption gate without metadata requests."""

    _, _, _, records, shortlist = _load_task8_inputs(config_path)
    counts = {
        state: sum(
            record["explanation_status"] == state for record in records
        )
        for state in ("reliable", "partial", "abstain")
    }
    return {
        "records": len(records),
        **counts,
        "shortlist": len(shortlist),
    }


def build_pair_reports(
    config_path: Path,
    *,
    output_root: Path | None = None,
    cache_root: Path | None = None,
) -> dict[str, int]:
    """Verify Task 7, fetch only its shortlist, and publish reports once."""

    config_path = Path(config_path)
    project = load_config(config_path)
    output_root = (
        Path(output_root)
        if output_root is not None
        else project.paths.project_root / "experiment2" / "pair_reports"
    )
    cache_root = (
        Path(cache_root)
        if cache_root is not None
        else project.paths.project_root / "cache" / "feature_metadata"
    )
    shortlist_manifest = cache_root / "shortlist_manifest.json"
    if output_root.exists() or shortlist_manifest.exists():
        raise FileExistsError(
            "formal Task 8 artifacts already exist; use a new output root"
        )
    (
        validated_project,
        canonical_path,
        task7_artifact_sha,
        records,
        shortlist,
    ) = _load_task8_inputs(config_path)
    if validated_project != project:
        raise RuntimeError("configuration changed during Task 8 validation")
    thresholds = _load_thresholds(project, config_path)

    old_cache_root = metadata_module._CACHE_ROOT
    metadata_module._CACHE_ROOT = cache_root
    snapshot = None
    if metadata_module._LOCAL_SNAPSHOT_PATH.is_file():
        snapshot = load_feature_snapshot(
            metadata_module._LOCAL_SNAPSHOT_PATH,
            require_complete=True,
        )

    request_count = 0
    snapshot_hit_count = 0
    cache_hit_count = 0
    official: dict[int, FeatureMetadata] = {}
    try:
        for feature_id in shortlist:
            cache_path = cache_root / f"{feature_id}.json"
            if cache_path.exists():
                cache_hit_count += 1
            elif snapshot is not None:
                snapshot_hit_count += 1
            else:
                request_count += 1
            official[feature_id] = get_feature_metadata(
                feature_id,
                snapshot=snapshot,
            )
    finally:
        metadata_module._CACHE_ROOT = old_cache_root

    implementation = {
        "metadata.py": file_sha256(Path(metadata_module.__file__)),
        "pair_report.py": file_sha256(Path(__file__)),
        "test_metadata_pair_report.py": file_sha256(
            Path(__file__).parent / "tests" / "test_metadata_pair_report.py"
        ),
    }
    metadata_source = (
        {
            "kind": "local_official_snapshot",
            "source_path": str(snapshot.source_path),
            "sha256": snapshot.sha256,
            "retrieved_at": snapshot.retrieved_at,
            "feature_count": len(snapshot.features),
        }
        if snapshot is not None
        else {
            "kind": "official_api",
            "base_url": FEATURE_API_ROOT,
        }
    )
    shortlist_payload = {
        "schema_version": 1,
        "task07_artifact_sha256": task7_artifact_sha,
        "task07_manifest_sha256": file_sha256(
            project.paths.project_root / "manifests" / "task07.sha256"
        ),
        "config_sha256": file_sha256(config_path),
        "implementation_sha256": implementation,
        "metadata_source": metadata_source,
        "features": [
            {
                "feature_id": feature_id,
                "source_url": official[feature_id].source_url,
                "cache_sha256": file_sha256(cache_root / f"{feature_id}.json"),
            }
            for feature_id in shortlist
        ],
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".tmp", dir=output_root.parent
        )
    )
    index_records = []
    try:
        for record in records:
            status = record["explanation_status"]
            report_metadata = official if status in {"reliable", "partial"} else {}
            mechanism_features = (
                list(
                    positive_support_feature_ids(
                        record,
                        single_reliable_abs_delta=(
                            thresholds.single_reliable_abs_delta
                        ),
                    )
                )
                if status in {"reliable", "partial"}
                else []
            )
            mechanism_eligible = (
                record["prediction_type"] == "TP"
                and status == "reliable"
                and bool(mechanism_features)
            )
            report = render_pair_report(
                record,
                report_metadata,
                single_reliable_abs_delta=(
                    thresholds.single_reliable_abs_delta
                ),
            )
            report_path = staging / f"{record['row_id']}.md"
            report_path.write_text(report, encoding="utf-8")
            features = (
                sorted(
                    {
                        int(item["feature_id"])
                        for item in record["singles"]
                    }
                )
                if status in {"reliable", "partial"}
                else []
            )
            index_records.append(
                {
                    "row_id": record["row_id"],
                    "gene_a": record["gene_a"],
                    "gene_b": record["gene_b"],
                    "prediction_type": record["prediction_type"],
                    "explanation_status": status,
                    "feature_ids": features,
                    "mechanism_feature_ids": mechanism_features,
                    "mechanism_eligible": mechanism_eligible,
                    "source_urls": [
                        official[feature_id].source_url
                        for feature_id in features
                    ],
                    "cache_sha256": {
                        str(feature_id): file_sha256(
                            cache_root / f"{feature_id}.json"
                        )
                        for feature_id in features
                    },
                    "report_sha256": file_sha256(report_path),
                    "task07_artifact_sha256": task7_artifact_sha,
                }
            )
        index_payload = {
            "schema_version": 1,
            "task07_artifact_sha256": task7_artifact_sha,
            "task07_manifest_sha256": shortlist_payload[
                "task07_manifest_sha256"
            ],
            "config_sha256": shortlist_payload["config_sha256"],
            "implementation_sha256": implementation,
            "metadata_source": metadata_source,
            "reports": index_records,
        }
        _write_json_create_only(staging / "report_index.json", index_payload)
        _write_json_create_only(shortlist_manifest, shortlist_payload)
        try:
            os.rename(staging, output_root)
        except FileExistsError:
            raise FileExistsError(
                "report output already exists; use a new output root"
            )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        "requests": request_count,
        "snapshot_hits": snapshot_hit_count,
        "cache_hits": cache_hit_count,
        "shortlist": len(shortlist),
        "reports": len(records),
        "errors": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                inspect_task8_inputs(args.config),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    counts = build_pair_reports(
        args.config,
        output_root=args.output_root,
        cache_root=args.cache_root,
    )
    print(json.dumps(counts, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
