"""Leakage-resistant held-out manifest, blind inference, and unblinding."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

from omegaconf import OmegaConf
from pytorch_lightning.trainer.states import RunningStage

from esmc_sae.attribution import (
    _load_validated_decoder,
    _verify_trusted_audit,
)
from esmc_sae.calibrate import (
    FrozenThresholds,
    _publish_temp_no_replace,
    _write_json_create_only,
    _write_jsonl_create_only,
)
from esmc_sae.config import ProjectConfig, load_config
from esmc_sae.dataset import ESMCSAELayer60ProtDataset
from esmc_sae.extract import _load_sequences
from esmc_sae.intervention import _run_pair
from esmc_sae.manifest import file_sha256
from esmc_sae.sl_adapter import FORMAL_CHECKPOINT_NAME, FrozenSLAdapter
from esmc_sae.train_dense import build_coach_config


_FORBIDDEN_LABEL_COLUMNS = frozenset({"2", "label", "y", "target"})


def _atomic_csv_create_only(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, object]],
) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}; use a new run ID")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temp_no_replace(Path(temporary), path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_text_create_only(path: Path, contents: str) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}; use a new run ID")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
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


def make_blind(
    source_csv: Path,
    blind_csv: Path,
    source_checksum_path: Path,
) -> None:
    """Copy stable row identity and gene IDs, recording source bytes separately."""

    source_csv = Path(source_csv)
    blind_csv = Path(blind_csv)
    source_checksum_path = Path(source_checksum_path)
    if blind_csv.exists() or source_checksum_path.exists():
        raise FileExistsError(
            "blind manifest or source checksum already exists; use a new run ID"
        )
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"0", "1", "2"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("held-out source requires columns 0, 1, and 2")
        rows = []
        for index, row in enumerate(reader):
            gene_a = int(row["0"])
            gene_b = int(row["1"])
            rows.append(
                {
                    "row_id": f"row_{index:06d}",
                    "0": gene_a,
                    "1": gene_b,
                }
            )
    if not rows:
        raise ValueError("held-out source must not be empty")
    digest = file_sha256(source_csv)
    _atomic_csv_create_only(blind_csv, ("row_id", "0", "1"), rows)
    try:
        _atomic_text_create_only(
            source_checksum_path, f"{digest}  {source_csv.name}\n"
        )
    except BaseException:
        blind_csv.unlink(missing_ok=True)
        raise


def _load_blind_rows(path: Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError("blind table is missing a header")
        forbidden = sorted(
            field for field in fields if field.strip().lower() in _FORBIDDEN_LABEL_COLUMNS
        )
        if forbidden:
            raise ValueError(f"blind table contains label column: {forbidden[0]}")
        if fields != ["row_id", "0", "1"]:
            raise ValueError("blind table columns must be exactly row_id, 0, 1")
        rows = []
        seen = set()
        for row in reader:
            row_id = row["row_id"]
            if not row_id or row_id in seen:
                raise ValueError("blind row_id values must be non-empty and unique")
            seen.add(row_id)
            rows.append(
                {
                    "row_id": row_id,
                    "gene_a": int(row["0"]),
                    "gene_b": int(row["1"]),
                }
            )
    if not rows:
        raise ValueError("blind table must not be empty")
    return rows


def _threshold_path(project: ProjectConfig) -> Path:
    return (
        project.paths.project_root
        / "configs"
        / (
            f"frozen_thresholds_{project.experiment.cell_line}_"
            f"{project.experiment.scene}_fold{project.experiment.fold}.json"
        )
    )


def _load_thresholds(
    project: ProjectConfig,
    config_path: Path,
) -> FrozenThresholds:
    path = _threshold_path(project)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "schema_version",
        "config_sha256",
        "candidate_k",
        "single_partial_abs_delta",
        "single_reliable_abs_delta",
        "interaction_reliable_abs_j",
        "classification_tau",
        "checkpoint_sha256",
        "validation_csv_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("frozen thresholds have an unexpected schema")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported frozen threshold schema")
    if payload["config_sha256"] != file_sha256(config_path):
        raise ValueError("frozen threshold config checksum mismatch")
    checkpoint = (
        project.training.output_dir / "best_ckpts" / FORMAL_CHECKPOINT_NAME
    )
    if payload["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError("frozen threshold checkpoint checksum mismatch")
    if payload["validation_csv_sha256"] != file_sha256(project.training.val_file):
        raise ValueError("frozen threshold validation CSV checksum mismatch")
    if payload["candidate_k"] not in (3, 5, 10):
        raise ValueError("frozen candidate_k is invalid")
    numeric_names = (
        "single_partial_abs_delta",
        "single_reliable_abs_delta",
        "interaction_reliable_abs_j",
        "classification_tau",
    )
    for name in numeric_names:
        value = payload[name]
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError(f"frozen {name} must be finite numeric")
    return FrozenThresholds(
        candidate_k=int(payload["candidate_k"]),
        single_partial_abs_delta=float(payload["single_partial_abs_delta"]),
        single_reliable_abs_delta=float(payload["single_reliable_abs_delta"]),
        interaction_reliable_abs_j=float(
            payload["interaction_reliable_abs_j"]
        ),
        classification_tau=float(payload["classification_tau"]),
        checkpoint_sha256=payload["checkpoint_sha256"],
        validation_csv_sha256=payload["validation_csv_sha256"],
    )


class _LabelFreeDataset:
    def __init__(self, dataset: ESMCSAELayer60ProtDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.dataset[index]

    @property
    def collate_fn(self):
        parent_collate = self.dataset.collate_fn

        def collate(batch):
            fixed_batch = parent_collate(batch)
            fixed_batch.pop("label", None)
            return fixed_batch

        return collate


def _build_label_free_dataset(
    project: ProjectConfig,
    blind_path: Path,
) -> _LabelFreeDataset:
    coach_config = build_coach_config(project, [])
    cfg = OmegaConf.create(OmegaConf.to_container(coach_config, resolve=True))
    cfg.DATASET.TEST_FILE = str(blind_path)
    cfg.MODEL.USE_SEQ = False
    dataset = ESMCSAELayer60ProtDataset(cfg, RunningStage.TESTING)
    dataset.df["2"] = 0.0
    return _LabelFreeDataset(dataset)


def _trim_candidates(payload: dict[str, object], k: int) -> None:
    singles = payload["singles"]
    selected = []
    selected_ids: dict[str, set[int]] = {"A": set(), "B": set()}
    for side in ("A", "B"):
        side_values = [item for item in singles if item["side"] == side][:k]
        selected.extend(side_values)
        selected_ids[side].update(int(item["feature_id"]) for item in side_values)
    payload["singles"] = selected
    payload["interactions"] = [
        item
        for item in payload["interactions"]
        if int(item["feature_a"]) in selected_ids["A"]
        and int(item["feature_b"]) in selected_ids["B"]
    ]


def run_blind(
    config: Path,
    *,
    blind_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    """Run label-free held-out intervention inference and atomically finalize."""

    config = Path(config)
    project = load_config(config)
    blind_path = (
        Path(blind_path)
        if blind_path is not None
        else project.paths.project_root / "manifests/heldout_pairs_blind.csv"
    )
    output_path = (
        Path(output_path)
        if output_path is not None
        else project.paths.project_root / "experiment2/heldout/blind_results.jsonl"
    )
    rows = _load_blind_rows(blind_path)
    if output_path.exists():
        raise FileExistsError(
            f"blind results already exist: {output_path}; use a new run ID"
        )
    thresholds = _load_thresholds(project, config)
    _verify_trusted_audit(project)
    checkpoint = (
        project.training.output_dir / "best_ckpts" / FORMAL_CHECKPOINT_NAME
    )
    dataset = _build_label_free_dataset(project, blind_path)
    if len(dataset) != len(rows):
        raise RuntimeError("blind dataset row count changed during loading")
    decoder, provenance_fingerprint = _load_validated_decoder(project)
    sequences = _load_sequences(project.paths.sequences)
    adapter = FrozenSLAdapter.from_checkpoint(project, checkpoint)
    artifact_bindings = {
        "blind_manifest_sha256": file_sha256(blind_path),
        "frozen_thresholds_sha256": file_sha256(_threshold_path(project)),
        "checkpoint_sha256": thresholds.checkpoint_sha256,
        "sae_provenance_fingerprint": provenance_fingerprint,
    }

    results = []
    for row_index, row in enumerate(rows):
        payload = _run_pair(
            project=project,
            dataset=dataset,
            row_index=row_index,
            sequences=sequences,
            decoder=decoder,
            provenance_fingerprint=provenance_fingerprint,
            adapter=adapter,
            checkpoint=checkpoint,
            split="test_blind",
            row_id=row["row_id"],
            include_control_interactions=True,
        )
        if (
            payload["gene_a"] != row["gene_a"]
            or payload["gene_b"] != row["gene_b"]
        ):
            raise RuntimeError("blind dataset gene identity mismatch")
        _trim_candidates(payload, thresholds.candidate_k)
        predicted = int(payload["probability"] >= thresholds.classification_tau)
        candidate_abs = [
            abs(float(item["delta_logit"])) for item in payload["singles"]
        ]
        interaction_abs = [
            abs(float(item["interaction_j"])) for item in payload["interactions"]
        ]
        payload["predicted_label"] = predicted
        payload["max_candidate_abs_delta"] = max(candidate_abs, default=0.0)
        payload["max_interaction_abs_j"] = max(interaction_abs, default=0.0)
        payload["artifact_bindings"] = artifact_bindings
        results.append(payload)
    _write_jsonl_create_only(output_path, results)


def classify_unblinded(
    *,
    true_label: int,
    probability: float,
    max_candidate_abs_delta: float,
    max_interaction_abs_j: float,
    max_control_abs_delta: float,
    max_control_interaction_abs_j: float,
    thresholds: FrozenThresholds,
    patching_enabled: bool,
) -> dict[str, object]:
    """Classify numeric outcome and explanation reliability on separate axes."""

    if type(true_label) is not int or true_label not in (0, 1):
        raise ValueError("true_label must be a binary integer")
    numeric = (
        probability,
        max_candidate_abs_delta,
        max_interaction_abs_j,
        max_control_abs_delta,
        max_control_interaction_abs_j,
    )
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in numeric):
        raise ValueError("classification inputs must be finite numeric values")
    predicted = int(probability >= thresholds.classification_tau)
    prediction_type = {
        (1, 1): "TP",
        (0, 1): "FP",
        (1, 0): "FN",
        (0, 0): "TN",
    }[(true_label, predicted)]
    if predicted == 0 and not patching_enabled:
        explanation_status = "abstain"
    elif (
        (
            max_candidate_abs_delta >= thresholds.single_reliable_abs_delta
            and max_candidate_abs_delta > max_control_abs_delta
        )
        or (
            max_interaction_abs_j >= thresholds.interaction_reliable_abs_j
            and max_interaction_abs_j > max_control_interaction_abs_j
        )
    ):
        explanation_status = "reliable"
    elif max_candidate_abs_delta >= thresholds.single_partial_abs_delta:
        explanation_status = "partial"
    else:
        explanation_status = "abstain"
    return {
        "true_label": true_label,
        "predicted_label": predicted,
        "prediction_type": prediction_type,
        "explanation_status": explanation_status,
    }


def _join_blind_results_by_row_id(
    source_rows: Sequence[dict[str, object]],
    blind_records: Sequence[dict[str, object]],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    by_id: dict[str, dict[str, object]] = {}
    for record in blind_records:
        row_id = record.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("blind result row_id must be a non-empty string")
        if row_id in by_id:
            raise ValueError(f"duplicate blind result row_id: {row_id}")
        by_id[row_id] = record

    expected_ids = {
        f"row_{index:06d}" for index in range(len(source_rows))
    }
    unknown = sorted(set(by_id).difference(expected_ids))
    if unknown:
        raise ValueError(f"unknown blind result row_id: {unknown[0]}")
    missing = sorted(expected_ids.difference(by_id))
    if missing:
        raise ValueError(f"missing blind result row_id: {missing[0]}")

    joined = []
    for index, source in enumerate(source_rows):
        row_id = f"row_{index:06d}"
        blind = by_id[row_id]
        if (
            int(source["0"]) != int(blind["gene_a"])
            or int(source["1"]) != int(blind["gene_b"])
        ):
            raise ValueError(f"gene identity mismatch for row_id: {row_id}")
        joined.append((source, blind))
    return joined


def _validate_blind_result_bindings(
    records: Sequence[dict[str, object]],
    expected: dict[str, str],
) -> None:
    required = {
        "blind_manifest_sha256",
        "frozen_thresholds_sha256",
        "checkpoint_sha256",
        "sae_provenance_fingerprint",
    }
    if set(expected) != required:
        raise ValueError("expected blind artifact bindings are incomplete")
    if not records:
        raise ValueError("blind results must not be empty")
    first = records[0].get("artifact_bindings")
    if not isinstance(first, dict) or set(first) != required:
        raise ValueError("blind result artifact bindings are missing or malformed")
    for record in records[1:]:
        if record.get("artifact_bindings") != first:
            raise ValueError("mixed blind artifact bindings")

    labels = {
        "blind_manifest_sha256": "blind manifest",
        "frozen_thresholds_sha256": "frozen thresholds",
        "checkpoint_sha256": "checkpoint",
        "sae_provenance_fingerprint": "SAE provenance",
    }
    for name, label in labels.items():
        if first[name] != expected[name]:
            raise ValueError(f"{label} binding mismatch")

    forbidden = {"true_label", "2", "label", "y", "target"}
    for record in records:
        if forbidden.intersection(str(key).lower() for key in record):
            raise ValueError("blind result contains a label field")
        checkpoint = record.get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("sha256") != first["checkpoint_sha256"]
        ):
            raise ValueError("checkpoint binding mismatch")
        provenance = record.get("provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("sae_fingerprint")
            != first["sae_provenance_fingerprint"]
        ):
            raise ValueError("SAE provenance binding mismatch")


def _validate_records_match_blind_manifest(
    blind_rows: Sequence[dict[str, object]],
    records: Sequence[dict[str, object]],
) -> None:
    expected = {str(row["row_id"]): row for row in blind_rows}
    actual: dict[str, dict[str, object]] = {}
    for record in records:
        row_id = record.get("row_id")
        if not isinstance(row_id, str) or row_id in actual:
            raise ValueError("duplicate or malformed blind result row_id")
        actual[row_id] = record
    unknown = sorted(set(actual).difference(expected))
    missing = sorted(set(expected).difference(actual))
    if unknown:
        raise ValueError(f"unknown blind result row_id: {unknown[0]}")
    if missing:
        raise ValueError(f"missing blind result row_id: {missing[0]}")
    for row_id, row in expected.items():
        record = actual[row_id]
        if (
            int(record["gene_a"]) != int(row["gene_a"])
            or int(record["gene_b"]) != int(row["gene_b"])
        ):
            raise ValueError(f"blind manifest gene identity mismatch: {row_id}")


def _read_expected_checksum(path: Path) -> str:
    fields = Path(path).read_text(encoding="ascii").split()
    if (
        len(fields) < 1
        or len(fields[0]) != 64
        or any(character not in "0123456789abcdef" for character in fields[0])
    ):
        raise ValueError("source checksum record is malformed")
    return fields[0]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"JSONL artifact is empty or malformed: {path}")
    return records


def _method_summary(records: Sequence[dict[str, object]]) -> dict[str, object]:
    prediction_counts = Counter(record["prediction_type"] for record in records)
    status_counts = Counter(record["explanation_status"] for record in records)
    correct = sum(
        record["true_label"] == record["predicted_label"] for record in records
    )
    positives = [record for record in records if record["true_label"] == 1]
    negatives = [record for record in records if record["true_label"] == 0]
    sensitivity = (
        sum(record["predicted_label"] == 1 for record in positives) / len(positives)
        if positives
        else None
    )
    specificity = (
        sum(record["predicted_label"] == 0 for record in negatives) / len(negatives)
        if negatives
        else None
    )

    def value_stats(values: Sequence[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "mean": sum(values) / len(values),
            "median": float(ordered[len(values) // 2])
            if len(values) % 2
            else float(
                (
                    ordered[len(values) // 2 - 1]
                    + ordered[len(values) // 2]
                )
                / 2
            ),
            "max": max(values),
        }

    def stats(name: str) -> dict[str, float]:
        return value_stats([float(record[name]) for record in records])

    control_singles = [
        abs(float(value))
        for record in records
        for value in record["control_abs_deltas"]
    ]
    control_interactions = [
        abs(float(item["interaction_j"]))
        for record in records
        for item in record["control_interactions"]
    ]
    return {
        "schema_version": 1,
        "n_pairs": len(records),
        "prediction_type_counts": dict(sorted(prediction_counts.items())),
        "explanation_status_counts": dict(sorted(status_counts.items())),
        "classification_statistics": {
            "accuracy": correct / len(records),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": (
                (sensitivity + specificity) / 2
                if sensitivity is not None and specificity is not None
                else None
            ),
        },
        "intervention_statistics": {
            "max_candidate_abs_delta": stats("max_candidate_abs_delta"),
            "max_interaction_abs_j": stats("max_interaction_abs_j"),
        },
        "control_statistics": {
            "single_abs_delta": value_stats(control_singles),
            "interaction_abs_j": value_stats(control_interactions),
        },
    }


def unblind(
    config: Path,
    *,
    source_csv: Path | None = None,
    source_checksum_path: Path | None = None,
    blind_manifest_path: Path | None = None,
    blind_results_path: Path | None = None,
    output_path: Path | None = None,
    summary_path: Path | None = None,
) -> None:
    """Verify source identity, join held-out labels, and emit numeric outcomes."""

    config = Path(config)
    project = load_config(config)
    source_csv = Path(source_csv) if source_csv is not None else project.training.test_file
    source_checksum_path = (
        Path(source_checksum_path)
        if source_checksum_path is not None
        else project.paths.project_root / "manifests/heldout_test_source.sha256"
    )
    blind_manifest_path = (
        Path(blind_manifest_path)
        if blind_manifest_path is not None
        else project.paths.project_root / "manifests/heldout_pairs_blind.csv"
    )
    blind_results_path = (
        Path(blind_results_path)
        if blind_results_path is not None
        else project.paths.project_root / "experiment2/heldout/blind_results.jsonl"
    )
    output_path = (
        Path(output_path)
        if output_path is not None
        else project.paths.project_root
        / "experiment2/heldout/unblinded_results.jsonl"
    )
    summary_path = (
        Path(summary_path)
        if summary_path is not None
        else project.paths.project_root / "reports/experiment2_summary.json"
    )
    if output_path.exists() or summary_path.exists():
        raise FileExistsError(
            "unblinded result or summary already exists; use a new run ID"
        )
    expected_checksum = _read_expected_checksum(source_checksum_path)
    if file_sha256(source_csv) != expected_checksum:
        raise ValueError("held-out source checksum mismatch")

    thresholds = _load_thresholds(project, config)
    blind_records = _read_jsonl(blind_results_path)
    _verify_trusted_audit(project)
    _, provenance_fingerprint = _load_validated_decoder(project)
    expected_bindings = {
        "blind_manifest_sha256": file_sha256(blind_manifest_path),
        "frozen_thresholds_sha256": file_sha256(_threshold_path(project)),
        "checkpoint_sha256": thresholds.checkpoint_sha256,
        "sae_provenance_fingerprint": provenance_fingerprint,
    }
    _validate_blind_result_bindings(blind_records, expected_bindings)
    _validate_records_match_blind_manifest(
        _load_blind_rows(blind_manifest_path), blind_records
    )
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    unblinded = []
    for source, blind in _join_blind_results_by_row_id(
        source_rows, blind_records
    ):
        true_label = int(source["2"])
        outcome = classify_unblinded(
            true_label=true_label,
            probability=float(blind["probability"]),
            max_candidate_abs_delta=float(blind["max_candidate_abs_delta"]),
            max_interaction_abs_j=float(blind["max_interaction_abs_j"]),
            max_control_abs_delta=max(
                abs(float(value))
                for value in blind["control_abs_deltas"]
            ),
            max_control_interaction_abs_j=max(
                abs(float(item["interaction_j"]))
                for item in blind["control_interactions"]
            ),
            thresholds=thresholds,
            patching_enabled=project.explanation.enable_patching,
        )
        unblinded.append({**blind, **outcome})

    summary = _method_summary(unblinded)
    _write_jsonl_create_only(output_path, unblinded)
    try:
        _write_json_create_only(summary_path, summary)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("make-blind", "run-blind", "unblind"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project = load_config(args.config)
    if args.command == "make-blind":
        make_blind(
            project.training.test_file,
            project.paths.project_root / "manifests/heldout_pairs_blind.csv",
            project.paths.project_root / "manifests/heldout_test_source.sha256",
        )
    elif args.command == "run-blind":
        run_blind(args.config)
    else:
        unblind(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
