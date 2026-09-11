"""Calibrate explanation thresholds while retaining the model's 0.5 rule."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import numpy as np
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.dataset import build_dataset
from esmc_sae.attribution import (
    _load_validated_decoder,
    _verify_trusted_audit,
)
from esmc_sae.config import load_config
from esmc_sae.extract import _load_sequences
from esmc_sae.intervention import (
    InteractionResult,
    InterventionResult,
    PairInterventionRecord,
    _run_pair,
)
from esmc_sae.manifest import file_sha256
from esmc_sae.sl_adapter import FORMAL_CHECKPOINT_NAME, FrozenSLAdapter
from esmc_sae.train_dense import build_coach_config


CLASSIFICATION_TAU = 0.5


@dataclass(frozen=True)
class FrozenThresholds:
    candidate_k: int
    single_partial_abs_delta: float
    single_reliable_abs_delta: float
    interaction_reliable_abs_j: float
    classification_tau: float
    checkpoint_sha256: str
    validation_csv_sha256: str


def _numeric_sequence(values: Sequence[object], label: str) -> list[float]:
    result = []
    for value in values:
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError(f"{label} must contain finite numeric values")
        result.append(float(value))
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _captured(record: PairInterventionRecord, k: int) -> float:
    by_side: dict[str, list[float]] = {}
    for result in record.singles:
        by_side.setdefault(result.side, []).append(abs(result.delta_logit))
    numerator = sum(sum(values[:k]) for values in by_side.values())
    denominator = sum(sum(values[:10]) for values in by_side.values())
    return numerator / max(denominator, np.finfo(np.float64).eps)


def calibrate(
    validation_results: Sequence[PairInterventionRecord],
    checkpoint_sha256: str,
    validation_csv_sha256: str,
) -> FrozenThresholds:
    """Calibrate from structured numeric intervention records only."""

    if not validation_results:
        raise ValueError("validation_results must not be empty")
    if any(type(record) is not PairInterventionRecord for record in validation_results):
        raise TypeError(
            "validation_results accepts PairInterventionRecord values only; "
            "text or LLM mappings are forbidden"
        )
    for label, digest in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("validation_csv_sha256", validation_csv_sha256),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    choices = (3, 5, 10)
    candidate_k = 10
    for choice in choices:
        mean_capture = sum(_captured(record, choice) for record in validation_results)
        mean_capture /= len(validation_results)
        if mean_capture >= 0.90:
            candidate_k = choice
            break

    controls = [
        abs(value)
        for record in validation_results
        for value in _numeric_sequence(
            record.control_abs_deltas, "control_abs_deltas"
        )
    ]
    control_js = [
        abs(value)
        for record in validation_results
        for value in _numeric_sequence(
            record.control_interaction_abs_js,
            "control_interaction_abs_js",
        )
    ]
    for record in validation_results:
        if record.true_label not in (0, 1):
            raise ValueError("every validation record requires a binary true_label")

    return FrozenThresholds(
        candidate_k=candidate_k,
        single_partial_abs_delta=float(np.median(controls)),
        single_reliable_abs_delta=float(np.quantile(controls, 0.95)),
        interaction_reliable_abs_j=float(np.quantile(control_js, 0.95)),
        classification_tau=CLASSIFICATION_TAU,
        checkpoint_sha256=checkpoint_sha256,
        validation_csv_sha256=validation_csv_sha256,
    )


def freeze_thresholds(
    path: Path,
    thresholds: FrozenThresholds,
    *,
    config_sha256: str,
) -> None:
    """Create one immutable threshold JSON; never replace an existing run."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"frozen thresholds already exist at {path}; use a new run ID"
        )
    payload = {"schema_version": 1, "config_sha256": config_sha256}
    payload.update(asdict(thresholds))
    _write_json_create_only(path, payload)


def record_from_payload(payload: dict[str, object]) -> PairInterventionRecord:
    singles = tuple(
        InterventionResult(
            feature_id=int(item["feature_id"]),
            side=item["side"],
            original_logit=float(item["original_logit"]),
            ablated_logit=float(item["ablated_logit"]),
            delta_logit=float(item["delta_logit"]),
        )
        for item in payload["singles"]
    )
    interactions = tuple(
        InteractionResult(
            feature_a=int(item["feature_a"]),
            feature_b=int(item["feature_b"]),
            delta_a=float(item["delta_a"]),
            delta_b=float(item["delta_b"]),
            delta_ab=float(item["delta_ab"]),
            interaction_j=float(item["interaction_j"]),
        )
        for item in payload["interactions"]
    )
    return PairInterventionRecord(
        row_id=str(payload["row_id"]),
        gene_a=int(payload["gene_a"]),
        gene_b=int(payload["gene_b"]),
        original_logit=float(payload["original_logit"]),
        probability=float(payload["probability"]),
        singles=singles,
        interactions=interactions,
        control_abs_deltas=tuple(
            float(value) for value in payload["control_abs_deltas"]
        ),
        control_interaction_abs_js=tuple(
            abs(float(item["interaction_j"]))
            for item in payload["control_interactions"]
        ),
        true_label=int(payload["true_label"]),
    )


def _write_jsonl_create_only(
    path: Path,
    records: Iterable[dict[str, object]],
) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}; use a new run ID")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temp_no_replace(Path(temporary), path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _publish_temp_no_replace(temporary: Path, path: Path) -> None:
    """Atomically publish one complete same-filesystem temp without replacement."""

    temporary = Path(temporary)
    path = Path(path)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(
            f"artifact already exists: {path}; use a new run ID"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_json_create_only(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}; use a new run ID")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temp_no_replace(Path(temporary), path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project = load_config(args.config)
    output_dir = project.paths.project_root / "experiment2/calibration"
    result_path = output_dir / "validation_results.jsonl"
    threshold_path = (
        project.paths.project_root
        / "configs"
        / (
            f"frozen_thresholds_{project.experiment.cell_line}_"
            f"{project.experiment.scene}_fold{project.experiment.fold}.json"
        )
    )
    if result_path.exists() or threshold_path.exists():
        raise RuntimeError(
            "NEEDS_CONTEXT: formal calibration artifact already exists; "
            "do not overwrite it"
        )

    _verify_trusted_audit(project)
    checkpoint = (
        project.training.output_dir / "best_ckpts" / FORMAL_CHECKPOINT_NAME
    )
    coach_config = build_coach_config(project, [])
    dataset = build_dataset(coach_config, RunningStage.VALIDATING)
    decoder, provenance_fingerprint = _load_validated_decoder(project)
    sequences = _load_sequences(project.paths.sequences)
    adapter = FrozenSLAdapter.from_checkpoint(project, checkpoint)

    payloads = []
    records = []
    for row_index in range(len(dataset)):
        payload = _run_pair(
            project=project,
            dataset=dataset,
            row_index=row_index,
            sequences=sequences,
            decoder=decoder,
            provenance_fingerprint=provenance_fingerprint,
            adapter=adapter,
            checkpoint=checkpoint,
            split="val",
            row_id=f"row_{row_index:06d}",
            include_control_interactions=True,
        )
        payload["true_label"] = int(dataset.df["2"][row_index])
        payloads.append(payload)
        records.append(record_from_payload(payload))

    checkpoint_sha256 = file_sha256(checkpoint)
    validation_sha256 = file_sha256(project.training.val_file)
    thresholds = calibrate(records, checkpoint_sha256, validation_sha256)
    _write_jsonl_create_only(result_path, payloads)
    freeze_thresholds(
        threshold_path,
        thresholds,
        config_sha256=file_sha256(args.config),
    )
    print(
        "CALIBRATION_RESULT "
        + json.dumps(asdict(thresholds), sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
