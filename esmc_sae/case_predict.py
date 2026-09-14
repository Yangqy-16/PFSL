"""Assemble OOF predictions and apply the locked-case prediction gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
import pandas as pd
from pytorch_lightning.trainer.states import RunningStage
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
import torch

from coach_pl.dataset import build_dataset
from coach_pl.module import build_module
from esmc_sae.artifacts import save_json
from esmc_sae.calibrate import CLASSIFICATION_TAU
from esmc_sae.case_train import (
    DATA_ROOT,
    MODEL_ROOT,
    build_case_coach_config,
)
import esmc_sae.dataset  # noqa: F401
from esmc_sae.manifest import file_sha256
import pfsl.model  # noqa: F401
import pfsl.module  # noqa: F401


OUTPUT_ROOT = (
    Path(__file__).resolve().parent
    / "case_study/ovcar8_brca2_parp1/predictions"
)


def assemble_oof_predictions(
    *,
    data_root: Path = DATA_ROOT,
    model_root: Path = MODEL_ROOT,
) -> pd.DataFrame:
    """Join fold predictions to validation rows with strict order checks."""

    frames = []
    for fold in range(5):
        validation = pd.read_csv(Path(data_root) / f"sl_val_{fold}.csv")
        prediction = pd.read_csv(
            Path(model_root) / f"fold_{fold}/test_predictions.csv"
        )
        if list(prediction.columns) != ["0", "1", "prob"]:
            raise RuntimeError(f"fold {fold} prediction columns changed")
        if len(validation) != len(prediction):
            raise RuntimeError(f"fold {fold} prediction row count changed")
        expected_pairs = validation[["0", "1"]].to_numpy(dtype=np.int64)
        observed_pairs = prediction[["0", "1"]].to_numpy(dtype=np.int64)
        if not np.array_equal(expected_pairs, observed_pairs):
            raise RuntimeError(f"fold {fold} prediction pair order changed")
        probabilities = prediction["prob"].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or (probabilities > 1.0).any()
        ):
            raise RuntimeError(f"fold {fold} probabilities are invalid")
        frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "row_id": validation["row_id"].astype(str),
                    "gene_a": validation["0"].astype(int),
                    "gene_b": validation["1"].astype(int),
                    "label": validation["2"].astype(int),
                    "probability": probabilities,
                }
            )
        )

    result = pd.concat(frames, ignore_index=True)
    if len(result) != 919 or result["row_id"].nunique() != 919:
        raise RuntimeError("OOF predictions do not exactly cover 919 rows")
    if int(result["label"].sum()) != 34:
        raise RuntimeError("OOF positive count changed")
    return result.sort_values("row_id", kind="stable").reset_index(drop=True)


def summarize_oof(frame: pd.DataFrame) -> dict[str, object]:
    """Evaluate OOF predictions at the model's fixed decision threshold."""

    labels = [int(value) for value in frame["label"]]
    probabilities = [float(value) for value in frame["probability"]]
    threshold = CLASSIFICATION_TAU
    predicted = [int(value >= threshold) for value in probabilities]
    return {
        "rows": len(frame),
        "positive_rows": sum(labels),
        "classification_threshold": threshold,
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predicted)
        ),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "predicted_positive_rows": sum(predicted),
    }


def _checkpoint_for_fold(fold: int, model_root: Path) -> tuple[Path, str]:
    metrics_path = Path(model_root) / f"fold_{fold}/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = Path(metrics["best_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = file_sha256(checkpoint)
    if digest != metrics["best_checkpoint_sha256"]:
        raise RuntimeError(f"fold {fold} checkpoint hash changed")
    if b"cuda:" in checkpoint.read_bytes():
        raise RuntimeError(f"fold {fold} checkpoint is not CPU-normalized")
    return checkpoint, digest


def predict_locked_case(
    fold: int,
    *,
    model_root: Path = MODEL_ROOT,
    data_root: Path = DATA_ROOT,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Strictly load one fold and predict BRCA2--PARP1 once."""

    device = torch.device(device)
    cfg = build_case_coach_config(fold, output_root=Path(model_root))
    cfg.DATASET.TEST_FILE = str(Path(data_root) / "locked_case.csv")
    checkpoint, digest = _checkpoint_for_fold(fold, Path(model_root))

    module = build_module(cfg)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"fold {fold} checkpoint lacks state_dict")
    module.load_state_dict(state_dict, strict=True)
    module.eval().requires_grad_(False).to(device)

    dataset = build_dataset(cfg, RunningStage.TESTING)
    if len(dataset) != 1:
        raise RuntimeError("locked case file must contain exactly one row")
    batch = dataset.collate_fn([dataset[0]])
    transferred = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        output = module.model(transferred)
        logit = output[0] if isinstance(output, tuple) else output
        scalar = float(logit.reshape(()).cpu())
    probability = 1.0 / (1.0 + math.exp(-scalar))
    return {
        "fold": fold,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "gene_a": int(batch["g1_idx"].item()),
        "gene_b": int(batch["g2_idx"].item()),
        "label": int(batch["label"].item()),
        "logit": scalar,
        "probability": probability,
    }


def _write_csv_create_only(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False, lineterminator="\n")
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def run_prediction_gate(
    *,
    output_root: Path = OUTPUT_ROOT,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Create immutable OOF and locked-case prediction artifacts."""

    output_root = Path(output_root)
    summary_path = output_root / "prediction_gate.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)

    oof = assemble_oof_predictions()
    oof_summary = summarize_oof(oof)
    threshold = float(oof_summary["classification_threshold"])
    case_rows = [
        predict_locked_case(fold, device=device) for fold in range(5)
    ]
    for row in case_rows:
        row["positive"] = bool(float(row["probability"]) >= threshold)
    votes = sum(bool(row["positive"]) for row in case_rows)

    oof_path = output_root / "oof_predictions.csv"
    case_path = output_root / "locked_case_predictions.csv"
    _write_csv_create_only(oof_path, oof)
    _write_csv_create_only(case_path, pd.DataFrame(case_rows))
    result: dict[str, object] = {
        "schema_version": 1,
        "case_id": "ovcar8_brca2_parp1",
        "oof": oof_summary,
        "case_positive_votes": votes,
        "required_positive_votes": 4,
        "prediction_gate_pass": votes >= 4,
        "oof_predictions": str(oof_path),
        "oof_predictions_sha256": file_sha256(oof_path),
        "locked_case_predictions": str(case_path),
        "locked_case_predictions_sha256": file_sha256(case_path),
    }
    save_json(summary_path, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_prediction_gate(device=args.device)
    print(
        "CASE_PREDICTION_GATE "
        + json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
