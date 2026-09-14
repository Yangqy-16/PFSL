"""Evaluate validation predictions and gate the locked BRCA2--PARP1 case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from esmc_sae.all_context_cache import (
    FULL_DENSE_ROOT,
    pooled_payload_from_dense,
)
import esmc_sae.all_context_dataset  # noqa: F401
from esmc_sae.all_context_train import (
    DATA_ROOT,
    MODEL_ROOT,
    build_all_context_coach_config,
)
from esmc_sae.artifacts import save_cpu_torch, save_json
from esmc_sae.calibrate import CLASSIFICATION_TAU
from esmc_sae.manifest import file_sha256
import pfsl.model  # noqa: F401
import pfsl.module  # noqa: F401


CASE_ROOT = (
    Path(__file__).resolve().parent
    / "case_study/ovcar8_brca2_parp1_all_context"
)
OUTPUT_ROOT = CASE_ROOT / "predictions"
TARGET_POOL_ROOT = CASE_ROOT / "target_pooled_cache"


def assemble_validation_predictions(
    *,
    data_root: Path = DATA_ROOT,
    model_root: Path = MODEL_ROOT,
    expected_rows: int | None = None,
    expected_positives: int | None = None,
) -> pd.DataFrame:
    """Join the single validation prediction file with strict row checks."""

    validation = pd.read_csv(Path(data_root) / "sl_val.csv")
    prediction = pd.read_csv(
        Path(model_root) / "seed_42/test_predictions.csv"
    )
    if list(prediction.columns) != ["0", "1", "prob"]:
        raise RuntimeError("validation prediction columns changed")
    if len(validation) != len(prediction):
        raise RuntimeError("validation prediction row count changed")
    expected_pairs = validation[["0", "1"]].to_numpy(dtype=np.int64)
    observed_pairs = prediction[["0", "1"]].to_numpy(dtype=np.int64)
    if not np.array_equal(expected_pairs, observed_pairs):
        raise RuntimeError("validation prediction pair order changed")
    probabilities = prediction["prob"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
    ):
        raise RuntimeError("validation probabilities are invalid")
    result = pd.DataFrame(
        {
            "row_id": validation["row_id"].astype(str),
            "context": validation["context"].astype(str),
            "gene_a": validation["0"].astype(int),
            "gene_b": validation["1"].astype(int),
            "label": validation["2"].astype(int),
            "probability": probabilities,
        }
    )
    row_count = len(result)
    positive_count = int(result["label"].sum())
    if expected_rows is None:
        expected_rows = row_count
    if expected_positives is None:
        expected_positives = positive_count
    if (
        row_count != expected_rows
        or positive_count != expected_positives
        or result["row_id"].nunique() != expected_rows
    ):
        raise RuntimeError("validation coverage changed")
    return result


def summarize_validation(frame: pd.DataFrame) -> dict[str, object]:
    """Evaluate validation predictions at the fixed model threshold."""

    labels = frame["label"].astype(int).tolist()
    probabilities = frame["probability"].astype(float).tolist()
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


def materialize_locked_target_pool(
    *,
    dense_root: Path = FULL_DENSE_ROOT,
    target_root: Path = TARGET_POOL_ROOT,
    gene_ids: tuple[int, ...] = (142, 675),
) -> dict[str, object]:
    """Create mean caches for quarantined genes from trusted dense caches."""

    dense_payloads: dict[int, dict[str, object]] = {}
    targets = [Path(target_root) / f"{gene_id}.pt" for gene_id in gene_ids]
    for target in targets:
        if target.exists():
            raise FileExistsError(target)
    for gene_id in gene_ids:
        dense = torch.load(
            Path(dense_root) / f"{gene_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        dense_payloads[gene_id] = pooled_payload_from_dense(
            dense,
            gene_id=gene_id,
            sequence_sha256=str(dense["sequence_sha256"]),
        )
    Path(target_root).mkdir(parents=True, exist_ok=True)
    for gene_id, payload in dense_payloads.items():
        save_cpu_torch(Path(target_root) / f"{gene_id}.pt", payload)
    return {
        "gene_ids": sorted(gene_ids),
        "target_root": str(target_root),
        "files": {
            str(gene_id): {
                "path": str(Path(target_root) / f"{gene_id}.pt"),
                "sha256": file_sha256(
                    Path(target_root) / f"{gene_id}.pt"
                ),
            }
            for gene_id in sorted(gene_ids)
        },
    }


def _checkpoint(
    model_root: Path = MODEL_ROOT,
) -> tuple[Path, str]:
    metrics_path = Path(model_root) / "seed_42/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(metrics["best_checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = file_sha256(checkpoint)
    if digest != metrics["best_checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash changed")
    if b"cuda:" in checkpoint.read_bytes():
        raise RuntimeError("checkpoint is not CPU-normalized")
    return checkpoint, digest


def predict_locked_case(
    *,
    target_pool_root: Path = TARGET_POOL_ROOT,
    model_root: Path = MODEL_ROOT,
    data_root: Path = DATA_ROOT,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Strictly load seed 42 and predict the held-out OVCAR8 pair once."""

    device = torch.device(device)
    cfg = build_all_context_coach_config(
        data_root=Path(data_root),
        pool_root=Path(target_pool_root),
        output_root=Path(model_root),
    )
    cfg.DATASET.TEST_FILE = str(Path(data_root) / "locked_case.csv")
    cfg.DATASET.POOLED_ESM_ROOT = str(target_pool_root)
    checkpoint, digest = _checkpoint(model_root)

    module = build_module(cfg)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("checkpoint lacks state_dict")
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
        probability = float(torch.sigmoid(logit.reshape(())).cpu())
    locked = pd.read_csv(Path(data_root) / "locked_case.csv").iloc[0]
    return {
        "seed": 42,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "row_id": str(locked["row_id"]),
        "context": str(locked["context"]),
        "gene_a": int(batch["g1_idx"].item()),
        "gene_b": int(batch["g2_idx"].item()),
        "label": int(batch["label"].item()),
        "logit": scalar,
        "probability": probability,
    }


def run_prediction_gate(
    *,
    output_root: Path = OUTPUT_ROOT,
    target_pool_root: Path = TARGET_POOL_ROOT,
    dense_root: Path = FULL_DENSE_ROOT,
    model_root: Path = MODEL_ROOT,
    data_root: Path = DATA_ROOT,
    gene_ids: tuple[int, ...] = (142, 675),
    case_id: str = "ovcar8_brca2_parp1_all_context",
    expected_rows: int | None = None,
    expected_positives: int | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Persist validation calibration and the single locked prediction."""

    output_root = Path(output_root)
    summary_path = output_root / "prediction_gate.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    target_pool = materialize_locked_target_pool(
        dense_root=dense_root,
        target_root=target_pool_root,
        gene_ids=gene_ids,
    )
    validation = assemble_validation_predictions(
        data_root=data_root,
        model_root=model_root,
        expected_rows=expected_rows,
        expected_positives=expected_positives,
    )
    validation_summary = summarize_validation(validation)
    threshold = float(validation_summary["classification_threshold"])
    case = predict_locked_case(
        target_pool_root=target_pool_root,
        model_root=model_root,
        data_root=data_root,
        device=device,
    )
    case["positive"] = bool(float(case["probability"]) >= threshold)

    output_root.mkdir(parents=True, exist_ok=True)
    validation_path = output_root / "validation_predictions.csv"
    case_path = output_root / "locked_case_prediction.json"
    validation.to_csv(validation_path, index=False, lineterminator="\n")
    save_json(case_path, case)
    result: dict[str, object] = {
        "schema_version": 1,
        "case_id": case_id,
        "seed": 42,
        "validation": validation_summary,
        "locked_case": case,
        "prediction_gate_pass": bool(case["positive"]),
        "target_pool": target_pool,
        "validation_predictions": str(validation_path),
        "validation_predictions_sha256": file_sha256(validation_path),
        "locked_case_prediction": str(case_path),
        "locked_case_prediction_sha256": file_sha256(case_path),
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
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--target-pool-root",
        type=Path,
        default=TARGET_POOL_ROOT,
    )
    parser.add_argument("--dense-root", type=Path, default=FULL_DENSE_ROOT)
    parser.add_argument("--gene-ids", type=int, nargs="+", default=[142, 675])
    parser.add_argument(
        "--case-id",
        default="ovcar8_brca2_parp1_all_context",
    )
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-positives", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_prediction_gate(
        output_root=args.output_root,
        target_pool_root=args.target_pool_root,
        dense_root=args.dense_root,
        model_root=args.model_root,
        data_root=args.data_root,
        gene_ids=tuple(args.gene_ids),
        case_id=args.case_id,
        expected_rows=args.expected_rows,
        expected_positives=args.expected_positives,
        device=args.device,
    )
    print(
        "ALL_CONTEXT_PREDICTION_GATE "
        + json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
