"""Local active-feature scoring for one frozen ProSL validation pair."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from pytorch_lightning.trainer.states import RunningStage
import torch

from coach_pl.dataset import build_dataset
from esmc_sae.audit import (
    _decoder_from_payload,
    _load_cache_pair,
    _load_trusted_decoder_payload,
)
from esmc_sae.config import ProjectConfig, load_config
from esmc_sae.extract import (
    DecoderBridge,
    SparseProteinFeatures,
    _load_sequences,
    _normalize_sequence,
)
from esmc_sae.sl_adapter import (
    FORMAL_CHECKPOINT_NAME,
    FrozenSLAdapter,
    _sha256,
    _verify_manifest_target,
)
from esmc_sae.train_dense import build_coach_config


SCORE_CHUNK_FEATURES = 64
TOP_FEATURES_PER_SIDE = 10


@dataclass(frozen=True)
class FeatureScore:
    feature_id: int
    side: Literal["A", "B"]
    signed_score: float
    activation_sum: float
    active_positions: tuple[int, ...]


def _validate_scoring_inputs(
    h: torch.Tensor,
    sparse: SparseProteinFeatures,
    decoder: DecoderBridge,
    grad_h: torch.Tensor,
    side: str,
) -> None:
    if side not in ("A", "B"):
        raise ValueError("side must be 'A' or 'B'")
    if h.ndim != 2 or grad_h.shape != h.shape:
        raise ValueError("hidden state and gradient must share shape [L, D]")
    if sparse.indices.ndim != 2 or sparse.values.shape != sparse.indices.shape:
        raise ValueError("sparse indices and values must share shape [L, K]")
    if sparse.indices.shape[0] != h.shape[0]:
        raise ValueError("sparse features and hidden state must align by residue")
    if sparse.indices.shape[1] != decoder.topk:
        raise ValueError("sparse width must match decoder Top-K")
    if h.shape[1] != decoder.d_model or decoder.W_dec.shape != (
        decoder.codebook_dim,
        decoder.d_model,
    ):
        raise ValueError("hidden state and decoder dimensions do not match")
    if not torch.is_floating_point(sparse.values):
        raise ValueError("sparse values must be floating point")
    if not bool(torch.isfinite(h).all()) or not bool(
        torch.isfinite(grad_h).all()
    ):
        raise ValueError("hidden state and gradient must be finite")

    indices = sparse.indices
    if bool((indices < -1).any()) or bool(
        (indices >= decoder.codebook_dim).any()
    ):
        raise ValueError("sparse feature index is outside the codebook")


def _compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    if any(tensor.dtype == torch.float64 for tensor in tensors):
        return torch.float64
    return torch.float32


def score_active_features(
    h: torch.Tensor,
    sparse: SparseProteinFeatures,
    decoder: DecoderBridge,
    grad_h: torch.Tensor,
    side: Literal["A", "B"],
    *,
    limit: int | None = TOP_FEATURES_PER_SIDE,
) -> list[FeatureScore]:
    """Score only active sparse IDs using original-coordinate decoder directions."""

    _validate_scoring_inputs(h, sparse, decoder, grad_h, side)
    flat_indices_cpu = sparse.indices.detach().to(
        device="cpu",
        dtype=torch.int64,
    ).reshape(-1)
    flat_values_cpu = sparse.values.detach().to(device="cpu").reshape(-1)
    active_cpu = flat_indices_cpu >= 0
    if not bool(active_cpu.any()):
        return []

    feature_ids_cpu, inverse_cpu = torch.unique(
        flat_indices_cpu[active_cpu],
        sorted=True,
        return_inverse=True,
    )
    width = sparse.indices.shape[1]
    residue_rows_cpu = (
        torch.arange(sparse.indices.shape[0], dtype=torch.int64)
        .unsqueeze(1)
        .expand(-1, width)
        .reshape(-1)[active_cpu]
    )
    active_values_cpu = flat_values_cpu[active_cpu]

    device = grad_h.device
    compute_dtype = _compute_dtype(
        h,
        grad_h,
        sparse.values,
        decoder.W_dec,
    )
    hidden = h.detach().to(device=device, dtype=compute_dtype)
    scaled_grad = grad_h.detach().to(device=device, dtype=compute_dtype)
    scaled_grad = scaled_grad * (
        hidden.std(dim=-1, keepdim=True) + decoder.eps
    )

    score_by_id: dict[int, float] = {}
    for start in range(0, feature_ids_cpu.numel(), SCORE_CHUNK_FEATURES):
        stop = min(start + SCORE_CHUNK_FEATURES, feature_ids_cpu.numel())
        chunk_ids_cpu = feature_ids_cpu[start:stop]
        chunk_size = stop - start
        directions = decoder.W_dec.index_select(
            0,
            chunk_ids_cpu.to(decoder.W_dec.device),
        ).to(device=device, dtype=compute_dtype)
        directional_derivatives = scaled_grad @ directions.transpose(0, 1)

        occurrence_mask_cpu = (inverse_cpu >= start) & (inverse_cpu < stop)
        occurrence_rows = residue_rows_cpu[occurrence_mask_cpu].to(device)
        occurrence_cols = (
            inverse_cpu[occurrence_mask_cpu] - start
        ).to(device)
        occurrence_values = active_values_cpu[occurrence_mask_cpu].to(
            device=device,
            dtype=compute_dtype,
        )
        activations = torch.zeros(
            (h.shape[0], chunk_size),
            device=device,
            dtype=compute_dtype,
        )
        activations.view(-1).scatter_add_(
            0,
            occurrence_rows * chunk_size + occurrence_cols,
            occurrence_values,
        )
        chunk_scores = (activations * directional_derivatives).sum(dim=0)
        for feature_id, score in zip(
            chunk_ids_cpu.tolist(),
            chunk_scores.detach().cpu().tolist(),
        ):
            score_by_id[int(feature_id)] = float(score)

    ranked_ids = sorted(
        score_by_id,
        key=lambda feature_id: (-abs(score_by_id[feature_id]), feature_id),
    )
    if limit is not None:
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a non-negative integer or None")
        ranked_ids = ranked_ids[:limit]
    features: list[FeatureScore] = []
    active_indices_cpu = flat_indices_cpu[active_cpu]
    for feature_id in ranked_ids:
        feature_mask = active_indices_cpu == feature_id
        positions = tuple(
            int(position)
            for position in torch.unique(
                residue_rows_cpu[feature_mask],
                sorted=True,
            ).add(1).tolist()
        )
        features.append(
            FeatureScore(
                feature_id=int(feature_id),
                side=side,
                signed_score=score_by_id[int(feature_id)],
                activation_sum=float(
                    active_values_cpu[feature_mask]
                    .to(torch.float64)
                    .sum()
                    .item()
                ),
                active_positions=positions,
            )
        )
    return features


def _verify_trusted_audit(project: ProjectConfig) -> dict[str, object]:
    audit_path = project.paths.reports / "experiment1_audit.json"
    _verify_manifest_target(
        project.paths.project_root / "manifests/task03.sha256",
        audit_path,
        code_root=project.paths.code_root,
        label="Task 3 audit",
    )
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if not isinstance(audit, dict):
        raise RuntimeError("Task 3 audit report is not a mapping")
    if audit.get("status") != "PASS" or not audit.get("stop_gate", {}).get(
        "passed"
    ):
        raise RuntimeError("Task 3 audit is not PASS")
    return audit


def _load_validated_decoder(
    project: ProjectConfig,
) -> tuple[DecoderBridge, str]:
    payload = _load_trusted_decoder_payload(project)
    decoder = _decoder_from_payload(payload, project, torch.device("cpu"))
    return decoder, str(payload["provenance_fingerprint"])


def _load_validated_cache(
    project: ProjectConfig,
    *,
    gene_id: int,
    sequence: str,
) -> tuple[torch.Tensor, SparseProteinFeatures]:
    return _load_cache_pair(
        project,
        gene_id,
        sequence,
    )


def _feature_json(score: FeatureScore) -> dict[str, object]:
    return {
        "feature_id": score.feature_id,
        "side": score.side,
        "signed_score": score.signed_score,
        "activation_sum": score.activation_sum,
        "active_positions": list(score.active_positions),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), required=True)
    parser.add_argument("--row-id", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project = load_config(args.config)
    _verify_trusted_audit(project)
    checkpoint = (
        project.training.output_dir
        / "best_ckpts"
        / FORMAL_CHECKPOINT_NAME
    )

    coach_config = build_coach_config(project, [])
    dataset = build_dataset(coach_config, RunningStage.VALIDATING)
    if args.row_id < 0 or args.row_id >= len(dataset):
        raise IndexError("row-id is outside the validation split")
    fixed_batch = dataset.collate_fn([dataset[args.row_id]])
    gene_a = int(fixed_batch["g1_idx"].item())
    gene_b = int(fixed_batch["g2_idx"].item())
    sequences = _load_sequences(project.paths.sequences)
    try:
        sequence_a = _normalize_sequence(sequences[gene_a])
        sequence_b = _normalize_sequence(sequences[gene_b])
    except KeyError as error:
        raise RuntimeError(f"pair gene lacks configured sequence: {error.args[0]}")
    decoder, provenance_fingerprint = _load_validated_decoder(project)
    dense_a, sparse_a = _load_validated_cache(
        project,
        gene_id=gene_a,
        sequence=sequence_a,
    )
    dense_b, sparse_b = _load_validated_cache(
        project,
        gene_id=gene_b,
        sequence=sequence_b,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    adapter = FrozenSLAdapter.from_checkpoint(project, checkpoint)
    h_a = (
        dense_a
        .to(device=adapter.device, dtype=torch.float32)
        .detach()
        .requires_grad_(True)
    )
    h_b = (
        dense_b
        .to(device=adapter.device, dtype=torch.float32)
        .detach()
        .requires_grad_(True)
    )
    logit = adapter.pair_logit(h_a, h_b, fixed_batch)
    grad_a, grad_b = torch.autograd.grad(logit, (h_a, h_b))
    if not bool(torch.isfinite(logit)) or not bool(
        torch.isfinite(grad_a).all() and torch.isfinite(grad_b).all()
    ):
        raise RuntimeError("pair logit or residue gradient is non-finite")
    if any(
        parameter.grad is not None for parameter in adapter.module.parameters()
    ):
        raise RuntimeError("frozen downstream parameters accumulated gradients")

    scores_a = score_active_features(
        h_a,
        sparse_a,
        decoder,
        grad_a,
        side="A",
    )
    scores_b = score_active_features(
        h_b,
        sparse_b,
        decoder,
        grad_b,
        side="B",
    )
    peak_vram = (
        int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0
    )
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "sae_provenance_fingerprint": provenance_fingerprint,
        "split": args.split,
        "row_id": args.row_id,
        "gene_a": gene_a,
        "gene_b": gene_b,
        "logit": float(logit.detach().cpu()),
        "grad_a_finite": bool(torch.isfinite(grad_a).all()),
        "grad_b_finite": bool(torch.isfinite(grad_b).all()),
        "model_parameter_grad_count": sum(
            parameter.grad is not None
            for parameter in adapter.module.parameters()
        ),
        "no_dense_codebook_tensor": True,
        "score_chunk_features": SCORE_CHUNK_FEATURES,
        "top_features_a": [_feature_json(score) for score in scores_a],
        "top_features_b": [_feature_json(score) for score in scores_b],
        "peak_vram_bytes": peak_vram,
    }
    print(
        "PAIR_ATTRIBUTION_RESULT "
        + json.dumps(result, allow_nan=False, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
