"""Direct SAE feature interventions for frozen ProSL validation pairs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Literal, Mapping, Sequence

from pytorch_lightning.trainer.states import RunningStage
import torch

from coach_pl.dataset import build_dataset
from esmc_sae.artifacts import save_json
from esmc_sae.attribution import (
    FeatureScore,
    _load_validated_cache,
    _load_validated_decoder,
    _verify_trusted_audit,
    score_active_features,
)
from esmc_sae.audit import apply_feature_delta
from esmc_sae.config import load_config
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
)
from esmc_sae.train_dense import build_coach_config


@dataclass(frozen=True)
class InterventionResult:
    feature_id: int
    side: Literal["A", "B"]
    original_logit: float
    ablated_logit: float
    delta_logit: float


@dataclass(frozen=True)
class InteractionResult:
    feature_a: int
    feature_b: int
    delta_a: float
    delta_b: float
    delta_ab: float
    interaction_j: float


@dataclass(frozen=True)
class PairInterventionRecord:
    row_id: str
    gene_a: int
    gene_b: int
    original_logit: float
    probability: float
    singles: tuple[InterventionResult, ...]
    interactions: tuple[InteractionResult, ...]
    control_abs_deltas: tuple[float, ...]
    control_interaction_abs_js: tuple[float, ...] = ()
    true_label: int | None = None


def _require_intervention_inputs(
    sparse: SparseProteinFeatures | None,
    decoder: DecoderBridge | None,
) -> tuple[SparseProteinFeatures, DecoderBridge]:
    if sparse is None or decoder is None:
        raise ValueError(
            "direct intervention requires validated sparse features and decoder"
        )
    return sparse, decoder


def ablate_feature(
    h: torch.Tensor,
    sparse: SparseProteinFeatures,
    decoder: DecoderBridge,
    feature_id: int,
) -> torch.Tensor:
    """Remove one active feature with Task 3's original-coordinate delta."""

    if type(feature_id) is not int:
        raise ValueError("feature_id must be an integer")
    active_rows = (sparse.indices == feature_id).any(dim=1)
    if not bool(active_rows.any()):
        return h
    result = apply_feature_delta(
        h,
        sparse,
        decoder,
        frozenset({feature_id}),
    )
    inactive_rows = ~active_rows.to(result.device)
    if bool(inactive_rows.any()):
        result[inactive_rows] = h[inactive_rows].to(result.device)
    return result


def _attempt_mean_pairs(
    adapter: FrozenSLAdapter,
    fixed_batch: Mapping[str, object],
    mean_a: torch.Tensor,
    mean_b: torch.Tensor,
) -> torch.Tensor | None:
    """Return None for CUDA OOM after the failed attempt frame is released."""
    try:
        return adapter.pair_logits_from_means(mean_a, mean_b, fixed_batch)
    except torch.cuda.OutOfMemoryError:
        return None


def _evaluate_mean_pairs(
    adapter: FrozenSLAdapter,
    fixed_batch: Mapping[str, object],
    mean_a: torch.Tensor,
    mean_b: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Try one vector batch; retry only that failed batch at size one on OOM."""

    logits = _attempt_mean_pairs(adapter, fixed_batch, mean_a, mean_b)
    if logits is not None:
        return logits, False
    del logits
    torch.cuda.empty_cache()
    outputs = []
    for index in range(mean_a.shape[0]):
        outputs.append(
            adapter.pair_logits_from_means(
                mean_a[index : index + 1],
                mean_b[index : index + 1],
                fixed_batch,
            )
        )
    return torch.cat(outputs, dim=0), True


def single_effect(
    adapter: FrozenSLAdapter,
    fixed_batch: dict[str, torch.Tensor],
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    side: Literal["A", "B"],
    feature_id: int,
    sparse: SparseProteinFeatures | None = None,
    decoder: DecoderBridge | None = None,
) -> InterventionResult:
    """Measure one feature's exact original-minus-ablated pair-logit effect."""

    if side not in ("A", "B"):
        raise ValueError("side must be 'A' or 'B'")
    sparse, decoder = _require_intervention_inputs(sparse, decoder)
    changed = ablate_feature(
        h_a if side == "A" else h_b,
        sparse,
        decoder,
        feature_id,
    )
    original_a = h_a.mean(dim=0)
    original_b = h_b.mean(dim=0)
    mean_a = torch.stack(
        [original_a, changed.mean(dim=0) if side == "A" else original_a]
    )
    mean_b = torch.stack(
        [original_b, changed.mean(dim=0) if side == "B" else original_b]
    )
    logits, _ = _evaluate_mean_pairs(adapter, fixed_batch, mean_a, mean_b)
    original_logit, ablated_logit = (
        float(value) for value in logits.detach().cpu().tolist()
    )
    return InterventionResult(
        feature_id=feature_id,
        side=side,
        original_logit=original_logit,
        ablated_logit=ablated_logit,
        delta_logit=original_logit - ablated_logit,
    )


def double_effect(
    adapter: FrozenSLAdapter,
    fixed_batch: dict[str, torch.Tensor],
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    feature_a: int,
    feature_b: int,
    sparse_a: SparseProteinFeatures | None = None,
    sparse_b: SparseProteinFeatures | None = None,
    decoder: DecoderBridge | None = None,
) -> InteractionResult:
    """Measure cross-side single, double, and inclusion-exclusion effects."""

    sparse_a, decoder = _require_intervention_inputs(sparse_a, decoder)
    sparse_b, _ = _require_intervention_inputs(sparse_b, decoder)
    changed_a = ablate_feature(h_a, sparse_a, decoder, feature_a)
    changed_b = ablate_feature(h_b, sparse_b, decoder, feature_b)
    original_a = h_a.mean(dim=0)
    original_b = h_b.mean(dim=0)
    ablated_a = changed_a.mean(dim=0)
    ablated_b = changed_b.mean(dim=0)
    mean_a = torch.stack([original_a, ablated_a, original_a, ablated_a])
    mean_b = torch.stack([original_b, original_b, ablated_b, ablated_b])
    logits, _ = _evaluate_mean_pairs(adapter, fixed_batch, mean_a, mean_b)
    z, z_a, z_b, z_ab = (
        float(value) for value in logits.detach().cpu().tolist()
    )
    delta_a = z - z_a
    delta_b = z - z_b
    delta_ab = z - z_ab
    return InteractionResult(
        feature_a=feature_a,
        feature_b=feature_b,
        delta_a=delta_a,
        delta_b=delta_b,
        delta_ab=delta_ab,
        interaction_j=delta_ab - delta_a - delta_b,
    )


def matched_controls(
    scores: Sequence[FeatureScore],
    side: Literal["A", "B"],
    excluded: frozenset[int],
    n: int = 5,
    seed: int = 42,
) -> list[int]:
    """Select same-side controls nearest to excluded candidate summaries."""

    if side not in ("A", "B"):
        raise ValueError("side must be 'A' or 'B'")
    if type(n) is not int or n < 0:
        raise ValueError("n must be a non-negative integer")
    side_scores = [score for score in scores if score.side == side]
    anchors = [score for score in side_scores if score.feature_id in excluded]
    if n == 0:
        return []
    if not anchors:
        raise ValueError("excluded must identify a scored candidate on the side")
    candidates = [
        score for score in side_scores if score.feature_id not in excluded
    ]

    def summary(score: FeatureScore) -> tuple[float, int]:
        magnitude = max(abs(float(score.activation_sum)), torch.finfo(torch.float64).tiny)
        return math.log(magnitude), len(score.active_positions)

    anchor_summaries = [summary(anchor) for anchor in anchors]
    tie_order = sorted(score.feature_id for score in candidates)
    random.Random(seed).shuffle(tie_order)
    tie_rank = {
        feature_id: index for index, feature_id in enumerate(tie_order)
    }

    def distance(score: FeatureScore) -> tuple[float, int]:
        log_activation, active_count = summary(score)
        nearest = min(
            (
                (log_activation - anchor_log) ** 2
                + float(active_count - anchor_count) ** 2
            )
            for anchor_log, anchor_count in anchor_summaries
        )
        return nearest, tie_rank[score.feature_id]

    return [
        score.feature_id
        for score in sorted(candidates, key=distance)[:n]
    ]


def _decision_margins(
    original_logit: float,
    ablated_logit: float,
) -> tuple[float, float, float]:
    decision_sign = 1.0 if original_logit >= 0.0 else -1.0
    original_margin = decision_sign * original_logit
    ablated_margin = decision_sign * ablated_logit
    return (
        original_margin,
        ablated_margin,
        ablated_margin - original_margin,
    )


def _score_payload(
    score: FeatureScore,
    result: InterventionResult,
    *,
    is_control: bool,
) -> dict[str, object]:
    original_margin, ablated_margin, margin_change = _decision_margins(
        result.original_logit,
        result.ablated_logit,
    )
    return {
        **asdict(result),
        "active_positions": list(score.active_positions),
        "activation_sum": score.activation_sum,
        "signed_score": score.signed_score,
        "original_decision_margin": original_margin,
        "ablated_decision_margin": ablated_margin,
        "decision_margin_change": margin_change,
        "is_control": is_control,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), required=True)
    parser.add_argument("--limit", type=int, required=True)
    return parser


def _run_pair(
    *,
    project,
    dataset,
    row_index: int,
    sequences: Mapping[int, str],
    decoder: DecoderBridge,
    provenance_fingerprint: str,
    adapter: FrozenSLAdapter,
    checkpoint: Path,
    split: str = "val",
    row_id: str | None = None,
    include_control_interactions: bool = False,
) -> dict[str, object]:
    fixed_batch = dataset.collate_fn([dataset[row_index]])
    gene_a = int(fixed_batch["g1_idx"].item())
    gene_b = int(fixed_batch["g2_idx"].item())
    try:
        sequence_a = _normalize_sequence(sequences[gene_a])
        sequence_b = _normalize_sequence(sequences[gene_b])
    except KeyError as error:
        raise RuntimeError(
            f"pair gene lacks configured sequence: {error.args[0]}"
        ) from error
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
    h_a = dense_a.to(adapter.device, dtype=torch.float32).requires_grad_(True)
    h_b = dense_b.to(adapter.device, dtype=torch.float32).requires_grad_(True)
    original = adapter.pair_logit(h_a, h_b, fixed_batch)
    grad_a, grad_b = torch.autograd.grad(original, (h_a, h_b))
    if not bool(
        torch.isfinite(original)
        and torch.isfinite(grad_a).all()
        and torch.isfinite(grad_b).all()
    ):
        raise RuntimeError("pair logit or residue gradient is non-finite")

    scores_a = score_active_features(
        h_a, sparse_a, decoder, grad_a, "A", limit=None
    )
    scores_b = score_active_features(
        h_b, sparse_b, decoder, grad_b, "B", limit=None
    )
    ceiling = project.explanation.candidate_ceiling
    candidate_a = scores_a[:ceiling]
    candidate_b = scores_b[:ceiling]
    excluded = frozenset(
        score.feature_id for score in (*candidate_a, *candidate_b)
    )
    controls_a = matched_controls(
        scores_a,
        "A",
        excluded,
        project.explanation.matched_controls_per_side,
        project.seed,
    )
    controls_b = matched_controls(
        scores_b,
        "B",
        excluded,
        project.explanation.matched_controls_per_side,
        project.seed,
    )
    score_a = {score.feature_id: score for score in scores_a}
    score_b = {score.feature_id: score for score in scores_b}

    evaluated = [
        *(("A", score, False) for score in candidate_a),
        *(("B", score, False) for score in candidate_b),
        *(("A", score_a[feature_id], True) for feature_id in controls_a),
        *(("B", score_b[feature_id], True) for feature_id in controls_b),
    ]
    original_a = h_a.detach().mean(dim=0)
    original_b = h_b.detach().mean(dim=0)
    means_a = []
    means_b = []
    for side, score, _ in evaluated:
        if side == "A":
            changed = ablate_feature(
                h_a.detach(), sparse_a, decoder, score.feature_id
            )
            means_a.append(changed.mean(dim=0))
            means_b.append(original_b)
        else:
            changed = ablate_feature(
                h_b.detach(), sparse_b, decoder, score.feature_id
            )
            means_a.append(original_a)
            means_b.append(changed.mean(dim=0))
    single_logits, single_oom = _evaluate_mean_pairs(
        adapter,
        fixed_batch,
        torch.stack(means_a),
        torch.stack(means_b),
    )
    original_float = float(original.detach().cpu())
    single_payloads = []
    control_payloads = []
    ablated_a = {}
    ablated_b = {}
    control_ablated_a = {}
    control_ablated_b = {}
    for (side, score, is_control), changed_logit in zip(
        evaluated, single_logits.detach().cpu().tolist()
    ):
        result = InterventionResult(
            feature_id=score.feature_id,
            side=side,
            original_logit=original_float,
            ablated_logit=float(changed_logit),
            delta_logit=original_float - float(changed_logit),
        )
        payload = _score_payload(score, result, is_control=is_control)
        (control_payloads if is_control else single_payloads).append(payload)
        if not is_control and side == "A":
            ablated_a[score.feature_id] = ablate_feature(
                h_a.detach(), sparse_a, decoder, score.feature_id
            ).mean(dim=0)
        elif not is_control:
            ablated_b[score.feature_id] = ablate_feature(
                h_b.detach(), sparse_b, decoder, score.feature_id
            ).mean(dim=0)
        elif side == "A":
            control_ablated_a[score.feature_id] = ablate_feature(
                h_a.detach(), sparse_a, decoder, score.feature_id
            ).mean(dim=0)
        else:
            control_ablated_b[score.feature_id] = ablate_feature(
                h_b.detach(), sparse_b, decoder, score.feature_id
            ).mean(dim=0)

    interaction_pairs = [
        (score_a_item.feature_id, score_b_item.feature_id)
        for score_a_item in candidate_a
        for score_b_item in candidate_b
    ]
    double_oom = False
    interactions = []
    if interaction_pairs:
        double_logits, double_oom = _evaluate_mean_pairs(
            adapter,
            fixed_batch,
            torch.stack([ablated_a[a] for a, _ in interaction_pairs]),
            torch.stack([ablated_b[b] for _, b in interaction_pairs]),
        )
        single_delta = {
            (payload["side"], payload["feature_id"]): payload["delta_logit"]
            for payload in single_payloads
        }
        for (feature_a, feature_b), ablated_logit in zip(
            interaction_pairs,
            double_logits.detach().cpu().tolist(),
        ):
            delta_a = float(single_delta[("A", feature_a)])
            delta_b = float(single_delta[("B", feature_b)])
            delta_ab = original_float - float(ablated_logit)
            interactions.append(
                asdict(
                    InteractionResult(
                        feature_a=feature_a,
                        feature_b=feature_b,
                        delta_a=delta_a,
                        delta_b=delta_b,
                        delta_ab=delta_ab,
                        interaction_j=delta_ab - delta_a - delta_b,
                    )
                )
            )

    control_interactions = []
    control_pairs = list(zip(controls_a, controls_b))
    if include_control_interactions and control_pairs:
        control_double_logits, control_double_oom = _evaluate_mean_pairs(
            adapter,
            fixed_batch,
            torch.stack([control_ablated_a[a] for a, _ in control_pairs]),
            torch.stack([control_ablated_b[b] for _, b in control_pairs]),
        )
        double_oom = double_oom or control_double_oom
        control_delta = {
            (payload["side"], payload["feature_id"]): payload["delta_logit"]
            for payload in control_payloads
        }
        for (feature_a, feature_b), ablated_logit in zip(
            control_pairs,
            control_double_logits.detach().cpu().tolist(),
        ):
            delta_a = float(control_delta[("A", feature_a)])
            delta_b = float(control_delta[("B", feature_b)])
            delta_ab = original_float - float(ablated_logit)
            control_interactions.append(
                asdict(
                    InteractionResult(
                        feature_a=feature_a,
                        feature_b=feature_b,
                        delta_a=delta_a,
                        delta_b=delta_b,
                        delta_ab=delta_ab,
                        interaction_j=delta_ab - delta_a - delta_b,
                    )
                )
            )

    return {
        "schema_version": 1,
        "split": split,
        "row_id": row_id if row_id is not None else str(row_index),
        "gene_a": gene_a,
        "gene_b": gene_b,
        "original_logit": original_float,
        "probability": float(torch.sigmoid(original.detach()).cpu()),
        "singles": single_payloads,
        "controls": control_payloads,
        "control_abs_deltas": [
            abs(float(payload["delta_logit"])) for payload in control_payloads
        ],
        "interactions": interactions,
        "control_interactions": control_interactions,
        "batching": {
            "single_variant_count": len(evaluated),
            "double_variant_count": (
                len(interaction_pairs)
                + (len(control_pairs) if include_control_interactions else 0)
            ),
            "oom_fallback": single_oom or double_oom,
            "fallback_batch_size": (
                1 if single_oom or double_oom else None
            ),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "provenance": {
            "sae_fingerprint": provenance_fingerprint,
            "decoder_bridge": str(project.paths.decoder_bridge),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.limit < 1:
        raise ValueError("limit must be at least one")
    project = load_config(args.config)
    _verify_trusted_audit(project)
    checkpoint = (
        project.training.output_dir
        / "best_ckpts"
        / FORMAL_CHECKPOINT_NAME
    )
    coach_config = build_coach_config(project, [])
    dataset = build_dataset(coach_config, RunningStage.VALIDATING)
    decoder, provenance_fingerprint = _load_validated_decoder(project)
    sequences = _load_sequences(project.paths.sequences)
    adapter = FrozenSLAdapter.from_checkpoint(project, checkpoint)
    output_dir = project.paths.reports / "interventions" / args.split
    for row_index in range(min(args.limit, len(dataset))):
        payload = _run_pair(
            project=project,
            dataset=dataset,
            row_index=row_index,
            sequences=sequences,
            decoder=decoder,
            provenance_fingerprint=provenance_fingerprint,
            adapter=adapter,
            checkpoint=checkpoint,
        )
        output_path = output_dir / f"row_{row_index:06d}.json"
        save_json(output_path, payload)
        print(
            "PAIR_INTERVENTION_RESULT "
            + json.dumps(payload, allow_nan=False, sort_keys=True),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
