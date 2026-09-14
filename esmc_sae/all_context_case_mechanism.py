"""Below-threshold illustrative SAE intervention and LLM mechanism report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from pytorch_lightning.trainer.states import RunningStage
import torch

from coach_pl.dataset import build_dataset
from coach_pl.module import build_module
from esmc_sae.all_context_predict import (
    TARGET_POOL_ROOT,
    _checkpoint,
)
from esmc_sae.all_context_train import (
    DATA_ROOT,
    MODEL_ROOT,
    build_all_context_coach_config,
)
from esmc_sae.attribution import (
    _load_validated_cache,
    _load_validated_decoder,
    _verify_trusted_audit,
    score_active_features,
)
from esmc_sae.calibrate import CLASSIFICATION_TAU
from esmc_sae.config import load_config
from esmc_sae.extract import _load_sequences, _normalize_sequence
from esmc_sae.intervention import (
    InterventionResult,
    _evaluate_mean_pairs,
    _score_payload,
    ablate_feature,
    matched_controls,
)
from esmc_sae.manifest import file_sha256
from esmc_sae.metadata import get_feature_metadata
from esmc_sae.sl_adapter import FrozenSLAdapter
import esmc_sae.all_context_dataset  # noqa: F401
import pfsl.model  # noqa: F401
import pfsl.module  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
MVP_CONFIG = ROOT / "esmc_sae/configs/mvp_a549_c1_fold0.yaml"
CASE_ID = "ovcar8_brca2_parp1_all_context"
OUTPUT_ROOT = (
    ROOT
    / "esmc_sae/case_study/ovcar8_brca2_parp1_all_context"
    / "illustrative_below_threshold"
)
OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.6"
@dataclass(frozen=True)
class CaseIdentity:
    case_id: str
    context: str
    gene_a: int
    gene_b: int
    gene_a_name: str
    gene_b_name: str
    row_id: str

    @property
    def gene_by_side(self) -> dict[str, str]:
        return {"A": self.gene_a_name, "B": self.gene_b_name}


DEFAULT_IDENTITY = CaseIdentity(
    case_id=CASE_ID,
    context="OVCAR8",
    gene_a=675,
    gene_b=142,
    gene_a_name="BRCA2",
    gene_b_name="PARP1",
    row_id="OVCAR8_BRCA2_PARP1_LOCKED",
)


def build_mechanism_schema(
    identity: CaseIdentity = DEFAULT_IDENTITY,
) -> dict[str, object]:
    return {
    "type": "object",
    "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "case_id": {"type": "string", "enum": [identity.case_id]},
        "feature_interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature_id": {"type": "integer"},
                    "side": {
                        "type": "string",
                        "enum": [identity.gene_a_name, identity.gene_b_name],
                    },
                    "semantic_role": {"type": "string"},
                    "link_to_hypothesis": {"type": "string"},
                },
                "required": [
                    "feature_id",
                    "side",
                    "semantic_role",
                    "link_to_hypothesis",
                ],
                "additionalProperties": False,
            },
        },
        "mechanism_hypothesis": {"type": "string"},
        "causal_chain": {
            "type": "array",
            "items": {"type": "string"},
        },
        "alternative_interpretations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "testable_predictions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "schema_version",
        "case_id",
        "feature_interpretations",
        "mechanism_hypothesis",
        "causal_chain",
        "alternative_interpretations",
        "limitations",
        "testable_predictions",
    ],
    "additionalProperties": False,
    }


MECHANISM_SCHEMA = build_mechanism_schema()


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return value


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def select_positive_support_features(
    interventions: Mapping[str, object],
    *,
    quantile: float = 0.95,
) -> dict[str, object]:
    """Retain positive logit-support candidates above same-side controls."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    controls = _sequence(interventions.get("controls"), "controls")
    singles = _sequence(interventions.get("singles"), "singles")
    control_values: dict[str, list[float]] = {"A": [], "B": []}
    for item in controls:
        if not isinstance(item, Mapping) or item.get("side") not in ("A", "B"):
            raise ValueError("control entry is malformed")
        control_values[str(item["side"])].append(
            abs(_finite(item.get("delta_logit"), "control delta_logit"))
        )
    if any(not values for values in control_values.values()):
        raise ValueError("each side requires at least one matched control")
    cutoffs = {
        side: float(np.quantile(values, quantile))
        for side, values in control_values.items()
    }
    selected: list[dict[str, object]] = []
    for item in singles:
        if not isinstance(item, Mapping) or item.get("side") not in ("A", "B"):
            raise ValueError("single intervention entry is malformed")
        side = str(item["side"])
        delta = _finite(item.get("delta_logit"), "single delta_logit")
        if delta > cutoffs[side]:
            selected.append(
                {
                    **dict(item),
                    "control_cutoff": cutoffs[side],
                    "passed_control": True,
                }
            )
    return {
        "control_quantile": quantile,
        "control_cutoffs": cutoffs,
        "selected_features": selected,
    }


def _position_summary(value: object) -> dict[str, object]:
    positions = [
        int(item) for item in _sequence(value, "active_positions")
    ]
    if not positions:
        raise ValueError("active_positions must not be empty")
    examples = (
        positions
        if len(positions) <= 10
        else [*positions[:5], *positions[-5:]]
    )
    return {
        "active_position_count": len(positions),
        "active_position_range": [min(positions), max(positions)],
        "example_positions": examples,
    }


def _prompt_feature_rows(
    feature_semantics: Sequence[Mapping[str, object]],
    *,
    identity: CaseIdentity = DEFAULT_IDENTITY,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in feature_semantics:
        side = feature.get("side")
        if side not in ("A", "B"):
            raise ValueError("feature side must be A or B")
        position_summary = _position_summary(feature.get("active_positions"))
        rows.append(
            {
                "feature_id": int(feature["feature_id"]),
                "gene": identity.gene_by_side[str(side)],
                "delta_logit": _finite(
                    feature.get("delta_logit"), "feature delta_logit"
                ),
                "same_side_control_cutoff": _finite(
                    feature.get("control_cutoff"), "feature control cutoff"
                ),
                "passed_control": bool(feature.get("passed_control")),
                **position_summary,
                "official_sae_description": _nonempty(
                    feature.get("description"), "feature description"
                ),
            }
        )
    if not rows:
        raise ValueError("at least one feature semantic is required")
    return rows


def build_llm_prompt(
    *,
    case_probability: float,
    classification_threshold: float,
    feature_semantics: Sequence[Mapping[str, object]],
    identity: CaseIdentity = DEFAULT_IDENTITY,
) -> str:
    """Build the evidence-bound prompt saved alongside the API response."""

    probability = _finite(case_probability, "case_probability")
    threshold = _finite(classification_threshold, "classification_threshold")
    rows = _prompt_feature_rows(feature_semantics, identity=identity)
    positive = probability >= threshold
    supplied = {
        "case_id": identity.case_id,
        "cell_context": identity.context,
        "pair": [identity.gene_a_name, identity.gene_b_name],
        "model_scope": f"{identity.context}-conditioned ESMC branch",
        "probability": probability,
        "classification_threshold": threshold,
        "below_threshold": not positive,
        "features": rows,
    }
    prediction_boundary = (
        "The overall model predicts this pair as SL: its probability "
        f"is {probability:.7g}, at or above the fixed threshold "
        f"{threshold:.7g}. Explain the listed positive-support components "
        "as an exploratory local ESMC-branch dependency hypothesis."
        if positive
        else
        "The overall model does not predict this pair as SL: its probability "
        f"is {probability:.7g}, below the fixed threshold {threshold:.7g}. "
        "Explain only the listed positive-support components; do not reinterpret "
        "the overall prediction as positive."
    )
    return "\n".join(
        [
            "Generate one exploratory mechanism hypothesis from the supplied "
            "model-internal information.",
            "",
            prediction_boundary,
            "",
            "Use only the supplied gene identities, cell context, intervention "
            "measurements, residue positions, and official SAE descriptions. "
            "Do not introduce external biological facts, papers, databases, "
            "citations, or unstated gene functions.",
            "",
            "Treat an official SAE description as a tentative semantic label, not "
            "as independent biological evidence. Distinguish direct model evidence "
            "from semantic interpretation and from speculation. Use cautious terms "
            "such as may, could, and hypothesis. Include alternative interpretations, "
            "the fixed-threshold boundary, the absence of external evidence, and "
            "testable predictions. Never use words such as proven or established "
            "for the proposed mechanism.",
            "",
            "Return only the JSON object required by the response schema.",
            "",
            "SUPPLIED_INPUT:",
            json.dumps(
                supplied,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            ),
        ]
    )


def build_openai_request(
    *,
    prompt: str,
    model: str = DEFAULT_MODEL,
    identity: CaseIdentity = DEFAULT_IDENTITY,
) -> dict[str, object]:
    """Construct a non-stored Responses API request with strict JSON output."""

    return {
        "model": _nonempty(model, "model"),
        "store": False,
        "max_output_tokens": 1800,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are a careful mechanistic-interpretability analyst. "
                    "Follow the evidence boundary exactly and return schema-valid JSON."
                ),
            },
            {"role": "user", "content": _nonempty(prompt, "prompt")},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "sae_mechanism_hypothesis",
                "strict": True,
                "schema": build_mechanism_schema(identity),
            }
        },
    }


def validate_llm_mechanism(
    payload: Mapping[str, object],
    *,
    allowed_features: Sequence[Mapping[str, object]],
    identity: CaseIdentity = DEFAULT_IDENTITY,
) -> None:
    """Reject an LLM result that escapes the supplied feature boundary."""

    required = set(build_mechanism_schema(identity)["required"])
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("LLM mechanism has an unexpected root schema")
    if (
        payload.get("schema_version") != 1
        or payload.get("case_id") != identity.case_id
    ):
        raise ValueError("LLM mechanism identity is invalid")
    allowed = {
        (identity.gene_by_side[str(item["side"])], int(item["feature_id"]))
        for item in allowed_features
    }
    if not allowed:
        raise ValueError("allowed feature set must not be empty")
    interpretations = _sequence(
        payload.get("feature_interpretations"), "feature_interpretations"
    )
    seen: set[tuple[str, int]] = set()
    for item in interpretations:
        if not isinstance(item, Mapping) or set(item) != {
            "feature_id",
            "side",
            "semantic_role",
            "link_to_hypothesis",
        }:
            raise ValueError("feature interpretation is malformed")
        feature_id = item.get("feature_id")
        side = item.get("side")
        key = (str(side), feature_id)
        if type(feature_id) is not int or key not in allowed:
            raise ValueError("LLM mechanism references an unknown feature")
        if key in seen:
            raise ValueError("LLM mechanism repeats a feature")
        seen.add(key)
        _nonempty(item.get("semantic_role"), "semantic_role")
        _nonempty(item.get("link_to_hypothesis"), "link_to_hypothesis")
    if not interpretations:
        raise ValueError("LLM mechanism must interpret at least one feature")

    _nonempty(payload.get("mechanism_hypothesis"), "mechanism_hypothesis")
    for field in (
        "causal_chain",
        "alternative_interpretations",
        "limitations",
        "testable_predictions",
    ):
        values = _sequence(payload.get(field), field)
        if not values:
            raise ValueError(f"{field} must not be empty")
        for value in values:
            _nonempty(value, field)
    limitations = " ".join(str(value) for value in payload["limitations"]).lower()
    if "threshold" not in limitations:
        raise ValueError("limitations must retain the below-threshold boundary")
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    if any(token in serialized for token in ("http://", "https://", "pmid", "doi:")):
        raise ValueError("LLM mechanism must not add external evidence or citations")


def render_illustrative_report(
    payload: Mapping[str, object],
    *,
    case_probability: float,
    classification_threshold: float,
    feature_semantics: Sequence[Mapping[str, object]],
    api_metadata: Mapping[str, object],
    identity: CaseIdentity = DEFAULT_IDENTITY,
) -> str:
    """Render validated LLM fields under an explicit prediction boundary."""

    validate_llm_mechanism(
        payload,
        allowed_features=feature_semantics,
        identity=identity,
    )
    probability = _finite(case_probability, "case_probability")
    threshold = _finite(classification_threshold, "classification_threshold")
    positive = probability >= threshold
    pair_name = f"{identity.gene_a_name}–{identity.gene_b_name}"
    feature_by_key = {
        (identity.gene_by_side[str(item["side"])], int(item["feature_id"])): item
        for item in feature_semantics
    }
    title = (
        f"# {pair_name} EXPLORATORY ESMC-DEPENDENCY HYPOTHESIS"
        if positive
        else f"# {pair_name} BELOW-THRESHOLD ILLUSTRATIVE HYPOTHESIS"
    )
    decision = (
        f"predicts {pair_name} as SL."
        if positive
        else f"does not predict {pair_name} as SL."
    )
    scope = (
        "This document explains local positive-support ESMC components of "
        "that prediction. It is an exploratory single-model ESMC-branch "
        "dependency hypothesis, not evidence that the proposed biological "
        "mechanism is true."
        if positive
        else
        "This document explains only local positive-support ESMC components. "
        "It is not a model-supported SL explanation and is not evidence that "
        "the proposed biological mechanism is true."
    )
    lines = [
        title,
        "",
        "## Interpretation boundary",
        "",
        (
            f"The model probability is `{probability:.7g}` at a fixed "
            f"classification threshold of `{threshold:.7g}`. The model therefore "
            + decision
        ),
        scope,
        (
            "No external biological evidence was provided to the LLM; the hypothesis "
            "uses only intervention measurements and official SAE descriptions."
        ),
        "",
        "## Model-internal feature inputs",
        "",
        "| Gene | Feature | Delta logit | Control cutoff | Passed | Positions | Official SAE description |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for feature in feature_semantics:
        position_summary = _position_summary(feature.get("active_positions"))
        start, stop = position_summary["active_position_range"]
        examples = ", ".join(
            str(value) for value in position_summary["example_positions"]
        )
        positions = (
            f"{position_summary['active_position_count']} positions; "
            f"range {start}–{stop}; examples {examples}"
        )
        description = str(feature["description"]).replace("|", "\\|")
        lines.append(
            f"| {identity.gene_by_side[str(feature['side'])]} | {int(feature['feature_id'])} "
            f"| {float(feature['delta_logit']):.7g} "
            f"| {float(feature['control_cutoff']):.7g} "
            f"| {str(bool(feature['passed_control'])).lower()} "
            f"| {positions} | {description} |"
        )
    lines.extend(
        [
            "",
            "## LLM feature interpretations",
            "",
        ]
    )
    for item in payload["feature_interpretations"]:
        feature = feature_by_key[(str(item["side"]), int(item["feature_id"]))]
        lines.extend(
            [
                f"### {item['side']} feature {item['feature_id']}",
                "",
                f"- Direct intervention status: {'passed' if feature['passed_control'] else 'did not pass'} the case-matched control cutoff.",
                f"- Semantic role: {item['semantic_role']}",
                f"- Hypothesis link: {item['link_to_hypothesis']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Illustrative mechanism hypothesis",
            "",
            str(payload["mechanism_hypothesis"]),
            "",
            "### Proposed causal chain",
            "",
            *[f"{index}. {value}" for index, value in enumerate(payload["causal_chain"], 1)],
            "",
            "### Alternative interpretations",
            "",
            *[f"- {value}" for value in payload["alternative_interpretations"]],
            "",
            "### Limitations",
            "",
            *[f"- {value}" for value in payload["limitations"]],
            "",
            "### Testable predictions",
            "",
            *[f"- {value}" for value in payload["testable_predictions"]],
            "",
            "## Generation provenance",
            "",
            f"- model: {api_metadata.get('model', 'unknown')}",
            f"- response_id: {api_metadata.get('response_id', 'unknown')}",
            f"- transport: {api_metadata.get('transport', 'responses_api')}",
            (
                "- API storage requested: "
                + str(api_metadata.get("store_requested", "not applicable")).lower()
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _load_case_adapter(
    *,
    device: torch.device,
    data_root: Path = DATA_ROOT,
    model_root: Path = MODEL_ROOT,
    target_pool_root: Path = TARGET_POOL_ROOT,
) -> tuple[FrozenSLAdapter, object, dict[str, object], Path, str]:
    cfg = build_all_context_coach_config(
        data_root=data_root,
        output_root=model_root,
    )
    cfg.DATASET.TEST_FILE = str(Path(data_root) / "locked_case.csv")
    cfg.DATASET.POOLED_ESM_ROOT = str(target_pool_root)
    checkpoint, checkpoint_sha256 = _checkpoint(Path(model_root))
    module = build_module(cfg)
    checkpoint_payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("checkpoint lacks state_dict")
    module.load_state_dict(state_dict, strict=True)
    module.eval().requires_grad_(False).to(device)
    dataset = build_dataset(cfg, RunningStage.TESTING)
    if len(dataset) != 1:
        raise RuntimeError("locked case dataset must contain exactly one row")
    fixed_batch = dataset.collate_fn([dataset[0]])
    return (
        FrozenSLAdapter(module, device=device),
        dataset,
        fixed_batch,
        checkpoint,
        checkpoint_sha256,
    )


def run_case_interventions(
    *,
    device: torch.device | str = "cpu",
    controls_per_side: int = 50,
    identity: CaseIdentity = DEFAULT_IDENTITY,
    data_root: Path = DATA_ROOT,
    model_root: Path = MODEL_ROOT,
    target_pool_root: Path = TARGET_POOL_ROOT,
) -> dict[str, object]:
    """Run singles-only intervention analysis for one locked case."""

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    project = load_config(MVP_CONFIG)
    _verify_trusted_audit(project)
    decoder, provenance_fingerprint = _load_validated_decoder(project)
    sequences = _load_sequences(project.paths.sequences)
    adapter, _, fixed_batch, checkpoint, checkpoint_sha256 = _load_case_adapter(
        device=device,
        data_root=Path(data_root),
        model_root=Path(model_root),
        target_pool_root=Path(target_pool_root),
    )
    gene_a = int(fixed_batch["g1_idx"].item())
    gene_b = int(fixed_batch["g2_idx"].item())
    if (gene_a, gene_b) != (identity.gene_a, identity.gene_b):
        raise RuntimeError("locked case identity or orientation changed")
    dense_a, sparse_a = _load_validated_cache(
        project,
        gene_id=gene_a,
        sequence=_normalize_sequence(sequences[gene_a]),
    )
    dense_b, sparse_b = _load_validated_cache(
        project,
        gene_id=gene_b,
        sequence=_normalize_sequence(sequences[gene_b]),
    )
    h_a = dense_a.to(device, dtype=torch.float32).requires_grad_(True)
    h_b = dense_b.to(device, dtype=torch.float32).requires_grad_(True)
    original = adapter.pair_logit(h_a, h_b, fixed_batch)
    grad_a, grad_b = torch.autograd.grad(original, (h_a, h_b))
    if not bool(
        torch.isfinite(original)
        and torch.isfinite(grad_a).all()
        and torch.isfinite(grad_b).all()
    ):
        raise RuntimeError("case logit or residue gradient is non-finite")
    scores_a = score_active_features(
        h_a, sparse_a, decoder, grad_a, "A", limit=None
    )
    scores_b = score_active_features(
        h_b, sparse_b, decoder, grad_b, "B", limit=None
    )
    candidates_a = scores_a[:10]
    candidates_b = scores_b[:10]
    excluded = frozenset(
        score.feature_id for score in (*candidates_a, *candidates_b)
    )
    controls_a = matched_controls(
        scores_a, "A", excluded, controls_per_side, seed=42
    )
    controls_b = matched_controls(
        scores_b, "B", excluded, controls_per_side, seed=42
    )
    if len(controls_a) != controls_per_side or len(controls_b) != controls_per_side:
        raise RuntimeError("insufficient active features for matched controls")
    score_by_side = {
        "A": {score.feature_id: score for score in scores_a},
        "B": {score.feature_id: score for score in scores_b},
    }
    evaluated = [
        *(("A", score, False) for score in candidates_a),
        *(("B", score, False) for score in candidates_b),
        *(("A", score_by_side["A"][feature_id], True) for feature_id in controls_a),
        *(("B", score_by_side["B"][feature_id], True) for feature_id in controls_b),
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
    logits, oom_fallback = _evaluate_mean_pairs(
        adapter,
        fixed_batch,
        torch.stack(means_a),
        torch.stack(means_b),
    )
    original_float = float(original.detach().cpu())
    singles: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    for (side, score, is_control), ablated_logit in zip(
        evaluated,
        logits.detach().cpu().tolist(),
    ):
        result = InterventionResult(
            feature_id=score.feature_id,
            side=side,
            original_logit=original_float,
            ablated_logit=float(ablated_logit),
            delta_logit=original_float - float(ablated_logit),
        )
        item = _score_payload(score, result, is_control=is_control)
        (controls if is_control else singles).append(item)
    result = {
        "schema_version": 1,
        "case_id": identity.case_id,
        "row_id": identity.row_id,
        "context": identity.context,
        "gene_a": gene_a,
        "gene_b": gene_b,
        "gene_a_name": identity.gene_a_name,
        "gene_b_name": identity.gene_b_name,
        "original_logit": original_float,
        "probability": float(torch.sigmoid(original.detach()).cpu()),
        "classification_threshold": CLASSIFICATION_TAU,
        "prediction_gate_bypassed": (
            float(torch.sigmoid(original.detach()).cpu())
            < CLASSIFICATION_TAU
        ),
        "singles": singles,
        "controls": controls,
        "batching": {
            "candidate_count": len(candidates_a) + len(candidates_b),
            "control_count": len(controls),
            "oom_fallback": oom_fallback,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
        },
        "provenance": {
            "sae_fingerprint": provenance_fingerprint,
            "decoder_bridge": str(project.paths.decoder_bridge),
            "decoder_bridge_sha256": file_sha256(project.paths.decoder_bridge),
        },
    }
    result["selection"] = select_positive_support_features(result)
    return result


def _choose_llm_features(
    interventions: Mapping[str, object],
) -> list[dict[str, object]]:
    selection = interventions.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("intervention selection is missing")
    selected = [
        dict(item)
        for item in _sequence(
            selection.get("selected_features"), "selected_features"
        )
        if isinstance(item, Mapping)
    ]
    if selected:
        return selected
    cutoffs = selection.get("control_cutoffs")
    if not isinstance(cutoffs, Mapping):
        raise ValueError("control cutoffs are missing")
    fallback: list[dict[str, object]] = []
    for side in ("A", "B"):
        candidates = [
            dict(item)
            for item in _sequence(interventions.get("singles"), "singles")
            if isinstance(item, Mapping)
            and item.get("side") == side
            and _finite(item.get("delta_logit"), "delta_logit") > 0.0
        ]
        candidates.sort(
            key=lambda item: (
                -float(item["delta_logit"]),
                int(item["feature_id"]),
            )
        )
        for item in candidates[:3]:
            fallback.append(
                {
                    **item,
                    "control_cutoff": _finite(
                        cutoffs.get(side), "control cutoff"
                    ),
                    "passed_control": False,
                }
            )
    if not fallback:
        raise RuntimeError("case has no positive-support SAE candidate")
    return fallback


def attach_official_semantics(
    interventions: Mapping[str, object],
    *,
    identity: CaseIdentity = DEFAULT_IDENTITY,
) -> list[dict[str, object]]:
    """Attach only official SAE descriptions to selected/fallback features."""

    result = []
    for feature in _choose_llm_features(interventions):
        metadata = get_feature_metadata(int(feature["feature_id"]))
        result.append(
            {
                **feature,
                "gene": identity.gene_by_side[str(feature["side"])],
                "description": metadata.description,
                "source_url": metadata.source_url,
                "metadata_retrieved_at": metadata.retrieved_at,
            }
        )
    return result


def _write_json_create_only(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")


def prepare_illustrative_run(
    *,
    output_root: Path = OUTPUT_ROOT,
    device: torch.device | str = "cpu",
    model: str = DEFAULT_MODEL,
    identity: CaseIdentity = DEFAULT_IDENTITY,
    data_root: Path = DATA_ROOT,
    model_root: Path = MODEL_ROOT,
    target_pool_root: Path = TARGET_POOL_ROOT,
) -> dict[str, object]:
    """Persist intervention evidence, semantics, prompt, and API request."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        interventions = run_case_interventions(
            device=device,
            identity=identity,
            data_root=data_root,
            model_root=model_root,
            target_pool_root=target_pool_root,
        )
        semantics = attach_official_semantics(
            interventions,
            identity=identity,
        )
        prompt = build_llm_prompt(
            case_probability=float(interventions["probability"]),
            classification_threshold=float(
                interventions["classification_threshold"]
            ),
            feature_semantics=semantics,
            identity=identity,
        )
        request_payload = build_openai_request(
            prompt=prompt,
            model=model,
            identity=identity,
        )
        _write_json_create_only(output_root / "interventions.json", interventions)
        _write_json_create_only(output_root / "llm_input.json", semantics)
        (output_root / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        _write_json_create_only(
            output_root / "openai_request.json", request_payload
        )
    except BaseException:
        if output_root.exists() and not any(output_root.iterdir()):
            output_root.rmdir()
        raise
    return {
        "output_root": str(output_root),
        "probability": interventions["probability"],
        "classification_threshold": interventions["classification_threshold"],
        "selected_feature_count": len(semantics),
        "validated_feature_count": sum(
            bool(item["passed_control"]) for item in semantics
        ),
        "prompt": str(output_root / "prompt.txt"),
        "request": str(output_root / "openai_request.json"),
    }


def _extract_response_text(response: Mapping[str, object]) -> str:
    if response.get("status") != "completed":
        raise RuntimeError(f"OpenAI response status is {response.get('status')!r}")
    for item in _sequence(response.get("output"), "response output"):
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for content in _sequence(item.get("content"), "message content"):
            if not isinstance(content, Mapping):
                continue
            if content.get("type") == "refusal":
                raise RuntimeError("OpenAI API refused the mechanism request")
            if content.get("type") == "output_text":
                return _nonempty(content.get("text"), "response output text")
    raise RuntimeError("OpenAI response contains no output text")


def call_openai_responses(
    request_payload: Mapping[str, object],
    *,
    api_key: str | None = None,
    endpoint: str = OPENAI_ENDPOINT,
    timeout: float = 120.0,
) -> tuple[dict[str, object], dict[str, object]]:
    """Call the Responses API without persisting credentials or raw headers."""

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; configure it in the process environment"
        )
    body = json.dumps(
        request_payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"OpenAI API returned HTTP {error.code}: {detail}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError("OpenAI Responses API request failed") from error
    try:
        response_payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("OpenAI API response is not valid JSON") from error
    if not isinstance(response_payload, Mapping):
        raise RuntimeError("OpenAI API response root is not an object")
    mechanism = json.loads(_extract_response_text(response_payload))
    if not isinstance(mechanism, dict):
        raise RuntimeError("structured mechanism output is not an object")
    usage = response_payload.get("usage")
    metadata = {
        "response_id": response_payload.get("id"),
        "model": response_payload.get("model", request_payload.get("model")),
        "status": response_payload.get("status"),
        "usage": usage if isinstance(usage, Mapping) else None,
        "store_requested": False,
        "transport": "responses_api",
    }
    return mechanism, metadata


def finalize_llm_artifacts(
    mechanism: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    output_root: Path = OUTPUT_ROOT,
    identity: CaseIdentity = DEFAULT_IDENTITY,
) -> dict[str, object]:
    """Validate, render, and create the final transport-independent artifacts."""

    output_root = Path(output_root)
    output_path = output_root / "llm_output.json"
    report_path = output_root / "mechanism_report.md"
    metadata_path = output_root / "api_metadata.json"
    if any(path.exists() for path in (output_path, report_path, metadata_path)):
        raise FileExistsError("LLM output already exists; use a new output root")
    semantics = json.loads(
        (output_root / "llm_input.json").read_text(encoding="utf-8")
    )
    interventions = json.loads(
        (output_root / "interventions.json").read_text(encoding="utf-8")
    )
    validate_llm_mechanism(
        mechanism,
        allowed_features=semantics,
        identity=identity,
    )
    report = render_illustrative_report(
        mechanism,
        case_probability=float(interventions["probability"]),
        classification_threshold=float(
            interventions["classification_threshold"]
        ),
        feature_semantics=semantics,
        api_metadata=metadata,
        identity=identity,
    )
    _write_json_create_only(output_path, mechanism)
    _write_json_create_only(metadata_path, metadata)
    report_path.write_text(report, encoding="utf-8")
    return {
        "llm_output": str(output_path),
        "api_metadata": str(metadata_path),
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
    }


def generate_illustrative_report(
    *,
    output_root: Path = OUTPUT_ROOT,
    api_key: str | None = None,
    identity: CaseIdentity = DEFAULT_IDENTITY,
) -> dict[str, object]:
    """Call the LLM, validate its structured output, and render the report."""

    output_root = Path(output_root)
    request_payload = json.loads(
        (output_root / "openai_request.json").read_text(encoding="utf-8")
    )
    mechanism, metadata = call_openai_responses(
        request_payload,
        api_key=api_key,
    )
    return finalize_llm_artifacts(
        mechanism,
        metadata,
        output_root=output_root,
        identity=identity,
    )


def _add_case_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument(
        "--target-pool-root",
        type=Path,
        default=TARGET_POOL_ROOT,
    )
    parser.add_argument("--case-id", default=DEFAULT_IDENTITY.case_id)
    parser.add_argument("--context", default=DEFAULT_IDENTITY.context)
    parser.add_argument("--gene-a", type=int, default=DEFAULT_IDENTITY.gene_a)
    parser.add_argument("--gene-b", type=int, default=DEFAULT_IDENTITY.gene_b)
    parser.add_argument(
        "--gene-a-name",
        default=DEFAULT_IDENTITY.gene_a_name,
    )
    parser.add_argument(
        "--gene-b-name",
        default=DEFAULT_IDENTITY.gene_b_name,
    )
    parser.add_argument("--row-id", default=DEFAULT_IDENTITY.row_id)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    _add_case_arguments(prepare)
    prepare.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    generate = subparsers.add_parser("generate")
    _add_case_arguments(generate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    identity = CaseIdentity(
        case_id=args.case_id,
        context=args.context,
        gene_a=args.gene_a,
        gene_b=args.gene_b,
        gene_a_name=args.gene_a_name,
        gene_b_name=args.gene_b_name,
        row_id=args.row_id,
    )
    result = (
        prepare_illustrative_run(
            output_root=args.output_root,
            device=args.device,
            model=args.model,
            identity=identity,
            data_root=args.data_root,
            model_root=args.model_root,
            target_pool_root=args.target_pool_root,
        )
        if args.command == "prepare"
        else generate_illustrative_report(
            output_root=args.output_root,
            identity=identity,
        )
    )
    print(
        "ILLUSTRATIVE_CASE_MECHANISM "
        + json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
