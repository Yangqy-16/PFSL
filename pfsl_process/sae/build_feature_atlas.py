"""Build, render, orchestrate, and validate the frozen SAE feature atlas."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "pfsl_process.sae"

from .concept_parsers import build_concept_index, validate_concept_index
from .external_snapshots import (
    SNAPSHOT_SPECS,
    download_external_snapshots,
    validate_external_snapshots,
)
from .feature_atlas_config import (
    AtlasPaths,
    AtlasSourceBinding,
    FeatureAtlasConfig,
    build_source_binding,
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
    write_torch_atomic,
)
from .feature_atlas_selection import (
    build_feature_atlas as build_statistical_atlas,
    select_deep_features,
    validate_selection_manifest,
)
from .feature_concept_discovery import (
    discover_train_candidates,
    validate_train_discovery,
)
from .heldout_concept_evaluation import (
    _frozen_candidate_mapping,
    evaluate_heldout_candidates,
    validate_heldout_evaluation,
)
from .llm_feature_descriptions import (
    CLAUDE_MODEL,
    GEMINI_MODEL,
    GPT_MODEL,
    PROVIDER_MODE_CLAUDE_ONLY,
    PROVIDER_MODE_DUAL,
    PROVIDER_MODE_GPT_ONLY,
    ClaudeCLIProvider,
    CodexCLIProvider,
    GeminiCLIProvider,
    run_feature_descriptions,
    validate_llm_feature_descriptions,
    build_train_packet,
    build_validation_packet,
    unavailable_agreement,
    validate_provider_family,
)
from .codex_cli_transport import (
    CodexCLITransportError,
    probe_codex_model_identity,
)
from .protein_feature_activations import (
    _decode_output_shard,
    _encode_shard,
    aggregate_protein_activations,
    build_protein_covariates,
    validate_protein_activations,
    validate_protein_covariates,
    iter_protein_activation_rows,
)
from .protrek_inference import load_text_embedder
from .sparse_feature_activations import (
    _resolve_device_identity,
    _validate_sparse_shard,
    build_data_index,
    extract_sparse_feature_activations,
    load_frozen_sae,
    validate_sparse_feature_activations,
)
from .span_attribution import (
    EmbeddingCache,
    _validate_production_context as _span_production_context,
    attribute_feature_spans,
    build_feature_plan,
    preflight_feature_plans,
    select_feature_texts,
    validate_feature_payload,
    validate_span_attribution,
    attribute_feature_texts,
    adopt_registered_cache,
    new_stage_manifest,
    publish_incremental_manifest,
    publish_provisional_feature,
    finalize_feature_family,
    TokenizedText,
)
from .sae_metrics import FeatureStatsAccumulator
from .sae_pilot_config import SaePilotConfig
from .sae_training_data import SaeDataIndex, SaeUnitMetadata, ShardMetadata
from .topk_sae import TopKSAE
import torch


SCHEMA_VERSION = 1
STAGES = (
    "external",
    "concepts",
    "sparse",
    "proteins",
    "discovery",
    "heldout",
    "selection",
    "spans",
    "llm",
    "cards",
    "all",
)
CORE_STAGES = (
    "external",
    "concepts",
    "sparse",
    "proteins",
    "discovery",
    "heldout",
    "selection",
    "cards",
)
STAGE_MANIFEST_KEYS_BY_STAGE = {
    "sparse": frozenset(
        {
            "atlas_source_binding",
            "batch_size",
            "complete",
            "device",
            "device_identity",
            "inference_sha256",
            "input_dim",
            "k",
            "model_config_fingerprint",
            "row_count",
            "schema_version",
            "shards",
            "source_provenance",
            "test_fixture_unbound",
            "width",
        }
    ),
    "proteins": frozenset(
        {
            "accession_count",
            "complete",
            "concept_binding",
            "filtered_uniprot_sha256",
            "proteins_per_shard",
            "schema_version",
            "shards",
            "source_provenance",
            "split_counts",
            "width",
        }
    ),
    "spans": frozenset(
        {
            "bindings",
            "budget",
            "cache",
            "complete",
            "completed_features",
            "features",
            "final_outputs",
            "provisional_outputs",
            "schema_version",
        }
    ),
    "llm": frozenset(
        {
            "artifacts",
            "complete",
            "draft_freeze",
            "family_order",
            "input_fingerprint",
            "label_binding",
            "max_attempts_per_phase",
            "max_calls",
            "output_fingerprint",
            "phase",
            "planned_calls",
            "provider_bindings",
            "provider_mode",
            "provider_statuses",
            "schema_version",
            "source_binding",
            "statistical_identities",
            "train_freeze",
            "validation_freeze",
        }
    ),
}
SELECTION_MANIFEST_KEYS = frozenset(
    {
        "atlas",
        "candidates",
        "complete",
        "config_fingerprint",
        "constraints",
        "counts",
        "frozen",
        "inference",
        "provenance",
        "schema_version",
        "seed",
        "selection_fingerprint",
        "selected",
        "width",
    }
)
CARD_KEYS = (
    "schema_version",
    "feature_index",
    "identity",
    "provenance",
    "frozen_selection",
    "top_texts",
    "top_proteins",
    "contrasts",
    "field_diagnostics",
    "source_exclusion",
    "train_concept_results",
    "heldout_concept_results",
    "splitting_diagnostics",
    "absorption_diagnostics",
    "spans",
    "raw_llm_references",
    "provider_mode",
    "llm_agreement",
    "counterevidence",
    "limitations",
    "statistical_classification",
    "masking_evidence_grade",
    "statistical_evidence_grade",
)
_CARD_KEY_SET = frozenset(CARD_KEYS)
_STATISTICAL_CLASSES = frozenset(
    {
        "field_detector",
        "validated_biological_concept",
        "mixed",
        "exploratory",
        "dead_or_too_sparse",
        "unresolved",
    }
)
_MASKING_GRADES = frozenset(
    {
        "strong_span",
        "distributed_evidence",
        "topk_competition_sensitive",
        "unstable_span",
        "insufficient_texts",
    }
)
_STATISTICAL_GRADES = frozenset(
    {"unvalidated", "insufficient", "descriptive", "confirmatory"}
)
_GRADE_RANK = {
    "unvalidated": 0,
    "insufficient": 1,
    "descriptive": 2,
    "confirmatory": 3,
}
_CAUSAL_WORDING = (
    " causes ",
    " caused ",
    " causally controls ",
    " drives the biological ",
    " is responsible for ",
    " proves the mechanism ",
    " establishes biological causality",
)
_DISALLOWED_WORDING = (
    re.compile(r"\bbiologic(?:al|ally)\s+causal(?:ity)?\b", re.IGNORECASE),
    re.compile(r"\bdefinitive(?:ly)?\b.*\bbiological mechanism\b", re.IGNORECASE),
    re.compile(r"\b(?:feature\s+)?(?:splitting|absorption)\s+is\s+proven\b", re.IGNORECASE),
    re.compile(r"\b(?:claude|gemini)\s+defined\s+the\s+feature\b", re.IGNORECASE),
    re.compile(r"\bsae\s+(?:improves|preserves)\b.*\b(?:sl|synthetic lethality)\s+prediction\b", re.IGNORECASE),
)
_DUAL_CONSENSUS_WORDING = (
    re.compile(
        r"\b(?:claude\s+and\s+gemini|both\s+(?:providers|models)|dual[- ]provider)\b"
        r".{0,80}\b(?:agree(?:ment|d)?|consensus)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:agree(?:ment|d)?|consensus)\b.{0,80}"
        r"\b(?:claude\s+and\s+gemini|both\s+(?:providers|models)|dual[- ]provider)\b",
        re.IGNORECASE,
    ),
)
_PROVIDER_PAIR_WORDING = re.compile(
    r"(?:\bclaude\b.*\bgemini\b|\bgemini\b.*\bclaude\b|"
    r"\bboth\s+(?:llms?|providers?|models?)\b|\bdual[- ]provider\b)",
    re.IGNORECASE,
)
_AGREEMENT_WORDING = re.compile(
    r"\b(?:agree(?:s|d|ment)?|consensus|match(?:es|ed|ing)?)\b",
    re.IGNORECASE,
)
_LLM_RESERVED_ARTIFACT = re.compile(
    r"^(?:train_feature_|validation_feature_|draft_|verdict_|agreement_feature_|"
    r"feature_|train_freeze_|draft_freeze_|validation_freeze_).*\.json$"
)
_GPT_CROSS_PROVIDER_CONSENSUS = re.compile(
    r"(?:\bgpt\b.{0,160}\b(?:claude|gemini)\b|"
    r"\b(?:claude|gemini)\b.{0,160}\bgpt\b|"
    r"\b(?:multiple|several|three|3)\s+(?:llms?|providers?|models?)\b)",
    re.IGNORECASE,
)
_NEGATED_CONSENSUS_SPANS = (
    re.compile(
        r"\b(?:gpt|claude|gemini)\b\s+do(?:es)?\s+not\s+agree\s+with\s+"
        r"\b(?:gpt|claude|gemini)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:gpt|claude|gemini)\b\s+and\s+\b(?:gpt|claude|gemini)\b\s+"
        r"do\s+not\s+agree\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+agreement\s+(?:(?:is|was)\s+)?(?:computed|available)\s+for\s+"
        r"\b(?:gpt|claude|gemini)\b.{0,40}?\b(?:gpt|claude|gemini)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+agreement\s+(?:between|among)\s+\b(?:gpt|claude|gemini)\b"
        r".{0,40}?\b(?:gpt|claude|gemini)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bagreement\s+(?:(?:is|was)\s+)?(?:not\s+computed|unavailable)\b",
        re.IGNORECASE,
    ),
)
_CARD_LIMITATIONS = (
    "This card describes a model-causal masking diagnostic, not biological causality.",
    "Same-source evidence is diagnostic only and is never primary evidence.",
    "LLM descriptions are candidate interpretations and cannot change statistical or masking grades.",
)


class FeatureAtlasBuildError(ValueError):
    """Raised when orchestration, evidence-card, or final validation fails."""


def _transaction_event(event: str) -> None:
    """Fault-injection seam for root/LLM publication transaction tests."""


@dataclass(frozen=True)
class SmokePlan:
    feature_count: int
    databases: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.feature_count) is not int or self.feature_count != 50:
            raise ValueError("bounded smoke feature_count must be exactly 50")
        if self.databases != ("go", "reactome"):
            raise ValueError("bounded smoke databases must be exactly go,reactome")

    @property
    def feature_indices(self) -> tuple[int, ...]:
        return tuple(range(self.feature_count))

    @property
    def selected_feature_cap(self) -> int:
        return 2

    def filter_rows(
        self, rows: Sequence[Mapping[str, object]]
    ) -> list[Mapping[str, object]]:
        allowed = set(self.feature_indices)
        databases = set(self.databases)
        return [
            row
            for row in rows
            if row.get("database") in databases
            and row.get("feature_index") in allowed
        ]


@dataclass(frozen=True)
class SmokeStageInputs:
    databases: tuple[str, ...]
    feature_indices: tuple[int, ...]
    protein_subset: tuple[str, ...]
    selected_features: tuple[int, ...]


@dataclass(frozen=True)
class _SmokeSAEOutput:
    activations: torch.Tensor
    preactivations: torch.Tensor


class _SmokeTextEmbedder:
    fixture_model_binding = {"fingerprint": "fixture-unbound"}

    def tokenize_with_offsets(self, text: str) -> TokenizedText:
        pieces = text.replace("[MASK]", " [MASK] ").replace(".", " . ").split()
        offsets: list[tuple[int, int]] = [(0, 0)]
        ids, tokens, cursor = [2], ["[CLS]"], 0
        for piece in pieces:
            start = text.index(piece, cursor)
            cursor = start + len(piece)
            offsets.append((start, cursor))
            tokens.append(piece.casefold())
            ids.append(4 if piece == "[MASK]" else 10 + sum(map(ord, piece)) % 100)
        offsets.append((0, 0))
        return TokenizedText(
            text=text,
            token_ids=tuple([*ids, 3]),
            tokens=tuple([*tokens, "[SEP]"]),
            offsets=tuple(offsets),
            special_tokens_mask=tuple([1, *([0] * len(pieces)), 1]),
            attention_mask=tuple([1] * (len(pieces) + 2)),
            mask_token_id=4,
        )

    def verify_rendered_perturbation(
        self, original: TokenizedText, rendered_text: str, planned: tuple[int, ...]
    ) -> TokenizedText:
        observed = self.tokenize_with_offsets(rendered_text)
        if observed.token_ids != planned:
            raise FeatureAtlasBuildError("smoke masking retokenization differs")
        return observed

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        rows = []
        for text in texts:
            lowered = text.casefold()
            row = torch.zeros(50, dtype=torch.float32)
            row[0] = 1.0 if "repair" in lowered else 0.05
            row[1] = 1.0 if "membrane" in lowered else 0.10
            rows.append(torch.nn.functional.normalize(row, dim=0))
        return torch.stack(rows)


class _SmokeMaskingSAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(
        self, inputs: torch.Tensor, update_dead_stats: bool = False
    ) -> _SmokeSAEOutput:
        pre = inputs[:, :50]
        winner = pre.argmax(dim=1, keepdim=True)
        activations = torch.zeros_like(pre).scatter(
            1, winner, pre.gather(1, winner)
        )
        return _SmokeSAEOutput(activations=activations, preactivations=pre)


def _smoke_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _smoke_concept(
    database: str, concept_id: str, members: Sequence[str]
) -> dict[str, object]:
    tracks = (
        [["primary", list(members)], ["secondary", list(members)]]
        if database == "go"
        else [["all", list(members)], ["leaf", list(members)]]
    )
    return {
        "concept_id": concept_id,
        "concept_name": "DNA repair fixture",
        "concept_type": "biological_process" if database == "go" else "pathway",
        "database": database,
        "evidence_tier": "official",
        "membership_tracks": tracks,
        "original_accessions": [],
        "parent_ids": [],
        "protein_accessions": list(members),
        "redundancy_cluster_id": None,
        "source_release": "smoke-1",
        "source_sha256": _smoke_digest([database, concept_id, list(members)]),
    }


def _smoke_feature_record(
    feature: int,
    *,
    candidate: Mapping[str, object] | None,
    heldout: Mapping[str, object] | None,
) -> dict[str, object]:
    texts = {
        "train": [{
            "accession": "T00", "activation": 2.0, "aspect_index": 0,
            "row_index": 0, "split": "train", "text": "DNA repair complex.",
            "unit_id": "T00:u0",
        }],
        "validation": [{
            "accession": "V00", "activation": 3.0, "aspect_index": 0,
            "row_index": 24, "split": "validation", "text": "repair complex factor.",
            "unit_id": "V00:u0",
        }],
    }
    train_count = 12 if feature == 0 else 4
    validation_count = 20 if feature == 0 else 4
    return {
        "contrasts": [], "dead": False, "decoder_norm": 1.0,
        "feature_index": feature, "firing_accessions": [],
        "heldout_concept_results": [dict(heldout)] if heldout else [],
        "preferred_aspect": feature % 24,
        "source_effect_loss": 0.9 if feature < 2 else 0.1,
        "source_lift": 3.0 if feature < 2 else 1.0,
        "source_purity": 0.9 if feature < 2 else 0.3,
        "split_reproducibility": min(train_count, validation_count) / max(train_count, validation_count),
        "splitting_cluster": None, "too_sparse": False,
        "top_protein_candidates": {"train": [], "validation": []},
        "top_text_candidates": texts,
        "train_activation_count": train_count,
        "train_concept_ids": [str(candidate["concept_id"])] if candidate else [],
        "train_concept_results": [dict(candidate)] if candidate else [],
        "train_max_activation": 3.0, "train_mean_activation": 1.0,
        "train_split_coverage": 0.5,
        "validation_activation_count": validation_count,
        "validation_concept_ids": [str(heldout["concept_id"])] if heldout else [],
        "validation_max_activation": 3.0, "validation_mean_activation": 1.0,
        "validation_split_coverage": 0.5,
    }


def _run_smoke_fixture_pipeline(
    paths: AtlasPaths,
    inputs: SmokeStageInputs,
    providers: Sequence[object],
) -> dict[str, Mapping[str, object]]:
    """Execute a bounded, fixture-bound Task 4--10 pipeline under /tmp."""

    try:
        _, provider_mode = validate_provider_family(
            providers,  # type: ignore[arg-type]
            allow_test_fixture_unbound=True,
            return_mode=True,
        )
    except Exception as error:
        raise FeatureAtlasBuildError(
            f"smoke Task 10 provider family is invalid: {error}"
        ) from error

    root, layout = paths.output_dir, _layout(paths)
    fixture = root / "smoke_fixture"
    run_dir, activation_dir = fixture / "run", fixture / "activations"
    run_dir.mkdir(parents=True, exist_ok=True)
    activation_dir.mkdir(parents=True, exist_ok=True)
    accessions = tuple([*(f"T{i:02d}" for i in range(24)), *(f"V{i:02d}" for i in range(40))])
    train_accessions, validation_accessions = accessions[:24], accessions[24:]
    unit_rows, units = [], []
    raw_embeddings = []
    for row, accession in enumerate(accessions):
        split = "train" if row < 24 else "validation"
        positive = row < 12 or 24 <= row < 44
        vector = [2.0, 0.1, 0.0, 0.0] if positive else [0.1, 2.0, 0.0, 0.0]
        raw_embeddings.append(vector)
        unit_id = f"{accession}:u0"
        unit_rows.append({
            "accession": accession, "aspect_index": 0, "aspect_name": "function",
            "parent_aspect_text": "function", "text": (
                "DNA repair complex." if positive else "membrane protein."
            ), "unit_id": unit_id, "unit_index": 0, "unit_type": "smoke",
        })
        units.append(SaeUnitMetadata(
            row_index=row, unit_id=unit_id, accession=accession, aspect_index=0,
            split=split,
        ))
    embeddings = torch.nn.functional.normalize(
        torch.tensor(raw_embeddings, dtype=torch.float32), dim=1
    )
    units_path = fixture / "sae_units.jsonl"
    write_jsonl_atomic(units_path, unit_rows)
    source_path = activation_dir / "shard_00000.pt"
    write_torch_atomic(source_path, {
        "embeddings": embeddings, "unit_ids": [unit.unit_id for unit in units],
    })
    provenance = {
        "activation_manifest_sha256": _smoke_digest("activation-manifest"),
        "assignment_sha256": _smoke_digest([train_accessions, validation_accessions]),
        "text_manifest_sha256": _smoke_digest("text-manifest"),
    }
    source_index = SaeDataIndex(
        units=tuple(units),
        shards=(ShardMetadata(source_path, 0, len(units), sha256_file(source_path)),),
        train_accessions=frozenset(train_accessions),
        validation_accessions=frozenset(validation_accessions),
        train_unit_count=24, validation_unit_count=40,
        source_train_unit_count=24, source_validation_unit_count=40,
        train_aspect_counts=(24, *([0] * 23)),
        validation_aspect_counts=(40, *([0] * 23)),
        assignment_sha256=provenance["assignment_sha256"],
        source_assignment_sha256=provenance["assignment_sha256"],
        text_manifest_sha256=provenance["text_manifest_sha256"],
        activation_manifest_sha256=provenance["activation_manifest_sha256"],
    )
    sae_config = SaePilotConfig(
        input_dim=4, width=50, k=2, aux_k=2, batch_size=8,
        seed=42, max_epochs=2, patience=1,
    )
    sae = TopKSAE(
        input_dim=4, width=50, k=2, aux_k=2,
        activation_threshold=1e-3, dead_steps_threshold=1,
    )
    with torch.no_grad():
        sae.encoder_weight.zero_()
        sae.encoder_weight[0, 0] = 1.0
        sae.encoder_weight[1, 1] = 1.0
        sae.encoder_bias.zero_()
        sae.pre_bias.zero_()
    inference_path = run_dir / "inference.pt"
    write_torch_atomic(inference_path, {
        "schema_version": 1, "config": sae_config.to_dict(),
        "config_fingerprint": sae_config.fingerprint(), "provenance": provenance,
        "model_state": sae.state_dict(),
    })
    with torch.inference_mode():
        dense = sae(embeddings, update_dead_stats=False).activations
    train_stats, validation_stats = (
        FeatureStatsAccumulator(50, 24), FeatureStatsAccumulator(50, 24)
    )
    train_stats.update(dense[:24], torch.zeros(24, dtype=torch.long))
    validation_stats.update(dense[24:], torch.zeros(40, dtype=torch.long))
    feature_stats_path = run_dir / "feature_stats.pt"
    write_torch_atomic(feature_stats_path, {
        "schema_version": 1, "config": sae_config.to_dict(),
        "config_fingerprint": sae_config.fingerprint(), "provenance": provenance,
        "width": 50, "train": train_stats.finalize(),
        "validation": validation_stats.finalize(),
    })
    sparse = extract_sparse_feature_activations(
        source_index, inference_path, layout["sparse"], device="cpu", batch_size=8,
        allow_test_fixture_unbound=True,
    )
    validate_sparse_feature_activations(
        source_index, inference_path, feature_stats_path, layout["sparse"],
        allow_test_fixture_unbound=True,
    )
    split_path = run_dir / "split_manifest.json"
    write_json_atomic(split_path, {
        "assignment_sha256": provenance["assignment_sha256"],
        "train_accession_count": 24, "train_accessions": list(train_accessions),
        "units_sha256": sha256_file(units_path),
        "validation_accession_count": 40,
        "validation_accessions": list(validation_accessions),
    })
    uniprot_path = fixture / "uniprot.tsv"
    uniprot_path.write_text(
        "Entry\tSequence\n" + "".join(f"{accession}\tAAAA\n" for accession in accessions),
        encoding="utf-8",
    )
    members = tuple([*train_accessions[:12], *validation_accessions[:20]])
    concepts = [
        _smoke_concept("go", "GO:0000001", members),
        _smoke_concept("reactome", "R-HSA-1", members),
    ]
    concepts.sort(key=lambda row: (str(row["database"]), str(row["concept_id"])))
    concept_dir = root / "concepts"
    concept_dir.mkdir(parents=True, exist_ok=True)
    concept_path = concept_dir / "concept_index.jsonl"
    write_jsonl_atomic(concept_path, concepts)
    write_json_atomic(concept_dir / "manifest.json", {
        "files": {"concept_index": {
            "bytes": concept_path.stat().st_size, "path": str(concept_path.resolve()),
            "sha256": sha256_file(concept_path),
        }}, "schema_version": 1,
    })
    proteins = aggregate_protein_activations(
        units_path, layout["sparse"], split_path, uniprot_path, layout["proteins"],
        concept_index_path=concept_path, proteins_per_shard=16,
        expected_accession_count=64, expected_split_counts=(24, 40),
        allow_test_fixture_unbound=True, expected_width=50, expected_k=2,
        allow_test_fixture_concept_unbound=True,
    )
    validate_protein_activations(
        units_path, layout["sparse"], split_path, uniprot_path, layout["proteins"],
        concept_index_path=concept_path, expected_accession_count=64,
        expected_split_counts=(24, 40), allow_test_fixture_unbound=True,
        expected_width=50, expected_k=2, allow_test_fixture_concept_unbound=True,
    )
    covariates = build_protein_covariates(
        uniprot_path, units_path, concept_path, layout["covariates"],
        expected_accession_count=64, allow_test_fixture_concept_unbound=True,
    )

    config = FeatureAtlasConfig(
        width=50, k=2, biological_slots=1, field_slots=2, control_slots=1,
        train_min_annotated=4, train_min_firing=4, train_min_overlap=4,
        validation_descriptive_min=4, validation_confirmatory_min=8,
        controls_per_positive=1, permutations=31,
    )

    class TrainRows:
        def iter_train_rows(self) -> object:
            yield from iter_protein_activation_rows(layout["proteins"], split="train")

    discovery = discover_train_candidates(TrainRows(), concepts, config, concept_dir)
    validate_train_discovery(
        concept_dir, config_fingerprint=config.fingerprint(),
        input_provenance=discovery["input_provenance"],
    )
    candidates = list(strict_json_lines(layout["train_candidates"]))
    if not candidates:
        raise FeatureAtlasBuildError("smoke Task 6 produced no candidate")

    bridge = root / "smoke_bridge"
    bridge_proteins, bridge_discovery = bridge / "task5", bridge / "task6"
    shutil.copytree(layout["proteins"], bridge_proteins)
    shutil.copytree(concept_dir, bridge_discovery)
    concept_manifest_path = concept_dir / "manifest.json"
    concept_binding = {
        "available": True, "index_sha256": sha256_file(concept_path),
        "manifest_sha256": sha256_file(concept_manifest_path),
        "test_fixture_unbound": True,
    }
    bridge_protein_manifest = strict_json_load(bridge_proteins / "manifest.json")
    bridge_protein_manifest["concept_binding"] = concept_binding
    write_json_atomic(bridge_proteins / "manifest.json", bridge_protein_manifest)
    bridge_discovery_manifest = strict_json_load(
        bridge_discovery / "discovery_manifest.json"
    )
    bridge_discovery_manifest["input_provenance"][
        "protein_activation_manifest_sha256"
    ] = sha256_file(bridge_proteins / "manifest.json")
    bridge_discovery_manifest["input_provenance"]["concept_index_sha256"] = sha256_file(
        concept_path
    )
    bridge_discovery_manifest["upstream_bindings"] = {
        "concept_binding": concept_binding,
        "source_provenance": bridge_protein_manifest["source_provenance"],
    }
    write_json_atomic(
        bridge_discovery / "discovery_manifest.json", bridge_discovery_manifest
    )
    bridge_covariates = bridge / "protein_covariates.jsonl"
    shutil.copy2(layout["covariates"], bridge_covariates)
    bridge_covariate_manifest = strict_json_load(
        layout["covariates"].with_suffix(".manifest.json")
    )
    bridge_covariate_manifest["concept_binding"] = concept_binding
    bridge_covariate_manifest["output"] = {
        "bytes": bridge_covariates.stat().st_size,
        "path": str(bridge_covariates.resolve()),
        "sha256": sha256_file(bridge_covariates),
    }
    bridge_covariate_manifest["output_bytes"] = bridge_covariates.stat().st_size
    bridge_covariate_manifest["output_sha256"] = sha256_file(bridge_covariates)
    write_json_atomic(
        bridge_covariates.with_suffix(".manifest.json"), bridge_covariate_manifest
    )
    task5_rows = [
        {"accession": row.accession, "all_sources": list(row.all_sources.items()),
         "eligibility": dict(row.eligibility),
         "source_excluded": {key: list(value.items()) for key, value in row.source_excluded.items()}}
        for row in iter_protein_activation_rows(layout["proteins"])
    ]
    bridge_rows = [
        {"accession": row.accession, "all_sources": list(row.all_sources.items()),
         "eligibility": dict(row.eligibility),
         "source_excluded": {key: list(value.items()) for key, value in row.source_excluded.items()}}
        for row in iter_protein_activation_rows(bridge_proteins)
    ]
    if task5_rows != bridge_rows:
        raise FeatureAtlasBuildError("smoke Task 5 bridge changed protein values")
    original_candidate_rows = list(strict_json_lines(layout["train_candidates"]))
    bridge_candidate_rows = list(
        strict_json_lines(bridge_discovery / "train_candidates.jsonl")
    )
    if original_candidate_rows != bridge_candidate_rows:
        raise FeatureAtlasBuildError("smoke Task 6 bridge changed candidate values")
    bridge_receipt = {
        "bridge_discovery_manifest_sha256": sha256_file(
            bridge_discovery / "discovery_manifest.json"
        ),
        "bridge_protein_manifest_sha256": sha256_file(
            bridge_proteins / "manifest.json"
        ),
        "bridge_protein_rows_sha256": _smoke_digest(bridge_rows),
        "candidate_rows_sha256": _smoke_digest(original_candidate_rows),
        "concept_manifest_sha256": sha256_file(concept_manifest_path),
        "discovery_manifest_sha256": sha256_file(
            concept_dir / "discovery_manifest.json"
        ),
        "discovery_provider_sha256": discovery["input_provenance"][
            "protein_activation_manifest_sha256"
        ],
        "protein_manifest_sha256": sha256_file(layout["proteins"] / "manifest.json"),
        "protein_rows_sha256": _smoke_digest(task5_rows),
        "schema_version": 1,
    }
    write_json_atomic(bridge / "manifest.json", bridge_receipt)
    if strict_json_load(bridge / "manifest.json") != bridge_receipt:
        raise FeatureAtlasBuildError("smoke bridge receipt failed canonical validation")
    heldout = evaluate_heldout_candidates(
        bridge_discovery / "train_candidates.jsonl", bridge_proteins,
        bridge_covariates, concept_path, config, layout["heldout"],
        allow_test_fixture_unbound=True,
    )
    heldout_rows = list(strict_json_lines(layout["heldout_results"]))
    candidate = next((row for row in candidates if row["feature_index"] == 0), None)
    heldout_row = next((row for row in heldout_rows if row["feature_index"] == 0), None)
    if candidate is None or heldout_row is None:
        raise FeatureAtlasBuildError("smoke Task 6/7 bridge lost feature 0")
    records = [
        _smoke_feature_record(
            feature,
            candidate=candidate if feature == 0 else None,
            heldout=heldout_row if feature == 0 else None,
        )
        for feature in inputs.feature_indices
    ]
    fixture_provenance = {"bridge_sha256": sha256_file(bridge / "manifest.json")}
    build_statistical_atlas(
        records, config, layout["atlas"], decoder=torch.eye(50, dtype=torch.float32),
        provenance=fixture_provenance, allow_test_fixture_unbound=True,
    )
    selection = select_deep_features(
        layout["atlas"], torch.eye(50, dtype=torch.float32), config,
        layout["selection"], provenance=fixture_provenance,
        allow_test_fixture_unbound=True,
    )
    validate_selection_manifest(
        layout["selection"], layout["atlas"], torch.eye(50, dtype=torch.float32),
        config, provenance=fixture_provenance, allow_test_fixture_unbound=True,
    )
    selected_features = sorted(
        int(row["feature_index"]) for row in selection["selected"]
    )
    if len(selected_features) != 2:
        raise FeatureAtlasBuildError(
            f"smoke Task 8 did not freeze exactly two features: {selected_features}"
        )

    atlas_rows = {int(row["feature_index"]): row for row in strict_json_lines(layout["atlas"])}
    embedder, masking_sae = _SmokeTextEmbedder(), _SmokeMaskingSAE()
    raw_spans = []
    for feature in selected_features:
        raw_spans.append(attribute_feature_texts(
            feature,
            {split: select_feature_texts(atlas_rows[feature], split)
             for split in ("train", "validation")},
            embedder, masking_sae, EmbeddingCache(layout["spans"] / "cache"),
            batch_size=8, seed=42, max_encodings=150_000, permutations=31,
        ))
    bindings = {
        "config_fingerprint": config.fingerprint(),
        "model_binding": {"fingerprint": "fixture-unbound"},
        "source_binding": {name: _smoke_digest(name) for name in AtlasSourceBinding.__dataclass_fields__},
        "task4_binding": {
            "feature_stats_sha256": sha256_file(feature_stats_path),
            "inference_sha256": sha256_file(inference_path),
            "sparse_manifest_sha256": sha256_file(layout["sparse"] / "manifest.json"),
        },
        "task8_binding": {
            "atlas_sha256": sha256_file(layout["atlas"]),
            "selection_fingerprint": selection["selection_fingerprint"],
            "selection_sha256": sha256_file(layout["selection"]),
        },
        "tokenizer_binding": {"fixture_unbound": True},
        "device_identity": {"kind": "cpu", "logical_device": "cpu"},
        "test_fixture_unbound": True,
    }
    span_manifest = new_stage_manifest(
        bindings, features=selected_features, budget=raw_spans[0]["budget"]
    )
    publish_incremental_manifest(layout["spans"], span_manifest)
    for payload in raw_spans:
        span_manifest = publish_provisional_feature(
            layout["spans"], span_manifest, payload
        )
    span_manifest = adopt_registered_cache(
        layout["spans"], span_manifest,
        {sha for payload in raw_spans for sha in payload["budget"]["executed_sha256"]},
    )
    publish_incremental_manifest(layout["spans"], span_manifest)
    span_manifest = finalize_feature_family(layout["spans"], span_manifest, raw_spans)
    for feature in selected_features:
        final_span = strict_json_load(
            layout["spans"] / f"feature_{feature:04d}.json"
        )
        validate_feature_payload(
            final_span, expected_fdr=float(final_span["metrics"]["fdr"])
        )
    plans = {
        feature: build_feature_plan(
            feature,
            {split: select_feature_texts(atlas_rows[feature], split)
             for split in ("train", "validation")},
            embedder, seed=42, permutations=31,
        )
        for feature in selected_features
    }
    validate_span_attribution(
        layout["spans"], current_bindings=bindings, atlas_rows=atlas_rows,
        sae=masking_sae, plans_by_feature=plans,
    )

    span_rows = {
        feature: strict_json_load(layout["spans"] / f"feature_{feature:04d}.json")
        for feature in selected_features
    }
    trains = [build_train_packet(atlas_rows[feature], span_rows[feature]) for feature in selected_features]
    validations = {
        feature: build_validation_packet(atlas_rows[feature], span_rows[feature])
        for feature in selected_features
    }
    identities = []
    for feature in selected_features:
        confirmed = [row for row in atlas_rows[feature]["heldout_concept_results"] if row.get("confirmed") is True]
        identities.append({
            "feature_index": feature,
            "masking_evidence_grade": span_rows[feature]["evidence_grade"],
            "statistical_classification": atlas_rows[feature]["classification"],
            "statistical_evidence_grade": (
                max((str(row["evidence_tier"]) for row in confirmed),
                    key={"unvalidated": 0, "insufficient": 1, "descriptive": 2, "confirmatory": 3}.get)
                if confirmed else "unvalidated"
            ),
            "validated_concept_ids": sorted(str(row["concept_id"]) for row in confirmed),
        })
    llm = run_feature_descriptions(
        trains, layout["llm"], providers=providers, label_embedder=embedder,
        statistical_identities=identities,
        validation_materializer=lambda _: validations,
        allow_test_fixture_unbound=True,
    )
    required_providers = (
        ("gpt",)
        if provider_mode == PROVIDER_MODE_GPT_ONLY
        else ("claude", "gemini")
        if provider_mode == PROVIDER_MODE_DUAL
        else ("claude",)
    )
    for provider_name in required_providers:
        status = _provider_status(
            provider_name,
            {"manifest": llm, "directory": str(layout["llm"])},
        )
        if status["success"] is not True:
            raise FeatureAtlasBuildError(
                f"smoke {provider_name} fake CLI draft/verdict failed closed"
            )
    validate_llm_feature_descriptions(
        layout["llm"], label_embedder=embedder,
        expected_statistical_identities=identities,
        expected_source_binding=llm["source_binding"],
        allow_test_fixture_unbound=True,
    )
    layout["cards"].mkdir(parents=True, exist_ok=True)
    outputs = []
    for feature in selected_features:
        llm_payload = strict_json_load(layout["llm"] / f"feature_{feature:04d}.json")
        card = build_evidence_card(
            atlas_rows[feature],
            next(row for row in selection["selected"] if row["feature_index"] == feature),
            span_rows[feature], llm_payload,
        )
        json_path = layout["cards"] / f"feature_{feature:04d}.json"
        markdown_path = layout["cards"] / f"feature_{feature:04d}.md"
        write_json_atomic(json_path, card)
        _write_text_atomic(markdown_path, render_evidence_card_markdown(card))
        for kind, path in (("json", json_path), ("markdown", markdown_path)):
            metadata = file_metadata(path)
            outputs.append({
                "bytes": metadata["bytes"], "feature_index": feature,
                "kind": kind, "path": path.name, "sha256": metadata["sha256"],
            })
    card_manifest = {
        "complete": True, "feature_count": 2, "features": selected_features,
        "outputs": outputs,
        "provider_mode": provider_mode,
        "schema_version": 1,
        "selection_fingerprint": selection["selection_fingerprint"],
    }
    write_json_atomic(layout["cards"] / "manifest.json", card_manifest)
    _validate_cards_stage(layout["cards"])
    return {
        "sparse": sparse, "proteins": proteins, "discovery": discovery,
        "heldout": heldout, "selection": selection, "spans": span_manifest,
        "llm": llm, "cards": card_manifest,
    }


def _run_smoke_stage_family(
    paths: AtlasPaths, plan: SmokePlan, providers: Sequence[object]
) -> dict[str, object]:
    resolved = paths.output_dir.resolve()
    production = AtlasPaths().output_dir.resolve()
    if resolved == production or not resolved.is_relative_to(Path("/tmp").resolve()):
        raise FeatureAtlasBuildError(
            "smoke adapter requires a dedicated /tmp output and refuses production output"
        )
    inputs = SmokeStageInputs(
        databases=plan.databases,
        feature_indices=plan.feature_indices,
        protein_subset=tuple(f"smoke-protein-{index:04d}" for index in range(64)),
        selected_features=plan.feature_indices[: plan.selected_feature_cap],
    )
    artifacts = _run_smoke_fixture_pipeline(paths, inputs, providers)
    selection = artifacts["selection"]
    selected = [int(row["feature_index"]) for row in selection["selected"]]
    results = {
        stage: {
            "artifact_manifest_sha256": sha256_file(
                (_layout(paths)[stage] / "manifest.json")
                if stage in {"sparse", "proteins", "spans", "llm", "cards"}
                else _layout(paths)["discovery_manifest"]
                if stage == "discovery"
                else _layout(paths)["heldout_manifest"]
                if stage == "heldout"
                else _layout(paths)["selection"]
            ),
            "databases": list(inputs.databases),
            "feature_count": len(inputs.feature_indices),
            "protein_count": len(inputs.protein_subset),
            "selected_features": selected,
            "stage": stage,
            "validated": True,
        }
        for stage in (
            "sparse", "proteins", "discovery", "heldout", "selection",
            "spans", "llm", "cards",
        )
    }
    llm_manifest = artifacts["llm"]
    status_context = {
        "manifest": llm_manifest,
        "directory": str(_layout(paths)["llm"]),
    }
    gpt = _provider_status("gpt", status_context)
    claude = _provider_status("claude", status_context)
    gemini = _provider_status("gemini", status_context)
    provider_mode = llm_manifest.get("provider_mode")
    if (
        provider_mode == PROVIDER_MODE_GPT_ONLY and gpt["success"] is not True
    ) or (
        provider_mode != PROVIDER_MODE_GPT_ONLY and not claude["success"]
    ) or (
        provider_mode == PROVIDER_MODE_DUAL and not gemini["success"]
    ):
        raise FeatureAtlasBuildError("smoke Task 10 fake CLI draft/verdict did not complete")
    statuses = {
        "statistical_atlas": {"core": False, "status": "bounded_smoke_complete"},
        "masking": {"success": True, "status": "complete"},
        "gpt": gpt,
        "claude": claude,
        "gemini": gemini,
    }
    result = {
        "provider_mode": provider_mode,
        "smoke": True,
        "stages": results,
        "statuses": statuses,
    }
    _validate_root_llm_provider_bindings(paths, llm_manifest)
    root_config = strict_json_load(paths.output_dir / "config.json")
    source_manifest = strict_json_load(paths.output_dir / "source_manifest.json")
    write_json_atomic(_layout(paths)["summary"], {
        "complete": True,
        "config_fingerprint": root_config["config_fingerprint"],
        "files": _summary_files(paths),
        "provider_mode": provider_mode,
        "schema_version": SCHEMA_VERSION,
        "source_binding": source_manifest["source_binding"],
        "statuses": statuses,
    })
    return result


def _clone(value: object) -> object:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise FeatureAtlasBuildError("evidence contains non-JSON or non-finite data") from error


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_wording(value: object, location: str = "card") -> None:
    if isinstance(value, str):
        folded = f" {value.casefold()} "
        safe_causality_negation = bool(
            re.search(
                r"\b(?:not|no\s+claim\s+of)\s+biological\s+causality\b",
                value,
                re.IGNORECASE,
            )
        )
        pattern_matches = [pattern.search(value) for pattern in _DISALLOWED_WORDING]
        if any(term in folded for term in _CAUSAL_WORDING) or any(
            match
            for index, match in enumerate(pattern_matches)
            if not (index == 0 and safe_causality_negation)
        ):
            raise FeatureAtlasBuildError(
                f"{location} makes a prohibited wording/biological-causality claim"
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_wording(item, f"{location}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_wording(item, f"{location}.{key}")


def _validate_no_dual_consensus_wording(
    value: object, location: str = "card"
) -> None:
    if isinstance(value, str):
        positive_claims = value
        for pattern in _NEGATED_CONSENSUS_SPANS:
            positive_claims = pattern.sub("", positive_claims)
        if (
            any(pattern.search(positive_claims) for pattern in _DUAL_CONSENSUS_WORDING)
            or _PROVIDER_PAIR_WORDING.search(positive_claims)
            and _AGREEMENT_WORDING.search(positive_claims)
            or _GPT_CROSS_PROVIDER_CONSENSUS.search(positive_claims)
            and _AGREEMENT_WORDING.search(positive_claims)
        ):
            raise FeatureAtlasBuildError(
                f"{location} claims dual-provider consensus when agreement is unavailable"
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_no_dual_consensus_wording(item, f"{location}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_no_dual_consensus_wording(item, f"{location}.{key}")


def _feature(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise FeatureAtlasBuildError(f"{label} feature identity is invalid")
    return value


def _statistical_grade(results: object) -> str:
    if not isinstance(results, list):
        raise FeatureAtlasBuildError("heldout concept results are invalid")
    grades = []
    for result in results:
        if not isinstance(result, Mapping):
            raise FeatureAtlasBuildError("heldout concept result is invalid")
        if result.get("confirmed") is True:
            tier = result.get("evidence_tier")
            if tier not in _STATISTICAL_GRADES:
                raise FeatureAtlasBuildError("confirmed heldout evidence tier is invalid")
            grades.append(str(tier))
    return max(grades, key=_GRADE_RANK.__getitem__) if grades else "unvalidated"


def _selection_failures(selection: Mapping[str, object]) -> list[dict[str, object]]:
    constraints = selection.get("constraints", [])
    if not isinstance(constraints, list):
        raise FeatureAtlasBuildError("selection constraints are invalid")
    failures = []
    for item in constraints:
        if not isinstance(item, Mapping) or type(item.get("passed")) is not bool:
            raise FeatureAtlasBuildError("selection constraint is invalid")
        if item["passed"] is False:
            failures.append(dict(item))
    return failures


def _llm_references(llm: Mapping[str, object] | None) -> dict[str, object]:
    if llm is None:
        return {
            "agreement_path": None,
            "provider_status_paths": {},
            "train_packet_path": None,
            "validation_packet_path": None,
        }
    return {
        "agreement_path": llm.get("agreement_path"),
        "provider_status_paths": _clone(llm.get("provider_status_paths", {})),
        "train_packet_path": llm.get("train_packet_path"),
        "validation_packet_path": llm.get("validation_packet_path"),
    }


def _card_provider_mode(llm: Mapping[str, object] | None) -> str:
    if llm is None:
        return PROVIDER_MODE_DUAL
    statuses = llm.get("provider_statuses")
    if not isinstance(statuses, Mapping):
        raise FeatureAtlasBuildError("card LLM provider statuses are invalid")
    if statuses.get("gpt") == "completed":
        if statuses != {
            "gpt": "completed",
            "claude": "skipped_by_user",
            "gemini": "skipped_by_user",
        }:
            raise FeatureAtlasBuildError("card GPT-only provider statuses are invalid")
        return PROVIDER_MODE_GPT_ONLY
    if statuses.get("claude") == "completed":
        return (
            PROVIDER_MODE_CLAUDE_ONLY
            if statuses.get("gemini") == "skipped_by_user"
            else PROVIDER_MODE_DUAL
        )
    raise FeatureAtlasBuildError("card LLM provider family is invalid")


def build_evidence_card(
    atlas_row: Mapping[str, object],
    selection: Mapping[str, object],
    spans: Mapping[str, object],
    llm: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one exact-schema card from frozen Task 8/9/10 evidence."""

    if not all(isinstance(value, Mapping) for value in (atlas_row, selection, spans)):
        raise TypeError("atlas_row, selection, and spans must be mappings")
    if llm is not None and not isinstance(llm, Mapping):
        raise TypeError("llm must be a mapping or None")
    feature = _feature(atlas_row.get("feature_index"), "atlas")
    if any(
        _feature(value.get("feature_index"), label) != feature
        for label, value in (("selection", selection), ("span", spans))
    ):
        raise FeatureAtlasBuildError("card inputs have different feature identities")
    if llm is not None and _feature(llm.get("feature_index"), "LLM") != feature:
        raise FeatureAtlasBuildError("card LLM feature identity differs")
    category = selection.get("category")
    if category not in {"biological", "field_detector", "control"}:
        raise FeatureAtlasBuildError("cards may be built only for selected features")
    if atlas_row.get("selection_category") != category:
        raise FeatureAtlasBuildError("atlas and frozen selection category differ")
    if atlas_row.get("selection_reason") != selection.get("reason"):
        raise FeatureAtlasBuildError("atlas and frozen selection reason differ")
    classification = atlas_row.get("classification")
    masking_grade = spans.get("evidence_grade")
    if classification not in _STATISTICAL_CLASSES:
        raise FeatureAtlasBuildError("statistical classification is invalid")
    if masking_grade not in _MASKING_GRADES:
        raise FeatureAtlasBuildError("masking evidence grade is invalid")
    heldout = atlas_row.get("heldout_concept_results", [])
    grade = _statistical_grade(heldout)
    counterevidence = {
        "failed_selection_checks": _selection_failures(selection),
        "failed_masking_checks": _clone(
            spans.get("metrics", {}).get("failed_checks", [])
            if isinstance(spans.get("metrics"), Mapping)
            else []
        ),
        "unconfirmed_heldout_results": [
            _clone(item)
            for item in heldout
            if isinstance(item, Mapping) and item.get("confirmed") is not True
        ],
        "llm_disagreement": (
            _clone(llm.get("agreement", {}))
            if llm is not None
            and isinstance(llm.get("agreement"), Mapping)
            and llm.get("agreement", {}).get("llm_consensus") is False
            else {}
        ),
    }
    card = {
        "schema_version": SCHEMA_VERSION,
        "feature_index": feature,
        "identity": {
            "feature_index": feature,
            "selection_fingerprint": atlas_row.get("selection_fingerprint"),
        },
        "provenance": _clone(atlas_row.get("source_provenance", {})),
        "frozen_selection": {
            "category": category,
            "reason": selection.get("reason"),
            "constraints": _clone(selection.get("constraints", [])),
        },
        "top_texts": _clone(atlas_row.get("top_texts", {})),
        "top_proteins": _clone(atlas_row.get("top_proteins", {})),
        "contrasts": _clone(atlas_row.get("contrasts", [])),
        "field_diagnostics": {
            name: atlas_row.get(name)
            for name in (
                "preferred_aspect",
                "source_purity",
                "source_lift",
                "source_effect_loss",
            )
        },
        "source_exclusion": {
            "primary_view": "source_excluded",
            "same_source_primary_allowed": False,
        },
        "train_concept_results": _clone(atlas_row.get("train_concept_results", [])),
        "heldout_concept_results": _clone(heldout),
        "splitting_diagnostics": _clone(
            atlas_row.get("splitting_diagnostics", {})
        ),
        "absorption_diagnostics": _clone(
            atlas_row.get("absorption_diagnostics", {})
        ),
        "spans": _clone(spans),
        "raw_llm_references": _llm_references(llm),
        "provider_mode": _card_provider_mode(llm),
        "llm_agreement": _clone(llm.get("agreement", {}) if llm else {}),
        "counterevidence": counterevidence,
        "limitations": list(_CARD_LIMITATIONS),
        "statistical_classification": classification,
        "masking_evidence_grade": masking_grade,
        "statistical_evidence_grade": grade,
    }
    checked = _validate_card(card)
    return checked


def _validate_card(card: object) -> dict[str, object]:
    if not isinstance(card, dict) or set(card) != _CARD_KEY_SET:
        raise FeatureAtlasBuildError("evidence card has invalid exact schema")
    feature = _feature(card.get("feature_index"), "card")
    identity = card.get("identity")
    if (
        not isinstance(identity, dict)
        or set(identity) != {"feature_index", "selection_fingerprint"}
        or identity.get("feature_index") != feature
        or not _is_sha256(identity.get("selection_fingerprint"))
    ):
        raise FeatureAtlasBuildError("evidence card identity is invalid")
    frozen = card.get("frozen_selection")
    if (
        not isinstance(frozen, dict)
        or set(frozen) != {"category", "reason", "constraints"}
        or frozen.get("category") not in {"biological", "field_detector", "control"}
        or type(frozen.get("reason")) is not str
    ):
        raise FeatureAtlasBuildError("evidence card selection is invalid")
    provenance = card.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise FeatureAtlasBuildError("evidence card provenance is invalid")
    for name in ("top_texts", "top_proteins"):
        values = card.get(name)
        if (
            not isinstance(values, dict)
            or set(values) != {"train", "validation"}
            or any(not isinstance(values[split], list) for split in ("train", "validation"))
        ):
            raise FeatureAtlasBuildError(f"evidence card {name} split schema is invalid")
        limit = 50 if name == "top_texts" else 20
        if any(len(values[split]) > limit for split in ("train", "validation")):
            raise FeatureAtlasBuildError(
                f"evidence card {name} exceeds bounded top-{limit} contract"
            )
    for name in ("contrasts", "train_concept_results", "heldout_concept_results"):
        if not isinstance(card.get(name), list):
            raise FeatureAtlasBuildError(f"evidence card {name} is invalid")
    if len(card["contrasts"]) > 20:
        raise FeatureAtlasBuildError("evidence card contrasts exceed bounded top-20 contract")
    field = card.get("field_diagnostics")
    if (
        not isinstance(field, dict)
        or set(field)
        != {"preferred_aspect", "source_purity", "source_lift", "source_effect_loss"}
        or type(field.get("preferred_aspect")) is not int
        or any(
            type(field.get(name)) not in (int, float)
            or type(field.get(name)) is bool
            or not math.isfinite(float(field[name]))
            for name in ("source_purity", "source_lift", "source_effect_loss")
        )
    ):
        raise FeatureAtlasBuildError("evidence card field diagnostics are invalid")
    if any(
        not isinstance(card.get(name), dict)
        for name in ("splitting_diagnostics", "absorption_diagnostics", "llm_agreement")
    ):
        raise FeatureAtlasBuildError("evidence card diagnostic object is invalid")
    if card.get("provider_mode") not in {
        PROVIDER_MODE_GPT_ONLY,
        PROVIDER_MODE_CLAUDE_ONLY,
        PROVIDER_MODE_DUAL,
    }:
        raise FeatureAtlasBuildError("evidence card provider mode is invalid")
    span_payload = card.get("spans")
    if (
        not isinstance(span_payload, dict)
        or span_payload.get("feature_index") != feature
        or span_payload.get("evidence_grade") != card.get("masking_evidence_grade")
    ):
        raise FeatureAtlasBuildError("evidence card span identity/grade is invalid")
    references = card.get("raw_llm_references")
    if (
        not isinstance(references, dict)
        or set(references)
        != {
            "agreement_path",
            "provider_status_paths",
            "train_packet_path",
            "validation_packet_path",
        }
        or not isinstance(references.get("provider_status_paths"), dict)
    ):
        raise FeatureAtlasBuildError("evidence card raw LLM references are invalid")
    counterevidence = card.get("counterevidence")
    if (
        not isinstance(counterevidence, dict)
        or set(counterevidence)
        != {
            "failed_selection_checks",
            "failed_masking_checks",
            "unconfirmed_heldout_results",
            "llm_disagreement",
        }
        or any(
            not isinstance(counterevidence[name], list)
            for name in (
                "failed_selection_checks",
                "failed_masking_checks",
                "unconfirmed_heldout_results",
            )
        )
        or not isinstance(counterevidence["llm_disagreement"], dict)
    ):
        raise FeatureAtlasBuildError("evidence card counterevidence is invalid")
    if card.get("statistical_classification") not in _STATISTICAL_CLASSES:
        raise FeatureAtlasBuildError("evidence card statistical class is invalid")
    if card.get("masking_evidence_grade") not in _MASKING_GRADES:
        raise FeatureAtlasBuildError("evidence card masking grade is invalid")
    expected_grade = _statistical_grade(card.get("heldout_concept_results"))
    if card.get("statistical_evidence_grade") != expected_grade:
        raise FeatureAtlasBuildError("evidence card statistical grade does not recompute")
    source_exclusion = card.get("source_exclusion")
    if source_exclusion != {
        "primary_view": "source_excluded",
        "same_source_primary_allowed": False,
    }:
        raise FeatureAtlasBuildError("same-source evidence guard changed")
    limitations = card.get("limitations")
    if limitations != list(_CARD_LIMITATIONS):
        raise FeatureAtlasBuildError("evidence card limitations changed")
    _clone(card)
    _validate_wording(card)
    agreement = card["llm_agreement"]
    if agreement.get("available") is False:
        expected_unavailable = (
            {
                "available": False,
                "llm_consensus": None,
                "reason": "single_gpt_provider",
            }
            if card["provider_mode"] == PROVIDER_MODE_GPT_ONLY
            else unavailable_agreement()
        )
        if agreement != expected_unavailable:
            raise FeatureAtlasBuildError(
                "evidence card unavailable LLM agreement is invalid"
            )
        _validate_no_dual_consensus_wording(card)
    return card


def render_evidence_card_markdown(card: Mapping[str, object]) -> str:
    """Render only a validated card; arbitrary text stays inert Markdown data."""

    checked = _validate_card(dict(card))
    canonical = json.dumps(
        {key: checked[key] for key in CARD_KEYS},
        allow_nan=False,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    escaped = html.escape(canonical, quote=True)
    indented = "\n".join(f"    {line}" for line in escaped.splitlines())
    return (
        f"# SAE feature {checked['feature_index']:04d} evidence card\n\n"
        "Candidate model diagnostic; no claim of biological causality.\n\n"
        "## Validated machine card\n\n"
        f"{indented}\n"
    )


def _layout(paths: AtlasPaths) -> dict[str, Path]:
    output = paths.output_dir
    return {
        "external": output / "external",
        "concepts": output / "concepts",
        "concept_index": output / "concepts" / "concept_index.jsonl",
        "sparse": output / "sparse_activations",
        "proteins": output / "protein_activations",
        "covariates": output / "protein_covariates.jsonl",
        "discovery": output / "concepts",
        "train_candidates": output / "concepts" / "train_candidates.jsonl",
        "discovery_manifest": output / "concepts" / "discovery_manifest.json",
        "heldout": output / "concepts",
        "heldout_results": output / "concepts" / "heldout_results.jsonl",
        "heldout_manifest": output / "concepts" / "heldout_manifest.json",
        "atlas": output / "feature_atlas.jsonl",
        "selection": output / "selection_manifest.json",
        "spans": output / "span_attribution",
        "llm": output / "llm",
        "cards": output / "cards",
        "summary": output / "summary.json",
    }


def _registration_rows(value: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if isinstance(value, dict):
        if {"bytes", "path", "sha256"} <= set(value):
            rows.append(value)
        for item in value.values():
            rows.extend(_registration_rows(item))
    elif isinstance(value, list):
        for item in value:
            rows.extend(_registration_rows(item))
    return rows


def _validate_local_payload(
    path: Path,
    stage: str,
    *,
    manifest: Mapping[str, object] | None = None,
    registration: Mapping[str, object] | None = None,
) -> None:
    suffix = path.suffix.casefold()
    try:
        if suffix == ".json":
            strict_json_load(path)
        elif suffix == ".jsonl":
            rows = list(strict_json_lines(path))
            identities = []
            for row in rows:
                if type(row.get("feature_index")) is int:
                    identities.append((int(row["feature_index"]),))
                elif type(row.get("database")) is str and type(row.get("concept_id")) is str:
                    identities.append((str(row["database"]), str(row["concept_id"])))
            if identities and identities != sorted(set(identities)):
                raise FeatureAtlasBuildError(
                    f"{stage} stage JSONL identities are reordered or duplicated"
                )
        elif suffix == ".pt":
            payload = torch.load(path, map_location="cpu", weights_only=True)

            def finite(item: object) -> None:
                if isinstance(item, torch.Tensor) and not bool(torch.isfinite(item).all()):
                    raise FeatureAtlasBuildError(
                        f"{stage} stage tensor contains nonfinite values"
                    )
                if isinstance(item, dict):
                    for nested in item.values():
                        finite(nested)
                elif isinstance(item, (list, tuple)):
                    for nested in item:
                        finite(nested)

            finite(payload)
            if stage == "sparse":
                if not isinstance(payload, dict):
                    raise FeatureAtlasBuildError("sparse stage shard is not an object")
                unit_ids = payload.get("unit_ids")
                if not isinstance(unit_ids, (list, tuple)):
                    raise FeatureAtlasBuildError("sparse stage unit IDs are invalid")
                _validate_sparse_shard(
                    payload,
                    start_row=int(payload.get("start_row", -1)),
                    end_row=int(payload.get("end_row", -1)),
                    expected_ids=tuple(unit_ids),
                    width=int((manifest or {}).get("width", -1)),
                    k=int((manifest or {}).get("k", -1)),
                    label=str(path),
                )
            elif stage == "proteins":
                if not isinstance(payload, dict) or registration is None:
                    raise FeatureAtlasBuildError("proteins stage shard is invalid")
                _decode_output_shard(
                    payload,
                    start=int(registration.get("start_protein", -1)),
                    end=int(registration.get("end_protein", -1)),
                    width=int((manifest or {}).get("width", -1)),
                )
        if stage == "spans" and path.name.startswith("feature_") and suffix == ".json":
            validate_feature_payload(strict_json_load(path))
    except FeatureAtlasBuildError:
        raise
    except Exception as error:
        raise FeatureAtlasBuildError(
            f"{stage} stage registered payload schema is invalid: {path.name}: {error}"
        ) from error


def _validate_registered_stage_bytes(
    directory: Path,
    stage: str,
    *,
    manifest_name: str = "manifest.json",
    require_complete: bool = True,
) -> dict[str, object]:
    """Validate all manifest-registered local bytes without model construction."""

    try:
        manifest = strict_json_load(directory / manifest_name)
    except (AtlasIOError, OSError, ValueError) as error:
        raise FeatureAtlasBuildError(f"{stage} stage manifest corruption: {error}") from error
    if type(manifest.get("schema_version")) is not int:
        raise FeatureAtlasBuildError(f"{stage} stage manifest schema_version is invalid")
    expected_keys = STAGE_MANIFEST_KEYS_BY_STAGE.get(stage)
    if expected_keys is not None and set(manifest) != expected_keys:
        raise FeatureAtlasBuildError(f"{stage} stage manifest exact schema is invalid")
    if require_complete and manifest.get("complete") is not True:
        raise FeatureAtlasBuildError(f"{stage} stage is incomplete")
    if stage == "spans":
        registrations = [
            row
            for key in ("provisional_outputs", "final_outputs", "cache")
            for row in _registration_rows(manifest.get(key))
        ]
    elif stage == "llm":
        registrations = _registration_rows(manifest.get("artifacts"))
    else:
        registrations = _registration_rows(manifest)
    if not registrations and (require_complete or manifest.get("complete") is True):
        if not (stage == "spans" and manifest.get("features") == []):
            raise FeatureAtlasBuildError(f"{stage} stage has no registered outputs")
    seen: set[str] = set()
    for row in registrations:
        relative = row.get("path")
        size = row.get("bytes")
        checksum = row.get("sha256")
        if (
            type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen
            or type(size) is not int
            or size <= 0
            or not _is_sha256(checksum)
        ):
            raise FeatureAtlasBuildError(f"{stage} stage registration schema is invalid")
        seen.add(relative)
        path = directory / relative
        if path.is_symlink():
            raise FeatureAtlasBuildError(
                f"{stage} stage registered output is a symlink: {relative}"
            )
        try:
            metadata = file_metadata(path)
        except (AtlasIOError, OSError, ValueError) as error:
            raise FeatureAtlasBuildError(
                f"{stage} stage registered output is invalid: {relative}: {error}"
            ) from error
        if metadata["bytes"] != size or metadata["sha256"] != checksum:
            raise FeatureAtlasBuildError(
                f"{stage} stage registered output checksum/size differs: {relative}"
            )
        _validate_local_payload(
            path, stage, manifest=manifest, registration=row
        )
    return manifest


def _resume_registered_stage(
    stage: str, directory: Path, *, manifest_name: str = "manifest.json"
) -> dict[str, object]:
    return _validate_registered_stage_bytes(
        directory, stage, manifest_name=manifest_name, require_complete=True
    )


def _validate_sparse_validation_record(paths: AtlasPaths) -> dict[str, object]:
    layout = _layout(paths)
    record_path = paths.output_dir / "sparse_validation.json"
    try:
        record = strict_json_load(record_path)
    except (AtlasIOError, OSError, ValueError) as error:
        raise FeatureAtlasBuildError(f"sparse stage validation record is invalid: {error}") from error
    if (
        set(record)
        != {
            "direct_forward_validated",
            "feature_stats_sha256",
            "schema_version",
            "sparse_manifest_sha256",
        }
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("direct_forward_validated") is not True
        or record.get("feature_stats_sha256")
        != sha256_file(paths.run_dir / "feature_stats.pt")
        or record.get("sparse_manifest_sha256")
        != sha256_file(layout["sparse"] / "manifest.json")
    ):
        raise FeatureAtlasBuildError(
            "sparse stage direct-forward record is stale or not bound to current shards"
        )
    return record


def _validate_task10_family(
    selection_manifest: Mapping[str, object], llm_manifest: Mapping[str, object]
) -> None:
    selected = selection_manifest.get("selected")
    if not isinstance(selected, list):
        raise FeatureAtlasBuildError("llm stage selection family is invalid")
    features_in_selection_order = [
        item.get("feature_index") for item in selected if isinstance(item, Mapping)
    ]
    if (
        len(features_in_selection_order) != len(selected)
        or len(features_in_selection_order) != len(set(features_in_selection_order))
    ):
        raise FeatureAtlasBuildError("llm stage selection family/order is invalid")
    features = sorted(features_in_selection_order)
    if llm_manifest.get("family_order") != features:
        raise FeatureAtlasBuildError("llm stage family_order differs from selection")
    artifacts = llm_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise FeatureAtlasBuildError("llm stage artifact registry is invalid")
    final_features = sorted(
        item.get("feature_index")
        for item in artifacts
        if isinstance(item, Mapping) and item.get("kind") == "feature"
    )
    if llm_manifest.get("complete") is True and final_features != features:
        raise FeatureAtlasBuildError("llm stage final feature family differs")
    if any(type(feature) is not int for feature in final_features):
        raise FeatureAtlasBuildError("llm stage final feature identity is invalid")


def _preflight_span_plans(
    atlas_rows: Mapping[int, Mapping[str, object]],
    features: Sequence[int],
    embedder: object,
    cache_dir: Path,
    *,
    seed: int,
    permutations: int,
    max_encodings: int,
) -> dict[str, object]:
    plans = [
        build_feature_plan(
            feature,
            {
                split: select_feature_texts(atlas_rows[feature], split)
                for split in ("train", "validation")
            },
            embedder,
            seed=seed,
            permutations=permutations,
        )
        for feature in features
    ]
    cache = EmbeddingCache(cache_dir)
    try:
        return preflight_feature_plans(
            plans, cache, max_encodings=max_encodings
        )
    except Exception as error:
        raise FeatureAtlasBuildError(f"spans preflight hard limit failed: {error}") from error


def _require_complete_core(
    stages: Mapping[str, object],
    atlas_rows: Sequence[Mapping[str, object]],
    *,
    expected_width: int,
) -> None:
    for stage in CORE_STAGES:
        if stage not in stages:
            raise FeatureAtlasBuildError(f"{stage} stage is missing from statistical core")
    if (
        expected_width != 8192
        or len(atlas_rows) != 8192
        or [row.get("feature_index") for row in atlas_rows] != list(range(8192))
    ):
        raise FeatureAtlasBuildError(
            "selection stage statistical core requires exact 8192 atlas records"
        )
    sparse = stages.get("sparse")
    if (
        not isinstance(sparse, Mapping)
        or sparse.get("row_count") != 243_947
        or sparse.get("width") != 8192
        or sparse.get("k") != 32
    ):
        raise FeatureAtlasBuildError(
            "sparse stage requires exact 243947 rows, width 8192, and k 32"
        )
    proteins = stages.get("proteins")
    if (
        not isinstance(proteins, Mapping)
        or proteins.get("accession_count") != 10_618
        or proteins.get("width") != 8192
        or proteins.get("split_counts") != {"train": 9539, "validation": 1079}
    ):
        raise FeatureAtlasBuildError(
            "proteins stage requires exact 10618 accessions and frozen split counts"
        )
    selection = stages.get("selection")
    cards = stages.get("cards")
    selected = selection.get("selected") if isinstance(selection, Mapping) else None
    card_features = cards.get("features") if isinstance(cards, Mapping) else None
    if (
        not isinstance(selected, list)
        or not isinstance(card_features, list)
        or len(selected) > 200
        or cards.get("feature_count") != len(card_features)
        or card_features != [
            item.get("feature_index") for item in selected if isinstance(item, Mapping)
        ]
    ):
        raise FeatureAtlasBuildError(
            "cards stage count/family differs from frozen selected feature count"
        )


def _is_exact_legacy_gpt_root_config(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    provider_configuration: Mapping[str, object],
    provider_mode: str,
) -> bool:
    placeholder = {
        "gpt": {"requested": GPT_MODEL, "resolved": GPT_MODEL},
        "claude": {"status": "skipped_by_user"},
        "gemini": {"status": "skipped_by_user"},
    }
    legacy_keys = {"config", "config_fingerprint", "schema_version", "smoke"}
    legacy_expected = {
        key: value
        for key, value in expected.items()
        if key not in {"provider_configuration", "provider_mode"}
    }
    return (
        provider_mode == PROVIDER_MODE_GPT_ONLY
        and dict(provider_configuration) == placeholder
        and set(observed) == legacy_keys
        and dict(observed) == legacy_expected
    )


def _publish_root_manifests(
    config: FeatureAtlasConfig,
    output_dir: Path,
    source_binding: Mapping[str, object],
    *,
    provider_models: Mapping[str, object],
    provider_mode: str | None = None,
    allow_provider_rebind: bool = False,
    smoke_features: int | None,
    smoke_databases: Sequence[str],
) -> dict[str, object]:
    if set(source_binding) != set(AtlasSourceBinding.__dataclass_fields__) or any(
        not _is_sha256(value) for value in source_binding.values()
    ):
        raise FeatureAtlasBuildError("source_manifest source binding is invalid")
    databases = list(smoke_databases)
    if databases != sorted(set(databases)) or not set(databases) <= {
        "go", "reactome", "interpro", "corum"
    }:
        raise FeatureAtlasBuildError("config smoke database filter is invalid")
    provider_configuration = _clone(dict(provider_models))
    if not isinstance(provider_configuration, dict) or set(provider_configuration) not in (
        {"claude", "gemini"},
        {"gpt", "claude", "gemini"},
    ):
        raise FeatureAtlasBuildError("root provider configuration is invalid")
    mode = provider_mode or (
        PROVIDER_MODE_CLAUDE_ONLY
        if provider_configuration.get("gemini") == {"status": "skipped_by_user"}
        else PROVIDER_MODE_DUAL
    )
    if mode not in {
        PROVIDER_MODE_GPT_ONLY, PROVIDER_MODE_CLAUDE_ONLY, PROVIDER_MODE_DUAL
    }:
        raise FeatureAtlasBuildError("root provider mode is invalid")
    if (
        mode == PROVIDER_MODE_GPT_ONLY
        and (
            set(provider_configuration) != {"gpt", "claude", "gemini"}
            or provider_configuration.get("claude") != {"status": "skipped_by_user"}
            or provider_configuration.get("gemini") != {"status": "skipped_by_user"}
        )
    ) or (
        mode == PROVIDER_MODE_CLAUDE_ONLY
        and (
            set(provider_configuration) != {"claude", "gemini"}
            or provider_configuration.get("gemini") != {"status": "skipped_by_user"}
        )
    ) or (
        mode == PROVIDER_MODE_DUAL
        and (
            set(provider_configuration) != {"claude", "gemini"}
            or provider_configuration.get("gemini") == {"status": "skipped_by_user"}
        )
    ):
        raise FeatureAtlasBuildError("root provider mode/configuration disagree")
    config_payload = {
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "provider_configuration": provider_configuration,
        "provider_mode": mode,
        "schema_version": SCHEMA_VERSION,
        "smoke": {
            "databases": databases,
            "feature_count": smoke_features,
            "selected_feature_cap": 2 if smoke_features is not None else None,
        },
    }
    source_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_binding": dict(source_binding),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "config.json": config_payload,
        "source_manifest.json": source_payload,
    }
    active_names = (
        ("gpt",)
        if mode == PROVIDER_MODE_GPT_ONLY
        else
        ("claude",)
        if mode == PROVIDER_MODE_CLAUDE_ONLY else ("claude", "gemini")
    )
    config_path = output_dir / "config.json"
    source_path = output_dir / "source_manifest.json"
    binding_path = output_dir / "provider_bindings.json"
    legacy_gpt_promotion = False
    if (
        (config_path.exists() or config_path.is_symlink())
        and (source_path.exists() or source_path.is_symlink())
        and not binding_path.exists()
        and not binding_path.is_symlink()
    ):
        try:
            observed_config = strict_json_load(config_path)
            observed_source = strict_json_load(source_path)
        except (AtlasIOError, OSError, ValueError):
            pass
        else:
            legacy_gpt_promotion = (
                _is_exact_legacy_gpt_root_config(
                    observed_config,
                    config_payload,
                    provider_configuration,
                    mode,
                )
                and observed_source == source_payload
            )
    for name, payload in expected.items():
        path = output_dir / name
        if path.exists() or path.is_symlink():
            try:
                observed = strict_json_load(path)
            except (AtlasIOError, OSError, ValueError) as error:
                raise FeatureAtlasBuildError(f"{name} corruption: {error}") from error
            if observed != payload:
                rebindable = dict(observed)
                observed_providers = observed.get("provider_configuration")
                placeholder_promotion = (
                    name == "config.json"
                    and observed.get("provider_mode") == mode
                    and isinstance(observed_providers, Mapping)
                    and (
                        mode == PROVIDER_MODE_DUAL
                        or all(
                            observed_providers.get(name)
                            == {"status": "skipped_by_user"}
                            for name in (
                                ("claude", "gemini")
                                if mode == PROVIDER_MODE_GPT_ONLY else ("gemini",)
                            )
                        )
                    )
                    and all(
                        observed_providers.get(provider) == {
                            "requested": (
                                GPT_MODEL if provider == "gpt"
                                else CLAUDE_MODEL if provider == "claude"
                                else GEMINI_MODEL
                            ),
                            "resolved": (
                                GPT_MODEL if provider == "gpt"
                                else CLAUDE_MODEL if provider == "claude"
                                else GEMINI_MODEL
                            ),
                        }
                        and isinstance(provider_configuration.get(provider), Mapping)
                        and provider_configuration[provider].get("provider") == provider
                        and _is_sha256(
                            provider_configuration[provider].get("binding_sha256")
                        )
                        for provider in active_names
                    )
                )
                if name == "config.json" and (
                    allow_provider_rebind
                    or placeholder_promotion
                    or legacy_gpt_promotion
                ):
                    rebindable["provider_configuration"] = provider_configuration
                    rebindable["provider_mode"] = mode
                if rebindable != payload:
                    raise FeatureAtlasBuildError(f"{name} differs from requested run")
                write_json_atomic(path, payload)
        else:
            write_json_atomic(path, payload)
        if allow_provider_rebind and name == "config.json":
            _transaction_event("root_config_replaced")
    bindings = provider_configuration
    if all(
        isinstance(bindings.get(name), Mapping)
        and _is_sha256(bindings[name].get("binding_sha256"))
        for name in active_names
    ):
        binding_path = output_dir / "provider_bindings.json"
        binding_payload = {
            "bindings": bindings,
            "provider_mode": mode,
            "schema_version": SCHEMA_VERSION,
        }
        if binding_path.exists() or binding_path.is_symlink():
            if strict_json_load(binding_path) != binding_payload:
                if not allow_provider_rebind:
                    raise FeatureAtlasBuildError(
                        "provider_bindings.json is frozen and cannot be mutated"
                    )
                write_json_atomic(binding_path, binding_payload)
        else:
            write_json_atomic(binding_path, binding_payload)
    return {name: file_metadata(output_dir / name) for name in sorted(expected)}


def _preflight_provider(
    provider: str, command: str, model: str
) -> object:
    candidate = shutil.which(command) or command
    resolved = Path(candidate).expanduser().resolve(strict=True)
    if not resolved.is_absolute() or not resolved.is_file() or resolved.is_symlink():
        raise FeatureAtlasBuildError(
            f"{provider} provider executable does not resolve to a regular realpath"
        )
    try:
        completed = subprocess.run(
            [str(resolved), "--version"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FeatureAtlasBuildError(f"{provider} provider preflight failed: {error}") from error
    version = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or not version:
        raise FeatureAtlasBuildError(
            f"{provider} provider --version failed with code {completed.returncode}"
        )
    sentinel = "FEATURE_ATLAS_PROVIDER_PREFLIGHT_OK_7f3d9a"
    prompt = (
        "Headless availability check. Return exactly this text and nothing else: "
        + sentinel
    )
    if provider == "claude" and model == CLAUDE_MODEL:
        preflight_argv = [
            str(resolved), "--print", "--model", model,
            "--output-format", "json", "--no-session-persistence",
            "--tools", "", "--max-budget-usd", "0.01", prompt,
        ]
    elif provider == "gemini" and model == GEMINI_MODEL:
        preflight_argv = [
            str(resolved), "-m", model, "-p", prompt,
            "--output-format", "json",
        ]
    else:
        raise FeatureAtlasBuildError(f"{provider} provider/model pair is not frozen")
    try:
        metadata_call = subprocess.run(
            preflight_argv,
            shell=False, capture_output=True, text=True, timeout=120.0, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FeatureAtlasBuildError(
            f"{provider} exact-model headless preflight failed: {error}"
        ) from error
    if metadata_call.returncode != 0:
        raise FeatureAtlasBuildError(
            f"{provider} exact-model headless preflight failed with code "
            f"{metadata_call.returncode}"
        )
    try:
        stripped = metadata_call.stdout.strip()
        if not stripped:
            raise ValueError("empty provider response")
        try:
            parsed = json.loads(stripped)
            documents = [parsed]
        except json.JSONDecodeError:
            documents = [json.loads(line) for line in stripped.splitlines() if line.strip()]
        if not documents or any(not isinstance(item, dict) for item in documents):
            raise ValueError("provider response must contain JSON objects")
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise FeatureAtlasBuildError(
            f"{provider} exact-model headless preflight returned invalid JSON: {error}"
        ) from error

    actual_models: set[str] = set()
    response_values: list[str] = []
    provider_versions: list[str] = []
    for document in documents:
        assert isinstance(document, dict)
        for key in ("result", "response"):
            value = document.get(key)
            if type(value) is str:
                response_values.append(value.strip())
        top_model = document.get("model")
        if type(top_model) is str and top_model:
            actual_models.add(top_model)
        for key in ("providerVersion", "provider_version", "apiVersion", "version"):
            value = document.get(key)
            if type(value) is str and value:
                provider_versions.append(value)
        if provider == "claude":
            usage = document.get("modelUsage")
            if isinstance(usage, Mapping):
                actual_models.update(
                    key for key in usage if type(key) is str and key
                )
        else:
            stats = document.get("stats")
            if isinstance(stats, Mapping):
                stats_model = stats.get("model")
                if type(stats_model) is str and stats_model:
                    actual_models.add(stats_model)
                for key in ("models", "modelUsage", "model_usage"):
                    usage = stats.get(key)
                    if isinstance(usage, Mapping):
                        actual_models.update(
                            name for name in usage if type(name) is str and name
                        )
    if response_values != [sentinel]:
        raise FeatureAtlasBuildError(
            f"{provider} exact-model headless preflight sentinel differs"
        )
    if actual_models != {model}:
        raise FeatureAtlasBuildError(
            f"{provider} exact model is unavailable or response identity differs"
        )
    reported_version = (
        provider_versions[0]
        if len(set(provider_versions)) == 1
        else model
        if not provider_versions
        else None
    )
    if reported_version is None:
        raise FeatureAtlasBuildError(
            f"{provider} provider response version metadata is ambiguous"
        )
    kwargs = {
        "executable": str(resolved),
        "resolved_executable": str(resolved),
        "cli_version": version,
        "reported_version": reported_version,
    }
    result = (
        ClaudeCLIProvider(**kwargs)
        if provider == "claude" and model == CLAUDE_MODEL
        else GeminiCLIProvider(**kwargs)
        if provider == "gemini" and model == GEMINI_MODEL
        else None
    )
    if result is None:
        raise FeatureAtlasBuildError(f"{provider} provider/model pair is not frozen")
    binding = result.binding()
    if (
        binding.get("reported_model") != model
        or binding.get("reported_version") == "unrecorded"
        or not _is_sha256(binding.get("executable_sha256"))
    ):
        raise FeatureAtlasBuildError(f"{provider} provider binding is incomplete")
    return result


def _smoke_plan_from_root(output_dir: Path) -> SmokePlan | None:
    path = output_dir / "config.json"
    if not path.exists():
        return None
    payload = strict_json_load(path)
    smoke = payload.get("smoke")
    if not isinstance(smoke, Mapping) or smoke.get("feature_count") is None:
        return None
    databases = smoke.get("databases")
    if not isinstance(databases, list):
        raise FeatureAtlasBuildError("config.json smoke databases are invalid")
    return SmokePlan(
        feature_count=int(smoke["feature_count"]),
        databases=tuple(databases),
    )


def _validate_root_provider_configuration(output_dir: Path) -> str:
    config = strict_json_load(output_dir / "config.json")
    mode = config.get("provider_mode")
    providers = config.get("provider_configuration")
    if (
        mode not in {
            PROVIDER_MODE_GPT_ONLY, PROVIDER_MODE_CLAUDE_ONLY, PROVIDER_MODE_DUAL
        }
        or not isinstance(providers, dict)
        or set(providers) != (
            {"gpt", "claude", "gemini"}
            if mode == PROVIDER_MODE_GPT_ONLY else {"claude", "gemini"}
        )
        or (
            mode == PROVIDER_MODE_GPT_ONLY
            and (
                providers.get("claude") != {"status": "skipped_by_user"}
                or providers.get("gemini") != {"status": "skipped_by_user"}
            )
        )
        or (
            mode == PROVIDER_MODE_CLAUDE_ONLY
            and providers.get("gemini") != {"status": "skipped_by_user"}
        )
        or (
            mode == PROVIDER_MODE_DUAL
            and providers.get("gemini") == {"status": "skipped_by_user"}
        )
    ):
        raise FeatureAtlasBuildError("root provider configuration is invalid")
    binding_path = output_dir / "provider_bindings.json"
    if binding_path.exists():
        bindings = strict_json_load(binding_path)
        if bindings != {
            "bindings": providers,
            "provider_mode": mode,
            "schema_version": SCHEMA_VERSION,
        }:
            raise FeatureAtlasBuildError(
                "root provider bindings differ from provider configuration"
            )
    return str(mode)


def _artifact_provider_mode(paths: AtlasPaths) -> str:
    modes: list[str] = []
    if (paths.output_dir / "config.json").exists():
        modes.append(_validate_root_provider_configuration(paths.output_dir))
    llm_manifest_path = _layout(paths)["llm"] / "manifest.json"
    if llm_manifest_path.exists():
        llm_mode = strict_json_load(llm_manifest_path).get("provider_mode")
        if llm_mode not in {
            PROVIDER_MODE_GPT_ONLY, PROVIDER_MODE_CLAUDE_ONLY, PROVIDER_MODE_DUAL
        }:
            raise FeatureAtlasBuildError("LLM provider mode is invalid")
        modes.append(str(llm_mode))
    if len(set(modes)) > 1:
        raise FeatureAtlasBuildError("root and LLM provider modes differ")
    return modes[0] if modes else PROVIDER_MODE_DUAL


def _validate_root_llm_provider_bindings(
    paths: AtlasPaths, llm_manifest: Mapping[str, object]
) -> None:
    mode = _validate_root_provider_configuration(paths.output_dir)
    binding_path = paths.output_dir / "provider_bindings.json"
    if not binding_path.is_file() or binding_path.is_symlink():
        raise FeatureAtlasBuildError(
            "root provider binding manifest is required for LLM artifacts"
        )
    root = strict_json_load(binding_path)
    root_bindings = root.get("bindings")
    llm_bindings = llm_manifest.get("provider_bindings")
    if not isinstance(root_bindings, dict) or not isinstance(llm_bindings, list):
        raise FeatureAtlasBuildError("root/LLM provider bindings are invalid")
    active_names = (
        ("gpt",)
        if mode == PROVIDER_MODE_GPT_ONLY
        else
        ("claude",)
        if mode == PROVIDER_MODE_CLAUDE_ONLY else ("claude", "gemini")
    )
    llm_by_name = {
        str(row.get("provider")): row
        for row in llm_bindings
        if isinstance(row, dict)
    }
    if set(llm_by_name) != set(active_names) or any(
        llm_by_name[name] != root_bindings.get(name) for name in active_names
    ):
        raise FeatureAtlasBuildError(
            "root and LLM active provider bindings differ"
        )


def _stage_artifact_exists(stage: str, layout: Mapping[str, Path]) -> bool:
    if stage == "concepts":
        return (layout["concepts"] / "manifest.json").exists()
    if stage == "discovery":
        return layout["discovery_manifest"].exists()
    if stage == "heldout":
        return layout["heldout_manifest"].exists()
    if stage == "selection":
        return layout["selection"].exists()
    if stage in {"external", "sparse", "proteins", "spans", "llm", "cards"}:
        return (layout[stage] / "manifest.json").exists()
    return False


def _resume_stage(
    stage: str,
    config: FeatureAtlasConfig,
    paths: AtlasPaths,
) -> dict[str, object] | None:
    layout = _layout(paths)
    if not _stage_artifact_exists(stage, layout):
        return None
    if stage in {"sparse", "proteins", "spans", "llm"}:
        manifest = _validate_registered_stage_bytes(
            layout[stage], stage, require_complete=False
        )
        if (
            stage == "llm"
            and manifest.get("provider_mode")
            != _validate_root_provider_configuration(paths.output_dir)
        ):
            raise FeatureAtlasBuildError(
                "llm resume provider mode differs from root configuration"
            )
        if stage == "llm":
            _validate_root_llm_provider_bindings(paths, manifest)
        if manifest.get("complete") is not True:
            return None
    return _validate_completed_stage_deep(stage, config, paths, device="cpu")


def _read_atlas(path: Path) -> list[dict[str, object]]:
    try:
        rows = list(strict_json_lines(path))
    except (AtlasIOError, OSError, ValueError) as error:
        raise FeatureAtlasBuildError("selection atlas is invalid") from error
    features = [row.get("feature_index") for row in rows]
    if any(type(feature) is not int for feature in features) or features != sorted(set(features)):
        raise FeatureAtlasBuildError("selection atlas rows are not sorted and unique")
    return rows


def _registered_payload(directory: Path, feature: int, stage: str) -> dict[str, object]:
    path = directory / f"feature_{feature:04d}.json"
    try:
        payload = strict_json_load(path)
    except (AtlasIOError, OSError, ValueError) as error:
        raise FeatureAtlasBuildError(f"{stage} feature {feature} is invalid") from error
    if payload.get("feature_index") != feature:
        raise FeatureAtlasBuildError(f"{stage} feature identity differs")
    return payload


def _validate_cards_root_llm_provider_bindings(paths: AtlasPaths) -> None:
    llm_directory = _layout(paths)["llm"]
    llm_manifest_path = llm_directory / "manifest.json"
    root_mode = _validate_root_provider_configuration(paths.output_dir)
    reserved_artifacts = (
        [
            path
            for path in llm_directory.iterdir()
            if _LLM_RESERVED_ARTIFACT.fullmatch(path.name)
        ]
        if llm_directory.is_dir() and not llm_directory.is_symlink()
        else []
    )
    if not llm_manifest_path.is_file() or llm_manifest_path.is_symlink():
        if root_mode == PROVIDER_MODE_GPT_ONLY or reserved_artifacts:
            raise FeatureAtlasBuildError(
                "complete LLM manifest is required before cards consumption"
            )
        return
    llm_manifest = strict_json_load(llm_manifest_path)
    if llm_manifest.get("complete") is not True:
        raise FeatureAtlasBuildError(
            "LLM manifest must be complete before cards consumption"
        )
    _validate_root_llm_provider_bindings(paths, llm_manifest)
    if root_mode == PROVIDER_MODE_GPT_ONLY:
        artifacts = llm_manifest.get("artifacts")
        identities = llm_manifest.get("statistical_identities")
        if not isinstance(artifacts, list) or not isinstance(identities, list):
            raise FeatureAtlasBuildError("GPT LLM feature family is invalid")
        expected_features = {
            row.get("feature_index")
            for row in identities
            if isinstance(row, Mapping) and type(row.get("feature_index")) is int
        }
        feature_rows = [
            row
            for row in artifacts
            if isinstance(row, Mapping) and row.get("kind") == "feature"
        ]
        feature_paths = {
            row.get("feature_index"): llm_directory / str(row.get("path"))
            for row in feature_rows
            if type(row.get("feature_index")) is int
            and type(row.get("path")) is str
        }
        if (
            len(expected_features) != len(identities)
            or set(feature_paths) != expected_features
            or any(
                not path.is_file() or path.is_symlink()
                for path in feature_paths.values()
            )
        ):
            raise FeatureAtlasBuildError(
                "GPT LLM feature family is incomplete before cards consumption"
            )


def _validate_completed_upstreams_for_cards(
    config: FeatureAtlasConfig, paths: AtlasPaths, device: str = "cpu"
) -> None:
    layout = _layout(paths)
    try:
        _validate_cards_root_llm_provider_bindings(paths)
        selection = strict_json_load(layout["selection"])
        provenance = selection.get("provenance")
        validate_selection_manifest(
            layout["selection"],
            layout["atlas"],
            paths.run_dir / "inference.pt",
            config,
            provenance=provenance if isinstance(provenance, Mapping) else None,
        )
        runtime = _build_llm_runtime(config, paths, device=device)
        span_manifest = validate_span_attribution(
            layout["spans"],
            current_bindings=runtime["context"]["bindings"],
            atlas_rows=runtime["context"]["rows"],
            sae=runtime["sae"],
            plans_by_feature=runtime["plans"],
        )
        if (layout["llm"] / "manifest.json").exists():
            validate_llm_feature_descriptions(
                layout["llm"],
                label_embedder=runtime["embedder"],
                selection_path=layout["selection"],
                atlas_path=layout["atlas"],
                span_output_dir=layout["spans"],
                inference_path=paths.run_dir / "inference.pt",
                config=config,
                provenance=selection.get("provenance"),
                current_bindings=runtime["context"]["bindings"],
                sae=runtime["sae"],
                plans_by_feature=runtime["plans"],
            )
        if span_manifest.get("complete") is not True:
            raise FeatureAtlasBuildError("spans stage is incomplete")
    except FeatureAtlasBuildError:
        raise
    except Exception as error:
        raise FeatureAtlasBuildError(f"cards upstream validation failed: {error}") from error


def _card_output_metadata(path: Path, feature: int, kind: str) -> dict[str, object]:
    metadata = file_metadata(path)
    return {
        "bytes": metadata["bytes"],
        "feature_index": feature,
        "kind": kind,
        "path": path.name,
        "sha256": metadata["sha256"],
    }


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _validate_cards_stage(cards_dir: Path) -> dict[str, object]:
    try:
        manifest = strict_json_load(cards_dir / "manifest.json")
    except (AtlasIOError, OSError, ValueError) as error:
        raise FeatureAtlasBuildError(f"cards manifest is invalid: {error}") from error
    expected_keys = {
        "complete",
        "feature_count",
        "features",
        "outputs",
        "provider_mode",
        "schema_version",
        "selection_fingerprint",
    }
    if set(manifest) != expected_keys or manifest.get("schema_version") != SCHEMA_VERSION:
        raise FeatureAtlasBuildError("cards manifest has invalid exact schema")
    features = manifest.get("features")
    outputs = manifest.get("outputs")
    if (
        manifest.get("complete") is not True
        or not isinstance(features, list)
        or len(features) != len(set(features))
        or len(features) > 200
        or manifest.get("feature_count") != len(features)
        or not isinstance(outputs, list)
        or len(outputs) != len(features) * 2
        or manifest.get("provider_mode") not in {
            PROVIDER_MODE_GPT_ONLY, PROVIDER_MODE_CLAUDE_ONLY, PROVIDER_MODE_DUAL
        }
        or not _is_sha256(manifest.get("selection_fingerprint"))
    ):
        raise FeatureAtlasBuildError("cards manifest family/count is invalid")
    expected_pairs = []
    for feature in features:
        _feature(feature, "cards")
        expected_pairs.extend([(feature, "json"), (feature, "markdown")])
    observed_pairs = []
    for row in outputs:
        if (
            not isinstance(row, dict)
            or set(row) != {"bytes", "feature_index", "kind", "path", "sha256"}
            or type(row.get("bytes")) is not int
            or row["bytes"] <= 0
            or not _is_sha256(row.get("sha256"))
            or row.get("kind") not in {"json", "markdown"}
        ):
            raise FeatureAtlasBuildError("cards output registration is invalid")
        feature = _feature(row.get("feature_index"), "cards output")
        suffix = "json" if row["kind"] == "json" else "md"
        expected_name = f"feature_{feature:04d}.{suffix}"
        if row.get("path") != expected_name:
            raise FeatureAtlasBuildError("cards output path is not canonical")
        path = cards_dir / expected_name
        if path.is_symlink() or not path.is_file():
            raise FeatureAtlasBuildError("cards registered output is missing or symlinked")
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise FeatureAtlasBuildError("cards registered output checksum differs")
        if row["kind"] == "json":
            card = strict_json_load(path)
            if _validate_card(card).get("feature_index") != feature:
                raise FeatureAtlasBuildError("cards JSON identity differs")
            markdown_path = cards_dir / f"feature_{feature:04d}.md"
            if markdown_path.exists():
                expected_markdown = render_evidence_card_markdown(card)
                if markdown_path.read_text(encoding="utf-8") != expected_markdown:
                    raise FeatureAtlasBuildError("cards Markdown does not correspond to JSON")
        observed_pairs.append((feature, row["kind"]))
    if observed_pairs != expected_pairs:
        raise FeatureAtlasBuildError("cards outputs are not in exact feature/kind order")
    return manifest


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _validate_cards_against_sources(paths: AtlasPaths) -> dict[str, object]:
    layout = _layout(paths)
    _validate_cards_root_llm_provider_bindings(paths)
    manifest = _validate_cards_stage(layout["cards"])
    try:
        selection_manifest = strict_json_load(layout["selection"])
    except (AtlasIOError, OSError, ValueError) as error:
        raise FeatureAtlasBuildError(f"cards source selection is invalid: {error}") from error
    selected = selection_manifest.get("selected")
    if not isinstance(selected, list):
        raise FeatureAtlasBuildError("cards source selection family is invalid")
    smoke = _smoke_plan_from_root(paths.output_dir)
    if smoke is not None:
        selected = [
            item
            for item in selected
            if isinstance(item, Mapping)
            and item.get("feature_index") in set(smoke.feature_indices)
        ][: smoke.selected_feature_cap]
    features = [
        item.get("feature_index") for item in selected if isinstance(item, Mapping)
    ]
    if (
        len(features) != len(selected)
        or len(features) != len(set(features))
        or manifest.get("features") != features
        or manifest.get("selection_fingerprint")
        != selection_manifest.get("selection_fingerprint")
    ):
        raise FeatureAtlasBuildError(
            "cards family/order/fingerprint differs from frozen selection"
        )
    if manifest.get("provider_mode") != _artifact_provider_mode(paths):
        raise FeatureAtlasBuildError(
            "cards provider mode differs from root/LLM artifacts"
        )
    rows = {int(row["feature_index"]): row for row in _read_atlas(layout["atlas"])}
    for selected_row, feature in zip(selected, features):
        if type(feature) is not int or feature not in rows:
            raise FeatureAtlasBuildError("cards feature is absent from frozen atlas")
        span = _registered_payload(layout["spans"], feature, "spans")
        llm_path = layout["llm"] / f"feature_{feature:04d}.json"
        llm_payload = (
            _registered_payload(layout["llm"], feature, "llm")
            if llm_path.exists()
            else None
        )
        expected = build_evidence_card(
            rows[feature], selected_row, span, llm_payload
        )
        card_path = layout["cards"] / f"feature_{feature:04d}.json"
        try:
            observed = strict_json_load(card_path)
        except (AtlasIOError, OSError, ValueError) as error:
            raise FeatureAtlasBuildError(
                f"cards feature {feature} JSON is invalid: {error}"
            ) from error
        if observed != expected or card_path.read_bytes() != _canonical_json_bytes(expected):
            raise FeatureAtlasBuildError(
                f"cards feature {feature} differs from deterministic source rebuild"
            )
        markdown = layout["cards"] / f"feature_{feature:04d}.md"
        if markdown.read_text(encoding="utf-8") != render_evidence_card_markdown(expected):
            raise FeatureAtlasBuildError(
                f"cards feature {feature} Markdown differs from source rebuild"
            )
    return manifest


def _validate_stage_only(
    stage: str, config: FeatureAtlasConfig, paths: AtlasPaths
) -> dict[str, object]:
    """Stage-specific local-byte validation used by ``--validate-only``."""

    layout = _layout(paths)
    if stage == "external":
        manifest = strict_json_load(layout["external"] / "manifest.json")
        if "files" in manifest:
            return validate_external_snapshots(layout["external"])
        return _validate_registered_stage_bytes(layout["external"], "external")
    if stage == "concepts":
        return validate_concept_index(
            layout["external"], paths.units_path, layout["concepts"]
        )
    if stage == "sparse":
        manifest = _validate_registered_stage_bytes(layout[stage], stage)
        _validate_sparse_validation_record(paths)
        return manifest
    if stage in {"proteins", "spans"}:
        return _validate_registered_stage_bytes(layout[stage], stage)
    if stage == "discovery":
        return validate_train_discovery(
            layout["discovery"],
            config_fingerprint=config.fingerprint(),
            input_provenance={
                "concept_index_sha256": sha256_file(layout["concept_index"]),
                "protein_activation_manifest_sha256": sha256_file(
                    layout["proteins"] / "manifest.json"
                ),
            },
        )
    if stage == "heldout":
        upstream = {
            "concept": strict_json_load(layout["concepts"] / "manifest.json"),
            "covariates": strict_json_load(
                layout["covariates"].with_suffix(".manifest.json")
            ),
            "discovery": strict_json_load(layout["discovery_manifest"]),
            "protein_activations": strict_json_load(
                layout["proteins"] / "manifest.json"
            ),
        }
        input_provenance = {
            "concept_index_sha256": sha256_file(layout["concept_index"]),
            "concept_manifest_sha256": sha256_file(
                layout["concepts"] / "manifest.json"
            ),
            "covariates_manifest_sha256": sha256_file(
                layout["covariates"].with_suffix(".manifest.json")
            ),
            "covariates_sha256": sha256_file(layout["covariates"]),
            "discovery_manifest_sha256": sha256_file(layout["discovery_manifest"]),
            "protein_activation_manifest_sha256": sha256_file(
                layout["proteins"] / "manifest.json"
            ),
            "train_candidates_sha256": sha256_file(layout["train_candidates"]),
        }
        return validate_heldout_evaluation(
            layout["heldout"], config_fingerprint=config.fingerprint(),
            input_provenance=input_provenance, config=config,
            upstream_manifests=upstream,
            frozen_candidates=_frozen_candidate_mapping(
                list(strict_json_lines(layout["train_candidates"]))
            ),
        )
    if stage == "selection":
        selection = strict_json_load(layout["selection"])
        rows = _read_atlas(layout["atlas"])
        if (
            set(selection) != SELECTION_MANIFEST_KEYS
            or selection.get("schema_version") != SCHEMA_VERSION
            or selection.get("complete") is not True
            or selection.get("frozen") is not True
            or selection.get("width") != config.width
            or selection.get("config_fingerprint") != config.fingerprint()
            or selection.get("seed") != config.seed
            or not _is_sha256(selection.get("selection_fingerprint"))
            or len(rows) != config.width
        ):
            raise FeatureAtlasBuildError("selection stage is incomplete")
        candidates = selection.get("candidates")
        selected = selection.get("selected")
        if (
            not isinstance(candidates, list)
            or len(candidates) != config.width
            or [item.get("feature_index") for item in candidates if isinstance(item, Mapping)]
            != list(range(config.width))
            or not isinstance(selected, list)
            or len(selected) > 200
        ):
            raise FeatureAtlasBuildError("selection stage candidate/family counts are invalid")
        atlas_meta = selection.get("atlas")
        if isinstance(atlas_meta, Mapping):
            if (
                atlas_meta.get("bytes") != layout["atlas"].stat().st_size
                or atlas_meta.get("sha256") != sha256_file(layout["atlas"])
            ):
                raise FeatureAtlasBuildError("selection stage atlas checksum differs")
        row_by_feature = {int(row["feature_index"]): row for row in rows}
        selected_features = []
        for item in selected:
            feature = item.get("feature_index") if isinstance(item, Mapping) else None
            if type(feature) is not int or feature in selected_features or feature not in row_by_feature:
                raise FeatureAtlasBuildError("selection stage selected identity is invalid")
            selected_features.append(feature)
            atlas_row = row_by_feature[feature]
            if (
                atlas_row.get("selection_category") != item.get("category")
                or atlas_row.get("selection_reason") != item.get("reason")
                or atlas_row.get("selection_fingerprint")
                != selection.get("selection_fingerprint")
            ):
                raise FeatureAtlasBuildError(
                    "selection stage reason/category cross-link differs from atlas"
                )
        if sorted(
            row["feature_index"]
            for row in rows
            if row.get("selection_category") is not None
        ) != sorted(selected_features):
            raise FeatureAtlasBuildError("selection stage selected atlas family differs")
        return selection
    if stage == "llm":
        manifest = _validate_registered_stage_bytes(
            layout["llm"], "llm", require_complete=False
        )
        selection = strict_json_load(layout["selection"])
        _validate_task10_family(selection, manifest)
        return manifest
    if stage == "cards":
        return _validate_cards_against_sources(paths)
    if stage == "all":
        return {"statuses": validate_feature_atlas(config, paths, _local_only=True)}
    raise FeatureAtlasBuildError(f"unknown validate-only stage: {stage}")


def _install_cards(stage: Path, destination: Path, old_manifest: object) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    registered = []
    if isinstance(old_manifest, Mapping):
        outputs = old_manifest.get("outputs")
        if isinstance(outputs, list):
            registered = [str(row["path"]) for row in outputs if isinstance(row, Mapping)]
    new_names = [path.name for path in stage.iterdir()]
    allowed = set(registered) | {"manifest.json"}
    collisions = [name for name in new_names if (destination / name).exists() and name not in allowed]
    if collisions:
        raise FeatureAtlasBuildError(
            f"cards publication would replace unrelated file: {collisions[0]}"
        )
    backup = Path(tempfile.mkdtemp(prefix=".cards-backup-", dir=destination.parent))
    moved_old: list[str] = []
    moved_new: list[str] = []
    try:
        for name in sorted(allowed):
            current = destination / name
            if current.exists() or current.is_symlink():
                if current.is_symlink() or not current.is_file():
                    raise FeatureAtlasBuildError("cards registered output is not a regular file")
                os.replace(current, backup / name)
                moved_old.append(name)
        for name in sorted(name for name in new_names if name != "manifest.json"):
            os.replace(stage / name, destination / name)
            moved_new.append(name)
        os.replace(stage / "manifest.json", destination / "manifest.json")
        moved_new.append("manifest.json")
        _validate_cards_stage(destination)
    except BaseException as error:
        for name in reversed(moved_new):
            current = destination / name
            if current.exists():
                os.replace(current, stage / name)
        for name in reversed(moved_old):
            os.replace(backup / name, destination / name)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(error, FeatureAtlasBuildError):
            raise
        raise FeatureAtlasBuildError(f"atomic cards publication failed: {error}") from error
    finally:
        shutil.rmtree(backup, ignore_errors=True)
    for name in registered:
        if name not in new_names:
            (destination / name).unlink(missing_ok=True)


def _run_cards(
    config: FeatureAtlasConfig,
    paths: AtlasPaths,
    *,
    overwrite: bool,
    device: str = "cpu",
) -> dict[str, object]:
    layout = _layout(paths)
    _validate_completed_upstreams_for_cards(config, paths, device)
    destination = layout["cards"]
    manifest_path = destination / "manifest.json"
    old_manifest: dict[str, object] | None = None
    if manifest_path.exists():
        old_manifest = _validate_cards_stage(destination)
        if not overwrite:
            return old_manifest
    selection_manifest = strict_json_load(layout["selection"])
    selected = selection_manifest.get("selected")
    if not isinstance(selected, list):
        raise FeatureAtlasBuildError("selection family is invalid")
    smoke = _smoke_plan_from_root(paths.output_dir)
    if smoke is not None:
        selected = [
            item
            for item in selected
            if isinstance(item, Mapping)
            and item.get("feature_index") in set(smoke.feature_indices)
        ][: smoke.selected_feature_cap]
    if len(selected) > 200:
        raise FeatureAtlasBuildError("selected feature family exceeds maximum 200")
    features = [_feature(item.get("feature_index"), "selection") for item in selected if isinstance(item, Mapping)]
    if len(features) != len(selected) or len(features) != len(set(features)):
        raise FeatureAtlasBuildError("selection family/order is not exact and unique")
    rows = {int(row["feature_index"]): row for row in _read_atlas(layout["atlas"])}
    if not set(features) <= set(rows):
        raise FeatureAtlasBuildError("selection contains feature absent from atlas")
    stage = Path(
        tempfile.mkdtemp(prefix=".cards-stage-", dir=paths.output_dir)
    )
    try:
        outputs = []
        for item, feature in zip(selected, features):
            span = _registered_payload(layout["spans"], feature, "spans")
            llm_path = layout["llm"] / f"feature_{feature:04d}.json"
            llm_payload = (
                _registered_payload(layout["llm"], feature, "llm")
                if llm_path.exists()
                else None
            )
            card = build_evidence_card(rows[feature], item, span, llm_payload)
            json_path = stage / f"feature_{feature:04d}.json"
            markdown_path = stage / f"feature_{feature:04d}.md"
            write_json_atomic(json_path, card)
            _write_text_atomic(markdown_path, render_evidence_card_markdown(card))
            outputs.extend(
                [
                    _card_output_metadata(json_path, feature, "json"),
                    _card_output_metadata(markdown_path, feature, "markdown"),
                ]
            )
        manifest = {
            "complete": True,
            "feature_count": len(features),
            "features": features,
            "outputs": outputs,
            "provider_mode": _artifact_provider_mode(paths),
            "schema_version": SCHEMA_VERSION,
            "selection_fingerprint": selection_manifest.get("selection_fingerprint"),
        }
        write_json_atomic(stage / "manifest.json", manifest)
        _validate_cards_stage(stage)
        _install_cards(stage, destination, old_manifest)
        return _validate_cards_stage(destination)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _source_binding_dict(binding: AtlasSourceBinding) -> dict[str, object]:
    return asdict(binding)


def _build_llm_runtime(
    config: FeatureAtlasConfig, paths: AtlasPaths, *, device: str
) -> dict[str, object]:
    layout = _layout(paths)
    resolved_device, _ = _resolve_device_identity(device)
    canonical_device = str(resolved_device)
    embedder, _ = load_text_embedder(device=canonical_device)
    sae = load_frozen_sae(paths.run_dir / "inference.pt", resolved_device)
    context = _span_production_context(
        layout["selection"],
        layout["atlas"],
        paths.units_path,
        paths.run_dir / "inference.pt",
        device=canonical_device,
        config=config,
        atlas_paths=paths,
        sparse_activation_dir=layout["sparse"],
        feature_stats_path=paths.run_dir / "feature_stats.pt",
        embedder=embedder,
        sae=sae,
        allow_test_fixture_unbound=False,
    )
    plans = {
        feature: build_feature_plan(
            feature,
            {
                split: select_feature_texts(context["rows"][feature], split)
                for split in ("train", "validation")
            },
            embedder,
            seed=config.seed,
            permutations=config.permutations,
        )
        for feature in context["features"]
    }
    validate_span_attribution(
        layout["spans"],
        current_bindings=context["bindings"],
        atlas_rows=context["rows"],
        sae=sae,
        plans_by_feature=plans,
    )
    return {"context": context, "embedder": embedder, "plans": plans, "sae": sae}


def _validate_completed_stage_deep(
    stage: str,
    config: FeatureAtlasConfig,
    paths: AtlasPaths,
    *,
    device: str,
) -> dict[str, object]:
    """Validate a complete stage from frozen upstream inputs, not its receipts."""

    local = _call_stage_validator(
        stage, lambda: _validate_stage_only(stage, config, paths)
    )
    if stage not in {"sparse", "proteins", "selection", "spans", "llm"}:
        assert isinstance(local, dict)
        return local

    layout = _layout(paths)
    binding = _call_stage_validator(stage, lambda: build_source_binding(paths))
    assert isinstance(binding, AtlasSourceBinding)
    if stage == "sparse":
        source_index = _call_stage_validator(
            stage,
            lambda: build_data_index(
                paths.units_path, paths.text_manifest_path, paths.activation_dir
            ),
        )
        result = _call_stage_validator(
            stage,
            lambda: validate_sparse_feature_activations(
                source_index,
                paths.run_dir / "inference.pt",
                paths.run_dir / "feature_stats.pt",
                layout["sparse"],
                atlas_paths=paths,
                source_binding=binding,
            ),
        )
    elif stage == "proteins":
        result = _call_stage_validator(
            stage,
            lambda: validate_protein_activations(
                paths.units_path,
                layout["sparse"],
                paths.run_dir / "split_manifest.json",
                paths.filtered_uniprot_tsv,
                layout["proteins"],
                concept_index_path=layout["concept_index"],
                atlas_paths=paths,
                source_binding=binding,
                expected_width=config.width,
                expected_k=config.k,
                concept_snapshot_dir=layout["external"],
            ),
        )
    elif stage == "selection":
        result = _call_stage_validator(
            stage,
            lambda: validate_selection_manifest(
                layout["selection"],
                layout["atlas"],
                paths.run_dir / "inference.pt",
                config,
                provenance=binding,
            ),
        )
    else:
        runtime = _call_stage_validator(
            stage, lambda: _build_llm_runtime(config, paths, device=device)
        )
        assert isinstance(runtime, dict)
        if stage == "spans":
            result = strict_json_load(layout["spans"] / "manifest.json")
        else:
            selection = strict_json_load(layout["selection"])
            result = _call_stage_validator(
                stage,
                lambda: validate_llm_feature_descriptions(
                    layout["llm"],
                    label_embedder=runtime["embedder"],
                    selection_path=layout["selection"],
                    atlas_path=layout["atlas"],
                    span_output_dir=layout["spans"],
                    inference_path=paths.run_dir / "inference.pt",
                    config=config,
                    provenance=selection.get("provenance"),
                    current_bindings=runtime["context"]["bindings"],
                    sae=runtime["sae"],
                    plans_by_feature=runtime["plans"],
                ),
            )
    assert isinstance(result, dict)
    if stage == "llm":
        if result.get("provider_mode") != _validate_root_provider_configuration(
            paths.output_dir
        ):
            raise FeatureAtlasBuildError(
                "llm stage provider mode differs from root configuration"
            )
        _validate_root_llm_provider_bindings(paths, result)
    return result


def _run_llm(
    config: FeatureAtlasConfig,
    paths: AtlasPaths,
    *,
    device: str,
    providers: Sequence[object],
    overwrite: bool,
) -> dict[str, object]:
    layout = _layout(paths)
    try:
        validate_provider_family(
            providers,  # type: ignore[arg-type]
            allow_test_fixture_unbound=False,
        )
    except Exception as error:
        raise FeatureAtlasBuildError(f"llm provider family is invalid: {error}") from error
    # Reuse Task 9's production gate so current bindings and plans are rebuilt,
    # rather than trusting the recorded span manifest.
    runtime = _build_llm_runtime(config, paths, device=device)
    context = runtime["context"]
    embedder = runtime["embedder"]
    plans = runtime["plans"]
    sae = runtime["sae"]
    selection_manifest = strict_json_load(layout["selection"])
    result = run_feature_descriptions(
        layout["selection"],
        layout["atlas"],
        layout["spans"],
        layout["llm"],
        providers=providers,  # type: ignore[arg-type]
        label_embedder=embedder,
        inference_path=paths.run_dir / "inference.pt",
        config=config,
        provenance=selection_manifest.get("provenance"),
        current_bindings=context["bindings"],
        sae=sae,
        plans_by_feature=plans,
        overwrite=overwrite,
    )
    status_context = {"manifest": result, "directory": str(layout["llm"])}
    gpt = _provider_status("gpt", status_context)
    claude = _provider_status("claude", status_context)
    gemini = _provider_status("gemini", status_context)
    if (
        result.get("provider_mode") == PROVIDER_MODE_GPT_ONLY
        and gpt["success"] is not True
    ):
        raise FeatureAtlasBuildError("llm GPT draft/verdict did not complete")
    if (
        result.get("provider_mode") != PROVIDER_MODE_GPT_ONLY
        and claude["success"] is not True
    ):
        raise FeatureAtlasBuildError("llm Claude draft/verdict did not complete")
    if (
        result.get("provider_mode") == PROVIDER_MODE_DUAL
        and gemini["success"] is not True
    ):
        raise FeatureAtlasBuildError("llm Gemini draft/verdict did not complete")
    return result


def _run_stage(
    stage: str,
    config: FeatureAtlasConfig,
    paths: AtlasPaths,
    *,
    device: str,
    overwrite: bool,
    providers: Sequence[object],
    source_binding: AtlasSourceBinding | None = None,
) -> object:
    layout = _layout(paths)
    if stage == "external":
        smoke = _smoke_plan_from_root(paths.output_dir)
        specs = (
            tuple(spec for spec in SNAPSHOT_SPECS if spec.database in smoke.databases)
            if smoke is not None
            else SNAPSHOT_SPECS
        )
        return download_external_snapshots(
            layout["external"], specs=specs, overwrite=overwrite
        )
    if stage == "concepts":
        validate_external_snapshots(layout["external"])
        return build_concept_index(
            layout["external"], paths.units_path, layout["concepts"], overwrite=overwrite
        )
    if stage == "cards":
        return _run_cards(config, paths, overwrite=overwrite, device=device)
    binding = source_binding or build_source_binding(paths)
    if stage == "sparse":
        validate_source_binding(binding, paths)
        index = build_data_index(paths.units_path, paths.text_manifest_path, paths.activation_dir)
        result = extract_sparse_feature_activations(
            index,
            paths.run_dir / "inference.pt",
            layout["sparse"],
            device=device,
            overwrite=overwrite,
            atlas_paths=paths,
            source_binding=binding,
        )
        validate_sparse_feature_activations(
            index,
            paths.run_dir / "inference.pt",
            paths.run_dir / "feature_stats.pt",
            layout["sparse"],
            atlas_paths=paths,
            source_binding=binding,
        )
        write_json_atomic(
            paths.output_dir / "sparse_validation.json",
            {
                "direct_forward_validated": True,
                "feature_stats_sha256": sha256_file(
                    paths.run_dir / "feature_stats.pt"
                ),
                "schema_version": SCHEMA_VERSION,
                "sparse_manifest_sha256": sha256_file(
                    layout["sparse"] / "manifest.json"
                ),
            },
        )
        return result
    if stage == "proteins":
        index = build_data_index(paths.units_path, paths.text_manifest_path, paths.activation_dir)
        validate_sparse_feature_activations(
            index,
            paths.run_dir / "inference.pt",
            paths.run_dir / "feature_stats.pt",
            layout["sparse"],
            atlas_paths=paths,
            source_binding=binding,
        )
        validate_concept_index(layout["external"], paths.units_path, layout["concepts"])
        result = aggregate_protein_activations(
            paths.units_path,
            layout["sparse"],
            paths.run_dir / "split_manifest.json",
            paths.filtered_uniprot_tsv,
            layout["proteins"],
            concept_index_path=layout["concept_index"],
            overwrite=overwrite,
            atlas_paths=paths,
            source_binding=binding,
            expected_width=config.width,
            expected_k=config.k,
            concept_snapshot_dir=layout["external"],
        )
        build_protein_covariates(
            paths.filtered_uniprot_tsv,
            paths.units_path,
            layout["concept_index"],
            layout["covariates"],
            concept_snapshot_dir=layout["external"],
            overwrite=overwrite,
        )
        return result
    if stage == "discovery":
        validate_protein_activations(
            paths.units_path,
            layout["sparse"],
            paths.run_dir / "split_manifest.json",
            paths.filtered_uniprot_tsv,
            layout["proteins"],
            concept_index_path=layout["concept_index"],
            atlas_paths=paths,
            source_binding=binding,
            expected_width=config.width,
            expected_k=config.k,
            concept_snapshot_dir=layout["external"],
        )
        return discover_train_candidates(
            layout["proteins"],
            layout["concept_index"],
            config,
            layout["discovery"],
            overwrite=overwrite,
        )
    if stage == "heldout":
        validate_train_discovery(layout["discovery"], config_fingerprint=config.fingerprint())
        validate_protein_covariates(
            paths.filtered_uniprot_tsv,
            paths.units_path,
            layout["concept_index"],
            layout["covariates"],
            concept_snapshot_dir=layout["external"],
        )
        return evaluate_heldout_candidates(
            layout["train_candidates"],
            layout["proteins"],
            layout["covariates"],
            layout["concept_index"],
            config,
            layout["heldout"],
            overwrite=overwrite,
            snapshot_dir=layout["external"],
            units_path=paths.units_path,
        )
    if stage == "selection":
        if overwrite and layout["selection"].exists():
            _validate_stage_only("selection", config, paths)
            raise FeatureAtlasBuildError(
                "selection stage is immutable; overwrite rejected before atlas write"
            )
        build_statistical_atlas(
            paths,
            binding,
            layout["sparse"],
            layout["proteins"],
            layout["covariates"],
            layout["concept_index"],
            layout["external"],
            layout["train_candidates"],
            layout["heldout_results"],
            layout["atlas"],
            config,
            inference_path=paths.run_dir / "inference.pt",
            feature_stats_path=paths.run_dir / "feature_stats.pt",
            overwrite=overwrite,
        )
        return select_deep_features(
            layout["atlas"],
            paths.run_dir / "inference.pt",
            config,
            layout["selection"],
            provenance=binding,
            overwrite=overwrite,
        )
    if stage == "spans":
        validate_selection_manifest(
            layout["selection"],
            layout["atlas"],
            paths.run_dir / "inference.pt",
            config,
            provenance=binding,
        )
        return attribute_feature_spans(
            layout["selection"],
            layout["atlas"],
            paths.units_path,
            paths.run_dir / "inference.pt",
            layout["spans"],
            device=device,
            overwrite=overwrite,
            config=config,
            atlas_paths=paths,
            sparse_activation_dir=layout["sparse"],
            feature_stats_path=paths.run_dir / "feature_stats.pt",
        )
    if stage == "llm":
        return _run_llm(
            config, paths, device=device, providers=providers, overwrite=overwrite
        )
    raise FeatureAtlasBuildError(f"unknown stage: {stage}")


def run_feature_atlas(
    config: FeatureAtlasConfig,
    paths: AtlasPaths,
    *,
    stage: str = "all",
    device: str = "cpu",
    resume: bool = False,
    overwrite: bool = False,
    providers: Sequence[object] = (),
    provider_mode: str | None = None,
) -> dict[str, object]:
    """Run one exact stage or all stages in dependency order."""

    if not isinstance(config, FeatureAtlasConfig) or not isinstance(paths, AtlasPaths):
        raise TypeError("config and paths have invalid types")
    if type(resume) is not bool or type(overwrite) is not bool:
        raise TypeError("resume and overwrite must be built-in bool values")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}")
    if type(device) is not str or not device:
        raise TypeError("device must be a nonempty string")
    if provider_mode not in {
        None, PROVIDER_MODE_GPT_ONLY, PROVIDER_MODE_CLAUDE_ONLY, PROVIDER_MODE_DUAL
    }:
        raise ValueError("provider_mode is invalid")
    try:
        binding = build_source_binding(paths)
    except Exception as error:
        raise FeatureAtlasBuildError(f"source stage corruption: {error}") from error
    existing_smoke = _smoke_plan_from_root(paths.output_dir)
    existing_config_path = paths.output_dir / "config.json"
    existing_root_config = (
        strict_json_load(existing_config_path) if existing_config_path.exists() else {}
    )
    if providers:
        try:
            provider_bindings, family_mode = validate_provider_family(
                providers,  # type: ignore[arg-type]
                allow_test_fixture_unbound=False,
                return_mode=True,
            )
        except Exception as error:
            raise FeatureAtlasBuildError(
                f"root provider family is invalid: {error}"
            ) from error
        if provider_mode is not None and provider_mode != family_mode:
            raise FeatureAtlasBuildError(
                "requested provider mode differs from provider family"
            )
        provider_mode = family_mode
        provider_configuration = {
            str(row["provider"]): row for row in provider_bindings
        }
        if family_mode == PROVIDER_MODE_GPT_ONLY:
            provider_configuration.update({
                "claude": {"status": "skipped_by_user"},
                "gemini": {"status": "skipped_by_user"},
            })
        elif family_mode == PROVIDER_MODE_CLAUDE_ONLY:
            provider_configuration["gemini"] = {"status": "skipped_by_user"}
    else:
        existing_configuration = existing_root_config.get("provider_configuration")
        requested_mode = provider_mode or str(
            existing_root_config.get("provider_mode", PROVIDER_MODE_DUAL)
        )
        if (
            isinstance(existing_configuration, Mapping)
            and existing_root_config.get("provider_mode") == requested_mode
        ):
            provider_configuration = dict(existing_configuration)
        else:
            provider_configuration = {
                **(
                    {
                        "gpt": {"requested": GPT_MODEL, "resolved": GPT_MODEL},
                        "claude": {"status": "skipped_by_user"},
                    }
                    if requested_mode == PROVIDER_MODE_GPT_ONLY else {
                        "claude": {"requested": CLAUDE_MODEL, "resolved": CLAUDE_MODEL},
                    }
                ),
                "gemini": (
                    {"status": "skipped_by_user"}
                    if requested_mode in {
                        PROVIDER_MODE_GPT_ONLY, PROVIDER_MODE_CLAUDE_ONLY
                    }
                    else {"requested": GEMINI_MODEL, "resolved": GEMINI_MODEL}
                ),
            }
        provider_mode = requested_mode
    transactional_rebind = overwrite and stage == "llm"
    root_paths = (
        paths.output_dir / "config.json",
        paths.output_dir / "provider_bindings.json",
    )
    root_snapshot = {
        path: path.read_bytes() if path.exists() else None
        for path in root_paths
    }
    transaction_backup: Path | None = None
    if transactional_rebind:
        transaction_backup = Path(tempfile.mkdtemp(
            prefix=".atlas-llm-transaction-", dir=paths.output_dir.parent
        ))
        llm_directory = _layout(paths)["llm"]
        if llm_directory.exists():
            shutil.copytree(llm_directory, transaction_backup / "llm")
        summary_path = _layout(paths)["summary"]
        if summary_path.exists():
            shutil.copy2(summary_path, transaction_backup / "summary.json")
    chosen = STAGES[:-1] if stage == "all" else (stage,)
    results = {}
    try:
        _publish_root_manifests(
            config, paths.output_dir, asdict(binding),
            provider_models=provider_configuration, provider_mode=provider_mode,
            allow_provider_rebind=transactional_rebind,
            smoke_features=(existing_smoke.feature_count if existing_smoke else None),
            smoke_databases=(existing_smoke.databases if existing_smoke else ()),
        )
        if transactional_rebind:
            _transaction_event("root_manifests_published")
        for name in chosen:
            try:
                if resume:
                    resumed = _resume_stage(name, config, paths)
                    if resumed is not None:
                        results[name] = resumed
                        continue
                results[name] = _run_stage(
                    name, config, paths, device=device, overwrite=overwrite,
                    providers=providers, source_binding=binding,
                )
                if transactional_rebind and name == "llm":
                    _transaction_event("llm_published_before_summary_invalidation")
                if overwrite and name in {"llm", "cards"}:
                    _layout(paths)["summary"].unlink(missing_ok=True)
            except FeatureAtlasBuildError:
                raise
            except Exception as error:
                raise FeatureAtlasBuildError(
                    f"{name} stage failed: {error}"
                ) from error
    except BaseException:
        if transactional_rebind:
            for path, content in root_snapshot.items():
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    _write_bytes_atomic(path, content)
            assert transaction_backup is not None
            llm_directory = _layout(paths)["llm"]
            if llm_directory.exists() or llm_directory.is_symlink():
                if llm_directory.is_symlink() or not llm_directory.is_dir():
                    llm_directory.unlink()
                else:
                    shutil.rmtree(llm_directory)
            backed_up_llm = transaction_backup / "llm"
            if backed_up_llm.exists():
                shutil.copytree(backed_up_llm, llm_directory)
            summary_path = _layout(paths)["summary"]
            summary_path.unlink(missing_ok=True)
            backed_up_summary = transaction_backup / "summary.json"
            if backed_up_summary.exists():
                shutil.copy2(backed_up_summary, summary_path)
            shutil.rmtree(transaction_backup, ignore_errors=True)
        raise
    if transaction_backup is not None:
        shutil.rmtree(transaction_backup, ignore_errors=True)
    if stage == "all":
        results["validation"] = validate_feature_atlas(config, paths, device=device)
    return results


def _call_stage_validator(stage: str, validator: Callable[[], object]) -> object:
    try:
        return validator()
    except Exception as error:
        raise FeatureAtlasBuildError(f"{stage} stage corruption: {error}") from error


def _validate_source() -> None:
    """Patch seam used by corruption-name tests."""


def _local_manifest(directory: Path, stage: str) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = strict_json_load(manifest_path)
    if manifest.get("complete") is not True:
        raise FeatureAtlasBuildError(f"{stage} stage is incomplete")
    return manifest


def _validate_all_completed_stages(
    config: FeatureAtlasConfig,
    paths: AtlasPaths,
    binding: AtlasSourceBinding,
    *,
    device: str = "cpu",
    local_only: bool = False,
) -> dict[str, object]:
    """Validate registered stages; local_only constructs no model or provider."""

    layout = _layout(paths)
    validated: dict[str, object] = {}
    validate_source_binding(binding, paths)
    try:
        root_config = strict_json_load(paths.output_dir / "config.json")
        root_source = strict_json_load(paths.output_dir / "source_manifest.json")
    except (AtlasIOError, OSError, ValueError) as error:
        raise FeatureAtlasBuildError(f"source stage root manifest corruption: {error}") from error
    if (
        root_config.get("config") != config.to_dict()
        or root_config.get("config_fingerprint") != config.fingerprint()
        or root_source.get("source_binding") != asdict(binding)
    ):
        raise FeatureAtlasBuildError("source stage root config/source binding differs")
    root_provider_mode = _validate_root_provider_configuration(paths.output_dir)
    if (layout["external"] / "manifest.json").exists():
        validated["external"] = _call_stage_validator(
            "external", lambda: validate_external_snapshots(layout["external"])
        )
    if (layout["concepts"] / "manifest.json").exists():
        validated["concepts"] = _call_stage_validator(
            "concepts",
            lambda: validate_concept_index(
                layout["external"], paths.units_path, layout["concepts"]
            ),
        )
    for stage in ("sparse", "proteins", "spans"):
        directory = layout[stage]
        if (directory / "manifest.json").exists():
            validated[stage] = _call_stage_validator(
                stage,
                lambda directory=directory, stage=stage: _validate_registered_stage_bytes(
                    directory, stage
                ),
            )
    if "sparse" in validated:
        _call_stage_validator("sparse", lambda: _validate_sparse_validation_record(paths))
    llm_manifest_path = layout["llm"] / "manifest.json"
    if llm_manifest_path.exists():
        validated["llm"] = _call_stage_validator(
            "llm",
            lambda: _validate_registered_stage_bytes(
                layout["llm"], "llm", require_complete=False
            ),
        )
        _validate_root_llm_provider_bindings(paths, validated["llm"])
    if layout["discovery_manifest"].exists():
        validated["discovery"] = _call_stage_validator(
            "discovery",
            lambda: _validate_stage_only("discovery", config, paths),
        )
    if layout["heldout_manifest"].exists():
        validated["heldout"] = _call_stage_validator(
            "heldout",
            lambda: _validate_stage_only("heldout", config, paths),
        )
    if layout["atlas"].exists():
        validated["selection_rows"] = _call_stage_validator(
            "selection", lambda: _read_atlas(layout["atlas"])
        )
    if layout["selection"].exists():
        selection = _call_stage_validator(
            "selection", lambda: _validate_stage_only("selection", config, paths)
        )
        assert isinstance(selection, dict)
        selected = selection.get("selected")
        if not isinstance(selected, list) or len(selected) > 200:
            raise FeatureAtlasBuildError("selection stage corruption: invalid family")
        validated["selection"] = selection
    if (layout["cards"] / "manifest.json").exists():
        validated["cards"] = _call_stage_validator(
            "cards", lambda: _validate_cards_against_sources(paths)
        )
    if "llm" in validated and "selection" in validated:
        _validate_task10_family(validated["selection"], validated["llm"])
    if (
        "llm" in validated
        and validated["llm"].get("provider_mode") != root_provider_mode
    ):
        raise FeatureAtlasBuildError(
            "llm stage provider mode differs from root configuration"
        )
    if "proteins" in validated:
        if (
            not layout["covariates"].is_file()
            or not layout["covariates"].with_suffix(".manifest.json").is_file()
        ):
            raise FeatureAtlasBuildError(
                "proteins stage required covariate data/manifest is missing"
            )
        _call_stage_validator(
            "proteins",
            lambda: validate_protein_covariates(
                paths.filtered_uniprot_tsv,
                paths.units_path,
                layout["concept_index"],
                layout["covariates"],
                concept_snapshot_dir=layout["external"],
            ),
        )
    if local_only:
        return validated

    source_index = None
    if "sparse" in validated:
        source_index = build_data_index(
            paths.units_path, paths.text_manifest_path, paths.activation_dir
        )
        validated["sparse"] = _call_stage_validator(
            "sparse",
            lambda: validate_sparse_feature_activations(
                source_index,
                paths.run_dir / "inference.pt",
                paths.run_dir / "feature_stats.pt",
                layout["sparse"],
                atlas_paths=paths,
                source_binding=binding,
            ),
        )
    if "proteins" in validated:
        validated["proteins"] = _call_stage_validator(
            "proteins",
            lambda: validate_protein_activations(
                paths.units_path,
                layout["sparse"],
                paths.run_dir / "split_manifest.json",
                paths.filtered_uniprot_tsv,
                layout["proteins"],
                concept_index_path=layout["concept_index"],
                atlas_paths=paths,
                source_binding=binding,
                expected_width=config.width,
                expected_k=config.k,
                concept_snapshot_dir=layout["external"],
            ),
        )
        _call_stage_validator(
            "proteins",
            lambda: validate_protein_covariates(
                paths.filtered_uniprot_tsv,
                paths.units_path,
                layout["concept_index"],
                layout["covariates"],
                concept_snapshot_dir=layout["external"],
            ),
        )
    if "selection" in validated:
        validated["selection"] = _call_stage_validator(
            "selection",
            lambda: validate_selection_manifest(
                layout["selection"],
                layout["atlas"],
                paths.run_dir / "inference.pt",
                config,
                provenance=binding,
            ),
        )
    runtime = None
    if "spans" in validated or "llm" in validated:
        runtime = _call_stage_validator(
            "spans", lambda: _build_llm_runtime(config, paths, device=device)
        )
        assert isinstance(runtime, dict)
        validated["spans"] = strict_json_load(layout["spans"] / "manifest.json")
    if "llm" in validated:
        assert isinstance(runtime, dict)
        selection = validated.get("selection")
        provenance = selection.get("provenance") if isinstance(selection, Mapping) else None
        validated["llm"] = _call_stage_validator(
            "llm",
            lambda: validate_llm_feature_descriptions(
                layout["llm"],
                label_embedder=runtime["embedder"],
                selection_path=layout["selection"],
                atlas_path=layout["atlas"],
                span_output_dir=layout["spans"],
                inference_path=paths.run_dir / "inference.pt",
                config=config,
                provenance=provenance,
                current_bindings=runtime["context"]["bindings"],
                sae=runtime["sae"],
                plans_by_feature=runtime["plans"],
            ),
        )
    return validated


def _validate_deterministic_samples(
    config: FeatureAtlasConfig, paths: AtlasPaths, completed: Mapping[str, object]
) -> None:
    """Recompute deterministic local links/counts without constructing a model."""

    layout = _layout(paths)
    if "selection" in completed and layout["atlas"].exists():
        rows = _read_atlas(layout["atlas"])
        if len(rows) != config.width:
            raise FeatureAtlasBuildError("selection stage exact atlas width differs")
        selection = completed["selection"]
        assert isinstance(selection, Mapping)
        selected = selection.get("selected")
        features = [item.get("feature_index") for item in selected] if isinstance(selected, list) else []
        if features != sorted(set(features)):
            raise FeatureAtlasBuildError("selection stage family is not sorted and unique")
        atlas_selected = [
            row["feature_index"] for row in rows if row.get("selection_category") is not None
        ]
        if features != atlas_selected:
            raise FeatureAtlasBuildError("selection stage atlas links differ")
    if "spans" in completed and "selection" in completed:
        span_manifest = completed["spans"]
        selection = completed["selection"]
        if isinstance(span_manifest, Mapping) and isinstance(selection, Mapping):
            features = sorted(
                row["feature_index"] for row in selection.get("selected", [])
            )
            if span_manifest.get("features") != features:
                raise FeatureAtlasBuildError("spans stage family differs from selection")
    if "llm" in completed and "selection" in completed:
        llm_manifest = completed["llm"]
        selection = completed["selection"]
        if isinstance(llm_manifest, Mapping) and isinstance(selection, Mapping):
            _validate_task10_family(selection, llm_manifest)


def _provider_status(name: str, context: object) -> dict[str, object]:
    if context is None:
        return {"success": False, "status": "not_run"}
    if (
        isinstance(context, Mapping)
        and isinstance(context.get("manifest"), Mapping)
        and type(context.get("directory")) is str
    ):
        llm_manifest = context["manifest"]
        directory = Path(context["directory"])
    elif isinstance(context, Mapping):
        llm_manifest = context
        directory = None
    else:
        return {"success": False, "status": "invalid_or_incomplete"}
    artifacts = llm_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return {"success": False, "status": "invalid_or_incomplete"}
    mode = llm_manifest.get("provider_mode")
    inactive = (
        name in {"claude", "gemini"} and mode == PROVIDER_MODE_GPT_ONLY
    ) or (
        name == "gemini" and mode == PROVIDER_MODE_CLAUDE_ONLY
    )
    if inactive:
        provider_statuses = llm_manifest.get("provider_statuses")
        if (
            isinstance(provider_statuses, Mapping)
            and provider_statuses.get(name) == "skipped_by_user"
            and not any(
                isinstance(item, Mapping) and item.get("provider") == name
                for item in artifacts
            )
        ):
            return {"success": None, "status": "skipped_by_user"}
        return {"success": False, "status": "invalid_or_incomplete"}
    provider_statuses = llm_manifest.get("provider_statuses")
    if (
        llm_manifest.get("complete") is True
        and llm_manifest.get("family_order") == []
        and isinstance(provider_statuses, Mapping)
        and provider_statuses.get(name) == "completed"
        and not any(
            isinstance(item, Mapping) and item.get("provider") == name
            for item in artifacts
        )
    ):
        return {"success": True, "status": "complete"}
    statuses = []
    for item in artifacts:
        if (
            isinstance(item, Mapping)
            and item.get("provider") == name
            and item.get("kind") in {"draft_status", "verdict_status"}
            and type(item.get("path")) is str
        ):
            if directory is None:
                continue
            path = directory / str(item["path"])
            try:
                payload = strict_json_load(path)
            except (AtlasIOError, OSError, ValueError):
                return {"success": False, "status": "invalid_or_incomplete"}
            statuses.append(payload.get("status"))
    if statuses and "llm_failed" in statuses:
        return {"success": False, "status": "llm_failed"}
    success = llm_manifest.get("complete") is True and bool(statuses)
    return {"success": success, "status": "complete" if success else "not_run"}


def _summary_files(paths: AtlasPaths) -> dict[str, object]:
    layout = _layout(paths)
    candidates = {
        "config.json": paths.output_dir / "config.json",
        "source_manifest.json": paths.output_dir / "source_manifest.json",
        "provider_bindings.json": paths.output_dir / "provider_bindings.json",
        "sparse_validation.json": paths.output_dir / "sparse_validation.json",
        "protein_covariates.jsonl": layout["covariates"],
        "protein_covariates.manifest.json": layout["covariates"].with_suffix(
            ".manifest.json"
        ),
        "cards/manifest.json": layout["cards"] / "manifest.json",
        "external/manifest.json": layout["external"] / "manifest.json",
        "concepts/manifest.json": layout["concepts"] / "manifest.json",
        "sparse_activations/manifest.json": layout["sparse"] / "manifest.json",
        "protein_activations/manifest.json": layout["proteins"] / "manifest.json",
        "concepts/discovery_manifest.json": layout["discovery_manifest"],
        "concepts/heldout_manifest.json": layout["heldout_manifest"],
        "span_attribution/manifest.json": layout["spans"] / "manifest.json",
        "llm/manifest.json": layout["llm"] / "manifest.json",
        "feature_atlas.jsonl": layout["atlas"],
        "selection_manifest.json": layout["selection"],
    }
    return {
        name: file_metadata(path)
        for name, path in sorted(candidates.items())
        if path.exists()
    }


def validate_feature_atlas(
    config: FeatureAtlasConfig,
    paths: AtlasPaths,
    *,
    device: str = "cpu",
    _local_only: bool = False,
) -> dict[str, object]:
    """Offline final validator returning independent statistical/provider statuses."""

    if not isinstance(config, FeatureAtlasConfig) or not isinstance(paths, AtlasPaths):
        raise TypeError("config and paths have invalid types")
    try:
        binding = build_source_binding(paths)
    except Exception as error:
        raise FeatureAtlasBuildError(f"source stage corruption: {error}") from error
    try:
        completed = _validate_all_completed_stages(
            config, paths, binding, device=device, local_only=_local_only
        )
        _validate_deterministic_samples(config, paths, completed)
        atlas_rows = _read_atlas(_layout(paths)["atlas"])
        _require_complete_core(completed, atlas_rows, expected_width=config.width)
        cards = completed["cards"]
        assert isinstance(cards, Mapping)
    except FeatureAtlasBuildError:
        raise
    except Exception as error:
        raise FeatureAtlasBuildError(f"statistical_atlas stage corruption: {error}") from error
    llm_manifest = completed.get("llm")
    if not isinstance(llm_manifest, Mapping):
        llm_manifest = None
    span_manifest = completed.get("spans")
    masking_success = isinstance(span_manifest, Mapping) and span_manifest.get("complete") is True
    statuses = {
        "statistical_atlas": {
            "core": True,
            "feature_count": len(atlas_rows),
            "selected_feature_count": len(
                completed["selection"].get("selected", [])
                if isinstance(completed["selection"], Mapping)
                else []
            ),
            "card_count": cards.get("feature_count"),
            "status": "complete",
        },
        "masking": {
            "success": masking_success,
            "status": "complete" if masking_success else "not_run",
        },
        "gpt": _provider_status(
            "gpt",
            {"manifest": llm_manifest, "directory": str(_layout(paths)["llm"])}
            if llm_manifest is not None
            else None,
        ),
        "claude": _provider_status(
            "claude",
            {"manifest": llm_manifest, "directory": str(_layout(paths)["llm"])}
            if llm_manifest is not None
            else None,
        ),
        "gemini": _provider_status(
            "gemini",
            {"manifest": llm_manifest, "directory": str(_layout(paths)["llm"])}
            if llm_manifest is not None
            else None,
        ),
    }
    summary = {
        "complete": True,
        "config_fingerprint": config.fingerprint(),
        "files": _summary_files(paths),
        "provider_mode": _validate_root_provider_configuration(paths.output_dir),
        "schema_version": SCHEMA_VERSION,
        "source_binding": (
            _source_binding_dict(binding)
            if isinstance(binding, AtlasSourceBinding)
            else _clone(getattr(binding, "__dict__", {}))
        ),
        "statuses": statuses,
    }
    if "summary.json" in summary["files"]:
        raise FeatureAtlasBuildError("summary cannot checksum itself")
    summary_path = _layout(paths)["summary"]
    if summary_path.exists() or summary_path.is_symlink():
        try:
            observed_summary = strict_json_load(summary_path)
        except (AtlasIOError, OSError, ValueError) as error:
            raise FeatureAtlasBuildError(f"summary stage corruption: {error}") from error
        if observed_summary != summary:
            raise FeatureAtlasBuildError("summary stage corruption: registered summary differs")
    else:
        write_json_atomic(summary_path, summary)
    return statuses


def _parse_smoke_databases(value: str) -> tuple[str, ...]:
    allowed = {"go", "reactome", "interpro", "corum"}
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(result) != len(set(result)) or not set(result) <= allowed:
        raise argparse.ArgumentTypeError("smoke databases must be unique known databases")
    return result


def _run_spans_preflight(
    config: FeatureAtlasConfig, paths: AtlasPaths, *, device: str
) -> dict[str, object]:
    layout = _layout(paths)
    before = set(paths.output_dir.rglob("*")) if paths.output_dir.exists() else set()
    try:
        context = _span_production_context(
            layout["selection"],
            layout["atlas"],
            paths.units_path,
            paths.run_dir / "inference.pt",
            device=device,
            config=config,
            atlas_paths=paths,
            sparse_activation_dir=layout["sparse"],
            feature_stats_path=paths.run_dir / "feature_stats.pt",
            embedder=None,
            sae=None,
            allow_test_fixture_unbound=False,
        )
        fingerprint = context["bindings"]["model_binding"]["fingerprint"]
        plans = [
            build_feature_plan(
                feature,
                {
                    split: select_feature_texts(context["rows"][feature], split)
                    for split in ("train", "validation")
                },
                context["embedder"],
                seed=config.seed,
                permutations=config.permutations,
            )
            for feature in context["features"]
        ]
        budget = preflight_feature_plans(
            plans,
            EmbeddingCache(
                layout["spans"] / "cache",
                model_binding_fingerprint=fingerprint,
            ),
            max_encodings=config.max_span_encodings,
        )
    except Exception as error:
        raise FeatureAtlasBuildError(f"spans preflight failed: {error}") from error
    after = set(paths.output_dir.rglob("*")) if paths.output_dir.exists() else set()
    if after != before:
        raise FeatureAtlasBuildError("spans preflight wrote output artifacts")
    return budget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=AtlasPaths().output_dir)
    parser.add_argument("--smoke-features", type=int)
    parser.add_argument("--smoke-databases", type=_parse_smoke_databases)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--claude-command", default="claude")
    parser.add_argument("--claude-model", default=CLAUDE_MODEL)
    parser.add_argument("--gemini-command", default="gemini")
    parser.add_argument("--gemini-model", default=GEMINI_MODEL)
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--gpt-only", action="store_true")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--gpt-model", default=GPT_MODEL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(raw_argv)
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    legacy_provider_flags = {
        "--skip-gemini",
        "--claude-command",
        "--claude-model",
        "--gemini-command",
        "--gemini-model",
    }
    if args.gpt_only and any(
        token == flag or token.startswith(flag + "=")
        for token in raw_argv
        for flag in legacy_provider_flags
    ):
        raise SystemExit("--gpt-only is incompatible with Claude/Gemini provider flags")
    if args.gpt_only and args.gpt_model != GPT_MODEL:
        raise SystemExit("--gpt-model must match the exact frozen GPT model")
    if (
        not args.gpt_only
        and (args.claude_model != CLAUDE_MODEL or args.gemini_model != GEMINI_MODEL)
    ):
        raise SystemExit("provider models must match the exact frozen Task 10 models")
    if (args.smoke_features is None) is not (args.smoke_databases is None):
        raise SystemExit("--smoke-features and --smoke-databases must be supplied together")
    paths = AtlasPaths(output_dir=args.output_dir)
    config = FeatureAtlasConfig()
    smoke_plan = (
        SmokePlan(args.smoke_features, args.smoke_databases)
        if args.smoke_features is not None
        else None
    )
    providers: tuple[object, ...] = ()
    if not args.validate_only and not args.preflight and args.stage in {"llm", "all"}:
        if args.gpt_only:
            try:
                identity = probe_codex_model_identity(
                    args.codex_command,
                    cwd=Path.cwd(),
                    model=args.gpt_model,
                )
            except CodexCLITransportError as error:
                raise FeatureAtlasBuildError(
                    f"gpt provider preflight failed: {error}"
                ) from error
            providers = (
                CodexCLIProvider(
                    executable=identity.resolved_executable,
                    resolved_executable=identity.resolved_executable,
                    cli_version=identity.cli_version,
                    reported_version=identity.reported_version,
                ),
            )
        else:
            providers = (
                _preflight_provider("claude", args.claude_command, args.claude_model),
            )
            if not args.skip_gemini:
                providers += (
                    _preflight_provider("gemini", args.gemini_command, args.gemini_model),
                )
    provider_mode = (
        PROVIDER_MODE_GPT_ONLY
        if args.gpt_only
        else PROVIDER_MODE_CLAUDE_ONLY if args.skip_gemini else PROVIDER_MODE_DUAL
    )
    if smoke_plan is not None:
        binding = (
            AtlasSourceBinding(**{
                name: _smoke_digest([name, smoke_plan.feature_count, smoke_plan.databases])
                for name in AtlasSourceBinding.__dataclass_fields__
            })
            if smoke_plan is not None
            else build_source_binding(paths)
        )
        provider_configuration = (
            {provider.provider: provider.binding() for provider in providers}
            if providers
            else {
                "claude": {"requested": args.claude_model, "resolved": args.claude_model},
                "gemini": {"requested": args.gemini_model, "resolved": args.gemini_model},
            }
        )
        if provider_mode == PROVIDER_MODE_GPT_ONLY:
            provider_configuration.update({
                "claude": {"status": "skipped_by_user"},
                "gemini": {"status": "skipped_by_user"},
            })
        elif provider_mode == PROVIDER_MODE_CLAUDE_ONLY:
            provider_configuration["gemini"] = {"status": "skipped_by_user"}
        _publish_root_manifests(
            config,
            paths.output_dir,
            asdict(binding),
            provider_models=provider_configuration,
            provider_mode=provider_mode,
            allow_provider_rebind=args.overwrite and args.stage == "llm",
            smoke_features=(smoke_plan.feature_count if smoke_plan else None),
            smoke_databases=(smoke_plan.databases if smoke_plan else ()),
        )
    if args.preflight:
        if args.stage != "spans" or args.validate_only:
            raise SystemExit("--preflight is valid only for the writing spans stage")
        result = _run_spans_preflight(config, paths, device=args.device)
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    if args.validate_only:
        result = (
            validate_feature_atlas(config, paths, device=args.device, _local_only=False)
            if args.stage == "all"
            else _validate_completed_stage_deep(
                args.stage, config, paths, device=args.device
            )
        )
    else:
        result = (
            _run_smoke_stage_family(paths, smoke_plan, providers)
            if smoke_plan is not None
            else run_feature_atlas(
                config,
                paths,
                stage=args.stage,
                device=args.device,
                resume=args.resume,
                overwrite=args.overwrite,
                providers=providers,
                provider_mode=provider_mode,
            )
        )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
