"""Immutable configuration and source binding for the SAE feature atlas."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .build_sae_corpus import validate_corpus
from .extract_protrek_activations import validate_activation_cache
from .feature_atlas_io import AtlasIOError, sha256_file, strict_json_load
from .train_topk_sae import validate_pilot_run


EXPECTED_INFERENCE_SHA256 = (
    "4a06d9b98474d01f661087c9229219e032803d6caeaaaad551f370e86b40a79c"
)
EXPECTED_FEATURE_STATS_SHA256 = (
    "faa9ef5fc296f39b7bfd9233b6ae636be91b52ab5cca1b27f65cea81729629ce"
)
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "6380251170b2ac95b12e3534a40d4f991138de80be97c47fb9c2b7d4148d3067"
)
EXPECTED_UNITS_SHA256 = (
    "5a6ccc86368a3f7f806081c17228e6b6446aa5de75772f6e8f66e7eea9aac49f"
)
EXPECTED_ACTIVATION_MANIFEST_SHA256 = (
    "313d40761b0f1e3f1cef271540f937d0b69bfdc79b10e911d77288270c5a2347"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT / "data/sae_runs/protrek_final_1024/topk_w8_k32_seed42"
)
DEFAULT_CORPUS_DIR = (
    PROJECT_ROOT / "data/sae_corpus/uniprot_human_reviewed/processed"
)
DEFAULT_ACTIVATION_DIR = (
    PROJECT_ROOT
    / "data/sae_corpus/uniprot_human_reviewed/activations/protrek_final_1024"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/sae_analysis/protrek_final_1024/topk_w8_k32_seed42/feature_atlas_v1"
)


class AtlasConfigError(ValueError):
    """Raised when atlas configuration or bound source provenance is invalid."""


@dataclass(frozen=True)
class FeatureAtlasConfig:
    schema_version: int = 1
    width: int = 8192
    k: int = 32
    seed: int = 42
    activation_threshold: float = 1e-3
    biological_slots: int = 160
    field_slots: int = 20
    control_slots: int = 20
    train_min_annotated: int = 20
    train_min_firing: int = 20
    train_min_overlap: int = 5
    train_candidate_fdr: float = 0.10
    validation_descriptive_min: int = 10
    validation_confirmatory_min: int = 20
    heldout_fdr: float = 0.05
    heldout_min_auroc: float = 0.70
    heldout_min_auprc_lift: float = 2.0
    controls_per_positive: int = 5
    propensity_caliper_sd: float = 0.2
    permutations: int = 1000
    max_span_encodings: int = 150_000
    max_llm_attempts: int = 3

    def __post_init__(self) -> None:
        integer_fields = (
            "schema_version",
            "width",
            "k",
            "seed",
            "biological_slots",
            "field_slots",
            "control_slots",
            "train_min_annotated",
            "train_min_firing",
            "train_min_overlap",
            "validation_descriptive_min",
            "validation_confirmatory_min",
            "controls_per_positive",
            "permutations",
            "max_span_encodings",
            "max_llm_attempts",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be a built-in int")
        float_fields = (
            "activation_threshold",
            "train_candidate_fdr",
            "heldout_fdr",
            "heldout_min_auroc",
            "heldout_min_auprc_lift",
            "propensity_caliper_sd",
        )
        for name in float_fields:
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a built-in float")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        for name in integer_fields:
            if name not in ("schema_version", "seed") and getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.k > self.width:
            raise ValueError("k must be no greater than width")
        if self.train_min_overlap > min(
            self.train_min_annotated, self.train_min_firing
        ):
            raise ValueError("train_min_overlap cannot exceed its source minimums")
        if self.validation_descriptive_min > self.validation_confirmatory_min:
            raise ValueError(
                "validation_descriptive_min cannot exceed validation_confirmatory_min"
            )
        if self.activation_threshold <= 0.0:
            raise ValueError("activation_threshold must be positive")
        for name in ("train_candidate_fdr", "heldout_fdr"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be greater than zero and at most one")
        if self.heldout_fdr > self.train_candidate_fdr:
            raise ValueError("heldout_fdr cannot exceed train_candidate_fdr")
        if not 0.0 <= self.heldout_min_auroc <= 1.0:
            raise ValueError("heldout_min_auroc must be between zero and one")
        for name in ("heldout_min_auprc_lift", "propensity_caliper_sd"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AtlasPaths:
    run_dir: Path = DEFAULT_RUN_DIR
    units_path: Path = DEFAULT_CORPUS_DIR / "sae_units.jsonl"
    activation_dir: Path = DEFAULT_ACTIVATION_DIR
    text_manifest_path: Path = DEFAULT_CORPUS_DIR / "text_corpus_manifest.json"
    filtered_uniprot_tsv: Path = (
        DEFAULT_CORPUS_DIR / "uniprotkb_human_reviewed_non_sl.tsv"
    )
    output_dir: Path = DEFAULT_OUTPUT_DIR

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if not isinstance(getattr(self, field_name), Path):
                raise TypeError(f"{field_name} must be a Path")


@dataclass(frozen=True)
class AtlasSourceBinding:
    inference_sha256: str
    feature_stats_sha256: str
    split_manifest_sha256: str
    summary_sha256: str
    units_sha256: str
    activation_manifest_sha256: str
    text_manifest_sha256: str
    filtered_uniprot_sha256: str
    assignment_sha256: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise TypeError(f"{field_name} must be a lowercase SHA-256 string")


def _assignment_from_summary(summary: object, label: str) -> str:
    provenance = summary.get("provenance") if isinstance(summary, dict) else None
    assignment = (
        provenance.get("assignment_sha256") if isinstance(provenance, dict) else None
    )
    if (
        type(assignment) is not str
        or len(assignment) != 64
        or any(character not in "0123456789abcdef" for character in assignment)
    ):
        raise AtlasConfigError(f"{label} has invalid assignment provenance")
    return assignment


def _exclusion_path(text_manifest_path: Path) -> Path:
    try:
        manifest = strict_json_load(text_manifest_path)
    except (AtlasIOError, OSError, ValueError) as error:
        raise AtlasConfigError(
            f"Invalid text_corpus_manifest.json: {text_manifest_path}"
        ) from error
    sources = manifest.get("sources")
    exclusion = sources.get("exclusion_table") if isinstance(sources, dict) else None
    raw_path = exclusion.get("path") if isinstance(exclusion, dict) else None
    if type(raw_path) is not str or not raw_path:
        raise AtlasConfigError(
            f"text_corpus_manifest.json has invalid exclusion_table path: {text_manifest_path}"
        )
    return Path(raw_path)


def _validate_upstreams(paths: AtlasPaths) -> tuple[dict[str, object], str]:
    exclusion_path = _exclusion_path(paths.text_manifest_path)
    activation_manifest_path = paths.activation_dir / "manifest.json"
    try:
        strict_json_load(activation_manifest_path)
    except (AtlasIOError, OSError, ValueError) as error:
        raise AtlasConfigError(
            f"Invalid activation manifest.json: {activation_manifest_path}"
        ) from error
    try:
        summary = validate_pilot_run(paths.run_dir)
    except Exception as error:
        raise AtlasConfigError(f"Invalid completed pilot run: {paths.run_dir}") from error
    try:
        validate_corpus(
            paths.text_manifest_path.parent,
            paths.filtered_uniprot_tsv,
            exclusion_path,
        )
    except Exception as error:
        raise AtlasConfigError(
            f"Invalid text corpus or filtered UniProt file: {paths.text_manifest_path}"
        ) from error
    try:
        validate_activation_cache(paths.units_path, paths.activation_dir)
    except Exception as error:
        raise AtlasConfigError(
            f"Invalid activation manifest.json: {activation_manifest_path}"
        ) from error
    return summary, _assignment_from_summary(summary, "pilot summary")


def _source_files(paths: AtlasPaths) -> tuple[tuple[str, str, Path], ...]:
    return (
        ("inference_sha256", "inference.pt", paths.run_dir / "inference.pt"),
        (
            "feature_stats_sha256",
            "feature_stats.pt",
            paths.run_dir / "feature_stats.pt",
        ),
        (
            "split_manifest_sha256",
            "split_manifest.json",
            paths.run_dir / "split_manifest.json",
        ),
        ("summary_sha256", "summary.json", paths.run_dir / "summary.json"),
        ("units_sha256", "sae_units.jsonl", paths.units_path),
        (
            "activation_manifest_sha256",
            "manifest.json",
            paths.activation_dir / "manifest.json",
        ),
        (
            "text_manifest_sha256",
            "text_corpus_manifest.json",
            paths.text_manifest_path,
        ),
        (
            "filtered_uniprot_sha256",
            paths.filtered_uniprot_tsv.name,
            paths.filtered_uniprot_tsv,
        ),
    )


def _observed_hashes(paths: AtlasPaths) -> dict[str, str]:
    observed: dict[str, str] = {}
    for field_name, label, path in _source_files(paths):
        try:
            observed[field_name] = sha256_file(path)
        except (AtlasIOError, OSError, ValueError) as error:
            raise AtlasConfigError(f"Cannot hash {label}: {path}") from error
    return observed


def build_source_binding(paths: AtlasPaths) -> AtlasSourceBinding:
    """Validate the completed inputs and freeze their exact provenance."""

    if not isinstance(paths, AtlasPaths):
        raise TypeError("paths must be AtlasPaths")
    _, assignment = _validate_upstreams(paths)
    observed = _observed_hashes(paths)
    expected = (
        ("inference_sha256", "inference.pt", EXPECTED_INFERENCE_SHA256),
        ("feature_stats_sha256", "feature_stats.pt", EXPECTED_FEATURE_STATS_SHA256),
        ("split_manifest_sha256", "split_manifest.json", EXPECTED_SPLIT_MANIFEST_SHA256),
        ("units_sha256", "sae_units.jsonl", EXPECTED_UNITS_SHA256),
        (
            "activation_manifest_sha256",
            "manifest.json",
            EXPECTED_ACTIVATION_MANIFEST_SHA256,
        ),
    )
    for field_name, label, expected_hash in expected:
        if observed[field_name] != expected_hash:
            raise AtlasConfigError(
                f"{label} checksum does not match the approved source"
            )
    return AtlasSourceBinding(**observed, assignment_sha256=assignment)


def validate_source_binding(
    binding: AtlasSourceBinding,
    paths: AtlasPaths,
) -> None:
    """Recompute every bound byte hash and assignment provenance."""

    if not isinstance(binding, AtlasSourceBinding):
        raise TypeError("binding must be AtlasSourceBinding")
    if not isinstance(paths, AtlasPaths):
        raise TypeError("paths must be AtlasPaths")
    observed = _observed_hashes(paths)
    for field_name, label, _ in _source_files(paths):
        if observed[field_name] != getattr(binding, field_name):
            raise AtlasConfigError(f"{label} checksum does not match source binding")
    try:
        summary = validate_pilot_run(paths.run_dir)
    except Exception as error:
        raise AtlasConfigError(f"Invalid completed pilot run: {paths.run_dir}") from error
    assignment = _assignment_from_summary(summary, "pilot summary")
    if assignment != binding.assignment_sha256:
        summary_path = paths.run_dir / "summary.json"
        raise AtlasConfigError(
            f"summary.json assignment checksum does not match source binding: {summary_path}"
        )
