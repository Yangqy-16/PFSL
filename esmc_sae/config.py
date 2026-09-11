"""Immutable configuration and path-safety contract for the ESMC-SAE MVP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


LAYER = 60
D_MODEL = 2560
CODEBOOK_DIM = 16384
TOPK = 64
SAE_EPS = 1e-5

PROJECT_ROOT = Path("/data/qingyuyang/SL/PFSL/esmc_sae")
CODE_ROOT = Path("/data/qingyuyang/SL/PFSL")
ESMC_REPO = Path("/data/qingyuyang/esm")
ESMC_MODEL = CODE_ROOT / "pfsl_process" / "ESMC"
ESMC_SAE = ESMC_REPO / "sae_weights"
SEQUENCES = CODE_ROOT / "data" / "SLKB" / "all_id_seq.pkl"
SPLIT_ROOT = CODE_ROOT / "data" / "SLKB" / "specific_1:1"

_FIXED_PATHS = {
    "project_root": PROJECT_ROOT,
    "code_root": CODE_ROOT,
    "esmc_repo": ESMC_REPO,
    "esmc_model": ESMC_MODEL,
    "esmc_sae": ESMC_SAE,
    "sequences": SEQUENCES,
    "split_root": SPLIT_ROOT,
}
_OUTPUT_PATH_NAMES = ("dense_cache", "sparse_cache", "decoder_bridge")
_TOP_LEVEL_KEYS = {
    "seed",
    "paths",
    "experiment",
    "model",
    "cache",
    "explanation",
    "training",
}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{label} must be a mapping with string keys")
    return value


def _strict_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _strict_float(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _strict_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _absolute_path(value: object, label: str) -> Path:
    path = Path(_strict_string(value, label))
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return path


def _resolved(path: Path) -> Path:
    return path.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    resolved_path = _resolved(path)
    resolved_root = _resolved(root)
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _validate_output_path(
    name: str,
    path: Path,
    project_root: Path,
    split_root: Path,
) -> Path:
    if _is_within(path, split_root):
        raise ValueError(
            f"output path '{name}' must not be under read-only SPLIT_ROOT"
        )
    if not _is_within(path, project_root):
        raise ValueError(
            f"output path '{name}' must resolve under PROJECT_ROOT"
        )
    return path


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    code_root: Path
    esmc_repo: Path
    esmc_model: Path
    esmc_sae: Path
    sequences: Path
    split_root: Path
    dense_cache: Path
    sparse_cache: Path
    decoder_bridge: Path
    reports: Path
    additional_outputs: tuple[tuple[str, Path], ...] = ()

    @classmethod
    def from_mapping(cls, value: object) -> "PathsConfig":
        raw = _mapping(value, "paths")
        missing = [
            name
            for name in (*_FIXED_PATHS, *_OUTPUT_PATH_NAMES)
            if name not in raw
        ]
        if missing:
            raise ValueError(f"paths is missing required path: {missing[0]}")

        fixed: dict[str, Path] = {}
        for name, expected in _FIXED_PATHS.items():
            actual = _absolute_path(raw[name], f"paths.{name}")
            if _resolved(actual) != _resolved(expected):
                raise ValueError(
                    f"paths.{name} must be the fixed dependency path {expected}"
                )
            fixed[name] = actual

        output_names = [
            name for name in raw if name not in _FIXED_PATHS
        ]
        outputs: dict[str, Path] = {}
        for name in output_names:
            path = _absolute_path(raw[name], f"paths.{name}")
            outputs[name] = _validate_output_path(
                name,
                path,
                fixed["project_root"],
                fixed["split_root"],
            )

        reports = outputs.pop("reports", fixed["project_root"] / "reports")
        required_outputs = {
            name: outputs.pop(name) for name in _OUTPUT_PATH_NAMES
        }
        return cls(
            **fixed,
            **required_outputs,
            reports=reports,
            additional_outputs=tuple(sorted(outputs.items())),
        )

    def output_path(self, name: str) -> Path:
        if name in (*_OUTPUT_PATH_NAMES, "reports"):
            return getattr(self, name)
        try:
            return dict(self.additional_outputs)[name]
        except KeyError as error:
            raise KeyError(f"Unknown output path: {name}") from error


@dataclass(frozen=True)
class ExperimentConfig:
    cell_line: str
    scene: str
    fold: int

    @classmethod
    def from_mapping(cls, value: object) -> "ExperimentConfig":
        raw = _mapping(value, "experiment")
        return cls(
            cell_line=_strict_string(raw.get("cell_line"), "experiment.cell_line"),
            scene=_strict_string(raw.get("scene"), "experiment.scene"),
            fold=_strict_int(raw.get("fold"), "experiment.fold", minimum=0),
        )


@dataclass(frozen=True)
class ModelConfig:
    layer: int
    d_model: int
    codebook_dim: int
    topk: int
    sae_eps: float

    @classmethod
    def from_mapping(cls, value: object) -> "ModelConfig":
        raw = _mapping(value, "model")
        expected = {
            "layer": LAYER,
            "d_model": D_MODEL,
            "codebook_dim": CODEBOOK_DIM,
            "topk": TOPK,
            "sae_eps": SAE_EPS,
        }
        parsed = {
            "layer": _strict_int(raw.get("layer"), "model.layer"),
            "d_model": _strict_int(raw.get("d_model"), "model.d_model"),
            "codebook_dim": _strict_int(
                raw.get("codebook_dim"), "model.codebook_dim"
            ),
            "topk": _strict_int(raw.get("topk"), "model.topk"),
            "sae_eps": _strict_float(raw.get("sae_eps"), "model.sae_eps"),
        }
        for name, expected_value in expected.items():
            if parsed[name] != expected_value:
                raise ValueError(
                    f"model.{name} must equal the fixed value {expected_value}"
                )
        return cls(**parsed)


@dataclass(frozen=True)
class CacheConfig:
    dense_dtype: str
    sparse_value_dtype: str
    extraction_batch_size: int

    @classmethod
    def from_mapping(cls, value: object) -> "CacheConfig":
        raw = _mapping(value, "cache")
        dense_dtype = _strict_string(
            raw.get("dense_dtype"), "cache.dense_dtype"
        )
        sparse_dtype = _strict_string(
            raw.get("sparse_value_dtype"), "cache.sparse_value_dtype"
        )
        if dense_dtype != "float16" or sparse_dtype != "float16":
            raise ValueError("cache tensor dtypes must be float16 for the MVP")
        return cls(
            dense_dtype=dense_dtype,
            sparse_value_dtype=sparse_dtype,
            extraction_batch_size=_strict_int(
                raw.get("extraction_batch_size"),
                "cache.extraction_batch_size",
                minimum=1,
            ),
        )


@dataclass(frozen=True)
class ExplanationConfig:
    candidate_ceiling: int
    candidate_choices: tuple[int, ...]
    captured_effect_target: float
    matched_controls_per_side: int
    enable_patching: bool

    @classmethod
    def from_mapping(cls, value: object) -> "ExplanationConfig":
        raw = _mapping(value, "explanation")
        raw_choices = raw.get("candidate_choices")
        if not isinstance(raw_choices, (list, tuple)) or not raw_choices:
            raise ValueError("explanation.candidate_choices must be a sequence")
        choices = tuple(
            _strict_int(choice, "explanation.candidate_choices", minimum=1)
            for choice in raw_choices
        )
        target = _strict_float(
            raw.get("captured_effect_target"),
            "explanation.captured_effect_target",
        )
        if not 0.0 < target <= 1.0:
            raise ValueError(
                "explanation.captured_effect_target must be in (0, 1]"
            )
        patching = raw.get("enable_patching")
        if type(patching) is not bool:
            raise ValueError("explanation.enable_patching must be boolean")
        return cls(
            candidate_ceiling=_strict_int(
                raw.get("candidate_ceiling"),
                "explanation.candidate_ceiling",
                minimum=1,
            ),
            candidate_choices=choices,
            captured_effect_target=target,
            matched_controls_per_side=_strict_int(
                raw.get("matched_controls_per_side"),
                "explanation.matched_controls_per_side",
                minimum=1,
            ),
            enable_patching=patching,
        )


@dataclass(frozen=True)
class TrainingConfig:
    base_config: Path
    dataset_name: str
    esm_root: Path
    train_file: Path
    val_file: Path
    test_file: Path
    output_dir: Path

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        paths: PathsConfig,
        experiment: ExperimentConfig,
    ) -> "TrainingConfig":
        raw = _mapping(value, "training")
        required = {
            "base_config",
            "dataset_name",
            "esm_root",
            "train_file",
            "val_file",
            "test_file",
            "output_dir",
        }
        missing = sorted(required.difference(raw))
        if missing:
            raise ValueError(f"training is missing required key: {missing[0]}")
        unknown = sorted(set(raw).difference(required))
        if unknown:
            raise ValueError(f"training has unknown key: {unknown[0]}")

        split_dir = (
            paths.split_root
            / experiment.cell_line
            / experiment.scene
        )
        expected = {
            "base_config": (
                paths.code_root / "esm4sl" / "configuration" / "prosl.yaml"
            ),
            "esm_root": paths.dense_cache,
            "train_file": split_dir / f"sl_train_{experiment.fold}.csv",
            "val_file": split_dir / f"sl_val_{experiment.fold}.csv",
            "test_file": split_dir / f"sl_test_{experiment.fold}.csv",
            "output_dir": (
                paths.project_root
                / "experiment2"
                / "model"
                / experiment.cell_line
                / experiment.scene
                / f"fold_{experiment.fold}"
            ),
        }
        parsed = {
            name: _absolute_path(raw[name], f"training.{name}")
            for name in expected
        }
        for name, expected_path in expected.items():
            if _resolved(parsed[name]) != _resolved(expected_path):
                raise ValueError(
                    f"training.{name} must equal the fixed path "
                    f"{expected_path}"
                )

        dataset_name = _strict_string(
            raw["dataset_name"],
            "training.dataset_name",
        )
        if dataset_name != "ESMCSAELayer60ProtDataset":
            raise ValueError(
                "training.dataset_name must equal "
                "ESMCSAELayer60ProtDataset"
            )
        return cls(dataset_name=dataset_name, **parsed)


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    paths: PathsConfig
    experiment: ExperimentConfig
    model: ModelConfig
    cache: CacheConfig
    explanation: ExplanationConfig
    training: TrainingConfig

    @classmethod
    def from_mapping(cls, value: object) -> "ProjectConfig":
        raw = _mapping(value, "configuration")
        missing = sorted(_TOP_LEVEL_KEYS.difference(raw))
        if missing:
            raise ValueError(f"configuration is missing required key: {missing[0]}")
        unknown = sorted(set(raw).difference(_TOP_LEVEL_KEYS))
        if unknown:
            raise ValueError(f"configuration has unknown key: {unknown[0]}")
        paths = PathsConfig.from_mapping(raw["paths"])
        experiment = ExperimentConfig.from_mapping(raw["experiment"])
        return cls(
            seed=_strict_int(raw["seed"], "seed", minimum=0),
            paths=paths,
            experiment=experiment,
            model=ModelConfig.from_mapping(raw["model"]),
            cache=CacheConfig.from_mapping(raw["cache"]),
            explanation=ExplanationConfig.from_mapping(raw["explanation"]),
            training=TrainingConfig.from_mapping(
                raw["training"],
                paths=paths,
                experiment=experiment,
            ),
        )


def load_config(path: Path) -> ProjectConfig:
    """Load a YAML project configuration into the immutable schema."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ProjectConfig.from_mapping(payload)
