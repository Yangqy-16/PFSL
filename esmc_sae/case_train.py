"""Train one OVCAR8 BRCA2--PARP1 case-study fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Sequence

from omegaconf import DictConfig, OmegaConf
import torch

from coach_pl.configuration import CfgNode
from coach_pl.tool import train as coach_train
from esmc_sae.artifacts import save_cpu_torch, save_json
import esmc_sae.dataset  # noqa: F401  # register layer-60 dataset adapter
from esmc_sae.manifest import file_sha256
from esmc_sae.train_dense import finalize_training_artifacts


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "esm4sl/configuration/prosl.yaml"
DATA_ROOT = ROOT / "data/SLKB/CASE_BRCA2_PARP1"
CACHE_ROOT = ROOT / "esmc_sae/cache/layer60_dense"
MODEL_ROOT = (
    ROOT / "esmc_sae/case_study/ovcar8_brca2_parp1/models"
)


def build_case_coach_config(
    fold: int,
    *,
    output_root: Path = MODEL_ROOT,
) -> DictConfig:
    """Reuse ``prosl.yaml`` with only case-study-specific overrides."""

    if fold not in range(5):
        raise ValueError(f"fold must be in [0, 4], got {fold}")

    cfg = CfgNode.load_yaml_with_base(str(BASE_CONFIG))
    cfg.SEED = 42 + fold
    cfg.OUTPUT_DIR = str(Path(output_root) / f"fold_{fold}")
    cfg.DATASET.NAME = "ESMCSAELayer60ProtDataset"
    cfg.DATASET.ESM_ROOT = str(CACHE_ROOT)
    cfg.DATASET.TRAIN_FILE = str(DATA_ROOT / f"sl_train_{fold}.csv")
    cfg.DATASET.VAL_FILE = str(DATA_ROOT / f"sl_val_{fold}.csv")
    cfg.DATASET.TEST_FILE = str(DATA_ROOT / f"sl_test_{fold}.csv")
    cfg.DATASET.CL = "OVCAR8"
    cfg.DATALOADER.SAMPLE = True
    cfg.TRAINER.DETERMINISTIC = True
    cfg.TRAINER.BENCHMARK = False
    cfg.TRAINER.LOG_EVERY_N_STEPS = 10
    return cfg


def normalize_checkpoint_to_cpu(path: Path) -> str:
    """Rewrite a Lightning checkpoint with every tensor stored on CPU."""

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    save_cpu_torch(path, payload)
    torch.load(path, map_location="cpu", weights_only=False)
    if b"cuda:" in path.read_bytes():
        raise RuntimeError(f"CUDA storage metadata remains in {path}")
    return file_sha256(path)


def _validate_fold_inputs(cfg: DictConfig) -> None:
    for field in ("TRAIN_FILE", "VAL_FILE", "TEST_FILE"):
        path = Path(cfg.DATASET[field])
        if not path.is_file():
            raise FileNotFoundError(path)
    if not CACHE_ROOT.is_dir():
        raise FileNotFoundError(CACHE_ROOT)


def train_case_fold(
    fold: int,
    *,
    output_root: Path = MODEL_ROOT,
) -> dict[str, object]:
    """Train and compact one fold without changing the shared base config."""

    cfg = build_case_coach_config(fold, output_root=output_root)
    _validate_fold_inputs(cfg)
    output_dir = Path(cfg.OUTPUT_DIR)
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise RuntimeError(
            f"formal output directory already exists and is not empty: "
            f"{output_dir}"
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(
        prefix=f"esmc-sae-case-fold-{fold}-"
    ) as temp_dir:
        config_path = Path(temp_dir) / "coach_config.yaml"
        OmegaConf.save(config=cfg, f=config_path, resolve=True)
        coach_train.main(
            argparse.Namespace(
                config_file=config_path,
                resume=None,
                num_gpus=1,
                num_nodes=1,
                opts=[],
            )
        )

    peak_vram_bytes = (
        torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else 0
    )
    result = finalize_training_artifacts(
        output_dir,
        peak_vram_bytes=peak_vram_bytes,
    )
    checkpoint = Path(str(result["best_checkpoint"]))
    result["best_checkpoint_sha256"] = normalize_checkpoint_to_cpu(checkpoint)
    result["checkpoint_cpu_storage"] = True
    result["fold"] = fold
    result["seed"] = int(cfg.SEED)
    OmegaConf.save(
        config=cfg,
        f=output_dir / "resolved_config.yaml",
        resolve=True,
    )
    save_json(output_dir / "metrics.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = train_case_fold(args.fold)
    print(
        "CASE_TRAIN_RESULT "
        + json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
