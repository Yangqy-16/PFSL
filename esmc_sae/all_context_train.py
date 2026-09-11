"""Train the single-seed all-context BRCA2--PARP1 case model."""

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
from esmc_sae.artifacts import save_json
from esmc_sae.case_train import normalize_checkpoint_to_cpu
from esmc_sae.train_dense import finalize_training_artifacts
import esmc_sae.all_context_dataset  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "esm4sl/configuration/prosl.yaml"
DATA_ROOT = ROOT / "data/SLKB/CASE_BRCA2_PARP1_ALL_CONTEXT"
POOL_ROOT = ROOT / "esmc_sae/cache/layer60_mean_all_context"
MODEL_ROOT = (
    ROOT
    / "esmc_sae/case_study/ovcar8_brca2_parp1_all_context/models"
)
NAME_TO_ID = ROOT / "data/mapping/name2id.pkl"


def build_all_context_coach_config(
    *,
    data_root: Path = DATA_ROOT,
    pool_root: Path = POOL_ROOT,
    output_root: Path = MODEL_ROOT,
    max_epochs: int = 50,
) -> DictConfig:
    """Build the one-seed ESMC-only, cell-context-conditioned config."""

    cfg = CfgNode.load_yaml_with_base(str(BASE_CONFIG))
    cfg.SEED = 42
    cfg.OUTPUT_DIR = str(Path(output_root) / "seed_42")
    cfg.DATASET.NAME = "AllContextESMCPooledDataset"
    cfg.DATASET.POOLED_ESM_ROOT = str(pool_root)
    cfg.DATASET.GENE_NAME_TO_ID = str(NAME_TO_ID)
    cfg.DATASET.TRAIN_FILE = str(Path(data_root) / "sl_train.csv")
    cfg.DATASET.VAL_FILE = str(Path(data_root) / "sl_val.csv")
    cfg.DATASET.TEST_FILE = str(Path(data_root) / "sl_val.csv")
    cfg.DATASET.CL = "MIXED"

    cfg.MODEL.USE_SEQ = True
    cfg.MODEL.SEQ_USE_ATT = False
    cfg.MODEL.USE_FUNC = False
    cfg.MODEL.FUNC_USE_ATT = False
    cfg.MODEL.USE_CL = False
    cfg.MODEL.USE_OMICS = True

    cfg.DATALOADER.SAMPLE = True
    cfg.DATALOADER.TRAIN.BATCH_SIZE = 512
    cfg.DATALOADER.VAL.BATCH_SIZE = 512
    cfg.DATALOADER.TEST.BATCH_SIZE = 512
    cfg.DATALOADER.TRAIN.NUM_WORKERS = 8
    cfg.DATALOADER.VAL.NUM_WORKERS = 8
    cfg.DATALOADER.TEST.NUM_WORKERS = 8
    cfg.TRAINER.MAX_EPOCHS = int(max_epochs)
    cfg.TRAINER.DETERMINISTIC = True
    cfg.TRAINER.BENCHMARK = False
    cfg.TRAINER.LOG_EVERY_N_STEPS = 50
    return cfg


def _validate_inputs(cfg: DictConfig) -> None:
    for field in ("TRAIN_FILE", "VAL_FILE", "TEST_FILE"):
        path = Path(cfg.DATASET[field])
        if not path.is_file():
            raise FileNotFoundError(path)
    if not Path(cfg.DATASET.POOLED_ESM_ROOT).is_dir():
        raise FileNotFoundError(cfg.DATASET.POOLED_ESM_ROOT)
    if not Path(cfg.DATASET.GENE_NAME_TO_ID).is_file():
        raise FileNotFoundError(cfg.DATASET.GENE_NAME_TO_ID)


def train_all_context(
    *,
    data_root: Path = DATA_ROOT,
    pool_root: Path = POOL_ROOT,
    output_root: Path = MODEL_ROOT,
    max_epochs: int = 50,
) -> dict[str, object]:
    """Train, test on validation, and compact the single formal run."""

    cfg = build_all_context_coach_config(
        data_root=data_root,
        pool_root=pool_root,
        output_root=output_root,
        max_epochs=max_epochs,
    )
    _validate_inputs(cfg)
    output_dir = Path(cfg.OUTPUT_DIR)
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise RuntimeError(
            "formal output directory already exists and is not empty: "
            f"{output_dir}"
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(
        prefix="esmc-sae-all-context-seed-42-"
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
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--pool-root", type=Path, default=POOL_ROOT)
    parser.add_argument("--output-root", type=Path, default=MODEL_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = train_all_context(
        data_root=args.data_root,
        pool_root=args.pool_root,
        output_root=args.output_root,
        max_epochs=args.max_epochs,
    )
    print(
        "ALL_CONTEXT_TRAIN_RESULT "
        + json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
