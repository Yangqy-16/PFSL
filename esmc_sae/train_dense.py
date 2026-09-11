"""Bind the existing coach_pl ProSL trainer to the layer-60 cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

from omegaconf import DictConfig, OmegaConf
import torch

from coach_pl.configuration import CfgNode
from coach_pl.tool import train as coach_train
from esmc_sae.artifacts import save_json
from esmc_sae.config import ProjectConfig, load_config
import esmc_sae.dataset  # noqa: F401  # register the thin adapter


_SMOKE_OVERRIDE_KEYS = frozenset(
    {
        "DATALOADER.TRAIN.BATCH_SIZE",
        "DATALOADER.VAL.BATCH_SIZE",
        "DATALOADER.TEST.BATCH_SIZE",
        "DATALOADER.TRAIN.NUM_WORKERS",
        "DATALOADER.VAL.NUM_WORKERS",
        "DATALOADER.TEST.NUM_WORKERS",
        "OUTPUT_DIR",
        "TRAINER.LOG_EVERY_N_STEPS",
        "TRAINER.MAX_EPOCHS",
    }
)


def _validate_overrides(opts: Sequence[str]) -> None:
    if len(opts) % 2:
        raise ValueError("coach overrides must be KEY VALUE pairs")
    unsupported = sorted(set(opts[::2]).difference(_SMOKE_OVERRIDE_KEYS))
    if unsupported:
        raise ValueError(f"unsupported coach override: {unsupported[0]}")
    pairs = dict(zip(opts[::2], opts[1::2]))
    if opts and "OUTPUT_DIR" not in pairs:
        raise ValueError(
            "non-empty coach overrides require an explicit smoke OUTPUT_DIR"
        )
    if "OUTPUT_DIR" in pairs:
        output_dir = Path(pairs["OUTPUT_DIR"]).resolve(strict=False)
        tmp_root = Path("/tmp").resolve(strict=False)
        if (
            not output_dir.is_relative_to(tmp_root)
            or output_dir == tmp_root
            or not output_dir.name.startswith("esmc_sae_t04_smoke_")
        ):
            raise ValueError(
                "smoke OUTPUT_DIR must be a scoped "
                "/tmp/esmc_sae_t04_smoke_* directory"
            )


def build_coach_config(
    project: ProjectConfig,
    opts: Sequence[str],
) -> DictConfig:
    """Create the trusted ProSL config with only Task-4 bindings changed."""

    _validate_overrides(opts)
    cfg = CfgNode.load_yaml_with_base(str(project.training.base_config))
    cfg.SEED = project.seed
    cfg.OUTPUT_DIR = str(project.training.output_dir)
    cfg.DATASET.NAME = project.training.dataset_name
    cfg.DATASET.ESM_ROOT = str(project.training.esm_root)
    cfg.DATASET.TRAIN_FILE = str(project.training.train_file)
    cfg.DATASET.VAL_FILE = str(project.training.val_file)
    cfg.DATASET.TEST_FILE = str(project.training.test_file)
    cfg.DATASET.CL = project.experiment.cell_line
    CfgNode.merge_with_dotlist(cfg, list(opts))
    return cfg


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_test_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for name, value in row.items():
                if name.startswith("test/") and value:
                    metrics[name] = float(value)
    if not metrics:
        raise RuntimeError(f"no test metrics found in {path}")
    return dict(sorted(metrics.items()))


def finalize_training_artifacts(
    output_dir: Path,
    *,
    peak_vram_bytes: int,
) -> dict[str, object]:
    """Compact one successful coach run inside its explicit output directory."""

    output_dir = Path(output_dir)
    checkpoints = sorted((output_dir / "best_ckpts").glob("*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"expected exactly one best checkpoint, found {len(checkpoints)}"
        )
    checkpoint = checkpoints[0]
    torch.load(checkpoint, map_location="cpu", weights_only=False)

    csv_versions = sorted(
        path
        for path in (output_dir / "csv_log").glob("version_*")
        if (path / "test_logits.csv").is_file()
        and (path / "metrics.csv").is_file()
    )
    if len(csv_versions) != 1:
        raise RuntimeError(
            "expected exactly one completed CSV logger version, "
            f"found {len(csv_versions)}"
        )
    csv_version = csv_versions[0]
    predictions = output_dir / "test_predictions.csv"
    shutil.copy2(csv_version / "test_logits.csv", predictions)

    result: dict[str, object] = {
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_cpu_reload": True,
        "metrics": _final_test_metrics(csv_version / "metrics.csv"),
        "peak_vram_bytes": int(peak_vram_bytes),
        "prediction_csv": str(predictions),
        "prediction_csv_sha256": _sha256(predictions),
    }

    for name in ("csv_log", "tb_log", "regular_ckpts"):
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
    save_json(output_dir / "metrics.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Restricted smoke overrides as KEY VALUE pairs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project = load_config(args.config)
    cfg = build_coach_config(project, args.opts)
    output_dir = Path(cfg.OUTPUT_DIR)
    if (
        not args.opts
        and output_dir.exists()
        and (not output_dir.is_dir() or any(output_dir.iterdir()))
    ):
        raise RuntimeError(
            f"formal output directory already exists and is not empty: "
            f"{output_dir}"
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix="esmc-sae-train-") as temp_dir:
        coach_config = Path(temp_dir) / "coach_config.yaml"
        OmegaConf.save(config=cfg, f=coach_config)
        coach_train.main(
            argparse.Namespace(
                config_file=coach_config,
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
    print(
        "DENSE_TRAIN_RESULT "
        + json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
