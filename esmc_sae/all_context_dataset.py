"""Mixed-context dataset backed by in-memory pooled ESMC embeddings."""

from __future__ import annotations

from pathlib import Path
import pickle
import random

from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
import torch

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY
from pfsl.dataset.prosl import CollateBatch
from pfsl.dataset.sl import SLDataset


OMICS_KINDS = ("cnv", "ess", "exp", "mut")


def _load_name_to_id(path: Path) -> dict[str, int]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("gene-name mapping must be a dictionary")
    return {str(name): int(gene_id) for name, gene_id in payload.items()}


def _load_omics(
    path: Path,
    name_to_id: dict[str, int],
) -> dict[int, float]:
    values: dict[int, float] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"invalid omics row in {path}")
            gene_id = name_to_id.get(fields[0])
            if gene_id is not None:
                values[gene_id] = float(fields[1])
    if not values:
        raise ValueError(f"omics file has no mapped genes: {path}")
    return values


@DATASET_REGISTRY.register()
class AllContextESMCPooledDataset(SLDataset):
    """Use each row's context to pair pooled ESMC with four omics values."""

    @configurable
    def __init__(self, cfg: DictConfig, stage: RunningStage) -> None:
        super().__init__(cfg, stage)
        if (
            not cfg.MODEL.USE_SEQ
            or not cfg.MODEL.USE_OMICS
            or cfg.MODEL.USE_FUNC
            or cfg.MODEL.USE_CL
            or cfg.MODEL.SEQ_USE_ATT
        ):
            raise ValueError(
                "AllContextESMCPooledDataset requires pooled ESMC + omics only"
            )
        required = {"context", "0", "1", "2"}
        if not required <= set(self.df.columns):
            raise ValueError(f"mixed-context CSV lacks {sorted(required)}")

        pooled_root = Path(cfg.DATASET.POOLED_ESM_ROOT)
        gene_ids = sorted(set(self.df["0"]) | set(self.df["1"]))
        self.esmc: dict[int, torch.Tensor] = {}
        for gene_id in gene_ids:
            path = pooled_root / f"{gene_id}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or "mean_h60" not in payload:
                raise ValueError(f"invalid pooled ESMC cache: {path}")
            tensor = payload["mean_h60"].to(dtype=torch.float32, device="cpu")
            if tensor.ndim != 1 or not torch.isfinite(tensor).all():
                raise ValueError(f"invalid pooled ESMC tensor: {path}")
            self.esmc[int(gene_id)] = tensor

        self.contexts = tuple(sorted(str(value) for value in self.df["context"].unique()))
        name_to_id = _load_name_to_id(Path(cfg.DATASET.GENE_NAME_TO_ID))
        omics_root = Path(cfg.DATASET.OMICS_ROOT)
        self.omics: dict[str, dict[str, dict[int, float]]] = {}
        self.omics_means: dict[str, dict[str, float]] = {}
        for context in self.contexts:
            by_kind = {
                kind: _load_omics(
                    omics_root / f"{context}_{kind}.txt",
                    name_to_id,
                )
                for kind in OMICS_KINDS
            }
            self.omics[context] = by_kind
            self.omics_means[context] = {
                kind: sum(values.values()) / len(values)
                for kind, values in by_kind.items()
            }

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        stage: RunningStage,
    ) -> dict[str, object]:
        return {"cfg": cfg, "stage": stage}

    def _gene_omics(self, context: str, gene_id: int) -> torch.Tensor:
        return torch.tensor(
            [
                self.omics[context][kind].get(
                    gene_id,
                    self.omics_means[context][kind],
                )
                for kind in OMICS_KINDS
            ],
            dtype=torch.float32,
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.df.iloc[index]
        gene_a = int(row["0"])
        gene_b = int(row["1"])
        if self.stage == RunningStage.TRAINING and random.random() > 0.5:
            gene_a, gene_b = gene_b, gene_a
        context = str(row["context"])
        return {
            "g1_idx": gene_a,
            "g2_idx": gene_b,
            "label": float(row["2"]),
            "g1_seq_emb": self.esmc[gene_a],
            "g2_seq_emb": self.esmc[gene_b],
            "g1_omics": self._gene_omics(context, gene_a),
            "g2_omics": self._gene_omics(context, gene_b),
        }

    @property
    def collate_fn(self) -> CollateBatch:
        return CollateBatch(self.cfg)
