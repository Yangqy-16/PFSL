"""Thin layer-60 cache adapter for the existing ProSL dataset."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.states import RunningStage
import torch

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY
from pfsl.dataset.prosl import CollateBatch, ProtDataset


@DATASET_REGISTRY.register()
class ESMCSAELayer60ProtDataset(ProtDataset):
    """Reuse ProtDataset while replacing only its sequence-cache loading."""

    @configurable
    def __init__(self, cfg: DictConfig, stage: RunningStage) -> None:
        self._layer60_cfg = cfg
        parent_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True)
        )
        parent_cfg.MODEL.USE_SEQ = False
        super().__init__(cfg=parent_cfg, stage=stage)

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        stage: RunningStage,
    ) -> dict[str, object]:
        return {"cfg": cfg, "stage": stage}

    def _load_sequence_cache(self, gene_id: int) -> torch.Tensor:
        path = Path(self._layer60_cfg.DATASET.ESM_ROOT) / f"{gene_id}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        tensor = payload["h60"]
        return tensor.to(device="cpu", dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, object]:
        data = super().__getitem__(index)
        if self._layer60_cfg.MODEL.USE_SEQ:
            gene1 = self._load_sequence_cache(data["g1_idx"])
            gene2 = self._load_sequence_cache(data["g2_idx"])
            if not self._layer60_cfg.MODEL.SEQ_USE_ATT:
                gene1 = gene1.mean(dim=0)
                gene2 = gene2.mean(dim=0)
            data["g1_seq_emb"] = gene1
            data["g2_seq_emb"] = gene2
        return data

    @property
    def collate_fn(self) -> CollateBatch:
        return CollateBatch(self._layer60_cfg)
