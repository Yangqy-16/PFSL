from pathlib import Path
import random
from typing import Any, Optional
import pickle

import torch
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY
from .sl import SLDataset

__all__ = ["SLrawDataset"]


class CollateBatch():
    def __init__(self):
        pass

    def __call__(self, batch: list):
        assert type(batch) == list

        outputs = {"gene1": [], "gene2": [], "label": []}
        for (gene1, seq1), (gene2, seq2), label in batch:
            outputs["gene1"].append((gene1, seq1))
            outputs["gene2"].append((gene2, seq2))
            outputs["label"].append(label)
        outputs["label"] = torch.tensor(outputs["label"])
        return outputs


@DATASET_REGISTRY.register()
class SLrawDataset(SLDataset):
    """
    Dataset for SL prediction.
    Each pair of SL genes generates: (cmb_gene_emb, label)
    """

    @configurable
    def __init__(self, cfg: DictConfig, stage: RunningStage) -> None:
        super().__init__(cfg, stage)
        
        with open(cfg.DATASET.GENE_SEQ_PATH, 'rb') as f:
            self.dic = dict(pickle.load(f))

    def __getitem__(self, index: int) -> tuple[tuple[str, str], tuple[str, str], float]:
        # NOTE: change df's col name if needed
        gene1 = self.df['0'][index]
        gene2 = self.df['1'][index]
        label = float(self.df['2'][index])

        seq1 = self.dic[gene1]
        seq2 = self.dic[gene2]

        if self.stage == RunningStage.TRAINING and random.uniform(0, 1) > 0.5:  # swap 1 and 2 in randomly
            seq1, seq2 = seq2, seq1
            gene1, gene2 = gene2, gene1

        return (gene1, seq1), (gene2, seq2), label

    # For DataLoader
    collate_fn = CollateBatch()
