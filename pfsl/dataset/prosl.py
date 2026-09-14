import os
import pickle
import random
from typing import Any

import json
import numpy as np
import torch
from torch import Tensor
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY
from .sl import SLDataset

# FUNC_MAP = [
#     'Protein names',
#     'Gene Names',
#     'Activity regulation',
#     'Developmental stage',
#     'Induction',
#     'Gene Ontology (biological process)',
#     'Gene Ontology (cellular component)',
#     'Gene Ontology (molecular function)',
#     'Involvement in disease',
#     'Subcellular location [CC]',
#     'Domain [CC]',
#     'Pathway',
#     'Cofactor',
#     'Catalytic activity',
#     'EC number',
#     'Caution',
#     'Function [CC]',
#     'Miscellaneous [CC]',
#     'Polymorphism',
#     'Post-translational modification',
#     'Proteomes',
#     'Sequence similarities',
#     'Subunit structure',
#     'Tissue specificity'
# ]

def mean(iterable: list[float]) -> float:
    return sum(iterable) / len(iterable)


class CollateBatch():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def padding(self, genes: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        count_len = [g.shape[0] for g in genes]
        max_len = self.cfg.DATASET.MAX_LEN

        batched_sample = torch.zeros((len(genes), max_len, genes[0].shape[1]))
        for i, (gene, length) in enumerate(zip(genes, count_len)):
            if length <= max_len:
                batched_sample[i, :length, :] = gene
            else:
                batched_sample[i, :max_len, :] = gene[:max_len, :]
        batched_sample = batched_sample.to(torch.float32)

        prot_mask = torch.ones((len(genes), max_len), dtype=bool)
        for i, seq_len in enumerate(count_len):
            if seq_len <= max_len:
                prot_mask[i, :seq_len] = 0
            else:
                prot_mask[i, :] = 0

        return batched_sample, prot_mask

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Tensor]:
        data = {}

        data['g1_idx'] = torch.tensor([d['g1_idx'] for d in batch])
        data['g2_idx'] = torch.tensor([d['g2_idx'] for d in batch])
        data['label'] = torch.tensor([d['label'] for d in batch])

        if self.cfg.MODEL.USE_SEQ:
            if self.cfg.MODEL.SEQ_USE_ATT:
                data['g1_seq_emb'], data['g1_seq_mask'] = self.padding([d['g1_seq_emb'] for d in batch])
                data['g2_seq_emb'], data['g2_seq_mask'] = self.padding([d['g2_seq_emb'] for d in batch])
            else:
                data['g1_seq_emb'] = torch.stack([d['g1_seq_emb'] for d in batch])
                data['g2_seq_emb'] = torch.stack([d['g2_seq_emb'] for d in batch])

        if self.cfg.MODEL.USE_FUNC:
            data['g1_func_emb'] = torch.stack([d['g1_func_emb'] for d in batch])
            data['g2_func_emb'] = torch.stack([d['g2_func_emb'] for d in batch])
            if self.cfg.MODEL.FUNC_USE_ATT:
                data['g1_func_mask'] = torch.stack([d['g1_func_mask'] for d in batch]) if batch[0]['g1_func_mask'] is not None else None
                data['g2_func_mask'] = torch.stack([d['g2_func_mask'] for d in batch]) if batch[0]['g2_func_mask'] is not None else None
        
        if self.cfg.MODEL.USE_CL:
            data['cl_emb'] = torch.stack([d['cl_emb'] for d in batch])
        
        if self.cfg.MODEL.USE_OMICS:
            data['g1_omics'] = torch.stack([d['g1_omics'] for d in batch])
            data['g2_omics'] = torch.stack([d['g2_omics'] for d in batch])

        return data                


@DATASET_REGISTRY.register()
class ProtDataset(SLDataset):
    @configurable
    def __init__(self, cfg: DictConfig, stage: RunningStage) -> None:
        super().__init__(cfg, stage)

        if cfg.MODEL.USE_FUNC:
            self.text_emb_root = cfg.DATASET.TEXT_EMB_ROOT
            with open(cfg.DATASET.TEXT_PATH, 'r') as f:
                self.text = json.load(f)

            # aspect_total_num = self.cfg.DATASET.ASPECT_NUM
            # unuse_aspects = [int(i) for i in str(self.cfg.MODEL.FUNC_UNUSE_ASPECT).split(',')] if self.cfg.MODEL.FUNC_UNUSE_ASPECT is not None else []
            self.use_aspects = list(range(24)) if self.cfg.MODEL.FUNC_USE_ASPECT is None \
                else [int(i) for i in str(self.cfg.MODEL.FUNC_USE_ASPECT).split(',')] #sorted(list(set(range(aspect_total_num)).difference(set(unuse_aspects))))
        
        if cfg.MODEL.USE_OMICS:
            with open('/data/qingyuyang/SL/PFSL/data/mapping/name2id.pkl', 'rb') as f:
                gn2gid = {key: int(value) for key, value in pickle.load(f).items()}
            
            omics_root, cl = cfg.DATASET.OMICS_ROOT, cfg.DATASET.CL
            
            with open(f'{omics_root}/{cl}_cnv.txt', 'r') as f:
                self.cnv = dict([l.split() for l in f.readlines()])
            self.cnv = {gn2gid[key]: float(value) for key, value in self.cnv.items() if key in gn2gid}

            with open(f'{omics_root}/{cl}_ess.txt', 'r') as f:
                self.ess = dict([l.split() for l in f.readlines()])
            self.ess = {gn2gid[key]: float(value) for key, value in self.ess.items() if key in gn2gid}

            with open(f'{omics_root}/{cl}_exp.txt', 'r') as f:
                self.exp = dict([l.split() for l in f.readlines()])
            self.exp = {gn2gid[key]: float(value) for key, value in self.exp.items() if key in gn2gid}

            with open(f'{omics_root}/{cl}_mut.txt', 'r') as f:
                self.mut = dict([l.split() for l in f.readlines()])
            self.mut = {gn2gid[key]: float(value) for key, value in self.mut.items() if key in gn2gid}

    def text_emb(self, gid: int):
        if os.path.exists(os.path.join(self.text_emb_root, f"{gid}.pt")):
            g_func = torch.load(os.path.join(self.text_emb_root, f"{gid}.pt"), map_location='cpu', weights_only=True)  # [24, 1024]
            
            use_indices = torch.tensor(self.use_aspects, dtype=torch.long)
            g_func = g_func.index_select(0, use_indices)
        
            g_text = self.text[str(gid)]
            g_text = [g_text[i] for i in self.use_aspects]

            g_mask = torch.zeros(len(self.use_aspects), dtype=torch.bool)

            for i in range(len(g_text)):
                if g_text[i] == '':
                    g_mask[i] = True
        else:
            g_func = torch.randn((len(self.use_aspects), 1024))  # Random initialization
            g_mask = torch.ones(len(self.use_aspects), dtype=torch.bool)

        if self.cfg.MODEL.FUNC_USE_ATT:
            return g_func, g_mask
        else:
            valid_mask = ~g_mask
            if valid_mask.any():
                g_func = g_func[valid_mask].mean(dim=0)
            else:
                g_func = g_func.mean(dim=0)
            return g_func, None

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        data = {}

        g1_idx = self.df['0'][index]
        g2_idx = self.df['1'][index]
        label = float(self.df['2'][index])

        if self.stage == RunningStage.TRAINING and random.uniform(0, 1) > 0.5:  # concat 1 and 2 in random order
            g2_idx, g1_idx = g1_idx, g2_idx

        data['g1_idx'], data['g2_idx'], data['label'] = g1_idx, g2_idx, label

        if self.cfg.MODEL.USE_SEQ:
            gene1 = torch.load(f"{self.cfg.DATASET.ESM_ROOT}/{g1_idx}.pt", weights_only=True)
            gene2 = torch.load(f"{self.cfg.DATASET.ESM_ROOT}/{g2_idx}.pt", weights_only=True)
            if not self.cfg.MODEL.SEQ_USE_ATT:
                gene1 = torch.mean(gene1, dim=0)
                gene2 = torch.mean(gene2, dim=0)
            data['g1_seq_emb'], data['g2_seq_emb'] = gene1, gene2

        if self.cfg.MODEL.USE_FUNC:
            data['g1_func_emb'], data['g1_func_mask'] = self.text_emb(data['g1_idx'])
            data['g2_func_emb'], data['g2_func_mask'] = self.text_emb(data['g2_idx'])
        
        if self.cfg.MODEL.USE_CL:
            data['cl_emb'] = torch.load(f'{self.cfg.DATASET.CL_ROOT}/{self.cfg.DATASET.CL}.pt', weights_only=True).squeeze()
        
        if self.cfg.MODEL.USE_OMICS:
            g1_cnv = self.cnv.get(g1_idx, mean(self.cnv.values()))
            g1_ess = self.ess.get(g1_idx, mean(self.ess.values()))
            g1_exp = self.exp.get(g1_idx, mean(self.exp.values()))
            g1_mut = self.mut.get(g1_idx, mean(self.mut.values()))
            data['g1_omics'] = torch.tensor([g1_cnv, g1_ess, g1_exp, g1_mut])

            g2_cnv = self.cnv.get(g2_idx, mean(self.cnv.values()))
            g2_ess = self.ess.get(g2_idx, mean(self.ess.values()))
            g2_exp = self.exp.get(g2_idx, mean(self.exp.values()))
            g2_mut = self.mut.get(g2_idx, mean(self.mut.values()))
            data['g2_omics'] = torch.tensor([g2_cnv, g2_ess, g2_exp, g2_mut])

        return data

    @property
    def collate_fn(self):
        return CollateBatch(self.cfg)
