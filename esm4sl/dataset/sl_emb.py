# from pathlib import Path
# import os
# import random
# from typing import Any

# import numpy as np
# import torch
# from torch import Tensor
# from omegaconf import DictConfig
# from pytorch_lightning.trainer.states import RunningStage

# from coach_pl.configuration import configurable
# from coach_pl.dataset import DATASET_REGISTRY
# from .sl import SLDataset


# @DATASET_REGISTRY.register()
# class SLembDataset(SLDataset):

#     def __getitem__(self, index: int, key: str) -> tuple[torch.Tensor, torch.Tensor, float, int, int] | tuple[torch.Tensor, torch.Tensor, float, int, int, np.ndarray]:
#         """
#         key == "mean_rep": [1280]
#         key == "rep": [L, 1280], L different across samples!
#         """
#         # change df's col name if needed
#         g1_idx = self.df['0'][index]
#         g2_idx = self.df['1'][index]
#         label = float(self.df['2'][index])

#         if self.stage == RunningStage.TRAINING and random.uniform(0, 1) > 0.5:  # concat 1 and 2 in random order
#             g2_idx, g1_idx = g1_idx, g2_idx

#         gene1 = np.load(os.path.join(self.cfg.DATASET.ESM_ROOT, f"{g1_idx}.npy"), allow_pickle=True).item()[key]
#         gene1 = torch.from_numpy(gene1)
#         gene2 = np.load(os.path.join(self.cfg.DATASET.ESM_ROOT, f"{g2_idx}.npy"), allow_pickle=True).item()[key]
#         gene2 = torch.from_numpy(gene2)
        
#         # if not self.cell_line:
#         #     return gene1, gene2, label, g1_idx, g2_idx
#         # else:
#         #     return gene1, gene2, label, g1_idx, g2_idx, self.cell_embedding

#     # def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
#     #     return self.common_getitem(index, "mean_rep")

#     @property
#     def collate_fn(self) -> Any:
#         return CollateBatch(self.cfg)



    # def __call__(self, 
    #     batch: list[tuple[torch.Tensor, torch.Tensor, float, int, int]]
    # ) -> tuple[torch.Tensor, ...]:
    #     if self.cell_line:
    #         g1, g2, labels, g1_indices, g2_indices, cell_lines = zip(*batch)
    #     else:
    #         g1, g2, labels, g1_indices, g2_indices = zip(*batch)

    #     batched_g1, g1_mask = self.padding(g1)
    #     batched_g2, g2_mask = self.padding(g2)

    #     if self.cell_line:
    #         return torch.tensor(g1_indices), batched_g1, g1_mask, \
    #                torch.tensor(g2_indices), batched_g2, g2_mask, \
    #                torch.tensor(labels), torch.from_numpy(np.array(cell_lines)).float()
    #     else:
    #         return torch.tensor(g1_indices), batched_g1, g1_mask, \
    #                torch.tensor(g2_indices), batched_g2, g2_mask, \
    #                torch.tensor(labels)


# @DATASET_REGISTRY.register()
# class SLwholeembDataset(SLembDataset):
#     def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
#         return self.common_getitem(index, "rep")


# @DATASET_REGISTRY.register()
# class ProSLDataset(SLwholeembDataset):
#     @configurable
#     def __init__(self, cfg: DictConfig, stage: RunningStage) -> None:  #sl_root: Path, test_fold: int, cell_line: str | None, esm_root: Path
#         super().__init__(cfg, stage)
#         self.text_emb_root = cfg.DATASET.TEXT_EMB_ROOT

#     def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
#         g1_esm, g2_esm, label, g1_idx, g2_idx = super().__getitem__(index)

#         if os.path.exists(os.path.join(self.text_emb_root, f"{g1_idx}.pt")):
#             g1_func = torch.load(os.path.join(self.text_emb_root, f"{g1_idx}.pt"), map_location='cpu', weights_only=True)
#         else:
#             g1_func = torch.randn((1, 1024))  # Random initialization
#         if os.path.exists(os.path.join(self.text_emb_root, f"{g2_idx}.pt")):
#             g2_func = torch.load(os.path.join(self.text_emb_root, f"{g2_idx}.pt"), map_location='cpu', weights_only=True)
#         else:
#             g2_func = torch.randn((1, 1024))

#         return g1_esm, g1_func, g2_esm, g2_func, label, g1_idx, g2_idx
    
#     @property
#     def collate_fn(self):
#         return ProSLCollate(self.cell_line, self.max_len)


# def funccollate(
#     batch: list[tuple[Tensor, Tensor, float, int, int]]
# ) -> tuple[Tensor, ...]:
#     g1_func, g2_func, labels, g1_indices, g2_indices = zip(*batch)

#     g1_func = torch.stack([func.mean(dim=0) for func in g1_func])
#     g2_func = torch.stack([func.mean(dim=0) for func in g2_func])

#     return torch.tensor(g1_indices), g1_func, \
#             torch.tensor(g2_indices), g2_func, \
#             torch.tensor(labels)


# @DATASET_REGISTRY.register()
# class PFDataset(ProSLDataset):
#     @configurable
#     def __init__(self, cfg: DictConfig, stage: RunningStage) -> None:  #sl_root: Path, test_fold: int, cell_line: str | None, esm_root: Path
#         super().__init__(cfg, stage)

#     def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
#         g1_idx = self.df['0'][index]
#         g2_idx = self.df['1'][index]
#         label = float(self.df['2'][index])

#         if os.path.exists(os.path.join(self.text_emb_root, f"{g1_idx}.pt")):
#             g1_func = torch.load(os.path.join(self.text_emb_root, f"{g1_idx}.pt"), map_location='cpu', weights_only=True)
#         else:
#             g1_func = torch.randn((1, 1024))  # Random initialization
#         if os.path.exists(os.path.join(self.text_emb_root, f"{g2_idx}.pt")):
#             g2_func = torch.load(os.path.join(self.text_emb_root, f"{g2_idx}.pt"), map_location='cpu', weights_only=True)
#         else:
#             g2_func = torch.randn((1, 1024))

#         return g1_func, g2_func, label, g1_idx, g2_idx
    
    # @property
    # def collate_fn(self):
    #     return funccollate