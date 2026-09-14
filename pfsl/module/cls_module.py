from collections import defaultdict
from typing import Any
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
# from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from omegaconf import DictConfig

from coach_pl.configuration import configurable
from coach_pl.model import build_model  #, build_criterion
from coach_pl.module import MODULE_REGISTRY
from pfsl.utils.metric import mean_loss, compute_metrics #, compute_reg_metrics


@MODULE_REGISTRY.register()
class ClsModule(pl.LightningModule):
    @configurable
    def __init__(self, model: nn.Module, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.model = torch.compile(model) if cfg.MODULE.COMPILE else model

        self.batch_size = cfg.DATALOADER.TRAIN.BATCH_SIZE
        self.base_lr = cfg.MODULE.OPTIMIZER.BASE_LR
        self.monitor = cfg.TRAINER.CHECKPOINT.MONITOR

        self.save_hyperparameters(cfg) #, ignore=['model', 'criterion']

        self.loss_func = nn.BCEWithLogitsLoss()

        self.losses = defaultdict(list)
        self.logits = defaultdict(list)
        self.labels = defaultdict(list)
        self.g1_indices = defaultdict(list)
        self.g2_indices = defaultdict(list)

        # Flush allocator cache periodically when using DeepSpeed to reduce
        # unsynchronized cache flushes under high memory pressure.
        # self.empty_cache_interval = max(1, int(cfg.TRAINER.LOG_EVERY_N_STEPS))

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        model = build_model(cfg)
        return {
            "model": model,
            "cfg": cfg,
        }

    def configure_gradient_clipping(self, optimizer: Optimizer, gradient_clip_val: int | float | None = None, gradient_clip_algorithm: str | None = None) -> None:
        if gradient_clip_algorithm == "value":
            nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), gradient_clip_val)
        elif gradient_clip_algorithm == "norm":
            nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_val)

    def configure_optimizers(self) -> dict[str, Optimizer | dict[str, str | Optimizer]]:
        lr = self.base_lr #* self.batch_size * self.trainer.accumulate_grad_batches * self.trainer.world_size / 16

        # if 'deepspeed' in self.cfg.TRAINER.STRATEGY: #_stage_2
        #     optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        # elif 'deepspeed_stage_3' in self.cfg.TRAINER.STRATEGY:
        #     optimizer = FusedAdam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        # else:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, self.cfg.TRAINER.CHECKPOINT.MONITOR_MODE)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/mean_loss",
            } 
        }

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        gene1, gene2, label, g1_idx, g2_idx = batch[0].cuda(), batch[1].cuda(), \
                batch[2].cuda(), batch[3].cuda(), batch[4].cuda()

        logit = self.model(torch.cat([gene1, gene2], dim=1))

        loss = self.loss_func(logit, label)
        return loss, logit, label, g1_idx, g2_idx

    def step_common(self, batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: RunningStage, stage_str: str) -> dict[str, Any]:
        loss, logit, label, g1_idx, g2_idx = self.forward(batch)

        if stage != RunningStage.TESTING:
            self.log(f"{stage_str}/loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=len(label), sync_dist=True, rank_zero_only=True)

        self.losses[stage].append((loss.detach().item(), len(label)))
        self.logits[stage].append(logit.detach().cpu())
        self.labels[stage].append(label.detach().cpu())
        self.g1_indices[stage].append(g1_idx.detach().cpu())
        self.g2_indices[stage].append(g2_idx.detach().cpu())
        return {"loss": loss}

    def end_common(self, stage: RunningStage, stage_str: str) -> None:
        ml = mean_loss(self.losses[stage])
        labels = torch.cat(self.labels[stage])  # [N]
        logits = torch.cat(self.logits[stage])  # [N]
        
        probs = torch.sigmoid(logits)  # NOTE: only for cls !!!
        metrics = compute_metrics(labels, probs) #logits

        self.log(f"{stage_str}/mean_loss", ml, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log(f"{stage_str}/auroc", metrics['auroc'], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log(f"{stage_str}/auprc", metrics['auprc'], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        # self.log(f"{stage_str}/mse", metrics['mse'], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        # self.log(f"{stage_str}/ci", metrics['ci'], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        if stage == RunningStage.VALIDATING:
            if self.monitor == 'avgmtr':
                self.log(self.monitor, (metrics['auroc'] + metrics['auprc']) / 2, logger=False, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
            elif self.monitor != 'loss':
                self.log(self.monitor, metrics[self.monitor.lower()], logger=False, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
            else:
                self.log(self.monitor, ml, logger=False, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        if stage == RunningStage.TESTING:
            for metric in ['acc', 'precision', 'recall', 'f1', 'bacc']:
            # for metric in ['r2', 'rm2', 'pearsonr', 'spearmanr', 'mae']:
                self.log(f"{stage_str}/{metric}", metrics[metric], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)

            g1_indices = torch.cat(self.g1_indices[stage])
            g2_indices = torch.cat(self.g2_indices[stage])
            df = pd.DataFrame({'0': g1_indices.detach().cpu().numpy(), '1': g2_indices.detach().cpu().numpy(), 'prob': probs.detach().cpu().numpy()}) #logits
            output_dir = f'{self.cfg.OUTPUT_DIR}/csv_log'
            version_num = len(os.listdir(output_dir)) - 1
            df.to_csv(f'{output_dir}/version_{version_num}/test_logits.csv', index=False)

        self.losses[stage].clear()
        self.logits[stage].clear()
        self.labels[stage].clear()
        self.g1_indices[stage].clear()
        self.g2_indices[stage].clear()

    def training_step(self, batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        return self.step_common(batch, RunningStage.TRAINING, 'train')

    # def _empty_cache_if_needed(self, batch_idx: int | None = None) -> None:
    #     if 'deepspeed' not in self.cfg.TRAINER.STRATEGY or not torch.cuda.is_available():
    #         return
    #     if batch_idx is not None and (batch_idx + 1) % self.empty_cache_interval != 0:
    #         return
    #     try:
    #         from deepspeed.accelerator import get_accelerator
    #         get_accelerator().empty_cache()
    #     except Exception:
    #         torch.cuda.empty_cache()

    # def on_train_batch_end(self, outputs: dict[str, Any], batch: Any, batch_idx: int) -> None:
    #     # All ranks hit this hook at the same batch index.
    #     self._empty_cache_if_needed(batch_idx=batch_idx)

    def on_train_epoch_end(self) -> None:
        self.end_common(RunningStage.TRAINING, 'train')
        # self._empty_cache_if_needed()

    def validation_step(self, batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        return self.step_common(batch, RunningStage.VALIDATING, 'val')

    def on_validation_epoch_end(self) -> None:
        self.end_common(RunningStage.VALIDATING, 'val')
        # self._empty_cache_if_needed()

    def test_step(self, batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, Any]:        
        return self.step_common(batch, RunningStage.TESTING, 'test')

    def on_test_epoch_end(self) -> None:
        self.end_common(RunningStage.TESTING, 'test')
        # self._empty_cache_if_needed()


@MODULE_REGISTRY.register()
class AttnModule(ClsModule):
    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        g1_idx, g1, g1_mask, g2_idx, g2, g2_mask, label = batch[0].cuda(), batch[1].cuda(), \
            batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda()
        cell_embedding = batch[7].cuda() if self.cfg.DATASET.CELL_LINE else None

        logit, attns = self.model(g1, g1_mask, g2, g2_mask, cell_embedding)

        loss = self.loss_func(logit, label)
        return loss, logit, label, g1_idx, g2_idx
    

@MODULE_REGISTRY.register()
class LoRAModule(ClsModule):
    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        label = batch['label'].cuda()
        g1_idx = torch.tensor([i[0] for i in batch['gene1']])
        g2_idx = torch.tensor([i[0] for i in batch['gene2']])

        logit = self.model(batch['gene1'], batch['gene2'])

        loss = self.loss_func(logit, label)
        return loss, logit, label, g1_idx, g2_idx
