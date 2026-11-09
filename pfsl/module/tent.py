from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.configuration import configurable
from coach_pl.module import MODULE_REGISTRY

from .prosl import PFnewModule


@MODULE_REGISTRY.register()
class TENTWrapper(PFnewModule):
    @configurable
    def __init__(self, model: nn.Module, cfg: DictConfig) -> None:
        super().__init__(model, cfg)
        self.automatic_optimization = False
        self._tent_optimizer: torch.optim.Optimizer | None = None

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return super().from_config(cfg)

    def _is_tent_mlp(self, module: nn.Module) -> bool:
        return (
            hasattr(module, "fc1")
            and hasattr(module, "fc2")
            and hasattr(module, "norm")
            and isinstance(getattr(module, "norm"), nn.LayerNorm)
        )

    def _tent_norm_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in self.model.modules():
            if self._is_tent_mlp(module):
                params.extend(list(module.norm.parameters()))
        return params

    def _freeze_for_tent(self) -> None:
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if self._is_tent_mlp(module):
                module.norm.requires_grad_(True)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self._tent_norm_params()
        if len(params) == 0:
            raise RuntimeError("TENTWrapper found no trainable LayerNorm parameters.")
        optimizer = torch.optim.Adam(params, lr=self.base_lr, weight_decay=0.0)
        self._tent_optimizer = optimizer
        return optimizer

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self._build_optimizer()

    def on_test_start(self) -> None:
        self._freeze_for_tent()
        self.model.eval()
        if self._tent_optimizer is None:
            self._build_optimizer()

    def _entropy_loss(self, logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # if logits.ndim == 1 or logits.size(-1) == 1:
        assert logits.ndim == 1
        probs = torch.sigmoid(logits)
        probs = probs.clamp(min=eps, max=1.0 - eps)
        entropy = -(probs * torch.log(probs + eps) + (1.0 - probs) * torch.log(1.0 - probs + eps))
        return entropy.mean()

        # probs = torch.softmax(logits, dim=-1).clamp_min(eps)
        # entropy = -(probs * torch.log(probs))
        # return entropy.sum(dim=-1).mean()

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch

    def _predict_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        batch = self._move_batch_to_device(batch)
        logit, atts = self.model(batch)
        return logit, batch["label"], batch["g1_idx"], batch["g2_idx"], atts

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        with torch.enable_grad():
            logit, label, g1_idx, g2_idx, _ = self._predict_batch(batch)
            loss = self._entropy_loss(logit)

            opt = self._tent_optimizer or self._build_optimizer()
            opt.zero_grad()
            loss.backward()
            opt.step()

        stage = RunningStage.TESTING
        self.losses[stage].append((loss.detach().item(), len(label)))
        self.logits[stage].append(logit.detach().cpu())
        self.labels[stage].append(label.detach().cpu())
        self.g1_indices[stage].append(g1_idx.detach().cpu())
        self.g2_indices[stage].append(g2_idx.detach().cpu())
        return {"loss": loss}
