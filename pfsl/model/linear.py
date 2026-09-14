import torch
import torch.nn as nn

from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY

__all__ = ["LinearProbing"]


@MODEL_REGISTRY.register()
class LinearProbing(nn.Module):

    @configurable
    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, out_channels, bias=True),
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.MODEL.IN_CHANNELS,
            "out_channels": cfg.MODEL.OUT_CHANNELS,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
