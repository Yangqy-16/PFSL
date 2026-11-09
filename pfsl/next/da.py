"""CDAN+E with separately balanced domains and explicit gradient routing.
Reference: https://github.com/thuml/CDAN/blob/master/pytorch/loss.py
The prediction conditioning is detached; entropy weights retain reversed grads.
Use the shared classifier logits so private cell-context heads are not aligned.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def reverse_gradient(x, coefficient=1.0):
    return _Reverse.apply(x, coefficient)


class CDANE(nn.Module):
    def __init__(self, feature_dim, hidden=64, entropy=True):
        super().__init__()
        self.use_entropy = entropy
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def domain_loss(self, features, logits, label, coefficient):
        p = logits.sigmoid()
        prob = torch.stack([1 - p, p], 1)
        conditional = (prob.detach().unsqueeze(2) * features.unsqueeze(1)).flatten(1)
        pred = self.discriminator(reverse_gradient(conditional, coefficient)).squeeze(-1)
        entropy = -(prob * prob.clamp_min(1e-6).log()).sum(1)
        weight = (
            1 + torch.exp(-reverse_gradient(entropy, coefficient))
            if self.use_entropy
            else torch.ones_like(logits)
        )
        weight = weight / weight.sum().detach().clamp_min(1e-8)
        return (
            weight
            * F.binary_cross_entropy_with_logits(
                pred, torch.full_like(pred, label), reduction="none"
            )
        ).sum()

    def forward(self, source, source_logits, target, target_logits, coefficient=1.0):
        return 0.5 * (
            self.domain_loss(source, source_logits, 1.0, coefficient)
            + self.domain_loss(target, target_logits, 0.0, coefficient)
        )
