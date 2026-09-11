"""Untied TopK sparse autoencoder for the ProTrek SAE pilot."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SaeForwardOutput:
    preactivations: torch.Tensor
    activations: torch.Tensor
    reconstruction: torch.Tensor
    auxiliary_activations: torch.Tensor | None
    auxiliary_reconstruction: torch.Tensor | None


@dataclass(frozen=True)
class SaeLossOutput:
    total: torch.Tensor
    main: torch.Tensor
    auxiliary: torch.Tensor


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


class TopKSAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int,
        k: int,
        aux_k: int,
        activation_threshold: float = 1e-3,
        dead_steps_threshold: int = 1,
    ) -> None:
        super().__init__()
        _require_positive_integer("input_dim", input_dim)
        _require_positive_integer("width", width)
        _require_positive_integer("k", k)
        _require_positive_integer("aux_k", aux_k)
        if k > width:
            raise ValueError("k must be no greater than width")
        if aux_k > width:
            raise ValueError("aux_k must be no greater than width")
        if isinstance(activation_threshold, bool) or not isinstance(
            activation_threshold, (int, float)
        ):
            raise TypeError("activation_threshold must be a real number")
        if not math.isfinite(activation_threshold) or activation_threshold < 0:
            raise ValueError("activation_threshold must be finite and nonnegative")
        if isinstance(dead_steps_threshold, bool) or not isinstance(
            dead_steps_threshold, int
        ):
            raise TypeError("dead_steps_threshold must be an integer")
        if dead_steps_threshold < 0:
            raise ValueError("dead_steps_threshold must be nonnegative")

        self.input_dim = input_dim
        self.width = width
        self.k = k
        self.aux_k = aux_k
        self.activation_threshold = float(activation_threshold)
        self.dead_steps_threshold = dead_steps_threshold

        self.encoder_weight = nn.Parameter(torch.empty(width, input_dim))
        self.decoder_weight = nn.Parameter(torch.empty(input_dim, width))
        self.encoder_bias = nn.Parameter(torch.zeros(width))
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("steps_since_active", torch.zeros(width, dtype=torch.long))

        with torch.no_grad():
            nn.init.normal_(self.decoder_weight)
            normalize_decoder_(self)
            self.encoder_weight.copy_(self.decoder_weight.T)

    @torch.no_grad()
    def initialize_pre_bias_(self, streaming_mean: torch.Tensor) -> TopKSAE:
        if not isinstance(streaming_mean, torch.Tensor):
            raise TypeError("streaming_mean must be a tensor")
        if streaming_mean.dtype != torch.float32:
            raise TypeError("streaming_mean must have dtype float32")
        if streaming_mean.shape != (self.input_dim,):
            raise ValueError(f"streaming_mean must have shape ({self.input_dim},)")
        if not torch.isfinite(streaming_mean).all().item():
            raise ValueError("streaming_mean must contain only finite values")
        self.pre_bias.copy_(streaming_mean)
        return self

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a tensor")
        if inputs.dtype != torch.float32:
            raise TypeError("inputs must have dtype float32")
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(f"inputs must have shape [B, {self.input_dim}]")
        if not torch.isfinite(inputs).all().item():
            raise ValueError("inputs must contain only finite values")
        norms = inputs.norm(dim=-1)
        if not torch.allclose(
            norms,
            torch.ones_like(norms),
            rtol=1e-4,
            atol=1e-4,
        ):
            raise ValueError("inputs must have unit norm along the final dimension")

    def forward(
        self,
        inputs: torch.Tensor,
        update_dead_stats: bool = True,
    ) -> SaeForwardOutput:
        self._validate_inputs(inputs)
        preactivations = F.linear(
            inputs - self.pre_bias,
            self.encoder_weight,
            self.encoder_bias,
        )
        selected_values, selected_indices = torch.topk(
            preactivations,
            self.k,
            dim=-1,
        )
        activations = torch.zeros_like(preactivations)
        activations.scatter_(-1, selected_indices, F.relu(selected_values))

        if update_dead_stats:
            with torch.no_grad():
                active = (activations > self.activation_threshold).any(dim=0)
                self.steps_since_active.add_(1)
                self.steps_since_active.masked_fill_(active, 0)

        dead_mask = self.steps_since_active >= self.dead_steps_threshold
        dead_count = int(dead_mask.sum().item())
        auxiliary_activations: torch.Tensor | None = None
        auxiliary_reconstruction: torch.Tensor | None = None
        if dead_count:
            selected_dead_count = min(self.aux_k, dead_count)
            masked_preactivations = preactivations.masked_fill(
                ~dead_mask.unsqueeze(0),
                -torch.inf,
            )
            auxiliary_values, auxiliary_indices = torch.topk(
                masked_preactivations,
                selected_dead_count,
                dim=-1,
            )
            auxiliary_activations = torch.zeros_like(preactivations)
            auxiliary_activations.scatter_(
                -1,
                auxiliary_indices,
                F.relu(auxiliary_values),
            )
            auxiliary_reconstruction = F.linear(
                auxiliary_activations,
                self.decoder_weight,
            )

        reconstruction = F.linear(
            activations,
            self.decoder_weight,
            self.pre_bias,
        )
        return SaeForwardOutput(
            preactivations=preactivations,
            activations=activations,
            reconstruction=reconstruction,
            auxiliary_activations=auxiliary_activations,
            auxiliary_reconstruction=auxiliary_reconstruction,
        )


def normalized_squared_error(
    inputs: torch.Tensor,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    numerator = (inputs - reconstruction).pow(2).sum(dim=-1)
    denominator = inputs.pow(2).sum(dim=-1).clamp_min(1e-8)
    return numerator / denominator


def compute_sae_loss_from_tensors(
    *,
    inputs: torch.Tensor,
    reconstruction: torch.Tensor,
    auxiliary_reconstruction: torch.Tensor | None,
    weights: torch.Tensor,
    auxiliary_coefficient: float,
) -> SaeLossOutput:
    main_errors = normalized_squared_error(inputs, reconstruction)
    main = (weights * main_errors).mean()

    if auxiliary_reconstruction is None:
        auxiliary = reconstruction.sum() * 0.0
    else:
        residual = inputs - reconstruction.detach()
        auxiliary_numerator = (residual - auxiliary_reconstruction).pow(2).sum(dim=-1)
        auxiliary_denominator = (
            residual.pow(2).sum(dim=-1).detach().clamp_min(1e-8)
        )
        auxiliary_errors = auxiliary_numerator / auxiliary_denominator
        auxiliary = (weights * auxiliary_errors).mean()

    total = main + auxiliary_coefficient * auxiliary
    return SaeLossOutput(total=total, main=main, auxiliary=auxiliary)


def compute_sae_loss(
    output: SaeForwardOutput,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    auxiliary_coefficient: float,
) -> SaeLossOutput:
    return compute_sae_loss_from_tensors(
        inputs=inputs,
        reconstruction=output.reconstruction,
        auxiliary_reconstruction=output.auxiliary_reconstruction,
        weights=weights,
        auxiliary_coefficient=auxiliary_coefficient,
    )


@torch.no_grad()
def project_decoder_gradients_(model: TopKSAE) -> None:
    gradient = model.decoder_weight.grad
    if gradient is None:
        raise RuntimeError("Decoder gradient is missing")
    if gradient.shape != model.decoder_weight.shape:
        raise RuntimeError("Decoder gradient has the wrong shape")
    if not torch.isfinite(model.decoder_weight).all().item():
        raise RuntimeError("Decoder contains non-finite values")
    decoder_norms = model.decoder_weight.norm(dim=0)
    if not torch.allclose(
        decoder_norms,
        torch.ones_like(decoder_norms),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError("Decoder columns must be unit norm before projection")
    if not torch.isfinite(gradient).all().item():
        raise RuntimeError("Decoder gradient contains non-finite values")
    parallel = (model.decoder_weight * gradient).sum(dim=0, keepdim=True)
    gradient.sub_(model.decoder_weight * parallel)


@torch.no_grad()
def normalize_decoder_(model: TopKSAE) -> None:
    if not torch.isfinite(model.decoder_weight).all().item():
        raise RuntimeError("Decoder contains non-finite values")
    raw_norms = model.decoder_weight.norm(dim=0, keepdim=True)
    if torch.any(raw_norms <= 1e-12).item():
        raise RuntimeError("Decoder columns must have norm greater than 1e-12")
    norms = raw_norms.clamp_min(1e-12)
    model.decoder_weight.div_(norms)


def assert_finite_parameters_and_gradients(model: TopKSAE) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise RuntimeError(f"Parameter {name} contains non-finite values")
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"Gradient for {name} contains non-finite values")
