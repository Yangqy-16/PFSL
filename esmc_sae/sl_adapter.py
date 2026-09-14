"""Frozen ProSL checkpoint adapter with differentiable mean-pooled inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from coach_pl.module import build_module
from esmc_sae.config import ProjectConfig, load_config
from esmc_sae.manifest import (
    file_sha256 as _sha256,
    verify_manifest_target as _verify_manifest_target,
)
from esmc_sae.train_dense import build_coach_config
import pfsl.model  # noqa: F401  # register the existing ProSL model
import pfsl.module  # noqa: F401  # register the existing Lightning module


FORMAL_CHECKPOINT_NAME = "epoch=4-loss=0.3476.ckpt"


def _verify_formal_checkpoint(
    project: ProjectConfig,
    checkpoint: Path,
) -> Path:
    expected = (
        project.training.output_dir
        / "best_ckpts"
        / FORMAL_CHECKPOINT_NAME
    )
    if Path(checkpoint).resolve(strict=False) != expected.resolve(strict=False):
        raise ValueError("checkpoint is not the formal Task 4 checkpoint path")
    _verify_manifest_target(
        project.paths.project_root / "manifests/task04.sha256",
        expected,
        code_root=project.paths.code_root,
        label="formal Task 4 checkpoint",
    )
    return expected


class FrozenSLAdapter:
    """Expose one frozen downstream pair logit as a function of residue states."""

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        device: torch.device,
    ) -> None:
        self.module = module
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        config: ProjectConfig | Path,
        checkpoint: Path,
    ) -> "FrozenSLAdapter":
        """Build the existing Lightning module and strictly load one checkpoint."""

        project = load_config(config) if isinstance(config, Path) else config
        checkpoint = _verify_formal_checkpoint(project, Path(checkpoint))
        coach_config = build_coach_config(project, [])
        module = build_module(coach_config)
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint is missing a Lightning state_dict")
        module.load_state_dict(state_dict, strict=True)
        module.eval()
        module.requires_grad_(False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module.to(device)
        device = next(module.parameters()).device
        if module.training:
            raise RuntimeError("downstream module must be in evaluation mode")
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("downstream module parameters must be frozen")
        return cls(module, device=device)

    def pair_logit(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        fixed_batch: dict[str, object],
    ) -> torch.Tensor:
        """Replace only pair sequence means and return the scalar frozen logit."""

        if h_a.ndim != 2 or h_b.ndim != 2:
            raise ValueError("hidden states must have shape [residue, d_model]")
        if h_a.shape[1] != h_b.shape[1]:
            raise ValueError("pair hidden-state widths must match")
        if h_a.device != self.device or h_b.device != self.device:
            raise ValueError("hidden states must be on the downstream model device")

        logits = self.pair_logits_from_means(
            h_a.float().mean(dim=0, keepdim=True),
            h_b.float().mean(dim=0, keepdim=True),
            fixed_batch,
        )
        if logits.numel() != 1:
            raise RuntimeError("downstream model must return exactly one pair logit")
        return logits.reshape(())

    def pair_logits_from_means(
        self,
        mean_a: torch.Tensor,
        mean_b: torch.Tensor,
        fixed_batch: Mapping[str, object],
    ) -> torch.Tensor:
        """Evaluate several sequence-mean variants against one fixed pair."""

        if mean_a.ndim != 2 or mean_b.ndim != 2:
            raise ValueError("mean embeddings must have shape [batch, d_model]")
        if mean_a.shape != mean_b.shape or mean_a.shape[0] < 1:
            raise ValueError("pair mean-embedding batches must share non-empty shape")
        if mean_a.device != self.device or mean_b.device != self.device:
            raise ValueError("mean embeddings must be on the downstream model device")

        batch_size = mean_a.shape[0]
        batch: dict[str, object] = {}
        for key, value in fixed_batch.items():
            if key in ("g1_seq_emb", "g2_seq_emb"):
                continue
            if not isinstance(value, torch.Tensor):
                batch[key] = value
                continue
            transferred = value.to(self.device)
            if transferred.ndim > 0 and transferred.shape[0] == 1:
                transferred = transferred.expand(
                    batch_size,
                    *transferred.shape[1:],
                )
            elif (
                transferred.ndim > 0
                and transferred.shape[0] != batch_size
            ):
                raise ValueError(
                    f"fixed batch field {key!r} cannot expand to variant batch"
                )
            batch[key] = transferred
        batch["g1_seq_emb"] = mean_a.float()
        batch["g2_seq_emb"] = mean_b.float()

        output = self.module.model(batch)
        logit = output[0] if isinstance(output, tuple) else output
        if not isinstance(logit, torch.Tensor) or logit.numel() != batch_size:
            raise RuntimeError(
                "downstream model must return one logit per pair variant"
            )
        return logit.reshape(batch_size)
