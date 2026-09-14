from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Hashable

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from coach_pl.configuration import configurable
from coach_pl.module import MODULE_REGISTRY

from .prosl import PFnewModule


class UQ(Enum):
    CONFORMAL = 1
    ENSEMBLE = 2
    MONDRIAN_CONFORMAL = 3


@MODULE_REGISTRY.register()
class UncertaintyModule(PFnewModule):
    @configurable
    def __init__(self, model: nn.Module, cfg: DictConfig) -> None:
        super().__init__(model, cfg)
        self.uq_results: dict[str, Any] = {}

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _forward_logits(self, batch: dict[str, Any]) -> tuple[torch.Tensor, Any]:
        output = self.model(batch)
        if isinstance(output, tuple):
            return output[0], output[1] if len(output) > 1 else None
        return output, None

    def _binary_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim > 1 and logits.size(-1) == 2:
            return torch.softmax(logits, dim=-1)

        pos_prob = torch.sigmoid(logits.reshape(-1))
        return torch.stack([1.0 - pos_prob, pos_prob], dim=-1)

    def _sample_key(
        self,
        sample: dict[str, Any],
        extract_name: Callable[[dict[str, Any]], Hashable] | None = None,
    ) -> Hashable:
        if extract_name is not None:
            return extract_name(sample)

        g1_idx = sample.get("g1_idx")
        g2_idx = sample.get("g2_idx")
        if isinstance(g1_idx, torch.Tensor):
            g1_idx = int(g1_idx.reshape(-1)[0].item())
        if isinstance(g2_idx, torch.Tensor):
            g2_idx = int(g2_idx.reshape(-1)[0].item())
        return (g1_idx, g2_idx)

    def _iter_samples(self, batch: dict[str, Any], logit: torch.Tensor, label: torch.Tensor) -> list[dict[str, Any]]:
        batch_size = int(label.shape[0])
        samples: list[dict[str, Any]] = []
        for i in range(batch_size):
            sample: dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
                    sample[key] = value[i]
                else:
                    sample[key] = value
            sample["logit"] = logit[i]
            sample["label"] = label[i]
            samples.append(sample)
        return samples

    def _score_against_calibration(self, calibration_scores: list[float], confidence: float) -> float:
        if not calibration_scores:
            return 1.0
        scores = np.asarray(calibration_scores, dtype=float)
        return float((np.sum(scores >= confidence) + 1) / (scores.size + 1))

    def calibrate(
        self,
        calibration_loader: Any,
        mondrian_class_dict: dict[Hashable, Hashable] | None = None,
        extract_name: Callable[[dict[str, Any]], Hashable] | None = None,
    ) -> dict[Hashable, list[float]]:
        self.model.eval()
        calibration_scores: dict[Hashable, list[float]] = defaultdict(list)

        with torch.no_grad():
            for batch in calibration_loader:
                batch = self._move_batch_to_device(batch)
                logits, _ = self._forward_logits(batch)
                probs = self._binary_probs(logits)
                labels = batch["label"].reshape(-1).long()

                for sample, sample_probs, sample_label in zip(
                    self._iter_samples(batch, logits, labels),
                    probs,
                    labels,
                ):
                    sample_key = self._sample_key(sample, extract_name)
                    mondrian_class = -1
                    if mondrian_class_dict is not None:
                        mondrian_class = mondrian_class_dict.get(sample_key, -1)
                    true_prob = float(sample_probs[int(sample_label.item())].item())
                    calibration_scores[mondrian_class].append(1.0 - true_prob)

        return dict(calibration_scores)

    def conformal_pred(
        self,
        calibration_loader: Any,
        test_loader: Any,
        mondrian_class_dict: dict[Hashable, Hashable] | None = None,
        extract_name: Callable[[dict[str, Any]], Hashable] | None = None,
    ) -> dict[str, Any]:
        calibration_scores = self.calibrate(
            calibration_loader,
            mondrian_class_dict=mondrian_class_dict,
            extract_name=extract_name,
        )

        pred_vals: list[int] = []
        corr: list[int] = []
        conf_scores: list[tuple[float, float]] = []
        test_names: list[Hashable] = []
        probs_list: list[torch.Tensor] = []

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = self._move_batch_to_device(batch)
                logits, _ = self._forward_logits(batch)
                probs = self._binary_probs(logits)
                labels = batch["label"].reshape(-1).long()
                preds = torch.argmax(probs, dim=-1)

                for sample, sample_probs, pred, label in zip(
                    self._iter_samples(batch, logits, labels),
                    probs,
                    preds,
                    labels,
                ):
                    sample_key = self._sample_key(sample, extract_name)
                    mondrian_class = -1
                    if mondrian_class_dict is not None:
                        mondrian_class = mondrian_class_dict.get(sample_key, -1)

                    calib_scores = calibration_scores.get(mondrian_class, [])
                    best_conf = float(1.0 - sample_probs[int(pred.item())].item())
                    worst_conf = float(1.0 - sample_probs[1 - int(pred.item())].item())

                    pred_vals.append(int(pred.item()))
                    corr.append(int(label.item()))
                    conf_scores.append(
                        (
                            self._score_against_calibration(calib_scores, best_conf),
                            self._score_against_calibration(calib_scores, worst_conf),
                        )
                    )
                    test_names.append(sample_key)
                    probs_list.append(sample_probs.detach().cpu())

        return {
            "pred_vals": pred_vals,
            "corr": corr,
            "conf_scores": conf_scores,
            "test_names": test_names,
            "calibration_scores": calibration_scores,
            "probs_list": probs_list,
        }

    def ensemble_pred(
        self,
        test_loader: Any,
        ensemble_state_dicts: list[dict[str, torch.Tensor]] | None = None,
        ensemble_ckpt_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        if ensemble_state_dicts is None and ensemble_ckpt_paths is None:
            raise ValueError("Provide ensemble_state_dicts or ensemble_ckpt_paths.")

        states: list[dict[str, torch.Tensor]] = []
        if ensemble_state_dicts is not None:
            states = [deepcopy(state) for state in ensemble_state_dicts]
        else:
            for path in ensemble_ckpt_paths or []:
                states.append(torch.load(path, map_location=self.device))

        original_state = deepcopy(self.model.state_dict())
        fold_preds: list[list[int]] = []
        fold_corr: list[int] = []
        fold_names: list[Hashable] = []

        try:
            for state in states:
                self.model.load_state_dict(state)
                self.model.eval()

                member_preds: list[int] = []
                member_corr: list[int] = []
                member_names: list[Hashable] = []

                with torch.no_grad():
                    for batch in test_loader:
                        batch = self._move_batch_to_device(batch)
                        logits, _ = self._forward_logits(batch)
                        probs = self._binary_probs(logits)
                        labels = batch["label"].reshape(-1).long()
                        preds = torch.argmax(probs, dim=-1)

                        for sample, pred, label in zip(self._iter_samples(batch, logits, labels), preds, labels):
                            member_preds.append(int(pred.item()))
                            member_corr.append(int(label.item()))
                            member_names.append(self._sample_key(sample))

                fold_preds.append(member_preds)
                fold_corr = member_corr
                fold_names = member_names
        finally:
            self.model.load_state_dict(original_state)

        pred_array = np.asarray(fold_preds, dtype=float)
        variance = np.var(pred_array, axis=0).tolist() if pred_array.size else []

        return {
            "pred_vals": fold_preds,
            "variance": variance,
            "corr": fold_corr,
            "test_names": fold_names,
        }

    def uncertainty_quantification(
        self,
        uq_type: UQ | str,
        test_loader: Any,
        calibration_loader: Any | None = None,
        num_ensemble: int = 5,
        ensemble_state_dicts: list[dict[str, torch.Tensor]] | None = None,
        ensemble_ckpt_paths: list[str] | None = None,
        mondrian_class_dict: dict[Hashable, Hashable] | None = None,
        extract_name: Callable[[dict[str, Any]], Hashable] | None = None,
    ) -> dict[str, Any]:
        if isinstance(uq_type, str):
            uq_type = UQ[uq_type.upper()]

        if uq_type == UQ.CONFORMAL:
            if calibration_loader is None:
                raise ValueError("calibration_loader is required for conformal prediction.")
            result = self.conformal_pred(
                calibration_loader,
                test_loader,
                mondrian_class_dict=None,
                extract_name=extract_name,
            )
        elif uq_type == UQ.MONDRIAN_CONFORMAL:
            if calibration_loader is None:
                raise ValueError("calibration_loader is required for Mondrian conformal prediction.")
            result = self.conformal_pred(
                calibration_loader,
                test_loader,
                mondrian_class_dict=mondrian_class_dict or {},
                extract_name=extract_name,
            )
        elif uq_type == UQ.ENSEMBLE:
            if ensemble_state_dicts is not None:
                num_ensemble = len(ensemble_state_dicts)
            elif ensemble_ckpt_paths is not None:
                num_ensemble = len(ensemble_ckpt_paths)
            result = self.ensemble_pred(
                test_loader,
                ensemble_state_dicts=ensemble_state_dicts,
                ensemble_ckpt_paths=ensemble_ckpt_paths,
            )
            result["num_ensemble"] = num_ensemble
        else:
            raise ValueError("Unknown uncertainty quantification type.")

        self.uq_results[uq_type.name.lower()] = result
        return result
