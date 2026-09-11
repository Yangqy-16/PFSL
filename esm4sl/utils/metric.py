import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, auc, average_precision_score, balanced_accuracy_score


def mean_loss(losses: list[tuple[float, float]]) -> float:
    """
    Mean loss of an epoch.
    Each item is a tuple of (loss, batch_size).
    """
    total_loss = sum([loss * batch_size for loss, batch_size in losses])
    total_batch_size = sum([batch_size for _, batch_size in losses])
    return total_loss / total_batch_size


def compute_metrics(labels: torch.Tensor, probs: torch.Tensor) -> dict[str, float]:
    preds = probs > 0.5

    metrics = dict()
    metrics['auroc'] = roc_auc_score(labels, probs)
    metrics['auprc'] = average_precision_score(labels, probs)  # an estimate
    metrics['acc'] = accuracy_score(labels, preds)
    metrics['bacc'] = balanced_accuracy_score(labels, preds)
    metrics['precision'] = precision_score(labels, preds)
    metrics['recall'] = recall_score(labels, preds)
    metrics['f1'] = f1_score(labels, preds)

    return metrics


import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lifelines.utils import concordance_index
import torch
from torch import Tensor


def compute_mean_loss(losses: list[tuple[float, int]]) -> torch.Tensor:
    """
    Mean loss of an epoch.
    Each item is a tuple of (loss, batch_size).
    """
    total_loss = sum([loss * batch_size for loss, batch_size in losses])
    total_batch_size = sum([batch_size for _, batch_size in losses])
    return total_loss / total_batch_size


def get_cindex(y: np.ndarray, p: np.ndarray) -> float:
    sum_m = 0
    pair = 0
    for i in range(1, len(y)):
        for j in range(0, i):
            if i is not j:
                if y[i] > y[j]:
                    pair += 1
                    sum_m += 1 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    if pair != 0:
        return sum_m / pair
    else:
        return 0


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down= sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)


def get_rm2(ys_orig: list[float], ys_line: list[float]) -> float:
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def compute_reg_metrics(y_true: np.ndarray | Tensor, y_pred: np.ndarray | Tensor) -> dict[str, float]:
    metrics = dict()
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['ci'] = concordance_index(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['rm2'] = get_rm2(y_true.tolist(), y_pred.tolist())
    # metrics['rmse'] = sqrt(metrics['mse'])
    metrics['pearsonr'] = pearsonr(y_true, y_pred)[0]
    metrics['spearmanr'] = spearmanr(y_true, y_pred)[0]
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    return metrics