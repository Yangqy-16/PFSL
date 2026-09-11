import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, auc, average_precision_score, balanced_accuracy_score


def compute_metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    preds = logits > 0.5

    metrics = dict()
    metrics['auroc'] = roc_auc_score(labels, logits)
    metrics['auprc'] = average_precision_score(labels, logits)  # an estimate
    metrics['acc'] = accuracy_score(labels, preds)
    metrics['bacc'] = balanced_accuracy_score(labels, preds)
    metrics['precision'] = precision_score(labels, preds)
    metrics['recall'] = recall_score(labels, preds)
    metrics['f1'] = f1_score(labels, preds)

    return metrics