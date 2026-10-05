from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def contact_metrics(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predictions = (probabilities > 0.5).float()
    true_positive = (predictions * labels).sum((0, 1))
    false_positive = (predictions * (1 - labels)).sum((0, 1))
    false_negative = ((1 - predictions) * labels).sum((0, 1))

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    labels_np = labels.flatten(0, 1).cpu().numpy()
    probabilities_np = probabilities.flatten(0, 1).cpu().numpy()
    auc = []
    for contact_index in range(labels_np.shape[1]):
        target = labels_np[:, contact_index]
        prediction = probabilities_np[:, contact_index]
        if 0 < target.mean() < 1:
            auc.append(roc_auc_score(target, prediction))
        else:
            auc.append(float("nan"))

    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "auroc": float(np.nanmean(auc)),
    }
