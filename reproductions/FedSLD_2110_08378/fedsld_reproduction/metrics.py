"""Dependency-free classification metrics."""

from typing import Dict

import numpy as np
import torch


def metrics_from_confusion(confusion: np.ndarray) -> Dict[str, object]:
    confusion = np.asarray(confusion, dtype=np.int64)
    true_count = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    tp = np.diag(confusion)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp, dtype=float), where=true_count > 0)
    precision = np.divide(tp, predicted_count, out=np.zeros_like(tp, dtype=float), where=predicted_count > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(recall),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(tp.sum() / max(1, confusion.sum())),
        "macro_f1": float(f1.mean()),
        "worst_class_recall": float(recall.min()),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, device: torch.device, num_classes: int) -> Dict[str, object]:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels in loader:
        images = images.to(device)
        targets = labels.reshape(-1).long().numpy()
        predictions = model(images).argmax(dim=1).cpu().numpy()
        np.add.at(confusion, (targets, predictions), 1)
    return metrics_from_confusion(confusion)

