"""Probability-calibration and conformal metrics for final FL evaluation."""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def collect_logits(model, data_loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one evaluation pass and return CPU logits and labels."""
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for images, targets in data_loader:
            outputs = model(images.to(device))
            logits.append(outputs.detach().cpu())
            labels.append(targets.squeeze().long().cpu())
    if not logits:
        raise ValueError("Cannot evaluate an empty dataset")
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one positive temperature by minimizing calibration-set NLL."""
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def probability_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    num_bins: int = 15,
) -> Dict[str, float]:
    """Compute NLL, multiclass Brier score, and fixed-bin ECE."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    probabilities = torch.softmax(logits / temperature, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(labels).float()
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()

    ece = torch.tensor(0.0)
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    for bin_index in range(num_bins):
        lower = boundaries[bin_index]
        upper = boundaries[bin_index + 1]
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            ece += in_bin.float().mean() * torch.abs(
                correctness[in_bin].mean() - confidence[in_bin].mean()
            )

    return {
        "nll": float(F.cross_entropy(logits / temperature, labels).item()),
        "brier": float(((probabilities - one_hot) ** 2).sum(dim=1).mean().item()),
        "ece": float(ece.item()),
    }


def fit_conformal_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.1,
    temperature: float = 1.0,
) -> float:
    """Fit split-conformal score q using 1 - true-class probability."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("conformal alpha must be in (0, 1)")
    probabilities = torch.softmax(logits / temperature, dim=1)
    scores = 1.0 - probabilities[torch.arange(len(labels)), labels]
    sorted_scores = torch.sort(scores).values
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha)) - 1
    rank = min(max(rank, 0), len(scores) - 1)
    return float(sorted_scores[rank].item())


def conformal_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    temperature: float = 1.0,
) -> Dict[str, object]:
    """Evaluate marginal/classwise coverage and prediction-set size."""
    probabilities = torch.softmax(logits / temperature, dim=1)
    prediction_sets = (1.0 - probabilities) <= threshold
    covered = prediction_sets[torch.arange(len(labels)), labels]
    classwise = {}
    for class_id in range(logits.shape[1]):
        mask = labels == class_id
        if mask.any():
            classwise[str(class_id)] = float(covered[mask].float().mean().item())
    return {
        "coverage": float(covered.float().mean().item()),
        "average_set_size": float(prediction_sets.sum(dim=1).float().mean().item()),
        "empty_set_rate": float((prediction_sets.sum(dim=1) == 0).float().mean().item()),
        "worst_class_coverage": float(min(classwise.values())),
        "classwise_coverage": classwise,
    }
