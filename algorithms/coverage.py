"""Local objective for class-coverage-aware federated learning.

The server still aggregates model parameters with ordinary sample-weighted
FedAvg.  Every value in this module is derived and used on the client only.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from datasets.labels import dataset_labels


def count_client_classes(dataset, num_classes: int) -> torch.Tensor:
    """Count labels in a client's training subset without returning raw data."""
    labels = dataset_labels(dataset)
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError("training label outside configured class range")
    return torch.bincount(torch.as_tensor(labels), minlength=num_classes).float()


def logit_adjustment(
    class_counts: torch.Tensor,
    tau: float = 1.0,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Return the smoothed local log-prior offset used during training."""
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if smoothing <= 0:
        raise ValueError("smoothing must be greater than zero")
    return tau * torch.log(class_counts.float() + smoothing)


def snapshot_classifier_head(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Copy the global classifier head at the start of a local FL round."""
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise ValueError("Coverage-aware training requires model.fc to be nn.Linear")
    return {
        "weight": model.fc.weight.detach().clone(),
        "bias": model.fc.bias.detach().clone(),
    }


def coverage_weights(class_counts: torch.Tensor, kappa: float) -> torch.Tensor:
    """Give poorly represented local classes a stronger global-head anchor."""
    if kappa <= 0:
        raise ValueError("coverage_kappa must be greater than zero")
    return kappa / (class_counts.float() + kappa)


def coverage_head_penalty(
    model: nn.Module,
    global_head: Dict[str, torch.Tensor],
    class_counts: torch.Tensor,
    head_mu: float,
    kappa: float,
) -> torch.Tensor:
    """Compute the class-row-weighted proximal penalty for the linear head."""
    if head_mu < 0:
        raise ValueError("head_mu must be non-negative")
    weights = coverage_weights(class_counts, kappa).to(model.fc.weight.device)
    weight_distance = (model.fc.weight - global_head["weight"]).pow(2).sum(dim=1)
    bias_distance = (model.fc.bias - global_head["bias"]).pow(2)
    return 0.5 * head_mu * torch.sum(weights * (weight_distance + bias_distance))
