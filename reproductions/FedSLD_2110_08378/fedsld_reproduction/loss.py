"""FedSLD objective from Equations (1)--(3) of arXiv:2110.08378v1."""

import torch
import torch.nn.functional as F


def federation_prior(client_class_counts: torch.Tensor) -> torch.Tensor:
    """Compute the global label prior from client class-count vectors."""
    counts = torch.as_tensor(client_class_counts, dtype=torch.float64)
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError("client_class_counts must have shape [clients, classes]")
    if torch.any(counts < 0) or not torch.isfinite(counts).all():
        raise ValueError("class counts must be finite and non-negative")
    totals = counts.sum(dim=0)
    if torch.any(totals <= 0):
        raise ValueError("every configured class must appear in the federation")
    return (totals / totals.sum()).to(dtype=torch.float32)


def fedsld_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    global_prior: torch.Tensor,
) -> torch.Tensor:
    """Weighted mean CE with paper weight p_batch(y) / p_global(y)."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    targets = targets.reshape(-1).long()
    if targets.numel() != logits.shape[0]:
        raise ValueError("one target is required for every logit row")
    prior = torch.as_tensor(global_prior, device=logits.device, dtype=logits.dtype)
    if prior.ndim != 1 or prior.numel() != logits.shape[1]:
        raise ValueError("global_prior must contain one value per class")
    if torch.any(prior <= 0) or not torch.isfinite(prior).all():
        raise ValueError("global_prior must be finite and strictly positive")
    if torch.any((targets < 0) | (targets >= logits.shape[1])):
        raise ValueError("target outside configured class range")

    batch_counts = torch.bincount(targets, minlength=logits.shape[1]).to(logits.dtype)
    batch_prior = batch_counts / targets.numel()
    sample_weights = batch_prior[targets] / prior[targets]
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    return torch.mean(sample_weights * per_sample)

