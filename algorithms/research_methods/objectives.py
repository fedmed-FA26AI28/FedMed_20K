"""Local losses used by the matched research runner.

Each named function is one experiment method's minibatch objective. The runner
passes training-count statistics; inference always uses raw model logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.research_methods.support import head_penalty


@dataclass(frozen=True)
class LocalBatch:
    model: nn.Module
    images: torch.Tensor
    labels: torch.Tensor
    counts: torch.Tensor
    global_counts: torch.Tensor
    sample_weights: torch.Tensor
    anchor: torch.Tensor
    reference: tuple[torch.Tensor, torch.Tensor]
    head_lambda: float
    fedlc_tau: float
    teacher: nn.Module | None
    fedvls_lambda: float
    logits: torch.Tensor
    ce: torch.Tensor


def fedavg(batch: LocalBatch) -> torch.Tensor:
    return batch.ce.mean()


def coverage_head_only(batch: LocalBatch) -> torch.Tensor:
    return batch.ce.mean() + head_penalty(
        batch.model, batch.reference, batch.anchor, batch.head_lambda)


def fedsld_literal(batch: LocalBatch) -> torch.Tensor:
    prior = torch.bincount(batch.labels, minlength=batch.logits.shape[1]).to(batch.logits)
    prior = prior / len(batch.labels)
    global_prior = batch.global_counts / batch.global_counts.sum()
    return (prior[batch.labels] / global_prior[batch.labels] * batch.ce).sum()


def coverage_iw(batch: LocalBatch) -> torch.Tensor:
    classes = batch.logits.shape[1]
    batch_counts = torch.bincount(batch.labels, minlength=classes).to(batch.logits)
    batch_prior = (batch_counts + 1) / (len(batch.labels) + classes)
    global_prior = (batch.global_counts + 1) / (batch.global_counts.sum() + classes)
    ratio = global_prior[batch.labels] / batch_prior[batch.labels]
    inverse_loss = (ratio * batch.ce).sum() / ratio.sum()
    return inverse_loss + head_penalty(
        batch.model, batch.reference, batch.anchor,
        batch.head_lambda * len(batch.anchor))


def fedlc(batch: LocalBatch) -> torch.Tensor:
    offset = batch.fedlc_tau * batch.counts.clamp_min(1).pow(-0.25)
    if batch.fedlc_tau > 0:
        offset = offset.masked_fill(batch.counts == 0, float("inf"))
    return F.cross_entropy(batch.logits - offset, batch.labels)


def fedvls(batch: LocalBatch) -> torch.Tensor:
    # Adapted mathematical losses from the authors' released clientvls.py.
    observed = batch.counts > 0
    prior = batch.counts / batch.counts.sum()
    log_prior = (prior + 1e-9).log()
    loss = F.cross_entropy(batch.logits + log_prior, batch.labels)
    adjusted = batch.logits - log_prior + batch.logits.new_tensor(1 / batch.logits.shape[1]).log()
    nonlabel = batch.labels[:, None] != torch.arange(batch.logits.shape[1], device=batch.logits.device)
    spread = torch.where(nonlabel, adjusted, torch.zeros_like(adjusted))
    suppress = (prior * (torch.logsumexp(spread, dim=0) -
                         batch.logits.new_tensor(len(batch.labels)).log())).sum()
    loss = loss + 0.005 * suppress
    vacant = torch.where(~observed)[0]
    if batch.teacher is not None and len(vacant) > 1:
        with torch.no_grad():
            teacher_prob = F.softmax(batch.teacher(batch.images)[:, vacant], dim=1)
        loss = loss + batch.fedvls_lambda * F.kl_div(
            F.log_softmax(batch.logits[:, vacant], dim=1), teacher_prob,
            reduction="batchmean",
        )
    return loss


def weighted_head(batch: LocalBatch) -> torch.Tensor:
    loss = (batch.sample_weights[batch.labels] * batch.ce).mean()
    return loss + head_penalty(
        batch.model, batch.reference, batch.anchor, batch.head_lambda)
