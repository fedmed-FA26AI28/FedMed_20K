"""Research algorithm registry for matched sequential FL experiments.

These are local objectives and aggregation rules for ``research.run``. They
are separate from Flower server strategies in ``algorithms.get_strategy``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.research_methods import objectives
from algorithms.research_methods.models import (
    TinyCNNBN, TinyCNNGN, classifier_head, make_model,
)
from algorithms.research_methods.support import (
    SupportStatistics, client_support_parameters, empirical_target_risk,
    fedswitch_oracle_weights, head_penalty, support_statistics,
)


@dataclass(frozen=True)
class ClientContext:
    local_counts: np.ndarray
    global_counts: np.ndarray
    coverage_sum: np.ndarray
    pooled_coverage_sum: np.ndarray
    total: int
    clients: int
    gamma: float
    coverage_kappa: float


def _plain(context: ClientContext) -> tuple[np.ndarray, np.ndarray]:
    return np.ones_like(context.local_counts), np.zeros_like(context.local_counts)


def _support(context: ClientContext) -> tuple[np.ndarray, np.ndarray]:
    return client_support_parameters(
        context.local_counts, context.global_counts, context.coverage_sum,
        context.total, context.clients, context.gamma)


def _support_uniform(context: ClientContext) -> tuple[np.ndarray, np.ndarray]:
    weights, _ = _support(context)
    return weights, np.ones_like(context.local_counts)


def _coverage_head(context: ClientContext) -> tuple[np.ndarray, np.ndarray]:
    _, anchor = _support(context)
    return np.ones_like(context.local_counts), anchor


def _pooled(context: ClientContext) -> tuple[np.ndarray, np.ndarray]:
    return client_support_parameters(
        context.local_counts, context.global_counts,
        context.pooled_coverage_sum, context.total, context.clients, math.inf)


def _coverage_iw(context: ClientContext) -> tuple[np.ndarray, np.ndarray]:
    counts = context.local_counts
    return np.ones_like(counts), context.coverage_kappa / (counts + context.coverage_kappa)


@dataclass(frozen=True)
class ResearchMethod:
    name: str
    objective: Callable[[objectives.LocalBatch], torch.Tensor]
    client_parameters: Callable[[ClientContext], tuple[np.ndarray, np.ndarray]] = _plain
    tuning_parameters: tuple[str, ...] = ()
    needs_teacher: bool = False
    oracle_aggregation: bool = False


METHOD_REGISTRY = {
    spec.name: spec for spec in (
        ResearchMethod("fedavg", objectives.fedavg),
        ResearchMethod("fedsld_literal", objectives.fedsld_literal),
        ResearchMethod("fedlc", objectives.fedlc, tuning_parameters=("fedlc_tau",)),
        ResearchMethod("fedvls", objectives.fedvls, tuning_parameters=("fedvls_lambda",),
                       needs_teacher=True),
        ResearchMethod("fedswitch_oracle", objectives.fedavg, oracle_aggregation=True),
        ResearchMethod("coverage_iw", objectives.coverage_iw, _coverage_iw,
                       tuning_parameters=("head_lambda",)),
        ResearchMethod("pooled_uniform_head", objectives.weighted_head, _pooled,
                       tuning_parameters=("head_lambda",)),
        ResearchMethod("coverage_head_only", objectives.coverage_head_only, _coverage_head,
                       tuning_parameters=("gamma", "head_lambda")),
        ResearchMethod("support_uniform_head", objectives.weighted_head, _support_uniform,
                       tuning_parameters=("gamma", "head_lambda")),
        ResearchMethod("support_calibrated", objectives.weighted_head, _support,
                       tuning_parameters=("gamma", "head_lambda")),
    )
}
METHODS = tuple(METHOD_REGISTRY)
TUNING_GRIDS = {"gamma": "gammas", "head_lambda": "head_lambdas",
                "fedlc_tau": "fedlc_taus", "fedvls_lambda": "fedvls_lambdas"}


def get_method(name: str) -> ResearchMethod:
    try:
        return METHOD_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown research method: {name}") from exc


def local_objective(
    method: str, model: nn.Module, images: torch.Tensor,
    labels: torch.Tensor, counts: torch.Tensor, global_counts: torch.Tensor,
    sample_weights: torch.Tensor, anchor: torch.Tensor,
    reference: tuple[torch.Tensor, torch.Tensor], head_lambda: float = 0.0,
    fedlc_tau: float = 0.5, teacher: nn.Module | None = None,
    fedvls_lambda: float = 0.1,
) -> torch.Tensor:
    """Dispatch one local minibatch loss through the algorithm registry."""
    spec = get_method(method)
    logits = model(images)
    ce = F.cross_entropy(logits, labels, reduction="none")
    batch = objectives.LocalBatch(
        model, images, labels, counts, global_counts, sample_weights, anchor,
        reference, head_lambda, fedlc_tau, teacher, fedvls_lambda, logits, ce,
    )
    return spec.objective(batch)


__all__ = [
    "ClientContext", "METHODS", "METHOD_REGISTRY", "ResearchMethod", "TUNING_GRIDS",
    "TinyCNNBN", "TinyCNNGN", "SupportStatistics", "classifier_head",
    "client_support_parameters", "empirical_target_risk", "fedswitch_oracle_weights",
    "get_method", "head_penalty", "local_objective", "make_model", "support_statistics",
]
