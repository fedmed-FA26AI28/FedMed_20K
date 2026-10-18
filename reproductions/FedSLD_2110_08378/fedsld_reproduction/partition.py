"""Reconstructed pathological and practical non-IID partition plans."""

from dataclasses import dataclass
from hashlib import sha256
from typing import List

import numpy as np


@dataclass(frozen=True)
class PartitionPlan:
    """Per-class fractions assigned to clients; rows sum to one."""

    fractions: np.ndarray
    name: str

    def __post_init__(self):
        values = np.asarray(self.fractions, dtype=np.float64)
        if values.ndim != 2 or np.any(values < 0):
            raise ValueError("fractions must have shape [classes, clients]")
        if not np.allclose(values.sum(axis=1), 1.0):
            raise ValueError("each class must be fully allocated")
        object.__setattr__(self, "fractions", values)

    @property
    def num_classes(self) -> int:
        return int(self.fractions.shape[0])

    @property
    def num_clients(self) -> int:
        return int(self.fractions.shape[1])


def practical_plan(num_classes: int, num_clients: int = 12, seed: int = 42) -> PartitionPlan:
    """Ten 1%, one 10%, and one 80% shard per class."""
    if num_clients != 12:
        raise ValueError("the paper's practical partition requires 12 clients")
    rng = np.random.default_rng(seed)
    base = np.asarray([0.80, 0.10] + [0.01] * 10, dtype=np.float64)
    fractions = np.vstack([rng.permutation(base) for _ in range(num_classes)])
    return PartitionPlan(fractions, "practical")


def pathological_plan(
    num_classes: int,
    num_clients: int = 12,
    seed: int = 42,
) -> PartitionPlan:
    """Assign exactly two classes to every client and random class amounts."""
    if num_classes < 2 or num_clients < 1:
        raise ValueError("at least two classes and one client are required")
    rng = np.random.default_rng(seed)
    slot_count = 2 * num_clients
    slots = np.arange(slot_count, dtype=np.int64) % num_classes
    pairs = None
    for _ in range(10_000):
        candidate = rng.permutation(slots).reshape(num_clients, 2)
        if np.all(candidate[:, 0] != candidate[:, 1]):
            pairs = candidate
            break
    if pairs is None:
        raise RuntimeError("could not construct distinct two-class assignments")

    fractions = np.zeros((num_classes, num_clients), dtype=np.float64)
    for class_id in range(num_classes):
        owners = np.flatnonzero(np.any(pairs == class_id, axis=1))
        if owners.size == 0:
            raise RuntimeError(f"class {class_id} has no assigned client")
        weights = rng.dirichlet(np.ones(owners.size))
        fractions[class_id, owners] = weights
    return PartitionPlan(fractions, "pathological")


def _integer_allocations(total: int, fractions: np.ndarray) -> np.ndarray:
    raw = total * fractions
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def partition_indices(labels, plan: PartitionPlan, seed: int = 42) -> List[List[int]]:
    """Apply a plan without duplicating or dropping examples."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if y.size == 0 or y.min() < 0 or y.max() >= plan.num_classes:
        raise ValueError("labels do not match the partition plan")
    rng = np.random.default_rng(seed)
    partitions: List[List[int]] = [[] for _ in range(plan.num_clients)]
    for class_id in range(plan.num_classes):
        indices = np.flatnonzero(y == class_id)
        rng.shuffle(indices)
        counts = _integer_allocations(len(indices), plan.fractions[class_id])
        start = 0
        for client_id, count in enumerate(counts):
            end = start + int(count)
            partitions[client_id].extend(indices[start:end].tolist())
            start = end
    for partition in partitions:
        rng.shuffle(partition)
    flattened = [index for partition in partitions for index in partition]
    if len(flattened) != len(y) or len(set(flattened)) != len(y):
        raise RuntimeError("partition must contain every example exactly once")
    return partitions


def partition_hash(indices: List[int]) -> str:
    values = np.asarray(sorted(indices), dtype=np.int64)
    return sha256(values.tobytes()).hexdigest()

