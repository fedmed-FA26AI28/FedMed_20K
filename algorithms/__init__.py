"""FL Algorithm strategies for FedMedAI.

Usage:
    from algorithms import get_strategy

    strategy = get_strategy("fedavg")       # default
    strategy = get_strategy("fedprox", proximal_mu=0.1)
    strategy = get_strategy("fednova")
"""

from algorithms.fedavg import FedAvgStrategy
from algorithms.fedprox import FedProxStrategy
from algorithms.fednova import FedNovaStrategy


STRATEGY_REGISTRY = {
    "fedavg": FedAvgStrategy,
    # Coverage-aware learning changes the client objective, not aggregation.
    "coverage": FedAvgStrategy,
    "balanced": FedAvgStrategy,
    "logit_only": FedAvgStrategy,
    "head_only": FedAvgStrategy,
    "fedprox": FedProxStrategy,
    "fednova": FedNovaStrategy,
}


def get_strategy(name: str = "fedavg", **kwargs):
    """Factory function to create a strategy by name.

    Args:
        name: One of 'fedavg', 'fedprox', 'fednova' (case-insensitive).
        **kwargs: Forwarded to the strategy constructor.

    Returns:
        An instance of the selected strategy.

    Raises:
        ValueError: If the strategy name is not recognised.
    """
    key = name.strip().lower()
    if key not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {available}"
        )
    return STRATEGY_REGISTRY[key](**kwargs)
