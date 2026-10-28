"""FL Algorithm strategies for FedMedAI.

Usage:
    from algorithms import get_strategy

    strategy = get_strategy("fedavg")       # default
    strategy = get_strategy("fedprox", proximal_mu=0.1)
    strategy = get_strategy("fednova")
    strategy = get_strategy("fedbn")
    strategy = get_strategy("scaffold")
    strategy = get_strategy("fedbns")       # FedBN + SCAFFOLD (alias: "proposed")
    strategy = get_strategy("cla")          # CLA strategy skeleton
    strategy = get_strategy("scr")          # SCR strategy skeleton
"""

from algorithms.base import FedStrategyBase
from algorithms.fedavg import FedAvgStrategy
from algorithms.fedprox import FedProxStrategy
from algorithms.fednova import FedNovaStrategy
from algorithms.fedbn import FedBNStrategy
from algorithms.scaffold import SCAFFOLDStrategy
from algorithms.fedbns import (
    FedBNSStrategy,
    FedProposedStrategy,
    ProposedStrategy,
    FedBNScaffoldStrategy,
    ScaffoldFedBNStrategy,
)
from algorithms.cla import CLAStrategy
from algorithms.scr import SCRStrategy


STRATEGY_REGISTRY = {
    # Core Baselines
    "fedavg": FedAvgStrategy,
    "fedprox": FedProxStrategy,
    "fednova": FedNovaStrategy,
    "fedbn": FedBNStrategy,
    "scaffold": SCAFFOLDStrategy,
    # FedBNS (Proposed FedBN + SCAFFOLD) & Aliases
    "fedbns": FedBNSStrategy,
    "proposed": ProposedStrategy,
    "fedbn_scaffold": FedBNScaffoldStrategy,
    "bn_scaffold": FedBNScaffoldStrategy,
    "scaffold_fedbn": ScaffoldFedBNStrategy,
    # New Strategies (Skeletons)
    "cla": CLAStrategy,
    "scr": SCRStrategy,
}


def get_strategy(name: str = "fedavg", **kwargs):
    """Factory function to create a strategy by name.

    Args:
        name: One of 'fedavg', 'fedprox', 'fednova', 'fedbn', 'scaffold', 'fedbns', 'proposed', 'cla', 'scr'.
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


__all__ = [
    "FedStrategyBase",
    "FedAvgStrategy",
    "FedProxStrategy",
    "FedNovaStrategy",
    "FedBNStrategy",
    "SCAFFOLDStrategy",
    "FedBNSStrategy",
    "FedProposedStrategy",
    "ProposedStrategy",
    "FedBNScaffoldStrategy",
    "ScaffoldFedBNStrategy",
    "CLAStrategy",
    "SCRStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
]
