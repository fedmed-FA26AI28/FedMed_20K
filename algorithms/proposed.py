"""Proposed FL Strategy Compatibility Wrapper.

This module provides backward-compatibility forwarding to `algorithms.fedbns.FedBNSStrategy`.
The 'proposed' strategy has been renamed to 'FedBNS'.
"""

from algorithms.fedbns import (
    FedBNSStrategy,
    FedProposedStrategy,
    ProposedStrategy,
    FedBNScaffoldStrategy,
    ScaffoldFedBNStrategy,
    _fedbn_scaffold_aggregate,
)

__all__ = [
    "FedBNSStrategy",
    "FedProposedStrategy",
    "ProposedStrategy",
    "FedBNScaffoldStrategy",
    "ScaffoldFedBNStrategy",
    "_fedbn_scaffold_aggregate",
]
