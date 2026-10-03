"""Proposed FL improvement method.

This module re-exports the selected improved strategy for the FedMedAI project.
After literature review, the chosen improvement can be changed here.

Current selection: FedProx (addresses client/system heterogeneity via proximal term).
Alternative: FedNova (addresses objective inconsistency with heterogeneous local steps).
"""

from algorithms.fedprox import FedProxStrategy as ProposedStrategy  # noqa: F401
from algorithms.fednova import FedNovaStrategy  # noqa: F401
from algorithms.fedbn import FedBNStrategy  # noqa: F401
from algorithms.scaffold import SCAFFOLDStrategy  # noqa: F401

__all__ = ["ProposedStrategy", "FedNovaStrategy", "FedBNStrategy", "SCAFFOLDStrategy"]
