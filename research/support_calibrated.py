"""Compatibility imports for pre-move SCR notebooks and tests.

Research method implementations now live in ``algorithms.research_methods``.
Use that package for new algorithm code.
"""

from algorithms.research_methods import (
    METHODS, TinyCNNBN, TinyCNNGN, SupportStatistics, classifier_head,
    client_support_parameters, empirical_target_risk, fedswitch_oracle_weights,
    head_penalty, local_objective, make_model, support_statistics,
)

__all__ = [
    "METHODS", "TinyCNNBN", "TinyCNNGN", "SupportStatistics", "classifier_head",
    "client_support_parameters", "empirical_target_risk", "fedswitch_oracle_weights",
    "head_penalty", "local_objective", "make_model", "support_statistics",
]
