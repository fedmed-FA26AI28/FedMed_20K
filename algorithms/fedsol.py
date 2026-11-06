"""FedSOL Strategy: Stabilized Orthogonal Learning with Proximal Restrictions.

Reference:
    "FedSOL: Stabilized Orthogonal Learning with Proximal Restrictions in Federated Learning"
    Gihun Lee, Minchan Jeong, Sangmook Kim, Jaehoon Oh, Se-Young Yun (CVPR 2024).

FedSOL addresses client drift and catastrophic forgetting under non-IID data distributions
by identifying parameter regions where learning on the local objective is naturally orthogonal
to directions affecting global knowledge preservation (proximal loss).
"""

from __future__ import annotations

from logging import INFO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from algorithms.base import FedStrategyBase


class FedSOLStrategy(FedStrategyBase):
    """Federated Stabilized Orthogonal Learning (FedSOL) Strategy (CVPR 2024).

    Args:
        fedsol_rho: Perturbation radius strength rho (default: 0.05 for fixed, 1.0 for adaptive).
        fedsol_loss_type: Discrepancy metric for proximal loss: 'kl' (KL-Divergence) or 'l2' (L2 weight distance).
        fedsol_temperature: Softmax temperature for KL divergence discrepancy (default: 3.0).
        fedsol_adaptive: Whether to use adaptive perturbation radius scaling based on parameter magnitude.
        model: Global model instance for inspection.
        **kwargs: Forwarded to FedStrategyBase.
    """

    def __init__(
        self,
        *,
        fedsol_rho: float = 0.05,
        fedsol_loss_type: str = "kl",
        fedsol_temperature: float = 3.0,
        fedsol_adaptive: bool = False,
        model: Optional[nn.Module] = None,
        strategy_name: str = "FedSOL",
        **kwargs,
    ):
        super().__init__(
            model=model,
            strategy_name=strategy_name,
            **kwargs,
        )
        self.fedsol_rho = float(fedsol_rho)
        self.fedsol_loss_type = str(fedsol_loss_type).lower().strip()
        if self.fedsol_loss_type not in ("kl", "l2"):
            raise ValueError(f"Unknown fedsol_loss_type: '{fedsol_loss_type}'. Must be 'kl' or 'l2'.")
        self.fedsol_temperature = float(fedsol_temperature)
        self.fedsol_adaptive = bool(fedsol_adaptive)

        log(
            INFO,
            "[FedSOL] Initialized strategy with rho=%g, loss_type=%s, temp=%g, adaptive=%s",
            self.fedsol_rho,
            self.fedsol_loss_type,
            self.fedsol_temperature,
            self.fedsol_adaptive,
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training, broadcasting FedSOL hyperparameters."""
        client_instructions = super().configure_fit(server_round, parameters, client_manager)

        for _, fit_ins in client_instructions:
            fit_ins.config["strategy"] = "fedsol"
            fit_ins.config["fedsol_rho"] = self.fedsol_rho
            fit_ins.config["fedsol_loss_type"] = self.fedsol_loss_type
            fit_ins.config["fedsol_temperature"] = self.fedsol_temperature
            fit_ins.config["fedsol_adaptive"] = self.fedsol_adaptive

        return client_instructions
