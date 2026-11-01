"""CLA (Minimalist AdaCoverage + Prior-Aligned Logit Adjustment) Federated Learning Strategy.

Version 2 (Minimalist 1-Hyperparameter `mu`):
- Prior-Aligned Logit Adjustment (la_mode='prior_aligned'): Bayes log-odds ratio offset
  delta_{k,c} = log(pi~_{k,c}) - log(pi_{global,c}), centering adjustments around zero
  without distorting global test priors.
- Empirical Bayes Adaptive Coverage (adaptive_kappa=True):
  a_{k,c}^{head} = n_k / (C * n_{k,c} + n_k), parameter-free adaptive anchoring.
- Back-Projected Multi-Linear Coverage (multi_linear_coverage=True):
  Propagates coverage weights backwards through all nn.Linear layers via reference weights W_ref^2,
  with scale-invariance normalisation rho_l = numel(W^(L)) / numel(W^(l)).
- Separated Optimizer Parameter Groups (disable_covered_wd=True):
  Weight decay is disabled on covered Linear layers to prevent pulling zero-sample weights to 0.
- Exact Pooled BatchNorm Aggregation (bn_mode='pooled_global'):
  Aggregates both within-client and between-client running variances accurately.

Backward-compatible with CLA v1 and standard FedAvg through ablation flags and presets.
Reference: docs/cla_v2.md
"""

from __future__ import annotations

import copy
import json
from logging import INFO, WARNING
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from algorithms.base import (
    FedStrategyBase,
    _format_device_details,
    _weighted_average_metrics,
)
from monitoring.metrics import FLMetricsRecorder


# -----------------------------------------------------------------------------
# Ablation Presets Definition
# -----------------------------------------------------------------------------
ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "A0_fedavg": {
        "mu": 0.0,
        "use_la": False,
        "disable_covered_wd": False,
        "bn_mode": "naive_avg",
    },
    "A1_cla_v1_legacy": {
        "mu": 0.1,
        "use_la": True,
        "la_mode": "v1_legacy",
        "adaptive_kappa": False,
        "multi_linear_coverage": False,
        "disable_covered_wd": False,
        "bn_mode": "naive_avg",
    },
    "A2_prior_aligned_la_only": {
        "mu": 0.0,
        "use_la": True,
        "la_mode": "prior_aligned",
        "disable_covered_wd": False,
        "bn_mode": "naive_avg",
    },
    "A3_head_bayes_cov": {
        "mu": 0.1,
        "use_la": True,
        "la_mode": "prior_aligned",
        "adaptive_kappa": True,
        "multi_linear_coverage": False,
        "disable_covered_wd": True,
        "bn_mode": "naive_avg",
    },
    "A4_multi_linear_cov": {
        "mu": 0.1,
        "use_la": True,
        "la_mode": "prior_aligned",
        "adaptive_kappa": True,
        "multi_linear_coverage": True,
        "disable_covered_wd": True,
        "bn_mode": "naive_avg",
    },
    "A5_full_proposed_v2": {
        "mu": 0.1,
        "use_la": True,
        "la_mode": "prior_aligned",
        "adaptive_kappa": True,
        "multi_linear_coverage": True,
        "disable_covered_wd": True,
        "bn_mode": "pooled_global",
    },
}


def get_linear_layers_in_order(model: nn.Module) -> List[nn.Linear]:
    """Extract list of nn.Linear layers in model definition order.
    
    The last element [-1] is the classifier head (d -> C).
    """
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linear_layers) == 0:
        raise ValueError("Model must contain at least one nn.Linear layer.")
    return linear_layers


def load_global_counts_from_partition(
    partition_path: Union[str, Path],
    num_classes: Optional[int] = None,
) -> List[int]:
    """Extract total dataset class counts from partition JSON."""
    p_path = Path(partition_path)
    if not p_path.exists():
        return []
    try:
        with open(p_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        clients = data.get("clients", {})
        if not clients:
            return []

        if num_classes is None:
            max_c = 0
            for entry in clients.values():
                dist = entry.get("class_distribution", {})
                for c_str in dist.keys():
                    max_c = max(max_c, int(c_str))
            num_classes = max_c + 1

        counts = [0] * num_classes
        for entry in clients.values():
            dist = entry.get("class_distribution", {})
            for c_str, count in dist.items():
                c = int(c_str)
                if c < num_classes:
                    counts[c] += int(count)
        return counts
    except Exception as e:
        log(WARNING, "Failed to load global class counts from %s: %s", partition_path, e)
        return []


class MinimalistAdaCoverageLACriterion(nn.Module):
    """AdaCoverage + Prior-Aligned LA Objective (1-hyperparameter `mu`).

    - Prior-Aligned Logit Adjustment: Bayes log-odds ratio offset (no tau, eps needed).
    - Empirical Bayes adaptive head coverage weights (no kappa needed).
    - Back-projected coverage weights for hidden linear layers with parameter-scale invariance.
    """

    def __init__(
        self,
        class_counts: List[int],
        global_class_counts: Optional[List[int]],
        num_classes: int,
        ref_linear_layers: List[nn.Linear],
        mu: float = 0.1,                        # Sole primary hyperparameter
        use_la: bool = True,
        la_mode: str = "prior_aligned",         # "prior_aligned" (v2) or "v1_legacy" (v1)
        adaptive_kappa: bool = True,            # True (v2: parameter-free) or False (v1)
        multi_linear_coverage: bool = True,     # True (v2: all Linear layers) or False (v1: head only)
        legacy_kappa: float = 512.0,            # Used only if adaptive_kappa=False
        legacy_tau: float = 1.0,                # Used only if la_mode="v1_legacy"
        legacy_eps: float = 1.0,                # Used only if la_mode="v1_legacy"
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.mu = float(mu)
        self.use_la = bool(use_la)
        self.la_mode = str(la_mode)
        self.multi_linear_coverage = bool(multi_linear_coverage)

        counts_k = torch.tensor(class_counts, dtype=torch.float32, device=device)
        assert counts_k.numel() == self.num_classes, (
            f"Expected {self.num_classes} classes, got {counts_k.numel()}"
        )
        n_k = counts_k.sum().clamp(min=1.0)
        C = float(self.num_classes)

        # ---------------------------------------------------------------------
        # 1. Logit Adjustment offset (delta_k)
        # ---------------------------------------------------------------------
        if not self.use_la:
            delta_k = torch.zeros_like(counts_k)
        elif self.la_mode == "prior_aligned" and global_class_counts is not None:
            counts_g = torch.tensor(global_class_counts, dtype=torch.float32, device=device)
            n_g = counts_g.sum().clamp(min=1.0)
            pi_k = (counts_k + 1.0) / (n_k + C)
            pi_g = (counts_g + 1.0) / (n_g + C)
            delta_k = torch.log(pi_k) - torch.log(pi_g)
        elif self.la_mode == "v1_legacy" or global_class_counts is None:
            delta_k = float(legacy_tau) * torch.log(counts_k + float(legacy_eps))
        else:
            raise ValueError(f"Unsupported la_mode: {self.la_mode}")

        self.register_buffer("delta_k", delta_k)

        # ---------------------------------------------------------------------
        # 2. Coverage weights: Head and back-projected hidden layers
        # ---------------------------------------------------------------------
        if adaptive_kappa:
            # Empirical Bayes Shrinkage: a_{k,c} = n_k / (C * n_{k,c} + n_k)
            a_head = n_k / (C * counts_k + n_k)
        else:
            k_val = float(legacy_kappa)
            a_head = k_val / (counts_k + k_val)

        active_refs = ref_linear_layers if self.multi_linear_coverage else [ref_linear_layers[-1]]
        self.num_active_layers = len(active_refs)
        head_numel = float(active_refs[-1].weight.numel())

        a_curr = a_head
        scale_list: List[float] = [1.0] * self.num_active_layers

        for idx in range(self.num_active_layers - 1, -1, -1):
            layer_ref = active_refs[idx]
            W_ref = layer_ref.weight.detach().clone().to(device)
            b_ref = layer_ref.bias.detach().clone().to(device) if layer_ref.bias is not None else None

            self.register_buffer(f"W_ref_{idx}", W_ref)
            if b_ref is not None:
                self.register_buffer(f"b_ref_{idx}", b_ref)
            else:
                setattr(self, f"b_ref_{idx}", None)

            self.register_buffer(f"a_{idx}", a_curr)
            scale_list[idx] = head_numel / float(W_ref.numel())

            # Back-project coverage weights to preceding linear layer
            if idx > 0:
                W_sq = W_ref ** 2  # [d_out, d_in]
                omega = W_sq / (W_sq.sum(dim=0, keepdim=True) + 1e-8)  # [d_out, d_in]
                a_curr = torch.matmul(a_curr.unsqueeze(0), omega).squeeze(0)  # [d_in]

        self.scale_factors = scale_list

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        curr_linear_layers: List[nn.Linear],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            ce_raw = F.cross_entropy(logits, targets)

        adjusted_logits = logits + self.delta_k.unsqueeze(0)
        ce_la = F.cross_entropy(adjusted_logits, targets)

        if self.mu <= 0.0:
            r_cov = torch.zeros((), device=logits.device, dtype=logits.dtype)
            r_head_val = 0.0
            r_hidden_val = 0.0
        else:
            active_currs = (
                curr_linear_layers if self.multi_linear_coverage else [curr_linear_layers[-1]]
            )
            assert len(active_currs) == self.num_active_layers, (
                f"Expected {self.num_active_layers} active linear layers, got {len(active_currs)}"
            )

            r_head = torch.zeros((), device=logits.device, dtype=logits.dtype)
            r_hidden = torch.zeros((), device=logits.device, dtype=logits.dtype)

            for idx, layer_curr in enumerate(active_currs):
                W_ref = getattr(self, f"W_ref_{idx}")
                b_ref = getattr(self, f"b_ref_{idx}")
                a_layer = getattr(self, f"a_{idx}")
                rho = self.scale_factors[idx]

                row_sq_diff = torch.sum((layer_curr.weight - W_ref) ** 2, dim=1)
                if b_ref is not None and layer_curr.bias is not None:
                    row_sq_diff = row_sq_diff + (layer_curr.bias - b_ref) ** 2

                penalty_layer = 0.5 * self.mu * rho * torch.sum(a_layer * row_sq_diff)
                if idx == self.num_active_layers - 1:
                    r_head = r_head + penalty_layer
                else:
                    r_hidden = r_hidden + penalty_layer

            r_cov = r_head + r_hidden
            r_head_val = float(r_head.detach().item())
            r_hidden_val = float(r_hidden.detach().item())

        total_loss = ce_la + r_cov
        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "ce_raw": float(ce_raw.item()),
            "ce_la": float(ce_la.detach().item()),
            "r_head": r_head_val,
            "r_hidden": r_hidden_val,
            "r_cov_total": float(r_cov.detach().item()),
        }
        return total_loss, metrics


def build_client_optimizer(
    model: nn.Module,
    covered_linear_layers: List[nn.Linear],
    optimizer_name: str = "adamw",
    lr: float = 0.001,
    weight_decay: float = 0.01,
    momentum: float = 0.9,
    disable_covered_wd: bool = True,
) -> torch.optim.Optimizer:
    """Build client optimizer separating covered linear parameters from backbone."""
    covered_param_ids = {
        id(p) for layer in covered_linear_layers for p in layer.parameters()
    }
    backbone_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in covered_param_ids
    ]
    covered_params = [
        p for layer in covered_linear_layers for p in layer.parameters() if p.requires_grad
    ]

    covered_wd = 0.0 if disable_covered_wd else weight_decay
    param_groups = [
        {"params": backbone_params, "weight_decay": weight_decay},
        {"params": covered_params, "weight_decay": covered_wd},
    ]

    opt_name = str(optimizer_name).lower().strip()
    if opt_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif opt_name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif opt_name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


@torch.no_grad()
def aggregate_models_with_pooled_bn(
    global_model: nn.Module,
    client_state_dicts: List[Dict[str, torch.Tensor]],
    client_sample_counts: List[int],
    bn_mode: str = "pooled_global",  # "pooled_global", "naive_avg", or "local_bn"
) -> Dict[str, torch.Tensor]:
    """Aggregate model state dictionaries with exact pooled BatchNorm statistics."""
    total_samples = float(sum(client_sample_counts))
    if total_samples <= 0:
        return copy.deepcopy(global_model.state_dict())

    weights = [float(n) / total_samples for n in client_sample_counts]
    new_state = copy.deepcopy(global_model.state_dict())

    bn_prefixes = {
        name for name, m in global_model.named_modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    }

    for key in new_state.keys():
        prefix = key.rsplit(".", 1)[0] if "." in key else ""
        if bn_mode == "local_bn" and prefix in bn_prefixes:
            continue

        if key.endswith("num_batches_tracked"):
            new_state[key] = client_state_dicts[0][key].clone()
            continue

        if bn_mode == "pooled_global" and key.endswith("running_var"):
            mean_key = key[:-11] + "running_mean"
            pooled_mean = torch.zeros_like(new_state[mean_key], dtype=torch.float32)
            for w_k, c_state in zip(weights, client_state_dicts):
                pooled_mean += w_k * c_state[mean_key].to(torch.float32)

            pooled_var = torch.zeros_like(new_state[key], dtype=torch.float32)
            for w_k, c_state in zip(weights, client_state_dicts):
                var_k = c_state[key].to(torch.float32)
                mean_k = c_state[mean_key].to(torch.float32)
                pooled_var += w_k * (var_k + (mean_k - pooled_mean) ** 2)

            new_state[key] = pooled_var.to(new_state[key].dtype)
            continue

        acc = torch.zeros_like(new_state[key], dtype=torch.float32)
        for w_k, c_state in zip(weights, client_state_dicts):
            acc += w_k * c_state[key].to(torch.float32)
        new_state[key] = acc.to(new_state[key].dtype)

    global_model.load_state_dict(new_state)
    return new_state


# -----------------------------------------------------------------------------
# Flower Strategy
# -----------------------------------------------------------------------------
class CLAStrategy(FedStrategyBase):
    """CLA (AdaCoverage + Prior-Aligned Logit Adjustment) Federated Strategy.

    Args:
        mu: Sole primary hyperparameter controlling Coverage penalty intensity (default: 0.1).
        use_la: Toggle Logit Adjustment (default: True).
        la_mode: "prior_aligned" (v2 Bayes Log-Odds) or "v1_legacy" (default: "prior_aligned").
        adaptive_kappa: Parameter-free shrinkage kappa = n_k / C (default: True).
        multi_linear_coverage: Back-project coverage across all nn.Linear layers (default: True).
        disable_covered_wd: Disable weight decay on covered Linear layers (default: True).
        bn_mode: BatchNorm aggregation mode: "pooled_global", "naive_avg", "local_bn" (default: "pooled_global").
        ablation_preset: Optional preset key from ABLATION_PRESETS (e.g. "A0_fedavg", "A5_full_proposed_v2").
        legacy_kappa: Fixed kappa value for legacy v1 mode (default: 512.0).
        legacy_tau: Fixed tau for legacy v1 LA mode (default: 1.0).
        legacy_eps: Fixed eps for legacy v1 LA mode (default: 1.0).
        coverage_mu_head: Legacy alias for mu (forwarded if mu is not customized).
        coverage_kappa: Legacy alias for legacy_kappa.
        la_tau: Legacy alias for legacy_tau.
        la_epsilon: Legacy alias for legacy_eps.
        model: Global model instance (used for pooled BN aggregation and architecture inspection).
        partition_path: Path to partition JSON for computing global class counts.
        global_class_counts: Explicit global dataset class distribution counts.
        early_stop_patience: Rounds without improvement before stopping.
        early_stop_metric: Target metric to track ('accuracy' or 'loss').
        metrics_recorder: Optional FLMetricsRecorder instance.
        local_test: Whether local test partitions are enabled.
        strategy_name: Name of the strategy (default: "CLA").
        **kwargs: Forwarded to Flower FedAvg.
    """

    def __init__(
        self,
        *,
        mu: float = 0.1,
        use_la: bool = True,
        la_mode: str = "prior_aligned",
        adaptive_kappa: bool = True,
        multi_linear_coverage: bool = True,
        disable_covered_wd: bool = True,
        bn_mode: str = "pooled_global",
        ablation_preset: Optional[str] = None,
        legacy_kappa: float = 512.0,
        legacy_tau: float = 1.0,
        legacy_eps: float = 1.0,
        coverage_mu_head: Optional[float] = None,
        coverage_kappa: Optional[float] = None,
        la_tau: Optional[float] = None,
        la_epsilon: Optional[float] = None,
        model: Optional[nn.Module] = None,
        partition_path: Optional[Union[str, Path]] = None,
        global_class_counts: Optional[List[int]] = None,
        early_stop_patience: int = 10,
        early_stop_metric: str = "accuracy",
        metrics_recorder: Optional[FLMetricsRecorder] = None,
        local_test: bool = False,
        strategy_name: str = "CLA",
        **kwargs,
    ):
        # Resolve legacy aliases
        if coverage_mu_head is not None and mu == 0.1:
            mu = coverage_mu_head
        if coverage_kappa is not None:
            legacy_kappa = coverage_kappa
        if la_tau is not None:
            legacy_tau = la_tau
        if la_epsilon is not None:
            legacy_eps = la_epsilon

        # Resolve ablation preset if provided
        resolved_preset = None
        if ablation_preset:
            p_key = None
            for key in ABLATION_PRESETS:
                if key.lower() == ablation_preset.strip().lower():
                    p_key = key
                    break
            if p_key is None:
                valid_presets = ", ".join(ABLATION_PRESETS.keys())
                raise ValueError(f"Unknown ablation preset '{ablation_preset}'. Available: {valid_presets}")
            
            resolved_preset = p_key
            preset_cfg = ABLATION_PRESETS[p_key]
            mu = preset_cfg.get("mu", mu)
            use_la = preset_cfg.get("use_la", use_la)
            la_mode = preset_cfg.get("la_mode", la_mode)
            adaptive_kappa = preset_cfg.get("adaptive_kappa", adaptive_kappa)
            multi_linear_coverage = preset_cfg.get("multi_linear_coverage", multi_linear_coverage)
            disable_covered_wd = preset_cfg.get("disable_covered_wd", disable_covered_wd)
            bn_mode = preset_cfg.get("bn_mode", bn_mode)

        if mu < 0:
            raise ValueError(f"mu must be >= 0, got {mu}")
        if legacy_kappa <= 0:
            raise ValueError(f"legacy_kappa must be > 0, got {legacy_kappa}")
        if legacy_tau < 0:
            raise ValueError(f"legacy_tau must be >= 0, got {legacy_tau}")
        if legacy_eps <= 0:
            raise ValueError(f"legacy_eps must be > 0, got {legacy_eps}")
        if la_mode not in ("prior_aligned", "v1_legacy"):
            raise ValueError(f"la_mode must be 'prior_aligned' or 'v1_legacy', got '{la_mode}'")
        if bn_mode not in ("pooled_global", "naive_avg", "local_bn"):
            raise ValueError(f"bn_mode must be 'pooled_global', 'naive_avg', or 'local_bn', got '{bn_mode}'")

        self.mu = float(mu)
        self.use_la = bool(use_la)
        self.la_mode = str(la_mode)
        self.adaptive_kappa = bool(adaptive_kappa)
        self.multi_linear_coverage = bool(multi_linear_coverage)
        self.disable_covered_wd = bool(disable_covered_wd)
        self.bn_mode = str(bn_mode)
        self.ablation_preset = resolved_preset
        self.legacy_kappa = float(legacy_kappa)
        self.legacy_tau = float(legacy_tau)
        self.legacy_eps = float(legacy_eps)

        self.partition_path = partition_path
        if global_class_counts is None and partition_path:
            self.global_class_counts = load_global_counts_from_partition(partition_path)
        else:
            self.global_class_counts = global_class_counts

        recorder = metrics_recorder or FLMetricsRecorder(
            strategy_name=strategy_name,
        )

        super().__init__(
            early_stop_patience=early_stop_patience,
            early_stop_metric=early_stop_metric,
            metrics_recorder=recorder,
            model=model,
            local_test=local_test,
            strategy_name=strategy_name,
            **kwargs,
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure client fit instructions, injecting CLA v2 configurations."""
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        if not client_instructions:
            return client_instructions

        for _, fit_ins in client_instructions:
            cfg = fit_ins.config
            # Inject CLA v2 hyperparameter configurations
            cfg["mu"] = self.mu
            cfg["coverage_mu_head"] = self.mu  # Legacy compatibility alias
            cfg["use_la"] = self.use_la
            cfg["la_mode"] = self.la_mode
            cfg["adaptive_kappa"] = self.adaptive_kappa
            cfg["multi_linear_coverage"] = self.multi_linear_coverage
            cfg["disable_covered_wd"] = self.disable_covered_wd
            cfg["bn_mode"] = self.bn_mode
            cfg["legacy_kappa"] = self.legacy_kappa
            cfg["coverage_kappa"] = self.legacy_kappa  # Legacy compatibility alias
            cfg["legacy_tau"] = self.legacy_tau
            cfg["la_tau"] = self.legacy_tau            # Legacy compatibility alias
            cfg["legacy_eps"] = self.legacy_eps
            cfg["la_epsilon"] = self.legacy_eps        # Legacy compatibility alias

            if self.ablation_preset:
                cfg["ablation_preset"] = self.ablation_preset
            if self.global_class_counts:
                cfg["global_class_counts"] = json.dumps(self.global_class_counts)

        return client_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client parameters using Exact Pooled BN Aggregation (or selected bn_mode)."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        if self.model is not None:
            keys = list(self.model.state_dict().keys())
            client_state_dicts: List[Dict[str, torch.Tensor]] = []
            client_sample_counts: List[int] = []

            for _, fit_res in results:
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                state = {k: torch.from_numpy(arr) for k, arr in zip(keys, ndarrays)}
                client_state_dicts.append(state)
                client_sample_counts.append(fit_res.num_examples)

            new_state = aggregate_models_with_pooled_bn(
                global_model=self.model,
                client_state_dicts=client_state_dicts,
                client_sample_counts=client_sample_counts,
                bn_mode=self.bn_mode,
            )
            aggregated_ndarrays = [val.cpu().numpy() for val in new_state.values()]
            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        else:
            parameters_aggregated, _ = super().aggregate_fit(
                server_round, results, failures
            )

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        metrics_aggregated = self._finalize_fit_aggregation(
            server_round=server_round,
            parameters_aggregated=parameters_aggregated,
            metrics_aggregated=metrics_aggregated,
            results=results,
            failures=failures,
        )

        preset_str = f" [preset={self.ablation_preset}]" if self.ablation_preset else ""
        log(
            INFO,
            "[%s] Round %d | %.1fs | clients=%d | failures=%d | "
            "mu=%.4f%s | la_mode=%s | adaptive_kappa=%s | multi_linear=%s | disable_wd=%s | bn_mode=%s",
            self.strategy_name,
            server_round,
            metrics_aggregated.get("round_time_seconds", 0.0),
            len(results),
            len(failures),
            self.mu,
            preset_str,
            self.la_mode,
            self.adaptive_kappa,
            self.multi_linear_coverage,
            self.disable_covered_wd,
            self.bn_mode,
        )

        return parameters_aggregated, metrics_aggregated

    def get_summary(self) -> Dict[str, Any]:
        """Return run summary including full CLA v2 hyperparameter metadata."""
        summary = super().get_summary()
        summary["mu"] = self.mu
        summary["use_la"] = self.use_la
        summary["la_mode"] = self.la_mode
        summary["adaptive_kappa"] = self.adaptive_kappa
        summary["multi_linear_coverage"] = self.multi_linear_coverage
        summary["disable_covered_wd"] = self.disable_covered_wd
        summary["bn_mode"] = self.bn_mode
        summary["ablation_preset"] = self.ablation_preset
        # Legacy compatibility keys
        summary["coverage_kappa"] = self.legacy_kappa
        summary["coverage_mu_head"] = self.mu
        summary["la_tau"] = self.legacy_tau
        summary["la_epsilon"] = self.legacy_eps
        return summary


class CoverageLA_Strategy(CLAStrategy):
    """Backward-compatibility subclass of CLAStrategy with default strategy_name='CoverageLA'."""

    def __init__(self, *args, strategy_name: str = "CoverageLA", **kwargs):
        super().__init__(*args, strategy_name=strategy_name, **kwargs)


__all__ = [
    "CLAStrategy",
    "CoverageLA_Strategy",
    "MinimalistAdaCoverageLACriterion",
    "build_client_optimizer",
    "aggregate_models_with_pooled_bn",
    "get_linear_layers_in_order",
    "load_global_counts_from_partition",
    "ABLATION_PRESETS",
]
