"""FL Client entry point.

Implements a Flower NumPyClient that:
- Loads a local data partition (from a pre-generated Dirichlet JSON).
- Trains the CNN model locally with LR scheduling and early stopping.
- Reports extended metrics: per-epoch time, total training time, weight size.
- Supports FedProx proximal term when instructed by the server config.
- Supports FedSLD shared label distribution reweighted loss.

Usage:
    python -m client.client --client_id 0 --server_address 127.0.0.1:8080
    python -m client.client --client_id 5 --server_address 192.168.1.15:8080 --partition_path data/partitions/partition_seed42_alpha0.3_clients10.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

# Ensure stdout and stderr handle unicode characters safely across platforms/encodings
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import flwr as fl
from flwr.common import NDArrays, Scalar

from models import (
    get_model,
    set_seed,
    get_parameters,
    set_parameters,
    set_parameters_fedbn,
    is_bn_key,
)
from datasets.medmnist_code import get_medmnist_datasets
from datasets.partition import load_partition, get_client_dataloader, get_partition_path
from utils.path_utils import sanitize_run_name


from monitoring.resource import get_resource_usage, get_device_type
from monitoring.network import ClientPingLogger, get_device_ipv4


# ──────────────────────────────────────────────────────────────
# FedSOL Adversarial Proximal Perturbation Optimizer (CVPR 2024)
# ──────────────────────────────────────────────────────────────

class ExpSAM(torch.optim.Optimizer):
    """Adversarial Proximal Perturbation Optimizer for FedSOL (CVPR 2024).

    Wraps a base optimizer (SGD, Adam, AdamW) to perform:
    1. first_step(): Perturbs target parameters (e.g., classifier head or all layers)
                     along the proximal gradient direction.
    2. second_step(): Restores perturbed parameters and steps base optimizer across
                      all trainable parameters using local loss gradients.
    """

    def __init__(
        self,
        params,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 2.0,
        adaptive: bool = False,
        ref_params: Optional[List[torch.Tensor]] = None,
    ):
        target_list = list(params)
        defaults = dict(rho=float(rho), adaptive=bool(adaptive))
        super().__init__(target_list, defaults)
        self.base_optimizer = base_optimizer
        self.target_params = target_list
        self.ref_params = list(ref_params) if ref_params is not None else None

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        scale = self.defaults["rho"] / (grad_norm + 1e-12)
        adaptive = self.defaults.get("adaptive", False)

        for idx, p in enumerate(self.target_params):
            if p.grad is None:
                continue
            self.state[p]["old_p"] = p.data.clone()
            if adaptive and self.ref_params is not None and idx < len(self.ref_params):
                diff = torch.abs(p - self.ref_params[idx])
                diff_norm = torch.norm(diff, p=2) + 1e-12
                lambda_vec = diff / diff_norm
                e_w = lambda_vec * p.grad * scale.to(p)
            else:
                e_w = p.grad * scale.to(p)
            p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for p in self.target_params:
            if p.grad is None or "old_p" not in self.state[p]:
                continue
            p.data.copy_(self.state[p]["old_p"])

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()
            self.base_optimizer.zero_grad()

    def _grad_norm(self):
        adaptive = self.defaults.get("adaptive", False)
        grads = []
        for idx, p in enumerate(self.target_params):
            if p.grad is not None:
                if adaptive and self.ref_params is not None and idx < len(self.ref_params):
                    diff = torch.abs(p - self.ref_params[idx])
                    diff_norm = torch.norm(diff, p=2) + 1e-12
                    lambda_vec = diff / diff_norm
                    grads.append((lambda_vec * p.grad).norm(p=2))
                else:
                    grads.append(p.grad.norm(p=2))
        if not grads:
            return torch.tensor(0.0)
        shared_device = self.target_params[0].device
        return torch.norm(torch.stack([g.to(shared_device) for g in grads]), p=2)


def _extract_class_counts(loader, num_classes: int) -> dict[str, int]:
    """Extract class occurrence frequencies from a DataLoader."""
    counts = np.zeros(num_classes, dtype=np.int64)
    if hasattr(loader.dataset, "labels"):
        lbls = np.asarray(loader.dataset.labels).flatten()
        b = np.bincount(lbls, minlength=num_classes)
        for i in range(min(num_classes, len(b))):
            counts[i] = int(b[i])
    elif hasattr(loader.dataset, "targets"):
        lbls = np.asarray(loader.dataset.targets).flatten()
        b = np.bincount(lbls, minlength=num_classes)
        for i in range(min(num_classes, len(b))):
            counts[i] = int(b[i])
    else:
        for _, labels in loader:
            lbls = labels.view(-1).cpu().numpy()
            for l in lbls:
                if 0 <= l < num_classes:
                    counts[l] += 1
    return {str(c): int(counts[c]) for c in range(num_classes)}


# ──────────────────────────────────────────────────────────────
# Local training function (FL-aware version with proximal term)
# ──────────────────────────────────────────────────────────────

def _local_train(
    model: nn.Module,
    train_loader,
    optimizer,
    device: torch.device,
    epochs: int,
    scheduler=None,
    proximal_mu: float = 0.0,
    global_params: list = None,
    scaffold_drift: list = None,
    is_scaffold: bool = False,
    min_lr: float = 1e-6,
    early_stop_patience: int = 5,
    is_fedsld: bool = False,
    fedsld_prior: "torch.Tensor | None" = None,
    fedsld_epsilon: float = 1e-6,
    fedsld_reduction: str = "mean",
    is_fedsol: bool = False,
    fedsol_ref_model: "nn.Module | None" = None,
    fedsol_rho: float = 2.0,
    fedsol_target: str = "head",
    fedsol_loss_type: str = "kl",
    fedsol_temperature: float = 3.0,
    fedsol_adaptive: bool = False,
    clip_grad_norm: "float | None" = None,
):
    """Train locally and return rich metrics.

    Args:
        scheduler: Optional persistent ReduceLROnPlateau scheduler.
        proximal_mu: If > 0 and global_params provided, adds FedProx proximal term.
        global_params: Global model parameters (list of tensors) for proximal term.
        scaffold_drift: List of tensor drift terms (c - c_i) for trainable parameters in SCAFFOLD.
        is_scaffold: Whether SCAFFOLD is active to track uncorrupted loss gradients (Option 2).
        is_fedsld: Whether FedSLD batch reweighting is active.
        fedsld_prior: Global prior label distribution P~(y) tensor of shape (C,).
        fedsld_epsilon: Smoothing epsilon to prevent zero division in FedSLD.
        fedsld_reduction: Batch loss reduction ('mean' or 'sum').
        is_fedsol: Whether FedSOL adversarial proximal perturbation is active.
        fedsol_ref_model: Reference global model for FedSOL proximal discrepancy.
        fedsol_rho: Perturbation radius strength for FedSOL (CVPR 2024 default: 2.0).
        fedsol_target: Perturbation target layer ('head' = classifier head only, 'all' = full model).
        fedsol_loss_type: Proximal discrepancy metric ('kl' or 'l2').
        fedsol_temperature: Softmax temperature for KL divergence.
        fedsol_adaptive: Whether to use adaptive perturbation radius in FedSOL.
        clip_grad_norm: Maximum L2 gradient norm for clipping (None or <= 0 disables).
    """
    criterion = nn.CrossEntropyLoss()

    if is_fedsld and fedsld_prior is not None:
        fedsld_prior = fedsld_prior.to(device)

    best_loss = float("inf")
    patience_counter = 0
    epoch_times = []
    epoch_losses = []
    epoch_accuracies = []
    epoch_lrs = []
    total_samples = 0
    total_steps = 0

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    grad_accum = [torch.zeros_like(p) for p in trainable_params] if is_scaffold else None

    # Identify the final classifier layer for FedSOL partial perturbation
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m

    head_params = []
    if last_linear is not None:
        head_params = [p for p in last_linear.parameters() if p.requires_grad]

    ref_head_params = []
    if fedsol_ref_model is not None and last_linear is not None:
        ref_last_linear = None
        for m in fedsol_ref_model.modules():
            if isinstance(m, nn.Linear):
                ref_last_linear = m
        if ref_last_linear is not None:
            ref_head_params = [p for p in ref_last_linear.parameters()]

    # FedSOL optimizer wrapper (SAM-based adversarial proximal perturbation)
    sam_optimizer = None
    if is_fedsol and fedsol_ref_model is not None:
        if str(fedsol_target).lower() == "head" and head_params:
            target_p = head_params
            ref_p = ref_head_params if ref_head_params else [p for p in fedsol_ref_model.parameters()]
        else:
            target_p = trainable_params
            ref_p = [p for p in fedsol_ref_model.parameters()]

        sam_optimizer = ExpSAM(
            params=target_p,
            base_optimizer=optimizer,
            rho=fedsol_rho,
            adaptive=fedsol_adaptive,
            ref_params=ref_p,
        )

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            # ── Special Branch: FedSOL Two-Pass Adversarial Proximal Update ──
            if is_fedsol and fedsol_ref_model is not None and sam_optimizer is not None:
                # Pass 1: Measure discrepancy to global reference model and perturb
                optimizer.zero_grad()
                local_logits = model(images)
                with torch.no_grad():
                    ref_logits = fedsol_ref_model(images)

                if fedsol_loss_type == "kl":
                    T = max(fedsol_temperature, 1e-4)
                    pred_probs = F.log_softmax(local_logits / T, dim=1)
                    target_probs = F.softmax(ref_logits / T, dim=1)
                    prox_loss = F.kl_div(pred_probs, target_probs, reduction="batchmean") * (T * T)
                else:  # 'l2' parameter distance
                    prox_loss = 0.0
                    for lp, rp in zip(model.parameters(), fedsol_ref_model.parameters()):
                        prox_loss = prox_loss + 0.5 * ((lp - rp) ** 2).sum()

                prox_loss.backward()
                sam_optimizer.first_step(zero_grad=True)

                # Pass 2: Evaluate local task loss on perturbed weights & update
                optimizer.zero_grad()
                perturbed_logits = model(images)
                loss = criterion(perturbed_logits, labels)
                loss.backward()

                if clip_grad_norm is not None and clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_grad_norm)

                sam_optimizer.second_step(zero_grad=True)

                total_steps += 1
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(perturbed_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                continue

            optimizer.zero_grad()
            outputs = model(images)

            # Strategy-specific loss formulation
            if is_fedsld and fedsld_prior is not None:
                # FedSLD: reweight local mini-batch CE by p_b(y) / (P~(y) + eps)
                batch_counts = torch.bincount(labels, minlength=fedsld_prior.size(0)).float()
                pb = batch_counts / float(labels.size(0))
                weights = pb[labels] / (fedsld_prior[labels] + fedsld_epsilon)
                # Self-normalize weights to unit mean across batch to preserve standard CE gradient scale
                # and prevent exponential divergence under severe non-IID label skew
                weights = weights / (weights.mean() + 1e-8)
                ce = F.cross_entropy(outputs, labels, reduction="none")
                if fedsld_reduction == "sum":
                    loss = (weights * ce).sum()
                else:
                    loss = (weights * ce).mean()
            else:
                loss = criterion(outputs, labels)

            # FedProx proximal term
            if proximal_mu > 0.0 and global_params is not None:
                proximal_loss = 0.0
                for local_p, global_p in zip(
                    model.parameters(), global_params
                ):
                    proximal_loss += ((local_p - global_p) ** 2).sum()
                loss = loss + (proximal_mu / 2.0) * proximal_loss

            loss.backward()

            # SCAFFOLD Option 2: Accumulate uncorrupted loss gradient before adding drift
            if grad_accum is not None:
                for accum, p in zip(grad_accum, trainable_params):
                    if p.grad is not None:
                        accum.add_(p.grad)

            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_grad_norm)

            optimizer.step()

            # SCAFFOLD drift correction (NVFlare decoupled post-step):
            if scaffold_drift is not None:
                step_lr = float(optimizer.param_groups[0]["lr"])
                with torch.no_grad():
                    for p, drift in zip(trainable_params, scaffold_drift):
                        p.data.add_(drift, alpha=-step_lr)
            total_steps += 1

            running_loss += loss.item() * images.size(0)
            # Accuracy always computed from raw logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_times.append(epoch_time)
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)
        epoch_lrs.append(current_lr)
        total_samples = total

        print(
            f"  [Client] Epoch [{epoch+1}/{epochs}] "
            f"LR={current_lr:.6f} | Loss={epoch_loss:.4f} Acc={epoch_acc:.4f} "
            f"| {epoch_time:.2f}s"
        )

        # LR scheduling on client's own train loss
        old_lr = current_lr
        if scheduler is not None:
            scheduler.step(epoch_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"    -> [Client LR Reducer] Loss plateaued: LR reduced {old_lr:.6f} -> {new_lr:.6f}")

        # Early stopping on train loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(
                    f"    -> Early stopping at epoch {epoch+1} "
                    f"(no improvement for {early_stop_patience} epochs)"
                )
                break

    metrics_out = {
        "train_loss": epoch_loss,
        "train_accuracy": epoch_acc,
        "epoch_times": epoch_times,
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "epoch_lrs": epoch_lrs,
        "epoch_time_avg": float(np.mean(epoch_times)),
        "epoch_time_min": float(np.min(epoch_times)),
        "epoch_time_max": float(np.max(epoch_times)),
        "num_samples": total_samples,
        "epochs_run": len(epoch_times),
        "total_steps": total_steps,
    }
    if grad_accum is not None:
        metrics_out["grad_accum"] = [
            (accum / max(1, total_steps)).cpu().numpy().astype(np.float64)
            for accum in grad_accum
        ]
    return metrics_out


# ──────────────────────────────────────────────────────────────
# Checkpoint & Resume Helpers
# ──────────────────────────────────────────────────────────────

def _canonical_strategy_name(name: Optional[str]) -> str:
    """Normalize strategy aliases to a canonical lowercase identifier."""
    s = str(name or "").lower().strip()
    if s in ("proposed", "fedbns", "fedbn_scaffold", "bn_scaffold", "scaffold_fedbn"):
        return "fedbns"
    if s in ("fedsld", "fed_sld"):
        return "fedsld"
    if s in ("fedsol", "fed_sol"):
        return "fedsol"
    if s in ("fedswitch", "fed_switch"):
        return "fedswitch"
    return s


def _clone_value_to_cpu(val: Any) -> Any:
    """Recursively detach and clone tensor values to CPU for device-agnostic checkpoints."""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().clone()
    if isinstance(val, dict):
        return {k: _clone_value_to_cpu(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_clone_value_to_cpu(v) for v in val]
    if isinstance(val, tuple):
        return tuple(_clone_value_to_cpu(v) for v in val)
    return val


def _clone_optimizer_state_to_cpu(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """Return a CPU-cloned copy of an optimizer's state_dict."""
    return _clone_value_to_cpu(optimizer.state_dict())


def _move_value_to_device(val: Any, device: torch.device) -> Any:
    """Recursively move tensor values to the target device."""
    if isinstance(val, torch.Tensor):
        return val.to(device)
    if isinstance(val, dict):
        return {k: _move_value_to_device(v, device) for k, v in val.items()}
    if isinstance(val, list):
        return [_move_value_to_device(v, device) for v in val]
    if isinstance(val, tuple):
        return tuple(_move_value_to_device(v, device) for v in val)
    return val


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move all tensor buffers inside optimizer.state to `device`."""
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for k, v in list(state.items()):
                state[k] = _move_value_to_device(v, device)


def _build_optimizer(
    optimizer_name: str,
    model_params,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Instantiate the configured local PyTorch optimizer."""
    opt_key = str(optimizer_name).lower().strip()
    if opt_key == "sgd":
        return optim.SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if opt_key == "adamw":
        return optim.AdamW(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
        )
    return optim.Adam(
        model_params,
        lr=lr,
        weight_decay=weight_decay,
    )


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Write a torch checkpoint atomically via a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


# ──────────────────────────────────────────────────────────────
# Flower NumPyClient
# ──────────────────────────────────────────────────────────────


class FedMedAIClient(fl.client.NumPyClient):
    """Flower client for FedMedAI federated learning."""

    def __init__(
        self,
        client_id: int,
        train_loader,
        test_loader,
        num_classes: int = 8,
        local_epochs: int = 5,
        learning_rate: float = 0.001,
        lr_mode: str = "client_loss",
        lr_patience: int = 2,
        lr_factor: float = 0.5,
        lr_min: float = 1e-6,
        device_type: str = None,
        save_local_metrics: bool = True,
        server_address: str = "127.0.0.1:8080",
        strategy: str = None,
        partition_path: Optional[str] = None,
        client_dir: Optional[Union[str, Path]] = None,
        model_name: str = "baseline_bn",
        optimizer_name: str = "adam",
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        name: Optional[str] = None,
        fedsol_rho: float = 2.0,
        fedsol_target: str = "head",
        fedsol_loss_type: str = "kl",
        fedsol_temperature: float = 3.0,
        fedsol_adaptive: bool = False,
        clip_grad_norm: Optional[float] = None,
        auto_resume: bool = True,
    ):
        self.client_id = client_id
        self.model_name = model_name
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.optimizer_name = str(optimizer_name).lower().strip() if optimizer_name else "adam"
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.lr_mode = str(lr_mode).lower().strip() if lr_mode else "client_loss"
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self._lr_initialized = False
        self.device_type = str(device_type).strip() if device_type else get_device_type(client_id)
        self.save_local_metrics = save_local_metrics
        self.server_address = server_address
        self.strategy = strategy.lower() if strategy else None
        self.partition_path = str(partition_path) if partition_path else None
        self.client_control_variate: Optional[List[np.ndarray]] = None

        self.fedsol_rho = float(fedsol_rho)
        self.fedsol_target = str(fedsol_target).lower().strip()
        self.fedsol_loss_type = str(fedsol_loss_type).lower().strip()
        self.fedsol_temperature = float(fedsol_temperature)
        self.fedsol_adaptive = bool(fedsol_adaptive)
        self.clip_grad_norm = float(clip_grad_norm) if clip_grad_norm is not None and clip_grad_norm > 0 else None

        self._class_counts_dict = _extract_class_counts(self.train_loader, self.num_classes)
        self._class_counts = torch.tensor(
            [self._class_counts_dict[str(c)] for c in range(self.num_classes)],
            dtype=torch.float32,
        )

        self.round_history = []
        self.epoch_history = []
        self.eval_history = []
        self.best_eval_accuracy: float = -1.0
        self.best_eval_round: Optional[int] = None
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.auto_resume: bool = bool(auto_resume)
        self._session_initialized: bool = False
        self._resumed_from_checkpoint: bool = False

        # Client ping logger
        if client_dir is not None:
            self.client_dir = Path(client_dir)
        elif name:
            safe_name = sanitize_run_name(name)
            self.client_dir = Path("results") / "clients" / safe_name / f"client_{self.client_id}"
        else:
            self.client_dir = Path("results") / "clients" / f"client_{self.client_id}"
        self.ping_logger = ClientPingLogger(
            client_id=self.client_id,
            server_address=self.server_address,
            device_type=self.device_type,
            log_dir=self.client_dir,
            enabled=self.save_local_metrics,
        )
        self.ping_logger.log_ping(server_round=0, event="init")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = get_model(model_name, num_classes=num_classes).to(self.device)

        self.optimizer = _build_optimizer(
            self.optimizer_name,
            self.model.parameters(),
            self.learning_rate,
            self.momentum,
            self.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min,
        )

        self.device_ipv4 = get_device_ipv4()

        print(
            f"[Client {self.client_id}] Initialized on {self.device} ({self.device_type}) | "
            f"IPv4: {self.device_ipv4} | train_samples={len(train_loader.dataset)} | "
            f"Optimizer: {self.optimizer_name.upper()} | LR Mode: {self.lr_mode} (Initial LR: {self.learning_rate})"
        )

    def _upsert_round_history(self, entry: Dict[str, Any]) -> None:
        """Insert or replace a round_history record keyed by server_round, sorted ascending."""
        r_num = int(entry.get("server_round", 0))
        by_round = {int(item.get("server_round", 0)): item for item in self.round_history}
        by_round[r_num] = entry
        self.round_history = [by_round[r] for r in sorted(by_round.keys())]

    def _upsert_eval_history(self, entry: Dict[str, Any]) -> None:
        """Insert or replace an eval_history record keyed by server_round, sorted ascending."""
        r_num = int(entry.get("server_round", 0))
        by_round = {int(item.get("server_round", 0)): item for item in self.eval_history}
        by_round[r_num] = entry
        self.eval_history = [by_round[r] for r in sorted(by_round.keys())]

    def _replace_round_epoch_history(self, server_round: int, entries: List[Dict[str, Any]]) -> None:
        """Replace epoch_history rows for `server_round` without duplicating prior rounds."""
        target_round = int(server_round)
        kept = [row for row in self.epoch_history if int(row.get("round", -1)) != target_round]
        kept.extend(entries)
        self.epoch_history = kept

    def _emit_resume_history(self, resumed_round: int, source_label: str) -> None:
        """Emit structured resume history marker so the dashboard restores prior curves."""
        compact_rounds = []
        for r in self.round_history:
            if r.get("server_round") is None:
                continue
            item: Dict[str, Any] = {
                "server_round": int(r["server_round"]),
            }
            if r.get("train_loss") is not None:
                item["train_loss"] = float(r["train_loss"])
            if r.get("train_accuracy") is not None:
                item["train_accuracy"] = float(r["train_accuracy"])
            rt = r.get("round_time", r.get("training_time"))
            if rt is not None:
                item["round_time"] = float(rt)
            if r.get("ping_ms") is not None:
                item["ping_ms"] = float(r["ping_ms"])
            if r.get("num_samples") is not None:
                item["num_samples"] = int(r["num_samples"])
            compact_rounds.append(item)

        compact_evals = []
        for e in self.eval_history:
            if e.get("server_round") is None:
                continue
            ev_item: Dict[str, Any] = {
                "server_round": int(e["server_round"]),
            }
            if e.get("eval_loss") is not None:
                ev_item["eval_loss"] = float(e["eval_loss"])
            if e.get("eval_accuracy") is not None:
                ev_item["eval_accuracy"] = float(e["eval_accuracy"])
            compact_evals.append(ev_item)

        pkg = {
            "client_id": int(self.client_id),
            "resumed_round": int(resumed_round),
            "source": str(source_label),
            "round_history": compact_rounds,
            "eval_history": compact_evals,
        }
        print(f"[CLIENT_RESUME_HISTORY] {json.dumps(pkg)}", flush=True)

    def _load_csv_histories_fallback(self, incoming_round: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load prior round and eval histories from local CSVs up to `incoming_round`."""
        import csv

        loaded_rounds: Dict[int, Dict[str, Any]] = {}
        r_path = self.client_dir / f"client_{self.client_id}_round_metrics.csv"
        if r_path.exists():
            try:
                with open(r_path, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        try:
                            sr = int(float(row["server_round"]))
                        except (KeyError, TypeError, ValueError):
                            continue
                        if not (0 < sr <= incoming_round):
                            continue
                        parsed: Dict[str, Any] = {"server_round": sr, "client_id": int(self.client_id)}
                        if row.get("device_type"):
                            parsed["device_type"] = row["device_type"]
                        for int_key in ("num_samples", "local_epochs"):
                            if row.get(int_key) not in (None, ""):
                                try:
                                    parsed[int_key] = int(float(row[int_key]))
                                except ValueError:
                                    pass
                        for flt_key in (
                            "learning_rate", "training_time", "epoch_time_avg", "epoch_time_min",
                            "epoch_time_max", "train_loss", "train_accuracy", "ping_ms",
                            "weight_size_kb", "cpu_percent", "ram_percent", "ram_used_mb", "gpu_memory_mb",
                        ):
                            if row.get(flt_key) not in (None, ""):
                                try:
                                    parsed[flt_key] = float(row[flt_key])
                                except ValueError:
                                    pass
                        loaded_rounds[sr] = parsed
            except Exception as exc:
                print(f"[Client {self.client_id}] Warning: Could not read {r_path.name} for resume: {exc}")

        loaded_evals: Dict[int, Dict[str, Any]] = {}
        ev_path = self.client_dir / f"client_{self.client_id}_eval_metrics.csv"
        if ev_path.exists():
            try:
                with open(ev_path, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        try:
                            sr = int(float(row["server_round"]))
                        except (KeyError, TypeError, ValueError):
                            continue
                        if not (0 < sr <= incoming_round):
                            continue
                        parsed_ev: Dict[str, Any] = {"server_round": sr}
                        for flt_key in ("eval_loss", "eval_accuracy"):
                            if row.get(flt_key) not in (None, ""):
                                try:
                                    parsed_ev[flt_key] = float(row[flt_key])
                                except ValueError:
                                    pass
                        if "eval_loss" in parsed_ev or "eval_accuracy" in parsed_ev:
                            loaded_evals[sr] = parsed_ev
            except Exception as exc:
                print(f"[Client {self.client_id}] Warning: Could not read {ev_path.name} for resume: {exc}")

        return (
            [loaded_rounds[r] for r in sorted(loaded_rounds.keys())],
            [loaded_evals[r] for r in sorted(loaded_evals.keys())],
        )

    def _save_latest_state(self, server_round: int, strategy: Optional[str] = None) -> Optional[Path]:
        """Persist latest local model weights (including BN layers) and full resume state atomically."""
        if not self.save_local_metrics:
            return None
        try:
            self.client_dir.mkdir(parents=True, exist_ok=True)
            model_sd = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            state_payload = {
                "schema_version": 1,
                "client_id": int(self.client_id),
                "model_name": str(self.model_name),
                "num_classes": int(self.num_classes),
                "strategy": _canonical_strategy_name(strategy or self.strategy),
                "optimizer_name": str(self.optimizer_name),
                "momentum": float(self.momentum),
                "weight_decay": float(self.weight_decay),
                "server_address": str(self.server_address),
                "partition_path": str(self.partition_path or ""),
                "last_server_round": int(server_round),
                "model_state_dict": model_sd,
                "optimizer_state_dict": _clone_optimizer_state_to_cpu(self.optimizer),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "lr_initialized": bool(self._lr_initialized),
                "lr_patience": int(self.lr_patience),
                "lr_factor": float(self.lr_factor),
                "lr_min": float(self.lr_min),
                "client_control_variate": (
                    [np.array(arr, copy=True) for arr in self.client_control_variate]
                    if self.client_control_variate is not None
                    else None
                ),
                "best_eval_accuracy": float(self.best_eval_accuracy),
                "best_eval_round": self.best_eval_round,
                "round_history": list(self.round_history),
                "epoch_history": list(self.epoch_history),
                "eval_history": list(self.eval_history),
            }
            state_path = self.client_dir / "latest_state.pt"
            latest_model_path = self.client_dir / "latest_model.pth"
            _atomic_torch_save(state_payload, state_path)
            _atomic_torch_save(model_sd, latest_model_path)
            return state_path
        except Exception as exc:
            print(f"[Client {self.client_id}] Warning: Could not save latest_state.pt: {exc}")
            return None

    def _try_resume_from_disk(
        self,
        incoming_round: int,
        incoming_strategy: str,
        config: Optional[dict] = None,
    ) -> bool:
        """Attempt to restore local model (with BN layers), optimizer, variates, and history from disk."""
        state_path = self.client_dir / "latest_state.pt"
        canon_incoming = _canonical_strategy_name(incoming_strategy or self.strategy)

        if state_path.exists():
            try:
                ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
                ckpt_cid = int(ckpt.get("client_id", -1))
                ckpt_model = str(ckpt.get("model_name", ""))
                ckpt_classes = int(ckpt.get("num_classes", -1))
                ckpt_strategy = _canonical_strategy_name(ckpt.get("strategy"))
                ckpt_part = str(ckpt.get("partition_path", "") or "")
                ckpt_round = int(ckpt.get("last_server_round", 0))

                if ckpt_cid != int(self.client_id):
                    print(f"[Client {self.client_id}] Resume skipped: client_id mismatch ({ckpt_cid} != {self.client_id}).")
                    return False
                if ckpt_model != str(self.model_name) or ckpt_classes != int(self.num_classes):
                    print(
                        f"[Client {self.client_id}] Resume skipped: model mismatch "
                        f"({ckpt_model}/{ckpt_classes} != {self.model_name}/{self.num_classes})."
                    )
                    return False
                if ckpt_strategy and canon_incoming and ckpt_strategy != canon_incoming:
                    print(
                        f"[Client {self.client_id}] Resume skipped: strategy mismatch "
                        f"({ckpt_strategy} != {canon_incoming})."
                    )
                    return False
                if ckpt_part and self.partition_path and Path(ckpt_part).name != Path(self.partition_path).name:
                    print(
                        f"[Client {self.client_id}] Resume skipped: partition mismatch "
                        f"({Path(ckpt_part).name} != {Path(self.partition_path).name})."
                    )
                    return False
                if not (0 < ckpt_round <= int(incoming_round)):
                    print(
                        f"[Client {self.client_id}] Resume skipped: checkpoint round {ckpt_round} "
                        f"incompatible with incoming round {incoming_round}."
                    )
                    return False

                model_sd = ckpt.get("model_state_dict")
                if not isinstance(model_sd, dict):
                    print(f"[Client {self.client_id}] Resume skipped: missing model_state_dict in {state_path.name}.")
                    return False

                self.model.load_state_dict(model_sd, strict=True)
                self.model.to(self.device)

                # Restore SCAFFOLD client control variate if shapes match
                ckpt_cv = ckpt.get("client_control_variate")
                if ckpt_cv is not None:
                    expected_shapes = [p.shape for p in get_parameters(self.model)]
                    if len(ckpt_cv) == len(expected_shapes) and all(
                        np.asarray(a).shape == s for a, s in zip(ckpt_cv, expected_shapes)
                    ):
                        self.client_control_variate = [np.array(a, copy=True) for a in ckpt_cv]

                # Restore LR / scheduler settings and optimizer state
                if ckpt.get("lr_patience") is not None:
                    self.lr_patience = int(ckpt["lr_patience"])
                if ckpt.get("lr_factor") is not None:
                    self.lr_factor = float(ckpt["lr_factor"])
                if ckpt.get("lr_min") is not None:
                    self.lr_min = float(ckpt["lr_min"])
                if ckpt.get("learning_rate") is not None:
                    self.learning_rate = float(ckpt["learning_rate"])
                if ckpt.get("lr_initialized") is not None:
                    self._lr_initialized = bool(ckpt["lr_initialized"])

                if config and "optimizer" in config:
                    srv_opt = str(config["optimizer"]).lower().strip()
                    if srv_opt != self.optimizer_name:
                        self.optimizer_name = srv_opt
                        if "momentum" in config:
                            self.momentum = float(config["momentum"])
                        self.optimizer = _build_optimizer(
                            self.optimizer_name,
                            self.model.parameters(),
                            self.learning_rate,
                            self.momentum,
                            self.weight_decay,
                        )

                ckpt_opt_name = str(ckpt.get("optimizer_name", "")).lower().strip()
                ckpt_opt_sd = ckpt.get("optimizer_state_dict")
                if ckpt_opt_name == self.optimizer_name and isinstance(ckpt_opt_sd, dict):
                    try:
                        self.optimizer.load_state_dict(ckpt_opt_sd)
                        _move_optimizer_state_to_device(self.optimizer, self.device)
                        self.learning_rate = float(self.optimizer.param_groups[0]["lr"])
                    except Exception as opt_exc:
                        print(f"[Client {self.client_id}] Warning: Could not restore optimizer state: {opt_exc}")

                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    min_lr=self.lr_min,
                )
                ckpt_sched_sd = ckpt.get("scheduler_state_dict")
                if isinstance(ckpt_sched_sd, dict):
                    try:
                        self.scheduler.load_state_dict(ckpt_sched_sd)
                    except Exception as sched_exc:
                        print(f"[Client {self.client_id}] Warning: Could not restore scheduler state: {sched_exc}")

                # Restore best metric tracking & best_model_state
                if ckpt.get("best_eval_accuracy") is not None:
                    self.best_eval_accuracy = float(ckpt["best_eval_accuracy"])
                self.best_eval_round = ckpt.get("best_eval_round")
                best_pth = self.client_dir / "best_model.pth"
                if best_pth.exists():
                    try:
                        best_sd = torch.load(best_pth, map_location="cpu", weights_only=False)
                        if isinstance(best_sd, dict):
                            self.best_model_state = best_sd
                    except Exception:
                        pass

                # Restore histories up to incoming_round
                self.round_history = [
                    r for r in (ckpt.get("round_history") or [])
                    if int(r.get("server_round", 0)) <= int(incoming_round)
                ]
                self.epoch_history = [
                    ep for ep in (ckpt.get("epoch_history") or [])
                    if int(ep.get("round", 0)) <= int(incoming_round)
                ]
                self.eval_history = [
                    ev for ev in (ckpt.get("eval_history") or [])
                    if int(ev.get("server_round", 0)) <= int(incoming_round)
                ]

                self._resumed_from_checkpoint = True
                print(
                    f"[Client {self.client_id}] Resumed latest local state from {state_path.name} "
                    f"(Round {ckpt_round}, best_acc={max(0.0, self.best_eval_accuracy) * 100:.2f}% @ "
                    f"R{self.best_eval_round or '-'}, rounds={len(self.round_history)}, evals={len(self.eval_history)})"
                )
                self._emit_resume_history(resumed_round=ckpt_round, source_label="latest_state.pt")
                return True
            except Exception as exc:
                print(f"[Client {self.client_id}] Warning: Failed to load {state_path.name} for resume: {exc}")
                return False

        # Fallback path when latest_state.pt does not exist: legacy model .pth + local CSVs
        csv_rounds, csv_evals = self._load_csv_histories_fallback(int(incoming_round))
        max_csv_round = max(
            [int(r["server_round"]) for r in csv_rounds]
            + [int(e["server_round"]) for e in csv_evals]
            + [0]
        )
        if not (0 < max_csv_round <= int(incoming_round)):
            return False

        for fname in ("latest_model.pth", "final_model.pth", "best_model.pth"):
            cand_path = self.client_dir / fname
            if not cand_path.exists():
                continue
            try:
                sd = torch.load(cand_path, map_location="cpu", weights_only=False)
                if not isinstance(sd, dict):
                    continue
                self.model.load_state_dict(sd, strict=True)
                self.model.to(self.device)
                self.round_history = csv_rounds
                self.eval_history = csv_evals
                for ev in csv_evals:
                    acc_val = ev.get("eval_accuracy")
                    if acc_val is not None and float(acc_val) > self.best_eval_accuracy:
                        self.best_eval_accuracy = float(acc_val)
                        self.best_eval_round = int(ev["server_round"])
                best_pth = self.client_dir / "best_model.pth"
                if best_pth.exists():
                    try:
                        best_sd = torch.load(best_pth, map_location="cpu", weights_only=False)
                        if isinstance(best_sd, dict):
                            self.best_model_state = best_sd
                    except Exception:
                        pass
                self._resumed_from_checkpoint = True
                print(
                    f"[Client {self.client_id}] Resumed fallback local weights from {fname} + CSVs "
                    f"(Round {max_csv_round}, rounds={len(self.round_history)}, evals={len(self.eval_history)})"
                )
                self._emit_resume_history(resumed_round=max_csv_round, source_label=fname)
                return True
            except Exception as exc:
                print(f"[Client {self.client_id}] Warning: Fallback resume from {fname} failed: {exc}")
        return False

    def _ensure_session_initialized(
        self,
        server_round: int,
        strategy: str,
        config: Optional[dict] = None,
        phase: str = "fit",
    ) -> None:
        """Initialize session once on first server message, resuming if reconnecting mid-run."""
        if self._session_initialized:
            return
        self._session_initialized = True

        rnd = int(server_round)
        should_try_resume = self.auto_resume and (rnd > 1 or (phase == "evaluate" and rnd >= 1))
        if should_try_resume:
            resumed = self._try_resume_from_disk(rnd, strategy, config)
            if not resumed:
                print(
                    f"[CLIENT_SESSION_START] {json.dumps({'client_id': int(self.client_id), 'server_round': rnd, 'resumed': False})}",
                    flush=True,
                )
        else:
            if ((phase == "fit" and rnd <= 1) or rnd <= 0) and self.save_local_metrics:
                for stale_name in ("latest_state.pt", "latest_model.pth"):
                    stale_path = self.client_dir / stale_name
                    if stale_path.exists():
                        try:
                            stale_path.unlink()
                        except Exception:
                            pass
            print(
                f"[CLIENT_SESSION_START] {json.dumps({'client_id': int(self.client_id), 'server_round': rnd, 'resumed': False})}",
                flush=True,
            )

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as numpy arrays."""
        return get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: dict):
        """Receive global model, train locally, return updated weights + metrics."""
        strategy = str(config.get("strategy", self.strategy or "fedavg")).lower()
        server_round = int(config.get("server_round", len(self.round_history) + 1))
        self._ensure_session_initialized(server_round=server_round, strategy=strategy, config=config, phase="fit")

        num_model_layers = len(get_parameters(self.model))
        fedbn_strategies = (
            "fedbn", "proposed", "fedbns", "fedbn_scaffold", "bn_scaffold", "scaffold_fedbn",
        )
        scaffold_strategies = (
            "scaffold", "proposed", "fedbns", "fedbn_scaffold", "bn_scaffold", "scaffold_fedbn",
        )
        is_scaffold = (strategy in scaffold_strategies) or (len(parameters) == 2 * num_model_layers)
        is_fedbn = strategy in fedbn_strategies

        initial_weights = None
        server_c = None
        scaffold_drift = None

        if is_scaffold:
            if len(parameters) == 2 * num_model_layers:
                initial_weights = parameters[:num_model_layers]
                server_c = parameters[num_model_layers:]
            else:
                initial_weights = parameters
                server_c = [np.zeros_like(p) for p in initial_weights]

            if is_fedbn:
                set_parameters_fedbn(self.model, initial_weights)
            else:
                set_parameters(self.model, initial_weights)

            if self.client_control_variate is None:
                self.client_control_variate = [np.zeros_like(p) for p in initial_weights]

            # Construct drift correction terms (server_c - client_c) for trainable parameters
            sd_keys = list(self.model.state_dict().keys())
            named_params = dict(self.model.named_parameters())
            scaffold_drift = [
                torch.as_tensor(s - c, device=self.device, dtype=named_params[k].dtype)
                if (not is_fedbn or not is_bn_key(self.model, k))
                else torch.zeros_like(named_params[k])
                for k, s, c in zip(sd_keys, server_c, self.client_control_variate)
                if k in named_params
            ]
        elif is_fedbn:
            set_parameters_fedbn(self.model, parameters)
        else:
            set_parameters(self.model, parameters)

        # Read config from server
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        proximal_mu = float(config.get("proximal_mu", 0.0))
        lr_mode = str(config.get("lr_mode", self.lr_mode)).lower().strip()

        if "optimizer" in config:
            srv_opt = str(config["optimizer"]).lower().strip()
            if srv_opt != self.optimizer_name:
                self.optimizer_name = srv_opt
                if "momentum" in config:
                    self.momentum = float(config["momentum"])
                self.optimizer = _build_optimizer(
                    self.optimizer_name,
                    self.model.parameters(),
                    self.learning_rate,
                    self.momentum,
                    self.weight_decay,
                )
                print(f"[Client {self.client_id}] Switched local optimizer to {self.optimizer_name.upper()} based on server config")

        active_scheduler = None

        if lr_mode == "fixed":
            # Fixed constant LR mode
            fixed_lr = float(config.get("learning_rate", self.learning_rate))
            for pg in self.optimizer.param_groups:
                pg["lr"] = fixed_lr
            self.learning_rate = fixed_lr
            active_scheduler = None
            print(f"[Client {self.client_id}] Round {server_round}: Fixed LR = {fixed_lr:.6f}")

        elif lr_mode in ("server_decay", "server", "step"):
            # Server-synchronized global step decay
            scheduled_lr = float(config.get("learning_rate", self.learning_rate))
            for pg in self.optimizer.param_groups:
                pg["lr"] = scheduled_lr
            self.learning_rate = scheduled_lr
            active_scheduler = None
            print(f"[Client {self.client_id}] Round {server_round}: Server decayed LR = {scheduled_lr:.6f}")

        else:
            # 'client_loss' mode
            if server_round == 1 or not self._lr_initialized:
                self._lr_initialized = True
                init_lr = config.get("learning_rate")
                if init_lr is None:
                    init_lr = config.get("initial_lr", self.learning_rate)
                init_lr = float(init_lr)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = init_lr
                self.learning_rate = init_lr

                if "client_lr_patience" in config:
                    self.lr_patience = int(config["client_lr_patience"])
                if "client_lr_factor" in config:
                    self.lr_factor = float(config["client_lr_factor"])
                if "client_lr_min" in config:
                    self.lr_min = float(config["client_lr_min"])

                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    min_lr=self.lr_min,
                )
                print(
                    f"[Client {self.client_id}] Training start (Round {server_round}): Initial LR = {init_lr:.6f} "
                    f"(local reducer: patience={self.lr_patience}, factor={self.lr_factor})"
                )
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"[Client {self.client_id}] Round {server_round}: Continuing with local LR = {current_lr:.6f} "
                    f"(managed independently by local loss)"
                )
            active_scheduler = self.scheduler

        # Measure network ping to server for this round
        ping_res = self.ping_logger.log_ping(server_round=server_round, event="fit")
        ping_ms = ping_res.get("ping_ms")

        # Prepare global params for FedProx proximal term
        global_params = None
        if proximal_mu > 0.0:
            global_params = [
                p.clone().detach() for p in self.model.parameters()
            ]

        # Prepare FedSLD parameters
        fedsld_prior_tensor = None
        is_fedsld = strategy in ("fedsld", "fed_sld")
        fedsld_epsilon = float(config.get("fedsld_epsilon", 1e-6))
        fedsld_reduction = str(config.get("fedsld_reduction", "mean")).lower()

        if is_fedsld:
            if "fedsld_prior" in config:
                prior_list = json.loads(config["fedsld_prior"])
                fedsld_prior_tensor = torch.tensor(prior_list, dtype=torch.float32, device=self.device)
            elif hasattr(self, "partition_path") and self.partition_path and Path(self.partition_path).exists():
                from algorithms.fedsld import load_prior_from_partition
                prior_list = load_prior_from_partition(self.partition_path, num_classes=self.num_classes)
                fedsld_prior_tensor = torch.tensor(prior_list, dtype=torch.float32, device=self.device)
            else:
                fedsld_prior_tensor = torch.ones(self.num_classes, dtype=torch.float32, device=self.device) / float(self.num_classes)

            print(
                f"[Client {self.client_id}] FedSLD: eps={fedsld_epsilon} reduction={fedsld_reduction} "
                f"| prior={[round(p, 4) for p in fedsld_prior_tensor.tolist()]}"
            )

        # Prepare FedSOL parameters
        is_fedsol = strategy in ("fedsol", "fed_sol")
        fedsol_rho = float(config.get("fedsol_rho", self.fedsol_rho))
        fedsol_target = str(config.get("fedsol_target", getattr(self, "fedsol_target", "head"))).lower().strip()
        fedsol_loss_type = str(config.get("fedsol_loss_type", self.fedsol_loss_type)).lower()
        fedsol_temperature = float(config.get("fedsol_temperature", self.fedsol_temperature))
        fedsol_adaptive = bool(config.get("fedsol_adaptive", self.fedsol_adaptive))
        clip_grad_norm_val = config.get("clip_grad_norm", self.clip_grad_norm)
        if clip_grad_norm_val is not None:
            clip_grad_norm_val = float(clip_grad_norm_val) if float(clip_grad_norm_val) > 0 else None

        fedsol_ref_model = None
        if is_fedsol:
            import copy
            fedsol_ref_model = copy.deepcopy(self.model)
            fedsol_ref_model.eval()
            for p in fedsol_ref_model.parameters():
                p.requires_grad = False
            print(
                f"[Client {self.client_id}] FedSOL: rho={fedsol_rho} target={fedsol_target} "
                f"loss_type={fedsol_loss_type} T={fedsol_temperature} adaptive={fedsol_adaptive}"
            )

        # Train locally
        train_start = time.time()
        train_metrics = _local_train(
            model=self.model,
            train_loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            epochs=local_epochs,
            scheduler=active_scheduler,
            proximal_mu=proximal_mu,
            global_params=global_params,
            scaffold_drift=scaffold_drift,
            is_scaffold=is_scaffold,
            is_fedsld=is_fedsld,
            fedsld_prior=fedsld_prior_tensor,
            fedsld_epsilon=fedsld_epsilon,
            fedsld_reduction=fedsld_reduction,
            is_fedsol=is_fedsol,
            fedsol_ref_model=fedsol_ref_model,
            fedsol_rho=fedsol_rho,
            fedsol_target=fedsol_target,
            fedsol_loss_type=fedsol_loss_type,
            fedsol_temperature=fedsol_temperature,
            fedsol_adaptive=fedsol_adaptive,
            clip_grad_norm=clip_grad_norm_val,
        )
        training_time = time.time() - train_start

        # Compute updated parameters
        updated_model_params = get_parameters(self.model)

        if is_scaffold:
            K = max(int(train_metrics.get("total_steps", 1)), 1)
            sd_keys = list(self.model.state_dict().keys())
            named_params = dict(self.model.named_parameters())
            grad_avg = train_metrics.get("grad_accum")

            delta_c = []
            new_client_c = []
            trainable_idx = 0
            for idx, k in enumerate(sd_keys):
                if k in named_params:
                    is_bn = is_fedbn and is_bn_key(self.model, k)
                    if not is_bn:
                        if grad_avg is not None and trainable_idx < len(grad_avg):
                            c_i_new = grad_avg[trainable_idx]
                            d_c = c_i_new - self.client_control_variate[idx]
                        else:
                            lr = float(self.optimizer.param_groups[0]["lr"])
                            d_c = (1.0 / (K * lr)) * (initial_weights[idx] - updated_model_params[idx]) - server_c[idx]
                            c_i_new = self.client_control_variate[idx] + d_c
                    else:
                        d_c = np.zeros_like(updated_model_params[idx], dtype=np.float64)
                        c_i_new = np.zeros_like(updated_model_params[idx], dtype=np.float64)
                    trainable_idx += 1
                else:
                    d_c = np.zeros_like(updated_model_params[idx], dtype=np.float64)
                    c_i_new = np.zeros_like(updated_model_params[idx], dtype=np.float64)
                delta_c.append(d_c)
                new_client_c.append(c_i_new)

            self.client_control_variate = new_client_c
            return_parameters = updated_model_params + delta_c
        else:
            return_parameters = updated_model_params

        # Sample hardware resources
        res = get_resource_usage()

        # Compute weight size in KB
        weight_size_bytes = sum(p.nbytes for p in return_parameters)
        weight_size_kb = weight_size_bytes / 1024.0

        # Build metrics dict
        metrics = {
            "client_id": int(self.client_id),
            "device_type": str(self.device_type),
            "device_ipv4": str(self.device_ipv4),
            "server_round": int(server_round),
            "num_samples": int(train_metrics["num_samples"]),
            "local_epochs": int(local_epochs),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["train_loss"]),
            "train_accuracy": float(train_metrics["train_accuracy"]),
            "training_time": float(training_time),
            "epoch_time_avg": float(train_metrics["epoch_time_avg"]),
            "epoch_time_min": float(train_metrics["epoch_time_min"]),
            "epoch_time_max": float(train_metrics["epoch_time_max"]),
            "epoch_times": json.dumps([round(t, 4) for t in train_metrics["epoch_times"]]),
            "epoch_losses": json.dumps([round(l, 4) for l in train_metrics["epoch_losses"]]),
            "epoch_accuracies": json.dumps([round(a, 4) for a in train_metrics["epoch_accuracies"]]),
            "epoch_lrs": json.dumps([round(lr, 6) for lr in train_metrics["epoch_lrs"]]),
            "epochs_run": int(train_metrics["epochs_run"]),
            "local_steps": int(train_metrics.get("total_steps", train_metrics["epochs_run"])),
            "ping_ms": float(ping_ms) if ping_ms is not None else -1.0,
            "weight_size_kb": float(weight_size_kb),
            "cpu_percent": float(res["cpu_percent"]),
            "ram_percent": float(res["ram_percent"]),
            "ram_used_mb": float(res["ram_used_mb"]),
            "gpu_memory_mb": float(res["gpu_memory_allocated_mb"]),
        }
        metrics["class_counts_json"] = json.dumps(self._class_counts_dict)
        if is_fedsol:
            metrics["fedsol_active"] = True
            metrics["fedsol_rho"] = fedsol_rho
            metrics["fedsol_target"] = fedsol_target
            metrics["fedsol_loss_type"] = fedsol_loss_type
        if is_fedsld:
            metrics["fedsld_active"] = True
            metrics["fedsld_epsilon"] = fedsld_epsilon
            metrics["fedsld_reduction"] = fedsld_reduction

        # Track history locally (deduplicated by round)
        self._upsert_round_history(metrics)
        cum_t = 0.0
        round_epochs = []
        for ep_idx, ep_time in enumerate(train_metrics["epoch_times"]):
            cum_t += ep_time
            round_epochs.append({
                "round": server_round,
                "client_id": self.client_id,
                "device_type": self.device_type,
                "device_ipv4": self.device_ipv4,
                "epoch": ep_idx + 1,
                "epoch_time_seconds": round(ep_time, 4),
                "cumulative_epoch_time_seconds": round(cum_t, 4),
                "train_loss": round(train_metrics["epoch_losses"][ep_idx], 6),
                "train_accuracy": round(train_metrics["epoch_accuracies"][ep_idx], 6),
                "learning_rate": train_metrics["epoch_lrs"][ep_idx],
            })
        self._replace_round_epoch_history(server_round, round_epochs)

        if self.save_local_metrics:
            self._save_client_csvs()
            self._save_client_plots()
            self._save_latest_state(server_round=server_round, strategy=strategy)

        print(
            f"[Client {self.client_id}] fit done | Round {server_round} | IPv4: {self.device_ipv4} | "
            f"loss={metrics['train_loss']:.4f} acc={metrics['train_accuracy']:.4f} "
            f"| round_time={training_time:.2f}s | avg_epoch={metrics['epoch_time_avg']:.2f}s "
            f"| weights={weight_size_kb:.1f}KB | ping={metrics['ping_ms']:.2f}ms"
        )

        # Check if this is the final round
        is_final_round = bool(config.get("is_final_round", False))
        if not is_final_round and "total_rounds" in config:
            is_final_round = (server_round >= int(config["total_rounds"]))

        # Build telemetry and performance metrics transmitted every round to server
        return_metrics = {
            "client_id": int(self.client_id),
            "server_round": int(server_round),
            "train_loss": float(metrics["train_loss"]),
            "train_accuracy": float(metrics["train_accuracy"]),
            "round_time": float(training_time),
            "epoch_time_avg": float(metrics["epoch_time_avg"]),
            "ping_ms": float(ping_ms) if ping_ms is not None else -1.0,
            "weight_size_kb": float(weight_size_kb),
            "num_samples": int(train_metrics["num_samples"]),
            "device_ipv4": str(self.device_ipv4),
            "device_type": str(self.device_type),
        }
        if self.eval_history and int(self.eval_history[-1].get("server_round", -1)) == int(server_round):
            return_metrics["eval_loss"] = float(self.eval_history[-1]["eval_loss"])
            return_metrics["eval_accuracy"] = float(self.eval_history[-1]["eval_accuracy"])

        if is_fedsol:
            return_metrics["fedsol_active"] = True
            return_metrics["fedsol_rho"] = fedsol_rho
        if is_fedsld:
            return_metrics["fedsld_active"] = True
            return_metrics["fedsld_epsilon"] = fedsld_epsilon

        if is_final_round:
            # Transmit accumulated multi-round metrics and class distribution once at training completion
            return_metrics.update(dict(metrics))
            return_metrics["final_client_history_json"] = json.dumps(self.round_history)
            return_metrics["class_counts_json"] = json.dumps(self._class_counts_dict)
            return_metrics["eval_history_json"] = json.dumps(self.eval_history)
            print(f"[Client {self.client_id}] Final round reached -> Packaging and returning full training history & class distribution to server.")

        return return_parameters, train_metrics["num_samples"], return_metrics

    def _save_checkpoint(self, filename: str, state_dict: Optional[dict] = None) -> Optional[Path]:
        """Save model checkpoint to client directory."""
        if not self.save_local_metrics:
            return None
        try:
            self.client_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.client_dir / filename
            if state_dict is None:
                state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            torch.save(state_dict, save_path)
            pth_size = save_path.stat().st_size / 1024.0
            print(f"[Client {self.client_id}] Checkpoint Saved: {filename} ({pth_size:.1f} KB)")
            return save_path
        except Exception as e:
            print(f"[Client {self.client_id}] Warning: Could not save checkpoint {filename}: {e}")
            return None

    def _save_client_csvs(self):
        """Export local client CSV metrics for on-device inspection."""
        try:
            client_dir = self.client_dir
            client_dir.mkdir(parents=True, exist_ok=True)

            # Round metrics
            r_path = client_dir / f"client_{self.client_id}_round_metrics.csv"
            r_cols = [
                "server_round", "client_id", "device_type", "num_samples", "local_epochs",
                "learning_rate", "training_time", "epoch_time_avg", "epoch_time_min",
                "epoch_time_max", "train_loss", "train_accuracy", "ping_ms", "weight_size_kb",
                "cpu_percent", "ram_percent", "ram_used_mb", "gpu_memory_mb"
            ]
            import csv
            with open(r_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=r_cols, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.round_history)

            # Epoch metrics
            ep_path = client_dir / f"client_{self.client_id}_epoch_metrics.csv"
            ep_cols = [
                "round", "client_id", "device_type", "epoch", "epoch_time_seconds",
                "cumulative_epoch_time_seconds", "train_loss", "train_accuracy", "learning_rate"
            ]
            with open(ep_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=ep_cols, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.epoch_history)

            # Evaluation metrics
            if self.eval_history:
                ev_path = client_dir / f"client_{self.client_id}_eval_metrics.csv"
                ev_cols = ["server_round", "eval_loss", "eval_accuracy"]
                with open(ev_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=ev_cols, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(self.eval_history)
        except Exception as e:
            print(f"[Client {self.client_id}] Warning: could not write local CSV: {e}")

    def _save_client_plots(self):
        """Generate and save dual-panel loss and accuracy curves (PNG)."""
        if not self.save_local_metrics:
            return
        try:
            import tempfile
            os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not self.round_history and not self.eval_history:
                return

            self.client_dir.mkdir(parents=True, exist_ok=True)
            fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
            fig.patch.set_facecolor("white")

            for ax in (ax_loss, ax_acc):
                ax.set_facecolor("white")
                ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
                for spine in ax.spines.values():
                    spine.set_color("#94a3b8")

            # --- Panel 1: Loss ---
            rounds_fit = [r["server_round"] for r in self.round_history]
            train_losses = [r["train_loss"] for r in self.round_history]
            if rounds_fit:
                ax_loss.plot(
                    rounds_fit,
                    train_losses,
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    color="#2563eb",
                    label="Train Loss",
                )

            rounds_eval = [e["server_round"] for e in self.eval_history]
            eval_losses = [e["eval_loss"] for e in self.eval_history]
            if rounds_eval:
                ax_loss.plot(
                    rounds_eval,
                    eval_losses,
                    marker="s",
                    markersize=5,
                    linewidth=2,
                    color="#dc2626",
                    label="Eval / Test Loss",
                )

            ax_loss.set_title("Training & Evaluation Loss", fontsize=12, fontweight="bold", pad=8)
            ax_loss.set_xlabel("Server Round", fontsize=11)
            ax_loss.set_ylabel("Loss", fontsize=11)
            ax_loss.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=10)

            # --- Panel 2: Accuracy ---
            train_accs = [r["train_accuracy"] * 100 for r in self.round_history]
            if rounds_fit:
                ax_acc.plot(
                    rounds_fit,
                    train_accs,
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    color="#3b82f6",
                    label="Train Accuracy (%)",
                )

            eval_accs = [e["eval_accuracy"] * 100 for e in self.eval_history]
            if rounds_eval:
                ax_acc.plot(
                    rounds_eval,
                    eval_accs,
                    marker="s",
                    markersize=5,
                    linewidth=2,
                    color="#059669",
                    label="Local Test Accuracy (TA %)",
                )
                # Cumulative Best Test Accuracy (BTA)
                cum_bta = []
                curr_best = 0.0
                for a in eval_accs:
                    curr_best = max(curr_best, a)
                    cum_bta.append(curr_best)
                ax_acc.plot(
                    rounds_eval,
                    cum_bta,
                    marker="^",
                    markersize=5,
                    linestyle="--",
                    linewidth=1.8,
                    color="#7c3aed",
                    label="Best Test Accuracy (BTA %)",
                )

            ax_acc.set_title("Training & Evaluation Accuracy", fontsize=12, fontweight="bold", pad=8)
            ax_acc.set_xlabel("Server Round", fontsize=11)
            ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
            ax_acc.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=10)

            strat_str = (self.strategy or "Federated").upper()
            fig.suptitle(
                f"Client {self.client_id} Training Progress | Strategy: {strat_str}",
                fontsize=13,
                fontweight="bold",
                y=0.98,
            )

            plt.tight_layout()
            out_path = self.client_dir / f"client_{self.client_id}_round_curves.png"
            plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[Client {self.client_id}] Warning: Could not generate client plots: {e}")

    def save_artifacts(self):
        """Export all client artifacts: models, CSV metrics, and training curves plot."""
        if not self.save_local_metrics:
            return
        if not self.round_history and not self.eval_history:
            print(
                f"[Client {self.client_id}] No rounds completed in this session; "
                f"preserving existing local artifacts."
            )
            return
        print(f"\n{'='*55}")
        print(f"[Client {self.client_id}] Finalizing and saving local client artifacts...")
        print(f"{'='*55}")
        self.client_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save final model
        self._save_checkpoint("final_model.pth")

        # 2. Ensure best model exists (if no evaluate improved best_eval_accuracy, use final model)
        best_path = self.client_dir / "best_model.pth"
        if not best_path.exists():
            if self.best_model_state is not None:
                self._save_checkpoint("best_model.pth", self.best_model_state)
            else:
                self._save_checkpoint("best_model.pth")

        # 3. Export CSVs
        self._save_client_csvs()

        # 4. Generate and save dual-panel round training curves (no epoch plot on disk)
        self._save_client_plots()

        # 5. Persist latest recoverable state
        last_rnd = 0
        if self.round_history:
            last_rnd = max(last_rnd, int(self.round_history[-1].get("server_round", 0)))
        if self.eval_history:
            last_rnd = max(last_rnd, int(self.eval_history[-1].get("server_round", 0)))
        if last_rnd > 0:
            self._save_latest_state(server_round=last_rnd, strategy=self.strategy)

        plot_path = self.client_dir / f"client_{self.client_id}_round_curves.png"
        print(f"  Client Dir : {self.client_dir}")
        print(f"  Best Model : {self.client_dir / 'best_model.pth'}")
        print(f"  Final Model: {self.client_dir / 'final_model.pth'}")
        if plot_path.exists():
            print(f"  Plot Curve : {plot_path}")
        print(f"{'='*55}\n")

    def evaluate(self, parameters: NDArrays, config: dict):
        """Evaluate the global model on the local test set."""
        eval_round = config.get("server_round")
        strategy = str(config.get("strategy", self.strategy or "fedavg")).lower()
        eval_round_int = int(eval_round) if eval_round is not None else 0
        self._ensure_session_initialized(server_round=eval_round_int, strategy=strategy, config=config, phase="evaluate")

        eval_ping_ms = None
        if eval_round is not None:
            eval_ping_res = self.ping_logger.log_ping(server_round=int(eval_round), event="evaluate")
            eval_ping_ms = eval_ping_res.get("ping_ms")

        fedbn_eval_strats = (
            "fedbn", "proposed", "fedbns", "fedbn_scaffold", "bn_scaffold", "scaffold_fedbn",
        )
        if strategy in fedbn_eval_strats:
            set_parameters_fedbn(self.model, parameters)
        else:
            set_parameters(self.model, parameters)

        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.view(-1).long().to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if total == 0:
            # A Dirichlet split can give a training client no local test images.
            # Report zero examples so sample-weighted evaluation ignores it.
            if not getattr(self, "_empty_test_warning_issued", False):
                print(f"[Client {self.client_id}] No local test samples; skipping evaluation")
                self._empty_test_warning_issued = True
            return 0.0, 0, {
                "accuracy": 0.0,
                "client_id": int(self.client_id),
                "server_round": int(eval_round) if eval_round is not None else 0,
            }

        avg_loss = total_loss / total
        accuracy = correct / total

        # Track evaluation metrics history (deduplicated by round)
        resolved_round = int(eval_round) if eval_round is not None else (
            self.round_history[-1]["server_round"] if self.round_history else (len(self.eval_history) + 1)
        )
        self._upsert_eval_history({
            "server_round": resolved_round,
            "eval_loss": float(avg_loss),
            "eval_accuracy": float(accuracy),
        })

        if accuracy > self.best_eval_accuracy:
            self.best_eval_accuracy = accuracy
            self.best_eval_round = resolved_round
            self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self._save_checkpoint("best_model.pth", self.best_model_state)
            print(
                f"[Client {self.client_id}] New Best Local Test Accuracy: {accuracy*100:.2f}% "
                f"(Round {resolved_round}) -> Saved best_model.pth"
            )

        if self.save_local_metrics:
            self._save_client_csvs()
            self._save_client_plots()
            self._save_latest_state(server_round=resolved_round, strategy=strategy)

        ping_str = f" | ping={eval_ping_ms:.2f}ms" if eval_ping_ms is not None and eval_ping_ms >= 0 else ""
        print(
            f"[Client {self.client_id}] evaluate | Round {resolved_round} | "
            f"loss={avg_loss:.4f} acc={accuracy:.4f}{ping_str}"
        )

        return float(avg_loss), total, {
            "accuracy": float(accuracy),
            "client_id": int(self.client_id),
            "server_round": int(resolved_round),
            "eval_loss": float(avg_loss),
            "eval_accuracy": float(accuracy),
            "num_samples": int(len(self.train_loader.dataset)),
            "class_counts_json": json.dumps(self._class_counts_dict),
        }


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

def load_yaml_config(config_path: str = "configs/experiment.yaml") -> dict:
    """Load configuration dictionary from YAML file if available."""
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except Exception:
        return {}


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to YAML experiment configuration file (default: configs/experiment.yaml)",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()
    yaml_cfg = load_yaml_config(pre_args.config)

    server_host = yaml_cfg.get("server_host", "127.0.0.1")
    if server_host == "0.0.0.0":
        server_host = "127.0.0.1"
    server_port = yaml_cfg.get("server_port", 8080)
    default_server_address = yaml_cfg.get("server_address") or f"{server_host}:{server_port}"

    active_clients = yaml_cfg.get("active_clients")
    default_num_clients = len(active_clients) if isinstance(active_clients, list) and active_clients else yaml_cfg.get("num_clients", yaml_cfg.get("min_clients", 12))

    parser = argparse.ArgumentParser(
        description="FedMedAI FL Client",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--client_id", type=int, required=True, help="Client ID integer (0, 1, 2, ...)"
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default=default_server_address,
        help=f"FL Server address (default: {default_server_address})",
    )
    parser.add_argument(
        "--partition_path",
        type=str,
        default=None,
        help="Path to partition JSON. If None, auto-detect from data/partitions/",
    )
    parser.add_argument(
        "--dataset", type=str, default=yaml_cfg.get("dataset", "bloodmnist"), help=f"Dataset name (bloodmnist, pathmnist, organamnist, dermamnist) (default: {yaml_cfg.get('dataset', 'bloodmnist')})"
    )
    parser.add_argument(
        "--partition_mode", type=str, default=yaml_cfg.get("partition_mode", yaml_cfg.get("partition", "dirichlet_0.3")), help=f"Partition mode (default: {yaml_cfg.get('partition_mode', yaml_cfg.get('partition', 'dirichlet_0.3'))})"
    )
    parser.add_argument(
        "--num_clients", type=int, default=default_num_clients, help=f"Total number of clients (default: {default_num_clients})"
    )
    parser.add_argument(
        "--local_epochs", type=int, default=yaml_cfg.get("local_epochs", 5), help=f"Local training epochs per round (default: {yaml_cfg.get('local_epochs', 5)})"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=yaml_cfg.get("learning_rate", 0.001), help=f"Initial base learning rate (default: {yaml_cfg.get('learning_rate', 0.001)})"
    )
    parser.add_argument(
        "--lr_patience",
        "--client_lr_patience",
        dest="lr_patience",
        type=int,
        default=yaml_cfg.get("client_lr_patience", 2),
        help=f"Client local epochs without loss improvement before reducing LR (default: {yaml_cfg.get('client_lr_patience', 2)})",
    )
    parser.add_argument(
        "--lr_factor",
        "--client_lr_factor",
        dest="lr_factor",
        type=float,
        default=yaml_cfg.get("client_lr_factor", 0.5),
        help=f"Multiplicative factor for client LR reduction (default: {yaml_cfg.get('client_lr_factor', 0.5)})",
    )
    parser.add_argument(
        "--lr_min",
        "--client_lr_min",
        dest="lr_min",
        type=float,
        default=yaml_cfg.get("client_lr_min", 1e-6),
        help=f"Minimum client learning rate bound (default: {yaml_cfg.get('client_lr_min', 1e-6)})",
    )
    parser.add_argument(
        "--lr_mode",
        type=str,
        default=yaml_cfg.get("lr_mode", "client_loss"),
        choices=["client_loss", "server_decay", "fixed"],
        help=f"LR schedule mode: 'client_loss' (each client reduces LR on local loss plateau), 'server_decay' (step decay on rounds), 'fixed' (constant LR) (default: {yaml_cfg.get('lr_mode', 'client_loss')})",
    )
    parser.add_argument(
        "--batch_size", type=int, default=yaml_cfg.get("batch_size", None), help="Override batch size from config (default: from config or jetson.yaml)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=yaml_cfg.get("num_workers", 0 if sys.platform == "win32" else None),
        help=f"DataLoader worker processes (default: 0 on Windows, or from config)",
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("pin_memory", None),
        help="Enable/disable pinned host memory for DataLoader transfers",
    )
    parser.add_argument(
        "--device_type",
        type=str,
        default=None,
        help="Device class/type override (e.g. 'pc', 'jetson_orin', 'jetson_nano', 'PC1'). Defaults to auto-detected from configs/jetson.yaml.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=yaml_cfg.get("strategy", "fedavg"),
        choices=[
            "fedavg",
            "fedprox",
            "fedbn",
            "scaffold",
            "fedbns",
            "proposed",
            "fedbn_scaffold",
            "fedsld",
            "fed_sld",
            "fedsol",
            "fed_sol",
            "fedswitch",
            "fed_switch",
        ],
        help=f"FL strategy (default: {yaml_cfg.get('strategy', 'fedavg')})",
    )
    parser.add_argument(
        "--client_dir",
        type=str,
        default=None,
        help="Custom directory to store client metrics and ping logs (default: results/clients/[<name>/]client_<id>)",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=yaml_cfg.get("name", None),
        help="Designated run name for client output directory (default: None)",
    )
    parser.add_argument(
        "--local_test",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("local_test", False),
        help="Evaluate on client-specific local test partition (following FedSLD paper)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=yaml_cfg.get("model", "baseline_bn"),
        choices=["baseline", "paper", "classic_cnn", "baseline_bn", "baselinecnn_bn", "classic_bn"],
        help=f"Model architecture (default: {yaml_cfg.get('model', 'baseline_bn')})",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=yaml_cfg.get("optimizer", "adam"),
        choices=["adam", "sgd", "adamw"],
        help=f"Client local optimizer ('adam', 'sgd', or 'adamw') (default: {yaml_cfg.get('optimizer', 'adam')})",
    )
    parser.add_argument(
        "--fedsol_rho",
        type=float,
        default=yaml_cfg.get("fedsol_rho", 2.0),
        help=f"FedSOL perturbation radius rho (default: {yaml_cfg.get('fedsol_rho', 2.0)})",
    )
    parser.add_argument(
        "--fedsol_target",
        type=str,
        default=yaml_cfg.get("fedsol_target", "head"),
        choices=["head", "all"],
        help=f"FedSOL perturbation target layer ('head' or 'all') (default: {yaml_cfg.get('fedsol_target', 'head')})",
    )
    parser.add_argument(
        "--fedsol_loss_type",
        type=str,
        default=yaml_cfg.get("fedsol_loss_type", "kl"),
        choices=["kl", "l2"],
        help=f"FedSOL proximal discrepancy loss type ('kl' or 'l2') (default: {yaml_cfg.get('fedsol_loss_type', 'kl')})",
    )
    parser.add_argument(
        "--fedsol_temperature",
        type=float,
        default=yaml_cfg.get("fedsol_temperature", 3.0),
        help=f"FedSOL softmax temperature T for KL divergence (default: {yaml_cfg.get('fedsol_temperature', 3.0)})",
    )
    parser.add_argument(
        "--fedsol_adaptive",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("fedsol_adaptive", False),
        help=f"Enable adaptive perturbation radius for FedSOL (default: {yaml_cfg.get('fedsol_adaptive', False)})",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=yaml_cfg.get("clip_grad_norm", None),
        help=f"Max gradient L2 norm for clipping (default: {yaml_cfg.get('clip_grad_norm', None)})",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=yaml_cfg.get("momentum", 0.0),
        help=f"Momentum factor for SGD optimizer (default: {yaml_cfg.get('momentum', 0.0)})",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=yaml_cfg.get("weight_decay", 0.0),
        help=f"Weight decay penalty (default: {yaml_cfg.get('weight_decay', 0.0)})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=yaml_cfg.get("seed", 42),
        help=f"Random seed for reproducibility (default: {yaml_cfg.get('seed', 42)})",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("augment", True),
        help=f"Enable/disable training data augmentation (default: {yaml_cfg.get('augment', True)})",
    )
    parser.add_argument(
        "--random_crop",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("random_crop", True),
        help=f"Enable/disable random crop during training (default: {yaml_cfg.get('random_crop', True)})",
    )
    parser.add_argument(
        "--random_flip",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("random_flip", True),
        help=f"Enable/disable random horizontal flip during training (default: {yaml_cfg.get('random_flip', True)})",
    )
    parser.add_argument(
        "--no_save_metrics",
        action="store_false",
        dest="save_metrics",
        help="Disable saving local client metrics to CSV",
    )
    parser.add_argument(
        "--auto_resume",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("auto_resume", True),
        help="Automatically resume latest local model/BN/optimizer/history when reconnecting at Round > 1 (default: True)",
    )
    parser.set_defaults(save_metrics=True)
    args = parser.parse_args()

    set_seed(args.seed)

    # Resolve partition path
    if args.partition_path:
        partition_path = args.partition_path
    else:
        partition_path = str(
            get_partition_path(
                dataset_name=args.dataset,
                partition_mode=args.partition_mode,
                seed=args.seed,
                num_clients=args.num_clients,
                local_test=args.local_test,
            )
        )

    if not Path(partition_path).exists():
        print(f"[Client {args.client_id}] Partition file not found: {partition_path}")
        print(
            f"[Client {args.client_id}] Auto-generating missing partition for "
            f"dataset={args.dataset}, mode={args.partition_mode}, clients={args.num_clients}, seed={args.seed}..."
        )
        try:
            from experiments.run_partition import run_one
            run_one(
                dataset_name=args.dataset,
                partition_mode=args.partition_mode,
                num_clients=args.num_clients,
                seed=args.seed,
                local_test=args.local_test,
            )
            print(f"[Client {args.client_id}] Auto-generation completed successfully.")
        except Exception as e:
            print(f"ERROR: Failed to auto-generate partition: {e}")
            sys.exit(1)

    print(f"[Client {args.client_id}] Loading partition from {partition_path}")

    # Load dataset and partition
    train_dataset, _, test_dataset, num_classes = get_medmnist_datasets(
        args.dataset,
        download=True,
        augment=args.augment,
        random_crop=args.random_crop,
        random_flip=args.random_flip,
        load_val=False,
    )
    print(f"[Client {args.client_id}] Augmentation: augment={args.augment}, crop={args.random_crop}, flip={args.random_flip}")
    client_indices = load_partition(partition_path, split="train")

    if args.client_id >= len(client_indices):
        print(
            f"ERROR: client_id={args.client_id} but partition only has "
            f"{len(client_indices)} clients (0-{len(client_indices)-1})"
        )
        sys.exit(1)

    my_indices = client_indices[args.client_id]
    resolved_device_type = str(args.device_type).strip() if args.device_type else get_device_type(args.client_id)
    print(f"[Client {args.client_id}] Assigned {len(my_indices)} training samples")

    # Create data loaders
    train_loader = get_client_dataloader(
        train_dataset,
        my_indices,
        client_id=args.client_id,
        device_type=resolved_device_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=True,
    )

    from torch.utils.data import DataLoader

    if args.local_test:
        client_test_indices = load_partition(partition_path, split="test")
        if client_test_indices is not None and args.client_id < len(client_test_indices):
            my_test_indices = client_test_indices[args.client_id]
            print(f"[Client {args.client_id}] Using local test partition: {len(my_test_indices)} samples")
            test_loader = get_client_dataloader(
                test_dataset,
                my_test_indices,
                client_id=args.client_id,
                device_type=resolved_device_type,
                batch_size=args.batch_size or 32,
                shuffle=False,
                num_workers=0,
            )
        else:
            print(f"[Client {args.client_id}] Warning: local_test requested but no test_indices found in partition. Falling back to global test set.")
            test_loader = DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=0
            )
    else:
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=0
        )

    # Create client and start
    client = FedMedAIClient(
        client_id=args.client_id,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        lr_mode=args.lr_mode,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
        device_type=resolved_device_type,
        save_local_metrics=args.save_metrics,
        server_address=args.server_address,
        strategy=args.strategy,
        partition_path=partition_path,
        client_dir=args.client_dir,
        model_name=args.model,
        optimizer_name=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        name=args.name,
        fedsol_rho=args.fedsol_rho,
        fedsol_target=args.fedsol_target,
        fedsol_loss_type=args.fedsol_loss_type,
        fedsol_temperature=args.fedsol_temperature,
        fedsol_adaptive=args.fedsol_adaptive,
        clip_grad_norm=args.clip_grad_norm,
        auto_resume=args.auto_resume,
    )

    print(f"[Client {args.client_id}] Connecting to {args.server_address}...")
    try:
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client,
        )
    except KeyboardInterrupt:
        print(f"\n[Client {args.client_id}] Interrupted by user (Ctrl+C). Exiting...")
    except Exception as e:
        print(f"\n[Client {args.client_id}] Session ended: {e}")
    finally:
        client.save_artifacts()


if __name__ == "__main__":
    main()
