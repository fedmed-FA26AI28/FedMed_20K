r"""Sequential FL development run: no test split is loaded here.

From the FedMed_20K directory:
    ..\.venv\Scripts\python.exe -m research.run --dataset bloodmnist \
        --partition dirichlet --method support_calibrated --seed 42
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from datasets.partition import _extract_labels, dirichlet_partition, practical_partition
from research.data import get_medmnist_datasets
from research.fast_data import FastImageDataset, prepare_batch
from algorithms.research_methods import (
    ClientContext, METHODS, TinyCNNGN, classifier_head, fedswitch_oracle_weights,
    get_method, local_objective, make_model, support_statistics,
)


REPO = Path(__file__).resolve().parents[1]


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bloodmnist", "pathmnist", "organamnist"), required=True)
    parser.add_argument("--partition", choices=("practical", "dirichlet"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model", "--norm", dest="norm",
                        choices=("baseline", "baseline_bn", "gn", "bn"), default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=28)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true",
                        help="Enable train-only random crop and horizontal flip")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--head-lambda", type=float, default=0.01)
    parser.add_argument("--coverage-kappa", type=float, default=10.0,
                        help="Fixed scalar head-support parameter for Coverage-IW")
    parser.add_argument("--fedlc-tau", type=float, default=0.5)
    parser.add_argument("--fedvls-lambda", type=float, default=0.1)
    parser.add_argument("--client-shift-strength", type=float, default=0.0,
                        help="Fixed client-specific RGB gain and offset on training images; synthetic feature shift")
    parser.add_argument("--limit-per-client", type=int, default=0,
                        help="Development smoke test only; uses first N shuffled indices per client")
    parser.add_argument("--slow-data", dest="fast_data", action="store_false",
                        help="Use the reference PIL transform path instead of tensor batches")
    parser.set_defaults(fast_data=True)
    parser.add_argument("--output-root", type=Path, default=REPO / "results" / "research")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluation_transform(dataset, resolution: int):
    channels = dataset.imgs.shape[-1] if dataset.imgs.ndim == 4 else 1
    shared = [transforms.Resize((resolution, resolution)), transforms.ToTensor()]
    if channels == 1:
        shared.append(transforms.Lambda(lambda image: image.repeat(3, 1, 1)))
    shared.append(transforms.Normalize([0.5] * 3, [0.5] * 3))
    return transforms.Compose(shared)


def set_transforms(train, val, resolution: int, augment: bool = False) -> None:
    shared = evaluation_transform(train, resolution).transforms
    augmentation = ([transforms.RandomCrop(28, padding=2, padding_mode="reflect"),
                     transforms.RandomHorizontalFlip()] if augment else [])
    train.transform = transforms.Compose([*augmentation, *shared])
    val.transform = evaluation_transform(val, resolution)


def partition_indices(train, config: argparse.Namespace) -> list[list[int]]:
    if config.partition == "practical":
        groups = practical_partition(train, config.clients, seed=config.seed)
    else:
        groups = dirichlet_partition(train, config.clients, config.alpha, seed=config.seed)
    groups = [list(map(int, group)) for group in groups]
    if config.limit_per_client:
        groups = [group[:config.limit_per_client] for group in groups]
    if any(len(group) == 0 for group in groups):
        raise ValueError("partition has an empty client; use a different partition seed")
    all_indices = [idx for group in groups for idx in group]
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("training partition contains duplicate indices")
    if not config.limit_per_client and set(all_indices) != set(range(len(train))):
        raise ValueError("training partition does not cover the complete train split")
    return groups


def class_counts(train, groups: list[list[int]], classes: int) -> np.ndarray:
    labels = _extract_labels(train)
    return np.stack([np.bincount(labels[group], minlength=classes) for group in groups])


def validation_groups(val, counts: np.ndarray, seed: int) -> list[list[int]]:
    """Assign validation images to virtual clients using training class shares.

    This partition is for reporting only; it never enters optimization or
    global validation checkpoint selection.
    """
    labels = _extract_labels(val)
    rng = np.random.RandomState(seed + 9109)
    groups = [[] for _ in range(len(counts))]
    for cls in range(counts.shape[1]):
        indices = np.where(labels == cls)[0]
        rng.shuffle(indices)
        total = counts[:, cls].sum()
        proportions = (counts[:, cls] / total if total else
                       np.full(len(counts), 1 / len(counts)))
        allocation = rng.multinomial(len(indices), proportions)
        offset = 0
        for client_id, quantity in enumerate(allocation):
            groups[client_id].extend(indices[offset:offset + quantity].tolist())
            offset += quantity
    assert sum(map(len, groups)) == len(val)
    return groups


def client_shift_parameters(seed: int, client_id: int, amount: float):
    rng = np.random.RandomState(seed + 9829 * client_id)
    gains = torch.as_tensor(1 + rng.uniform(-amount, amount, size=3),
                            dtype=torch.float32).view(3, 1, 1)
    offsets = torch.as_tensor(rng.uniform(-amount / 2, amount / 2, size=3),
                              dtype=torch.float32).view(3, 1, 1)
    return gains, offsets


class ClientShiftDataset:
    """Apply deterministic synthetic client-specific RGB shift after normalization."""

    def __init__(self, dataset, groups: list[list[int]], seed: int, amount: float):
        self.dataset = dataset
        self.client_for_index = np.full(len(dataset), -1, dtype=np.int64)
        self.parameters = [client_shift_parameters(seed, cid, amount)
                           for cid in range(len(groups))]
        for cid, indices in enumerate(groups):
            if np.any(self.client_for_index[indices] != -1):
                raise ValueError("synthetic evaluation groups overlap")
            self.client_for_index[indices] = cid
        if np.any(self.client_for_index < 0):
            raise ValueError("synthetic evaluation groups do not cover the split")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        gain, offset = self.parameters[self.client_for_index[index]]
        return (image * gain + offset).clamp(-1, 1), label


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "-infinity"
    return value


def run_name(config: argparse.Namespace) -> str:
    payload = {key: _jsonable(value) for key, value in vars(config).items()
               if key not in ("output_root", "download", "device")}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"{config.dataset}_{config.partition}_{config.method}_s{config.seed}_{digest}"


def metrics_from_probabilities(probs: np.ndarray, labels: np.ndarray) -> dict:
    classes = probs.shape[1]
    predicted = probs.argmax(axis=1)
    confusion = np.bincount(labels * classes + predicted, minlength=classes * classes).reshape(classes, classes)
    true_positive = np.diag(confusion).astype(float)
    supports = confusion.sum(axis=1).astype(float)
    pred_support = confusion.sum(axis=0).astype(float)
    recalls = np.divide(true_positive, supports, out=np.zeros(classes), where=supports > 0)
    f1 = np.divide(2 * true_positive, supports + pred_support,
                   out=np.zeros(classes), where=(supports + pred_support) > 0)
    chosen = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    confidence = probs.max(axis=1)
    correct = (predicted == labels).astype(float)
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    one_hot = np.eye(classes)[labels]
    return {
        "macro_f1": float(f1.mean()), "accuracy": float(correct.mean()),
        "balanced_accuracy": float(recalls[supports > 0].mean()),
        "worst_class_recall": float(recalls[supports > 0].min()),
        "class_recall": recalls.tolist(), "class_f1": f1.tolist(),
        "confusion": confusion.tolist(), "nll": float(-np.log(chosen).mean()),
        "brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "ece_15": float(ece), "samples": int(len(labels)),
    }


@torch.inference_mode()
def evaluate(model: TinyCNNGN, dataset, device: torch.device, batch_size: int,
             client_groups: list[list[int]] | None = None) -> dict:
    model.eval()
    probabilities, labels = [], []
    for images, targets in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        images = images.to(device)
        if images.dtype == torch.uint8:
            images = prepare_batch(images, dataset.resolution, training=False)
        logits = model(images)
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
        labels.append(np.asarray(targets).reshape(-1))
    probs, true = np.concatenate(probabilities), np.concatenate(labels)
    metrics = metrics_from_probabilities(probs, true)
    if client_groups is not None:
        client_scores = []
        for indices in client_groups:
            if not indices:
                client_scores.append(None)
                continue
            score = metrics_from_probabilities(probs[indices], true[indices])
            supported = np.asarray(score["confusion"]).sum(axis=1) > 0
            client_scores.append({"samples": len(indices), "accuracy": score["accuracy"],
                                  "observed_class_macro_f1": float(np.asarray(score["class_f1"])[supported].mean())})
        metrics["virtual_client_validation"] = client_scores
        metrics["worst_virtual_client_accuracy"] = min(
            score["accuracy"] for score in client_scores if score is not None)
    return metrics


def train_client(
    model: TinyCNNGN, teacher: TinyCNNGN | None, train, indices: list[int],
    counts: np.ndarray, global_counts: np.ndarray, weights: np.ndarray,
    anchor: np.ndarray, config: argparse.Namespace, device: torch.device,
    round_number: int, client_id: int,
) -> tuple[dict[str, torch.Tensor], float]:
    # Identical sampling and augmentation seeds across matched method runs.
    set_seed(config.seed + round_number * 100003 + client_id)
    loader_seed = torch.Generator().manual_seed(config.seed + round_number * 100003 + client_id)
    loader = DataLoader(Subset(train, indices), batch_size=config.batch_size, shuffle=True,
                        generator=loader_seed, num_workers=0, pin_memory=device.type == "cuda")
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                     weight_decay=config.weight_decay)
    local_counts = torch.as_tensor(counts, dtype=torch.float32, device=device)
    global_counts_t = torch.as_tensor(global_counts, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    anchor_t = torch.as_tensor(anchor, dtype=torch.float32, device=device)
    head = classifier_head(model)
    reference = (head.weight.detach().clone(), head.bias.detach().clone())
    total_loss, batches = 0.0, 0
    model.train()
    if config.client_shift_strength:
        gains, offsets = client_shift_parameters(config.seed, client_id,
                                                  config.client_shift_strength)
        gains, offsets = gains.to(device), offsets.to(device)
    for _ in range(config.epochs):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            if images.dtype == torch.uint8:
                images = prepare_batch(images, train.resolution,
                                       training=getattr(config, "augment", False))
            if config.client_shift_strength:
                images = (images * gains + offsets).clamp(-1, 1)
            targets = targets.reshape(-1).long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = local_objective(
                config.method, model, images, targets, local_counts, global_counts_t,
                weights_t, anchor_t, reference, config.head_lambda,
                config.fedlc_tau, teacher, config.fedvls_lambda,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at round {round_number}, client {client_id}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return state, total_loss / batches


def main() -> None:
    config = arguments()
    if config.clients < 2 or config.rounds < 1 or config.epochs < 1 or config.batch_size < 1:
        raise ValueError("clients, rounds, epochs and batch size must be positive")
    if config.limit_per_client < 0:
        raise ValueError("limit-per-client must be nonnegative")
    if not 0 <= config.client_shift_strength <= 1:
        raise ValueError("client-shift-strength must be between 0 and 1")
    if config.coverage_kappa <= 0:
        raise ValueError("coverage-kappa must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight-decay must be nonnegative")
    if not 0 <= config.momentum < 1:
        raise ValueError("momentum must be in [0, 1)")
    if config.client_shift_strength and config.fast_data:
        raise ValueError("synthetic client shift requires --slow-data")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available()
                          else "cpu" if config.device == "auto" else config.device)
    torch.set_num_threads(1)
    set_seed(config.seed)
    run_dir = config.output_root / run_name(config)
    result_file = run_dir / "development.json"
    if result_file.exists():
        from research.plots import plot_run
        plot_path = run_dir / "validation_curves.png"
        if not plot_path.exists():
            plot_run(json.loads(result_file.read_text(encoding="utf-8")), plot_path)
        print(f"Completed run already exists: {result_file}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    # Deliberately no test dataset: this runner cannot inspect test during training.
    train, val, _, classes = get_medmnist_datasets(
        config.dataset, download=config.download, augment=False,
        load_train=True, load_val=True, load_test=False,
    )
    if config.fast_data:
        train = FastImageDataset(train, config.resolution)
        val = FastImageDataset(val, config.resolution)
    else:
        set_transforms(train, val, config.resolution, getattr(config, "augment", False))
    groups = partition_indices(train, config)
    counts = class_counts(train, groups, classes)
    val_groups = validation_groups(val, counts, config.seed)
    if config.client_shift_strength:
        val = ClientShiftDataset(val, val_groups, config.seed, config.client_shift_strength)
    total_per_class = counts.sum(axis=0)
    support = support_statistics(counts, config.gamma)
    pooled = support_statistics(counts, math.inf)
    switch_weights = fedswitch_oracle_weights(counts)
    method = get_method(config.method)
    partition_hash = hashlib.sha256(json.dumps(groups, separators=(",", ":")).encode()).hexdigest()
    model = make_model(classes, config.norm).to(device)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_state = copy.deepcopy(initial)
    best_round, best_metrics = 0, None
    history = []
    model_bytes = sum(value.numel() * value.element_size() for value in initial.values())
    progress_file = run_dir / "progress.pt"
    start_round = 1
    if progress_file.exists():
        progress = torch.load(progress_file, map_location="cpu", weights_only=True)
        if progress["partition_sha256"] != partition_hash:
            raise ValueError("saved progress has a different training partition")
        model.load_state_dict(progress["state_dict"])
        best_state, best_round = progress["best_state"], progress["best_round"]
        best_metrics, history = progress["best_metrics"], progress["history"]
        start_round = progress["round"] + 1
        print(f"Resuming {run_name(config)} at round {start_round}", flush=True)
    for round_number in range(start_round, config.rounds + 1):
        round_start = time.time()
        global_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        teacher = None
        if method.needs_teacher:
            teacher = make_model(classes, config.norm).to(device)
            teacher.load_state_dict(global_state)
            teacher.eval()
            teacher.requires_grad_(False)
        accumulated = {key: torch.zeros_like(value, dtype=torch.float64) for key, value in global_state.items()}
        client_losses = []
        for client_id, indices in enumerate(groups):
            model.load_state_dict(global_state)
            weights, anchor = method.client_parameters(ClientContext(
                counts[client_id], total_per_class, support.coverage.sum(axis=0),
                pooled.coverage.sum(axis=0), int(counts.sum()), len(groups),
                config.gamma, config.coverage_kappa,
            ))
            local_state, local_loss = train_client(
                model, teacher, train, indices, counts[client_id], total_per_class,
                weights, anchor, config, device, round_number, client_id,
            )
            portion = (switch_weights[client_id]
                       if method.oracle_aggregation and round_number > 1
                       else len(indices) / sum(map(len, groups)))
            for key in accumulated:
                accumulated[key] += local_state[key].to(torch.float64) * portion
            client_losses.append(local_loss)
        # BatchNorm's num_batches_tracked is an integer buffer. Its default
        # momentum does not use this count, but it must remain integer-valued.
        aggregated = {
            key: (value if global_state[key].is_floating_point() else value.round())
            .to(global_state[key].dtype)
            for key, value in accumulated.items()
        }
        model.load_state_dict(aggregated)
        val_metrics = evaluate(model, val, device, config.batch_size, val_groups)
        history.append({"round": round_number, "validation": val_metrics,
                        "mean_client_loss": float(np.mean(client_losses)),
                        "seconds": time.time() - round_start})
        if best_metrics is None or val_metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_round, best_metrics = round_number, val_metrics
        progress = {
            "round": round_number, "partition_sha256": partition_hash,
            "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "best_state": best_state, "best_round": best_round,
            "best_metrics": best_metrics, "history": history,
        }
        temporary = run_dir / "progress.tmp"
        torch.save(progress, temporary)
        temporary.replace(progress_file)
        print(f"{run_name(config)} round {round_number}/{config.rounds}: "
              f"val macro-F1={val_metrics['macro_f1']:.4f}, acc={val_metrics['accuracy']:.4f}, "
              f"time={history[-1]['seconds']:.1f}s", flush=True)
    temporary_best = run_dir / "best.tmp"
    torch.save({"state_dict": best_state, "round": best_round}, temporary_best)
    temporary_best.replace(run_dir / "best.pt")
    metadata = {
        "phase": "development", "configuration": {key: _jsonable(value) for key, value in vars(config).items()},
        "dataset_split": "official MedMNIST train/val; test never loaded",
        "partition_sha256": partition_hash, "train_class_counts": counts.tolist(),
        "global_class_counts": total_per_class.tolist(),
        "support_normalizer": support.coverage.sum(axis=0).tolist(),
        "variance_factor": support.variance_factor.tolist(),
        "fedswitch_oracle_aggregation_weights": switch_weights.tolist()
        if method.oracle_aggregation else None,
        "model_parameters": sum(value.numel() for value in model.parameters()),
        "model_bytes": model_bytes, "estimated_communication_bytes": 2 * model_bytes * len(groups) * config.rounds,
        "best_validation_round": best_round, "best_validation": best_metrics,
        "history": history, "wall_seconds": time.time() - start,
    }
    temporary_result = run_dir / "development.tmp"
    temporary_result.write_text(json.dumps(metadata, indent=2, allow_nan=False),
                                encoding="utf-8")
    temporary_result.replace(result_file)
    print(f"Saved development result: {result_file}")
    from research.plots import plot_run
    plot_path = plot_run(metadata, run_dir / "validation_curves.png")
    print(f"Validation plot: {plot_path}")


if __name__ == "__main__":
    main()
