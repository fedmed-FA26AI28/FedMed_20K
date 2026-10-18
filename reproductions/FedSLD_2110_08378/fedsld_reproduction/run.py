"""CLI for the independent FedSLD paper reconstruction."""

import argparse
import datetime as dt
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import load_medmnist
from .federated import aggregate_states, train_client
from .loss import federation_prior
from .metrics import evaluate
from .model import PaperCNN
from .partition import (
    partition_hash,
    partition_indices,
    pathological_plan,
    practical_plan,
)


ALGORITHM_DESCRIPTIONS = {
    "fedavg": "Unmodified cross-entropy control.",
    "fedprox": "Cross-entropy plus the configured FedProx proximal term.",
    "fedsld": (
        "Literal Equation (3): p_batch/p_global weights with a batch-size mean."
    ),
    "fedsld_weightnorm": (
        "Diagnostic hypothesis: printed p_batch/p_global weights normalized by "
        "the sum of sample weights."
    ),
    "fedsld_inverse": (
        "Diagnostic hypothesis: reversed p_global/p_batch importance weights "
        "normalized by the sum of sample weights."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["organamnist", "pathmnist", "bloodmnist"], required=True)
    parser.add_argument("--algorithm", choices=sorted(ALGORITHM_DESCRIPTIONS), required=True)
    parser.add_argument("--partition", choices=["practical", "pathological"], required=True)
    parser.add_argument("--num-clients", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--proximal-mu", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results/reference")
    parser.add_argument(
        "--dataset-split",
        choices=["official", "stratified"],
        default="official",
        help="Official MedMNIST split or an explicit paper-style stratified hypothesis.",
    )
    parser.add_argument(
        "--stratified-test-fraction",
        type=float,
        default=None,
        help=(
            "Test fraction for --dataset-split stratified; by default it matches "
            "the official MedMNIST test-set size."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save a resumable checkpoint every N rounds; use 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Checkpoint output path; defaults to the method result directory.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume a matching run from a checkpoint, including a read-only Kaggle input.",
    )
    parser.add_argument(
        "--exclude-official-val",
        action="store_true",
        help="Use only official train as training pool; default merges train+val.",
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _state_dict_cpu(model: torch.nn.Module):
    """Clone a model state to CPU so retained checkpoints do not occupy GPU memory."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _run_signature(args, rounds: int):
    """Fields which must match before a paper run can be resumed safely."""
    return {
        "dataset": args.dataset,
        "algorithm": args.algorithm,
        "partition": args.partition,
        "num_clients": args.num_clients,
        "rounds": rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "proximal_mu": args.proximal_mu,
        "seed": args.seed,
        "dataset_split": args.dataset_split,
        "stratified_test_fraction": args.stratified_test_fraction,
        "official_validation_merged_into_train": not args.exclude_official_val,
    }


def _save_progress_checkpoint(
    path: Path,
    signature,
    completed_round: int,
    model: torch.nn.Module,
    best_test_accuracy: float,
    best_test_round,
    best_test_state,
    history,
    elapsed_seconds: float,
):
    """Atomically save all state needed to continue at the next FL round."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "signature": signature,
        "completed_round": completed_round,
        "model_state": _state_dict_cpu(model),
        "best_test_accuracy": best_test_accuracy,
        "best_test_round": best_test_round,
        "best_test_state": best_test_state,
        "history": history,
        "elapsed_seconds": elapsed_seconds,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_progress_checkpoint(path: Path, expected_signature):
    """Load a checkpoint only when it describes the exact requested experiment."""
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported resume checkpoint format")
    actual_signature = payload.get("signature")
    if not isinstance(actual_signature, dict):
        raise ValueError("resume checkpoint does not contain a valid run signature")
    if actual_signature != expected_signature:
        differences = {
            key: {"checkpoint": actual_signature.get(key), "requested": value}
            for key, value in expected_signature.items()
            if actual_signature.get(key) != value
        }
        raise ValueError(f"resume checkpoint configuration mismatch: {differences}")
    return payload


def main():
    args = parse_args()
    if args.num_clients != 12:
        raise ValueError("paper reconstruction requires exactly 12 clients")
    if args.local_epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint-every must be non-negative")
    if args.stratified_test_fraction is not None and not (
        0 < args.stratified_test_fraction < 1
    ):
        raise ValueError("stratified-test-fraction must be in (0, 1)")
    if args.dataset_split == "official" and args.stratified_test_fraction is not None:
        raise ValueError("stratified-test-fraction requires --dataset-split stratified")
    if args.dataset_split == "stratified" and args.exclude_official_val:
        raise ValueError("stratified mode pools official train, val, and test")
    rounds = args.rounds if args.rounds is not None else (80 if args.partition == "practical" else 160)
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)

    bundle = load_medmnist(
        args.dataset,
        root=args.data_root,
        include_validation_in_train=not args.exclude_official_val,
        split_mode=args.dataset_split,
        split_seed=args.seed,
        stratified_test_fraction=args.stratified_test_fraction,
    )
    plan = (
        practical_plan(bundle.num_classes, args.num_clients, args.seed)
        if args.partition == "practical"
        else pathological_plan(bundle.num_classes, args.num_clients, args.seed)
    )
    train_partitions = partition_indices(bundle.train_labels, plan, args.seed)
    test_partitions = partition_indices(bundle.test_labels, plan, args.seed + 1)
    if any(len(partition) == 0 for partition in train_partitions + test_partitions):
        raise RuntimeError("paper partition produced an empty client")

    class_counts = np.zeros((args.num_clients, bundle.num_classes), dtype=np.int64)
    for client_id, indices in enumerate(train_partitions):
        class_counts[client_id] = np.bincount(
            bundle.train_labels[indices], minlength=bundle.num_classes
        )
    prior = federation_prior(torch.from_numpy(class_counts))
    model = PaperCNN(bundle.in_channels, bundle.num_classes, bundle.image_size).to(device)
    combined_test_loader = DataLoader(bundle.test, batch_size=args.batch_size, shuffle=False)

    output = Path(args.output_dir) / args.dataset / args.partition / args.algorithm
    output.mkdir(parents=True, exist_ok=True)
    run_label = f"seed{args.seed}"
    if args.dataset_split == "stratified":
        run_label += "_stratified"
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else output / f"{run_label}_resume.pt"
    )
    signature = _run_signature(args, rounds)

    history = []
    best_test_accuracy = -1.0
    best_test_round = None
    best_test_state = None
    completed_round = 0
    previous_elapsed = 0.0
    if args.resume_from:
        payload = _load_progress_checkpoint(Path(args.resume_from), signature)
        completed_round = int(payload["completed_round"])
        if completed_round < 0 or completed_round > rounds:
            raise ValueError("resume checkpoint has an invalid completed round")
        model.load_state_dict(payload["model_state"], strict=True)
        best_test_accuracy = float(payload["best_test_accuracy"])
        best_test_round = payload["best_test_round"]
        best_test_state = payload["best_test_state"]
        history = payload["history"]
        previous_elapsed = float(payload.get("elapsed_seconds", 0.0))
        if len(history) != completed_round:
            raise ValueError("resume checkpoint history length does not match its round")
        print(
            f"Resuming from {args.resume_from} after round {completed_round}/{rounds}",
            flush=True,
        )

    print(
        f"FedSLD reconstruction: dataset={args.dataset}, algorithm={args.algorithm}, "
        f"partition={args.partition}, clients={args.num_clients}, rounds={rounds}, "
        f"local_epochs={args.local_epochs}, split={args.dataset_split}, device={device}",
        flush=True,
    )
    print(f"Client training samples: {[len(values) for values in train_partitions]}", flush=True)
    if args.checkpoint_every:
        print(
            f"Saving a resume checkpoint every {args.checkpoint_every} rounds to "
            f"{checkpoint_path}",
            flush=True,
        )

    start = time.time()
    for server_round in range(completed_round + 1, rounds + 1):
        states = []
        losses = []
        sizes = []
        for client_id, indices in enumerate(train_partitions):
            state, loss = train_client(
                model,
                bundle.train,
                indices,
                args.algorithm,
                prior,
                device,
                args.batch_size,
                args.local_epochs,
                args.learning_rate,
                args.proximal_mu,
                seed=args.seed + server_round * 10_000 + client_id,
            )
            states.append(state)
            losses.append(loss)
            sizes.append(len(indices))
        model.load_state_dict(aggregate_states(states, sizes), strict=True)
        test_metrics = evaluate(model, combined_test_loader, device, bundle.num_classes)
        if test_metrics["accuracy"] > best_test_accuracy:
            best_test_accuracy = test_metrics["accuracy"]
            best_test_round = server_round
            best_test_state = _state_dict_cpu(model)
        weighted_loss = float(np.average(losses, weights=sizes))
        history.append(
            {
                "round": server_round,
                "train_loss": weighted_loss,
                "combined_test": test_metrics,
            }
        )
        print(
            f"Round {server_round}/{rounds}: loss={weighted_loss:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f}",
            flush=True,
        )
        if args.checkpoint_every and (
            server_round % args.checkpoint_every == 0 or server_round == rounds
        ):
            elapsed_so_far = previous_elapsed + time.time() - start
            _save_progress_checkpoint(
                checkpoint_path,
                signature,
                server_round,
                model,
                best_test_accuracy,
                best_test_round,
                best_test_state,
                history,
                elapsed_so_far,
            )
            print(f"Resume checkpoint: {checkpoint_path}", flush=True)

    elapsed = previous_elapsed + time.time() - start
    result = {
        "study": "FedSLD independent reconstruction",
        "source": "arXiv:2110.08378v1",
        "algorithm_interpretation": ALGORITHM_DESCRIPTIONS[args.algorithm],
        "reconstruction_hypothesis": args.algorithm in {
            "fedsld_weightnorm", "fedsld_inverse"
        } or args.dataset_split == "stratified",
        "paper_test_used_every_round": True,
        "metric": "BTA: highest combined test accuracy across rounds",
        "warning": "BTA selects the best test round and is not a locked-test estimate.",
        "config": {
            **vars(args),
            "rounds": rounds,
            "device_resolved": str(device),
            "checkpoint_path_resolved": str(checkpoint_path),
            "official_validation_merged_into_train": not args.exclude_official_val,
            "model": "PaperCNN conv20-conv50-fc500",
            "optimizer": "SGD without momentum or weight decay",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "dataset": {
            "train_samples": len(bundle.train),
            "test_samples": len(bundle.test),
            "split_mode": bundle.split_mode,
            "test_fraction": bundle.test_fraction,
            "split_seed": args.seed if bundle.split_mode == "stratified" else None,
            "num_classes": bundle.num_classes,
            "in_channels": bundle.in_channels,
        },
        "partition": {
            "train_sizes": [len(values) for values in train_partitions],
            "test_sizes": [len(values) for values in test_partitions],
            "train_hashes": [partition_hash(values) for values in train_partitions],
            "test_hashes": [partition_hash(values) for values in test_partitions],
            "client_class_counts": class_counts.tolist(),
            "federation_prior": prior.tolist(),
            "fraction_plan": plan.fractions.tolist(),
        },
        "best_test_accuracy": best_test_accuracy,
        "best_test_accuracy_round": best_test_round,
        "elapsed_seconds": elapsed,
        "history": history,
    }
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output / f"{run_label}_{timestamp}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    final_checkpoint = output / f"{run_label}_{timestamp}_final.pt"
    best_checkpoint = output / f"{run_label}_{timestamp}_best_bta.pt"
    torch.save(_state_dict_cpu(model), final_checkpoint)
    if best_test_state is None:
        raise RuntimeError("no best-test checkpoint was captured")
    torch.save(best_test_state, best_checkpoint)
    print(f"Result: {path}")
    print(f"Final checkpoint: {final_checkpoint}")
    print(f"Best-BTA checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()

