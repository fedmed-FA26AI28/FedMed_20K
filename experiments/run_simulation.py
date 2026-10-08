"""Run many isolated Flower clients on one machine using Ray.

Examples:
    python -m experiments.run_simulation --num_clients 5 --rounds 2 --local_epochs 1
    python -m experiments.run_simulation --num_clients 10 --rounds 50 --strategy fedavg
"""

import argparse
import datetime
import hashlib
import json
import math
import random
import time
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from flwr.common import Context, ndarrays_to_parameters

from algorithms import get_strategy
from client.client import FedMedAIClient
from datasets.medmnist_code import (
    build_transform, fit_train_normalization, get_bloodmnist_dataset,
)
from datasets.partition import (
    dirichlet_partition,
    get_client_dataloader,
    stratified_holdout_indices,
    stratified_subsample_indices,
    stratified_validation_partition,
)
from models.cnn import build_model, get_parameters
from monitoring.metrics import FLMetricsRecorder
from server.server import (
    _evaluate_final_global_model,
    _evaluate_metrics_aggregation_fn,
    _make_on_evaluate_config_fn,
    _make_on_fit_config_fn,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _partition_hash(indices) -> str:
    values = np.asarray(sorted(indices), dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _make_client_fn(
    train_partitions,
    val_partitions,
    num_classes: int,
    local_epochs: int,
    learning_rate: float,
    batch_size: int,
    client_cpus: float,
    client_gpus: float = 0.0,
    model_name: str = "legacy",
    seed: int = 42,
    size: int = 28,
    augment: bool = False,
    normalization=None,
):
    """Build ephemeral clients with assigned train and validation subsets."""

    def client_fn(context: Context) -> fl.client.Client:
        client_id = int(context.node_config["partition-id"])
        torch.set_num_threads(max(1, int(math.ceil(client_cpus))))
        from datasets.medmnist_code import load_simulation_datasets
        train_dataset, val_dataset = load_simulation_datasets(size, augment, normalization)

        client_train = get_client_dataloader(
            train_dataset,
            train_partitions[client_id],
            client_id=client_id,
            device_type="simulation",
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        client_val = get_client_dataloader(
            val_dataset,
            val_partitions[client_id],
            client_id=client_id,
            device_type="simulation",
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        return FedMedAIClient(
            client_id=client_id,
            train_loader=client_train,
            val_loader=client_val,
            num_classes=num_classes,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            device_type="simulation",
            save_local_metrics=False,
            server_address="127.0.0.1:1",
            model_name=model_name,
            seed=seed,
            use_gpu=client_gpus > 0,
        ).to_client()

    return client_fn


def run_simulation(args) -> Path:
    """Run one simulation and return its result directory."""
    if args.num_clients <= 0:
        raise ValueError("num_clients must be greater than zero")
    if not 0 < args.client_fraction <= 1:
        raise ValueError("client_fraction must be in (0, 1]")
    if args.client_gpus > 0 and not torch.cuda.is_available():
        raise ValueError("client_gpus > 0 was requested, but CUDA is unavailable")
    if args.ray_cpus is not None and args.ray_cpus < args.client_cpus:
        raise ValueError("ray_cpus must be at least client_cpus")
    if args.ray_object_store_mb is not None and args.ray_object_store_mb < 80:
        raise ValueError("ray_object_store_mb must be at least 80")
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if args.final_test and not args.locked_config:
        raise ValueError("final test requires --locked_config from a development run")
    if args.strategy in {"vacant_distill", "coverage_distill"}:
        if not math.isfinite(args.distill_mu) or args.distill_mu <= 0:
            raise ValueError("distill_mu must be finite and positive")
        if not math.isfinite(args.distill_temperature) or args.distill_temperature <= 0:
            raise ValueError("distill_temperature must be finite and positive")
        if args.distill_max_count < 0 or args.distill_warmup_rounds < 0:
            raise ValueError("distillation count and warmup must be non-negative")
        if args.rounds <= args.distill_warmup_rounds:
            raise ValueError("rounds must exceed distill_warmup_rounds")

    _seed_everything(args.seed)
    train_dataset, num_classes = get_bloodmnist_dataset(
        "train", download=True, size=args.size
    )

    eligible_train_indices = stratified_subsample_indices(
        train_dataset, num_samples=args.train_samples, seed=args.seed
    )
    normalization = None
    if args.normalization == "train":
        normalization = fit_train_normalization(train_dataset, eligible_train_indices)
    train_dataset.transform = build_transform(
        "train", augment=args.augment, normalization=normalization
    )
    val_dataset, _ = get_bloodmnist_dataset(
        "val", download=True, size=args.size, normalization=normalization
    )
    train_partitions = dirichlet_partition(
        train_dataset,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
        eligible_indices=eligible_train_indices,
    )
    local_val_indices, calibration_indices = stratified_holdout_indices(
        val_dataset,
        holdout_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    val_partitions = stratified_validation_partition(
        val_dataset,
        num_clients=args.num_clients,
        seed=args.seed,
        eligible_indices=local_val_indices,
    )
    if any(not partition for partition in train_partitions):
        raise ValueError(
            "At least one client has no training samples; use fewer clients, "
            "a larger alpha, or a different seed."
        )
    if any(not partition for partition in val_partitions):
        raise ValueError("There are more clients than usable validation samples")

    partition_sizes = [len(partition) for partition in train_partitions]
    print(
        f"Simulation: clients={args.num_clients}, rounds={args.rounds}, "
        f"strategy={args.strategy}, alpha={args.alpha}"
    )
    print(
        f"Train partitions: total={sum(partition_sizes)}, "
        f"min={min(partition_sizes)}, max={max(partition_sizes)}"
    )

    # Partition label reads may invoke random training transforms. Reset the
    # model-initialization seed here so centralized and FL arms start identically.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    initial_model = build_model(args.model, num_classes=num_classes)
    initial_parameters = ndarrays_to_parameters(get_parameters(initial_model))
    selected_clients = max(
        1, math.ceil(args.num_clients * args.client_fraction)
    )
    recorder = FLMetricsRecorder(
        strategy_name=args.strategy,
        proximal_mu=args.proximal_mu if args.strategy == "fedprox" else None,
    )
    strategy_kwargs = {
        "fraction_fit": args.client_fraction,
        "fraction_evaluate": args.client_fraction,
        "min_fit_clients": selected_clients,
        "min_evaluate_clients": selected_clients,
        "min_available_clients": args.num_clients,
        "initial_parameters": initial_parameters,
        "metrics_recorder": recorder,
        "on_fit_config_fn": _make_on_fit_config_fn(
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            proximal_mu=(
                args.proximal_mu if args.strategy == "fedprox" else 0.0
            ),
            logit_tau=args.logit_tau if args.strategy in {"coverage", "logit_only", "coverage_distill"} else 0.0,
            prior_smoothing=args.prior_smoothing,
            head_mu=args.head_mu if args.strategy in {"coverage", "head_only", "coverage_distill"} else 0.0,
            coverage_kappa=args.coverage_kappa,
            balanced_sampling=args.strategy == "balanced",
            distill_mu=args.distill_mu if args.strategy in {"vacant_distill", "coverage_distill"} else 0.0,
            distill_temperature=args.distill_temperature,
            distill_max_count=args.distill_max_count,
            distill_warmup_rounds=args.distill_warmup_rounds,
        ),
        "on_evaluate_config_fn": _make_on_evaluate_config_fn(),
        "evaluate_metrics_aggregation_fn": _evaluate_metrics_aggregation_fn,
        "early_stop_patience": args.early_stop_patience,
    }
    if args.strategy == "fedprox":
        strategy_kwargs["proximal_mu"] = args.proximal_mu
    strategy = get_strategy(args.strategy, **strategy_kwargs)
    strategy.latest_parameters = initial_parameters

    client_fn = _make_client_fn(
        train_partitions=train_partitions,
        val_partitions=val_partitions,
        num_classes=num_classes,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        client_cpus=args.client_cpus,
        client_gpus=args.client_gpus,
        model_name=args.model,
        seed=args.seed,
        size=args.size,
        augment=args.augment,
        normalization=normalization,
    )

    start_time = time.time()
    ray_init_args = {"ignore_reinit_error": True, "include_dashboard": False}
    if args.ray_cpus is not None:
        ray_init_args["num_cpus"] = args.ray_cpus
    if args.ray_object_store_mb is not None:
        ray_init_args["object_store_memory"] = args.ray_object_store_mb * 1024 * 1024
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": args.client_cpus,
            "num_gpus": args.client_gpus,
        },
        ray_init_args=ray_init_args,
    )
    if strategy.latest_parameters is initial_parameters:
        raise RuntimeError("No client model updates were aggregated; inspect Flower client failures")
    elapsed = time.time() - start_time

    from flwr.common import parameters_to_ndarrays
    from monitoring.research_validation import evaluate_validation
    final_arrays = parameters_to_ndarrays(strategy.latest_parameters)
    final_validation_metrics = evaluate_validation(
        final_arrays, args.model, val_dataset, local_val_indices,
        num_classes=num_classes, batch_size=args.batch_size,
    )
    final_test_metrics = None
    if args.final_test:
        # The test split is first loaded here, after all FL rounds and validation.
        final_test_metrics = _evaluate_final_global_model(
            final_arrays,
            num_classes=num_classes,
            calibration_indices=calibration_indices,
            conformal_alpha=args.conformal_alpha,
            model_name=args.model,
            size=args.size,
            normalization=normalization,
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = (
        Path(args.output_dir)
        / args.strategy
        / f"clients{args.num_clients}_alpha{args.alpha}_{timestamp}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "mode": "single_machine_simulation",
        "strategy": args.strategy,
        "model": args.model,
        "size": args.size,
        "augment": args.augment,
        "normalization": args.normalization,
        "normalization_stats": normalization,
        "final_test_enabled": args.final_test,
        "num_virtual_clients": args.num_clients,
        "clients_per_round": selected_clients,
        "client_fraction": args.client_fraction,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "alpha": args.alpha,
        "seed": args.seed,
        "train_samples_requested": args.train_samples,
        "train_samples_used": len(eligible_train_indices),
        "local_validation_samples": len(local_val_indices),
        "calibration_samples": len(calibration_indices),
        "calibration_fraction": args.calibration_fraction,
        "conformal_alpha": args.conformal_alpha,
        "coverage_objective": {
            "enabled": args.strategy in {"coverage", "logit_only", "head_only", "coverage_distill"},
            "logit_tau": args.logit_tau if args.strategy in {"coverage", "logit_only", "coverage_distill"} else 0.0,
            "prior_smoothing": args.prior_smoothing,
            "head_mu": args.head_mu if args.strategy in {"coverage", "head_only", "coverage_distill"} else 0.0,
            "coverage_kappa": args.coverage_kappa,
        },
        "vacant_distillation": {
            "enabled": args.strategy in {"vacant_distill", "coverage_distill"},
            "distill_mu": args.distill_mu if args.strategy in {"vacant_distill", "coverage_distill"} else 0.0,
            "temperature": args.distill_temperature,
            "max_count": args.distill_max_count,
            "warmup_rounds": args.distill_warmup_rounds,
        },
        "client_resources": {
            "num_cpus": args.client_cpus,
            "num_gpus": args.client_gpus,
        },
        "ray_cpus": args.ray_cpus,
        "ray_object_store_mb": args.ray_object_store_mb,
        "train_partition_sizes": partition_sizes,
        "train_partition_hashes": [
            _partition_hash(partition) for partition in train_partitions
        ],
        "local_validation_hash": _partition_hash(local_val_indices),
        "calibration_hash": _partition_hash(calibration_indices),
        "elapsed_seconds": round(elapsed, 3),
        "final_test_metrics": final_test_metrics,
        "final_validation_metrics": final_validation_metrics,
    }
    strategy.save_artifacts(str(result_dir), extra_metadata=metadata)
    if final_test_metrics is not None:
        with open(result_dir / "final_test_metrics.json", "w", encoding="utf-8") as file:
            json.dump(final_test_metrics, file, indent=2)
    with open(
        result_dir / "simulation_config.json", "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    if not args.final_test:
        run_spec = {key: value for key, value in vars(args).items()
                    if key not in {"final_test", "locked_config", "output_dir"}}
        with open(result_dir / "run_spec.json", "w", encoding="utf-8") as file:
            json.dump(run_spec, file, indent=2)
    print(f"Simulation complete in {elapsed:.1f}s | "
          f"mode={'final' if args.final_test else 'development'}")
    print(f"Results: {result_dir}")
    return result_dir


def main():
    parser = argparse.ArgumentParser(
        description="Simulate many federated clients on one machine"
    )
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--size", type=int, choices=[28, 64], default=28)
    parser.add_argument("--model", choices=["legacy", "tiny_cnn", "tiny_cnn_gn", "mobilenet_v3_small"], default="legacy")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--normalization", choices=["fixed", "train"], default="fixed")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=["fedavg", "fedprox", "fednova", "coverage", "balanced", "logit_only", "head_only", "vacant_distill", "coverage_distill"],
        default="fedavg",
    )
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--proximal_mu", type=float, default=0.1)
    parser.add_argument("--logit_tau", type=float, default=1.0)
    parser.add_argument("--prior_smoothing", type=float, default=1.0)
    parser.add_argument("--head_mu", type=float, default=0.01)
    parser.add_argument("--coverage_kappa", type=float, default=32.0)
    parser.add_argument("--distill_mu", type=float, default=0.1)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--distill_max_count", type=int, default=0)
    parser.add_argument("--distill_warmup_rounds", type=int, default=1)
    parser.add_argument(
        "--train_samples",
        type=int,
        default=None,
        help="Stratified training budget; default uses the full training split",
    )
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--conformal_alpha", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument(
        "--client_fraction",
        type=float,
        default=1.0,
        help="Fraction of virtual clients selected per round",
    )
    parser.add_argument("--client_cpus", type=float, default=1.0)
    parser.add_argument("--client_gpus", type=float, default=0.0)
    parser.add_argument("--ray_cpus", type=int, default=None,
                        help="Limit concurrent Ray actors; 1 with client_cpus=1 serializes virtual clients")
    parser.add_argument("--ray_object_store_mb", type=int, default=None,
                        help="Cap Ray object-store memory in MB; 256 is a low-memory starting point")
    parser.add_argument(
        "--output_dir", default="results/simulations"
    )
    parser.add_argument("--final_test", action="store_true")
    parser.add_argument("--locked_config", type=str, default=None)
    args = parser.parse_args()
    if args.locked_config:
        if not args.final_test:
            parser.error("--locked_config requires --final_test")
        with open(args.locked_config, encoding="utf-8") as file:
            locked_spec = json.load(file)
        for key, value in locked_spec.items():
            if not hasattr(args, key) or key in {"final_test", "locked_config", "output_dir"}:
                parser.error(f"invalid locked setting: {key}")
            setattr(args, key, value)
    run_simulation(args)


if __name__ == "__main__":
    main()
