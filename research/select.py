"""Lock hyperparameters from development validation results only."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from algorithms.research_methods import METHODS, TUNING_GRIDS, get_method
from research.run import REPO


def tuning_axes(method: str, args: argparse.Namespace) -> list[tuple[str, list[float]]]:
    return [("lr", args.lrs)] + [
        (name, getattr(args, TUNING_GRIDS[name]))
        for name in get_method(method).tuning_parameters
    ]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bloodmnist", "pathmnist", "organamnist"), required=True)
    parser.add_argument("--partition", choices=("practical", "dirichlet"), required=True)
    parser.add_argument("--model", "--norm", dest="norm",
                        choices=("baseline", "baseline_bn", "gn", "bn"), default="baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--clients", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=28)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--client-shift-strength", type=float, default=0.0)
    parser.add_argument("--slow-data", dest="fast_data", action="store_false")
    parser.set_defaults(fast_data=True)
    parser.add_argument("--coverage-kappa", type=float, default=10.0)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--head-lambdas", type=float, nargs="+", default=[0.0, 0.01, 0.1])
    parser.add_argument("--fedlc-taus", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--fedvls-lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output-root", type=Path, default=REPO / "results" / "research")
    parser.add_argument("--lock-file", type=Path)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("seeds must be unique")
    variants: dict[str, dict[str, dict[int, dict]]] = {method: {} for method in args.methods}
    for file in sorted(args.output_root.glob("*/development.json")):
        result = json.loads(file.read_text(encoding="utf-8"))
        cfg = result["configuration"]
        protocol = ("dataset", "partition", "rounds", "clients", "alpha", "epochs",
                    "batch_size", "resolution", "device", "client_shift_strength", "coverage_kappa",
                    "norm", "weight_decay", "optimizer", "momentum", "augment",
                    "fast_data")
        defaults = {"norm": "baseline", "weight_decay": 0.0, "augment": False,
                    "optimizer": "adam", "momentum": 0.9}
        if any(cfg.get(name, defaults.get(name)) != getattr(args, name) for name in protocol):
            continue
        if (cfg["method"] not in variants or cfg["seed"] not in args.seeds
                or cfg["limit_per_client"] or cfg["lr"] not in args.lrs):
            continue
        method = cfg["method"]
        defaults = {"gamma": 1.0, "head_lambda": 0.0,
                    "fedlc_tau": 0.5, "fedvls_lambda": 0.1}
        if any(cfg.get(name, defaults[name]) not in getattr(args, TUNING_GRIDS[name])
               for name in get_method(method).tuning_parameters):
            continue
        settings = {key: value for key, value in cfg.items()
                    if key not in ("seed", "device", "download", "output_root")}
        key = json.dumps(settings, sort_keys=True)
        checkpoint = file.parent / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"completed development run lacks checkpoint: {checkpoint}")
        record = {
            "seed": cfg["seed"], "validation_macro_f1": result["best_validation"]["macro_f1"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "development": str(file), "development_sha256": file_sha256(file),
            "partition_sha256": result["partition_sha256"],
        }
        variants[cfg["method"]].setdefault(key, {})[cfg["seed"]] = record
    selected = {}
    for method, choices in variants.items():
        complete = []
        completed_settings = set()
        axes = tuning_axes(method, args)
        for key, records in choices.items():
            if set(records) == set(args.seeds):
                ordered = [records[seed] for seed in args.seeds]
                complete.append((sum(r["validation_macro_f1"] for r in ordered) / len(ordered), key, ordered))
                settings = json.loads(key)
                completed_settings.add(tuple(settings.get(name) for name, _ in axes))
        expected_settings = set(itertools.product(*(values for _, values in axes)))
        missing = expected_settings - completed_settings
        if missing:
            raise RuntimeError(
                f"{method} has {len(missing)} incomplete validation settings for seeds "
                f"{args.seeds}; first missing: {dict(zip((name for name, _ in axes), sorted(missing)[0]))}"
            )
        score, key, ordered = max(complete, key=lambda item: (item[0], item[1]))
        selected[method] = {"settings": json.loads(key), "mean_validation_macro_f1": score,
                            "runs": ordered}
    for seed in args.seeds:
        hashes = {selection["runs"][args.seeds.index(seed)]["partition_sha256"]
                  for selection in selected.values()}
        if len(hashes) != 1:
            raise ValueError(f"selected methods use different training partitions for seed {seed}")
    lock = {"selection_metric": "mean best-round validation macro-F1", "dataset": args.dataset,
            "partition": args.partition, "seeds": args.seeds, "methods": selected,
            "protocol": {name: getattr(args, name) for name in protocol},
            "learning_rate_grid": args.lrs, "fedlc_tau_grid": args.fedlc_taus,
            "fedvls_lambda_grid": args.fedvls_lambdas,
            "gamma_grid": args.gammas, "head_lambda_grid": args.head_lambdas}
    path = args.lock_file or args.output_root / f"lock_{args.dataset}_{args.partition}_{args.norm}.json"
    if path.exists():
        raise FileExistsError(f"lock already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"Validation-only selection locked: {path}")


if __name__ == "__main__":
    main()
