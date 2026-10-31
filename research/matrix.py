"""Enumerate or run matched development experiments sequentially.

The default action prints the matrix. Pass --execute to train. Completed runs
are skipped by research.run, so the command can be restarted after a crash.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

from algorithms.research_methods import METHODS, get_method


def commands(args):
    for dataset, partition, norm, method, seed, lr in itertools.product(
        args.datasets, args.partitions, args.norms, args.methods, args.seeds, args.lrs,
    ):
        tuned = get_method(method).tuning_parameters
        gamma_values = args.gammas if "gamma" in tuned else [1.0]
        lambda_values = args.head_lambdas if "head_lambda" in tuned else [0.0]
        tau_values = args.fedlc_taus if "fedlc_tau" in tuned else [0.5]
        fedvls_values = args.fedvls_lambdas if "fedvls_lambda" in tuned else [0.1]
        for gamma, head_lambda, tau, fedvls_lambda in itertools.product(
            gamma_values, lambda_values, tau_values, fedvls_values,
        ):
            yield [
                sys.executable, "-m", "research.run", "--dataset", dataset,
                "--partition", partition, "--method", method, "--seed", str(seed),
                "--norm", norm, "--clients", str(args.clients),
                "--lr", str(lr), "--gamma", str(gamma),
                "--optimizer", args.optimizer, "--momentum", str(args.momentum),
                "--weight-decay", str(args.weight_decay),
                "--head-lambda", str(head_lambda), "--rounds", str(args.rounds),
                "--coverage-kappa", str(args.coverage_kappa),
                "--fedlc-tau", str(tau), "--fedvls-lambda", str(fedvls_lambda),
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                "--resolution", str(args.resolution), "--device", args.device,
                "--client-shift-strength", str(args.client_shift_strength),
                "--output-root", str(args.output_root),
            ] + (["--download"] if args.download else []) + (["--slow-data"] if args.slow_data else []) + (["--augment"] if args.augment else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("bloodmnist", "pathmnist", "organamnist"),
                        default=["bloodmnist", "pathmnist", "organamnist"])
    parser.add_argument("--partitions", nargs="+", choices=("practical", "dirichlet"),
                        default=["practical", "dirichlet"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--models", "--norms", dest="norms", nargs="+",
                        choices=("baseline", "baseline_bn", "gn", "bn"),
                        default=["baseline", "baseline_bn"])
    parser.add_argument("--clients", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--head-lambdas", nargs="+", type=float, default=[0.0, 0.01, 0.1])
    parser.add_argument("--coverage-kappa", type=float, default=10.0)
    parser.add_argument("--fedlc-taus", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    parser.add_argument("--fedvls-lambdas", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=28)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "research")
    parser.add_argument("--client-shift-strength", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--slow-data", action="store_true")
    parser.add_argument("--list-methods", action="store_true",
                        help="List registered research algorithms and their tuning axes")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.list_methods:
        for name in METHODS:
            axes = ("lr",) + get_method(name).tuning_parameters
            print(f"{name}: tune {', '.join(axes)}")
        return
    planned = list(commands(args))
    print(f"Planned sequential development runs: {len(planned)}", flush=True)
    if not args.execute:
        for command in planned[:10]:
            print(subprocess.list2cmdline(command))
        if len(planned) > 10:
            print(f"... {len(planned) - 10} additional runs. Add --execute to train.")
        return
    try:
        for number, command in enumerate(planned, 1):
            print(f"[{number}/{len(planned)}] {subprocess.list2cmdline(command)}", flush=True)
            subprocess.run(command, check=True)
    finally:
        # A stopped matrix still gets a plot of completed development runs.
        from research.plots import plot_matrix
        from research.run import arguments, run_name
        signature = hashlib.sha256(json.dumps([command[3:] for command in planned],
                                            sort_keys=True).encode()).hexdigest()[:10]
        files = [args.output_root / run_name(arguments(command[3:])) / "development.json"
                 for command in planned]
        for output in plot_matrix(files, args.output_root, signature):
            print(f"Validation comparison plot: {output}", flush=True)


if __name__ == "__main__":
    main()
