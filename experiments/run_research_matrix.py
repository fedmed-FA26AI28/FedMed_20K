"""List or execute the full-train validation-only comparison/ablation matrix."""

import argparse
import subprocess
import sys


def jobs(phase):
    strategies = {
        "main": ("fedavg", "fedprox", "balanced", "coverage"),
        "ablation": ("logit_only", "head_only"),
    }[phase]
    alphas = (0.1, 0.3) if phase == "main" else (0.1,)
    for seed in (42, 43, 44):
        for alpha in alphas:
            for strategy in strategies:
                yield {"strategy": strategy, "alpha": alpha, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["main", "ablation"])
    parser.add_argument("--execute", action="store_true", help="Run every job; default only prints commands")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--client_gpus", type=float, default=0.0)
    args = parser.parse_args()
    for job in jobs(args.phase):
        command = [sys.executable, "-m", "experiments.run_simulation",
                   "--num_clients", "10", "--rounds", str(args.rounds),
                   "--local_epochs", str(args.local_epochs),
                   "--model", "tiny_cnn", "--size", "64", "--augment",
                   "--strategy", job["strategy"], "--alpha", str(job["alpha"]),
                   "--seed", str(job["seed"]), "--client_gpus", str(args.client_gpus)]
        print(" ".join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
