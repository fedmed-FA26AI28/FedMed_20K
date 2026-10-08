"""List or execute the full-train validation-only comparison/ablation matrix."""

import argparse
import subprocess
import sys


def jobs(phase):
    if phase == "distillation":
        for seed in (42, 43, 44):
            for alpha in (0.1, 0.3):
                for strategy in ("fedavg", "coverage", "vacant_distill", "coverage_distill"):
                    yield {"strategy": strategy, "alpha": alpha, "seed": seed}
                # A simple calibration + near-uniform-head control. With this
                # count range and FP32, kappa=1e12 makes all row weights 1.
                yield {"strategy": "coverage", "alpha": alpha, "seed": seed,
                       "coverage_kappa": 1e12}
        return
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
    parser.add_argument("phase", choices=["main", "ablation", "distillation"])
    parser.add_argument("--execute", action="store_true", help="Run every job; default only prints commands")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--client_gpus", type=float, default=0.0)
    parser.add_argument("--client_cpus", type=float, default=None)
    parser.add_argument("--ray_cpus", type=int, default=None)
    parser.add_argument("--ray_object_store_mb", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Run one seed at a time for resumable notebook sessions")
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()
    for job in jobs(args.phase):
        if args.seed is not None and job["seed"] != args.seed:
            continue
        if args.alpha is not None and job["alpha"] != args.alpha:
            continue
        command = [sys.executable, "-m", "experiments.run_simulation",
                   "--num_clients", "10", "--rounds", str(args.rounds),
                   "--local_epochs", str(args.local_epochs),
                   "--model", "tiny_cnn", "--size", "64", "--augment",
                   "--strategy", job["strategy"], "--alpha", str(job["alpha"]),
                   "--seed", str(job["seed"]), "--client_gpus", str(args.client_gpus)]
        if "coverage_kappa" in job:
            command.extend(("--coverage_kappa", str(job["coverage_kappa"])))
        for flag, value in (
            ("--client_cpus", args.client_cpus),
            ("--ray_cpus", args.ray_cpus),
            ("--ray_object_store_mb", args.ray_object_store_mb),
            ("--output_dir", args.output_dir),
        ):
            if value is not None:
                command.extend((flag, str(value)))
        print(" ".join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
