"""Run matched FedAvg/SCR experiments on the existing Dirichlet(alpha=0.1) split.

Run from the repository root with ``python -m scripts.run_scr_dirichlet01``.
    Each dataset, seed, and strategy has its own output directory. A completed
80-round run is skipped; an interrupted run is retained and a retry gets a
new directory. No results are committed or pushed automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARTITION_MODE = "dirichlet_0.1"
NUM_CLIENTS = 12
ROUNDS = 80


def partition_path(dataset: str, seed: int) -> Path:
    return ROOT / "data" / "partitions" / (
        f"partition_{dataset}_{PARTITION_MODE}_localtest_seed{seed}_clients{NUM_CLIENTS}.json"
    )


def check_partition(path: Path, dataset: str, seed: int) -> None:
    """Reject a mismatched or training-empty split before starting workers."""
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = (dataset, PARTITION_MODE, seed, NUM_CLIENTS)
    actual = (
        data.get("dataset"), data.get("partition_mode"),
        data.get("seed"), data.get("num_clients"),
    )
    if actual != expected or not data.get("has_local_test"):
        raise ValueError(f"Partition metadata mismatch: {path}")
    clients = data["clients"]
    if len(clients) != NUM_CLIENTS:
        raise ValueError(f"Expected {NUM_CLIENTS} clients in {path}")
    train_sizes = [int(clients[str(i)]["num_samples"]) for i in range(NUM_CLIENTS)]
    test_sizes = [int(clients[str(i)]["test_num_samples"]) for i in range(NUM_CLIENTS)]
    if min(train_sizes) <= 0:
        raise ValueError(f"Empty training client in {path}; SCR cannot fit this split")
    if sum(train_sizes) != int(data["total_samples"]) or sum(test_sizes) != int(data["total_test_samples"]):
        raise ValueError(f"Partition sample totals disagree in {path}")
    empty_test = [i for i, count in enumerate(test_sizes) if count == 0]
    print(
        f"[Partition] {dataset} seed {seed}: train min={min(train_sizes)}, "
        f"test min={min(test_sizes)}, empty-test clients={empty_test}",
        flush=True,
    )


def check_existing_partition_hash(output_dir: Path, dataset: str, seed: int,
                                  partition: Path) -> None:
    """Keep the same sample assignment across arms and resumed sessions."""
    digest = hashlib.sha256(partition.read_bytes()).hexdigest()
    for strategy in ("fedavg", "scr"):
        for run_dir in output_dir.glob(f"{dataset}_s{seed}_{strategy}*"):
            manifest_path = run_dir / "experiment_manifest.json"
            if manifest_path.is_file():
                old = json.loads(manifest_path.read_text(encoding="utf-8"))
                if old.get("partition_sha256") != digest:
                    raise ValueError(
                        f"Partition changed since {run_dir}; use the original partition "
                        "or a new output directory"
                    )


def completed_run(path: Path, dataset: str, strategy: str, rounds: int) -> bool:
    summary_path = path / "simulation_results.json"
    result_path = path / dataset / PARTITION_MODE / strategy / "fl_results.json"
    if not summary_path.is_file() or not result_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return (
            summary.get("completed_runs") == 1
            and summary.get("failed_runs") == 0
            and int(result.get("metadata", {}).get("num_rounds_completed", 0)) == rounds
        )
    except (OSError, ValueError, TypeError):
        return False


def choose_run_dir(output_dir: Path, dataset: str, seed: int, strategy: str, rounds: int) -> tuple[Path, bool]:
    name = f"{dataset}_s{seed}_{strategy}"
    existing = [output_dir / name]
    existing.extend(sorted(output_dir.glob(f"{name}_retry[0-9]*")))
    for path in existing:
        if completed_run(path, dataset, strategy, rounds):
            return path, True
    if not existing[0].exists():
        return existing[0], False
    retry = 1
    while (output_dir / f"{name}_retry{retry}").exists():
        retry += 1
    return output_dir / f"{name}_retry{retry}", False


def simulation_command(dataset: str, seed: int, strategy: str, rounds: int, run_dir: Path) -> list[str]:
    return [
        sys.executable, "-u", "-m", "experiments.run_simulation",
        "--config", "configs/simulation.yaml",
        "--datasets", dataset,
        "--partitions", PARTITION_MODE,
        "--strategies", strategy,
        "--num_clients", str(NUM_CLIENTS),
        "--rounds", str(rounds),
        "--local_epochs", "5",
        "--batch_size", "256",
        "--learning_rate", "0.01",
        "--lr_mode", "fixed",
        "--model", "baseline_bn",
        "--optimizer", "sgd",
        "--momentum", "0.9",
        "--weight_decay", "0",
        "--seed", str(seed),
        "--no-augment", "--no-random_crop", "--no-random_flip",
        "--local_test", "--no-server_eval",
        "--early_stop_patience", "0",
        "--num_workers", "0", "--client_cpu_threads", "1",
        "--client_module", "client.client",
        "--scr_gamma", "1", "--scr_head_lambda", "0.01",
        "--no_timeout",
        "--output_dir", str(run_dir.parent), "--name", run_dir.name,
    ]


def write_manifest(run_dir: Path, dataset: str, seed: int, strategy: str,
                   partition: Path, command: list[str]) -> None:
    """Record the resolved command and exact partition used by this arm."""
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "dataset": dataset,
        "seed": seed,
        "strategy": strategy,
        "partition_mode": PARTITION_MODE,
        "partition_path": str(partition),
        "partition_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
        "num_clients": NUM_CLIENTS,
        "rounds": ROUNDS,
        "model": "baseline_bn",
        "weight_decay": 0,
        "command": command,
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def write_summary(output_dir: Path) -> None:
    """Collect completed jobs and draw the matched BTA curves after each run."""
    rows = []
    seen = set()
    name_pattern = re.compile(r"^(bloodmnist|organamnist|pathmnist)_s(\d+)_(fedavg|scr)(?:_retry\d+)?$")
    for run_dir in sorted(output_dir.iterdir()) if output_dir.exists() else []:
        match = name_pattern.fullmatch(run_dir.name)
        if match is None:
            continue
        dataset, seed_text, strategy = match.groups()
        key = (dataset, int(seed_text), strategy)
        if key in seen or not completed_run(run_dir, dataset, strategy, ROUNDS):
            continue
        seen.add(key)
        result_path = run_dir / dataset / PARTITION_MODE / strategy / "fl_results.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        summary = result["summary"]
        rows.append({
            "dataset": dataset,
            "seed": int(seed_text),
            "strategy": strategy,
            "bta": summary.get("bta"),
            "bmcta": summary.get("bmcta"),
            "wall_clock_seconds": summary.get("total_wall_clock_time_seconds"),
            "run_dir": str(run_dir),
        })
    if not rows:
        return
    rows.sort(key=lambda row: (row["dataset"], row["seed"], row["strategy"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dirichlet01_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({row["dataset"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        for strategy in ("fedavg", "scr"):
            points = sorted(
                (row for row in rows if row["dataset"] == dataset
                 and row["strategy"] == strategy and row["bta"] is not None),
                key=lambda row: row["seed"],
            )
            if points:
                axis.plot([row["seed"] for row in points],
                          [100 * float(row["bta"]) for row in points],
                          marker="o", label=strategy.upper())
        axis.set_title(dataset)
        axis.set_xlabel("Seed")
        axis.set_ylabel("BTA (%)")
        axis.set_xticks(seeds)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
    fig.suptitle("Dirichlet alpha=0.1 | baseline_bn | 12 clients")
    fig.tight_layout()
    plot_path = output_dir / "dirichlet01_bta_by_seed.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"[Summary] {csv_path} | {plot_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["bloodmnist", "organamnist", "pathmnist"],
                        default=["bloodmnist", "organamnist", "pathmnist"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)))
    parser.add_argument("--strategies", nargs="+", choices=["fedavg", "scr"], default=["fedavg", "scr"])
    parser.add_argument("--output_dir", type=Path, default=Path("results/scr_bn_dirichlet01_parallel"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    for dataset in args.datasets:
        for seed in args.seeds:
            path = partition_path(dataset, seed)
            if not path.is_file():
                command = [
                    sys.executable, "-u", "-m", "experiments.run_partition",
                    "--dataset", dataset, "--partition_mode", PARTITION_MODE,
                    "--num_clients", str(NUM_CLIENTS), "--seed", str(seed), "--local_test",
                ]
                print("[Generate partition]", " ".join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=ROOT, check=True)
            if path.is_file():
                check_partition(path, dataset, seed)
                check_existing_partition_hash(output_dir, dataset, seed, path)
            elif not args.dry_run:
                raise FileNotFoundError(f"Partition generation did not create {path}")

            for strategy in args.strategies:
                run_dir, complete = choose_run_dir(output_dir, dataset, seed, strategy, ROUNDS)
                if complete:
                    print(f"[Skip completed] {run_dir}", flush=True)
                    continue
                command = simulation_command(dataset, seed, strategy, ROUNDS, run_dir)
                print("[Run]", " ".join(command), flush=True)
                if args.dry_run:
                    continue
                write_manifest(run_dir, dataset, seed, strategy, path, command)
                subprocess.run(command, cwd=ROOT, check=True)
                if not completed_run(run_dir, dataset, strategy, ROUNDS):
                    raise RuntimeError(f"Run ended without {ROUNDS} completed rounds: {run_dir}")
                write_summary(output_dir)

    if not args.dry_run:
        write_summary(output_dir)


if __name__ == "__main__":
    main()
