"""Compare validation-locked GN and BN test runs as a paired 2x2 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.select import file_sha256


METHODS = ("fedavg", "support_calibrated")
METRICS = ("macro_f1", "accuracy", "worst_class_recall")


def read_arm(lock_path: Path, expected_norm: str) -> tuple[dict, dict]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result_path = lock_path.with_name(lock_path.stem + "_test.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("lock_sha256") != file_sha256(lock_path):
        raise ValueError(f"test result does not match validation lock: {lock_path}")
    if lock["protocol"].get("norm", "gn") != expected_norm:
        raise ValueError(f"expected {expected_norm} lock: {lock_path}")
    if not set(METHODS).issubset(lock["methods"]):
        raise ValueError(f"lock must contain {METHODS}: {lock_path}")
    return lock, result


def paired_difference(left: dict[int, float], right: dict[int, float]) -> dict:
    if set(left) != set(right) or not left:
        raise ValueError("paired comparison requires identical, nonempty seeds")
    seeds = sorted(left)
    differences = np.asarray([left[seed] - right[seed] for seed in seeds])
    rng = np.random.default_rng(2026)
    draws = rng.choice(differences, size=(10000, len(seeds)), replace=True).mean(axis=1)
    return {"seeds": seeds, "per_seed_difference": differences.tolist(),
            "mean_difference": float(differences.mean()),
            "bootstrap_95_percentile_interval": np.quantile(draws, [0.025, 0.975]).tolist()}


def compare(gn_lock: dict, gn_result: dict, bn_lock: dict, bn_result: dict) -> dict:
    for key in ("dataset", "partition", "seeds"):
        if gn_lock[key] != bn_lock[key]:
            raise ValueError(f"GN and BN locks have different {key}")
    for key in gn_lock["protocol"]:
        if key != "norm" and gn_lock["protocol"][key] != bn_lock["protocol"].get(key):
            raise ValueError(f"GN and BN protocols have different {key}")
    expected = set(gn_lock["seeds"])
    scores = {}
    for norm, lock, result in (("gn", gn_lock, gn_result), ("bn", bn_lock, bn_result)):
        for method in METHODS:
            rows = result["methods"][method]
            if len(rows) != len(expected) or {row["seed"] for row in rows} != expected:
                raise ValueError(f"{norm}/{method} has missing or duplicate test seeds")
            locked = {item["seed"]: item["partition_sha256"]
                      for item in lock["methods"][method]["runs"]}
            if set(locked) != expected:
                raise ValueError(f"{norm}/{method} has incomplete partition hashes")
            scores[(norm, method)] = {
                metric: {row["seed"]: row["metrics"][metric] for row in rows}
                for metric in METRICS
            }
    for seed in expected:
        hashes = {next(item["partition_sha256"] for item in lock["methods"][method]["runs"]
                       if item["seed"] == seed)
                  for lock in (gn_lock, bn_lock) for method in METHODS}
        if len(hashes) != 1:
            raise ValueError(f"GN and BN use different partitions for seed {seed}")

    table = {}
    for (norm, method), metrics in scores.items():
        table[f"{norm}_{method}"] = {
            metric: {"mean": float(np.mean(list(values.values()))),
                     "sd": float(np.std(list(values.values()), ddof=1)) if len(values) > 1 else 0.0}
            for metric, values in metrics.items()
        }
    contrasts = {}
    for metric in METRICS:
        gn_gain = scores[("gn", "support_calibrated")][metric]
        gn_base = scores[("gn", "fedavg")][metric]
        bn_gain = scores[("bn", "support_calibrated")][metric]
        bn_base = scores[("bn", "fedavg")][metric]
        interaction = {seed: (bn_gain[seed] - bn_base[seed]) -
                       (gn_gain[seed] - gn_base[seed]) for seed in expected}
        contrasts[metric] = {
            "scr_vs_fedavg_gn": paired_difference(gn_gain, gn_base),
            "scr_vs_fedavg_bn": paired_difference(bn_gain, bn_base),
            "bn_vs_gn_under_scr": paired_difference(bn_gain, gn_gain),
            "normalization_by_method_interaction": paired_difference(
                interaction, {seed: 0.0 for seed in expected}),
        }
    return {"dataset": gn_lock["dataset"], "partition": gn_lock["partition"],
            "seeds": gn_lock["seeds"], "arms": table, "paired_differences": contrasts,
            "note": "Intervals describe paired simulation seeds, not independent hospitals."}


def render_markdown(report: dict) -> str:
    lines = [f"# {report['dataset']} / {report['partition']}: GN versus BN", "",
             "All checkpoints and settings were selected by validation before test evaluation.", "",
             "| Normalization | Method | Macro-F1 | Accuracy | Worst-class recall |",
             "|---|---|---:|---:|---:|"]
    for norm in ("gn", "bn"):
        for method in METHODS:
            metrics = report["arms"][f"{norm}_{method}"]
            cells = [f"{metrics[name]['mean']:.4f} ± {metrics[name]['sd']:.4f}"
                     for name in METRICS]
            lines.append(f"| {norm.upper()} | {method} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences (first minus second)", "",
              "| Metric | SCR − FedAvg (GN) | SCR − FedAvg (BN) | BN − GN (SCR) | Interaction |",
              "|---|---:|---:|---:|---:|"]
    names = ("scr_vs_fedavg_gn", "scr_vs_fedavg_bn", "bn_vs_gn_under_scr",
             "normalization_by_method_interaction")
    for metric in METRICS:
        values = report["paired_differences"][metric]
        lines.append(f"| {metric} | " + " | ".join(
            f"{values[name]['mean_difference']:+.4f}"
            for name in names) + " |")
    lines += ["", "Interaction = (SCR − FedAvg under BN) − (SCR − FedAvg under GN).",
              "Full per-seed differences and bootstrap intervals are in the JSON report.",
              report["note"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gn-lock", type=Path, required=True)
    parser.add_argument("--bn-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(*read_arm(args.gn_lock, "gn"), *read_arm(args.bn_lock, "bn"))
    output = args.output or args.gn_lock.with_name(args.gn_lock.stem + "_vs_bn.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Comparison: {output}")


if __name__ == "__main__":
    main()
