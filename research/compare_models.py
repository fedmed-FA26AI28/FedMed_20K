"""Compare two validation-locked architectures on paired test seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.select import file_sha256
from algorithms.research_methods import METHODS as AVAILABLE_METHODS


REQUIRED = ("fedavg", "support_calibrated")
METRICS = ("macro_f1", "accuracy", "worst_class_recall")
AUX_METRICS = ("balanced_accuracy", "nll", "brier", "ece_15")


def load_arm(path: Path) -> tuple[dict, dict]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    test_path = path.with_name(path.stem + "_test.json")
    test = json.loads(test_path.read_text(encoding="utf-8"))
    if test.get("lock_sha256") != file_sha256(path):
        raise ValueError(f"test results do not match lock: {path}")
    if not set(REQUIRED).issubset(lock["methods"]):
        raise ValueError(f"lock lacks FedAvg or SCR: {path}")
    return lock, test


def paired(left: dict[int, float], right: dict[int, float]) -> dict:
    if not left or set(left) != set(right):
        raise ValueError("paired comparison requires identical, nonempty seed sets")
    seeds = sorted(left)
    differences = np.array([left[seed] - right[seed] for seed in seeds])
    rng = np.random.default_rng(2026)
    draws = rng.choice(differences, size=(10000, len(seeds)), replace=True).mean(axis=1)
    return {"per_seed": dict(zip(map(str, seeds), differences.tolist())),
            "mean": float(differences.mean()),
            "bootstrap_95_interval": np.quantile(draws, [.025, .975]).tolist()}


def compare(first_lock: dict, first_test: dict,
            second_lock: dict, second_test: dict) -> dict:
    for key in ("dataset", "partition", "seeds"):
        if first_lock[key] != second_lock[key]:
            raise ValueError(f"model arms differ in {key}")
    first, second = (lock["protocol"]["norm"] for lock in (first_lock, second_lock))
    if first == second:
        raise ValueError("model arms must use different architectures")
    if (first, second) != ("baseline", "baseline_bn"):
        raise ValueError("expected baseline first and baseline_bn second")
    if set(first_lock["methods"]) != set(second_lock["methods"]):
        raise ValueError("model arms must contain the same methods")
    protocol_keys = set(first_lock["protocol"]) | set(second_lock["protocol"])
    for key in protocol_keys - {"norm"}:
        if first_lock["protocol"].get(key) != second_lock["protocol"].get(key):
            raise ValueError(f"model arms differ in protocol field {key}")
    if first_lock["protocol"].get("clients") != 12:
        raise ValueError("expected 12 simulated clients")
    if first_lock["protocol"].get("weight_decay") != 0:
        raise ValueError("expected zero weight decay")
    methods = [method for method in AVAILABLE_METHODS
               if method in first_lock["methods"] and method in second_lock["methods"]]
    seeds = set(first_lock["seeds"])
    scores: dict[tuple[str, str, str], dict[int, float]] = {}
    auxiliary: dict[str, dict] = {}
    hashes: dict[int, set[str]] = {seed: set() for seed in seeds}
    for model, lock, test in ((first, first_lock, first_test),
                              (second, second_lock, second_test)):
        for method in methods:
            locked = lock["methods"][method]["runs"]
            rows = test["methods"].get(method, [])
            if (len(locked) != len(seeds) or len(rows) != len(seeds)
                    or {item["seed"] for item in locked} != seeds
                    or {item["seed"] for item in rows} != seeds):
                raise ValueError(f"incomplete or duplicate seed for {model}/{method}")
            for item in locked:
                hashes[item["seed"]].add(item["partition_sha256"])
            for metric in METRICS:
                scores[(model, method, metric)] = {
                    row["seed"]: row["metrics"][metric] for row in rows}
            extras = {}
            for metric in AUX_METRICS:
                if all(metric in row["metrics"] for row in rows):
                    extras[metric] = float(np.mean([row["metrics"][metric] for row in rows]))
            if all("class_recall" in row["metrics"] for row in rows):
                extras["class_recall"] = np.mean(
                    [row["metrics"]["class_recall"] for row in rows], axis=0).tolist()
            development = [json.loads(Path(item["development"]).read_text(encoding="utf-8"))
                           for item in locked if "development" in item]
            if len(development) == len(seeds):
                for source, destination in (("wall_seconds", "mean_training_seconds"),
                                            ("estimated_communication_bytes", "mean_communication_bytes")):
                    if all(source in item for item in development):
                        extras[destination] = float(np.mean([item[source] for item in development]))
            auxiliary[f"{model}/{method}"] = extras
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError("model and method arms used different train partitions")
    report = {"dataset": first_lock["dataset"], "partition": first_lock["partition"],
              "models": [first, second], "seeds": sorted(seeds), "methods": methods,
              "means": {}, "standard_deviation": {}, "auxiliary": auxiliary,
              "paired_differences": {}}
    for model in (first, second):
        for method in methods:
            report["means"][f"{model}/{method}"] = {
                metric: float(np.mean(list(scores[(model, method, metric)].values())))
                for metric in METRICS}
            report["standard_deviation"][f"{model}/{method}"] = {
                metric: float(np.std(list(scores[(model, method, metric)].values()), ddof=1))
                if len(seeds) > 1 else 0.0 for metric in METRICS}
    for metric in METRICS:
        contrasts = {
            f"scr_minus_fedavg/{model}": paired(
                scores[(model, "support_calibrated", metric)],
                scores[(model, "fedavg", metric)]) for model in (first, second)}
        contrasts[f"{second}_minus_{first}/scr"] = paired(
            scores[(second, "support_calibrated", metric)],
            scores[(first, "support_calibrated", metric)])
        first_gain = contrasts[f"scr_minus_fedavg/{first}"]["per_seed"]
        second_gain = contrasts[f"scr_minus_fedavg/{second}"]["per_seed"]
        contrasts["method_by_model_interaction"] = paired(
            {int(seed): value for seed, value in second_gain.items()},
            {int(seed): value for seed, value in first_gain.items()})
        for comparator in methods:
            if comparator in REQUIRED:
                continue
            for model in (first, second):
                contrasts[f"scr_minus_{comparator}/{model}"] = paired(
                    scores[(model, "support_calibrated", metric)],
                    scores[(model, comparator, metric)])
        report["paired_differences"][metric] = contrasts
    return report


def markdown(report: dict) -> str:
    first, second = report["models"]
    lines = [f"# {report['dataset']} {report['partition']}: {first} vs {second}", "",
             "Settings and checkpoints were selected on validation before test evaluation.", "",
             "| Model | Method | Macro-F1 | Accuracy | Worst recall |",
             "|---|---|---:|---:|---:|"]
    for model in (first, second):
        for method in report["methods"]:
            values = report["means"][f"{model}/{method}"]
            lines.append(f"| {model} | {method} | " +
                         " | ".join(f"{values[name]:.4f}" for name in METRICS) + " |")
    lines += ["", "## Paired mean differences", "",
              "Positive values favor the first named method or model.", ""]
    for metric, contrasts in report["paired_differences"].items():
        lines.append(f"### {metric}")
        lines.append("")
        for name, values in contrasts.items():
            low, high = values["bootstrap_95_interval"]
            lines.append(f"- {name}: {values['mean']:+.4f} (paired bootstrap 95%: {low:+.4f}, {high:+.4f})")
        lines.append("")
    if any(report["auxiliary"].values()):
        lines += ["## Additional arm metrics", ""]
        for arm, values in report["auxiliary"].items():
            lines.append(f"- {arm}: " + ", ".join(
                f"{key}={value}" for key, value in values.items()))
        lines.append("")
    lines.append("Intervals describe simulated partition seeds, not independent hospitals.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-lock", type=Path, required=True)
    parser.add_argument("--baseline-bn-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(*load_arm(args.baseline_lock), *load_arm(args.baseline_bn_lock))
    output = args.output or args.baseline_lock.with_name(
        args.baseline_lock.stem + "_vs_baseline_bn.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown(report), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    from research.plots import plot_test_report
    figure = plot_test_report(report, output.with_suffix(".png"))
    print(f"Comparison: {output}")
    print(f"Test comparison plot: {figure}")


if __name__ == "__main__":
    main()
