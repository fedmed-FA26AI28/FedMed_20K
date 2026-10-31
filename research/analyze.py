"""Summarize a locked final test evaluation with paired-seed differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.select import file_sha256


METRICS = ("macro_f1", "accuracy", "worst_class_recall", "nll", "brier", "ece_15")


def paired_summary(values: dict[str, dict[int, float]], reference: str,
                   seed: int = 2026) -> dict[str, dict]:
    """Bootstrap paired seed differences; no claim of independent patient sites."""
    if reference not in values:
        raise ValueError(f"reference method {reference!r} is absent")
    rng = np.random.default_rng(seed)
    reference_values = values[reference]
    result = {}
    for method, scores in values.items():
        if method == reference:
            continue
        seeds = sorted(set(reference_values) & set(scores))
        if set(seeds) != set(reference_values) or set(seeds) != set(scores):
            raise ValueError(f"methods {method} and {reference} have unmatched seeds")
        differences = np.array([scores[item] - reference_values[item] for item in seeds])
        draws = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
        result[method] = {
            "reference": reference, "seeds": seeds,
            "per_seed_difference": differences.tolist(),
            "mean_difference": float(differences.mean()),
            "bootstrap_95_percentile_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    return result


def build_report(lock: dict, final: dict) -> dict:
    expected_methods = set(lock["methods"])
    if set(final["methods"]) != expected_methods:
        raise ValueError("final result has a different method set from the lock")
    summary = {}
    macro_f1_by_seed = {}
    for method, selection in lock["methods"].items():
        rows = final["methods"][method]
        expected = set(lock["seeds"])
        if {row["seed"] for row in rows} != expected or len(rows) != len(expected):
            raise ValueError(f"{method} is missing one or more locked test evaluations")
        metrics = {name: np.array([row["metrics"][name] for row in rows]) for name in METRICS}
        class_recall = np.array([row["metrics"]["class_recall"] for row in rows])
        runtime, communication = [], []
        for run in selection["runs"]:
            development_file = Path(run["development"])
            if (run.get("development_sha256")
                    and file_sha256(development_file) != run["development_sha256"]):
                raise ValueError(f"locked development result changed: {development_file}")
            development = json.loads(development_file.read_text(encoding="utf-8"))
            # Sum recorded round durations so a resumed run has the full time.
            runtime.append(sum(item["seconds"] for item in development["history"]))
            communication.append(development["estimated_communication_bytes"])
        summary[method] = {
            "metrics": {name: {"mean": float(values.mean()),
                                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
                        for name, values in metrics.items()},
            "class_recall_mean": class_recall.mean(axis=0).tolist(),
            "class_recall_sd": class_recall.std(axis=0, ddof=1).tolist()
            if len(rows) > 1 else [0.0] * class_recall.shape[1],
            "mean_train_seconds": float(np.mean(runtime)),
            "mean_communication_bytes": float(np.mean(communication)),
        }
        macro_f1_by_seed[method] = {row["seed"]: row["metrics"]["macro_f1"] for row in rows}
    paired = {}
    for reference in ("fedavg", "pooled_uniform_head", "fedvls"):
        if reference in macro_f1_by_seed:
            paired[reference] = paired_summary(macro_f1_by_seed, reference)
    return {"dataset": lock["dataset"], "partition": lock["partition"],
            "seeds": lock["seeds"], "summary": summary,
            "paired_macro_f1": paired,
            "note": "Bootstrap intervals describe paired simulated seeds, not hospitals or patients."}


def render_markdown(report: dict) -> str:
    lines = [f"# {report['dataset']} / {report['partition']} locked test results", "",
             "Validation selected all settings and rounds before these test metrics were read.", "",
             "| Method | Macro-F1 mean ± SD | Accuracy | Worst recall | ECE-15 | Train hours | Communication MiB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, row in report["summary"].items():
        metrics = row["metrics"]
        lines.append(f"| {method} | {metrics['macro_f1']['mean']:.4f} ± {metrics['macro_f1']['sd']:.4f} "
                     f"| {metrics['accuracy']['mean']:.4f} | {metrics['worst_class_recall']['mean']:.4f} "
                     f"| {metrics['ece_15']['mean']:.4f} | {row['mean_train_seconds']/3600:.2f} "
                     f"| {row['mean_communication_bytes']/1048576:.1f} |")
    lines += ["", "## Paired macro-F1 differences", "",
              "Intervals are percentile bootstrap intervals over paired seeds; five seeds give limited precision.", ""]
    for reference, contrasts in report["paired_macro_f1"].items():
        lines += [f"### Relative to {reference}", "", "| Method | Mean difference | 95% bootstrap interval |",
                  "|---|---:|---:|"]
        for method, result in contrasts.items():
            lo, hi = result["bootstrap_95_percentile_interval"]
            lines.append(f"| {method} | {result['mean_difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
        lines.append("")
    lines += ["## Mean class recall", ""]
    for method, row in report["summary"].items():
        lines.append(f"- {method}: " + ", ".join(f"c{index}={value:.3f}"
                     for index, value in enumerate(row["class_recall_mean"])))
    lines += ["", report["note"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock_file.read_text(encoding="utf-8"))
    final_path = args.lock_file.with_name(args.lock_file.stem + "_test.json")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if final.get("lock_sha256") != file_sha256(args.lock_file):
        raise ValueError("validation lock changed after final test evaluation")
    report = build_report(lock, final)
    base = args.lock_file.with_name(args.lock_file.stem + "_analysis")
    base.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    base.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Analysis: {base.with_suffix('.md')}")


if __name__ == "__main__":
    main()
