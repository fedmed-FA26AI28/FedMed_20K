"""Headless validation and locked-test plots for the SCR experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_run(result: dict, output: Path) -> Path:
    """Plot a completed run using validation history only."""
    history = result.get("history", [])
    if not history:
        raise ValueError("completed development result has no round history")
    rounds = [item["round"] for item in history]
    series = (
        ("macro_f1", "Validation macro-F1"),
        ("accuracy", "Validation accuracy"),
        ("worst_class_recall", "Validation worst-class recall"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, (key, label) in zip(axes.flat, series):
        axis.plot(rounds, [item["validation"][key] for item in history],
                  color="#1849a9", marker="o" if len(rounds) <= 12 else None,
                  markersize=3, linewidth=1.8)
        axis.axvline(result["best_validation_round"], color="#d66a00",
                     linestyle="--", linewidth=1.2, label="Selected by validation")
        axis.set(title=label, xlabel="Communication round", ylabel="Score", ylim=(0, 1.02))
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    loss_axis = axes.flat[3]
    loss_axis.plot(rounds, [item["mean_client_loss"] for item in history],
                   color="#6b399c", linewidth=1.8,
                   marker="o" if len(rounds) <= 12 else None, markersize=3)
    loss_axis.set(title="Mean client training loss", xlabel="Communication round",
                  ylabel="Loss")
    loss_axis.grid(alpha=0.25)
    cfg = result["configuration"]
    fig.suptitle(f"{cfg['dataset']} / {cfg['partition']} / {cfg['norm']} / "
                 f"{cfg['method']} / seed {cfg['seed']}  |  validation only", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    return output


def _read_development(paths: list[Path]) -> list[tuple[Path, dict]]:
    completed = []
    for path in paths:
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("phase") != "development" or not result.get("history"):
            continue
        completed.append((path, result))
    return completed


def plot_matrix(paths: list[Path], output_root: Path, signature: str | None = None,
                *, include_smoke: bool = False) -> list[Path]:
    """Plot only completed development runs; never inspect locked test data."""
    completed = _read_development(paths)
    if not include_smoke:
        completed = [(path, result) for path, result in completed
                     if not result["configuration"].get("limit_per_client", 0)]
    cohorts: dict[tuple[str, str], list[tuple[Path, dict]]] = defaultdict(list)
    for path, result in completed:
        cfg = result["configuration"]
        cohorts[(cfg["dataset"], cfg["partition"])].append((path, result))
    outputs = []
    for (dataset, partition), cohort in sorted(cohorts.items()):
        digest = signature or hashlib.sha256(
            "\n".join(sorted(str(path) for path, _ in cohort)).encode()).hexdigest()[:10]
        folder = output_root / "plots" / f"{dataset}_{partition}_{digest}"
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for path, result in cohort:
            cfg, best = result["configuration"], result["best_validation"]
            rows.append({
                "model": cfg["norm"], "method": cfg["method"], "seed": cfg["seed"],
                "lr": cfg["lr"], "gamma": cfg["gamma"],
                "head_lambda": cfg["head_lambda"],
                "selected_round": result["best_validation_round"],
                "validation_macro_f1": best["macro_f1"],
                "validation_accuracy": best["accuracy"],
                "validation_worst_class_recall": best["worst_class_recall"],
                "development_file": str(path),
            })
        rows.sort(key=lambda row: (row["model"], row["method"], row["seed"],
                                   row["lr"], row["gamma"], row["head_lambda"]))
        summary_path = folder / "validation_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        groups = defaultdict(list)
        for row in rows:
            groups[(row["model"], row["method"])].append(row)
        ordered = sorted(groups, key=lambda key: (key[0] != "baseline", key[0], key[1]))
        colors = {"fedavg": "#4e79a7", "pooled_uniform_head": "#f28e2b",
                  "support_calibrated": "#59a14f"}
        fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.5),
                                         constrained_layout=True)
        if len(rows) <= 12:
            for path, result in cohort:
                cfg = result["configuration"]
                label = f"{cfg['norm']} / {cfg['method']} / s{cfg['seed']}"
                left.plot([item["round"] for item in result["history"]],
                          [item["validation"]["macro_f1"] for item in result["history"]],
                          label=label, linewidth=1.6,
                          color=colors.get(cfg["method"]),
                          marker="o" if len(result["history"]) <= 12 else None,
                          markersize=4,
                          linestyle="--" if cfg["norm"] == "baseline_bn" else "-")
            left.set(title="Validation macro-F1 by round", xlabel="Round", ylabel="Macro-F1",
                     ylim=(0, 1.02))
            left.legend(fontsize=7, loc="best")
        else:
            for index, group in enumerate(ordered):
                values = [row["validation_macro_f1"] for row in groups[group]]
                jitter = np.linspace(-0.14, 0.14, len(values)) if len(values) > 1 else [0]
                left.scatter(np.array(jitter) + index, values, alpha=0.65, s=24,
                             color=colors.get(group[1], "#777777"))
            left.set(title="All completed settings (not selected)",
                     xlabel="Model / method", ylabel="Best validation macro-F1",
                     xticks=range(len(ordered)),
                     xticklabels=[f"{model}\n{method}" for model, method in ordered],
                     ylim=(0, 1.02))
            left.tick_params(axis="x", labelrotation=20, labelsize=8)
        left.grid(axis="y", alpha=0.25)
        if len(rows) <= 12 and all(len(groups[group]) == 1 for group in ordered):
            right.bar(range(len(ordered)),
                      [groups[group][0]["validation_macro_f1"] for group in ordered],
                      color=[colors.get(group[1], "#777777") for group in ordered])
            right.set(title="Best validation macro-F1 per arm", ylabel="Macro-F1",
                      ylim=(0, 1.02))
        else:
            right.bar(range(len(ordered)), [len(groups[group]) for group in ordered],
                      color=[colors.get(group[1], "#777777") for group in ordered])
            right.set(title="Completed development runs per arm", ylabel="Run count")
        right.set_xticks(range(len(ordered)),
                         [f"{model}\n{method}" for model, method in ordered],
                         rotation=20, fontsize=8)
        right.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{dataset} / {partition}: {len(rows)} completed development runs\n"
                     "Validation scores only; choose settings before final test", fontsize=12)
        output = folder / "validation_comparison.png"
        fig.savefig(output, dpi=160, facecolor="white")
        plt.close(fig)
        outputs.append(output)
    return outputs


def plot_test_report(report: dict, output: Path) -> Path:
    """Plot the already finalized, validation-locked test report."""
    methods, models = report["methods"], report["models"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {models[0]: "#4e79a7", models[1]: "#f28e2b"}
    for axis, (metric, title) in zip(axes.flat[:3], (
        ("macro_f1", "Test macro-F1"), ("accuracy", "Test accuracy"),
        ("worst_class_recall", "Test worst-class recall"),
    )):
        centers = np.arange(len(methods), dtype=float)
        for offset, model in zip((-.19, .19), models):
            means = [report["means"][f"{model}/{method}"][metric]
                     for method in methods]
            sds = [report.get("standard_deviation", {}).get(f"{model}/{method}", {}).get(metric, 0)
                   for method in methods]
            axis.bar(centers + offset, means, width=.34, yerr=sds, capsize=3,
                     label=model, color=colors[model])
        axis.set(title=title, ylabel="Score", ylim=(0, 1.02), xticks=centers,
                 xticklabels=methods)
        axis.tick_params(axis="x", labelrotation=20, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    recall = [report.get("auxiliary", {}).get(f"{model}/support_calibrated", {})
              .get("class_recall") for model in models]
    axis = axes.flat[3]
    if all(values is not None for values in recall):
        image = axis.imshow(np.array(recall), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        axis.set(title="SCR test recall by class", xlabel="Class", ylabel="Model",
                 xticks=range(len(recall[0])), yticks=range(len(models)), yticklabels=models)
        for row, values in enumerate(recall):
            for column, value in enumerate(values):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=7, color="white" if value > .6 else "black")
        fig.colorbar(image, ax=axis, fraction=.05, pad=.04)
    else:
        axis.axis("off")
        axis.text(.5, .5, "Per-class recall unavailable", ha="center", va="center")
    fig.suptitle(f"{report['dataset']} / {report['partition']} | validation-locked test results",
                 fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--partition")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    files = []
    for path in sorted(args.output_root.glob("*/development.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        cfg = result["configuration"]
        if ((args.dataset is None or cfg["dataset"] == args.dataset)
                and (args.partition is None or cfg["partition"] == args.partition)):
            plot_run(result, path.with_name("validation_curves.png"))
            files.append(path)
    outputs = plot_matrix(files, args.output_root, include_smoke=args.include_smoke)
    for output in outputs:
        print(f"Validation plot: {output}")
    if not outputs:
        print("No matching completed development runs to compare.")


if __name__ == "__main__":
    main()
