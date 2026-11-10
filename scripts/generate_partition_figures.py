"""Generate publication-ready data-partition figures and count matrices.

Features:
- Step 1: Count matrices saved to results/partition_counts.npz
- Step 2: Main figures (seed 42 only, 1x3 panels, stacked bars, shared y-axis, no bar borders, clean 'k' number formatting)
- Step 3: Appendix figures (all 5 seeds, 5x3 grid, column-shared y-axis, no bar borders, well-spaced 'k' digits, clean row labels)
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from medmnist import INFO

# Set matplotlib publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    "figure.titlesize": 10.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DATASETS = ["pathmnist", "organamnist", "bloodmnist"]
REGIMES = ["practical", "dirichlet_0.3", "dirichlet_0.1"]
REGIME_TITLES = {
    "practical": "Practical",
    "dirichlet_0.3": r"Dirichlet ($\alpha=0.3$)",
    "dirichlet_0.1": r"Dirichlet ($\alpha=0.1$)",
}
SEEDS = [42, 43, 44, 45, 46]
NUM_CLIENTS = 12
PARTITION_DIR = Path("data/partitions")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("data/partitions/plots")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARTITION_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def format_k(x, pos):
    """Format sample counts compactly (e.g. 25000 -> 25k, 2500 -> 2.5k, 0 -> 0)."""
    if x == 0:
        return "0"
    elif x >= 1000:
        if x % 1000 == 0:
            return f"{int(x/1000)}k"
        else:
            return f"{x/1000:.1f}k"
    else:
        return str(int(x))


k_formatter = ticker.FuncFormatter(format_k)


def get_class_names(dataset_name: str) -> list[str]:
    """Retrieve official MedMNIST class names with clean legend formatting."""
    info_entry = INFO.get(dataset_name.lower())
    if info_entry and "label" in info_entry:
        label_dict = info_entry["label"]
        names = []
        for i in range(len(label_dict)):
            name = label_dict[str(i)]
            # Clean up excessively long BloodMNIST label for compact publication legend
            if "immature granulocytes" in name.lower():
                name = "immature granulocytes"
            names.append(name)
        return names
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_color_palette(num_classes: int) -> list:
    """Return colorblind-safe distinct palette for num_classes."""
    if num_classes <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(num_classes)]
    else:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(num_classes)]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Build & Save Count Matrices
# ──────────────────────────────────────────────────────────────────────────────

def build_all_count_matrices() -> dict:
    """Build count matrices counts[k, c] for every (dataset, regime, seed)."""
    matrices = {}
    print("=" * 70)
    print("STEP 1: Building Count Matrices for all (dataset, regime, seed)")
    print("=" * 70)

    for ds in DATASETS:
        class_names = get_class_names(ds)
        num_classes = len(class_names)
        for reg in REGIMES:
            for s in SEEDS:
                json_path = PARTITION_DIR / f"partition_{ds}_{reg}_localtest_seed{s}_clients12.json"
                if not json_path.exists():
                    raise FileNotFoundError(f"Missing partition file: {json_path}")
                
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                counts = np.zeros((NUM_CLIENTS, num_classes), dtype=int)
                for k in range(NUM_CLIENTS):
                    c_dist = data["clients"][str(k)]["class_distribution"]
                    for c in range(num_classes):
                        counts[k, c] = int(c_dist[str(c)])
                
                key = f"{ds}_{reg}_seed{s}"
                matrices[key] = counts
                
                # If pathmnist, also store default alias without dataset prefix
                if ds == "pathmnist":
                    matrices[f"{reg}_seed{s}"] = counts

                total_samples = counts.sum()
                print(f"  [{ds:<11} | {reg:<14} | seed {s}] Shape: {counts.shape} | Total train samples: {total_samples}")

    npz_path = RESULTS_DIR / "partition_counts.npz"
    np.savez_compressed(npz_path, **matrices)
    print(f"\n[Saved] Count matrices saved to: {npz_path.resolve()} ({len(matrices)} matrices stored)\n")
    return matrices


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Main Figure (Seed 42 Only, 1 Row x 3 Panels)
# ──────────────────────────────────────────────────────────────────────────────

def plot_main_figure(matrices: dict, dataset_name: str, out_prefix: str = None):
    """Plot 1 row x 3 panels stacked bar chart for seed 42 (without bar borders)."""
    class_names = get_class_names(dataset_name)
    num_classes = len(class_names)
    colors = get_color_palette(num_classes)
    client_indices = np.arange(1, NUM_CLIENTS + 1)  # 1..12

    # Determine shared y-axis limit across all 3 regimes for seed 42
    max_samples = 0
    for reg in REGIMES:
        counts = matrices[f"{dataset_name}_{reg}_seed42"]
        totals = counts.sum(axis=1)
        max_samples = max(max_samples, totals.max())
    y_limit = max_samples * 1.06

    # Adjust vertical space dynamically to ensure generous separation between 'Client Index' and legend
    fig_height = 3.15 if num_classes > 9 else 3.0
    bottom_margin = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(7.2, fig_height), sharey=True, dpi=300)
    fig.subplots_adjust(wspace=0.14, bottom=bottom_margin, top=0.88, left=0.09, right=0.98)

    for col_idx, reg in enumerate(REGIMES):
        ax = axes[col_idx]
        counts = matrices[f"{dataset_name}_{reg}_seed42"]
        bottom = np.zeros(NUM_CLIENTS, dtype=int)

        for c in range(num_classes):
            values = counts[:, c]
            ax.bar(
                client_indices,
                values,
                bottom=bottom,
                color=colors[c],
                edgecolor="none",     # No bar borders as requested
                linewidth=0,
                width=0.80,
            )
            bottom += values

        ax.set_title(REGIME_TITLES[reg], fontsize=9.5, fontweight="semibold", pad=6)
        ax.set_xlim(0.3, NUM_CLIENTS + 0.7)
        ax.set_ylim(0, y_limit)
        ax.set_xticks(client_indices)
        ax.set_xticklabels(client_indices, fontsize=7.2)
        ax.set_xlabel("Client Index", fontsize=8.5, labelpad=3)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)

        # Compact 'k' digit formatting on y-axis
        ax.yaxis.set_major_formatter(k_formatter)

        if col_idx == 0:
            ax.set_ylabel("Number of Samples", fontsize=8.5, labelpad=4)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # Shared legend below figure (no patch borders)
    legend_elements = [
        Patch(facecolor=colors[c], edgecolor="none", label=f"C{c}: {class_names[c]}")
        for c in range(num_classes)
    ]
    ncol = 4 if num_classes > 8 else min(num_classes, 5)
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.01),
        ncol=ncol,
        frameon=True,
        framealpha=0.95,
        edgecolor="#dddddd",
        fontsize=7.2,
        handletextpad=0.35,
        columnspacing=0.8,
        handlelength=1.1,
        handleheight=0.7,
    )

    if out_prefix is None:
        out_prefix = f"part_main_{dataset_name}_seed42"

    targets = [
        PARTITION_DIR / out_prefix,
        PLOTS_DIR / out_prefix,
    ]
    for target in targets:
        plt.savefig(target.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        plt.savefig(target.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Main Figure Saved] {out_prefix}.pdf & .png in data/partitions & plots/")


def plot_combined_main_figure(matrices: dict):
    """Plot 3 rows (datasets) x 3 columns (regimes) for seed 42 to show all datasets together with legends."""
    fig, axes = plt.subplots(3, 3, figsize=(8.8, 6.6), sharey="row", dpi=300)
    fig.subplots_adjust(wspace=0.18, hspace=0.36, bottom=0.08, top=0.94, left=0.08, right=0.74)
    client_indices = np.arange(1, NUM_CLIENTS + 1)

    for row_idx, ds in enumerate(DATASETS):
        class_names = get_class_names(ds)
        num_classes = len(class_names)
        colors = get_color_palette(num_classes)

        # row y-limit
        row_max = 0
        for reg in REGIMES:
            row_max = max(row_max, matrices[f"{ds}_{reg}_seed42"].sum(axis=1).max())
        y_lim = row_max * 1.06

        for col_idx, reg in enumerate(REGIMES):
            ax = axes[row_idx, col_idx]
            counts = matrices[f"{ds}_{reg}_seed42"]
            bottom = np.zeros(NUM_CLIENTS, dtype=int)
            for c in range(num_classes):
                values = counts[:, c]
                ax.bar(
                    client_indices,
                    values,
                    bottom=bottom,
                    color=colors[c],
                    edgecolor="none",     # No bar borders
                    linewidth=0,
                    width=0.80,
                )
                bottom += values

            if row_idx == 0:
                ax.set_title(REGIME_TITLES[reg], fontsize=9.5, fontweight="semibold", pad=5)
            ax.set_xlim(0.3, NUM_CLIENTS + 0.7)
            ax.set_ylim(0, y_lim)
            ax.set_xticks(client_indices)
            ax.set_xticklabels(client_indices, fontsize=6.8)
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
            ax.yaxis.set_major_formatter(k_formatter)

            if col_idx == 0:
                ds_display = {"pathmnist": "PathMNIST", "organamnist": "OrganAMNIST", "bloodmnist": "BloodMNIST"}[ds]
                ax.set_ylabel(f"{ds_display}\nSamples", fontsize=8.0, fontweight="medium", labelpad=3)
            else:
                ax.tick_params(axis="y", labelleft=False)

            if row_idx == 2:
                ax.set_xlabel("Client Index", fontsize=8.0, labelpad=2)

        # Dedicated right-side legend for this dataset row
        ds_display = {"pathmnist": "PathMNIST", "organamnist": "OrganAMNIST", "bloodmnist": "BloodMNIST"}[ds]
        legend_elements = [
            Patch(facecolor=colors[c], edgecolor="none", label=f"C{c}: {class_names[c]}")
            for c in range(num_classes)
        ]
        axes[row_idx, 2].legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(1.04, 0.5),
            ncol=1,
            frameon=True,
            framealpha=0.95,
            edgecolor="#dddddd",
            fontsize=6.5,
            handletextpad=0.25,
            handlelength=0.9,
            handleheight=0.6,
            labelspacing=0.22,
            title=f"{ds_display} Classes",
            title_fontsize=7.2,
        )

    targets = [
        PARTITION_DIR / "part_main_all3_seed42",
        PLOTS_DIR / "part_main_all3_seed42",
    ]
    for target in targets:
        plt.savefig(target.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        plt.savefig(target.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Combined Main Figure Saved] part_main_all3_seed42.pdf & .png in data/partitions & plots/")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Appendix Figure (5 Seeds x 3 Regimes Grid)
# ──────────────────────────────────────────────────────────────────────────────

def plot_appendix_figure(matrices: dict, dataset_name: str):
    """Plot 5 rows (seeds) x 3 columns (regimes) grid for the appendix."""
    class_names = get_class_names(dataset_name)
    num_classes = len(class_names)
    colors = get_color_palette(num_classes)
    client_indices = np.arange(1, NUM_CLIENTS + 1)

    # Determine shared y-axis limits WITHIN EACH COLUMN
    col_max_samples = {}
    for col_idx, reg in enumerate(REGIMES):
        m = 0
        for s in SEEDS:
            m = max(m, matrices[f"{dataset_name}_{reg}_seed{s}"].sum(axis=1).max())
        col_max_samples[reg] = m * 1.06

    fig_height = 10.6 if num_classes > 9 else 10.2
    bottom_margin = 0.16 if num_classes > 9 else 0.14

    # Generous wspace (0.28) and left margin (0.17) to prevent any digit collision
    fig, axes = plt.subplots(5, 3, figsize=(7.4, fig_height), dpi=300)
    fig.subplots_adjust(wspace=0.28, hspace=0.26, bottom=bottom_margin, top=0.96, left=0.17, right=0.98)

    for row_idx, s in enumerate(SEEDS):
        for col_idx, reg in enumerate(REGIMES):
            ax = axes[row_idx, col_idx]
            counts = matrices[f"{dataset_name}_{reg}_seed{s}"]
            bottom = np.zeros(NUM_CLIENTS, dtype=int)

            for c in range(num_classes):
                values = counts[:, c]
                ax.bar(
                    client_indices,
                    values,
                    bottom=bottom,
                    color=colors[c],
                    edgecolor="none",     # No bar borders
                    linewidth=0,
                    width=0.80,
                )
                bottom += values

            # Column titles ONLY on top row
            if row_idx == 0:
                ax.set_title(REGIME_TITLES[reg], fontsize=9.5, fontweight="semibold", pad=5)

            # Shared y-limit per column
            ax.set_ylim(0, col_max_samples[reg])
            ax.set_xlim(0.3, NUM_CLIENTS + 0.7)
            ax.set_xticks(client_indices)
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)

            # Compact 'k' formatting for all y-axis digits (prevents horizontal overflow)
            ax.yaxis.set_major_formatter(k_formatter)
            ax.tick_params(axis="y", labelsize=6.8, pad=2)

            # y-axis label ONLY on first column
            if col_idx == 0:
                ax.set_ylabel("Samples", fontsize=8.0, labelpad=3)
            else:
                ax.tick_params(axis="y", labelleft=True)

            # x-axis labels ONLY on bottom row
            if row_idx == len(SEEDS) - 1:
                ax.set_xlabel("Client", fontsize=8.0, labelpad=2)
                ax.set_xticklabels(client_indices, fontsize=6.8)
            else:
                ax.set_xticklabels([])

    # Add dedicated, prominent HORIZONTAL row labels on the far left
    fig.canvas.draw_idle()
    for row_idx, s in enumerate(SEEDS):
        bbox = axes[row_idx, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(
            0.025, y_center, f"Seed {s}",
            va="center", ha="left",
            fontsize=8.5,
            fontweight="bold"
        )

    # Shared legend below (no borders on patches)
    legend_elements = [
        Patch(facecolor=colors[c], edgecolor="none", label=f"C{c}: {class_names[c]}")
        for c in range(num_classes)
    ]
    ncol = 4 if num_classes > 8 else min(num_classes, 5)
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.57, 0.01),
        ncol=ncol,
        frameon=True,
        framealpha=0.95,
        edgecolor="#dddddd",
        fontsize=7.2,
        handletextpad=0.35,
        columnspacing=0.8,
        handlelength=1.1,
        handleheight=0.7,
    )

    # Save to both data/partitions/ and data/partitions/plots/
    targets = [
        PARTITION_DIR / f"part_appendix_{dataset_name}_all_seeds",
        PLOTS_DIR / f"part_appendix_{dataset_name}_all_seeds",
    ]
    if dataset_name == "pathmnist":
        targets.extend([
            PARTITION_DIR / "part_appendix_all_seeds",
            PLOTS_DIR / "part_appendix_all_seeds",
        ])

    for target in targets:
        pdf_path = target.with_suffix(".pdf")
        png_path = target.with_suffix(".png")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        print(f"[Appendix Saved] {pdf_path} & {png_path.name}")
    plt.close(fig)


def main():
    matrices = build_all_count_matrices()

    print("\n" + "=" * 70)
    print("STEP 2: Generating Main Figures (Seed 42)")
    print("=" * 70)
    plot_main_figure(matrices, "pathmnist", out_prefix="part_main_seed42")
    plot_main_figure(matrices, "pathmnist", out_prefix="part_main_pathmnist_seed42")
    plot_main_figure(matrices, "organamnist", out_prefix="part_main_organamnist_seed42")
    plot_main_figure(matrices, "bloodmnist", out_prefix="part_main_bloodmnist_seed42")
    plot_combined_main_figure(matrices)

    print("\n" + "=" * 70)
    print("STEP 3: Generating Appendix Figures (All 5 Seeds x 3 Regimes)")
    print("=" * 70)
    for ds in DATASETS:
        plot_appendix_figure(matrices, ds)

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
