"""
Plot distractor scaling curves: success rate vs. number of distractors.

Generates a 2x2 grid figure:
  - Rows: spoon on towel, carrot on plate
  - Columns: semantic confusors, random clutter
  - Each subplot: 3 models overlaid (solid = +CGVD, dashed = baseline)

Usage:
    python scripts/plot_distractor_scaling.py \
        --results_dir logs/clutter_eval \
        --output figures/distractor_scaling.pdf

    # Generate with placeholder data:
    python scripts/plot_distractor_scaling.py --placeholder
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})

# Tasks: (task key, row label)
TASKS = [
    ("spoon", "Spoon on towel"),
    ("carrot", "Carrot on plate"),
]

CATEGORIES = [
    ("semantic", "Semantic Confusors"),
    ("random", "Random Clutter"),
]

MODELS = ["pi0", "groot", "openpi"]
MODEL_LABELS = {"pi0": "Pi0", "groot": "GR00T", "openpi": "OpenVLA"}
MODEL_COLORS = {"pi0": "#2171b5", "groot": "#d94801", "openpi": "#238b45"}

DISTRACTOR_COUNTS = [0, 2, 4, 6, 8, 10]


def load_from_logs(results_dir: str) -> dict:
    """Load results from log directory structure."""
    data = {}
    results_path = Path(results_dir)

    for model in MODELS:
        data[model] = {}
        for task, _ in TASKS:
            data[model][task] = {}
            for category, _ in CATEGORIES:
                data[model][task][category] = {}
                for n in DISTRACTOR_COUNTS:
                    task_dir = results_path / model / task / category
                    pattern = f"n{n}_*"
                    matches = sorted(task_dir.glob(pattern)) if task_dir.exists() else []

                    if matches:
                        summary_file = matches[-1] / "summary.csv"
                        if summary_file.exists():
                            baseline_rates = []
                            cgvd_rates = []
                            with open(summary_file) as f:
                                header = f.readline().strip().split(",")
                                bl_idx = header.index("baseline_success_rate")
                                cg_idx = header.index("cgvd_success_rate")
                                for line in f:
                                    if line.strip():
                                        vals = line.strip().split(",")
                                        baseline_rates.append(float(vals[bl_idx]))
                                        cgvd_rates.append(float(vals[cg_idx]))

                            data[model][task][category][str(n)] = {
                                "baseline": baseline_rates,
                                "cgvd": cgvd_rates,
                            }
    return data


def load_from_json(data_file: str) -> dict:
    """Load results from a single JSON file."""
    with open(data_file) as f:
        return json.load(f)


def plot_scaling(data: dict, output: str):
    """Generate a 2x2 scaling figure."""
    fig, axes = plt.subplots(
        2, 2, figsize=(7.16, 5.0), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.25, "wspace": 0.08},
    )

    x = np.array(DISTRACTOR_COUNTS)

    for row_idx, (task, row_label) in enumerate(TASKS):
        for col_idx, (category, cat_label) in enumerate(CATEGORIES):
            ax = axes[row_idx, col_idx]

            for model in MODELS:
                color = MODEL_COLORS[model]
                label = MODEL_LABELS[model]
                task_data = data.get(model, {}).get(task, {}).get(category, {})

                for condition, linestyle, lw, alpha_fill, suffix in [
                    ("baseline", "--", 1.2, 0.08, ""),
                    ("cgvd", "-", 1.8, 0.12, " +CGVD"),
                ]:
                    means = []
                    stds = []
                    valid_x = []

                    for n in x:
                        entry = task_data.get(str(n), {})
                        rates = entry.get(condition, [])
                        if rates:
                            means.append(np.mean(rates))
                            stds.append(np.std(rates))
                            valid_x.append(n)

                    if not valid_x:
                        continue

                    valid_x = np.array(valid_x)
                    means = np.array(means)
                    stds = np.array(stds)

                    ax.plot(
                        valid_x, means,
                        color=color, linestyle=linestyle, marker="o",
                        markersize=3, linewidth=lw,
                        label=f"{label}{suffix}",
                    )
                    ax.fill_between(
                        valid_x,
                        np.clip(means - stds, 0, 100),
                        np.clip(means + stds, 0, 100),
                        color=color, alpha=alpha_fill,
                    )

                    # Label at n=10 endpoint only
                    if len(means) > 0:
                        end_x = valid_x[-1]
                        end_y = means[-1]
                        offset_y = 4 if condition == "cgvd" else -4
                        va = "bottom" if condition == "cgvd" else "top"
                        ax.annotate(
                            f"{end_y:.0f}",
                            (end_x, end_y),
                            textcoords="offset points",
                            xytext=(0, offset_y),
                            ha="center", va=va,
                            fontsize=5.5, color=color, fontweight="bold",
                        )

            # Formatting
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(0, 105)
            ax.set_xticks(DISTRACTOR_COUNTS)
            ax.grid(True, alpha=0.3, linewidth=0.5)

            if row_idx == 0:
                ax.set_title(cat_label, fontweight="bold")
            if row_idx == 1:
                ax.set_xlabel("Number of distractors")
            if col_idx == 0:
                ax.set_ylabel(f"Success rate (%)\n({row_label})")

    # Shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.06),
        framealpha=0.9, edgecolor="0.8",
        fontsize=7.5,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved figure to {output}")
    plt.close(fig)


def create_placeholder_data() -> dict:
    """Create placeholder data for testing the plot layout."""
    np.random.seed(42)
    data = {}

    for model in MODELS:
        data[model] = {}
        model_offset = {"pi0": 0, "groot": -5, "openpi": -10}[model]

        for task, _ in TASKS:
            data[model][task] = {}
            for category, _ in CATEGORIES:
                data[model][task][category] = {}
                base_sr = 70 + model_offset
                # Random clutter causes less degradation than semantic
                degrade_bl = 2.5 if category == "semantic" else 1.5
                degrade_cg = 1.0 if category == "semantic" else 0.5

                for n in DISTRACTOR_COUNTS:
                    bl_mean = max(10, base_sr - n * degrade_bl)
                    cg_mean = max(15, base_sr - n * degrade_cg)

                    bl_rates = [bl_mean + np.random.normal(0, 5) for _ in range(5)]
                    cg_rates = [cg_mean + np.random.normal(0, 4) for _ in range(5)]

                    data[model][task][category][str(n)] = {
                        "baseline": bl_rates,
                        "cgvd": cg_rates,
                    }
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot distractor scaling curves")
    parser.add_argument(
        "--results_dir", type=str, default="logs/clutter_eval",
        help="Directory containing evaluation logs",
    )
    parser.add_argument(
        "--data_file", type=str, default=None,
        help="JSON file with all results (overrides --results_dir)",
    )
    parser.add_argument(
        "--output", type=str, default="figures/distractor_scaling.pdf",
        help="Output file path",
    )
    parser.add_argument(
        "--placeholder", action="store_true",
        help="Generate plot with placeholder data (for testing layout)",
    )
    args = parser.parse_args()

    if args.placeholder:
        data = create_placeholder_data()
    elif args.data_file:
        data = load_from_json(args.data_file)
    else:
        data = load_from_logs(args.results_dir)

    plot_scaling(data, args.output)
