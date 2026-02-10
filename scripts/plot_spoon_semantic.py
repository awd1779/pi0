#!/usr/bin/env python3
"""Plot distractor scaling for spoon-on-towel semantic confusors (Pi0).

Reads real evaluation data from logs/clutter_eval/pi0/spoon/semantic/
and produces a single-panel figure: success rate vs. number of distractors.

Usage:
    python scripts/plot_spoon_semantic.py
    python scripts/plot_spoon_semantic.py --output figures/spoon_semantic.pdf
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
})

LOGS_DIR = Path("logs/clutter_eval/pi0/spoon/semantic")


def load_data():
    """Load summary CSVs, returning {n_distractors: (baseline_rates, cgvd_rates)}."""
    data = {}
    for d in sorted(LOGS_DIR.iterdir()):
        if not d.is_dir():
            continue
        # Parse distractor count from directory name (e.g. n6_e20_r10_...)
        n = int(d.name.split("_")[0][1:])
        summary = d / "summary.csv"
        if not summary.exists():
            continue

        with open(summary) as f:
            reader = csv.DictReader(f)
            for row in reader:
                bl = float(row["baseline_success_rate"])
                cg = float(row["cgvd_success_rate"])
                if n not in data:
                    data[n] = ([], [])
                data[n][0].append(bl)
                data[n][1].append(cg)
    return data


def plot(data, output):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ns = sorted(data.keys())
    x = np.array(ns)

    bl_means = np.array([np.mean(data[n][0]) for n in ns])
    bl_stds = np.array([np.std(data[n][0]) for n in ns])
    cg_means = np.array([np.mean(data[n][1]) for n in ns])
    cg_stds = np.array([np.std(data[n][1]) for n in ns])

    color_bl = "#2171b5"
    color_cg = "#d62728"

    # Baseline
    ax.plot(x, bl_means, color=color_bl, linestyle="--", marker="o",
            markersize=4, linewidth=1.3, label="Pi0 (baseline)")
    ax.fill_between(x,
                     np.clip(bl_means - bl_stds, 0, 100),
                     np.clip(bl_means + bl_stds, 0, 100),
                     color=color_bl, alpha=0.10)

    # +CGVD
    ax.plot(x, cg_means, color=color_cg, linestyle="-", marker="s",
            markersize=4, linewidth=1.8, label="Pi0 + CGVD")
    ax.fill_between(x,
                     np.clip(cg_means - cg_stds, 0, 100),
                     np.clip(cg_means + cg_stds, 0, 100),
                     color=color_cg, alpha=0.10)

    # Annotate endpoints with mean values
    for i, n in enumerate(ns):
        offset_bl = -6 if bl_means[i] < cg_means[i] else 6
        offset_cg = 6 if cg_means[i] > bl_means[i] else -6
        ax.annotate(f"{bl_means[i]:.0f}%", (n, bl_means[i]),
                    textcoords="offset points", xytext=(0, offset_bl),
                    ha="center", va="top" if offset_bl < 0 else "bottom",
                    fontsize=6, color=color_bl, fontweight="bold")
        ax.annotate(f"{cg_means[i]:.0f}%", (n, cg_means[i]),
                    textcoords="offset points", xytext=(0, offset_cg),
                    ha="center", va="bottom" if offset_cg > 0 else "top",
                    fontsize=6, color=color_cg, fontweight="bold")

    # Improvement arrows at each point
    for i, n in enumerate(ns):
        diff = cg_means[i] - bl_means[i]
        mid = (bl_means[i] + cg_means[i]) / 2
        ax.annotate(f"+{diff:.0f}",
                    (n + 0.35, mid),
                    fontsize=5.5, color="#388E3C", fontweight="bold",
                    ha="left", va="center")

    ax.set_xlim(1, 11)
    ax.set_ylim(0, 105)
    ax.set_xticks(ns)
    ax.set_xlabel("Number of semantic distractors")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Spoon on Towel — Semantic Confusors", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.85")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved to {output}")

    # Print summary table
    print(f"\n{'n':>3}  {'Baseline':>10}  {'CGVD':>10}  {'Δ':>6}  {'runs':>5}")
    print("-" * 42)
    for n in ns:
        bl, cg = data[n]
        print(f"{n:>3}  {np.mean(bl):>9.1f}%  {np.mean(cg):>9.1f}%  "
              f"{np.mean(cg)-np.mean(bl):>+5.1f}  {len(bl):>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/spoon_semantic.pdf")
    args = parser.parse_args()
    data = load_data()
    plot(data, args.output)
