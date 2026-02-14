"""Plot 2x2 distractor sweep: rows = tasks, cols = distractor type.

Usage:
    python scripts/plot_distractor_sweep.py [--output paper/figures/distractor_sweep.pdf]

Expected data directory structure:
    logs/clutter_eval/pi0/{task_short}/{category}/n{X}_e*_r*_*/summary.csv
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# RA-L / IEEE style
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

BASE_DIR = Path("logs/clutter_eval/pi0")

# Colors (one color per model; dashed=baseline, solid=+CGVD)
COLOR_PI0 = "#2471a3"       # muted blue

# 2x2 grid: rows = tasks, cols = distractor type
PANELS = [
    (0, 0, "spoon",  "semantic"),
    (0, 1, "spoon",  "control"),
    (1, 0, "carrot", "semantic"),
    (1, 1, "carrot", "control"),
]

ROW_LABELS = ["Spoon on Towel", "Carrot on Plate"]
COL_LABELS = ["Semantic Distractors", "Random Distractors"]

# ---------- Hardcoded data from Results Tables.md ----------
# Format: {nd: {"baseline": [val], "cgvd": [val], "baseline_hard": [val], "cgvd_hard": [val]}}
HARDCODED = {
    ("spoon", "semantic"): {
        2:  {"baseline": [65.5], "cgvd": [69.5], "baseline_hard": [59.0], "cgvd_hard": [69.5]},
        4:  {"baseline": [62.5], "cgvd": [65.0], "baseline_hard": [50.5], "cgvd_hard": [65.0]},
        6:  {"baseline": [58.0], "cgvd": [64.5], "baseline_hard": [50.0], "cgvd_hard": [63.5]},
        8:  {"baseline": [52.0], "cgvd": [66.5], "baseline_hard": [43.5], "cgvd_hard": [66.0]},
        10: {"baseline": [44.5], "cgvd": [58.0], "baseline_hard": [40.0], "cgvd_hard": [56.5]},
        12: {"baseline": [44.5], "cgvd": [67.0], "baseline_hard": [39.0], "cgvd_hard": [66.0]},
        14: {"baseline": [47.5], "cgvd": [63.5], "baseline_hard": [42.0], "cgvd_hard": [61.5]},
    },
    ("spoon", "control"): {
        2:  {"baseline": [75.0], "cgvd": [65.0], "baseline_hard": [74.5], "cgvd_hard": [65.0]},
        4:  {"baseline": [76.5], "cgvd": [64.5], "baseline_hard": [75.5], "cgvd_hard": [63.5]},
        6:  {"baseline": [68.0], "cgvd": [64.5], "baseline_hard": [66.5], "cgvd_hard": [64.0]},
        8:  {"baseline": [70.5], "cgvd": [66.5], "baseline_hard": [68.5], "cgvd_hard": [66.0]},
        10: {"baseline": [68.0], "cgvd": [68.5], "baseline_hard": [65.0], "cgvd_hard": [67.0]},
        12: {"baseline": [58.5], "cgvd": [67.5], "baseline_hard": [55.0], "cgvd_hard": [67.5]},
        14: {"baseline": [57.5], "cgvd": [65.5], "baseline_hard": [52.5], "cgvd_hard": [63.0]},
    },
}


def load_condition(task_short: str, category: str) -> dict:
    """Load all distractor-count conditions for a (task, category) pair.

    Tries CSV files first; falls back to HARDCODED data.
    """
    cat_dir = BASE_DIR / task_short / category
    results = {}

    if cat_dir.exists():
        for run_dir in sorted(cat_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary = run_dir / "summary.csv"
            if not summary.exists():
                continue

            m = re.match(r"n(\d+)_", run_dir.name)
            if not m:
                continue
            nd = int(m.group(1))

            with open(summary) as f:
                rows = list(csv.DictReader(f))

            if not rows:
                continue

            results[nd] = {
                "baseline": [float(r["baseline_success_rate"]) for r in rows],
                "cgvd": [float(r["cgvd_success_rate"]) for r in rows],
                "baseline_hard": [float(r["baseline_hard_success_rate"]) for r in rows],
                "cgvd_hard": [float(r["cgvd_hard_success_rate"]) for r in rows],
            }

    # Fall back to hardcoded data if no CSVs found
    if not results:
        results = HARDCODED.get((task_short, category), {})

    return results


def _stats(values):
    """Return mean and standard error."""
    n = len(values)
    m = statistics.mean(values)
    se = statistics.stdev(values) / (n ** 0.5) if n > 1 else 0
    return m, se


def plot_panel(ax, data: dict, show_xlabel: bool, show_ylabel: bool):
    """Plot one panel: baseline vs CGVD SR vs number of distractors."""
    if not data:
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#888888",
                fontstyle="italic")
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        if show_xlabel:
            ax.set_xlabel("Number of Distractors")
        if show_ylabel:
            ax.set_ylabel("Success Rate (%)")
        ax.grid(True, alpha=0.25, linestyle="--")
        return

    nds = sorted(data.keys())
    x = np.array(nds)

    bl_m, bl_se = zip(*[_stats(data[n]["baseline"]) for n in nds])
    cg_m, cg_se = zip(*[_stats(data[n]["cgvd"]) for n in nds])
    bl_m, bl_se = np.array(bl_m), np.array(bl_se)
    cg_m, cg_se = np.array(cg_m), np.array(cg_se)

    # Shaded error bands (1 SE)
    ax.fill_between(x, bl_m - bl_se, bl_m + bl_se, color=COLOR_PI0, alpha=0.12)
    ax.fill_between(x, cg_m - cg_se, cg_m + cg_se, color=COLOR_PI0, alpha=0.12)

    # Lines
    ax.plot(x, bl_m, "o--", color=COLOR_PI0, markerfacecolor="white",
            markeredgewidth=1.2, zorder=3, label="$\\pi_0$")
    ax.plot(x, cg_m, "s-", color=COLOR_PI0, markerfacecolor="white",
            markeredgewidth=1.2, zorder=3, label="$\\pi_0$ + CGVD")

    # Annotate each point with its value
    for i, xi in enumerate(x):
        ax.annotate(f"{bl_m[i]:.0f}", (xi, bl_m[i]), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=6.5, color=COLOR_PI0)
        ax.annotate(f"{cg_m[i]:.0f}", (xi, cg_m[i]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.5, color=COLOR_PI0)

    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25, linestyle="--")

    if show_xlabel:
        ax.set_xlabel("Number of Distractors")
    if show_ylabel:
        ax.set_ylabel("Success Rate (%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="paper/figures/distractor_sweep.pdf")
    args = parser.parse_args()

    # IEEE single-column width ~3.5in, double ~7.16in
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.5), sharex=False, sharey=True)

    for (r, c, task_short, category) in PANELS:
        data = load_condition(task_short, category)
        plot_panel(axes[r, c], data,
                   show_xlabel=(r == 1),
                   show_ylabel=(c == 0))

    # Column titles
    for c, label in enumerate(COL_LABELS):
        axes[0, c].set_title(label, fontweight="bold", pad=8)

    # Row labels as combined ylabel on left column
    for r, label in enumerate(ROW_LABELS):
        axes[r, 0].set_ylabel(f"{label}\nSuccess Rate (%)", fontsize=9, fontweight="bold",
                               linespacing=1.2)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.subplots_adjust(hspace=0.25, wspace=0.08)

    # Single shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   frameon=True, fancybox=False, edgecolor="#cccccc",
                   bbox_to_anchor=(0.5, 0.0))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved to {out_path}")

    if out_path.suffix == ".pdf":
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        print(f"Saved to {png_path}")


if __name__ == "__main__":
    main()
