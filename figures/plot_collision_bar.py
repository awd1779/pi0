"""Plot grouped bar chart of average collision rates: Baseline vs CGVD.

Usage:
    python figures/plot_collision_bar.py [--output figures/collision_bar.pdf]
"""

import argparse
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
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ---------- Collision data (from Experimental Results Table) ----------
# Format: {n: (baseline_coll, cgvd_coll)}
COLLISION_DATA = {
    "Spoon\nSemantic": {
        4:  (17.5, 4.5),
        8:  (25.0, 8.5),
        10: (26.5, 11.5),
        12: (29.5, 11.0),
        14: (29.0, 13.5),
        16: (25.5, 14.0),
        18: (29.0, 9.5),
    },
    "Spoon\nRandom": {
        4:  (2.0, 1.0),
        8:  (10.5, 5.0),
        10: (12.5, 5.0),
        12: (19.0, 8.5),
        14: (16.5, 7.0),
        16: (20.5, 8.0),
        18: (25.5, 8.5),
    },
    "Carrot\nRandom": {
        4:  (3.0, 1.0),
        8:  (6.5, 5.0),
        10: (9.5, 2.5),
        12: (11.0, 8.5),
        14: (12.0, 5.5),
        16: (16.0, 7.0),
        18: (15.0, 9.0),
    },
}

COLOR_BASELINE = "#c0392b"  # muted red
COLOR_CGVD = "#2471a3"      # muted blue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="figures/collision_bar.pdf")
    args = parser.parse_args()

    conditions = list(COLLISION_DATA.keys())
    baseline_avgs = []
    cgvd_avgs = []

    for cond in conditions:
        data = COLLISION_DATA[cond]
        bl = [v[0] for v in data.values()]
        cg = [v[1] for v in data.values()]
        baseline_avgs.append(np.mean(bl))
        cgvd_avgs.append(np.mean(cg))

    x = np.arange(len(conditions))
    width = 0.32

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    bars_bl = ax.bar(x - width / 2, baseline_avgs, width, color=COLOR_BASELINE,
                     edgecolor="white", linewidth=0.5, label="$\\pi_0$", zorder=3)
    bars_cg = ax.bar(x + width / 2, cgvd_avgs, width, color=COLOR_CGVD,
                     edgecolor="white", linewidth=0.5, label="$\\pi_0$ + CGVD", zorder=3)

    # Value labels on bars
    for bar in bars_bl:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7, color=COLOR_BASELINE)
    for bar in bars_cg:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7, color=COLOR_CGVD)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Collision Rate (%)")
    ax.set_ylim(0, max(baseline_avgs) * 1.25)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")

    fig.tight_layout()

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
