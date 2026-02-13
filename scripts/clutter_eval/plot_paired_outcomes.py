#!/usr/bin/env python3
"""Plot 3-panel paired outcome figure: SR curves, stacked bars, diverging bars."""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

BASE = "logs/clutter_eval/pi0/spoon/semantic"
COUNTS = [2, 4, 6, 8, 10, 12]
OUT_PATH = "logs/clutter_eval/pi0/paired_outcomes.png"

# Paper-appropriate muted palette
C_GREEN = "#4a9c6d"   # both succeed
C_BLUE = "#3a7ebf"    # rescued
C_ORANGE = "#d4843e"  # regressed
C_RED = "#c44e52"     # both fail
C_BASELINE = "#c44e52"  # baseline SR line
C_CGVD = "#3a7ebf"      # CGVD SR line


def load_data():
    """Load per-run SR stats and per-episode paired outcomes."""
    paired = []
    sr_baseline_runs = []  # list of lists: [[run0, run1, ...], ...]
    sr_cgvd_runs = []

    for n in COUNTS:
        dirs = sorted(d for d in os.listdir(BASE) if d.startswith(f"n{n}_e20_r5_"))
        run_dir = os.path.join(BASE, dirs[0])

        # -- Per-run success rates from summary.csv --
        bl_rates, cg_rates = [], []
        with open(os.path.join(run_dir, "summary.csv")) as f:
            for row in csv.DictReader(f):
                bl_rates.append(float(row["baseline_success_rate"]))
                cg_rates.append(float(row["cgvd_success_rate"]))
        sr_baseline_runs.append(bl_rates)
        sr_cgvd_runs.append(cg_rates)

        # -- Per-episode paired outcomes from results.csv --
        both_success = both_fail = rescued = regressed = 0
        with open(os.path.join(run_dir, "results.csv")) as f:
            for row in csv.DictReader(f):
                bs = row["baseline_failure_mode"] == "success"
                cs = row["cgvd_failure_mode"] == "success"
                if bs and cs:
                    both_success += 1
                elif not bs and not cs:
                    both_fail += 1
                elif not bs and cs:
                    rescued += 1
                else:
                    regressed += 1

        total = both_success + both_fail + rescued + regressed
        paired.append({
            "n": n,
            "both_success": both_success,
            "both_fail": both_fail,
            "rescued": rescued,
            "regressed": regressed,
            "total": total,
        })

    return paired, np.array(sr_baseline_runs), np.array(sr_cgvd_runs)


def main():
    paired, sr_bl, sr_cg = load_data()
    ns = np.array(COUNTS)
    x = np.arange(len(ns))

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), height_ratios=[3, 3, 2],
                              gridspec_kw={"hspace": 0.38})
    ax_sr, ax_stack, ax_div = axes

    # ===== Panel A: Success Rate Curves =====
    bl_mean = sr_bl.mean(axis=1)
    bl_std = sr_bl.std(axis=1)
    cg_mean = sr_cg.mean(axis=1)
    cg_std = sr_cg.std(axis=1)

    ax_sr.fill_between(ns, bl_mean - bl_std, bl_mean + bl_std,
                        alpha=0.15, color=C_BASELINE, linewidth=0)
    ax_sr.fill_between(ns, cg_mean - cg_std, cg_mean + cg_std,
                        alpha=0.15, color=C_CGVD, linewidth=0)
    ax_sr.plot(ns, bl_mean, "o-", color=C_BASELINE, linewidth=2, markersize=6,
               label="Baseline", zorder=3)
    ax_sr.plot(ns, cg_mean, "s-", color=C_CGVD, linewidth=2, markersize=6,
               label="CGVD", zorder=3)

    # Annotate endpoints — position to avoid overlaps
    ax_sr.annotate(f"{bl_mean[0]:.0f}%", (ns[0], bl_mean[0]),
                    textcoords="offset points", xytext=(-24, 8),
                    fontsize=9, color=C_BASELINE, fontweight="bold")
    ax_sr.annotate(f"{bl_mean[-1]:.0f}%", (ns[-1], bl_mean[-1]),
                    textcoords="offset points", xytext=(8, -4),
                    fontsize=9, color=C_BASELINE, fontweight="bold")
    ax_sr.annotate(f"{cg_mean[0]:.0f}%", (ns[0], cg_mean[0]),
                    textcoords="offset points", xytext=(-24, -16),
                    fontsize=9, color=C_CGVD, fontweight="bold")
    ax_sr.annotate(f"{cg_mean[-1]:.0f}%", (ns[-1], cg_mean[-1]),
                    textcoords="offset points", xytext=(8, -12),
                    fontsize=9, color=C_CGVD, fontweight="bold")

    ax_sr.set_ylabel("Success Rate (%)", fontsize=10.5)
    ax_sr.set_xlabel("Number of Distractors", fontsize=10.5)
    ax_sr.set_xticks(ns)
    ax_sr.set_ylim(25, 85)
    ax_sr.legend(fontsize=9.5, loc="upper right", framealpha=0.9)
    ax_sr.set_title("(a)  Success Rate vs. Distractor Count  (mean ± 1 std, n=5 runs)",
                     fontsize=10.5, fontweight="bold", loc="left", pad=6)
    ax_sr.spines["top"].set_visible(False)
    ax_sr.spines["right"].set_visible(False)
    ax_sr.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_sr.grid(axis="y", alpha=0.25, linewidth=0.5)

    # ===== Panel B: Stacked Paired Outcomes =====
    width = 0.55
    keys = ["both_success", "rescued", "regressed", "both_fail"]
    colors = [C_GREEN, C_BLUE, C_ORANGE, C_RED]
    labels = [
        "Both succeed",
        "CGVD rescued",
        "CGVD regressed",
        "Both fail",
    ]

    bottom = np.zeros(len(ns))
    for key, color, label in zip(keys, colors, labels):
        vals = np.array([d[key] for d in paired], dtype=float)
        ax_stack.bar(x, vals, width, bottom=bottom, color=color, label=label,
                     edgecolor="white", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 5:
                ax_stack.text(x[i], b + v / 2, f"{int(v)}", ha="center", va="center",
                              fontsize=9, fontweight="bold", color="white")
        bottom += vals

    ax_stack.set_ylabel("Episodes (out of 100)", fontsize=10.5)
    ax_stack.set_ylim(0, 108)
    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels([str(n) for n in ns], fontsize=10)
    ax_stack.set_xlabel("Number of Distractors", fontsize=10.5)
    ax_stack.legend(fontsize=8.5, loc="upper center", ncol=4, framealpha=0.9,
                     bbox_to_anchor=(0.5, 1.02), columnspacing=1.0)
    ax_stack.set_title("(b)  Paired Episode Outcomes  (5 runs × 20 episodes)",
                        fontsize=10.5, fontweight="bold", loc="left", pad=6)
    ax_stack.spines["top"].set_visible(False)
    ax_stack.spines["right"].set_visible(False)

    # ===== Panel C: Diverging Rescued vs Regressed =====
    y = np.arange(len(ns))
    rescued = np.array([d["rescued"] for d in paired], dtype=float)
    regressed_neg = np.array([-d["regressed"] for d in paired], dtype=float)
    net = rescued + regressed_neg
    bar_h = 0.50

    ax_div.barh(y, rescued, height=bar_h, color=C_BLUE,
                edgecolor="white", linewidth=0.5)
    ax_div.barh(y, regressed_neg, height=bar_h, color=C_ORANGE,
                edgecolor="white", linewidth=0.5)

    for i in range(len(ns)):
        if rescued[i] >= 4:
            ax_div.text(rescued[i] / 2, y[i], f"+{int(rescued[i])}", ha="center",
                        va="center", fontsize=9, fontweight="bold", color="white")
        if -regressed_neg[i] >= 4:
            ax_div.text(regressed_neg[i] / 2, y[i], f"{int(-regressed_neg[i])}",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white")
        net_val = int(net[i])
        color = "#2a6099" if net_val > 0 else ("#b5651d" if net_val < 0 else "#666")
        ax_div.text(max(rescued[i], 0) + 1.5, y[i], f"net {net_val:+d}",
                    ha="left", va="center", fontsize=9, fontweight="bold", color=color)

    ax_div.axvline(0, color="black", linewidth=0.8)
    ax_div.set_yticks(y)
    ax_div.set_yticklabels([str(n) for n in ns], fontsize=10)
    ax_div.set_ylabel("Distractors", fontsize=10.5)
    ax_div.set_xlabel("Episodes", fontsize=10.5)
    ax_div.set_title("(c)  Rescued vs. Regressed",
                      fontsize=10.5, fontweight="bold", loc="left", pad=6)
    ax_div.spines["top"].set_visible(False)
    ax_div.spines["right"].set_visible(False)
    ax_div.invert_yaxis()
    # Direction labels
    ax_div.text(-0.5, -0.65, "← regressed", ha="right", va="bottom", fontsize=8.5,
                color=C_ORANGE, fontstyle="italic", transform=ax_div.transData)
    ax_div.text(0.5, -0.65, "rescued →", ha="left", va="bottom", fontsize=8.5,
                color=C_BLUE, fontstyle="italic", transform=ax_div.transData)
    ax_div.set_ylim(len(ns) - 0.5, -1.0)

    # ===== Save =====
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.suptitle("Paired Evaluation: Baseline vs. CGVD  (widowx_spoon_on_towel, semantic)",
                 fontsize=12, fontweight="bold", y=0.995)
    plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
