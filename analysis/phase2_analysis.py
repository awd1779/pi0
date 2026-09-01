#!/usr/bin/env python3
"""Phase 2 re-analysis of CGVD rollout logs (revision items R24-R30).

Reads per-episode results.csv logs under logs/, computes per-seed success
rates, paired seed-level statistics, confidence intervals, ICC, and trend
tests, and emits:
  analysis/out/main_results.csv          - every condition, both arms, CIs
  analysis/out/per_seed.csv              - per-seed rates and deltas
  analysis/out/ablation.csv              - ablation rows with CIs
  analysis/out/attribute.csv             - Table I cells with CIs
  analysis/out/replication_checks.txt    - R29 noise-floor / reuse checks
  analysis/out/timing.txt                - Table III provenance check
  paper/generated/main_results_table.tex - R27 numeric main-results table
  paper/generated/attribute_table.tex    - regenerated Table I (when both arms exist)

Rules honored: no number is invented; every value traces to a results.csv /
summary JSON row. Duplicate (model,task,category,count) run dirs resolve to
the LATEST timestamp; older duplicates are reported as replication estimates.
"""
import csv, glob, json, math, os, re, sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
OUT = os.path.join(ROOT, "analysis", "out")
GEN = os.path.join(ROOT, "paper", "generated")
os.makedirs(OUT, exist_ok=True)
os.makedirs(GEN, exist_ok=True)

DIR_RE = re.compile(r"n(\d+)_e(\d+)_r(\d+)_(\d{8}_\d{6})$")


def wilson_ci(k, n, z=1.959964):
    """Wilson score interval for a single proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def newcombe_diff_ci(k1, n1, k2, n2):
    """Newcombe hybrid score CI for p1 - p2 (unpaired)."""
    l1, u1 = wilson_ci(k1, n1)
    l2, u2 = wilson_ci(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def seed_delta_ci(deltas, conf=0.95):
    """t-interval over per-seed paired deltas (the design-respecting CI)."""
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n < 2:
        return (float("nan"), float("nan"))
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    tcrit = stats.t.ppf(0.5 + conf / 2, n - 1)
    return (m - tcrit * se, m + tcrit * se)


def icc_oneway(groups):
    """One-way ANOVA ICC(1) over episode outcomes grouped by seed."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 1]
    k = len(groups)
    if k < 2:
        return float("nan")
    n_per = np.array([len(g) for g in groups])
    N = n_per.sum()
    grand = np.concatenate(groups).mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    n0 = (N - (n_per ** 2).sum() / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else float("nan")


def load_results_csv(path):
    rows = list(csv.DictReader(open(path)))
    return rows


def find_condition_dirs():
    """Map (model, task, category, count) -> latest run dir; keep older dups."""
    latest, dups = {}, defaultdict(list)
    pats = [
        (os.path.join(LOGS, "clutter_eval", "gr00t", "*", "*", "n*_e20_r10_*"), "gr00t"),
        (os.path.join(LOGS, "clutter_eval", "pi0", "spoon", "spoon", "*", "n*_e20_r10_*"), "pi0"),
        (os.path.join(LOGS, "clutter_eval", "pi0", "carrot", "*", "n*_e20_r10_*"), "pi0"),
        (os.path.join(LOGS, "clutter_eval", "pi0", "carrot", "DOne", "*", "n*_e20_r10_*"), "pi0"),
    ]
    for pat, model in pats:
        for d in glob.glob(pat):
            m = DIR_RE.search(os.path.basename(d))
            if not m:
                continue
            count, stamp = int(m.group(1)), m.group(4)
            parts = d.split(os.sep)
            category = parts[-2]
            task = "spoon" if "spoon" in d.split("clutter_eval")[1].split(os.sep)[2] else "carrot"
            key = (model, task, category, count)
            dups[key].append((stamp, d))
    for key, lst in dups.items():
        lst.sort()
        latest[key] = lst[-1][1]
    older = {k: [d for _, d in sorted(v)[:-1]] for k, v in dups.items() if len(v) > 1}
    return latest, older


def condition_stats(rows):
    """Per-condition stats from per-episode rows (paired by seed)."""
    by_seed_b, by_seed_c = defaultdict(list), defaultdict(list)
    for r in rows:
        s = int(r["seed"])
        by_seed_b[s].append(int(r["baseline_success"]))
        by_seed_c[s].append(int(r["cgvd_success"]))
    seeds = sorted(by_seed_b)
    b_rates = np.array([np.mean(by_seed_b[s]) for s in seeds]) * 100
    c_rates = np.array([np.mean(by_seed_c[s]) for s in seeds]) * 100
    deltas = c_rates - b_rates
    kb = sum(sum(by_seed_b[s]) for s in seeds)
    kc = sum(sum(by_seed_c[s]) for s in seeds)
    n = sum(len(by_seed_b[s]) for s in seeds)
    out = {
        "n_episodes": n, "n_seeds": len(seeds),
        "baseline_mean": kb / n * 100, "cgvd_mean": kc / n * 100,
        "delta": (kc - kb) / n * 100,
        "baseline_sd_seed": float(b_rates.std(ddof=1)) if len(seeds) > 1 else float("nan"),
        "cgvd_sd_seed": float(c_rates.std(ddof=1)) if len(seeds) > 1 else float("nan"),
        "delta_ci_paired": seed_delta_ci(deltas),
        "delta_ci_newcombe": tuple(100 * x for x in newcombe_diff_ci(kc, n, kb, n)),
        "icc_baseline": icc_oneway([by_seed_b[s] for s in seeds]),
        "icc_cgvd": icc_oneway([by_seed_c[s] for s in seeds]),
        "seed_rates_baseline": b_rates.tolist(), "seed_rates_cgvd": c_rates.tolist(),
        "seed_deltas": deltas.tolist(),
    }
    if len(seeds) > 1 and np.any(deltas != 0):
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(deltas).pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    else:
        out["wilcoxon_p"] = float("nan")
    return out


def main():
    latest, older = find_condition_dirs()
    results = {}
    for key, d in sorted(latest.items()):
        f = os.path.join(d, "results.csv")
        rows = load_results_csv(f) if os.path.exists(f) else []
        if not rows:
            # Fallback: reconstruct exact episode-level outcomes from per-seed
            # summary.csv (binary outcomes, 20 eps/seed -> k = rate*20/100 exactly).
            sf = os.path.join(d, "summary.csv")
            if not os.path.exists(sf):
                continue
            srows = load_results_csv(sf)
            rows = []
            for sr in srows:
                seed = int(sr["seed"])
                kb = round(float(sr["baseline_success_rate"]) * 20 / 100)
                kc = round(float(sr["cgvd_success_rate"]) * 20 / 100)
                for i in range(20):
                    rows.append({"seed": str(seed),
                                 "baseline_success": "1" if i < kb else "0",
                                 "cgvd_success": "1" if i < kc else "0"})
        results[key] = condition_stats(rows)
        results[key]["dir"] = os.path.relpath(d, ROOT)

    # ---------- main_results.csv + per_seed.csv ----------
    with open(os.path.join(OUT, "main_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "task", "category", "count", "n_episodes", "n_seeds",
                    "baseline", "cgvd", "delta", "baseline_sd_seed", "cgvd_sd_seed",
                    "delta_ci_paired_lo", "delta_ci_paired_hi",
                    "delta_ci_newcombe_lo", "delta_ci_newcombe_hi",
                    "wilcoxon_p", "icc_baseline", "icc_cgvd", "excludes0_paired", "dir"])
        for (model, task, cat, cnt), s in sorted(results.items()):
            lo, hi = s["delta_ci_paired"]
            w.writerow([model, task, cat, cnt, s["n_episodes"], s["n_seeds"],
                        f"{s['baseline_mean']:.1f}", f"{s['cgvd_mean']:.1f}", f"{s['delta']:+.1f}",
                        f"{s['baseline_sd_seed']:.1f}", f"{s['cgvd_sd_seed']:.1f}",
                        f"{lo:.1f}", f"{hi:.1f}",
                        f"{s['delta_ci_newcombe'][0]:.1f}", f"{s['delta_ci_newcombe'][1]:.1f}",
                        f"{s['wilcoxon_p']:.4f}", f"{s['icc_baseline']:.3f}", f"{s['icc_cgvd']:.3f}",
                        int(lo > 0 or hi < 0), s["dir"]])
    with open(os.path.join(OUT, "per_seed.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "task", "category", "count", "seed", "baseline_rate", "cgvd_rate", "delta"])
        for (model, task, cat, cnt), s in sorted(results.items()):
            for i in range(s["n_seeds"]):
                w.writerow([model, task, cat, cnt, i,
                            f"{s['seed_rates_baseline'][i]:.1f}",
                            f"{s['seed_rates_cgvd'][i]:.1f}",
                            f"{s['seed_deltas'][i]:+.1f}"])

    # ---------- R27: gap-widens trend test (per-seed slope of delta vs count) ----------
    trend_lines = []
    for model in ("pi0", "gr00t"):
        for task in ("spoon", "carrot"):
            for cat in ("semantic", "control"):
                pts = {cnt: s for (m, t, c, cnt), s in results.items()
                       if m == model and t == task and c == cat}
                if len(pts) < 3:
                    continue
                counts = sorted(pts)
                nseeds = pts[counts[0]]["n_seeds"]
                slopes = []
                for i in range(nseeds):
                    xs = np.array(counts, dtype=float)
                    ys = np.array([pts[c]["seed_deltas"][i] for c in counts])
                    slopes.append(float(np.polyfit(xs, ys, 1)[0]))
                slopes = np.array(slopes)
                t_stat, p = stats.ttest_1samp(slopes, 0.0)
                trend_lines.append(
                    f"{model} {task} {cat}: mean per-seed slope of (CGVD-baseline) vs count = "
                    f"{slopes.mean():+.2f} pp/distractor (SD {slopes.std(ddof=1):.2f}, "
                    f"t({nseeds-1})={t_stat:.2f}, p={p:.4f})")
    with open(os.path.join(OUT, "trend_tests.txt"), "w") as fh:
        fh.write("\n".join(trend_lines) + "\n")

    # ---------- R27: LaTeX main-results table ----------
    def cell(s, arm):
        mean = s[f"{arm}_mean"]; sd = s[f"{arm}_sd_seed"]
        return f"{mean:.1f}$\\pm${sd:.1f}"

    lines = []
    lines.append("% AUTO-GENERATED by analysis/phase2_analysis.py — do not edit by hand.")
    lines.append("% Mean success rate (%) ± SD across the 10 matched seeds (20 episodes/seed).")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{\\textbf{Main results (numeric).} Success rate (\\%), mean $\\pm$ SD over 10 matched seeds (200 episodes/cell); $\\Delta$ with 95\\% CI from per-seed paired deltas ($t_9$), bold when the CI excludes zero.}")
    lines.append("\\label{tab:main_numeric}")
    lines.append("\\renewcommand{\\arraystretch}{1.05}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{ll" + "ccc" * 4 + "}")
    lines.append("\\hline")
    lines.append(" & & \\multicolumn{6}{c}{\\textbf{Semantic distractors}} & \\multicolumn{6}{c}{\\textbf{Random distractors}} \\\\")
    lines.append("\\cline{3-8}\\cline{9-14}")
    lines.append(" & & \\multicolumn{3}{c}{$\\pi_0$} & \\multicolumn{3}{c}{GR00T} & \\multicolumn{3}{c}{$\\pi_0$} & \\multicolumn{3}{c}{GR00T} \\\\")
    lines.append("\\textbf{Task} & \\textbf{\\#D} & base & +CGVD & $\\Delta$ [95\\% CI] & base & +CGVD & $\\Delta$ [95\\% CI] & base & +CGVD & $\\Delta$ [95\\% CI] & base & +CGVD & $\\Delta$ [95\\% CI] \\\\")
    lines.append("\\hline")
    task_names = {"spoon": "\\textit{spoon on towel}", "carrot": "\\textit{carrot on plate}"}
    for task in ("spoon", "carrot"):
        counts = sorted({k[3] for k in results if k[1] == task})
        for j, cnt in enumerate(counts):
            row = [task_names[task] if j == 0 else "", str(cnt)]
            for cat in ("semantic", "control"):
                for model in ("pi0", "gr00t"):
                    s = results.get((model, task, cat, cnt))
                    if s is None:
                        row += ["--", "--", "--"]
                        continue
                    lo, hi = s["delta_ci_paired"]
                    dtxt = f"{s['delta']:+.1f} [{lo:+.1f},{hi:+.1f}]"
                    if lo > 0 or hi < 0:
                        dtxt = "\\textbf{" + dtxt + "}"
                    row += [f"{s['baseline_mean']:.1f}", f"{s['cgvd_mean']:.1f}", dtxt]
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table*}")
    with open(os.path.join(GEN, "main_results_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------- Ablation (Table II) ----------
    abl_lines = ["condition,n_episodes,n_seeds,sr,sd_seed,delta_vs_full,ci_lo,ci_hi,source"]
    full = results.get(("pi0", "spoon", "semantic", 18))
    if full:
        abl_lines.append(
            f"baseline,{full['n_episodes']},{full['n_seeds']},{full['baseline_mean']:.1f},"
            f"{full['baseline_sd_seed']:.1f},,,,{full['dir']}")
        abl_lines.append(
            f"full_pipeline,{full['n_episodes']},{full['n_seeds']},{full['cgvd_mean']:.1f},"
            f"{full['cgvd_sd_seed']:.1f},,,,{full['dir']}")
    for variant in ("cgvd_no_inpaint", "cgvd_no_crossval", "cgvd_no_robot"):
        jf = glob.glob(os.path.join(LOGS, "ablation_spoon_semantic_18dist", variant, "sweep_results_*.json"))
        if not jf:
            continue
        data = json.load(open(jf[0]))
        seed_rates, per_seed_eps = [], []
        for entry in data:
            eps = entry["cgvd"]["episodes"]
            per_seed_eps.append([1 if e["success"] else 0 for e in eps])
            seed_rates.append(entry["cgvd"]["success_rate"])
        allk = sum(sum(g) for g in per_seed_eps)
        alln = sum(len(g) for g in per_seed_eps)
        sr = allk / alln * 100
        sd = float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else float("nan")
        if full:
            # paired per-seed delta vs full pipeline (matched seeds)
            fr = full["seed_rates_cgvd"]
            deltas = [seed_rates[i] - fr[i] for i in range(min(len(fr), len(seed_rates)))]
            lo, hi = seed_delta_ci(deltas)
        else:
            lo = hi = float("nan")
        abl_lines.append(f"{variant},{alln},{len(seed_rates)},{sr:.1f},{sd:.1f},"
                         f"{sr - full['cgvd_mean']:+.1f},{lo:.1f},{hi:.1f},{os.path.relpath(jf[0], ROOT)}")
    with open(os.path.join(OUT, "ablation.csv"), "w") as fh:
        fh.write("\n".join(abl_lines) + "\n")

    # ---------- Attribute (Table I) ----------
    attr_rows = ["arm,count,n_episodes,n_seeds,baseline,cgvd,delta,ci_paired_lo,ci_paired_hi,ci_newcombe_lo,ci_newcombe_hi,dir"]
    attr_stats = {}
    for arm, base in (("complex", os.path.join(LOGS, "attribute_spoon", "spoon", "attribute")),
                      ("simple", os.path.join(LOGS, "attribute_spoon_simple", "spoon", "attribute"))):
        for d in sorted(glob.glob(os.path.join(base, "n*_e20_r*_*"))):
            m = DIR_RE.search(os.path.basename(d))
            if not m:
                continue
            f = os.path.join(d, "results.csv")
            if not os.path.exists(f):
                continue
            rows = load_results_csv(f)
            if not rows:
                continue
            s = condition_stats(rows)
            attr_stats[(arm, int(m.group(1)))] = s
            lo, hi = s["delta_ci_paired"]
            nlo, nhi = s["delta_ci_newcombe"]
            attr_rows.append(f"{arm},{m.group(1)},{s['n_episodes']},{s['n_seeds']},"
                             f"{s['baseline_mean']:.1f},{s['cgvd_mean']:.1f},{s['delta']:+.1f},"
                             f"{lo:.1f},{hi:.1f},{nlo:.1f},{nhi:.1f},{os.path.relpath(d, ROOT)}")
    with open(os.path.join(OUT, "attribute.csv"), "w") as fh:
        fh.write("\n".join(attr_rows) + "\n")

    # regenerate Table I tex when both arms are present
    counts_both = sorted({c for (a, c) in attr_stats if ("simple", c) in attr_stats and ("complex", c) in attr_stats})
    if counts_both:
        tl = ["% AUTO-GENERATED by analysis/phase2_analysis.py — do not edit by hand."]
        ns = attr_stats[("complex", counts_both[0])]["n_seeds"]
        ne = attr_stats[("complex", counts_both[0])]["n_episodes"]
        tl.append("\\begin{table}[!t]\n\\centering")
        tl.append("\\caption{\\textbf{Attribute Distractor Sensitivity} on \\textit{put spoon on towel} ($\\pi_0$, "
                  f"{ne} episodes/cell, {ns} matched seeds). "
                  "$\\Delta$: 95\\% CI from per-seed paired deltas; bold = CI excludes zero; otherwise within run-to-run noise.}")
        tl.append("\\label{tab:combined_attribute_scaling}")
        tl.append("\\renewcommand{\\arraystretch}{1.05}")
        tl.append("\\resizebox{\\columnwidth}{!}{%")
        tl.append("\\begin{tabular}{l ccc ccc}\n\\hline")
        tl.append("& \\multicolumn{3}{c}{\\textbf{Simple Prompt}}\n& \\multicolumn{3}{c}{\\textbf{Complex Prompt}} \\\\")
        tl.append("\\cline{2-4} \\cline{5-7}")
        tl.append("\\textbf{\\# Distractors}\n  & \\textbf{$\\pi_0$} & \\textbf{+CGVD} & \\textbf{$\\Delta$ [95\\% CI]}\n  & \\textbf{$\\pi_0$} & \\textbf{+CGVD} & \\textbf{$\\Delta$ [95\\% CI]} \\\\")
        tl.append("\\hline")
        for c in counts_both:
            row = [str(c)]
            for arm in ("simple", "complex"):
                s = attr_stats[(arm, c)]
                lo, hi = s["delta_ci_paired"]
                dtxt = f"{s['delta']:+.1f} [{lo:+.1f},{hi:+.1f}]"
                if lo > 0 or hi < 0:
                    dtxt = "\\textbf{" + dtxt + "}"
                row += [f"{s['baseline_mean']:.1f}", f"{s['cgvd_mean']:.1f}", dtxt]
            tl.append(" & ".join(row) + " \\\\")
        tl.append("\\hline\n\\end{tabular}%\n}\n\\end{table}")
        with open(os.path.join(GEN, "attribute_table.tex"), "w") as fh:
            fh.write("\n".join(tl) + "\n")

    # ---------- R29: replication / noise-floor checks ----------
    rep = []
    rep.append("== 0-distractor physical-identity check (semantic vs control at n=0 are the same condition) ==")
    for model in ("pi0", "gr00t"):
        for task in ("spoon", "carrot"):
            a = results.get((model, task, "semantic", 0))
            b = results.get((model, task, "control", 0))
            if a and b:
                rep.append(f"{model} {task}: baseline {a['baseline_mean']:.1f} vs {b['baseline_mean']:.1f} "
                           f"(|diff| {abs(a['baseline_mean']-b['baseline_mean']):.1f} pp); "
                           f"CGVD {a['cgvd_mean']:.1f} vs {b['cgvd_mean']:.1f} "
                           f"(|diff| {abs(a['cgvd_mean']-b['cgvd_mean']):.1f} pp)")
    rep.append("")
    rep.append("== duplicate-run replication estimates (older batch vs latest) ==")
    for key, dirs in sorted(older.items()):
        for d in dirs:
            f = os.path.join(d, "results.csv")
            if not os.path.exists(f):
                continue
            rows = load_results_csv(f)
            if not rows:
                continue
            s_old = condition_stats(rows)
            s_new = results.get(key)
            if s_new:
                rep.append(f"{key}: older {os.path.basename(d)} baseline {s_old['baseline_mean']:.1f} / "
                           f"cgvd {s_old['cgvd_mean']:.1f}  vs latest baseline {s_new['baseline_mean']:.1f} / "
                           f"cgvd {s_new['cgvd_mean']:.1f}")
    rep.append("")
    rep.append("== Table II reuse check: does pi0/spoon/semantic/n18 reproduce 43.0 / 77.5? ==")
    if full:
        rep.append(f"pi0 spoon semantic n18 ({full['dir']}): baseline {full['baseline_mean']:.1f}, "
                   f"cgvd {full['cgvd_mean']:.1f}  (paper Table II: 43.0 / 77.5)")
    with open(os.path.join(OUT, "replication_checks.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")

    # ---------- Timing provenance (Table III) ----------
    tim = []
    if full:
        f = os.path.join(ROOT, full["dir"], "results.csv")
        rows = load_results_csv(f)
        def col(name):
            vals = [float(r[name]) for r in rows if r.get(name) not in (None, "", "nan")]
            return (np.mean(vals), np.std(vals), len(vals)) if vals else (float("nan"),) * 3
        for name in ("init_time_ms", "avg_runtime_time_ms", "baseline_time", "cgvd_time",
                     "cgvd_pipeline_time", "sam3_time", "lama_time"):
            if rows and name in rows[0]:
                m, sd, n = col(name)
                tim.append(f"{name}: mean {m:.1f}, sd {sd:.1f} (n={n})")
        tim.append("paper Table III: init 4914 ms; per-step base 317 ms vs CGVD 421 ms (+104)")
    with open(os.path.join(OUT, "timing.txt"), "w") as fh:
        fh.write("\n".join(tim) + "\n")

    # ---------- console summary ----------
    print(f"conditions analysed: {len(results)}")
    conf = results.get(("pi0", "spoon", "semantic", 18))
    if conf:
        lo, hi = conf["delta_ci_paired"]
        print(f"CONFIRMATORY (pi0 spoon semantic 18): delta {conf['delta']:+.1f} pp, "
              f"paired 95% CI [{lo:+.1f}, {hi:+.1f}], Wilcoxon p={conf['wilcoxon_p']:.4f}")
    iccs = [s["icc_baseline"] for s in results.values() if not math.isnan(s["icc_baseline"])] + \
           [s["icc_cgvd"] for s in results.values() if not math.isnan(s["icc_cgvd"])]
    print(f"ICC(seed) across conditions: median {np.median(iccs):.3f}, "
          f"IQR [{np.percentile(iccs,25):.3f}, {np.percentile(iccs,75):.3f}]")
    print("outputs in analysis/out/ and paper/generated/")


if __name__ == "__main__":
    main()
