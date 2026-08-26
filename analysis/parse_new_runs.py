#!/usr/bin/env python3
"""Parse revision-run artifacts that live in logs, not results.csv:
- R28: per-episode TARGET_ERASURE stats from cgvd.log (collapse consecutive
  duplicate lines; the cached mask is constant within an episode).
- Table III: per-episode `vla=XXX.Xms/f` prints from the queue logs
  (per-inference-step forward time incl. any wrapper overhead), split by arm.
- R33: SAM3_ROBOT_MASK per-frame latency and detection reliability.
Outputs to analysis/out/new_runs_summary.txt.
"""
import glob, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
os.makedirs(OUT, exist_ok=True)

lines_out = []

def p(s):
    print(s)
    lines_out.append(s)

# ---------- R28: target erasure (replication run) ----------
# TARGET_ERASURE prints only reach the queue stdout log; slice out job 2.
er_eps = []
qlog = os.path.join(ROOT, "analysis/queue_run.log")
if os.path.exists(qlog):
    text = open(qlog).read()
    m = re.search(r"=== \[2/4\].*?===(.*?)=== \[2/4\] done", text, re.S)
    if m:
        # Episode boundary = the per-episode progress line printed after each
        # episode ("  CGVD  ep N/20 ..."). First frac in an episode = t=0
        # erasure (mask built on that frame); max frac = in-flight exposure of
        # the (moving) target to previously-inpainted regions.
        first_fracs, max_fracs, cur = [], [], []
        for line in m.group(1).splitlines():
            mm = re.search(r"TARGET_ERASURE frac=([0-9.]+)", line)
            if mm:
                cur.append(float(mm.group(1)))
            elif re.search(r"CGVD\s+ep \d+/\d+", line):
                if cur:
                    first_fracs.append(cur[0])
                    max_fracs.append(max(cur))
                cur = []
        er_eps = first_fracs
        if max_fracs:
            mx = np.array(max_fracs)
            lines_out.append("")  # spacer
            p(f"R28 in-flight exposure (max per-episode frac, target moves through scene): "
              f"episodes with max>1%: {(mx>0.01).sum()}/{len(mx)} ({100*(mx>0.01).mean():.1f}%), "
              f">10%: {(mx>0.10).sum()} ({100*(mx>0.10).mean():.1f}%), median max {np.median(mx)*100:.1f}%")
if er_eps:
    er = np.array(er_eps)
    p(f"R28 target-erasure audit (full pipeline, n18 semantic replication): "
      f"{len(er)} episode-level entries")
    p(f"  episodes with ANY target pixel inside the inpaint mask (frac>0): "
      f"{(er>0).sum()} ({100*(er>0).mean():.1f}%)")
    p(f"  episodes with frac>1%: {(er>0.01).sum()} ({100*(er>0.01).mean():.1f}%); "
      f">10%: {(er>0.10).sum()} ({100*(er>0.10).mean():.1f}%); "
      f">50%: {(er>0.50).sum()} ({100*(er>0.50).mean():.1f}%)")
    p(f"  mean frac {er.mean()*100:.2f}%, median {np.median(er)*100:.2f}%, max {er.max()*100:.1f}%")

# ---------- Table III: vla ms/f by arm ----------
for name, log in (("stage1", "analysis/queue_run.log"), ("stage2", "analysis/queue2_run.log")):
    path = os.path.join(ROOT, log)
    if not os.path.exists(path):
        continue
    base, cg = [], []
    for m in re.finditer(r"(Baseline|CGVD)\s+ep \d+/\d+.*?vla=([0-9.]+)ms/f", open(path).read()):
        (base if m.group(1) == "Baseline" else cg).append(float(m.group(2)))
    if base:
        b = np.array(base)
        p(f"{name} baseline vla ms/f: n={len(b)} mean {b.mean():.1f} sd {b.std():.1f} "
          f"median {np.median(b):.1f} IQR [{np.percentile(b,25):.1f},{np.percentile(b,75):.1f}]")
    if cg:
        c = np.array(cg)
        p(f"{name} CGVD-arm vla ms/f: n={len(c)} mean {c.mean():.1f} sd {c.std():.1f} "
          f"median {np.median(c):.1f} IQR [{np.percentile(c,25):.1f},{np.percentile(c,75):.1f}]")

# ---------- R33: SAM3 robot mask latency + reliability ----------
path = os.path.join(ROOT, "analysis/queue2_run.log")
if os.path.exists(path):
    ms, px = [], []
    for m in re.finditer(r"SAM3_ROBOT_MASK frame=(\d+) ms=([0-9.]+) px=(\d+)", open(path).read()):
        ms.append(float(m.group(2)))
        px.append(int(m.group(3)))
    if ms:
        a, q = np.array(ms), np.array(px)
        p(f"R33 SAM3 robot mask: n={len(a)} frames; latency mean {a.mean():.1f} ms "
          f"sd {a.std():.1f} median {np.median(a):.1f}")
        p(f"  frames with NO robot detected (px=0): {(q==0).sum()} ({100*(q==0).mean():.1f}%)")
        p(f"  detected-mask size median {np.median(q[q>0]) if (q>0).any() else 0:.0f} px")

with open(os.path.join(OUT, "new_runs_summary.txt"), "w") as fh:
    fh.write("\n".join(lines_out) + "\n")
