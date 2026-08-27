# CGVD Revision — where everything lives

All revision work is on git branch **`revision-panel-r1`**.

## Read these first
| File | What it is |
|---|---|
| `REVISION_CHANGELOG.md` | **The master status document** — every review item (R1–R33) → what changed, where, and its status. This is the response-to-reviewers skeleton. |
| `PHASE0_INVENTORY.md` | What was found on this machine: LaTeX, code, logs, the distractor vocabulary D, every verified hyperparameter. |
| `CGVD_REVISION_BRIEF.md` | The instructions this revision executes (from the review panel). |

## The paper
| Path | What it is |
|---|---|
| `../paper/root.tex` | The revised ICRA LaTeX (compiles with `pdflatex root && bibtex root && pdflatex root && pdflatex root`). Red `[TODO: …]` markers = data or decisions that must come from you — never guessed values. |
| `../paper/root.pdf` | Last compiled PDF. |
| `../paper/generated/` | Auto-generated tables (`main_results_table.tex`, `attribute_table.tex`) — regenerate with `python analysis/phase2_analysis.py`, do not hand-edit. |
| `overleaf_source.zip` | Your original Overleaf upload (untouched baseline; also git commit c810075). |

## The reviews (inputs)
`seat1_venue_fit.md` (AE / venue), `seat2_methodology.md` (stats), `seat3_domain.md` (VLA domain), `seat4_perspective.md` (deployment/safety), `seat5_devils_advocate.md`, `paper_transcription.md` (arXiv v1 text).

## Analysis & experiments
| Path | What it is |
|---|---|
| `../analysis/phase2_analysis.py` | Re-analysis of all rollout logs: per-seed rates, paired CIs, ICC, trend tests, replication checks. Re-run any time; safe. |
| `../analysis/out/` | Its outputs: `main_results.csv`, `per_seed.csv`, `ablation.csv`, `attribute.csv`, `trend_tests.txt`, `replication_checks.txt`, `timing.txt`. |
| `../analysis/run_revision_queue.sh` | GPU queue stage 1 (done): ① Table I Simple-arm re-run ② n18 replication + target-erasure audit ③④ OOV conditions (v1, superseded). |
| `../analysis/run_revision_queue2.sh` / `2b.sh` | Stage 2 (done): SAM3 robot-mask substitution (R33); stage 2b (done): OOV v2 with frozen vocabulary + descriptive-target attribute arm. |
| `../analysis/run_revision_queue3.sh` | Stage 3: BYOVLA comparator — `smoke` (done) then `full` (running). |
| `../analysis/parse_new_runs.py` | Parses erasure audit, timing, robot-mask stats from `runlogs/` → `out/new_runs_summary.txt`. |
| `../analysis/runlogs/` | Raw stdout of every revision run (gitignored; kept on disk as provenance). |

New rollout logs (gitignored, on this machine): `../logs/attribute_spoon_simple{,_desc}`, `../logs/replication_masks_n18`, `../logs/oov_n18_{miss,generic}_v2`, `../logs/robotmask_sam3_n18`, `../logs/byovla_n18{,_smoke}`.
