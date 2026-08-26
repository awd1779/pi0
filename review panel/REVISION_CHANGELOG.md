# CGVD Revision Changelog — response-to-reviewers skeleton

Statuses: **done** | **todo-data** (needs data the author must supply/locate) | **todo-compute** (needs GPU runs, awaiting approval) | **skipped+why**.
Every item ID is from `CGVD_REVISION_BRIEF.md`; findings are in the seat reports (same folder).

## Phase 0 — Setup
| Item | Status | Notes |
|---|---|---|
| Locate LaTeX / code / logs / D | done | See `PHASE0_INVENTORY.md`. LaTeX at `paper/root.tex` (from user's Overleaf zip). D is derived from the injected asset list → oracle vocabulary confirmed; Phase 3 gate triggers. |

## Phase 1 — Tier A editing pass
| Item | Status | Where | Notes |
|---|---|---|---|
| R1 superiority claim | pending | Abstract, §I | |
| R2 headline condition | pending | Abstract, §I, §IV | |
| R3 carrot regression surfaced | pending | Abstract, §VI, §V | |
| R4 OOD claim | pending | §VI | |
| R5 latency wording | pending | §IV-E, Table III, §V | |
| R6 "statistical significance" captions | pending | Fig. 3, Tables I–II | |
| R9 mechanism → hypothesis (B3) | pending | Abstract, §I, §II, Fig. 4 | |
| "Architecturally excluded" | pending | §I | |
| R10 ablation scope | pending | §IV-D | |
| R11 hyperparameter table | pending | §IV-A | values verified in Phase 0 |
| R12 selection statement | pending | §IV-A | |
| R13 Eq. (3) empty-set convention | pending | §III-D | code: g = σ_safe when no overlap; η = 0.3 |
| R14 protocol paragraph | pending | §IV-A | |
| R15 α definition + sole-channel + failure mode | pending | §III-G | |
| R8+R20 limitations rewrite | pending | §V, §IV-A | |
| R16 Table II relabel | pending | Table II | |
| R17 §II restructure | pending | §II | |
| R21 §II-A causal fixes | pending | §II-A | |
| R19 M_robot/r_e definitions, robot role | pending | §III | |
| R18 §IV-C type conflation | pending | §IV-C | |
| R22 geometry-preservation qualifier | pending | Abstract, §V | |
| R23 copy-edit sweep | pending | throughout | |

## Phase 2 — Tier B re-analysis
| Item | Status | Notes |
|---|---|---|
| R24 per-seed + paired analysis | pending | logs available for Fig. 3 grid + ablation |
| R25 CIs beside headline numbers | pending | |
| R26 confirmatory contrast | pending | |
| R27 numeric main-results table | pending | GR00T + carrot numbers exist in logs |
| R28 false-erasure rate | todo-data | masks not logged (`run_*/cgvd/` empty); needs instrumented re-run |
| R29 0-distractor noise floor + Table II reuse | pending | |
| R30 Table I rounding/n | in progress | logs show n=100 (5 seeds), one arm only, n0 off by 1 pt vs table — needs author confirmation + missing Simple-arm logs |

## Phase 3 — R7 / R32 distractor vocabulary
| Item | Status | Notes |
|---|---|---|
| R7 disclosure of D | pending | D = category names of injected assets (oracle) — confirmed in `load_distractors_from_file()` |
| R32 OOV condition | todo-compute | REQUIRED (gate triggered). Config/script to prepare; needs approval |

## Phase 4 — R33 SAM3 robot mask
| Item | Status | Notes |
|---|---|---|
| R33 SAM3-mask substitution + timing | todo-compute | to prepare; needs approval |

## Phase 5 — R31 BYOVLA comparator
| Item | Status | Notes |
|---|---|---|
| R31 BYOVLA on same setup | todo-compute | check public code; needs approval |

## Open questions for the author
1. **Table I:** local logs = 5 seeds (n=100) matching the Complex column (n1–4 exactly; n0 86.0 vs printed 85.0). Where are the Simple-arm logs, and was the Complex n0 cell 85 or 86? If the missing data can't be located, the caption must say 5 seeds/100 episodes and the n0 discrepancy must be resolved from whatever source produced the table.
2. **π0 carrot n18-control appears in two run batches** (2026-02-17 `DOne/` and 2026-03-04) — which one fed Fig. 3?
3. **Table III:** which machine produced 4,914 / 317 / 421 ms (this A10G box, or RunPod)? Repetition count?
