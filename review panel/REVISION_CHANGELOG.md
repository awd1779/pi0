# CGVD Revision Changelog — response-to-reviewers skeleton

Statuses: **done** | **todo-data** (needs data the author must supply/locate) | **todo-compute** (needs GPU runs, awaiting approval) | **skipped+why**.
Every item ID is from `CGVD_REVISION_BRIEF.md`; findings are in the seat reports (same folder).

## Phase 0 — Setup
| Item | Status | Notes |
|---|---|---|
| Locate LaTeX / code / logs / D | done | See `PHASE0_INVENTORY.md`. LaTeX at `paper/root.tex` (from user's Overleaf zip). D is derived from the injected asset list → oracle vocabulary confirmed; Phase 3 gate triggers. |

## Phase 1 — Tier A editing pass (all edits in `paper/root.tex` unless noted; compare against baseline commit c810075)
| Item | Status | Where | Notes |
|---|---|---|---|
| R1 superiority claim | done | Abstract, §I contrib. 3, §IV-A | "significantly outperforms state-of-the-art baselines" removed; explicit "all comparisons are against the un-augmented policies"; availability sentence added (BYOVLA public; no public DTP/OBEYED found as of Aug 2026 — verified via web search) |
| R2 headline condition | done | Abstract, §I, §IV-D | every 77.5/43.0 now reads "(π0, *put spoon on towel*, 18 semantic distractors; Table II)" |
| R3 carrot regression surfaced | done | Abstract, §I contrib. 3, §V, §VI | "prevents performance collapse" now conditioned on *semantic* clutter; regression stated in Abstract, contribution list, Limitations, Conclusion; §V "slight degradation" replaced with §IV-B's "consistently underperforms" |
| R4 OOD claim | done | §VI | deleted; "highly efficient" → "moderate (+32.8%)" |
| R5 latency wording | done | §IV-E, Table III caption, §V | one consistent description: +104 ms (+32.8%), ≈3.15→2.38 Hz; "negligible…native control frequency" removed; "negligible compared to mechanical movement time" dropped; hardware/reps = \todo (provenance unknown, see open questions) |
| R6 significance captions | done | Fig. 3 + Table II captions | phrase deleted; plain episode counts; Fig. 3 caption also warns episodes within a seed are not independent; Table I "non-monotonic variance" wording removed (CI statement todo for Phase 2) |
| R9 mechanism → hypothesis (B3) | done | Abstract, §I, §II-A, §II-D, §IV-B, Fig. 4 | all feature-dilution/attention-corruption claims now "we hypothesize"; Fig. 4 retitled "Qualitative illustration of the attention-repair hypothesis" + entailment caveat + \todo for extraction details (candidate: `scripts/visualize_pi0_attention.py` — π0 backbone hooks, softmax(QKᵀ/√d), query→256 SigLIP tokens, mean over layers+heads — author must confirm); Figs. 1, 2, 4 now cited from body; §IV-B names the two rival explanations (distribution restoration, choice-set reduction) and links mean-color-fill evidence |
| "Architecturally excluded" | done | §I, §IV-D, §III-B | → "excluded conditional on correct safe-set segmentation", cross-referenced to the refinement-ablation failure |
| R10 ablation scope | done | §IV-D | scoped to operating point; pi0-choice justification fixed ("headroom", not "more demanding testbed") |
| R11 hyperparameter table | done | §IV-A Table (tab:hyper) | all values pulled from code (see PHASE0_INVENTORY.md); availability statement = \todo URL |
| R12 selection statement | done (with \todo) | §IV-A | "fixed during development, held constant" + \todo author confirmation that no tuning used the reported cells |
| R13 Eq. (3) empty-set convention | done | §III-D | max over empty set = 0 → g = σ_safe (verified in `_compute_genuineness`); η in tab:hyper (0.30) |
| R14 protocol paragraph | done | §IV-A | success predicate, 60-step horizon, seed semantics (RNG seed; episode_id=(s+e)%24 of 24 canonical layouts; distractor seed 10⁴s+e), placement-failure rule (random fallback then hidden off-scene; episode never dropped), "no episodes excluded" (verified: 48/48 Fig.-3 results.csv have exactly 200 rows), counts {0,4,8,10,14,18} |
| R15 α definition etc. | done | §III-G | α equation + fixed-at-t0 statement; robot overwrite = only per-frame part; distilled frame = sole visual input; false-positive-erasure failure mode named incl. post-t0 intrusion; deployment scoped to static/human-excluded/fixed-camera/known-vocabulary; FK-projection named as intended real-robot path (Phase 4 sentence) |
| R8+R20 limitations rewrite | done | §V, §IV-A | limitations now: static scene (intrusion = hazard), contextual clutter regression, sim-only + two privileged inputs, single-target-instance, occlusion fabrication, startup latency; §IV-A sim-to-real bounding claim replaced with scoped version |
| R16 Table II relabel | done | Table II | row "LaMa → mean-color fill"; caption "removes or substitutes"; adversarial-patch → "we hypothesize… out-of-distribution patches… not measured" |
| R17 §II restructure | done | §II | §II-B retitled "Vision Foundation Models for Robotic Perception"; old §II-C duplication deleted; new §II-C "Adaptation and Training-Time Augmentation" covers [obeyed]+[rosie,genaug,nice] incl. NICE positioning paragraph; taxonomy now matches §I; [6] credited for distractor-type framing in §IV-A |
| R21 §II-A causal fixes | done | §II-A, §II-B, §II-D | action-discretization clause removed (π0 flow-matching, GR00T diffusion); "high-pass filter" removed, frequency language dropped; BC-IB bottleneck explicitly analogical; intriguing_vit citation removed from both disputed uses (claims now stated as conjecture without false anchor) |
| R19 M_robot/r_e, robot role | done | §III-B, §III-F | S = {c_tgt, c_anc}; robot = separate channel; M_robot + r_e defined at first use in Eq. (6) |
| R18 §IV-C type conflation | done | §IV-C | "0–4 attribute distractors"; coverage asymmetry stated (π0-only, one task, 0–4); CGVD prompt strings disclosed (verified from shell history) |
| R22 geometry qualifier | done | Abstract, §III-F, §V | "preserving background context and scene layout outside the edited regions"; t=0 occlusion-fabrication consequence in §III-F + §V |
| R23 copy-edit sweep | done | throughout + references.bib | fragment fixed, "in clutter", articles, Fig. 2 "tar"→(target, anchor), Table I headers π0/π0+CGVD, "SAM 3" unified, task names italic lowercase, robogen author list restored, Open X-Embodiment corporate author, obeyed title corrected to "Clutter-Robust" (per arXiv), author-name typo "Sangmim"→"Sangmin" in \thanks (email left as-is — author should verify) |
| Page budget | in progress | — | compiles cleanly, currently 9 pp. with red \todo markers; final squeeze after Phase 2 replaces todos with data and adds the main-results table; planned cuts: Fig. 1 (~⅓ col) and further §II trims if needed |

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
