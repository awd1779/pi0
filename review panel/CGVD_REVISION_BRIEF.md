# CGVD Revision Brief — execute priorities 1–5 from the review panel

You are revising the paper **"Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation" (arXiv 2603.10340)** — a training-free SAM3+LaMa perception wrapper for VLA policies, evaluated in SimplerEnv on π0 and GR00T. A simulated five-seat ICRA review panel returned **Major Revision (4/4 seats)**. Your job is to execute the revision work list below (priorities 1–5), producing a submission-ready draft for ICRA.

**Supporting material provided alongside this brief** (uploaded with it, or in a `review_panel/` folder — locate them before starting): the five full referee reports (`seat1_venue_fit.md`, `seat2_methodology.md`, `seat3_domain.md`, `seat4_perspective.md`, `seat5_devils_advocate.md`) and a text transcription of the current paper (`paper_transcription.md`). Every R-number below is defined in the reports; read the relevant finding before making its fix. The consolidated decision is at https://claude.ai/code/artifact/ba260fea-4a80-47c6-985f-4beef8929ad3 if you can fetch it, but this brief plus the reports are sufficient. If any of these files are missing from your environment, ask the user to upload them before proceeding.

---

## Ground rules (non-negotiable)

1. **Never fabricate a number.** Every new statistic (CI, per-seed value, ICC, GR00T number, carrot-on-plate number, timing) must be computed from real logs or real runs. If the data isn't available, insert a clearly marked `\todo{...}` placeholder and list it in the handoff report — never a plausible-looking value.
2. **Preserve and promote the negative results.** The panel unanimously praised the honest carrot-on-plate regression and Table I's negative deltas. Do not soften, remove, or bury them — the required direction is the opposite: surface them in the Abstract and Conclusion.
3. **Locate before editing.** First find the actual LaTeX source and the experiment code/logs. If they are not in the working directory, ask the user for the paths before doing anything else. Work on a git branch (`revision-panel-r1`) if the source is a repo; otherwise keep timestamped backups before editing.
4. **Ask before long compute.** Prepare scripts and configs for anything needing GPU rollouts (Phases 3b, 4, 5), report the estimated runtime, and get the user's go-ahead before launching jobs longer than ~30 minutes.
5. **Verify claims against the source, not this brief.** Quotes below are from the arXiv v1 transcription; the local source may have drifted. Match on meaning, not exact bytes.
6. **Keep a change log.** Maintain `REVISION_CHANGELOG.md` mapping every item ID below → what was changed, where, and its status (done / todo-data / todo-compute / skipped+why). This becomes the response-to-reviewers skeleton.

**Branch decisions already made by the author (bake these in):**
- **B1 → claim-softening baseline:** rewrite all superiority text to "compared against the un-augmented policies" now. If Phase 5's BYOVLA comparison later succeeds, a measured comparative sentence may be reinstated using its actual numbers.
- **B3 → downgrade branch:** attention/feature-dilution language becomes *hypothesis*, not fact. No attention-measurement experiment is planned.

---

## Phase 0 — Setup

- Locate: (a) LaTeX source of the paper; (b) experiment code (the CGVD pipeline, SimplerEnv eval harness); (c) rollout logs — per-episode outcomes, seeds, masks if saved; (d) the config that defines the distractor vocabulary D. Report what exists before proceeding.
- Read `review_panel/paper_transcription.md` end to end, then skim the five reports (Weaknesses sections at minimum).

## Phase 1 — Tier A editing pass (LaTeX only, no experiments)

**Claim language**
- R1: Abstract + §I contribution 3 — remove "significantly outperforms state-of-the-art baselines"; state comparisons are against un-augmented π0/GR00T. Add one sentence noting whether DTP [30] / OBEYED-VLA [8] have public implementations (check; if not, say so — silence reads as omission).
- R2: Everywhere 77.5/43.0 appears — attach its condition: "(π0, *put spoon on towel*, 18 semantic distractors; Table II)".
- R3: Abstract + §VI — add the carrot-on-plate regression (one clause/sentence each); replace "prevents performance collapse" and "critical prerequisite" with distractor-type-conditional wording ("under semantic and attribute clutter"); reconcile §V "slight degradation" with §IV-B "consistently underperforms".
- R4: §VI — delete "improves generalization to out-of-distribution targets" (or restate as attribute-adherence under compositional prompts per Table I). Reassess "highly efficient".
- R5: §IV-E + Table III caption + §V — one consistent latency description: +104 ms/step (+32.8%), ≈3.15→2.38 Hz; remove "negligible… maintaining the VLA's native control frequency"; drop the "negligible compared to mechanical movement time" comparison unless a measured movement time is added; add hardware/repetitions/dispersion to Table III (from logs, or `\todo{}`).
- R6: Fig. 3 + Table II captions — delete "To ensure statistical significance"; state plain episode counts. Reserve the word "variance" for reported dispersion.
- R9 (B3 downgrade): Abstract, §I, §II-A, §II-D — every "attention corruption", "feature dilution", "Precision-Reasoning Gap"-as-mechanism assertion becomes hypothesis language ("we hypothesize", "consistent with"). Retitle Fig. 4 (e.g., "Qualitative illustration of the attention-repair hypothesis"), cite Figs. 1, 2 and 4 from body text (currently uncited), and reconcile the Abstract's causal sentence with Table II's mean-color-fill row (61% of the gain tracks fill appearance — see DA report C2).
- "Architecturally excluded" (§I) → "excluded conditional on correct safe-set segmentation" (seat3 W9; §IV-D itself reports targets being erroneously inpainted).
- R10: §IV-D — scope component-attribution language to the tested operating point only.

**Disclosure / specification**
- R11: Add a compact hyperparameter table: η, r_d, r_s, r_e, compositing Gaussian σ, SAM3 confidence thresholds + exact prompt strings, LaMa checkpoint, π0/GR00T checkpoints, SimplerEnv variant, placement-grid resolution, seeds, hardware. **Pull every value from the actual code/configs** — this is a Phase 0 dependency. Add a one-line code/video availability statement.
- R12: One sentence on how those values were selected and on what data ("upstream repository defaults" is acceptable if true — verify in code).
- R13: Eq. (3) — state the empty-set convention (what g(s_i) is when no distractor exceeds η; read the actual implementation and document what the code does).
- R14: §IV-A — protocol paragraph: success predicate used, episode horizon, what each seed randomizes, placement-failure rule at high densities, episodes attempted vs analysed (verify in the eval harness).
- R15: §III-G — define α precisely (from code: equation or procedure; fixed at t=0 or not); state whether the distilled frame is the policy's only visual input; name false-positive erasure of a physically present object as the characteristic failure mode; scope §I/§VI deployment framing to static, human-excluded workspaces with fixed camera.
- R8 + R20: §V — add "simulation-only evaluation" and "single-target-instance assumption" as explicit limitations; declare the ground-truth robot mask simulation-only with replacement cost unmeasured (until Phase 4 measures it); soften §IV-A's "sim-to-real risk is bounded to policy-level transfer"; fix "two key assumptions" vs three paragraphs.

**Structure / accuracy / copy**
- R16: Table II — relabel "– Mean-color fill" → "LaMa → mean-color fill"; caption → "removes or substitutes one component"; mark the "adversarial patches" explanation as a hypothesis.
- R17: §II restructure — retitle §II-B (its content is vision foundation models); merge §II-C into §I/§II-A and delete the remainder (it duplicates §I's fourth paragraph); align §II subsections with §I's three categories (adaptation / inference-time / training-time augmentation — the last currently has no subsection); add a short paragraph positioning against [10]–[12], especially [12] (NICE scene surgery — nearest pixel-space prior); credit [6] for the semantic/random/attribute taxonomy if it is the source. **This is also the main source of page-budget space.**
- R21: §II-A — remove action-discretization as a cause (π0 is flow-matching, GR00T N1 is diffusion-head; neither discretizes); fix "high-pass filter that blocks clutter" (wrong direction — pick the semantic framing and drop frequency language); present [24]'s information bottleneck as analogy; verify [18] (Naseer et al.) actually supports the use made of it, else re-anchor.
- R19: Define M_robot and r_e at first use; reconcile the robot's role across §III-B (in safe set S), Eq. (2) (absent from M_safe), Eq. (6) (added to inpainting mask).
- R18: §IV-C — fix "0–4 random attribute distractors" (conflates two defined types); state the coverage asymmetry (attribute type: Table I only, π0 only, 0–4 counts).
- R22: Abstract — qualify "preserving critical spatial geometry"; note in §V that Eq. (6) fabricates whatever the arm occluded at t=0 for the whole episode.
- R23: Copy-edit sweep: §IV-B sentence fragment ("Conversely, in the Carrot on Plate task."); "in the clutter" → "in clutter"; §I contribution-2 trailing comma; "If model misidentifies" → "If the model misidentifies"; "To make inpainting artifacts do not obscure" → "To ensure that inpainting artifacts do not obscure"; §III-F missing article/comma; Fig. 2 caption's truncated "tar" → "target"; Table I column headers → "π0" / "π0 + CGVD"; move Fig. 3 adjacent to §IV-B; unify "SAM3"/"SAM 3" and task-name capitalization; fix refs [22] (author list) and [27] (corporate author).

**Page budget (do alongside R17):** target ≤8 pages IEEE two-column. Recover: merge/delete §II-C (~¼ col), compress §II-B to ~3 sentences (~⅕ col), compress §III-A to ~3 sentences (~⅙ col), drop Fig. 1 or make it single-column (~⅓ col). Spend on: numeric main-results table (~0.4 col), hyperparameter table (~0.15 col), comparator rows if Phase 5 runs (~0.3 col).

## Phase 2 — Tier B re-analysis (existing logs only; no new rollouts)

If per-episode/per-seed logs exist, write analysis scripts (Python) and generate:
- R24: Per-seed success rates and per-seed deltas for Tables I–II and the Fig. 3 grid; a paired analysis over the 10 matched seed pairs (Wilcoxon signed-rank at minimum; preferably a mixed-effects logistic regression: method fixed effect, random intercept per seed, distractor count covariate, cluster-robust CIs). Report the observed ICC.
- R25: 95% CIs (or per-seed spread) beside every headline number; identify which Table I deltas exclude zero; then either delete the 1-distractor "non-monotonic variance" interpretation or state it is within run-to-run noise. (Reference: the methodology report recomputed unpaired CIs — 6 of 10 Table I deltas contain zero.)
- R26: Declare one confirmatory contrast (the 18-semantic-distractor spoon-on-towel cell), apply correction within the declared family, label the rest exploratory.
- R27: A numeric main-results table replacing prose-only §IV-B: both policies × both tasks × both clutter types × all six distractor counts, with dispersion — **this is where GR00T and carrot-on-plate finally get numbers**. List the six distractor-count settings explicitly in §IV-A. Support "the gap widens" with a method × density interaction estimate.
- R28: From logged masks, compute the target/safe-set false-erasure rate at the headline condition (with and without the two-layer refinement if logs allow). If masks weren't logged, note it and keep the conditional-guarantee wording from Phase 1.
- R29: Spread across the physically identical 0-distractor conditions (free noise floor — the methodology seat's top-priority item); state whether Table II's 43.0/77.5 rows reuse the Fig. 3 runs.
- R30: State Table I's rounding convention / analytic n (all 20 cells are integers on a 0.5-pt attainable grid at n=200; Table II uses half-integers — explain the discrepancy from the logs).

If logs do NOT exist for some part, mark the affected items todo-data in the changelog and tell the user exactly which runs must be re-executed to obtain them.

## Phase 3 — R7: the distractor vocabulary D

- **Find D's construction in the code** (config file, hard-coded list, or derivation). Write the 1–2 sentence disclosure in §III-B/§IV-A stating exactly how D was built for the experiments and how it would be built at deployment. Scope the "no additional API" claim to the closed-vocabulary setting.
- **Decision gate:** if D overlaps or derives from the injected RoboCasa/YCB category list → **R32 becomes required**: prepare an out-of-vocabulary condition (inject distractor categories not in D) at the 18-distractor spoon-on-towel cell, ~200–400 rollouts, reported beside the headline. Prepare the config/script, estimate runtime, ask before launching.

## Phase 4 — R33: SAM3 robot-mask substitution (~200 rollouts + timing)

- Implement a variant where the compositing robot mask is **SAM3-predicted per frame** instead of SimplerEnv's ground-truth mask. Re-run the Table II robot-mask condition (same seeds, 200 episodes) and report the success-rate delta.
- Measure per-frame SAM3 robot-mask latency on stated hardware; report the reconciled real per-step time next to the 421 ms figure.
- Add one sentence naming forward-kinematics-projected self-masking as the intended real-robot deployment path.
- Prepare first, confirm with the user, then run.

## Phase 5 — R31: BYOVLA comparator (~400 rollouts + reimplementation)

- Obtain or reimplement BYOVLA (Hancock, Ren & Majumdar, "Run-time observation interventions make VLA models more visually robust", ICRA 2025 — ref [9]; check for public code first) on the same SimplerEnv setup.
- Run at the 18-semantic-distractor spoon-on-towel condition, matched seeds, 200 episodes/arm. Report **success rate and wall-clock per step** side by side with CGVD.
- If it succeeds: reinstate a *measured* comparative sentence using the actual numbers (honest even if CGVD loses on some axis). If it cannot be made to run after reasonable effort: one sentence in the paper stating why, and keep the softened claims from Phase 1.
- Prepare first, confirm with the user, then run.

## Deliverables

1. Revised LaTeX compiling cleanly at ≤8 pages, on branch `revision-panel-r1`.
2. Analysis scripts + generated tables (Phase 2) checked into the repo.
3. `REVISION_CHANGELOG.md` — every item ID → change made, location, status.
4. A short list of open TODOs that need the author (missing data, decisions, compute approvals).

Work through phases in order (Phase 1 has no dependencies; Phases 2–5 depend on what Phase 0 finds). Report progress after each phase.
