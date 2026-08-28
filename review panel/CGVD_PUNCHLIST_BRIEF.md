# CGVD Round-2 Punch List — final pre-submission pass

You are executing the final revision pass on **"Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation" (CGVD)**, targeting ICRA. A five-seat simulated re-review of the current draft returned **Minor Revision**: all Round-1 blockers are discharged, all four Devil's Advocate CRITICALs withdrawn, zero regressions. What remains is **wording, disclosure, citation accuracy, repo hygiene, and a page-count check — no new experiments are required**. Your job is to execute items F1–F8 below exactly, on the LaTeX source and the code repository.

**Provided alongside this brief:** `revised_paper.txt` — a pdftotext extraction of the current draft, for locating the quoted passages (its figure-axis garbling and flattened tables are extraction noise; edit the LaTeX, never this file). The full review record is at https://claude.ai/code/artifact/ba260fea-4a80-47c6-985f-4beef8929ad3 if fetchable.

## Ground rules

1. **Never fabricate a number.** Where an item asks for a value (chunk length, vocabulary list, audit protocol, init timing), pull it from the code/configs/logs. If it is not recoverable, insert `\todo{ASK AUTHOR: ...}` and list it in the handoff — never a plausible guess.
2. **Do not soften the negative results or the self-damaging disclosures.** The carrot regression, the oracle-D admission, the 2.0% erasure rate, the 35% blending audit, and the fill-appearance concession are credited strengths. Every fix below adds precision or caveats; none removes honesty.
3. **Locate before editing**: find the LaTeX source and the repo (github.com/awd1779/pi0 or its local clone). If not present, ask the user for paths/access first. Work on a git branch (`punchlist-r2`); keep `REVISION_CHANGELOG.md` mapping each item → change → status.
4. **Ask before anything heavy.** The only candidate compute is optional (marked); everything required is editing.

---

## F1 — Front-matter register on the new results (Abstract, §I, §VI)

- Abstract, the BYOVLA sentence ("...BYOVLA reaches 32.5% at roughly 34× CGVD's per-step cost"): append a caveat clause, e.g. "—a dense-clutter regime outside BYOVLA's design envelope (§IV-E)". Keep §IV-E's word "total" wherever "per-step cost" appears (the Abstract currently drops it).
- §VI, "at the headline condition it outperforms a vocabulary-matched BYOVLA reimplementation by 45 pp at a small fraction of its per-step cost": add the caveat subordinate clause ("in a regime outside its design envelope, with its sensitivity threshold untuned") **and** fix the cost phrasing — 11.3 s is per *intervention* (per action chunk), not per step. Either state it per intervention or amortize per step using the true chunk length (see F3).
- Abstract, "CGVD substantially improves robustness to semantic clutter…and, under descriptive compositional prompts, to attribute clutter": the attribute half rests on ONE exploratory, unadjusted, 5-seed cell (+16.0 [+1.2, +30.8]) that §IV-C itself labels descriptive. Drop "substantially" from that conjunct, or rescope: "and, in one exploratory condition with compositional prompts, improves attribute adherence." Apply the same scoping touch to §I contribution 3 and §VI's attribute clause.
- §IV-E: change "BYOVLA reaches 32.5%—below the 43.0% baseline (∆ = −10.5, 95% CI [−25.7, +4.7])" → "numerically below the 43.0% baseline (∆ = −10.5, 95% CI [−25.7, +4.7], within noise)" — the paper's own rule says CIs containing zero are run-to-run noise. Optionally soften the opener "The result is one-sided."

## F2 — Abstract still misdescribes D (contradicts §III-A)

- Abstract: "parses the instruction into safe and distractor sets" → "parses the instruction into a safe set and applies a supplied distractor vocabulary". Add one clause naming the closed-vocabulary precondition (the Abstract currently states the static-scene bound but not the vocabulary bound; §VI already has both).
- §III opening, "The key insight: the instruction already specifies which objects matter": scope to the safe set (e.g. "...specifies which objects must remain visible; the clutter vocabulary is supplied separately (§III-A)").

## F3 — Comparator auditability (§IV-E)

- **Name the inpainter used in the BYOVLA arm** (read the reimplementation code — if it shares LaMa with CGVD, say "the shared LaMa inpainter", which also makes the flicker diagnosis clean; if it differs, disclose and note the confound).
- **State the action-chunk length** (steps per chunk, from the π0 config) and that the per-chunk intervention cadence follows the reference implementation. Then verify the "34×" arithmetic against the stated basis and correct the number or the label if needed (11.3 s ÷ chunk_len vs 334 ms/step, or keep intervention-vs-total-per-step with the basis explicit).

## F4 — Two undisclosed re-measurements vs the previous version (one sentence each)

- **Latency**: v1 reported 421 ms/step for CGVD; the draft reports 318.5 + 15.6 ms with no note. Check the git history / code for the true reason (the arithmetic suggests v1 folded amortized initialization into the per-step average: 318 + 4914/60 + 16 ≈ 416). Add one sentence to §IV-F, e.g. "Earlier internal figures folded amortized initialization into the per-step average; Table IV reports decomposed timings." State only what the history supports; else `\todo{ASK AUTHOR}`.
- **Attribute table**: the Simple-prompt column was re-executed under the instruction-consistent prompt protocol (the baseline column moved too), and n changed to 5 seeds/100 episodes. Add one sentence to §IV-C acknowledging the re-execution and the reason (removal of the scene-informed-prompt information asymmetry — a correction that ran against the authors' interest; say so). If a reason for 5 seeds exists, state it. *(Optional compute, ask first: extending this study back to 10 seeds would materially firm up the Abstract's attribute claim.)*

## F5 — Verifiability of the new evidence

- **Enumerate the generic 12-category vocabulary** in §IV-D (pull the exact list from the repo's `scripts/clutter_eval/distractors/` files); the miss-vocabulary is already enumerated, the load-bearing one is not. Also scope §III-A's forward-reference ("a generic site-level vocabulary recovers the oracle gain in full") to the tested operating point.
- **Realized vs nominal distractor count**: the off-scene placement fallback means realized count can fall below nominal at 14–18, where the headline and the +1.54 pp/distractor slope live. From logs, report the fallback frequency / realized-count distribution; if unrecoverable, add one sentence noting the fallback direction makes the slope conservative in realized-count terms.
- **Protocols for the two audits**: the 35% in-flight blending figure needs n, condition/arm, overlap definition and ground-truth source (mirror the erasure audit's specification); the 20.1% SAM3 no-detection rate needs its frame denominator.
- **The ≈3.3 s initialization figure** is "wall-time derived": measure it directly if logs/one cheap rerun allow (timer around the init call), else label it explicitly as an upper-bound estimate and note what the wall-time residual includes.

## F6 — Deployment implications (two-to-three sentences total, §IV-D/§V)

- **Vocabulary-miss semantics**: state that under a mis-specified vocabulary CGVD still runs the full pipeline — the scene is still edited, the cache still frozen, the erasure and blending risks still live — so 45.5% ≈ baseline means forfeited benefit, *not* absent intervention. Optionally add the free monitor: an (almost) empty distractor mask is a one-line vocabulary-miss detector.
- **20.1% fallback consequence**: one sentence on what a stale robot mask does (leading edge of the arm unprotected over former-distractor regions; no staleness bound), and that FK projection avoids the failure class entirely.
- **Equivalence over-read**: replace "primarily a latency-and-reliability privilege rather than an accuracy one" with a calibrated form: "no accuracy difference was detected (∆ = −2.0 pp); the experiment excludes accuracy costs larger than ≈11 pp."

## F7 — Page budget (do this early — it gates everything)

- Compile the ICRA two-column build. Hard limit 8 pages (6 + 2 at fee); the estimate from the review is ~8.3. If over, trim in this order: (i) §IV-E's mechanism-fidelity enumeration → footnote (~80 words); (ii) §IV-D's vocabulary and deployable-mask paragraphs each state their conclusion twice — deduplicate (~120 words); (iii) §III-C's worked spatula example, now redundant with the stated empty-set convention (~60 words); (iv) §IV-C's scene-informed-phrase digression (~50 words); (v) last resort: merge Table I's ∆ and CI into one column per arm-pair. Do not cut limitations, negative results, or caveats.

## F8 — Citation-accuracy and repo corrections (a web fact-check verified these; they are the items a Googling reviewer finds)

1. **OBEYED-VLA [8], two fixes.** (a) §I + §II-C mischaracterize it as "fine-tune attention adapters" / "train attention layers": per its abstract it *augments VLAs with a perception module* producing task-conditioned, object-centric, geometry-grounded observations, with the policy fine-tuned on clean single-object demos. Re-describe it accurately, move it beside BYOVLA in §II-D as observation-space prior work, and state the real distinction (CGVD trains and fine-tunes nothing). Also re-anchor or drop [8] as the citation for §I's feature-dilution hypothesis. (b) §IV-A's "we found no public implementations of DTP [28] or OBEYED-VLA [8]" is **false for OBEYED-VLA** — github.com/UARK-AICV/OBEYED_VLA has been public since 2025-12. Rescope: "OBEYED-VLA's released code covers its perception module but not the action policy or robot interface, so an end-to-end comparison was infeasible; DTP's only advertised release is an anonymized review-time repository that has since expired."
2. **Taxonomy credit**: §IV-A's "Following the distractor-type framing of [6]" is wrong — [6] uses a continuous clutter measure and explicitly removes ambiguity-inducing distractors. Claim the Semantic/Random/Attribute taxonomy as this paper's own; cite [6] only for the clutter-degradation magnitude. Also fix §I's similarity-concentration claim ("failure concentrates around distractors sharing visual or semantic properties… [6]"): [6] only relays prior reports — cite its underlying sources or soften to "consistent with prior reports [6]".
3. **Reference [7] (Eva-VLA)**: restore the dropped second author (Shouwei Ruan; every later position shifts), and stop citing it as clutter evidence — its axes are 3D transforms, illumination, adversarial regions. Remove it from the "[6], [7]" clutter-degradation citations or recategorize.
4. **Reference [4]**: add a footnote/parenthetical that arXiv 2503.14734 documents GR00T **N1** while the evaluated checkpoint is **N1.6** (different backbone/head); point to the NVIDIA N1.6 model card as the secondary reference.
5. **Repo (github.com/awd1779/pi0)**: the README is still the unmodified upstream open-pi-zero README with zero mention of CGVD — add a CGVD section (or CGVD.md) pointing at `src/cgvd/` and `scripts/clutter_eval/`, with a mapping note that the repo's `control` distractor set = the paper's Random type. **Release the BYOVLA reimplementation and the vocabulary-sensitivity scripts** — §IV-E and the B2-discharging experiment are currently unreproducible from the released artifact, which contradicts §IV-A's availability sentence.

---

## Deliverables

1. Edited LaTeX compiling at ≤8 pages, branch `punchlist-r2`.
2. Repo updates (README/CGVD.md + the missing scripts committed).
3. `REVISION_CHANGELOG.md`: every F-item → what changed, where, status (done / todo-author / todo-compute).
4. A short ASK-AUTHOR list for anything unrecoverable from code/logs (expected candidates: the true latency-change reason if git history is silent; the 35%-audit protocol details if unlogged).

Work order: F7's compile first (it constrains the additions), then F8 (factual), then F1–F6 (wording), then re-compile and confirm the page count.
