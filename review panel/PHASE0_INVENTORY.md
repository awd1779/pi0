# Phase 0 — Setup Inventory (CGVD Revision)

Date: 2026-08-26. Branch: `revision-panel-r1`.

## (a) LaTeX source — FOUND
- `paper/root.tex` — full ICRA source (user-uploaded Overleaf zip; also archived at `review panel/overleaf_source.zip`), with `references.bib`, `ieeeconf.cls`, `figures/`.
- `doc/methodology.tex` — longer methodology draft containing implementation details absent from root.tex (asymmetric thresholding, renderer-segmentation robot mask, robot-free initialization). Useful source for disclosure text; verified against code.

## (b) Experiment code — FOUND
- Pipeline: `src/cgvd/` (`cgvd_wrapper.py`, `sam3_segmenter.py`, `lama_inpainter.py`, `instruction_parser.py`, `distractor_wrapper.py`, `collision_tracker.py`, `grasp_analyzer.py`).
- Eval harness: `scripts/clutter_eval/batch_eval.py` (π0), `batch_eval_groot.py` (GR00T); run scripts `run_category_sweep_fast*.sh`, `run_attribute_spoon.sh`, `run_ablation_spoon.sh`.

## (c) Rollout logs — FOUND (with gaps)
Per-episode CSVs (`results.csv`) with: run, seed, episode, episode_id, baseline/cgvd success, hard success (success ∧ zero collisions), times, collision counts, failure mode, CGVD pipeline timing (init_time_ms, avg_runtime_time_ms), mask coverage.

| Paper element | Log location | Status |
|---|---|---|
| Fig. 3, GR00T (spoon+carrot × semantic+control × n∈{0,4,8,10,14,18}) | `logs/clutter_eval/gr00t/` | complete, e20_r10 |
| Fig. 3, π0 spoon (semantic+control, all 6 counts) | `logs/clutter_eval/pi0/spoon/spoon/` | complete, e20_r10 |
| Fig. 3, π0 carrot | `logs/clutter_eval/pi0/carrot/` + `pi0/carrot/DOne/` | complete but split across two batches (2026-02-17 and 2026-03-04/05); n18-control exists in BOTH batches — must decide which the paper used |
| Table I (attribute) | `logs/attribute_spoon/spoon/attribute/n0–n4_e20_r5` | **only one prompt arm locally**; 5 seeds × 20 eps = **n=100**, not the caption's claimed 10 seeds/200. Matches Table I *Complex* column exactly at n1–4; n0 baseline is 86.0 in logs vs 85.0 in the table. **Simple-prompt arm: missing locally (todo-data).** |
| Table II ablation (3 removal rows) | `logs/ablation_spoon_semantic_18dist/{cgvd_no_crossval, cgvd_no_inpaint, cgvd_no_robot}` | complete, n18 e20_r10 |
| Table II baseline 43.0 / full 77.5 | presumed reuse of π0 spoon semantic n18 run | verify in Phase 2 (R29) |
| Table III latency | per-episode init/runtime ms in results.csv | provenance of 4,914/317/421 and hardware: to establish in Phase 2 |
| Masks (for R28 false-erasure rate) | `run_*/{baseline,cgvd}/` dirs are **empty** (no videos/masks saved) | **todo-data** — needs an instrumented re-run with `--cgvd_save_debug` |

- Fig. 3's six distractor counts (never stated in the paper): **{0, 4, 8, 10, 14, 18}** — answers methodology-seat Q3.
- The paper's "Random" distractor type = category `control` in code/logs.

## (d) Distractor vocabulary D — FOUND, and the Phase 3 gate TRIGGERS
`batch_eval.py::load_distractors_from_file()` reads the *same* file that defines the injected distractors (e.g. `distractors_spoon_semantic.txt`: `rc_fork_0:0.1 …`) and derives the CGVD prompt list from the injected asset IDs (`rc_fork_0` → "fork"). **D is the category list of the injected clutter — closed-vocabulary, derived from the injection list.** R32 (out-of-vocabulary condition) becomes *required* per the brief's decision gate.

## Verified hyperparameters (from code — for R11/R13)
| Symbol / name | Value | Source |
|---|---|---|
| Safe-set (target/anchor) SAM3 threshold | 0.3 | `batch_eval.py` CLI default `--cgvd_safe_threshold` |
| Robot SAM3 threshold | 0.3 | CLI default (unused for masks in sim; robot mask from renderer) |
| Distractor SAM3 threshold | 0.20 | CLI default `--cgvd_distractor_threshold` |
| η (cross-val IoU threshold, Eq. 3) | 0.3 | hardcoded `iou > 0.3` in `_compute_genuineness` (the `distractor_iou_threshold=0.15` param is legacy, unused) |
| Eq. (3) empty-set convention | `max_dist_score` initialized to 0 → g(s_i) = σ_safe when no distractor overlaps | `cgvd_wrapper.py` cross-val loop (R13 answer) |
| r_d (distractor dilation) | 11 px (`lama_dilation`) | wrapper default |
| r_s (safe dilation, gating) | max(`safe_dilation`=5, `lama_dilation`=11) = **11 px**; r_s ≥ r_d enforced by the `max()` | `_step4_safe_dilation` |
| r_e (robot dilation in LaMa mask) | `_reinforce_size` = 11 + 3·⌈σ⌉ = **20 px** | `_build_inpaint_mask` |
| Compositing Gaussian σ | 3.0 (`blend_sigma`) | wrapper default |
| Mask binarization | 0.5 | wrapper |
| Min connected-component size | 50 px | `min_component_pixels` |
| Safe-set warmup frames | 1 | `safeset_warmup_frames` |
| Cache refresh | never (`cache_refresh_interval=0`) | wrapper |
| SAM3 checkpoint | `facebook/sam3` (HF) | `sam3_segmenter.py` |
| LaMa | `simple-lama-inpainting` (big-lama) | `lama_inpainter.py` |
| π0 checkpoint | open-pi-zero `bridge_beta.pt` | run scripts |
| GR00T checkpoint | `nvidia/GR00T-N1.6-bridge` (**paper says "GR00T"/cites N1 — actual model is N1.6**) | `batch_eval_groot.py` |
| Robot mask (sim) | SimplerEnv renderer segmentation buffer, per-link actor IDs | wrapper `_get_gt_*`; `doc/methodology.tex` |
| Placement | collision-aware fixed grid, 6 cm cells, one object/cell | `distractor_wrapper.py` |
| Episode horizon | 60 control steps | logs (`steps: 60`) |
| Success predicate | SimplerEnv built-in task success signal from `env.step()`; logs also record "hard success" = success ∧ zero collisions (not used in the paper) | `batch_eval.py` |
| Seed semantics | per run-seed s∈{0..9}: `random`/`np`/`torch` seeded with s (policy stochasticity); per episode: `episode_id=(s+ep)%24` selects one of SimplerEnv's 24 canonical initial layouts; `distractor_seed = s·10000+ep` randomizes distractor selection/placement | `batch_eval.py` run_episode |
| Hardware (this machine) | NVIDIA A10G 23 GB | nvidia-smi (Table III provenance TBD) |

## Immediate implications for the revision
1. **Table I caption is wrong as printed** — data is 5 seeds × 20 episodes (n=100), one prompt arm is not on this machine, and n0 (Complex baseline) disagrees with logs by 1 pt (86 vs 85). Needs author confirmation / the missing Simple-arm logs.
2. **D is oracle-derived** → R7 disclosure must say so; R32 OOV run required.
3. Eq. (5) as printed (r_s ≥ r_d) is correct *because* the code takes max(5, 11); the nominal `safe_dilation=5` alone would violate it — the disclosure should give effective values.
4. Masks were not logged → R28 needs a small instrumented re-run (~200 episodes) or stays conditional-wording only.
