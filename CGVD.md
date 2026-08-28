# CGVD — Concept-Gated Visual Distillation

This repository contains the reference implementation and full evaluation harness for
**"Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual
Distillation" (CGVD)** — a training-free, inference-time perception wrapper that removes
semantically confusable distractors from a VLA policy's observations via language-gated
SAM 3 segmentation and LaMa inpainting.

## Code map

| Path | Contents |
|---|---|
| `src/cgvd/cgvd_wrapper.py` | The CGVD pipeline (gym wrapper): concept gating, two-layer target refinement (cross-validation + spatial disambiguation), mask composition, LaMa clean-plate caching, temporally consistent compositing, robot-mask handling (`robot_mask_source="gt"` renderer buffer or `"sam3"` per-frame prediction), and the ground-truth target-erasure audit. |
| `src/cgvd/sam3_segmenter.py` | SAM 3 text-prompted instance segmentation (in-process or client–server). |
| `src/cgvd/lama_inpainter.py` | LaMa (big-lama) inpainting singleton. |
| `src/cgvd/instruction_parser.py` | Deterministic instruction → (target, anchor) parsing. |
| `src/cgvd/distractor_wrapper.py` | Controlled distractor injection: RoboCasa/YCB assets, collision-aware 6 cm grid placement with random/off-scene fallback. |
| `src/cgvd/byovla_intervention.py` | Vocabulary-matched BYOVLA reimplementation (per-object Gaussian-blur sensitivity probe, reference threshold 0.002 m, per-chunk cadence) used for the paper's §IV-E comparison. |
| `scripts/clutter_eval/batch_eval.py` | π0 evaluation harness (matched-seed paired protocol). Key flags: `--cgvd_distractor_names` (freeze/override the vocabulary D — used for the out-of-vocabulary study), `--cgvd_robot_mask_source gt|sam3`, `--eval_byovla` (+`--byovla_thresh`). |
| `scripts/clutter_eval/batch_eval_groot.py` | GR00T (N1.6) evaluation harness. |
| `scripts/clutter_eval/distractors/` | Distractor pool definitions per task/type. |
| `analysis/phase2_analysis.py` | Per-seed paired statistics (t₉ CIs, Wilcoxon, ICC, trend tests) and auto-generated paper tables. |
| `analysis/parse_new_runs.py` | Erasure-audit / timing / robot-mask-reliability parsing from run logs. |
| `analysis/run_revision_queue*.sh` | Exact invocations of every experiment in the paper's revision (replication, erasure audit, out-of-vocabulary cells, SAM 3 robot-mask substitution, BYOVLA comparison). |

## Terminology mapping (paper ↔ repo)

- The paper's **Random** distractor type = this repo's `control` category
  (`distractors_<task>_control.txt`, `--categories control`).
- The paper's **Semantic** and **Attribute** types = `semantic` and `attribute` categories.
- The paper's headline condition = `--task widowx_spoon_on_towel --categories semantic
  --distractor_counts 18 --episodes 20 --runs 10 --randomize_distractors --placement spread`.

## Reproducing the paper's experiments

All parameters are fixed in code defaults (see the paper's §IV-A Implementation paragraph).
The vocabulary-sensitivity (out-of-vocabulary) cells and the BYOVLA comparison are launched
exactly as in `analysis/run_revision_queue2b.sh` (jobs 6–7) and `analysis/run_revision_queue3.sh`.
Evaluations run inside the project Docker image with the repo bind-mounted; see
`docker/DEPLOY_RUNPOD.md`.
