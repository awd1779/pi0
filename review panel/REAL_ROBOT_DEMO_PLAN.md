# Real-Robot Qualitative Demo — Plan

Goal: a small, honest hardware demonstration that closes the paper's biggest stated gap
("simulation-only evaluation," §V) at *qualitative* strength — one paragraph plus
supplement clips, no statistical claims. Every reviewer seat listed this as the highest-value
strengthening item, and the original project notes planned it ("do qualitative analysis with
real robot").

## What it must show (tied to the paper's open claims)

1. **The pipeline runs on real images**: SAM 3 segments a real cluttered tabletop from the
   generic vocabulary, LaMa produces a usable clean plate, compositing is stable over an
   episode. This directly addresses §V's biggest unvalidated component (LaMa fill fidelity on
   real textured scenes — the paper's own largest ablation factor).
2. **The FK-projected robot mask works** — the deployment path §III-F/§IV-D argue for but
   mark untested.
3. **Behavioral effect (stretch goal, stage-gated)**: with a Bridge-like rig, baseline π0
   confuses utensils under clutter while CGVD does not — mirroring Fig. 1 on hardware.

## Hardware prerequisites (ASK-AUTHOR: what do you have access to?)

| Item | Requirement | Why |
|---|---|---|
| Arm | **Interbotix WidowX-250 (6DOF)** | bridge_beta is Bridge-trained; any other embodiment near-guarantees policy failure. (The BYOVLA reference code drives a wx250s via the Interbotix ROS stack — reusable as the action-interface template, `$CLAUDE_JOB_DIR`-cloned copy exists.) |
| Camera | Fixed third-person RGB (RealSense/USB), mounted to mimic the Bridge over-shoulder view | Policy is view-sensitive; §IV-A's envelope is "single fixed third-person camera" |
| Scene | Table or toy-kitchen surface + real utensils (forks/knives/spatulas/scissors as semantic clutter), spoon + towel | Matches the headline task; Bridge-domain props improve transfer odds |
| GPU | Any ≥12 GB near the robot (or stream frames to this A10G box) | π0 ~318 ms/forward; SAM3+LaMa at init only |

## Integration work (code, before any robot time)

1. **`scripts/real_eval.py` driver** (~1 day): camera capture → CGVD transform → π0 inference
   → Interbotix action execution, mirroring `batch_eval.py`'s loop. Action de-normalization for
   real WidowX exists in the Bridge/Interbotix ecosystem; the BYOVLA reference file shows the
   exact wx250s command pattern.
2. **FK robot mask** (~1–2 days): URDF + joint states → project link geometry through the
   calibrated camera (ArUco/checkerboard extrinsics) → binary mask, as `robot_mask_source="fk"`
   alongside the existing `gt`/`sam3` options. *Fallback that needs zero new code*: the
   already-implemented `sam3` per-frame mask (241 ms/frame, 20% dropout — acceptable for a
   qualitative demo, and honestly reportable either way).
3. **Deployment vocabulary**: the sim-validated generic 12-category list (fork, knife, spatula,
   scissors, ladle, whisk, plate, bowl, cup, mug, pan, pot) — no per-scene authoring, which is
   itself the point (§IV-D's vocabulary result on hardware).
4. **Pilot config check** (½ day): SAM 3 thresholds (0.30/0.20) sanity on real frames; LaMa
   fill quality eyeball on 5–10 real cluttered photos *before* involving the robot at all.
   (This photo-only step can happen this week with a phone camera — no robot needed — and
   already produces supplement-grade evidence for point 1.)

## Protocol (stage-gated)

- **Gate 0 — photos only (no robot)**: 10 real cluttered-tabletop photos → run the full
  perception pipeline offline → inspect masks/fills/vocabulary hits. If LaMa fills are unusable,
  we learn it for the cost of an afternoon.
- **Gate 1 — policy sanity**: 10 clutter-free `put spoon on towel` episodes, baseline only.
  Proceed to Gate 2 only if success ≳ 30%; otherwise the demo pivots to **perception-only**
  (still valuable and publishable as such — the paper's hardware gap is perception-level).
- **Gate 2 — paired qualitative comparison**: 10–15 layouts, each photographed, then run
  baseline and CGVD on the *same* layout (manual reset between arms). ~8–10 semantic
  distractors. Record: success, wrong-object grasps, collisions, raw video, distilled-view
  video, per-episode debug panels, init/compositing timing on the real GPU.
- **Extras worth 10 minutes each**: one vocabulary-miss episode (wrong vocabulary → near-empty
  mask → the paper's one-line miss detector fires); one episode with a mid-episode intrusion
  (demonstrates the stated static-scene hazard honestly — reviewers respect shown limitations).

## Paper integration

- New short paragraph ("Real-robot qualitative demonstration") in §IV or folded into §V,
  worded strictly qualitatively; clips into the supplement video.
- If perception-only: report it as validating SAM 3/LaMa/compositing on real scenes while
  leaving policy-level transfer future work — precisely narrows §V, never oversells.
- Anonymity: no faces, no lab logos or identifiable posters in frame (RAS rules); shoot tight
  on the table.

## Timeline estimate (part-time)

| Step | Time |
|---|---|
| Gate 0 photo study | 0.5 day (no robot; can start now) |
| Driver + calibration | 1–2 days |
| FK mask (or skip via sam3 fallback) | 0–2 days |
| Gates 1–2 capture | 1 day |
| Video edit + paper paragraph | 0.5 day |

## What I need from you

1. Confirm hardware: WidowX-250 access? Which camera? GPU near the robot, or stream to this box?
2. If no WidowX exists, decide now to target the **perception-only** demo (Gate 0 + real-photo
   study) — it needs only a phone camera and covers the most-attacked gap.
3. 10–20 phone photos of a real cluttered tabletop (utensils + a spoon + a towel) to start
   Gate 0 immediately — upload them here and I'll run the full pipeline on them today.
