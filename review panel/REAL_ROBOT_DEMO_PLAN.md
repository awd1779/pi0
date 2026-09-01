# Real-World Qualitative Demo — Step-by-Step Plan (UR3 + RTX 4070 laptop)

**Decision recorded**: rig = UR3 + fixed RGB camera + 4070 laptop (x86 → the existing Docker
image runs unmodified). Because `bridge_beta` is Bridge/WidowX-trained, policy-level transfer
to a UR3 is not credible zero-shot, so this is the **perception-level** demonstration: the CGVD
pipeline running live on real scenes with the real arm in frame. It closes §V's stated gaps
(SAM 3 recall on real clutter, LaMa fill fidelity on real textures, compositing stability,
FK-projected robot masking on hardware) and is written into the paper strictly qualitatively.

The Gate-0 tool already exists: `scripts/cgvd_standalone.py` (folder of images in → per-image
distilled output + masks out). No new code needed to start.

---

## Phase A — Desk work, no robot (½ day; can start today)

**A1. Laptop environment** (choose one):
   - *Easiest*: install Docker + nvidia-container-toolkit, `docker pull drodii/open-pi-zero:latest`
     (~40 GB), clone `github.com/awd1779/pi0`, run with the repo bind-mounted (same pattern as
     the sim evals). SAM 3 + LaMa need ~4–5 GB VRAM — comfortable on the 4070.
   - *Slim*: python 3.10 venv with `torch` (cu12x), `transformers` (git main, for SAM 3),
     `simple-lama-inpainting`, `opencv-python`, `pillow`; clone the repo; only `src/cgvd/` and
     `scripts/cgvd_standalone.py` are needed for the perception track.

**A2. Gate 0 — real-photo study**: take **15–20 photos** with the demo camera (or a phone):
   real spoon + towel + utensil clutter (forks/knives/spatulas/scissors) at 3 densities
   (~0/5/10 objects), tabletop framing similar to the intended camera view, fixed exposure.
   Run:
   ```
   python scripts/cgvd_standalone.py \
     --input_dir ~/photos --target "spoon" \
     --distractors "fork. knife. spatula. scissors. ladle. whisk. plate. bowl. cup. mug. pan. pot"
   ```
   Inspect the outputs: does SAM 3 find the utensils (vocabulary recall)? does it protect the
   spoon (safe-set)? are LaMa fills usable on the real table texture? Record timings.
   *(Alternative: upload the photos to this session and I run it on the A10G the same day.)*

**A3. Threshold sanity**: if real-image confidences differ from sim, adjust
   `--presence_threshold` / `--distractor_threshold` on the Gate-0 set only, note the values,
   and freeze them before any robot work (mirrors the paper's frozen-parameters discipline).

**Go/no-go**: if LaMa fills or vocabulary recall are unusable at Gate 0, stop and report that
finding honestly — it costs one afternoon and is itself informative.

## Phase B — Rig bring-up (½–1 day)

**B1. Camera**: fixed tripod/frame mount, third-person over-shoulder view of the workspace,
   ≥640×480 RGB. **Disable auto-exposure/auto-white-balance** (compositing assumes a stable
   background appearance; AE drift breaks the cache visually).
**B2. UR3 link**: `ur_rtde` (pip) from the laptop; verify joint-state streaming and slow
   scripted moveL trajectories over the table. Set conservative speed/force limits and the
   controller's workspace box; e-stop within reach; nobody in frame (RAS anonymity + the
   paper's own static-scene envelope).
**B3. Hand–eye calibration** (for the FK mask): print a ChArUco board; capture ~15 poses of a
   board fixed to the UR3 flange; solve with OpenCV `calibrateHandEye` → camera intrinsics +
   camera-to-base extrinsics saved to yaml.

## Phase C — FK robot mask (~1 day of desk coding; parallel with B)

**C1.** Implement `robot_mask_source="fk"` next to the existing `gt`/`sam3` options:
   PyBullet in DIRECT mode with the UR3 URDF (`ur_description`), set live joint states from
   RTDE, render a segmentation image from a virtual camera at the calibrated
   intrinsics/extrinsics → binary robot mask → dilate by r_e. (The UR3's first-class URDF/ROS
   support is why this is easy — easier than on a WidowX.)
**C2.** Validate: overlay the FK mask on live frames across the workspace; nudge extrinsics
   until visually tight; measure per-frame cost (expect ~5–15 ms — record it for the paper,
   next to the 241 ms/frame SAM 3 alternative already measured in sim).

## Phase D — Live distillation episodes (½ day)

**D1.** Demo driver (`scripts/real_demo.py`, I will write it; it is `cgvd_standalone.py`'s
   pipeline + the wrapper's caching/compositing loop + FK mask): at t=0 parse "put the spoon
   on the towel", segment safe set + the generic 12-category vocabulary, two-layer refinement,
   LaMa clean plate; per frame composite + FK-mask overwrite; record raw | distilled
   side-by-side at ~10 fps.
**D2.** **10–15 layouts** at ~8–10 real utensil distractors. For each: photograph the layout,
   then run one 20–30 s episode with the UR3 executing a scripted approach-and-sweep
   trajectory over the scene (the moving arm is what stresses the FK mask and compositing).
   Capture per episode: raw video, distilled video, 4-panel debug frames, init/per-frame
   timings on the 4070.
**D3.** Honesty extras (10 min each):
   - *Vocabulary miss*: run one layout with a non-utensil vocabulary → near-empty mask →
     log the mask-coverage number (the paper's one-line miss detector, demonstrated).
   - *Intrusion*: drop a new object mid-episode → show it blending toward the cache — the
     stated static-scene hazard, shown deliberately.
   - *Erasure audit*: hand-review the panels; report "spoon masked in k of N layouts."

## Phase E — Assembly (½ day)

**E1.** Select clips into the supplement reel (tight table framing; no faces, logos, or
   identifiable lab features — RAS double-anonymous rules apply to videos too).
**E2.** Paper: one qualitative paragraph (§IV or §V): pipeline validated on real scenes with
   the generic vocabulary; FK mask demonstrated at X ms/frame; timings on consumer hardware;
   policy-level transfer requires a Bridge-compatible embodiment and remains future work.
**E3.** Changelog entry + updated ASK-AUTHOR list.

## Split of work

| Who | What |
|---|---|
| You | Photos (A2), rig + calibration captures (B), robot episodes (D2–D3) |
| Me (this session) | Gate-0 processing if you upload photos; `real_demo.py`; the FK-mask module (testable here in PyBullet with the UR3 URDF before it ever touches your rig); video assembly; paper paragraph |

## Shopping/prep list
Printed ChArUco board (A4, rigid backing) · camera mount/tripod · real utensils (≥10:
forks, knives, spatulas, scissors + a distinctive spoon) · a towel · USB/Ethernet to the UR3.
