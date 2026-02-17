# CGVD: Concept-Gated Visual Distillation

## The Problem

Vision-Language-Action (VLA) models like Pi0 learn manipulation policies from demonstrations — "pick the spoon and place it on the towel." They work well in clean environments that match training conditions, but **fail catastrophically when visual distractors are present on the table**.

This is the **Precision-Reasoning Gap**: the policy sees a fork next to the target spoon and either grasps the wrong object (semantic confusion), avoids the region entirely (decision paralysis), or collides with distractors during transport. The failure is not uniform — **semantic distractors** (fork, spatula, knife — objects similar to the target) cause far more damage than random clutter (apple, mug, bread).

Empirically, on the `spoon_on_towel` task with a WidowX robot in SimplerEnv:
- **0 distractors**: ~80% success rate
- **8 semantic distractors**: drops to ~57% baseline
- **14 semantic distractors**: drops to ~48% baseline

Retraining the VLA on cluttered scenes is expensive and doesn't generalize. We need an **inference-time, training-free, model-agnostic** solution.

## The Solution: CGVD

CGVD is a `gym.Wrapper` that sits between the simulator camera and the VLA policy. Every frame, it intercepts the observation image, removes distractors via neural inpainting, and returns a clean image to the policy. The policy never sees the clutter.

### Core Pipeline

```
Instruction: "pick the spoon and place it on the towel"
                    │
                    ▼
         ┌─────────────────────┐
         │  Instruction Parser  │  → target = "spoon", anchor = "towel"
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   SAM3 Segmentation  │  Two independent queries:
         │                      │  safe-set:    "spoon. towel. robot arm"  (threshold 0.15)
         │                      │  distractors: "fork. knife. spatula..."  (threshold 0.30)
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Safe-Set Subtraction │  final_mask = distractor AND NOT dilate(safe)
         │  (Architectural       │  ← Guarantees target is NEVER inpainted,
         │   Safety Guarantee)   │     regardless of SAM3 errors
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   LaMa Inpainting    │  Neural inpainting fills distractor regions
         │                      │  with realistic table/background texture
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   Alpha Compositing   │  Feathered blend: distractors → clean plate,
         │                      │  target + robot → live frame pixels
         └─────────────────────┘
                    │
                    ▼
              Clean observation → VLA policy
```

### Key Design Decisions

**Asymmetric thresholds**: Safe-set uses low threshold (0.15) to maximize recall — we'd rather over-protect than miss the target. Distractors use high threshold (0.30) to minimize false positives — we'd rather miss a distractor than accidentally mask the target.

**Warmup accumulation**: During environment reset, run 1+ no-op frames with the robot stationary. SAM3 masks are unioned across frames via `np.maximum`. This handles SAM3's unreliable single-frame detection (scores can drop to 0.00 between frames).

**Robot-free rendering**: The robot arm is hidden during SAM3 segmentation queries so it doesn't occlude the target object. Robot visibility is handled separately in compositing.

**Automatic fallback**: If CGVD can't detect the target at all, it passes through the raw observation unchanged. This guarantees monotonicity: `SR_CGVD >= SR_baseline`.

## The Hard Problem: Confusable Objects

The pipeline above works when SAM3 can clearly distinguish spoon from fork. But SAM3 often **detects a spatula as "spoon"** (they look similar), which means the spatula enters the safe-set and becomes protected instead of inpainted. This is the central technical challenge.

### Multi-Layer Defense

We've built a layered system to handle this:

#### Layer 1: Cross-Validation (before top-1 selection)

For each target instance, compute:
```
genuineness = safe_confidence("spoon") - max_overlapping_distractor_confidence("spatula", "fork", ...)
```
If genuineness < threshold (−0.1) AND the instance isn't the best one, remove it. This catches cases where SAM3 gives 2+ "spoon" detections — one real, one actually a spatula.

**Critical constraint**: `must_keep_best = True` — always protect the highest-genuineness instance. Without this, the sole real spoon gets removed when genuineness is ambiguous (spoon vs spatula scores differ by <0.1).

#### Layer 2: Top-1 Selection with Distractor-Penalized Priority

After cross-validation, pick ONE target instance per frame. The priority formula:
```
priority = safe_score + genuineness_weight * max(genuineness, 0) - dist_ev_weight * dist_ev
```

Where `dist_ev` (distractor evidence) = `sum(IoU * distractor_score)` across ALL matching distractors, capped at 1.0. A spatula-as-spoon matches "ladle" AND "spatula" AND "fork" with high IoU → high evidence (~0.8+). A real spoon matches few distractors with low IoU → low evidence (~0.1).

#### Layer 3: Post-Warmup Spatial Cleanup

After warmup, the accumulated target mask may contain multiple disconnected components (from different frames picking different physical objects). Connected-component analysis scores each component:
```
score = pixel_count * (1.0 - dist_ev_weight * dist_ev) + genuineness_weight * max(genuineness, 0)
```
Keep only the best-scoring component.

#### Layer 4: Post-Layer-3 Rejection Gate

If the surviving target component STILL has overwhelming distractor evidence AND poor genuineness, reject it entirely:
```
if dist_ev > 0.6 AND genuineness <= 0.0:
    cached_target_mask = zeros  # safe-set = anchor only
```
Dual gate prevents false rejection — a real spoon in a cluttered scene may have moderate dist_ev but positive genuineness.

#### Layer 5: Pixel-Difference Enhancement

Catches distractors SAM3 missed entirely:
```
diff = |live_frame - clean_plate|.mean(axis=2)
mask any pixel where diff > 30
```
Re-enforcement protects target + robot. No SAM3 needed — purely geometric.

## Compositing Pipeline (the other hard problem)

Even with perfect masks, blending clean-plate and live-frame pixels without visible artifacts is difficult.

```
1. GaussianBlur(distractor_mask, sigma=3.0) → feathered alpha
2. Mechanism 2: clamp feathered to 1.0 in distractor regions gated by binary_target
3. Re-enforcement: binarized robot mask punches through (forces live pixels)
4. Blend: output = feathered * clean_plate + (1 - feathered) * live_frame
```

## What Has Been Tried (and What Failed)

### Compositing Approaches

| Approach | Result | Why |
|----------|--------|-----|
| **Poisson blending** (cv2.seamlessClone) | Failed — fog/color-shift artifacts | Solves global Poisson equation; shifts colors across entire masked region. Designed for pasting small objects, not replacing large regions |
| **Large GaussianBlur sigma** (sigma=5.0) | Visible halos around distractors | ~15px blur spread leaked distractor edges. Reduced to sigma=3.0 (~9px spread) |
| **Soft SAM3 masks in compositing** | Distractor leakage at boundaries | SAM3 returns soft values (0.6–0.7) at edges; `1.0 - 0.6 = 0.4` lets 40% of distractor through. Fixed by binarizing all masks with `> 0.5` threshold |
| **Including robot in cached_mask** | Flickering artifacts | Robot-shaped holes shift every frame as arm moves → GaussianBlur produces different feathered values → flicker. Fixed by decoupling robot from cached_mask; robot handled via binarized re-enforcement |
| **Eroded robot mask in distractor zones** | Table-texture halo around robot | ~2px gap where Mechanism 2 forces feathered=1.0, showing clean plate through the gap. Fixed by using raw binarized robot mask (>0.5 threshold is sufficient) |
| **Undilated safe mask in compositing** | GaussianBlur bleeds table texture at target boundary | SAM3 under-segments by 1–3px → blur leaks clean-plate texture into the gap. Fixed by dilating binary_target by `safe_dilation=5` in compositing |

### Safe-Set / Target Identification

| Approach | Result | Why |
|----------|--------|-----|
| **Cross-validation AFTER accumulation** (subtracting fp_mask from cached_safe_mask) | Destructive — one bad frame wipes all frames | Cached_safe_mask is a union; subtracting removes pixels accumulated across ALL previous good frames. Fixed by subtracting from per-frame raw mask BEFORE `np.maximum` accumulation |
| **Removing top-1 entirely** | Robot ghosting artifacts | Mechanism unclear, but top-1 must stay |
| **Single-max fraction-overlap** for distractor evidence | Saturated to 1.0, no discrimination | A single IoU calculation can't distinguish real vs fake. Changed to summed `IoU * score` across ALL matching distractors |
| **Single-max IoU** for distractor evidence | Missed multi-label signal | A spatula matches "spatula" (IoU 0.9) but also "ladle" (0.7) and "fork" (0.5). Summing captures total evidence |
| **must_keep_best = False** in cross-validation | Sole real spoon gets removed | When genuineness is a coin flip (spoon/spatula differ by <0.1), the only target instance gets filtered out |
| **5-frame warmup** | 5x redundant computation, zero benefit | Robot-free image is static during no-op steps and SAM3 is deterministic → all frames produce identical results. Reduced to 1 frame |
| **dist_ev computed but not used in scoring** | Spatulas still entering safe-set | The signal was logged but never influenced priority or component selection. Just wired in (this session) |

### Inpainting

| Approach | Result | Why |
|----------|--------|-----|
| **Mean-color fill** (ablation) | Visible rectangular patches | No texture continuity; policy can "see" something was removed |
| **LaMa with small dilation** | Inpainting artifacts at mask boundaries | LaMa needs margin around the mask to blend properly. `lama_dilation=11` provides sufficient context |
| **Refreshing clean plate frequently** | Robot arm baked into clean plate | Clean plate must be generated from robot-free warmup frame only |

## Current Parameters (Defaults)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `safeset_warmup_frames` | 1 | Frames to accumulate safe-set (1 is sufficient for static scenes) |
| `presence_threshold` | 0.15 | Safe-set detection threshold (permissive) |
| `distractor_presence_threshold` | 0.30 | Distractor detection threshold (strict) |
| `robot_presence_threshold` | 0.05 | Robot detection threshold (very permissive) |
| `blend_sigma` | 3.0 | GaussianBlur sigma for feathered compositing |
| `lama_dilation` | 11 | Dilation for LaMa inpainting mask |
| `safe_dilation` | 5 | Dilation for safe-set protection in Step 4 and compositing |
| `genuineness_margin` | −0.1 | Threshold for cross-validation rejection |
| `genuineness_weight` | 0.2 | Weight of genuineness in priority/scoring formulas |
| `dist_ev_weight` | 0.5 | Penalty weight for distractor evidence in scoring |
| `dist_ev_reject` | 0.6 | Threshold for post-Layer-3 rejection gate |
| `min_component_pixels` | 50 | Minimum pixels for a target component to survive cleanup |
| `overlap_penalty_cap` | 0.7 | Cap on overlap penalty in detection filtering |

## Results

**Spoon on Towel (semantic distractors):**
- 8 distractors: 57% baseline → 69% CGVD (**+12pp**)
- 14 distractors: 48% baseline → 66% CGVD (**+18pp**)

**Collision reduction:** ~60% fewer gripper-distractor contacts with CGVD.

**Comparison to prior work:**
- **BYOVLA**: Requires K VLA forward passes per region + GPT-4o API calls. Probabilistic protection.
- **ARRO**: Destroys scene context, no fallback when tracking fails.
- **OBEYED-VLA**: Requires expensive retraining with attention adapters.
- **CGVD**: Zero VLA passes, architectural safety guarantee, preserves scene context, automatic fallback.

## File Map

| File | Purpose |
|------|---------|
| `src/cgvd/cgvd_wrapper.py` | Main wrapper — masks, compositing, all layers |
| `src/cgvd/sam3_segmenter.py` | SAM3 segmentation (client-server architecture) |
| `src/cgvd/lama_inpainter.py` | LaMa neural inpainting |
| `src/cgvd/instruction_parser.py` | Extract target/anchor from language instruction |
| `src/cgvd/distractor_wrapper.py` | Spawn distractors in SimplerEnv |
| `src/cgvd/collision_tracker.py` | Track gripper-distractor contacts |
| `src/cgvd/grasp_analyzer.py` | Classify failure modes |
| `scripts/clutter_eval/batch_eval.py` | Full sweep evaluation |
| `doc/cgvd_algorithm_reference.md` | Detailed algorithm pseudocode |
| `cgvd_debug/` | Debug images (6-column layout) and logs |
