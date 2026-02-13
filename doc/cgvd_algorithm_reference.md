# CGVD Algorithm Reference

> Concept-Gated Visual Distillation — a model-agnostic perception preprocessing module
> that removes visual distractors from robot manipulation scenes before observations
> reach Vision-Language-Action (VLA) policies.

## 1. System Overview

CGVD is implemented as a `gym.Wrapper` that intercepts camera observations, removes
clutter objects via segmentation + inpainting, and returns a "clean" image to the
policy. The key guarantee: **task-relevant objects (target, anchor, robot) are never
altered**, even if the segmenter confuses them with distractors.

### Files

| File | Lines | Role |
|------|-------|------|
| `src/cgvd/cgvd_wrapper.py` | 1235 | Core pipeline (gym wrapper) |
| `src/cgvd/sam3_segmenter.py` | 577 | SAM3 text-prompted instance segmentation |
| `src/cgvd/lama_inpainter.py` | 98 | LaMa inpainting backend |
| `src/cgvd/instruction_parser.py` | 142 | NLP: instruction → (target, anchor) |
| `src/cgvd/distractor_wrapper.py` | 1122 | Physical distractor placement in simulation |
| `src/cgvd/collision_tracker.py` | 194 | Evaluation: robot-distractor collision detection |
| `src/cgvd/grasp_analyzer.py` | 237 | Evaluation: grasp failure categorization |

### Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `presence_threshold` | 0.15 | SAM3 confidence for safe-set (target/anchor) |
| `distractor_presence_threshold` | 0.3 | SAM3 confidence for distractors |
| `robot_presence_threshold` | 0.05 | SAM3 confidence for robot arm |
| `safeset_warmup_frames` | 5 | Frames to accumulate masks before compositing |
| `deferred_detection_frames` | 10 | Extra frames to find target if missed in warmup |
| `blend_sigma` | 3.0 | Gaussian blur sigma for feathered compositing |
| `lama_dilation` | 11 | Distractor mask dilation before safe-set subtraction |
| `safe_dilation` | 5 | Protective buffer around target/anchor edges |
| `cache_refresh_interval` | 0 | Re-inpaint interval (0 = disabled) |

Derived: `_reinforce_size = safe_dilation + 2 * ceil(blend_sigma)` = **11px** with defaults.

---

## 2. Pipeline Overview

```
reset()                          step()
  │                                │
  ├─ warmup (frames 0..4)         │
  │   └─ _apply_cgvd() × 5       │
  │      (accumulate masks,       │
  │       skip compositing)       │
  │                               │
  ├─ first post-warmup frame      ├─ _apply_cgvd()
  │   └─ _apply_cgvd()           │    (composite only,
  │      (inpaint + composite)    │     reuse cached inpainted)
  │                               │
  └─ return obs to VLA            └─ return obs to VLA
```

### Per-Frame Pipeline (`_apply_cgvd`)

```
Step 1: Parse instruction → (target, anchor)
Step 2: Segment distractors (SAM3, accumulate during warmup, freeze after)
Step 3a: Segment safe-set (SAM3, accumulate during warmup, freeze after)
Step 3b: Segment robot (SAM3, every frame)
Step 3.5: Dilate distractor mask by lama_dilation
Step 4: Gate: cached_mask = dilated_distractor AND NOT dilated_safe
Step 5: Inpaint (LaMa, first post-warmup frame only)
Step 6: Composite (feathered alpha blend, every frame)
```

---

## 3. Distractor Tracking

### Segmentation
- User provides distractor names (e.g., `["fork", "spatula", "knife"]`)
- Joined into SAM3 query: `"fork. spatula. knife"` (line 521)
- Queried at `distractor_presence_threshold=0.3` — higher threshold reduces false
  positives (e.g., carrot detected as banana)
- SAM3 queries each concept independently, returns per-instance masks + scores

### Accumulation (warmup)
- Frames 0 through `safeset_warmup_frames - 1`:
  ```python
  cached_distractor_mask = np.maximum(cached_distractor_mask, raw_distractor_mask)
  ```
- Union ensures objects missed on one frame are caught on another
- After warmup: mask is **frozen** (distractors are stationary tabletop objects)

### Post-Warmup Behavior
- When `cache_distractor_once=True` (default): mask stays frozen permanently
- When `cache_distractor_once=False`: mask is **replaced** (not accumulated) every
  `update_freq` frames — loses warmup accumulation

### Dilation (Step 3.5)
- Before safe-set subtraction, dilate by `lama_dilation=11` px (line 696-700)
- Expands coverage to catch shadows/edges SAM3 misses
- Dilation happens **before** gating by safe-set, so it cannot bleed into protected regions
- Applied to a local `distractor_mask` variable; `cached_distractor_mask` stays raw/undilated

### Dead Code
- `_suppress_overlapping_distractors()` (line 257-280) is defined but **never called**.
  Comment at line 674-678 explains it was removed because the Step 4 gating formula
  already handles overlap, and pre-suppression was causing distractors to show through.
- `distractor_iou_threshold` parameter (default 0.15) is stored but never used.

---

## 4. Target + Anchor (Safe-Set) Tracking

### Instruction Parsing (`instruction_parser.py`)
- Pattern matching first: `"spoon.*towel"` → `("spoon", "towel")` (line 66-69)
- Fallback: heuristic noun extraction — strip action verbs, first remaining noun =
  target, noun after preposition = anchor (lines 76-121)
- Builds SAM3 prompt: `"spoon. towel"` (robot excluded, tracked separately)

### Segmentation
- SAM3 query at `presence_threshold=0.15` — low threshold to prefer over-protection
  over missing the target (line 582-583)
- Robot is excluded from this query (`include_robot=False` at line 580)

### Top-1 Filtering (`_filter_overlapping_detections`, line 208-255)
- SAM3 may return multiple instances (e.g., `spoon_0`=real spoon, `spoon_1`=spatula)
- Groups by base concept name (strips `_0`, `_1` suffixes)
- Keeps only the **highest-scoring** instance per concept
- Applied **before** accumulation — filtered-out instances never enter `cached_safe_mask`
- Called every frame during warmup (and deferred detection)

### Cross-Validation (`_cross_validate_safeset`, line 282-364)
- For each target instance, computes:
  ```
  genuineness = safe_score - max_overlapping_distractor_score
  ```
  where overlap is measured by IoU > 0.3 between the safe instance and any distractor
- Instances with negative genuineness are flagged as false positives
- **Always keeps the most genuine instance** (highest genuineness score)
- **Used for logging only** — the returned `fp_mask` is computed but never subtracted
  from `cached_safe_mask` (line 633-636). Top-1 filtering handles false positive
  rejection in practice.

### Accumulation (warmup)
- Frames 0 through `safeset_warmup_frames - 1`:
  ```python
  cached_safe_mask = np.maximum(cached_safe_mask, raw_target_mask)  # union
  _safe_mask_votes += (raw_target_mask > 0.5).astype(float32)       # vote count
  ```
- `np.maximum` union: safe mask only grows, never shrinks
- Vote counting (`_safe_mask_votes`) is **diagnostic only** — logged on last warmup
  frame (line 641-646) but never used for filtering or thresholding
- After warmup: mask is **frozen**

### Deferred Detection (line 570-574)
- If target was not detected during warmup (e.g., occluded by robot gripper):
  - Safe-set segmentation continues for `deferred_detection_frames=10` more frames
  - Condition: `not _target_detected_in_warmup AND frame_count < warmup + deferred`
- Safe because: if SAM3 never saw the target, it also never labeled it as a distractor
  → it shows through naturally at `mask=0` pixels
- When target is finally detected, it's added to `cached_safe_mask` for explicit
  protection going forward

### Dilation (Step 4 gating)
- `cached_safe_mask` (target+anchor only, no robot) dilated by `safe_dilation=5` px
  (line 686-692)
- Creates ~2.5px protective buffer around target edges
- Counters the encroachment from `lama_dilation` on the distractor side
- Used in the gating formula: `cached_mask = distractor AND NOT dilated_safe`

---

## 5. Robot Arm Tracking

### Segmentation (every frame)
- SAM3 query: `"robot arm. robot gripper"` at `robot_presence_threshold=0.05` (line 655-658)
- Runs **every frame** — not gated by warmup or caching conditions
- Uses `last_distilled_image` as input when available (line 656):
  ```python
  robot_image = self.last_distilled_image if self.last_distilled_image is not None else image
  ```
  The already-cleaned image gives SAM3 a cleaner view of the robot.
  During warmup, `last_distilled_image` is `None`, so the raw frame is used.

### Warmup Accumulation
- During warmup: `cached_robot_mask = np.maximum(cached_robot_mask, robot_mask)` (line 662-666)
- This accumulated mask is used for clean-plate generation (`_build_inpaint_mask`)
  so the robot's starting position is also inpainted away

### Decoupling from `cached_mask`
- The gating formula (Step 4) uses `cached_safe_mask` (target+anchor **only**):
  ```python
  safe_mask_for_gating = dilate(cached_safe_mask > 0.5)     # no robot
  cached_mask = distractor AND NOT safe_mask_for_gating      # stable across frames
  ```
- Robot is excluded so `cached_mask` doesn't have robot-shaped holes that shift every
  frame (which would cause GaussianBlur to produce different feathered values → flicker)
- Robot visibility is handled entirely in `_composite()` via re-enforcement

### Clean-Plate Inpainting (`_build_inpaint_mask`, line 897-920)
- Inpaint mask = `cached_mask` + robot (dilated by `_reinforce_size`)
- Prefers `cached_robot_mask` (accumulated warmup union) over `last_robot_mask` (single frame)
- Robot dilation covers the GaussianBlur spread (~2σ) so no stale arm-color pixels
  remain after the arm moves away

---

## 6. Compositing (`_composite`, line 797-895)

The compositing algorithm blends the cached inpainted background with the live camera
frame using a multi-stage feathered alpha approach.

### Stage 1: Feathered Alpha
```python
feathered = cv2.GaussianBlur(mask, (0, 0), sigmaX=blend_sigma, sigmaY=blend_sigma)
```
- `mask` = `cached_mask` (binary, stable across frames)
- GaussianBlur creates smooth 0→1 transitions at mask boundaries
- With `blend_sigma=3.0`, blur spread is ~9px (3σ on each side)

### Stage 2: Binarize Inputs
```python
safe = (current_safe_mask > 0.5).astype(float32)           # target + anchor + robot
binary_target = (cached_safe_mask > 0.5).astype(float32)   # target + anchor only
binary_target = cv2.dilate(binary_target, _reinforce_size)  # dilate for buffer
```

### Stage 3: Mechanism 2 — Distractor Clamping
```python
binary_distractor = (cached_distractor_mask > 0.5).astype(float32)   # raw, undilated
non_safe_distractor = binary_distractor * (1.0 - binary_target)       # exclude target
feathered = np.maximum(feathered, non_safe_distractor)                # clamp to 1.0
```
- Uses `cached_distractor_mask` (raw/undilated), NOT the dilated local variable
- Binarization prevents soft SAM3 edge values (0.6-0.7) from only partially clamping
- Gated by `binary_target` so target pixels are never clamped to show inpainted bg
- Effect: distractor pixels always show inpainted background (feathered=1.0)

### Stage 4: Re-enforcement — Safe-Set Protection
```python
# Isolate robot contribution
robot_contrib = np.clip(safe - binary_target, 0.0, 1.0)
robot_binary = (robot_contrib > 0.5).astype(float32)

# Dilate robot (halo elimination outside distractors)
robot_binary_dilated = cv2.dilate(robot_binary, _reinforce_size)

# Hybrid mask: dilated outside distractors, raw inside distractors
if cached_distractor_mask is not None:
    distractor_zone = (cached_distractor_mask > 0.5).astype(float32)
    non_distractor = 1.0 - distractor_zone
    robot_binary_dilated = (
        robot_binary_dilated * non_distractor    # dilated outside
        + robot_binary * distractor_zone          # raw binarized inside
    )

# Combine target protection + robot protection
reinforce_mask = np.maximum(binary_target, robot_binary_dilated)
feathered = feathered * (1.0 - reinforce_mask)   # force to 0.0 = show live frame
```
- **Outside distractor zones**: robot dilated by `_reinforce_size` (11px) to eliminate
  halo from GaussianBlur spread
- **Inside distractor zones**: raw binarized robot (no dilation) to prevent SAM3 false
  positive pixels from punching holes in the distractor coverage
- Effect: target + robot pixels always show live frame (feathered=0.0)

### Stage 5: Final Blend
```python
feathered_3d = feathered[..., None]
result = feathered_3d * inpainted + (1.0 - feathered_3d) * live_frame
```
- `feathered=1.0` → pure inpainted background (distractors removed)
- `feathered=0.0` → pure live frame (target, robot, non-distractor areas)
- `0 < feathered < 1` → smooth transition at boundaries

### Hard Compositing Fallback
When `blend_sigma=0`, skips all feathering and uses hard binary selection:
```python
mask_3d = mask[..., None] > 0.5
return np.where(mask_3d, inpainted, image)
```

---

## 7. Interaction Diagram

```
                    ┌──────────────────┐
                    │   Distractors    │  SAM3 @ 0.3 threshold
                    │  (frozen after   │  np.maximum accumulation
                    │   warmup)        │  dilated by lama_dilation=11
                    └────────┬─────────┘
                             │
                      AND NOT│
                             │
                    ┌────────┴─────────┐
                    │    Safe-Set      │  SAM3 @ 0.15 threshold
                    │  (target+anchor  │  np.maximum accumulation
                    │   only, no robot │  dilated by safe_dilation=5
                    │   frozen after   │  top-1 filtering per concept
                    │   warmup)        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   cached_mask    │  Binary, stable across frames
                    │  (no robot)      │  No robot-shaped holes → no flicker
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
      GaussianBlur     Mechanism 2      Re-enforcement
      (sigma=3.0)      (clamp=1.0       (force=0.0 at
                        at distractor    target + robot)
                        pixels outside
                        target)
            │                │                │
            └────────────────┼────────────────┘
                             ▼
                    ┌──────────────────┐
                    │  feathered blend │
                    │  f*inpainted +   │
                    │  (1-f)*live      │
                    └──────────────────┘

    Robot (SAM3 @ 0.05 threshold, every frame):
    ├── Excluded from cached_mask (prevents flicker)
    ├── Included in _build_inpaint_mask (clean plate)
    └── Re-enforced in _composite (punches through distractors)
```

---

## 8. Warmup Sequence Detail

```
Frame 0:  Segment distractors → init cached_distractor_mask
          Segment safe-set    → init cached_safe_mask + votes
          Segment robot       → init cached_robot_mask
          Skip compositing (VLA never sees this frame)

Frame 1-4: Same as frame 0 but accumulate via np.maximum
           Cross-validate (logging only)
           On frame 4: log vote statistics

Frame 5:  (First post-warmup frame)
          Distractor mask: frozen (not recomputed)
          Safe-set mask: frozen (not recomputed, unless deferred)
          Robot mask: fresh segmentation
          Step 3.5: Dilate distractor mask
          Step 4: Gate = distractor AND NOT safe
          Step 5: Build inpaint mask (cached_mask + robot)
                  Run LaMa → cached_inpainted_image
                  Composite → first VLA observation

Frame 6+: Robot re-segmented each frame
          Composite using cached_inpainted_image + fresh robot mask
          Optional: cache refresh every cache_refresh_interval frames
```

---

## 9. SAM3 Segmenter Architecture (`sam3_segmenter.py`)

### Three Backends

| Class | Use Case | Communication |
|-------|----------|---------------|
| `SAM3Segmenter` | Default, direct HuggingFace model | In-process GPU |
| `SAM3ClientSegmenter` | When transformers version conflicts (e.g., GR00T) | File-based IPC via `/tmp/sam3_server` |
| `MockSAM3Segmenter` | Testing without GPU/model | Center-weighted Gaussian |

### Segmentation Flow (`SAM3Segmenter.segment`)
1. Parse dot-separated concepts: `"fork. spatula"` → `["fork", "spatula"]`
2. Pre-compute vision embeddings once (shared across concepts):
   ```python
   vision_embeds = model.get_vision_features(pixel_values)
   ```
3. Query each concept independently with shared vision embeddings
4. Per-concept: `post_process_instance_segmentation` → per-instance masks + scores
5. Combine all instances via `np.maximum` → combined mask
6. Binarize at 0.5 threshold
7. Store per-concept scores in `last_scores` and per-instance masks in `last_individual_masks`

### Singleton Pattern
- `get_sam3_segmenter()` returns a shared instance (~2.5GB model, loaded once)
- `get_lama_inpainter()` returns a shared LaMa instance (~2GB)
- Prevents redundant model loading across multiple `CGVDWrapper` instances

---

## 10. LaMa Inpainting (`lama_inpainter.py`)

- Wraps `simple_lama_inpainting.SimpleLama`
- Uses Fast Fourier Convolutions (FFC) for global receptive field
- Generates realistic textures (not just local patch copying)
- Called with `dilate_mask=0` from CGVD (dilation handled in Step 3.5 instead)
- Runs once per episode (first post-warmup frame), result cached as `cached_inpainted_image`
- Optional periodic refresh via `cache_refresh_interval` (disabled by default)

---

## 11. Ablation Flags

| Flag | Effect |
|------|--------|
| `disable_safeset=True` | Skip safe-set protection entirely; mask = raw dilated distractors |
| `disable_inpaint=True` | Replace LaMa with mean-color fill (`_apply_mean_fill`) |

---

## 12. Timing Characteristics

| Component | Frequency | Approximate Time |
|-----------|-----------|-----------------|
| SAM3 segmentation | Every frame (robot) + warmup (distractors, safe-set) | 50-200ms |
| LaMa inpainting | Once per episode (+ optional refresh) | 100-500ms |
| Compositing | Every frame | <5ms |
| Total per-frame (post-warmup) | Every frame | ~55-205ms |

Timing stats available via `get_timing_stats()` method.
