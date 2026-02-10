# CGVD Fix Plan: Closing the Gap to Clean Baseline

Related: [cgvd_carrot_investigation.md](cgvd_carrot_investigation.md)

## Goal

CGVD + distractors should match the **clean baseline (57%)**, not the distractor baseline.
Current gap: -9% (n=2), -1.5% (n=4). Target: <2% gap across all distractor counts.

## Implementation Order

Fixes are ordered by expected impact. Each is independently testable.

---

## Fix 1: Soft Compositing with Feathered Mask

**Why:** The hard `np.where` at the mask boundary creates a pixel-level seam between the stale inpainted background and the live frame. This is the primary cause of the increased drop failures (20→29).

**What:** Replace binary compositing with Gaussian-feathered alpha blending.

**File:** `src/cgvd/cgvd_wrapper.py`

**Current code (lines 501-502, repeated at 521-522):**
```python
distractor_mask_3d = self.cached_mask[..., None] > 0.5
distilled = np.where(distractor_mask_3d, self.cached_inpainted_image, image)
```

**New code:**
```python
feathered = cv2.GaussianBlur(self.cached_mask, (0, 0), sigmaX=self.blend_sigma, sigmaY=self.blend_sigma)
feathered_3d = feathered[..., None]
distilled = (feathered_3d * self.cached_inpainted_image.astype(np.float32) +
             (1.0 - feathered_3d) * image.astype(np.float32)).astype(np.uint8)
```

**Changes to `__init__` (line ~50):** Add parameter `blend_sigma: float = 5.0`.

**Locations to update:**
- Lines 501-502 (first frame compositing)
- Lines 521-522 (subsequent frames compositing)

Extract a helper method `_composite(image, inpainted, mask)` to avoid duplicating the blending logic.

---

## Fix 2: Move Mask Dilation Before Safe-Set Subtraction

**Why:** LaMa dilates the mask by 11px internally (`lama_inpainter.py:87-91`) after safe-set subtraction has already been applied. This means the inpainted background has hallucinated content in the 11px border around distractors — including where the carrot might be. Moving dilation before subtraction ensures the safe-set re-excludes any dilated pixels that overlap task objects.

**What:** Dilate `distractor_mask` in the wrapper before the safe-set subtraction step. Pass `dilate_mask=0` to LaMa.

**File:** `src/cgvd/cgvd_wrapper.py`

**Current code (lines 469-474):**
```python
# Step 4: SUBTRACT safe set from distractors
self.cached_mask = np.logical_and(
    distractor_mask > 0.5, safe_mask < 0.5
).astype(np.float32)
```

**New code (insert before subtraction):**
```python
# Step 3.5: Dilate distractor mask BEFORE safe-set subtraction
# This ensures dilation is gated by the safe-set (carrot never bleeds in)
if self.lama_dilation > 0:
    dilation_kernel = np.ones((self.lama_dilation, self.lama_dilation), np.uint8)
    distractor_mask = cv2.dilate(
        (distractor_mask > 0.5).astype(np.uint8), dilation_kernel, iterations=1
    ).astype(np.float32)

# Step 4: SUBTRACT safe set from (dilated) distractors
self.cached_mask = np.logical_and(
    distractor_mask > 0.5, safe_mask < 0.5
).astype(np.float32)
```

**File:** `src/cgvd/lama_inpainter.py`

Pass `dilate_mask=0` from the wrapper to LaMa at all call sites:
- `cgvd_wrapper.py:498` → `self.inpainter.inpaint(image, cache_mask, dilate_mask=0)`
- `cgvd_wrapper.py:514` → `self.inpainter.inpaint(image, cache_mask, dilate_mask=0)`

**Changes to `__init__` (line ~50):** Add parameter `lama_dilation: int = 11`.

---

## Fix 3: Periodic Cache Refresh

**Why:** The inpainted background is captured at frame ~6 and reused for ~294 frames. Lighting/shadows shift as the robot moves, worsening the seam mismatch over time.

**What:** Re-inpaint the background every N frames.

**File:** `src/cgvd/cgvd_wrapper.py`

**Insert after line 516** (after the warmup re-inpaint block):
```python
# Periodic refresh to reduce background staleness
elif (self.cache_refresh_interval > 0 and
      self.frame_count % self.cache_refresh_interval == 0):
    if self.include_robot and self.last_robot_mask is not None:
        cache_mask = np.maximum(self.cached_mask, self.last_robot_mask)
    else:
        cache_mask = self.cached_mask
    self.cached_inpainted_image = self.inpainter.inpaint(
        image, cache_mask, dilate_mask=0
    )
    if self.verbose:
        print(f"[CGVD] Cache refresh at frame {self.frame_count}")
```

**Changes to `__init__` (line ~50):** Add parameter `cache_refresh_interval: int = 50`.

**Performance note:** LaMa takes ~50-100ms per call. With refresh every 50 frames over ~300 frames, that's ~5 extra inpaintings per episode (~0.5s total). Negligible vs the 26s SAM3 overhead.

---

## Fix 4: Expose Ablation Parameters via CLI

**Why:** Currently dilation, blend sigma, cache refresh, and warmup frames are hardcoded. Exposing them via CLI enables ablation testing without code edits.

### File: `scripts/try_checkpoint_in_simpler.py`

**Add after line 420** (after existing ablation flags):
```python
parser.add_argument(
    "--cgvd_blend_sigma", type=float, default=5.0,
    help="Gaussian sigma for soft compositing at mask edges (0=hard, default 5.0)",
)
parser.add_argument(
    "--cgvd_lama_dilation", type=int, default=11,
    help="Mask dilation in pixels before safe-set subtraction (default 11, 0=disable)",
)
parser.add_argument(
    "--cgvd_cache_refresh", type=int, default=50,
    help="Re-inpaint cached background every N frames (0=never, default 50)",
)
parser.add_argument(
    "--cgvd_safeset_warmup", type=int, default=5,
    help="Frames to accumulate safe-set detections during warmup (default 5)",
)
```

### File: `scripts/try_checkpoint_in_simpler.py` `wrap_env_with_cgvd()` (lines 39-55)

**Add to CGVDWrapper constructor call:**
```python
blend_sigma=getattr(args, 'cgvd_blend_sigma', 5.0),
lama_dilation=getattr(args, 'cgvd_lama_dilation', 11),
cache_refresh_interval=getattr(args, 'cgvd_cache_refresh', 50),
safeset_warmup_frames=getattr(args, 'cgvd_safeset_warmup', 5),
```

### File: `scripts/clutter_eval/run_paired_eval.sh`

**Add variables (around line 50):**
```bash
CGVD_BLEND_SIGMA=5.0
CGVD_LAMA_DILATION=11
CGVD_CACHE_REFRESH=50
```

**Add to CGVD_ARGS block (around line 427):**
```bash
CGVD_ARGS="$CGVD_ARGS --cgvd_blend_sigma $CGVD_BLEND_SIGMA"
CGVD_ARGS="$CGVD_ARGS --cgvd_lama_dilation $CGVD_LAMA_DILATION"
CGVD_ARGS="$CGVD_ARGS --cgvd_cache_refresh $CGVD_CACHE_REFRESH"
```

---

## Fix 5: Enhanced Debug Visualization

**Why:** Current debug images show 5 columns (Original | Distractors | Safe Set | Final | Distilled) but don't show compositing seams or the cached background. Adding these makes future diagnosis much faster.

**File:** `src/cgvd/cgvd_wrapper.py` method `_save_debug_images()` (lines 574-738)

### 5a: Seam heatmap column

After computing the 5-column comparison at line 600, add a 6th column showing the absolute pixel difference between distilled output and live frame:

```python
# Seam visualization: where does distilled differ from live frame?
seam_diff = np.abs(distilled.astype(np.float32) - original.astype(np.float32)).mean(axis=2)
seam_vis = (np.clip(seam_diff * 3, 0, 255)).astype(np.uint8)
seam_vis_rgb = cv2.applyColorMap(seam_vis, cv2.COLORMAP_JET)
seam_vis_rgb = cv2.cvtColor(seam_vis_rgb, cv2.COLOR_BGR2RGB)
comparison = np.hstack([comparison, seam_vis_rgb])
cv2.putText(comparison, "Seam Diff", (5 * w + 10, 30), font, 0.5, (255, 255, 255), 1)
```

### 5b: Target missing warning

After line 667, add a warning when the target is not detected in the safe-set:

```python
# Warn if target object not detected in safe-set
if self.current_target:
    target_detected = any(
        self.current_target in k and v >= self.presence_threshold
        for k, v in self.safe_scores.items()
    )
    if not target_detected:
        cv2.putText(
            comparison, "WARNING: TARGET NOT IN SAFE SET",
            (10, h - 40), font, 0.6, (0, 0, 255), 2,
        )
```

### 5c: Mask boundary contours on Distilled column

Draw the compositing boundary on the Distilled column so you can see exactly where seams are:

```python
# Draw compositing boundary on Distilled column
contours, _ = cv2.findContours(
    (mask > 0.5).astype(np.uint8),
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
)
cv2.drawContours(comparison, contours, -1, (0, 0, 255), 1, offset=(4 * w, 0))
```

---

## Verification Plan

### Quick smoke test (5 episodes)
```bash
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_carrot_on_plate \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --use_bf16 --num_episodes 5 \
    --distractors rc_corn_1 rc_cucumber_1 \
    --use_cgvd --cgvd_save_debug --cgvd_verbose \
    --cgvd_blend_sigma 5.0 --cgvd_lama_dilation 11 --cgvd_cache_refresh 50 \
    --output_dir logs/cgvd_fix_smoke_test
```

Inspect debug images in `logs/cgvd_fix_smoke_test/cgvd_debug/` — verify:
- Seam heatmap column shows low-intensity (minimal pixel difference at boundaries)
- No "TARGET NOT IN SAFE SET" warnings
- Mask contours on Distilled column don't overlap the carrot

### Full paired eval (20 episodes x 3 runs)
```bash
./scripts/clutter_eval/run_paired_eval.sh \
    --task widowx_carrot_on_plate -e 20 -r 3 -n 2
```

**Success criteria:**
- CGVD + 2 distractors ≥ 55% (within ~2% of 57% clean baseline)
- "Dropped" count returns to baseline levels (~20, down from 29)
- No regression on n=4 performance (should stay ≥ 55%)

---

## Fix 6: Re-enforce Safe-Set After Feathering (B.6)

**Why:** Fix 1 (soft compositing) introduced a new bug: `GaussianBlur` feathers ALL mask edges, including the inner edges where the safe-set carved holes for the carrot/plate. This bleeds the cached background (which contains the carrot at its original position) into the live frame after the robot picks up the carrot, creating a ghost/afterimage.

**What:** After applying GaussianBlur, multiply the feathered mask by `(1 - cached_safe_mask)` to zero out any bleed into safe-set regions.

**File:** `src/cgvd/cgvd_wrapper.py` method `_composite()`

**New code (after GaussianBlur, before constructing feathered_3d):**
```python
if self.cached_safe_mask is not None:
    feathered = feathered * (1.0 - self.cached_safe_mask)
```

**Effect:**
- Outer edges (distractor → background): still feathered smoothly
- Inner edges (safe-set holes for carrot/plate/robot): hard boundary, 100% live frame
- After carrot pickup: no ghost, carrot region always shows current live frame

---

### Ablation matrix (optional, for paper)
```bash
# Hard compositing (before fix)
--cgvd_blend_sigma 0 --cgvd_lama_dilation 11 --cgvd_cache_refresh 0

# Soft compositing only
--cgvd_blend_sigma 5 --cgvd_lama_dilation 11 --cgvd_cache_refresh 0

# Soft + safe dilation
--cgvd_blend_sigma 5 --cgvd_lama_dilation 11 --cgvd_cache_refresh 0

# Full fix
--cgvd_blend_sigma 5 --cgvd_lama_dilation 11 --cgvd_cache_refresh 50
```
