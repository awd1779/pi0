# CGVD Carrot-on-Plate Performance Investigation

## Problem

When CGVD removes distractors, the VLA should see what looks like a clean scene — so performance should match the **no-distractor baseline (57%)**. It doesn't:

| Scenario | Success Rate | Gap from Clean Baseline |
|---|---|---|
| No distractors (clean baseline) | 57.0% ± 6.4% | — |
| 2 semantic distractors (no CGVD) | 50.5% ± 6.5% | -6.5% |
| **2 semantic distractors + CGVD** | **48.0% ± 9.3%** | **-9.0%** |
| 4 semantic distractors (no CGVD) | 53.5% ± 9.8% | -3.5% |
| **4 semantic distractors + CGVD** | **55.5% ± 9.6%** | **-1.5%** |

Key observations:
- **CGVD should recover to 57%, but it doesn't.** The 9% gap (n=2) and 1.5% gap (n=4) represent the "cost" of CGVD's imperfect image reconstruction.
- With 2 distractors, CGVD is actually worse than no CGVD at all (48% vs 50.5%) — its artifact cost exceeds its distractor-removal benefit.
- CGVD increases "dropped" failures from 20→29 (n=2), meaning the VLA grasps the carrot but loses it mid-trajectory.
- The n=0 CGVD test (57% = baseline) is misleading — with no distractors in the scene, SAM3 finds nothing and the image passes through completely unchanged. It never actually tests whether inpainting degrades image quality.

## Root Cause Analysis

CGVD does not produce images that look like a naturally clean scene. Instead it creates a **composite Frankenstein image** stitched from two different sources (a stale cached background and the live frame) with visible artifacts at the seams.

### Cause 1: Hard Compositing Seams (Primary — causes drops)

`src/cgvd/cgvd_wrapper.py:521-522`

```python
distractor_mask_3d = self.cached_mask[..., None] > 0.5
distilled = np.where(distractor_mask_3d, self.cached_inpainted_image, image)
```

This is a **binary pixel switch**:
- Pixel in distractor mask → from cached inpainted background (generated at frame ~6)
- Pixel outside mask → from live camera frame

At the mask boundary there is a hard pixel discontinuity — colors, lighting, and textures abruptly change. The VLA was trained on clean, consistent images and has never seen these artifacts. A naturally clean scene has no such boundary.

**Why this causes drops:** The robot picks up the carrot and moves it across the scene. As it passes over a distractor-masked region, the VLA sees a visual "glitch" at the mask boundary (abrupt pixel transition), misreads the carrot position, adjusts the gripper incorrectly, and drops it. This directly explains the 20→29 increase in drop failures.

### Cause 2: Stale Cached Inpainted Background

`src/cgvd/cgvd_wrapper.py:491-516`

The inpainted background is generated once at frame ~6 and reused for all remaining ~294 frames. As the robot moves, lighting and shadows change, making the color/texture mismatch at compositing boundaries progressively worse over time.

### Cause 3: LaMa Mask Dilation Bypasses Safe-Set Protection

The safe-set subtraction correctly protects the carrot (`cgvd_wrapper.py:472-474`):
```python
self.cached_mask = logical_and(distractor_mask > 0.5, safe_mask < 0.5)
```

But LaMa then dilates the mask by 11 pixels internally (`lama_inpainter.py:87-91`):
```python
kernel = np.ones((dilate_mask, dilate_mask), np.uint8)
mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
```

This dilation happens **after** safe-set subtraction, inside the inpainter. While the compositing uses the undilated mask (so carrot pixels come from the live frame), the inpainted background has hallucinated content where the carrot's edges should be. This degrades seam quality specifically near the carrot.

### Why n=4 Works Better Than n=2

With 4 distractors, the baseline VLA is genuinely confused by clutter (53.5%). CGVD's benefit (removing 4 confusing vegetables) outweighs its artifact cost, narrowing the gap to just 1.5% below clean baseline. With only 2 distractors, the VLA handles them naturally (50.5%), so CGVD's artifact cost exceeds its removal benefit, widening the gap to 9% below clean baseline.

## Fix Plan

### B.1: Soft Compositing with Feathered Mask (Highest Priority)

Replace the hard `np.where` with Gaussian-feathered alpha blending:

```python
feathered = cv2.GaussianBlur(self.cached_mask, (0, 0), sigmaX=5, sigmaY=5)
feathered_3d = feathered[..., None]
distilled = (feathered_3d * self.cached_inpainted_image.astype(np.float32) +
             (1.0 - feathered_3d) * image.astype(np.float32)).astype(np.uint8)
```

Smoothly blends inpainted and live regions over ~5 pixels, eliminating the hard seam that causes drop failures. The VLA should no longer see jarring pixel transitions at mask boundaries.

**Files:** `src/cgvd/cgvd_wrapper.py` lines 501-502 and 521-522

### B.2: Move Mask Dilation Before Safe-Set Subtraction

Dilate the distractor mask in the wrapper before safe-set subtraction, then pass `dilate_mask=0` to LaMa:

```python
dilation_kernel = np.ones((11, 11), np.uint8)
distractor_mask_dilated = cv2.dilate(
    (distractor_mask > 0.5).astype(np.uint8), dilation_kernel, iterations=1
).astype(np.float32)

self.cached_mask = np.logical_and(
    distractor_mask_dilated > 0.5, safe_mask < 0.5
).astype(np.float32)
```

Ensures dilation respects safe-set protection — any dilated pixels overlapping the carrot get re-excluded by the safe-set.

**Files:** `src/cgvd/cgvd_wrapper.py` lines 468-474, `src/cgvd/lama_inpainter.py` line 67

### B.3: Periodic Cache Refresh

Add `cache_refresh_interval` parameter (default 50 frames) to re-inpaint the background periodically, reducing the staleness that worsens compositing seams over time.

**Files:** `src/cgvd/cgvd_wrapper.py` `__init__` and lines 505-516

### B.4: Expose Ablation Parameters via CLI

Make dilation size, warmup frames, cache refresh interval, and blend sigma configurable from CLI for easy ablation testing without code edits.

**Files:** `scripts/try_checkpoint_in_simpler.py`, `scripts/clutter_eval/run_paired_eval.sh`

### B.5: Enhanced Debug Visualization

Add seam heatmap (`|distilled - current_frame|`), mask boundary contours, and "TARGET NOT IN SAFE SET" warning to debug images to make future diagnosis easier.

**Files:** `src/cgvd/cgvd_wrapper.py` method `_save_debug_images()`

## Post-B.5 Issue: Carrot Afterimage from Soft Compositing

After implementing fixes B.1-B.5 (soft compositing, safe dilation, cache refresh, CLI params, debug viz), a **carrot afterimage** appeared at the original plate position after the robot picks up the carrot. Performance did not improve.

### Root Cause

The `_composite` method applies `GaussianBlur(mask, sigma=5)` to feather ALL mask edges — including the inner edges where the safe-set carved a hole for the carrot. This bleeds the cached background (which contains the carrot at its original position) into the live frame (which has an empty plate after pickup).

```
cached_inpainted_image:  has carrot at original position (safe-set protected it)
live_frame:              carrot gone (robot picked it up)
feathered mask:          non-zero gradient at safe-set boundary
result at boundary:      0.2 × cached_bg(carrot) + 0.8 × live(empty) = ghost carrot
```

### Fix (B.6)

Re-enforce safe-set protection after GaussianBlur in `_composite`. Zero out any feathered values that overlap the safe-set region:

```python
feathered = cv2.GaussianBlur(mask, (0, 0), sigmaX=self.blend_sigma, sigmaY=self.blend_sigma)
if self.cached_safe_mask is not None:
    feathered = feathered * (1.0 - self.cached_safe_mask)
```

This ensures:
- **Outer edges** (distractor → background): still feathered (smooth transition)
- **Inner edges** (safe-set holes): hard boundary, 100% live frame
- **After carrot pickup**: no ghost, carrot region always shows live frame

## Verification

1. Run debug eval to visually inspect seam artifacts before/after fixes
2. Run paired eval: `run_paired_eval.sh --task widowx_carrot_on_plate -e 20 -r 3 -n 2`
3. **Target:** CGVD + distractors should approach the clean baseline (~57%), not just beat the distractor baseline
4. **Target:** "Dropped" count should decrease back toward baseline levels (~20)
5. Inspect "Seam Diff" heatmap — should show NO diff at carrot's original position after pickup
