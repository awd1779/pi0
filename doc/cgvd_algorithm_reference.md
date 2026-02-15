# CGVD Algorithm Reference

**Concept-Gated Visual Distillation: Complete Algorithmic Description**

*Derived entirely from source code analysis. All line references are to the codebase at time of writing.*

---

## 1. Architecture Overview

CGVD is implemented as a `gym.Wrapper` (`CGVDWrapper` in `src/cgvd/cgvd_wrapper.py`) that intercepts visual observations from a robot manipulation environment and removes visual distractors while preserving task-relevant objects.

### 1.1 System Components

| Component | File | Role |
|-----------|------|------|
| CGVDWrapper | `src/cgvd/cgvd_wrapper.py` | Main pipeline: mask computation, compositing, clean plate management |
| SAM3Segmenter | `src/cgvd/sam3_segmenter.py` | Text-prompted instance segmentation (SAM3 model) |
| LamaInpainter | `src/cgvd/lama_inpainter.py` | Neural inpainting for clean plate generation |
| InstructionParser | `src/cgvd/instruction_parser.py` | Extracts target/anchor objects from language instructions |

### 1.2 Pipeline Summary

For each frame, CGVD performs the following:

```
Language instruction  -->  InstructionParser  -->  (target, anchor)
                                                        |
Camera image  --------->  SAM3 Segmenter  --------->  distractor masks
                    |                                   safe-set masks
                    |                                   robot mask
                    v
              Mask Computation:  final_mask = distractor AND NOT safe
                    |
                    v
              LaMa Inpainting  -->  clean plate (distractors + robot removed)
                    |
                    v
              Compositing  -->  distilled image (distractors hidden, target + robot preserved)
```

### 1.3 Wrapper Integration

CGVDWrapper overrides two methods:

- **`reset()`**: Clears all cached state, runs the warmup phase (N no-op frames to accumulate masks), then returns the first distilled observation. The VLA never sees warmup frames.
- **`step(action)`**: Takes environment step, applies CGVD to the resulting observation, returns the distilled observation.

---

## 2. Instruction Parsing

**File:** `src/cgvd/instruction_parser.py`

### 2.1 Target and Anchor Extraction

The `InstructionParser` maps natural language instructions to a `(target, anchor)` tuple:

- **Target**: The object being manipulated (e.g., "spoon")
- **Anchor**: The destination or reference object (e.g., "towel"), or `None` if absent

Parsing uses a two-tier strategy:

1. **Pattern matching** (priority): A table of regex patterns maps known task instructions to fixed `(target, anchor)` pairs. Examples:
   - `"spoon.*towel"` -> `("spoon", "towel")`
   - `"carrot.*plate"` -> `("carrot", "plate")`
   - `"pick.*coke"` -> `("coke can", None)`

2. **Heuristic fallback**: If no pattern matches, the parser strips action verbs ("pick", "place", "move", ...) and articles, then takes the first remaining noun as target. Anchor is extracted from prepositional phrases ("on the plate" -> "plate").

### 2.2 SAM3 Concept Prompt Construction

The parser builds a dot-separated prompt string for SAM3:

```
build_concept_prompt(target="spoon", anchor="towel", include_robot=True)
  -->  "spoon. towel. robot arm. robot gripper"
```

Each concept is queried independently by SAM3, sharing pre-computed vision embeddings for efficiency.

---

## 3. SAM3 Segmentation

**File:** `src/cgvd/sam3_segmenter.py`

### 3.1 Model Architecture

CGVD uses SAM3 (`facebook/sam3`), a text-prompted instance segmentation model. The segmenter supports three backends:

- **SAM3Segmenter**: Direct model inference (default). Uses `float16` precision on CUDA.
- **SAM3ClientSegmenter**: Client-server architecture for environments with dependency conflicts. Communicates via file-based IPC (`/tmp/sam3_server`).
- **MockSAM3Segmenter**: Testing backend that returns center-weighted Gaussian masks.

All backends use a singleton pattern to avoid redundant model loading across wrapper instances.

### 3.2 Multi-Concept Segmentation

The `segment()` method processes a dot-separated concept string:

1. Parse concepts: `"spoon. carrot. robot arm"` -> `["spoon", "carrot", "robot arm"]`
2. **Pre-compute vision embeddings** once via `model.get_vision_features()` (avoids redundant encoder forward passes across concepts)
3. For each concept, run `_segment_single_concept()` with the shared vision embeddings
4. Combine per-concept masks via `np.maximum()` (union)
5. Binarize the combined mask at threshold 0.5

### 3.3 Instance-Level Output

For each concept, SAM3 may return multiple instances (e.g., two spoons). The segmenter stores:

- `last_scores`: Dict mapping instance name -> confidence score
- `last_individual_masks`: Dict mapping instance name -> binary mask `(H, W)`

**Naming convention:**
- Single instance: stored under the base concept name (e.g., `"spoon"`)
- Multiple instances: indexed names without the base name (e.g., `"spoon_0"`, `"spoon_1"`)

This per-instance output enables the downstream cross-validation and top-1 filtering algorithms.

### 3.4 Three-Tier Confidence Thresholds

CGVD uses different SAM3 confidence thresholds for different object categories:

| Category | Threshold | Rationale |
|----------|-----------|-----------|
| Robot arm/gripper | 0.05 | Very permissive — robot must always be detected to prevent proprioception artifacts |
| Safe-set (target/anchor) | 0.15 | Moderately permissive — target may be partially occluded by gripper during warmup |
| Distractors | 0.30 | Strict — reduces false positives (e.g., carrot misdetected as banana) |

---

## 4. Two-Phase Operation

### 4.1 Warmup Phase

During `reset()`, CGVD runs `safeset_warmup_frames` (default: 5) internal steps before the VLA receives its first observation:

```python
for i in range(safeset_warmup_frames):
    _apply_cgvd(obs)          # Accumulate masks, skip compositing
    obs = env.step(zeros(7))  # No-op action (hold position, gripper unchanged)
```

**Purpose**: Accumulate robust distractor and safe-set masks across multiple frames. SAM3 detection is unreliable on individual frames (scores can drop to 0.0 between frames), so union accumulation across warmup frames provides coverage.

**Key properties:**
- The VLA never sees warmup frames
- Compositing is skipped during warmup (line 891: `if frame_count <= safeset_warmup_frames: return obs`)
- The environment's `TimeLimit` is extended by `safeset_warmup_frames` steps so the VLA retains its full action budget

### 4.2 Robot-Free Rendering

During warmup, CGVD renders the scene with the robot hidden for safe-set SAM3 queries:

```python
if in_warmup:
    safe_query_image = _render_robot_free_image(camera_name)
```

This hides robot visual meshes via `link.hide_visual()`, re-renders the scene, then restores the robot. The robot-free image gives SAM3 an unoccluded view of the target object.

**Motivation**: Without this, the robot gripper hovering over the real spoon suppresses its SAM3 confidence, while a spatula (fully visible) may win top-1 filtering and lock into the safe-set incorrectly.

**Important**: The robot-free image is used ONLY for SAM3 queries. The clean plate is computed from the real image (with robot visible) because SAPIEN's IBL renderer recomputes lighting when the robot is hidden, causing a global color shift that creates visible seams during compositing.

### 4.3 Clean Plate Pre-Computation

On the **last warmup frame**, CGVD pre-computes the clean plate:

```python
if frame_count == safeset_warmup_frames:
    cached_inpainted_image = inpainter.inpaint(image, _build_inpaint_mask(), dilate_mask=0)
```

The clean plate is computed once and reused for all subsequent frames. The inpaint mask includes both distractors and the robot (see Section 10).

---

## 5. Distractor Mask Accumulation

### 5.1 Detection

Each warmup frame, CGVD queries SAM3 for all distractors simultaneously. During warmup, the robot-free image (`safe_query_image`) is used for both distractor and safe-set queries so SAM3 sees objects unoccluded by the robot arm:

```python
distractor_concepts = ". ".join(distractor_names)  # e.g., "carrot. banana. fork"
raw_distractor_mask = segmenter.segment(safe_query_image, distractor_concepts, presence_threshold=0.3)
```

### 5.2 Accumulation Strategy

| Phase | Behavior |
|-------|----------|
| First frame | `cached_distractor_mask = raw` (initialize) |
| Warmup (frames 1..N-1) | `cached_distractor_mask = max(cached, raw)` (union accumulation) |
| Post-warmup | Frozen (when `cache_distractor_once=True`, default) |

Union accumulation (`np.maximum`) means the distractor mask can only grow during warmup, never shrink. This handles transient SAM3 misdetections where a distractor is missed on one frame but detected on the next.

### 5.3 Morphological Dilation

Before safe-set subtraction, the distractor mask is dilated:

```python
if lama_dilation > 0:
    kernel = ones((lama_dilation, lama_dilation))   # default: 11x11
    distractor_mask = dilate(distractor_mask > 0.5, kernel)
```

**Purpose**: Expand the distractor region so that LaMa inpainting covers the full extent of each distractor object including any GaussianBlur spread. The dilation is applied BEFORE safe-set subtraction, so the dilated region is properly gated by the safe set (task objects never bleed into the inpaint region).

---

## 6. Safe-Set Construction: Three-Layer Filtering

The safe-set identifies pixels that must be **preserved** (shown from the live frame, never inpainted). It consists of three object categories:

- **Target**: The object being manipulated (e.g., spoon)
- **Anchor**: The destination or reference object (e.g., towel)
- **Robot**: The robot arm and gripper (handled separately, see Section 7)

CGVD uses a three-layer sequential filtering strategy to build a robust safe-set from potentially noisy SAM3 detections.

### 6.1 Layer 1: Cross-Validation (Genuineness Scoring)

**Method:** `_cross_validate_safeset()` (line 312)

**Problem**: SAM3 may return multiple instances for the target concept (e.g., `spoon_0` = real spoon, `spoon_1` = spatula misclassified as spoon). If the wrong instance is accumulated, a distractor enters the safe-set and becomes permanently visible.

**Algorithm**: For each TARGET instance (anchors are never filtered):

1. Compute IoU between the target instance mask and every distractor mask
2. For overlapping distractors (IoU > 0.3), find the highest-scoring distractor
3. Compute **genuineness**:

```
genuineness = safe_score - max_overlapping_distractor_score
```

4. **Decision rule**:
   - The instance with the highest genuineness is ALWAYS kept (even if negative)
   - Other instances with `genuineness < genuineness_margin` (default: -0.1) are removed

**Intuition**: A real spoon might score 0.85 as "spoon" and overlap with a spatula scoring 0.70 as a distractor -> genuineness = +0.15 (keep). A spatula misdetected as "spoon" might score 0.75 but overlap with its own correct distractor detection at 0.82 -> genuineness = -0.07 (remove, since < -0.1 would apply to non-best instances).

**Critical ordering**: Cross-validation runs BEFORE top-1 filtering, on the full multi-instance set. This allows false positives to be removed before top-1 could inadvertently select them.

**Critical accumulation rule**: The false-positive mask is subtracted from THIS FRAME's masks only (before accumulation), not from the cumulative `cached_safe_mask`. Subtracting from the accumulated union would be destructive: one bad frame where the real spoon has negative genuineness would wipe pixels accumulated across all previous good frames.

### 6.2 Layer 2: Top-1 Filtering

**Method:** `_filter_overlapping_detections()` (line 263)

After cross-validation cleans the multi-instance set, top-1 filtering keeps only the highest-confidence instance per concept:

```python
for each concept:
    instances.sort(by=score, descending)
    keep instances[0]  # highest score wins
```

This reduces the per-concept output to at most one mask, preventing SAM3 multi-instance confusion from propagating downstream.

### 6.3 Layer 2b: IoU-Gated Target Accumulation

**Method:** `_accumulate_target()` (line 396)

After filtering, the surviving target mask is accumulated into `cached_target_mask` using `np.maximum` (union). However, to prevent spatially inconsistent detections from contaminating the cache, an IoU gate is applied:

**Algorithm:**

| Condition | Behavior |
|-----------|----------|
| First detection ever | Initialize `cached_target_mask` and per-pixel vote counter |
| Early frames (`frame < iou_gate_start_frame`, default: 2) | Accumulate unconditionally |
| Normal warmup frames | Compute IoU between new detection and `cached_target_mask`. If `IoU > iou_gate_threshold` (default: 0.15), accumulate. Otherwise, reject. |

**Per-pixel vote tracking**: Each time a pixel is accumulated, its vote counter is incremented:

```python
_target_votes += (new_mask > 0.5).astype(float32)
```

This records how many warmup frames detected each pixel, providing a temporal consistency measure used in Layer 3.

**Anchor accumulation**: Anchor masks are accumulated unconditionally (never filtered, never IoU-gated).

### 6.4 Layer 3: Post-Warmup Spatial Cleanup

**Method:** `_cleanup_target_mask()` (line 443)

**Problem**: Despite Layers 1-2, the accumulated target mask may contain multiple spatially disjoint components (e.g., the real spoon + pixels from a fork that was misdetected as "spoon" on some frames).

**Algorithm:**

1. **Connected component analysis** on the binarized `cached_target_mask`:
   ```python
   num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=4)
   ```
   Uses 4-connectivity (not 8) to reduce false merges from diagonally-adjacent masks.

2. **Score each component** using temporal consistency and distractor contamination:
   ```python
   avg_votes = target_votes[component].mean()       # How consistently detected
   dist_overlap = AND(component, distractor > 0.5).sum() / pixel_count  # Distractor contamination
   overlap_penalty = min(dist_overlap, overlap_penalty_cap)             # Cap at 0.7
   score = avg_votes * (1.0 - overlap_penalty)
   ```

3. **Keep only the best-scoring component**, discard all others.

**Why this works**: Consider two components:
- Real spoon: detected in 4/5 warmup frames (avg_votes ~ 0.8), 20% distractor overlap -> score = 0.8 * 0.8 = 0.64
- Fork-as-spoon: detected in 2/5 frames (avg_votes ~ 0.4), 95% distractor overlap (it IS a fork distractor) -> score = 0.4 * 0.3 = 0.12

The real spoon wins decisively.

**Trigger conditions:**
- Runs once at the end of warmup (last warmup frame)

After cleanup, `_recompute_cached_safe_mask()` merges the cleaned target with the anchor:
```python
cached_safe_mask = max(cached_target_mask, cached_anchor_mask)
```

---

## 7. Robot Arm Protection

The robot arm requires special handling because it is:
1. Always present and always task-relevant (must never be inpainted)
2. Moving every frame (unlike stationary target/anchor)

### 7.1 Per-Frame Detection

The robot is segmented from the **live frame** (not the robot-free render) every frame:

```python
robot_mask = segmenter.segment(image, "robot arm. robot gripper", presence_threshold=0.05)
```

The very low threshold (0.05) ensures the robot is always detected, even with low SAM3 confidence.

### 7.2 Decoupling from Cached Mask

**Critical design decision**: The robot mask is NOT used in the Step 4 mask computation (`cached_mask = distractor AND NOT safe`). Instead, Step 4 uses only `cached_safe_mask` (target + anchor, no robot):

```python
safe_mask_for_gating = dilate(cached_safe_mask > 0.5, kernel)  # No robot
cached_mask = AND(distractor > 0.5, safe_mask_for_gating < 0.5)
```

**Rationale**: The robot moves every frame. If robot pixels were subtracted from `cached_mask`, the robot-shaped holes would shift each frame. When GaussianBlur is applied during compositing, different feathered values would be produced at different robot positions -> temporal flicker.

By excluding the robot from `cached_mask`, the mask is stable across frames (only depends on stationary distractors and safe-set). Robot visibility is instead handled in the compositing stage via re-enforcement (Section 9).

### 7.3 Robot in Warmup Accumulation

During warmup, the robot mask is accumulated for clean plate generation:

```python
if in_warmup:
    cached_robot_mask = max(cached_robot_mask, robot_mask)
```

### 7.4 Safe Mask Composition

The per-frame safe mask combines the stationary cached safe-set with the fresh robot detection:

```python
current_safe_mask = max(cached_safe_mask, robot_mask)  # target + anchor + robot
```

This `current_safe_mask` is used in compositing re-enforcement (Section 9), not in Step 4 gating.

---

## 8. Final Mask Computation (Step 4)

### 8.1 Safe-Set Dilation

Before subtraction, the safe-set mask is dilated to create a protective buffer:

```python
# _step4_safe_dilation = max(safe_dilation, lama_dilation)  # default: max(5, 11) = 11
safe_dilation_kernel = ones((_step4_safe_dilation, _step4_safe_dilation))  # default: 11x11
safe_mask_for_gating = dilate(cached_safe_mask > 0.5, safe_dilation_kernel)
```

**Purpose**: SAM3 tends to under-segment object boundaries by 1-3 pixels. Without dilation, the GaussianBlur feathering in compositing would bleed inpainted (table texture) values into the gap between SAM3's boundary and the actual object edge, creating a visible halo. The 11px dilation (~5.5px buffer) prevents this. The dilation uses `_step4_safe_dilation = max(safe_dilation, lama_dilation)` to ensure the safe buffer is at least as large as the distractor dilation, preventing distractor dilation from encroaching into safe-set territory.

### 8.2 Mask Subtraction

```python
cached_mask = AND(distractor_mask > 0.5, safe_mask_for_gating < 0.5)
```

Verbally: a pixel is in the final mask if and only if it is (a) detected as a distractor AND (b) not in the dilated safe set.

**Binary thresholding**: Both masks are binarized at 0.5 before the logical operation. This prevents soft SAM3 boundary values (e.g., 0.6-0.7) from leaking through the logical AND.

### 8.3 Mask Variable Summary

| Variable | Contents | Updated When |
|----------|----------|--------------|
| `cached_distractor_mask` | Raw (undilated) distractor accumulation | Warmup: union each frame. Post-warmup: frozen |
| `distractor_mask` | Dilated version of cached_distractor_mask | Computed each frame from cached + dilation |
| `cached_safe_mask` | max(target, anchor) — stationary objects only | Warmup + Layer 3 cleanup |
| `cached_target_mask` | Accumulated + cleaned target detections | Warmup + IoU gating + Layer 3 cleanup |
| `cached_anchor_mask` | Accumulated anchor detections | Warmup (unconditional) |
| `cached_robot_mask` | Accumulated robot during warmup | Warmup only |
| `last_robot_mask` | This frame's robot detection | Every frame |
| `current_safe_mask` | max(cached_safe_mask, last_robot_mask) — includes robot | Every frame |
| `cached_mask` | Inpaint-region mask: **dilated** distractor AND NOT dilated_safe | Every frame (stable post-warmup) |
| `cached_compositing_mask` | Compositing mask: **undilated** distractor AND NOT dilated_safe | Every frame (stable post-warmup) |
| `safe_mask_for_gating` | Dilated cached_safe_mask (for Step 4 only) | Every frame |

---

## 9. Compositing Pipeline

**Method:** `_composite()` (line 958)

The compositing pipeline blends the cached clean plate (inpainted background) with the live camera frame, using `cached_compositing_mask` to determine which pixels show the clean plate vs. the live frame. Note: compositing uses `cached_compositing_mask` (undilated distractor), not `cached_mask` (dilated distractor used for inpainting). This ensures the GaussianBlur transition starts at the actual distractor boundary rather than `lama_dilation` pixels beyond it.

### 9.1 Feathered Blending (Step 1)

```python
feathered = GaussianBlur(cached_compositing_mask, sigma=blend_sigma)  # default: sigma=3.0
```

The binary compositing mask is blurred to create smooth alpha transitions at distractor boundaries. With `sigma=3.0`, the blur has visible effect within ~9 pixels (3 sigma) of each boundary.

### 9.2 Binary Mask Preparation (Step 2)

All masks are binarized to prevent soft SAM3 values from leaking:

```python
safe = (current_safe_mask > 0.5).astype(float32)         # target + anchor + robot
binary_target = (cached_safe_mask > 0.5).astype(float32)  # target + anchor only
```

The `binary_target` mask is dilated by `_reinforce_size = _step4_safe_dilation + 3 * ceil(blend_sigma)` pixels (default: 11 + 9 = 20):

```python
binary_target = dilate(binary_target, kernel=(reinforce_size, reinforce_size))
```

This dilation ensures that GaussianBlur's feathered values are negligible (< 2%) at the re-enforcement boundary, eliminating any table-color outline artifact.

### 9.3 Mechanism 2: Distractor Clamping (Step 3)

```python
binary_distractor = (cached_distractor_mask > 0.5).astype(float32)
non_safe_distractor = binary_distractor * (1.0 - binary_target)
feathered = max(feathered, non_safe_distractor)
```

**Problem**: GaussianBlur spreads the feathered values outward from the mask boundary. At pixels that are distractor but not currently in `cached_mask` (e.g., due to mask boundary imprecision), the feathered value might be low (0.3-0.6), allowing 40-70% of the distractor to leak through.

**Solution**: For every pixel that is a distractor AND not in the dilated safe-set, clamp `feathered` to at least 1.0. This forces 100% of the inpainted background at distractor locations.

**Binarization is critical**: Without binarizing `cached_distractor_mask`, soft SAM3 edge values (e.g., 0.6) would cause `np.maximum(feathered, 0.6)` to clamp to only 60%, leaking 40% of the distractor through.

### 9.4 Re-Enforcement: Safe-Set Protection (Step 4)

```python
reinforce_mask = max(safe, binary_target)
feathered = feathered * (1.0 - reinforce_mask)
```

**Purpose**: Zero out the feathered value at all safe-set pixels (target + anchor + robot), guaranteeing they show 100% of the live frame.

The `reinforce_mask` combines two masks:
- `safe` (binarized `current_safe_mask`): includes this frame's robot detection
- `binary_target` (dilated `cached_safe_mask`): provides extra halo protection for stationary target/anchor

Robot uses the raw SAM3 mask via `safe` — the 1-2px SAM3 boundary under-segmentation is imperceptible since re-enforcement shows the live frame directly.

### 9.5 Final Alpha Blend (Step 5)

```python
result = feathered * inpainted + (1.0 - feathered) * image
```

At each pixel:
- `feathered = 1.0`: Show 100% clean plate (distractor fully hidden)
- `feathered = 0.0`: Show 100% live frame (target/robot fully visible)
- `0 < feathered < 1`: Smooth blend at boundaries

### 9.6 Compositing Summary

The four steps form a defense-in-depth:

1. **GaussianBlur**: Creates smooth alpha transitions (prevents hard seams)
2. **Mechanism 2 (clamp)**: Forces distractor pixels to show clean plate (prevents leakage from soft SAM3 values)
3. **Re-enforcement**: Forces safe-set pixels to show live frame (prevents target/robot from being inpainted)
4. **Alpha blend**: Produces final seamless composite

---

## 10. Clean Plate Generation

### 10.1 Inpaint Mask Construction

**Method:** `_build_inpaint_mask()` (line 1028)

The inpaint mask includes both distractors and the robot:

```python
mask = cached_mask.copy()           # Distractors (already subtracted safe-set)
robot = cached_robot_mask or last_robot_mask
robot = dilate(robot > 0.5, kernel=(reinforce_size, reinforce_size))
mask = max(mask, robot)
```

**Why include robot**: The clean plate should show only the table/background. Including the robot in the inpaint mask means LaMa removes both distractors and the robot, producing a clean background.

**Why dilate robot**: The gap between the undilated robot mask and the dilated-safe hole in `cached_mask` leaves un-inpainted pixels that retain stale robot arm color. Dilating by `_reinforce_size` (default: 20px) covers the GaussianBlur spread.

**Why NOT include target**: Keeping the target visible in the clean plate means that even if compositing feathering is imperfect near a distractor boundary, the target still appears (from the clean plate side). A minor ghost at the target's old position after pickup is far less harmful than the target disappearing during approach.

### 10.2 LaMa Inpainting

**File:** `src/cgvd/lama_inpainter.py`

The `LamaInpainter` wraps the `SimpleLama` model:

```python
def inpaint(image, mask, dilate_mask=11):
    mask_uint8 = (mask * 255).astype(uint8)
    if dilate_mask > 0:
        mask_uint8 = dilate(mask_uint8, kernel=(dilate_mask, dilate_mask))
    result = model(image, mask_uint8)
    return array(result)
```

The inpainter supports optional mask dilation (default 11px), but when called from `_build_inpaint_mask()`, dilation is already applied upstream, so `dilate_mask=0` is passed.

The inpainter uses a singleton pattern with lazy model loading to avoid redundant initialization.

### 10.3 Clean Plate Lifecycle

| Event | Action |
|-------|--------|
| Last warmup frame | Compute clean plate from **real image** (with robot), using full inpaint mask (distractors + dilated robot) |
| Post-warmup frames | Reuse cached clean plate (no recomputation by default) |
| Optional periodic refresh | If `cache_refresh_interval > 0`, recompute every N frames (disabled by default; introduces visual jumps) |

---

## 11. Complete Frame-by-Frame Trace

### 11.1 Warmup Frame (frame 0 through N-1)

```
1. Extract image from observation
2. Parse instruction -> (target, anchor)
3. Segment distractors from robot-free image (threshold=0.3)
   -> cached_distractor_mask = max(cached, raw)  [union]
4. Render robot-free image (hide robot meshes, re-render)
5. Segment target+anchor from robot-free image (threshold=0.15)
   a. Layer 1: Cross-validate instances (genuineness scoring)
   b. Top-1: Keep highest-scoring instance per concept
   c. Accumulate target (IoU-gated) and anchor (unconditional)
   d. Recompute: cached_safe_mask = max(target, anchor)
   e. If last warmup frame: Layer 3 cleanup (connected components)
6. Segment robot from live image (threshold=0.05)
   -> cached_robot_mask = max(cached, robot)
7. Dilate safe mask (11x11 kernel, using _step4_safe_dilation)
8. Dilate distractor mask (11x11 kernel, using lama_dilation)
9. cached_mask = dilated_distractor AND NOT dilated_safe
   cached_compositing_mask = undilated_distractor AND NOT dilated_safe
10. If last warmup frame: pre-compute clean plate via LaMa
11. Skip compositing, return observation unchanged
```

### 11.2 Post-Warmup Frame (frame N onward)

```
1. Extract image from observation
2. Parse instruction -> (target, anchor) [usually cached]
3. Use frozen cached_distractor_mask (no recomputation)
4. Use frozen cached_safe_mask (no recomputation)
5. Segment robot from live image (fresh every frame, threshold=0.05)
   -> current_safe_mask = max(cached_safe_mask, robot_mask)
6. Dilate cached_safe_mask for gating (11x11 kernel, using _step4_safe_dilation)
7. Dilate cached_distractor_mask for inpainting (11x11 kernel, using lama_dilation)
8. cached_mask = dilated_distractor AND NOT dilated_safe  [stable, since all inputs are frozen except robot, which is excluded]
   cached_compositing_mask = undilated_distractor AND NOT dilated_safe
9. Composite (using cached_compositing_mask):
   a. GaussianBlur(cached_compositing_mask, sigma=3.0) -> feathered
   b. Mechanism 2: Clamp feathered at binary distractor pixels
   c. Re-enforcement: Zero feathered at safe-set pixels (incl. robot)
   d. result = feathered * clean_plate + (1-feathered) * live_image
10. Write distilled image back to observation
```

---

## 12. Parameters Reference

### 12.1 Core Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `update_freq` | 1 | Frames between SAM3 updates (1 = every frame) |
| `include_robot` | True | Include robot arm/gripper in safe-set |
| `distractor_names` | [] | List of distractor object names to remove |
| `cache_distractor_once` | True | Freeze distractor mask after warmup |

### 12.2 Detection Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `presence_threshold` | 0.15 | SAM3 confidence threshold for safe-set (target/anchor) |
| `robot_presence_threshold` | 0.05 | SAM3 confidence threshold for robot detection |
| `distractor_presence_threshold` | 0.30 | SAM3 confidence threshold for distractor detection |

### 12.3 Warmup Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `safeset_warmup_frames` | 5 | Number of no-op frames for mask accumulation |

### 12.4 Compositing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `blend_sigma` | 3.0 | GaussianBlur sigma for feathered blending (~9px visible spread) |
| `lama_dilation` | 11 | Distractor mask dilation before inpainting (11x11 kernel) |
| `safe_dilation` | 5 | Safe-set mask dilation for protective buffer (5x5 kernel) |
| `cache_refresh_interval` | 0 | Frames between clean plate refresh (0 = never refresh) |

**Derived parameters**:
- `_step4_safe_dilation = max(safe_dilation, lama_dilation)` = max(5, 11) = 11 pixels. Used for dilating the safe-set mask in Step 4 gating.
- `_reinforce_size = _step4_safe_dilation + 3 * ceil(blend_sigma)` = 11 + 3*3 = 20 pixels. Used for dilating `binary_target` in compositing re-enforcement and robot mask in inpaint mask construction.

### 12.5 Safe-Set Robustness Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `genuineness_margin` | -0.1 | Threshold for cross-validation: instances with genuineness below this are removed |
| `iou_gate_threshold` | 0.15 | Minimum IoU for accepting a new target detection into the accumulator |
| `iou_gate_start_frame` | 2 | Frame index when IoU gating becomes active |
| `min_component_pixels` | 50 | Minimum pixel count for a valid connected component |
| `overlap_penalty_cap` | 0.7 | Maximum penalty for distractor overlap in Layer 3 scoring |

### 12.6 Ablation Flags

| Parameter | Default | Description |
|-----------|---------|-------------|
| `disable_safeset` | False | Skip safe-set subtraction (mask distractors without protecting target) |
| `disable_inpaint` | False | Use mean-color fill instead of LaMa inpainting |

---

## 13. Algorithmic Formulas Summary

### Genuineness (Layer 1 Cross-Validation)

```
genuineness(instance_i) = confidence(instance_i, target_concept)
                        - max_{d in distractors, IoU(instance_i, d) > 0.3} confidence(d)
```

Decision: Keep if `genuineness >= genuineness_margin` OR if `instance_i` has the highest genuineness among all target instances.

### IoU Gate (Layer 2b Target Accumulation)

```
IoU = |new_binary AND cached_binary| / |new_binary OR cached_binary|

Accumulate if IoU > iou_gate_threshold (and frame >= iou_gate_start_frame)
```

### Component Score (Layer 3 Cleanup)

```
avg_votes = mean(target_votes[component_pixels])
dist_overlap = |component AND distractor_mask| / |component|
overlap_penalty = min(dist_overlap, overlap_penalty_cap)
score = avg_votes * (1.0 - overlap_penalty)
```

Keep the component with the highest score.

### Compositing

```
feathered = GaussianBlur(cached_compositing_mask, sigma=blend_sigma)

// Mechanism 2: Clamp at distractor pixels
non_safe_distractor = binarize(distractor) * (1 - dilated_binary_target)
feathered = max(feathered, non_safe_distractor)

// Re-enforcement: Protect safe-set
reinforce = max(binarize(current_safe), dilated_binary_target)
feathered = feathered * (1 - reinforce)

// Blend
result = feathered * clean_plate + (1 - feathered) * live_frame
```

### Mask Computation (Step 4)

```
cached_mask = binarize(dilated_distractor) AND NOT binarize(dilated_safe)
cached_compositing_mask = binarize(undilated_distractor) AND NOT binarize(dilated_safe)
```

Where:
- `dilated_distractor = morphological_dilate(cached_distractor_mask, kernel=lama_dilation)`
- `undilated_distractor = binarize(cached_distractor_mask)` (before lama_dilation)
- `dilated_safe = morphological_dilate(cached_safe_mask, kernel=_step4_safe_dilation)`
- `_step4_safe_dilation = max(safe_dilation, lama_dilation)` (default: 11)
- `binarize(x) = (x > 0.5)`

`cached_mask` is used for inpaint mask construction; `cached_compositing_mask` is used for compositing.
