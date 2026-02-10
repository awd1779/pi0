# Concept-Gated Visual Distillation (CGVD)

## Overview

CGVD is a visual preprocessing method that removes distractor objects from scenes using **LaMa inpainting**. It uses a two-stage segmentation approach with **safe-set protection** to guarantee that target objects are never affected.

## Motivation

Vision-Language-Action (VLA) models can be confused by cluttered scenes containing objects visually or semantically similar to the target. CGVD addresses this by:

1. Identifying distractor objects via SAM3 segmentation
2. Removing them via AI inpainting (LaMa) - filling with realistic background texture
3. Preserving task-relevant objects (target, anchor, robot) via safe-set protection

## Algorithm

### Step 1: Instruction Parsing

Parse the natural language instruction to extract task-relevant object names.

```
Input:  "put the spoon on the towel"
Output: target = "spoon", anchor = "towel"
```

### Step 2: Distractor Segmentation

Query SAM3 (Segment Anything Model 3) with known distractor object names.

```
Input:  image, distractor_names = ["fork", "scissors", "marker"]
Output: distractor_mask ∈ {0, 1}^(H×W)
        where 1 = distractor pixel
```

### Step 3: Safe-Set Segmentation

Query SAM3 with task-relevant concepts to create a protection mask.

```
Input:  image, safe_concepts = "spoon. towel. robot arm. robot gripper"
Output: safe_mask ∈ {0, 1}^(H×W)
        where 1 = task-relevant pixel
```

### Step 4: Safe-Set Subtraction

Compute the final blur mask by subtracting the safe-set from distractors.

```
final_mask = distractor_mask ∧ ¬safe_mask
```

This guarantees that task-relevant objects are **never** blurred, even if SAM3 incorrectly segments them as distractors.

### Step 5: LaMa Inpainting

Remove the masked distractor regions using LaMa (Large Mask Inpainting), which fills them with realistic background texture.

```python
# Apply LaMa inpainting to remove distractors
mask_dilated = dilate(final_mask, kernel=3)  # Clean up edges
output = LaMa.inpaint(image, mask_dilated)
```

**How LaMa works:**
- Uses Fast Fourier Convolutions (FFC) for global receptive field
- "Sees" the entire image context, not just local patches
- Generates realistic wood grain/table texture to fill masked regions
- Result looks like distractors were never there

**Performance:**
- ~100-800ms per frame (vs ~5ms for blur)
- ~2GB VRAM
- Still fast enough for robot control at 3-5Hz

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `update_freq` | Frames between SAM3 segmentation updates | 1 |
| `presence_threshold` | SAM3 confidence threshold for safe-set | 0.6 |
| `distractor_threshold` | SAM3 confidence threshold for distractors | 0.6 |
| `robot_threshold` | SAM3 confidence threshold for robot arm | 0.30 |

## Key Properties

### 1. Safe-Set Guarantee

The safe-set subtraction step ensures that target and anchor objects are never affected, regardless of SAM3 segmentation errors. If SAM3 incorrectly identifies the target spoon as a "fork" distractor, the safe-set mask will protect it.

### 2. LaMa Inpainting Quality

Unlike blur+darken approaches that create unnatural patches, LaMa produces seamless results:
- Fills masked regions with realistic table texture
- Uses global image context via Fast Fourier Convolutions
- Output is in-distribution for VLAs trained on clean scenes

### 3. Robot Preservation

The robot arm and gripper are always included in the safe-set to maintain proprioception alignment. The robot is never masked or inpainted.

## Visual Pipeline

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Input     │     │  SAM3 Segment   │     │  SAM3 Segment   │
│   Image     │────▶│  Distractors    │     │  Safe-Set       │
└─────────────┘     └────────┬────────┘     └────────┬────────┘
                             │                       │
                             ▼                       ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │ distractor_mask │     │   safe_mask     │
                    └────────┬────────┘     └────────┬────────┘
                             │                       │
                             └───────────┬───────────┘
                                         ▼
                              ┌─────────────────────┐
                              │   final_mask =      │
                              │   distractor ∧ ¬safe │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │   LaMa Inpainting   │
                              │   (fill with table) │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │   Output Image      │
                              │   (VLA input)       │
                              └─────────────────────┘
```

## Implementation Files

- `src/cgvd/cgvd_wrapper.py` - Main Gym wrapper implementing the pipeline
- `src/cgvd/lama_inpainter.py` - LaMa inpainting for distractor removal
- `src/cgvd/sam3_segmenter.py` - SAM3 interface for concept-based segmentation
- `src/cgvd/instruction_parser.py` - Natural language instruction parsing
- `src/cgvd/distractor_wrapper.py` - Physical distractor object placement in simulation

---

## Distractor Placement

The `DistractorWrapper` handles physical placement of distractor objects in SimplerEnv simulation. It uses **grid-based placement** to ensure all distractors fit on the table without overlapping.

### Placement Algorithm

1. **Load distractor objects** into the SAPIEN scene with appropriate scaling
2. **Compute XY bounding radius** for each object from its collision geometry
3. **Compute task object centroid** from safety bubble positions
4. **Task-centered grid placement**:
   - Create a ~30cm x 40cm grid centered around task objects (not full table)
   - Grid bounds: ±15cm in X, ±20cm in Y from centroid, clamped to table
   - Divide this centered region into a 4x4 grid (16 cells)
   - Filter out cells that overlap with task object safety bubbles
   - Shuffle available cells and assign one distractor per cell
   - Add random jitter within each cell for natural appearance
   - Overflow objects placed within centered grid if more distractors than cells
5. **Physics settling** lets objects fall and stabilize on the table

### Grid Layout

The grid is centered around task objects, not the full table. This ensures distractors appear in the same visual region as task objects rather than at distant table edges.

```
Table bounds: X: -0.35 to -0.02, Y: -0.28 to 0.28 (meters)
Grid radius: ±15cm in X, ±20cm in Y from task centroid

Example: If task objects are at centroid (-0.20, 0.05):
  Grid X: max(-0.35, -0.35) to min(-0.02, -0.05) = [-0.35, -0.05]
  Grid Y: max(-0.28, -0.15) to min(0.28, 0.25)  = [-0.15, 0.25]

     Y=grid_y_min      centroid    Y=grid_y_max
        ┌─────┬─────┬─────┬─────┐
grid_x  │  0  │  1  │  2  │  3  │
_min    ├─────┼─────┼─────┼─────┤
        │  4  │  5  │ [T] │  7  │  [T] = task objects
        ├─────┼─────┼─────┼─────┤
        │  8  │  9  │ 10  │ 11  │
        ├─────┼─────┼─────┼─────┤
grid_x  │ 12  │ 13  │ 14  │ 15  │
_max    └─────┴─────┴─────┴─────┘

Grid size: ~30cm x 40cm (smaller than full table)
Cell size: ~6cm x 9cm (varies based on centroid position)
Edge margin: 2cm from grid edges
```

### Bounding Box Computation

Each distractor's XY footprint is computed from its SAPIEN collision shapes:

```python
def get_actor_xy_radius(actor):
    for shape in actor.get_collision_shapes():
        geom = shape.geometry
        if isinstance(geom, ConvexMeshGeometry):
            verts = geom.vertices * geom.scale
            extent = verts[:, :2].max(axis=0) - verts[:, :2].min(axis=0)
            radius = np.sqrt(extent[0]**2 + extent[1]**2) / 2
        elif isinstance(geom, BoxGeometry):
            half = geom.half_lengths
            radius = np.sqrt(half[0]**2 + half[1]**2)
        # ... handle other geometry types
    return max_radius
```

### Placement Constraints

| Constraint | Description |
|------------|-------------|
| **Safety bubbles** | Circular exclusion zones around task objects (bounding box radius, no padding) |
| **Task centroid** | Grid centered on mean position of task objects |
| **Grid assignment** | One object per cell, cells with safety bubble overlap are excluded |
| **Table bounds** | X: -0.35 to -0.02, Y: -0.28 to 0.28 (meters, grid clamped to these) |
| **Edge margin** | 2cm inset from grid edges |
| **Spawn height** | 10cm above table surface, physics settles objects down |

### Placement Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `GRID_COLS` | Number of grid columns | 4 |
| `GRID_ROWS` | Number of grid rows | 4 |
| `GRID_RADIUS_X` | Grid half-width from centroid | 15cm |
| `GRID_RADIUS_Y` | Grid half-height from centroid | 20cm |
| `EDGE_MARGIN` | Inset from grid edges | 2cm |
| `SAFETY_PADDING` | Extra padding around task objects | 0cm |
| `FALLBACK_RADIUS` | Default radius if bounding box unavailable | 6cm |

### Object Scaling

External dataset objects (YCB, RoboCasa) are scaled to appropriate sizes:

| Object Type | Scale | Reason |
|-------------|-------|--------|
| Built-in (eggplant, cubes) | 1.0 | Already correctly sized |
| Utensils (fork, knife, spoon) | 1.0 | Realistic utensil size |
| External (ycb_*, rc_*) | 0.1 | External assets tend to be oversized |
| Per-object override | `object_id:scale` syntax | Fine-tuning specific objects |

## Usage Example

```bash
# Via paired eval script (recommended)
./scripts/clutter_eval/run_paired_eval.sh --task widowx_spoon_on_towel

# Or direct Python usage
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --distractors ycb_030_fork:0.55 ycb_033_spatula:0.50 \
    --use_cgvd \
    --cgvd_use_inpaint
```
