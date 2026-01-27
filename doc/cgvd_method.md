# Concept-Gated Visual Distillation (CGVD)

## Overview

CGVD is a visual preprocessing method that selectively blurs distractor objects while preserving task-relevant regions. It uses a two-stage segmentation approach with **safe-set protection** to guarantee that target objects are never affected by the blur operation.

## Motivation

Vision-Language-Action (VLA) models can be confused by cluttered scenes containing objects visually or semantically similar to the target. CGVD addresses this by:

1. Identifying and blurring distractor objects to reduce visual salience
2. Preserving sharp details for task-relevant objects (target, anchor, robot)
3. Blending blurred regions into local background colors for natural appearance

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

### Step 5: Table Surface Detection and Color Sampling

Segment the table surface to sample the actual background color, then apply blur and blend.

```python
# Step 5a: Segment table surface (first frame only)
table_mask = SAM3.segment(image, "wooden table. table surface")
table_color = mean(image[table_mask])  # RGB color of table

# Step 5b: Apply blur and blend toward table color
blurred = GaussianBlur(image, σ)
darkened = (1 - α) × blurred + α × table_color

# Step 5c: Composite final image
output = (1 - M) × image + M × darkened
```

The table color is **cached** after the first frame since the table doesn't move.

Where:
- `σ` = blur sigma (controls blur strength)
- `α` = darken_strength (controls background blending)
- `M` = final_mask (binary mask of regions to blur)
- `table_color` = RGB color sampled from table surface via SAM3

## Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `blur_sigma` | Gaussian blur standard deviation. With table color sampling, lower values (15-50) are sufficient since color comes from SAM3, not blur. | 15 - 50 |
| `darken_strength` | Blend factor toward table color (0 = pure blur, 1 = solid table color) | 0.5 - 0.8 |
| `update_freq` | Frames between SAM3 segmentation updates | 1 - 10 |
| `presence_threshold` | SAM3 confidence threshold for accepting masks | 0.4 - 0.6 |

## Key Properties

### 1. Safe-Set Guarantee

The safe-set subtraction step ensures that target and anchor objects are never blurred, regardless of SAM3 segmentation errors. If SAM3 incorrectly identifies the target spoon as a "fork" distractor, the safe-set mask will protect it.

### 2. Table Color Sampling

CGVD detects the table/surface that objects rest on using SAM3, then samples its actual color. This ensures:
- Distractors blend to the real table color (brown), not a gray average
- Only 1 extra SAM3 call per episode (cached after first frame)
- Fallback to sampling non-distractor pixels if table detection fails

### 3. Robot Preservation

The robot arm and gripper are always included in the safe-set to maintain proprioception alignment. Blurring the robot would cause a mismatch between the sharp robot in training data and blurred robot at inference.

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
                              │  Blur + Darken      │
                              │  masked regions     │
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
- `src/cgvd/spectral_abstraction.py` - Gaussian blur and blending operations
- `src/cgvd/sam3_segmenter.py` - SAM3 interface for concept-based segmentation
- `src/cgvd/instruction_parser.py` - Natural language instruction parsing

## Usage Example

```bash
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --distractors ycb_030_fork:0.55 ycb_037_scissors:0.50 \
    --use_cgvd \
    --cgvd_blur_sigma 30.0 \       # Lower sigma now OK (color from SAM3)
    --cgvd_darken_strength 0.8 \   # Blend toward table color
    --cgvd_update_freq 1
```
