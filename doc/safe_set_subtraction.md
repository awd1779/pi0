# Safe-Set Subtraction for CGVD Distractor Mode

## Problem

When using CGVD in distractor-only mode, SAM3 (Segment Anything Model 3) can sometimes confuse visually similar objects. For example:

- **Task**: "put the spoon on the towel"
- **Target**: spoon
- **Distractors**: fork, knife, spatula

If SAM3 confuses the spoon with the spatula (both are elongated metallic utensils), it may include the spoon in the "spatula" segmentation mask. This causes the spoon to be blurred, making the task impossible for the VLA to complete.

## Solution: Safe-Set Subtraction

We explicitly query SAM3 for both:
1. **Distractor objects** (fork, knife, spatula)
2. **Safe-set objects** (spoon, towel, robot arm, robot gripper)

Then we compute the final blur mask using set subtraction:

```
final_mask = distractor_mask AND (NOT safe_mask)
```

This **guarantees** that the target object is never blurred, even if SAM3 incorrectly includes it in a distractor mask.

## Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Image                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│  Query SAM3 for   │                     │  Query SAM3 for   │
│   DISTRACTORS     │                     │    SAFE SET       │
│ (fork, knife,     │                     │ (spoon, towel,    │
│  spatula)         │                     │  robot arm, etc.) │
└───────────────────┘                     └───────────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│ distractor_mask   │                     │    safe_mask      │
│                   │                     │                   │
│  [may include     │                     │  [definitely      │
│   spoon by        │                     │   includes        │
│   mistake!]       │                     │   spoon]          │
└───────────────────┘                     └───────────────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  final_mask =          │
                 │  distractor AND        │
                 │  (NOT safe)            │
                 │                        │
                 │  [spoon REMOVED from   │
                 │   blur region]         │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Apply Gaussian blur   │
                 │  to final_mask regions │
                 └────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Distilled Image      │
                 │   (spoon is SHARP,     │
                 │    distractors blurred)│
                 └────────────────────────┘
```

## Implementation Details

### Safe Set Composition

The safe set always includes:
- **Target object**: The object being manipulated (parsed from instruction)
- **Anchor object**: The destination/reference object (parsed from instruction)
- **Robot arm**: Always included to prevent proprioception issues
- **Robot gripper**: Always included to prevent proprioception issues

### Debug Visualization

When `--cgvd_save_debug` is enabled, the debug images show 5 columns:

| Column | Name | Description |
|--------|------|-------------|
| 1 | Original | Raw camera input |
| 2 | Distractors | SAM3 mask for distractor objects (may include target) |
| 3 | Safe Set | SAM3 mask for target + anchor + robot |
| 4 | Final (D-S) | Result after subtracting safe set from distractors |
| 5 | Distilled | Final blurred image sent to VLA |

### Console Output

With `--cgvd_verbose`, you'll see coverage percentages:

```
[CGVD] Instruction: 'put the spoon on the towel' -> target='spoon', anchor='towel'
[CGVD] Distractor: 4.1%, Safe: 21.8%, Final: 2.6%
```

- **Distractor**: Coverage before subtraction (may include wrongly segmented target)
- **Safe**: Coverage of protected regions
- **Final**: Coverage after subtraction (should be smaller than Distractor)

## Usage

```bash
xvfb-run -a uv run python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --num_episodes 5 \
    --distractors rc_fork_11 rc_knife_26 rc_spatula_1 \
    --use_cgvd \
    --cgvd_distractor_names fork knife spatula \
    --cgvd_save_debug \
    --cgvd_verbose \
    --recording \
    --use_bf16
```

## Success Criteria

1. Debug images show target object in Safe Set column (green label)
2. Debug images show target object REMOVED from Final (D-S) column
3. Video recording shows target object is SHARP (never blurred)
4. Task success rate maintained or improved compared to non-protected mode

## Trade-offs

**Pros:**
- Guarantees target object is never blurred
- Robust to SAM3 confusion between similar objects
- Minimal computational overhead (one additional SAM3 query)

**Cons:**
- Requires instruction parsing to work correctly
- If instruction parsing fails, safe set may be incomplete
- Two SAM3 queries per frame instead of one (when mask updates)

## Troubleshooting

### YCB Objects Falling Through Table

If YCB objects (e.g., `ycb_032_knife`, `ycb_030_fork`) fall through the table, the collision files may be corrupted. The symptom is:

```
[SAPIEN] [error] OBJ: Invalid face indice
[Distractor] WARNING: distractor_ycb_032_knife fell off table (z=-11.312)
```

**Cause**: YCB collision files are symlinks to PLY format instead of actual OBJ files.

**Fix**: Convert PLY to OBJ:

```bash
cd /home/ubuntu/allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom/models

for dir in ycb_030_fork ycb_032_knife; do
    if [ -L "$dir/collision.obj" ]; then
        echo "Fixing $dir..."
        rm "$dir/collision.obj"
        python3 -c "
import trimesh
mesh = trimesh.load('$dir/collision.ply')
mesh.export('$dir/collision.obj')
print(f'  Converted: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces')
"
    fi
done
```

**Alternative**: Use RC objects instead (e.g., `rc_fork_11`, `rc_knife_26`) which have correct collision files.

### Per-Object Scale

Utensils can be too large at default scale. Use per-object scaling:

```bash
DISTRACTORS="rc_fork_11:0.5 rc_knife_26:0.5 rc_spatula_1:0.5"
```

Format: `object_id:scale` where scale is a multiplier (0.5 = half size).

## Related Files

- `src/cgvd/cgvd_wrapper.py`: Main implementation
- `src/cgvd/instruction_parser.py`: Parses instructions to extract target/anchor
- `src/cgvd/sam3_segmenter.py`: SAM3 segmentation interface
- `src/cgvd/distractor_wrapper.py`: Distractor object spawning and positioning
