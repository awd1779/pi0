# Clutter Assets for VLA Testing

This guide explains how to set up and use diverse object assets for testing VLA clutter resistance in SimplerEnv.

## Overview

Two asset sources are available:
- **YCB Dataset**: Standard robotics benchmark objects (cans, boxes, fruits) - ~78 objects
- **RoboCasa Dataset**: Kitchen objects including **utensils** (fork, knife, spoon, spatula) - ~914 objects

## Quick Start

```bash
# 1. Download and convert assets (one-time setup)
source ~/miniconda3/bin/activate maniskill
python scripts/setup_clutter_assets.py

# 2. List available objects
python scripts/list_available_objects.py

# 3. Run evaluation with distractors
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --distractors rc_fork_11 rc_knife_26 rc_spoon_11 \
    --num_episodes 5 --recording --use_bf16
```

## CRITICAL: SimplerEnv Path

The `uv run` environment uses **`~/allenzren_SimplerEnv/`** (NOT `~/SimplerEnv-OpenVLA/`).

Assets must be in the correct location:
```
~/allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom/
├── models/           # Converted model files
└── info_bridge_custom_v0.json  # Model database
```

## Folder Structure

```
~/.maniskill/data/                              # Downloaded assets
├── assets/mani_skill2_ycb/                    # YCB source
└── scene_datasets/robocasa_dataset/           # RoboCasa source
    └── assets/objects/objaverse/              # Fork, knife, spoon, etc.

~/allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom/
├── models/
│   ├── rc_fork_11/                            # RoboCasa converted
│   │   ├── collision.obj
│   │   ├── textured.obj
│   │   └── *.png
│   ├── ycb_011_banana/                        # YCB converted
│   ├── bridge_spoon_blue/                     # Built-in
│   └── ...
└── info_bridge_custom_v0.json
```

## Available Objects

### Built-in (Always Available)
- `green_cube_3cm`, `yellow_cube_3cm`
- `eggplant`, `apple`, `orange`, `sponge`
- `bridge_carrot_generated_modified`, `bridge_spoon_generated_modified`
- `bridge_plate_objaverse`

### RoboCasa (After Setup)

| Category | Object IDs |
|----------|-----------|
| **Utensils** | `rc_fork_11`, `rc_knife_26`, `rc_spoon_11`, `rc_spatula_1`, `rc_ladle_3` |
| Fruits | `rc_apple_7`, `rc_banana_20`, `rc_orange_1`, `rc_lemon_4`, `rc_lime_0` |
| Vegetables | `rc_carrot_7`, `rc_tomato_6`, `rc_potato_5` |
| Containers | `rc_bowl_1`, `rc_cup_2`, `rc_mug_15`, `rc_plate_1` |

### YCB (After Setup)

| Category | Object IDs |
|----------|-----------|
| Utensils | `ycb_030_fork`, `ycb_031_spoon`, `ycb_032_knife`, `ycb_033_spatula` |
| Fruits | `ycb_011_banana`, `ycb_013_apple`, `ycb_014_lemon`, `ycb_017_orange` |
| Cans | `ycb_002_master_chef_can`, `ycb_005_tomato_soup_can` |
| Containers | `ycb_024_bowl`, `ycb_025_mug`, `ycb_029_plate` |

Run `python scripts/list_available_objects.py` for the full list.

## Usage Examples

### Basic Clutter (2 objects)
```bash
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --distractors green_cube_3cm yellow_cube_3cm \
    --num_episodes 10 --recording --use_bf16
```

### Kitchen Utensils (4 objects)
```bash
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --distractors rc_fork_11 rc_knife_26 ycb_011_banana eggplant \
    --num_episodes 10 --recording --use_bf16
```

### Heavy Clutter (6 objects)
```bash
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --distractors rc_fork_11 rc_knife_26 rc_spoon_11 rc_spatula_1 green_cube_3cm eggplant \
    --num_episodes 10 --recording --use_bf16
```

### With CGVD Enabled
```bash
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --distractors rc_fork_11 rc_knife_26 rc_spoon_11 rc_spatula_1 ycb_011_banana eggplant \
    --use_cgvd \
    --cgvd_blur_sigma 15.0 \
    --num_episodes 10 --recording --use_bf16
```

### Using Evaluation Scripts
```bash
# Baseline (no CGVD)
./scripts/clutter_eval/run_clutter_6_baseline.sh 10

# With CGVD
./scripts/clutter_eval/run_clutter_6.sh 10
```

## Troubleshooting

### "Object not in model_db" Error

```
[Distractor] Warning: 'rc_fork_11' not in model_db, skipping
```

**Cause**: Assets not downloaded/converted, or in wrong location.

**Solution**:
```bash
# 1. Run setup
source ~/miniconda3/bin/activate maniskill
python scripts/setup_clutter_assets.py

# 2. Check which SimplerEnv is being used
uv run python -c "import mani_skill2_real2sim; print(mani_skill2_real2sim.__path__)"

# 3. Copy assets to correct location if needed
cp -r ~/SimplerEnv-OpenVLA/ManiSkill2_real2sim/data/custom/models/* \
      ~/allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom/models/
cp ~/SimplerEnv-OpenVLA/ManiSkill2_real2sim/data/custom/*.json \
   ~/allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom/
```

### Check Available Objects

```bash
# List all available objects
python scripts/list_available_objects.py

# Verbose mode (shows source for each object)
python scripts/list_available_objects.py --verbose

# Output as JSON
python scripts/list_available_objects.py --json
```

### Re-download Assets

```bash
rm -rf ~/.maniskill/data/assets/mani_skill2_ycb
rm -rf ~/.maniskill/data/scene_datasets/robocasa_dataset
python scripts/setup_clutter_assets.py
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/setup_clutter_assets.py` | Download and convert YCB/RoboCasa assets |
| `scripts/list_available_objects.py` | List all available distractor objects |
| `scripts/try_checkpoint_in_simpler.py` | Run evaluation with `--distractors` flag |
| `scripts/clutter_eval/run_clutter_*.sh` | Pre-configured evaluation scripts |

## Adding Custom Objects

1. Create model folder:
   ```
   ~/allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom/models/my_object/
   ├── collision.obj    # Collision mesh
   ├── textured.obj     # Visual mesh
   └── texture.png      # Texture (optional)
   ```

2. Add to `info_bridge_custom_v0.json`:
   ```json
   "my_object": {
     "bbox": {
       "min": [-0.05, -0.05, -0.05],
       "max": [0.05, 0.05, 0.05]
     },
     "scales": [1.0],
     "density": 1000
   }
   ```

3. Use with `--distractors my_object`
