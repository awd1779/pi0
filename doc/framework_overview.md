# open-pi-zero Framework Overview

This document provides a comprehensive overview of the open-pi-zero codebase for understanding the full pipeline from training to evaluation.

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Training Pipeline](#3-training-pipeline)
4. [Evaluation Pipeline](#4-evaluation-pipeline)
5. [CGVD System](#5-cgvd-system)
6. [Clutter Testing Pipeline](#6-clutter-testing-pipeline)
7. [Key Scripts Reference](#7-key-scripts-reference)
8. [Configuration Notes](#8-configuration-notes)
9. [Common Workflows](#9-common-workflows)

---

## 1. Project Overview

**open-pi-zero** is a re-implementation of the Pi0 Vision-Language-Action (VLA) model from Physical Intelligence.

### Key Architecture
- **VLM Expert**: Pre-trained 3B PaliGemma (2.291B fine-tuned)
- **Action Expert**: New 0.315B trainable parameters
- **Training**: Flow matching for continuous action prediction
- **Innovation**: Concept-Gated Visual Distillation (CGVD) for handling cluttered environments

### Supported Robots
- **Bridge (WidowX)**: `widowx_carrot_on_plate`, `widowx_spoon_on_towel`, `widowx_stack_cube`, `widowx_put_eggplant_in_basket`
- **Fractal (Google Robot)**: `google_robot_pick_horizontal_coke_can`, `google_robot_move_near`, `google_robot_close_drawer`

---

## 2. Directory Structure

```
open-pi-zero/
├── config/
│   ├── train/
│   │   ├── bridge.yaml           # Bridge dataset training config
│   │   └── fractal.yaml          # Fractal dataset training config
│   ├── eval/
│   │   ├── bridge.yaml           # Bridge evaluation config
│   │   └── fractal_*.yaml        # Fractal task configs
│   ├── bridge_statistics.json    # Normalization stats (p01/p99)
│   └── fractal_statistics.json
│
├── scripts/
│   ├── run.py                    # Main Hydra entry point
│   ├── try_checkpoint_in_simpler.py  # Quick inference with options
│   ├── set_path.sh               # Environment setup
│   │
│   ├── setup_clutter_assets.py   # Download & convert YCB/RoboCasa
│   ├── list_available_objects.py # List available distractor objects
│   │
│   ├── clutter_eval/             # Pre-configured evaluation scripts
│   │   ├── run_clutter_2.sh
│   │   ├── run_clutter_4.sh
│   │   ├── run_clutter_6.sh      # With CGVD
│   │   ├── run_clutter_2_baseline.sh
│   │   ├── run_clutter_4_baseline.sh
│   │   └── run_clutter_6_baseline.sh  # Without CGVD
│   │
│   └── data/
│       └── modify_rlds_dataset.py  # Preprocess datasets
│
├── src/
│   ├── agent/
│   │   ├── train.py              # TrainAgent class
│   │   ├── eval.py               # EvalAgent class
│   │   ├── dataset.py            # TorchRLDSInterleavedDataset
│   │   └── env_adapter/
│   │       ├── base.py           # BaseEnvAdapter (normalization)
│   │       └── simpler.py        # SimplerAdapter, BridgeSimplerAdapter
│   │
│   ├── model/
│   │   ├── vla/
│   │   │   ├── pizero.py         # Main PiZero & PiZeroInference
│   │   │   ├── joint_model.py    # Joint transformer (18 layers)
│   │   │   ├── mixture.py        # Expert mixtures
│   │   │   ├── modules.py        # Time embedding, encoders
│   │   │   └── processing.py     # Tokenization
│   │   └── paligemma/
│   │       └── siglip.py         # Vision encoder
│   │
│   ├── cgvd/
│   │   ├── __init__.py           # Exports CGVDWrapper
│   │   ├── cgvd_wrapper.py       # Main gym.Wrapper
│   │   ├── sam3_segmenter.py     # SAM3 segmentation
│   │   ├── instruction_parser.py # Parse NL instructions
│   │   ├── spectral_abstraction.py  # Gaussian blur
│   │   └── distractor_wrapper.py # Add clutter objects
│   │
│   └── utils/
│       ├── metrics.py
│       ├── optimization.py
│       └── monitor.py
│
├── checkpoints/                  # Model checkpoints
│   └── bridge_beta.pt
│
├── slurm/                        # SLURM job scripts
│
└── doc/
    ├── framework_overview.md     # This file
    ├── clutter_assets.md         # Clutter testing guide
    └── notes.md                  # Training observations
```

---

## 3. Training Pipeline

### Entry Point
```bash
uv run scripts/run.py --config-name=bridge
```

### Data Flow
```
RLDS Datasets (Bridge/Fractal)
    ↓
TorchRLDSInterleavedDataset
    - Interleave multiple sources with weights
    - Apply trajectory transforms
    - Window: 1 frame proprio, 4 frames action
    ↓
Image Augmentation
    - Random resized crop
    - Brightness, contrast, saturation, hue
    ↓
Normalization
    - Actions: p01/p99 → [-1, 1]
    - Proprioception: same
    ↓
Flow Matching Training
    - Sample t ~ Beta(1.5, 1)
    - ψₜ = (1-t)·x₀ + t·x₁
    - Loss = MSE(v_pred, x₁ - x₀)
    ↓
Checkpoints → logs/{dataset}/{timestamp}/
```

### Key Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 15 | Training epochs |
| `global_batch_size` | 1024 | Total batch size |
| `per_device_batch_size` | 16 | Per GPU |
| `action_lr` | 5e-5 | Action expert LR |
| `vlm_lr` | 5e-5 | VLM fine-tuning LR |
| `flow_sampling` | beta | `beta` or `uniform` |
| `horizon_steps` | 4 | Action chunk size |
| `num_inference_steps` | 10 | ODE solver steps |

---

## 4. Evaluation Pipeline

### Quick Inference
```bash
uv run scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --num_episodes 10 \
    --recording \
    --use_bf16
```

### Inference Data Flow
```
SimplerEnv Observation
    ↓
[Optional: CGVD Wrapper]
    - Instruction parsing
    - SAM3 segmentation (1Hz)
    - Gaussian blur background
    ↓
Environment Adapter (preprocess)
    - Resize image to 224×224
    - Tokenize instruction
    - Normalize proprioception
    ↓
PiZeroInference Forward
    - SigLIP encoder → 256 tokens
    - Joint transformer (18 layers)
    - Block-wise causal attention
    ↓
Flow Matching Inference
    - x₀ ~ N(0, I)
    - 10 Euler steps: x_{t+1} = x_t + 0.1 * v_pred
    ↓
Environment Adapter (postprocess)
    - Denormalize action
    - Euler → axis-angle
    ↓
SimplerEnv.step(action)
```

### CLI Arguments
```bash
--task                    # SimplerEnv task name
--checkpoint_path         # Path to .pt file
--num_episodes           # Number of episodes
--recording              # Save video
--output_dir             # Video output directory
--use_bf16               # bfloat16 inference
--use_torch_compile      # Enable torch.compile (3x speedup)
--distractors            # Add clutter objects
--use_cgvd               # Enable visual distillation
--cgvd_blur_sigma        # Blur strength (default: 15.0)
--cgvd_update_freq       # Mask update frequency (default: 1)
```

---

## 5. CGVD System

**Concept-Gated Visual Distillation** - A model-agnostic wrapper that cleans visual observations to prevent feature dilution in cluttered environments.

### Architecture (3 Stages)

#### Stage 1: Interaction-Aware Decomposition
- **File**: `src/cgvd/instruction_parser.py`
- **Input**: Natural language instruction
- **Output**: Target object + Anchor object
- **Example**: "put the spoon on the towel" → target="spoon", anchor="towel"

#### Stage 2: Concept-Driven Grounding
- **File**: `src/cgvd/sam3_segmenter.py`
- **Model**: Facebook SAM3 (Segment Anything 3)
- **Update**: Every N frames (configurable, default 1Hz)
- **Output**: Binary masks for target, anchor, robot arm, gripper
- **Validation**: IoU > 0.4 to filter hallucinations

#### Stage 3: Spectral Visual Abstraction
- **File**: `src/cgvd/spectral_abstraction.py`
- **Method**: Gaussian blur on background
- **Formula**: `I_distilled = M * I_raw + (1-M) * GaussianBlur(I)`
- **Default sigma**: 15.0

### Integration
- **File**: `src/cgvd/cgvd_wrapper.py`
- Implements `gym.Wrapper` interface
- Overrides `step()` and `reset()`
- **CRITICAL**: Always includes robot arm/gripper in mask to prevent proprioception misalignment

### Usage
```bash
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --use_cgvd \
    --cgvd_blur_sigma 15.0 \
    --cgvd_save_debug  # Save debug images
```

---

## 6. Clutter Testing Pipeline

### Overview
Test VLA robustness by adding distractor objects to the scene.

### Asset Sources
| Source | Objects | Download |
|--------|---------|----------|
| **YCB** | Cans, boxes, fruits (~78 objects) | `python -m mani_skill.utils.download_asset ycb -y` |
| **RoboCasa** | Kitchen items + **utensils** (~914 objects) | `python -m mani_skill.utils.download_asset RoboCasa -y` |

### Setup Process
```bash
# 1. Download and convert assets
source ~/miniconda3/bin/activate maniskill
python scripts/setup_clutter_assets.py

# 2. List available objects
python scripts/list_available_objects.py

# 3. Run with distractors
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --distractors rc_fork_11 rc_knife_26 rc_spoon_11 \
    --num_episodes 5 --recording --use_bf16
```

### CRITICAL: SimplerEnv Path Issue

The `uv run` environment uses a **different** SimplerEnv installation:

| Environment | SimplerEnv Path |
|-------------|-----------------|
| `uv run` / `.venv` | `~/allenzren_SimplerEnv/` |
| `conda maniskill` | `~/SimplerEnv-OpenVLA/` |

**Assets must be in the correct location:**
```
~/allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom/
├── models/
│   ├── rc_fork_11/
│   ├── rc_knife_26/
│   ├── ycb_011_banana/
│   └── ...
└── info_bridge_custom_v0.json
```

### Folder Structure
```
~/.maniskill/data/                              # ManiSkill downloads
├── assets/mani_skill2_ycb/                    # YCB source
│   ├── models/
│   │   ├── 002_master_chef_can/
│   │   └── ...
│   └── info_pick_v0.json
└── scene_datasets/robocasa_dataset/           # RoboCasa source
    └── assets/objects/
        └── objaverse/
            ├── fork/fork_0/, fork_1/, ...
            ├── knife/knife_0/, ...
            └── spoon/spoon_0/, ...

~/allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom/  # Converted assets
├── models/
│   ├── rc_fork_11/                            # RoboCasa converted
│   │   ├── collision.obj
│   │   ├── textured.obj
│   │   └── *.png (textures)
│   ├── ycb_011_banana/                        # YCB converted
│   └── bridge_spoon_blue/                     # Built-in
└── info_bridge_custom_v0.json                 # Model database
```

### Available Object IDs

**Built-in (always available):**
- `green_cube_3cm`, `yellow_cube_3cm`
- `eggplant`, `apple`, `orange`, `sponge`
- `bridge_carrot_generated_modified`, `bridge_spoon_generated_modified`
- `bridge_plate_objaverse`

**RoboCasa Utensils (after setup):**
- `rc_fork_11`, `rc_knife_26`, `rc_spoon_11`, `rc_spatula_1`, `rc_ladle_3`

**YCB Utensils (after setup):**
- `ycb_030_fork`, `ycb_031_spoon`, `ycb_032_knife`, `ycb_033_spatula`

**YCB Fruits:**
- `ycb_011_banana`, `ycb_013_apple`, `ycb_017_orange`, `ycb_014_lemon`

### Evaluation Scripts
```bash
# Baseline (no CGVD)
./scripts/clutter_eval/run_clutter_6_baseline.sh 10

# With CGVD
./scripts/clutter_eval/run_clutter_6.sh 10
```

---

## 7. Key Scripts Reference

| Script | Purpose | Key Args |
|--------|---------|----------|
| `scripts/run.py` | Main Hydra launcher | `--config-name=bridge` |
| `scripts/try_checkpoint_in_simpler.py` | Quick inference | `--task`, `--checkpoint_path`, `--use_cgvd`, `--distractors` |
| `scripts/setup_clutter_assets.py` | Download & convert assets | `--skip-download`, `--max-objects` |
| `scripts/list_available_objects.py` | List distractors | `--verbose`, `--json` |
| `scripts/data/modify_rlds_dataset.py` | Preprocess datasets | `--dataset`, `--mods=resize_and_jpeg_encode` |

---

## 8. Configuration Notes

### RoPE Settings (CRITICAL)
**Old checkpoints (provided):**
```yaml
time_max_period: 10000.0
action_expert_rope_theta: 10000.0
```

**New training config:**
```yaml
time_max_period: 100.0
action_expert_rope_theta: 100.0
```

Match the config to your checkpoint or results will be wrong.

### Normalization
- **Method**: p01/p99 percentiles → [-1, 1]
- **Why**: Better than Gaussian due to outliers in Bridge data
- **Files**: `config/bridge_statistics.json`, `config/fractal_statistics.json`

### Action Chunk
- **Size**: 4 steps
- **Bridge**: Execute all 4 steps
- **Fractal**: Execute 2/4 steps (3Hz vs 5Hz)

### Inference Speed
| Setup | Time | VRAM |
|-------|------|------|
| float32 | 237ms | 13.6GB |
| bf16 | 245ms | 6.7GB |
| bf16 + torch.compile | **75ms** | **6.7GB** |

---

## 9. Common Workflows

### Train a New Model
```bash
# 1. Set environment
source scripts/set_path.sh

# 2. Preprocess data (if needed)
uv run python scripts/data/modify_rlds_dataset.py \
    --dataset=bridge_dataset \
    --mods=resize_and_jpeg_encode

# 3. Train
uv run scripts/run.py --config-name=bridge
```

### Evaluate with Clutter
```bash
# 1. Setup assets (one-time)
source ~/miniconda3/bin/activate maniskill
python scripts/setup_clutter_assets.py

# 2. Run evaluation
./scripts/clutter_eval/run_clutter_6_baseline.sh 10   # Baseline
./scripts/clutter_eval/run_clutter_6.sh 10            # With CGVD
```

### Add New Distractor Objects
1. Add model files to `~/allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom/models/{object_id}/`
   - `collision.obj` - Collision mesh
   - `textured.obj` - Visual mesh
   - `*.png` - Textures
2. Add entry to `info_bridge_custom_v0.json`:
   ```json
   "my_object": {
     "bbox": {"min": [-0.05, -0.05, -0.05], "max": [0.05, 0.05, 0.05]},
     "scales": [1.0],
     "density": 1000
   }
   ```
3. Use with `--distractors my_object`

### Debug CGVD
```bash
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --use_cgvd \
    --cgvd_save_debug \
    --cgvd_verbose \
    --num_episodes 1
```
Debug images saved to `cgvd_debug/{task}/`

---

## Quick Reference

### Environment Variables
```bash
export VLA_DATA_DIR=/path/to/data
export VLA_LOG_DIR=/path/to/logs
export TRANSFORMERS_CACHE=/path/to/paligemma
export VLA_WANDB_ENTITY=your_entity
```

### Model Architecture Summary
```
Input:  Image (224×224) + Text (≤20 tokens) + Proprio (7D)
        ↓
Encode: SigLIP (256 tok) + Tokenizer + Linear
        ↓
Joint:  18-layer transformer with block-wise causal masking
        - VLM block: 2048 dim, bidirectional
        - Proprio block: 1024 dim, attends to VLM
        - Action block: 1024 dim, attends to all
        ↓
Output: Flow matching → 4-step action chunk (7D each)
```

### Troubleshooting

**"Object not in model_db"**
→ Run `python scripts/setup_clutter_assets.py` to download/convert assets

**Wrong SimplerEnv path**
→ Check which SimplerEnv is being used: `uv run python -c "import mani_skill2_real2sim; print(mani_skill2_real2sim.__path__)"`

**Poor CGVD segmentation**
→ Check `cgvd_debug/` images, adjust `--cgvd_presence_threshold`

**RoPE mismatch**
→ Use correct `time_max_period` and `action_expert_rope_theta` for your checkpoint
