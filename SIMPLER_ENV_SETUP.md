# Pi0 Evaluation on SimplerEnv - Setup Guide

This document describes the setup for running Pi0 (open-pi-zero) evaluation on SimplerEnv Bridge tasks.

## Overview

We achieved **80% success rate** on the `widowx_spoon_on_towel` task using the open-pi-zero implementation with the Bridge-Beta checkpoint.

| Model | Success Rate | Notes |
|-------|-------------|-------|
| Octo-base | 10% | Baseline |
| OpenPI + incompatible checkpoint | 0% | Wrong preprocessing |
| **open-pi-zero + Bridge-Beta** | **80%** | Working setup |
| Reported (paper) | 84.6% | With full evaluation protocol |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     open-pi-zero                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ PaliGemma   │  │ Flow        │  │ Action Expert       │ │
│  │ VLM (3B)    │→ │ Matching    │→ │ (0.315B params)     │ │
│  │             │  │ (10 steps)  │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              SimplerEnv (allenzren fork)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ ManiSkill2  │  │ SAPIEN      │  │ WidowX Robot        │ │
│  │ Environment │→ │ Simulator   │→ │ (Bridge setup)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. open-pi-zero
- **Repository**: https://github.com/allenzren/open-pi-zero
- **Location**: `/home/ubuntu/open-pi-zero`
- **Purpose**: Pi0 model implementation with flow matching for action generation

### 2. SimplerEnv (allenzren fork)
- **Repository**: https://github.com/allenzren/SimplerEnv
- **Location**: `/home/ubuntu/allenzren_SimplerEnv`
- **Purpose**: Simulation environment with proper `eef_pos` (end-effector position) support
- **Critical**: Must use this fork, not the main SimplerEnv-OpenVLA repo

### 3. Checkpoints

#### Bridge-Beta Checkpoint
- **Path**: `/home/ubuntu/open-pi-zero/checkpoints/bridge_beta.pt`
- **Source**: https://huggingface.co/allenzren/open-pi-zero/blob/main/bridge_beta_step19296_2024-12-26_22-30_42.pt
- **Size**: 7.2GB
- **Training**: 19k steps on Bridge dataset with Beta flow matching timestep sampling

#### PaliGemma VLM
- **Path**: `/home/ubuntu/.cache/transformers/paligemma-3b-pt-224`
- **Source**: https://huggingface.co/google/paligemma-3b-pt-224 (gated model - requires access request)
- **Purpose**: Vision-Language Model backbone for image/text understanding

## Installation

### Prerequisites
- NVIDIA GPU with 8GB+ VRAM (tested on A10G with 24GB)
- CUDA 12.x
- Python 3.10
- uv package manager

### Step 1: Clone Repositories
```bash
cd /home/ubuntu
git clone https://github.com/allenzren/open-pi-zero
git clone https://github.com/allenzren/SimplerEnv --recurse-submodules allenzren_SimplerEnv
```

### Step 2: Install Dependencies
```bash
cd /home/ubuntu/open-pi-zero
uv sync
uv add /home/ubuntu/allenzren_SimplerEnv --editable
uv add /home/ubuntu/allenzren_SimplerEnv/ManiSkill2_real2sim --editable
```

### Step 3: Download Checkpoints

#### Bridge-Beta checkpoint:
```bash
mkdir -p /home/ubuntu/open-pi-zero/checkpoints
cd /home/ubuntu/open-pi-zero/checkpoints
wget "https://huggingface.co/allenzren/open-pi-zero/resolve/main/bridge_beta_step19296_2024-12-26_22-30_42.pt" -O bridge_beta.pt
```

#### PaliGemma (requires HuggingFace login and access approval):
```bash
# First request access at: https://huggingface.co/google/paligemma-3b-pt-224
huggingface-cli login
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/paligemma-3b-pt-224', local_dir='/home/ubuntu/.cache/transformers/paligemma-3b-pt-224')"
```

### Step 4: Fix TimeLimit Wrapper Issue
Edit `/home/ubuntu/open-pi-zero/scripts/try_checkpoint_in_simpler.py`:
```python
# Replace all occurrences of:
env.get_language_instruction()
# With:
env.unwrapped.get_language_instruction()
```

## Running Evaluation

### Environment Variables
```bash
export TRANSFORMERS_CACHE=/home/ubuntu/.cache/transformers
export VLA_LOG_DIR=/home/ubuntu/open-pi-zero/logs
export VLA_WANDB_ENTITY=none
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

### Single Episode
```bash
cd /home/ubuntu/open-pi-zero
xvfb-run -a -s "-screen 0 1024x768x24" uv run python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path /home/ubuntu/open-pi-zero/checkpoints/bridge_beta.pt \
    --recording \
    --use_bf16
```

### Using the Provided Scripts
```bash
cd /home/ubuntu/open-pi-zero
bash run_bridge_eval.sh        # Single episode
bash run_bridge_eval_multi.sh  # 10 episodes
```

## Available Tasks

Bridge (WidowX) tasks:
- `widowx_spoon_on_towel` - Put spoon on towel (84.6% reported)
- `widowx_carrot_on_plate` - Put carrot on plate (55.8% reported)
- `widowx_put_eggplant_in_basket` - Put eggplant in basket (85.4% reported)
- `widowx_stack_cube` - Stack green cube on yellow (47.9% reported)

Google Robot (Fractal) tasks (requires fractal checkpoint):
- `google_robot_pick_horizontal_coke_can`
- `google_robot_pick_vertical_coke_can`
- `google_robot_move_near_v0`
- `google_robot_open_drawer`
- `google_robot_close_drawer`

## Troubleshooting

### 1. CUDA Out of Memory
Kill any existing processes using GPU:
```bash
pkill -f "serve_policy.py"
nvidia-smi  # Check GPU memory
```

### 2. TimeLimit AttributeError
Use `env.unwrapped.get_language_instruction()` instead of `env.get_language_instruction()`.

### 3. Missing eef_pos KeyError
Must use allenzren's SimplerEnv fork which computes `eef_pos` properly:
```python
# In allenzren's fork, base_agent.py computes:
eef_pos = np.concatenate([pos, quat_wxyz, [gripper_nwidth]])
obs = OrderedDict(qpos=..., qvel=..., eef_pos=eef_pos)
```

### 4. PaliGemma Access Denied
Request access at https://huggingface.co/google/paligemma-3b-pt-224 and login with `huggingface-cli login`.

### 5. Vulkan/Display Errors
Use xvfb-run for headless rendering:
```bash
xvfb-run -a -s "-screen 0 1024x768x24" python ...
```

## Why OpenPI Didn't Work

We initially tried using OpenPI (Physical Intelligence's official inference server) with a checkpoint from HuggingFace (`glory-hyeok/pi0-fast-bridge-simpler`). This resulted in 0% success rate because:

1. **State dimension mismatch**: OpenPI expected 8-dim state, checkpoint was trained with 7-dim
2. **Normalization key mismatch**: Checkpoint used `p01`/`p99`, OpenPI used `q01`/`q99`
3. **Input format mismatch**: Different image key names and preprocessing

The open-pi-zero implementation is the correct one for the official allenzren checkpoints.

## Results

Final evaluation results on `widowx_spoon_on_towel` (10 episodes):

| Episode | Result |
|---------|--------|
| 0 | SUCCESS |
| 1 | SUCCESS |
| 2 | FAILURE |
| 3 | FAILURE |
| 4 | SUCCESS |
| 5 | SUCCESS |
| 6 | SUCCESS |
| 7 | SUCCESS |
| 8 | SUCCESS |
| 9 | SUCCESS |

**Success Rate: 8/10 = 80%** (vs 84.6% reported with full evaluation protocol)

## File Structure

```
/home/ubuntu/
├── open-pi-zero/                    # Pi0 implementation
│   ├── checkpoints/
│   │   └── bridge_beta.pt           # Bridge checkpoint (7.2GB)
│   ├── config/
│   │   ├── eval/bridge.yaml         # Bridge evaluation config
│   │   └── bridge_statistics.json   # Normalization stats
│   ├── scripts/
│   │   └── try_checkpoint_in_simpler.py  # Evaluation script
│   ├── src/
│   │   ├── agent/env_adapter/simpler.py  # SimplerEnv adapter
│   │   ├── model/vla/pizero.py          # Pi0 model
│   │   └── utils/geometry.py            # Quaternion/Euler utils
│   ├── run_bridge_eval.sh           # Single episode script
│   └── run_bridge_eval_multi.sh     # Multi-episode script
│
├── allenzren_SimplerEnv/            # SimplerEnv with eef_pos support
│   ├── simpler_env/
│   └── ManiSkill2_real2sim/
│
└── .cache/transformers/
    └── paligemma-3b-pt-224/         # PaliGemma VLM weights
```

## References

- [open-pi-zero GitHub](https://github.com/allenzren/open-pi-zero)
- [SimplerEnv Paper](https://arxiv.org/abs/2405.05941)
- [Pi0 Paper](https://www.physicalintelligence.company/download/pi0.pdf)
- [Bridge Dataset](https://rail-berkeley.github.io/bridgedata/)
