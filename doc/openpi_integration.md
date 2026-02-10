# OpenPI Integration with SimplerEnv

This document describes the integration of Physical Intelligence's OpenPI Pi0 models with SimplerEnv for evaluation.

## Overview

OpenPI provides open-source implementations of Pi0 vision-language-action models. We integrated these models to evaluate CGVD effectiveness on cross-embodiment models.

## Setup

### Prerequisites

- Conda installed
- SimplerEnv at `/home/ubuntu/allenzren_SimplerEnv`
- ManiSkill2 at `/home/ubuntu/allenzren_SimplerEnv/ManiSkill2_real2sim`

### Installation

Run the setup script:

```bash
./scripts/setup_openpi_env.sh
```

This will:
1. Create `openpi` conda environment with Python 3.11 (required by OpenPI)
2. Clone OpenPI repo to `/home/ubuntu/openpi`
3. Install OpenPI and dependencies
4. Add `open-pi-zero` to PYTHONPATH (not pip installed due to Python version conflict)

### Manual Installation

```bash
conda create -n openpi python=3.11 -y
conda activate openpi

# Clone OpenPI
cd /home/ubuntu
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi

# Install OpenPI
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Install SimplerEnv
pip install -e /home/ubuntu/allenzren_SimplerEnv
pip install -e /home/ubuntu/allenzren_SimplerEnv/ManiSkill2_real2sim

# Install additional dependencies
pip install imageio imageio-ffmpeg opencv-python pytest gcsfs
```

## Available Models

OpenPI provides models for **inference** and **fine-tuning**. Only inference-ready models work out of the box:

| Model | Type | Description |
|-------|------|-------------|
| `pi0_fast_droid` | Inference | Cross-embodiment, fast inference (recommended) |
| `pi05_droid` | Inference | Cross-embodiment, better language following |
| `pi0_fast_libero` | Inference | LIBERO simulation |
| `pi05_libero` | Inference | LIBERO simulation |
| `pi0_aloha` | Inference | ALOHA bimanual robot |

**Note:** Base models (`pi0_base`, `pi0_fast_base`) are for fine-tuning only and don't have inference configs.

## Files Created

### Model Wrapper

**`src/model/vla/openpi.py`**

Wraps OpenPI policy for SimplerEnv:
- Loads model from Google Cloud Storage checkpoint
- Provides `forward()` method matching our interface
- Handles input/output format conversion

### Environment Adapters

**`src/agent/env_adapter/openpi_simpler.py`**

Two adapter classes:
- `OpenPIBridgeSimplerAdapter` - For WidowX/Bridge tasks
- `OpenPIFractalSimplerAdapter` - For Google Robot/Fractal tasks

These handle:
- Preprocessing: SimplerEnv obs → OpenPI format (224x224 images, proprioception)
- Postprocessing: OpenPI actions → SimplerEnv format (axis-angle rotation)

### Evaluation Script

**`scripts/eval_openpi.py`**

Full evaluation script supporting:
- All SimplerEnv tasks
- Distractor injection
- CGVD wrapper
- Video recording
- Success rate tracking

### Shell Scripts

**`scripts/setup_openpi_env.sh`** - Environment setup

**`scripts/clutter_eval/run_openpi_comparison.sh`** - Standalone OpenPI evaluation

**`scripts/clutter_eval/run_paired_eval.sh`** - Updated to support `--model openpi`

## Usage

### Run with run_paired_eval.sh

```bash
# OpenPI evaluation
./scripts/clutter_eval/run_paired_eval.sh --model openpi --task widowx_carrot_on_plate --episodes 10

# Compare with Pi0 or GR00T
./scripts/clutter_eval/run_paired_eval.sh --model pi0 --task widowx_carrot_on_plate --episodes 10
./scripts/clutter_eval/run_paired_eval.sh --model groot --task widowx_carrot_on_plate --episodes 10
```

### Run directly

```bash
conda activate openpi
python scripts/eval_openpi.py \
    --task widowx_carrot_on_plate \
    --model pi0_fast_droid \
    --num_episodes 10 \
    --use_cgvd
```

## Key Learnings

### OpenPI Architecture

OpenPI separates:
- **Config**: Defines model architecture, input/output transforms, data format
- **Checkpoint**: Provides trained weights

Configs are task/robot specific (droid, aloha, libero). Base checkpoints exist but have no standalone config - they're meant as starting points for fine-tuning.

### Python Version Conflict

- OpenPI requires Python 3.11+
- open-pi-zero requires Python 3.10

Solution: Add open-pi-zero to PYTHONPATH instead of pip installing:
```bash
export PYTHONPATH="/home/ubuntu/open-pi-zero:$PYTHONPATH"
```

### Checkpoint Download

Checkpoints are downloaded from Google Cloud Storage on first use:
- Location: `gs://openpi-assets/checkpoints/{model_name}`
- Cache: `~/.cache/openpi/`
- Size: Several GB per model

### Cross-Embodiment Limitations

`pi0_fast_droid` is trained on DROID (Franka robot), not WidowX. The model may not perform well on Bridge tasks without fine-tuning. However, it's useful for:
- Testing CGVD on a model that hasn't seen Bridge tasks
- Evaluating cross-embodiment generalization

## Troubleshooting

### Missing modules

```bash
conda run -n openpi pip install pytest gcsfs
```

### Download stuck

Checkpoints are large. Check download progress:
```bash
watch -n 2 'du -sh ~/.cache/openpi/'
```

### Config not found

Only inference-ready models have configs. Use `pi0_fast_droid`, not `pi0_base`.
