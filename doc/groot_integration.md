# GR00T N1.6 Integration for SimplerEnv

This document describes the integration of NVIDIA's GR00T N1.6 model into the open-pi-zero evaluation pipeline for comparison with Pi0.

## Overview

GR00T (Generalist Robot 00 Technology) is NVIDIA's foundation model for humanoid and manipulation robots. We integrated GR00T-N1.6-bridge to enable direct comparison with Pi0 on SimplerEnv tasks, with and without CGVD (Concept-Gated Visual Distillation).

### Key Design Decision: Separate Conda Environments

GR00T, Pi0, and SAM3 have conflicting dependencies. We use **three separate conda environments**:

```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│  Environment: .venv (uv)        │    │  Environment: groot (conda)     │
│  - Pi0 model                    │    │  - GR00T N1.6 model             │
│  - transformers 4.45.0          │    │  - transformers 4.53.0          │
│  - try_checkpoint_in_simpler.py │    │  - eval_groot.py                │
└─────────────────────────────────┘    └─────────────────────────────────┘
              │                                      │
              └──────────────┬───────────────────────┘
                             ▼
              ┌─────────────────────────────────────────┐
              │  Shared Components                      │
              │  - SimplerEnv (allenzren_SimplerEnv)    │
              │  - CGVD wrapper (src/cgvd/)             │
              │  - Distractor assets                    │
              └─────────────────────────────────────────┘

┌─────────────────────────────────┐
│  Environment: sam3 (conda)      │
│  - transformers 5.0.1.dev0      │
│  - SAM3 segmentation model      │
│  - scripts/sam3_server.py       │
└─────────────────────────────────┘
```

## Files Created

| File | Purpose |
|------|---------|
| `src/model/vla/groot.py` | GR00T model wrapper |
| `src/agent/env_adapter/groot_simpler.py` | GR00T-specific SimplerEnv adapters |
| `scripts/eval_groot.py` | Main GR00T evaluation script |
| `scripts/sam3_server.py` | Standalone SAM3 segmentation server |
| `run_groot_bridge_eval.sh` | Bridge task evaluation runner |
| `run_groot_fractal_eval.sh` | Fractal task evaluation runner |

## Files Modified

### Eagle Processor Patches

GR00T uses the Eagle-Block2A-2B-v2 vision processor which has compatibility issues with transformers 4.53.0. The following files in `~/.cache/huggingface/modules/transformers_modules/Eagle-Block2A-2B-v2/` were patched:

**`processing_eagle3_vl.py`**
```python
# Added local VideoInput type definition (not available in transformers 4.53.0)
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
VideoInput = Union[List[Image.Image], List[np.ndarray], List[torch.Tensor]]

# Fixed from_args_and_dict compatibility
if isinstance(unused_kwargs, dict):
    # ... handle dict case

# Added image_sizes fallback for SiglipImageProcessor
if 'image_sizes' in image_inputs:
    image_height, image_width = image_inputs['image_sizes'][0]
else:
    pv = image_inputs['pixel_values']
    # ... compute from pixel_values shape
```

**`image_processing_eagle3_vl_fast.py`**
```python
# Disabled - requires transformers >= 5.0.0
raise ImportError(
    "Eagle3_VLImageProcessorFast requires transformers >= 4.57.0. "
    "Falling back to slow image processor."
)
```

**`preprocessor_config.json`**
```json
{
    "image_processor_type": "SiglipImageProcessor"
}
```

### CGVD Server Support

**`src/cgvd/sam3_segmenter.py`**
- Added `SAM3ClientSegmenter` class for file-based IPC with SAM3 server
- Updated `create_segmenter()` factory with `use_server` parameter

**`src/cgvd/cgvd_wrapper.py`**
- Added `use_server_segmenter` parameter to enable server mode

## Environment Setup

### GR00T Environment

```bash
conda create -n groot python=3.10 -y
conda activate groot

# Install Isaac-GR00T
git clone https://github.com/NVIDIA/Isaac-GR00T.git ~/Isaac-GR00T
cd ~/Isaac-GR00T && pip install -e .

# Install SimplerEnv
pip install -e ~/allenzren_SimplerEnv
pip install -e ~/allenzren_SimplerEnv/ManiSkill2_real2sim

# Install open-pi-zero (for CGVD)
pip install -e ~/open-pi-zero
```

### SAM3 Environment (for CGVD with GR00T)

SAM3 requires transformers >= 5.0.0, but GR00T requires 4.53.0. We use a separate environment with a server architecture:

```bash
conda create -n sam3 python=3.10 -y
conda activate sam3

# Install transformers from main branch (has SAM3)
pip install git+https://github.com/huggingface/transformers.git@main
pip install torch torchvision pillow numpy scipy pydantic
```

## SAM3 Server Architecture

The SAM3 server runs in a separate process to avoid transformers version conflicts:

```
┌─────────────────┐     file-based IPC      ┌─────────────────┐
│  GR00T Process  │ ──────────────────────► │  SAM3 Server    │
│  (groot env)    │                         │  (sam3 env)     │
│                 │ ◄────────────────────── │                 │
│  SAM3Client     │     /tmp/sam3_server/   │  Sam3Model      │
└─────────────────┘                         └─────────────────┘
```

Communication files in `/tmp/sam3_server/`:
- `ready` - Server ready signal
- `request.json` - Segmentation request (image path, concepts, threshold)
- `response.npz` - Segmentation mask response

## Usage

### Basic GR00T Evaluation

```bash
./run_groot_bridge_eval.sh widowx_carrot_on_plate 10
```

### With Distractors

```bash
./run_groot_bridge_eval.sh widowx_carrot_on_plate 10 \
    --distractors scripts/clutter_eval/distractors/distractors_carrot.txt
```

### With Distractors + CGVD

**Step 1:** Start SAM3 server in a separate terminal:

```bash
conda activate sam3
python scripts/sam3_server.py
```

**Step 2:** Run GR00T with CGVD:

```bash
./run_groot_bridge_eval.sh widowx_carrot_on_plate 10 \
    --distractors scripts/clutter_eval/distractors/distractors_carrot.txt \
    --cgvd
```

## Available Tasks

Bridge/WidowX tasks:
- `widowx_spoon_on_towel`
- `widowx_carrot_on_plate`
- `widowx_put_eggplant_in_basket`
- `widowx_stack_cube`

## Output Structure

Each run creates a timestamped output directory:

```
videos/
├── groot_baseline/           # No distractors, no CGVD
│   └── widowx_carrot_on_plate_20260131_120000/
├── groot_distractors/        # Distractors, no CGVD
│   └── widowx_carrot_on_plate_20260131_120100/
├── groot_cgvd/               # No distractors, with CGVD
│   └── widowx_carrot_on_plate_20260131_120200/
└── groot_distractors_cgvd/   # Distractors + CGVD
    └── widowx_carrot_on_plate_20260131_120300/
        ├── episode_0.mp4
        ├── episode_1.mp4
        └── cgvd_debug/       # Debug visualizations (if --cgvd_save_debug)
```

## Evaluation Matrix

| Model | Distractors | CGVD | Environment | Command |
|-------|-------------|------|-------------|---------|
| Pi0 | No | No | uv/.venv | `uv run python scripts/try_checkpoint_in_simpler.py ...` |
| Pi0 | Yes | No | uv/.venv | `uv run python scripts/try_checkpoint_in_simpler.py --distractors ...` |
| Pi0 | Yes | Yes | uv/.venv | `uv run python scripts/try_checkpoint_in_simpler.py --distractors ... --use_cgvd` |
| GR00T | No | No | groot | `./run_groot_bridge_eval.sh TASK N` |
| GR00T | Yes | No | groot | `./run_groot_bridge_eval.sh TASK N --distractors FILE` |
| GR00T | Yes | Yes | groot + sam3 | `./run_groot_bridge_eval.sh TASK N --distractors FILE --cgvd` |

## GR00T Model Details

**Model:** `nvidia/GR00T-N1.6-bridge`
- Fine-tuned on Bridge dataset
- Embodiment tag: `new_embodiment_widowxv2`

**Input/Output:**
- Images: 224x224 RGB uint8
- State: 7D proprioception (xyz + euler + gripper)
- Actions: 16-step horizon, normalized to [-1, 1]

## Troubleshooting

### Eagle Processor Errors

If you see errors about `VideoInput` or `image_sizes`, the Eagle processor patches may have been overwritten. Re-apply the patches or clear the cache:

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/Eagle-Block2A-2B-v2
# Model will re-download and need patching again
```

### SAM3 Server Not Found

If you see "SAM3 server not running", ensure:
1. The sam3 conda environment is activated
2. `python scripts/sam3_server.py` is running
3. `/tmp/sam3_server/ready` file exists

### Distractor Loading Failures

Distractors are loaded from SimplerEnv assets at:
```
~/allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom/models/
```

Ensure SimplerEnv is properly installed with assets.

### CUDA Out of Memory

GR00T requires ~8GB VRAM. If you run out of memory:
- Use `--use_bf16` flag
- Reduce batch size (already 1 by default)
