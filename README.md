# open-pi-zero

A re-implementation of the [Pi0](https://www.physicalintelligence.company/download/pi0.pdf) Vision-Language-Action (VLA) model from Physical Intelligence (Pi).

This project implements a **Mixture-of-Experts (MoE)-like architecture** using a pre-trained 3B PaliGemma VLM (2.291B fine-tuned) with a new 0.315B action expert. The model is trained using **flow matching** for continuous action prediction.

<img src="media/open-pi-zero-overview.png" alt="open-pi-zero-overview" width="70%"/>

> If you find a bug or think I may have misunderstood part of the architecture based on the paper, please raise an issue or email me.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
  - [Pipeline](#pipeline)
  - [Block-wise Causal Attention](#block-wise-causal-attention)
  - [Flow Matching](#flow-matching)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
  - [Data Preparation](#data-preparation)
  - [Training Commands](#training-commands)
  - [Training Configuration](#training-configuration)
- [Evaluation](#evaluation)
- [Pre-trained Checkpoints](#pre-trained-checkpoints)
- [Benchmark Results](#benchmark-results)
- [Project Structure](#project-structure)
- [Technical Notes](#technical-notes)
- [Acknowledgements](#acknowledgements)

---

## Architecture Overview

### Pipeline

The model follows a three-stage pipeline that processes visual, language, and proprioceptive inputs to predict robot actions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT STAGE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │    Image     │    │    Text      │    │  Proprio     │                  │
│   │  (224x224)   │    │  Instruction │    │  (7D pose)   │                  │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│          │                   │                   │                          │
│          ▼                   ▼                   ▼                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │   SigLIP     │    │  Tokenizer   │    │   Linear     │                  │
│   │   Encoder    │    │  + Embed     │    │   Encoder    │                  │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│          │                   │                   │                          │
│          ▼                   │                   │                          │
│   ┌──────────────┐           │                   │                          │
│   │  Projector   │           │                   │                          │
│   │ (1152→2048)  │           │                   │                          │
│   └──────┬───────┘           │                   │                          │
│          │                   │                   │                          │
│          ▼                   ▼                   ▼                          │
│      256 tokens         ≤20 tokens           1 token                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         JOINT TRANSFORMER (18 layers)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Each layer processes three expert mixtures with block-wise attention:     │
│                                                                              │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐           │
│   │   VLM Expert    │   │ Proprio Expert  │   │  Action Expert  │           │
│   │   (2048 dim)    │   │   (1024 dim)    │   │   (1024 dim)    │           │
│   │                 │   │                 │   │                 │           │
│   │ Attends to:     │   │ Attends to:     │   │ Attends to:     │           │
│   │ • Self only     │   │ • Self          │   │ • Self          │           │
│   │                 │   │ • VLM           │   │ • Proprio       │           │
│   │                 │   │                 │   │ • VLM           │           │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘           │
│         │                       │                     │                     │
│         └───────────────────────┴─────────────────────┘                     │
│                                 │                                           │
│                    (Proprio & Action share weights)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT STAGE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────────────────────────────────────────────┐              │
│   │                   Action Decoder                          │              │
│   │              (Linear: 1024 → 7D action)                   │              │
│   └──────────────────────────┬───────────────────────────────┘              │
│                              │                                              │
│                              ▼                                              │
│   ┌──────────────────────────────────────────────────────────┐              │
│   │               Flow Matching ODE Solver                    │              │
│   │                   (10 Euler steps)                        │              │
│   └──────────────────────────┬───────────────────────────────┘              │
│                              │                                              │
│                              ▼                                              │
│                   Action Chunk (4 steps × 7D)                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Block-wise Causal Attention

The model uses **block-wise causal masking** where each expert attends to specific blocks:

```
                    VLM tokens    Proprio    Action tokens
                   (256 + text)   (1 tok)      (4 tok)
                  ┌───────────┬─────────┬───────────────┐
    VLM tokens    │    ✓      │    ✗    │      ✗        │
                  ├───────────┼─────────┼───────────────┤
    Proprio       │    ✓      │    ✓    │      ✗        │
                  ├───────────┼─────────┼───────────────┤
    Action tokens │    ✓      │    ✓    │      ✓        │
                  └───────────┴─────────┴───────────────┘
                           (✓ = can attend, ✗ = masked)
```

**Key properties:**
- VLM block is **bidirectional** within itself (image + text attend to each other)
- Proprio attends to VLM and itself
- Action attends to everything (VLM, proprio, and other action tokens)
- Proprio and action experts **share transformer weights** but use different input embeddings

### Flow Matching

Instead of discrete action tokens, Pi0 uses **flow matching** (continuous diffusion) for action prediction:

```
Training:
─────────
1. Sample timestep t ~ Beta(1.5, 1)     # Higher density at earlier timesteps
2. Sample noise x₀ ~ N(0, I)
3. Interpolate: ψₜ = (1 - t)·x₀ + t·x₁   # x₁ = ground truth action
4. Predict velocity: v = model(ψₜ, t)
5. Loss = MSE(v, x₁ - x₀)               # Target is direction to data

Inference:
──────────
1. Start with noise x₀ ~ N(0, I)
2. For t in [0, 0.1, 0.2, ..., 1.0]:    # 10 Euler steps
     v = model(xₜ, t)
     xₜ₊₁ = xₜ + Δt · v
3. Return x₁ as predicted action
```

**Why flow matching?**
- Continuous output space (no discretization artifacts)
- Stable training with simple MSE loss
- Efficient inference with few ODE steps
- Natural multimodal action distributions

---

## Installation

### 1. Clone Repositories

```bash
# Clone this repo
git clone https://github.com/allenzren/open-pi-zero
cd open-pi-zero

# For evaluation, clone the SimplerEnv fork (adds proprio support)
git clone https://github.com/allenzren/SimplerEnv --recurse-submodules
```

### 2. Install Dependencies

**Using uv (recommended):**
```bash
# Install uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync

# If using SimplerEnv for evaluation
uv pip install -e ../SimplerEnv
uv pip install -e ../SimplerEnv/ManiSkill2_real2sim
```

**Using pip/conda:**
```bash
pip install -e .
pip install -e ../SimplerEnv
pip install -e ../SimplerEnv/ManiSkill2_real2sim
```

### 3. Set Environment Variables

```bash
source scripts/set_path.sh
```

This sets:
- `VLA_DATA_DIR` - Training data directory
- `VLA_LOG_DIR` - Logging and checkpoint directory
- `VLA_WANDB_ENTITY` - Weights & Biases entity
- `TRANSFORMERS_CACHE` - PaliGemma weights location

### 4. Download PaliGemma Weights

```bash
cd $TRANSFORMERS_CACHE
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

### 5. Verify Installation

```bash
# Test PaliGemma text generation
uv run src/model/vla/pizero.py --text_only --load_pretrained_weights --use_bf16
```

---

## Quick Start

### Try Pre-trained Checkpoints

Download a checkpoint and run inference in SimplerEnv:

```bash
# Download checkpoint (see Pre-trained Checkpoints section)
# Then run:
uv run scripts/try_checkpoint_in_simpler.py \
    --task google_robot_pick_horizontal_coke_can \
    --checkpoint_path /path/to/fractal_beta.pt \
    --recording \
    --use_bf16 \
    --use_torch_compile  # First batch will be slow due to compilation
```

**Available tasks:**
- **Bridge (WidowX):** `widowx_carrot_on_plate`, `widowx_spoon_on_towel`, `widowx_stack_cube`, `widowx_put_eggplant_in_basket`
- **Fractal (Google Robot):** `google_robot_pick_horizontal_coke_can`, `google_robot_move_near`, `google_robot_close_drawer`, `google_robot_open_drawer`, `google_robot_place_apple_in_closed_top_drawer`

---

## Training

### Data Preparation

#### Download Datasets

**Bridge Dataset:**
```bash
# Download from RAIL (at $VLA_DATA_DIR)
# See: https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/
```

**Fractal Dataset:**
```bash
# Download from Google Cloud (following OXE)
uv run gsutil -m cp -r gs://gresearch/robotics/fractal20220817_data/0.1.0 $VLA_DATA_DIR/
```

#### Preprocess Data

Resize images to 224x224 for PaliGemma:

```bash
# Using the provided script
uv run python scripts/data/modify_rlds_dataset.py \
    --dataset=bridge_dataset \
    --data_dir=$VLA_DATA_DIR \
    --target_dir=$VLA_DATA_DIR/resize_224 \
    --mods=resize_and_jpeg_encode \
    --n_workers=40 \
    --max_episodes_in_memory=200
```

Or use the SLURM script: `slurm/modify_rlds.sh`

### Training Commands

**Single GPU:**
```bash
uv run scripts/run.py --config-name=bridge
```

**Multi-GPU (8 GPUs on 1 node):**
```bash
HYDRA_FULL_ERROR=1 uv run torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --standalone \
    scripts/run.py \
    --config-name=bridge
```

**Multi-Node:**
```bash
# See slurm/train_multi_node.sh for full example
```

### Training Configuration

Key parameters in `config/train/bridge.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 15 | Number of training epochs |
| `global_batch_size` | 1024 | Total batch size across all GPUs |
| `per_device_batch_size` | 16 | Batch size per GPU |
| `action_lr` | 5e-5 | Learning rate for action expert |
| `vlm_lr` | 5e-5 | Learning rate for VLM |
| `flow_sampling` | beta | Time sampling: `uniform` or `beta` |
| `horizon_steps` | 4 | Action chunk size |
| `num_inference_steps` | 10 | ODE solver steps during eval |
| `use_torch_compile` | True | Enable torch.compile |
| `use_bf16` | True | Use bfloat16 precision |

**Memory optimization options:**
```yaml
quantize: True      # 4-bit quantization
lora: True          # LoRA fine-tuning
lora_r: 32          # LoRA rank
```

### Training Tips

1. **RAM Usage:** TFDS dataloading requires ~300-400GB system RAM
2. **VRAM Usage:** ~40GB peak with batch size 16 on single GPU
3. **Training Time:** ~1.5-2 days on L40, 8-12 hours on H100s
4. **Normalization:** Actions normalized to [-1, 1] using p01/p99 percentiles

---

## Evaluation

### SimplerEnv Evaluation

Run evaluation on SimplerEnv tasks:

```bash
uv run scripts/run.py \
    --config-name=bridge \
    --config-path=config/eval \
    env.task=widowx_spoon_on_towel \
    checkpoint_path=/path/to/checkpoint.pt \
    n_eval_episode=240
```

**Important:** The provided checkpoints use different RoPE settings than the current training config:
- Provided checkpoints: `time_max_period=10000`, `action_expert_rope_theta=10000`
- New training config: `time_max_period=100`, `action_expert_rope_theta=100`

Update the eval config accordingly when evaluating newly trained checkpoints.

### Quick Evaluation Scripts

We provide convenience scripts for running evaluations:

**Single episode:**
```bash
./run_bridge_eval.sh
```

**Multiple episodes (10 runs with different seeds):**
```bash
./run_bridge_eval_multi.sh
```

This runs 10 episodes and reports the success rate. Results are logged to `/tmp/episode_*.log`.

### Inference Speed

| Setup | Time | Peak VRAM |
|:------|:----:|:---------:|
| float32 | 237ms | 13.6GB |
| bf16 | 245ms | 6.7GB |
| float32 + torch.compile | 89ms | 13.6GB |
| bf16 + torch.compile | **75ms** | **6.7GB** |
| Pi0 paper | 73ms* | - |

*Pi0 paper uses 3 images and chunk size 50; this implementation uses 1 image and chunk size 4.

---

## Pre-trained Checkpoints

| Model | Dataset | Sampling | Download |
|-------|---------|----------|----------|
| Bridge-Uniform | Bridge | Uniform | [HuggingFace](https://huggingface.co/allenzren/open-pi-zero/blob/main/bridge_uniform_step19296_2024-12-26_22-31_42.pt) |
| Bridge-Beta | Bridge | Beta | [HuggingFace](https://huggingface.co/allenzren/open-pi-zero/blob/main/bridge_beta_step19296_2024-12-26_22-30_42.pt) |
| Fractal-Uniform | Fractal | Uniform | [HuggingFace](https://huggingface.co/allenzren/open-pi-zero/blob/main/fractal_uniform_step29576_2024-12-31_22-26_42.pt) |
| Fractal-Beta | Fractal | Beta | [HuggingFace](https://huggingface.co/allenzren/open-pi-zero/blob/main/fractal_beta_step29576_2024-12-29_13-10_42.pt) |

**Sampling strategy:**
- **Uniform:** Sample flow matching timesteps uniformly in [0, 1]
- **Beta:** Sample from Beta(1.5, 1) distribution, giving higher density at earlier timesteps

---

## Benchmark Results

Success rates in SimplerEnv **visual matching** setting (averaged over 240-2400 trials per task):

### Bridge Tasks (WidowX Robot)

| Policy | Dtype | Carrot on plate | Eggplant in basket | Spoon on towel | Stack cube |
|:------:|:-----:|:---------------:|:------------------:|:--------------:|:----------:|
| Bridge-Uniform | float32 | 58.8% | 79.2% | 63.3% | 21.3% |
| Bridge-Uniform | bf16 | 58.8% | 81.3% | 61.7% | 23.8% |
| Bridge-Beta | float32 | 55.8% | 85.4% | 84.6% | 47.9% |
| Bridge-Beta | bf16 | 52.5% | 87.9% | 83.8% | **52.5%** |

### Fractal Tasks (Google Robot)

| Policy | Dtype | Pick Coke | Move Near | Close Drawer | Open Drawer | Put Apple |
|:------:|:-----:|:---------:|:---------:|:------------:|:-----------:|:---------:|
| Fractal-Uniform | float32 | 88.0% | 80.3% | 66.7% | 45.2% | 52.2% |
| Fractal-Uniform | bf16 | 88.9% | 80.5% | 65.4% | 45.3% | 53.0% |
| Fractal-Beta | float32 | **97.9%** | 78.7% | **75.0%** | 49.5% | 46.6% |
| Fractal-Beta | bf16 | 97.8% | 78.4% | 74.7% | **51.7%** | 46.1% |

**Notes:**
- Bridge policies: Execute all 4 action chunk steps
- Fractal policies: Execute 2 out of 4 steps (dataset is 3Hz vs 5Hz for Bridge)
- bf16 inference causes slight distribution shift due to KV cache precision

> Disclaimer: Please do not associate these results with possible results from Pi.

---

## Project Structure

```
open-pi-zero/
├── config/
│   ├── train/
│   │   ├── bridge.yaml          # Bridge dataset training
│   │   └── fractal.yaml         # Fractal dataset training
│   ├── eval/
│   │   └── *.yaml               # Evaluation configs
│   ├── bridge_statistics.json   # Normalization statistics
│   └── fractal_statistics.json
│
├── scripts/
│   ├── run.py                   # Main entry point
│   ├── set_path.sh              # Environment setup
│   ├── try_checkpoint_in_simpler.py  # Quick inference demo
│   └── data/                    # Data processing scripts
│
├── slurm/                       # SLURM job scripts
│
├── src/
│   ├── agent/
│   │   ├── train.py             # Training loop
│   │   ├── eval.py              # Evaluation loop
│   │   └── env_adapter/         # Environment adapters
│   │       ├── base.py          # Base normalization
│   │       └── simpler.py       # SimplerEnv adapters
│   │
│   ├── model/
│   │   ├── vla/
│   │   │   ├── pizero.py        # Main PiZero model
│   │   │   ├── joint_model.py   # Joint transformer
│   │   │   ├── mixture.py       # Expert mixture
│   │   │   ├── modules.py       # Time embedding, encoders
│   │   │   └── processing.py    # Tokenization
│   │   │
│   │   └── paligemma/
│   │       ├── siglip.py        # Vision encoder
│   │       └── modules.py       # Gemma components
│   │
│   └── data/                    # Dataset loading (Octo/DLIMP style)
│
└── doc/
    ├── notes.md                 # Training observations
    ├── error.md                 # Common errors
    └── convention.md            # Data conventions
```

---

## Technical Notes

### Observations from Training

- **Beta vs Uniform sampling:** Beta achieves better validation loss and higher-threshold accuracy; Uniform sometimes matches or outperforms at low thresholds (e.g., 0.05)
- **Learning rate:** Stable training up to 3e-4 with batch size 1024
- **Normalization:** [-1, 1] bounds work better than unit Gaussian due to outliers in Bridge data
- **Pre-training:** Fine-tuning PaliGemma is essential; freezing VLM (training action expert only) fails

### Data Conventions

**End-effector pose:**
- Bridge: XYZ position + sxyz Euler angles (relative to top-down pose)
- Fractal: XYZ position + XYZW quaternion

**Gripper:**
- Bridge: 1 = open, -1 = closed (state); 1 = open, -1 = close (action)
- Fractal: -1 = open, 1 = closed (state); 1 = close, -1 = open (action)

### Important Config Notes

When evaluating provided checkpoints, use these settings:
```yaml
time_max_period: 10000.0
action_expert_rope_theta: 10000.0
```

New training configs use:
```yaml
time_max_period: 100.0
action_expert_rope_theta: 100.0
```

---

## Acknowledgements

- **PaliGemma:** [Open-source PaliGemma](https://github.com/hkproj/pytorch-paligemma/tree/main)
- **Datasets:** [Octo](https://octo-models.github.io/) and [dlimp](https://github.com/kvablack/dlimp)
- **Data preprocessing:** [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod/tree/main)
- **References:** [Pi0 Paper](https://www.physicalintelligence.company/download/pi0.pdf), [OpenVLA](https://github.com/openvla/openvla), [Flow Matching](https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb), [lucidrains implementation](https://github.com/lucidrains/pi-zero-pytorch), [SimplerEnv](https://github.com/simpler-env/SimplerEnv), [QLoRA-LLM](https://github.com/michaelnny/QLoRA-LLM)

Special thanks to [Asher Hancock](https://aasherh.github.io/) for discussions on block-wise causal masking.

---

## License

See [LICENSE](LICENSE) for details.
