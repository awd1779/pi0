# Concept-Gated Visual Distillation (CGVD) - RSS 2026

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| CGVDWrapper | **Done** | `src/cgvd/cgvd_wrapper.py` |
| SAM3Segmenter | **Done** | `src/cgvd/sam3_segmenter.py` |
| InstructionParser | **Done** | `src/cgvd/instruction_parser.py` |
| SpectralAbstraction | **Done** | `src/cgvd/spectral_abstraction.py` |
| CLI Integration | **Done** | `scripts/try_checkpoint_in_simpler.py` |
| Dependencies | **Done** | `pyproject.toml` |
| Eval Script (CGVD) | **Done** | `run_bridge_eval.sh` |
| Eval Script (Baseline) | **Done** | `run_bridge_eval_baseline.sh` |

### Quick Start

```bash
# Run baseline evaluation (no CGVD)
./run_bridge_eval_baseline.sh

# Run CGVD evaluation (with background blur + debug output)
./run_bridge_eval.sh
```

### Manual Usage

```bash
# Test with mock segmenter (no SAM3 model required)
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --use_cgvd \
    --cgvd_use_mock \
    --cgvd_verbose \
    --recording

# Test with real SAM3 (requires transformers main branch)
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --use_cgvd \
    --cgvd_blur_sigma 15 \
    --recording
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_cgvd` | False | Enable CGVD wrapper |
| `--cgvd_blur_sigma` | 15.0 | Gaussian blur sigma for background |
| `--cgvd_update_freq` | 1 | Frames between mask updates (1=every frame for smooth tracking) |
| `--cgvd_presence_threshold` | 0.15 | SAM3 confidence threshold (lower=more permissive) |
| `--cgvd_use_mock` | False | Use mock segmenter for testing |
| `--cgvd_feather_edges` | False | Apply edge feathering to masks |
| `--cgvd_verbose` | False | Print debug information |
| `--cgvd_save_debug` | False | Save debug images to `cgvd_debug/` |

---

## Overview

Implement a model-agnostic perception framework called **Concept-Gated Visual Distillation (CGVD)**. The goal is to "clean" visual observations before they reach the robotic policy to prevent feature dilution in cluttered environments.

**Key Decisions:**
- **Segmentation**: SAM 3 (text → masks directly via `Sam3Processor`/`Sam3Model`)
- **Background**: Gaussian blur (spectral abstraction), NOT neutral gray
- **Integration**: `gym.Wrapper` with `step()`/`reset()` overrides
- **Frame Rate**: 1Hz SAM (every 10 frames), per-frame blur
- **Robot Arm**: ALWAYS include in mask (critical for proprioception alignment)
- **Eval Platform**: SimplerEnv simulation
- **VLA Models**: Pi0, OpenVLA, GR00T, Gemini Robotics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CGVDWrapper (gym.Wrapper)                 │
│                                                              │
│  step() / reset():                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Get image via get_image_from_maniskill2_obs_dict()│   │
│  │ 2. Get instruction via env.get_language_instruction()│   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Stage 1: Interaction-Aware Decomposition              │   │
│  │   "pick apple and place in basket"                    │   │
│  │   → target: "apple", anchor: "basket"                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Stage 2: Concept-Driven Grounding (1Hz)               │   │
│  │   SAM 3 prompt: "apple. basket. robot arm. gripper"   │   │
│  │   → Binary masks (hallucination check: IoU > 0.4)     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Stage 3: Spectral Visual Abstraction (every frame)    │   │
│  │   I_distilled = M * I_raw + (1-M) * GaussianBlur(I)   │   │
│  │   sigma=15 (configurable)                             │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Write distilled image back into obs dict in-place    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         SimplerAdapter (unchanged) → VLA Model              │
│         Pi0 / OpenVLA / GR00T / Gemini Robotics             │
└─────────────────────────────────────────────────────────────┘
```

**Critical Design Decisions:**
1. **Robot arm ALWAYS in mask** - Prevents proprioception alignment failure
2. **Blur, not gray** - Preserves low-freq shapes for collision avoidance
3. **Per-frame SAM updates** - Smooth mask tracking as robot moves
4. **No fallback behavior** - SAM3 output used directly for debugging visibility

## VLA Model Integration

| Model | Type | Integration |
|-------|------|-------------|
| **Pi0** | Flow matching | This repo - wrap env with CGVD |
| **OpenVLA** | Discrete tokens | Same wrapper, different policy |
| **GR00T** | NVIDIA foundation | API wrapper (have access) |
| **Gemini Robotics** | Google | API wrapper (have access) |

## Files Created

```
src/cgvd/
├── __init__.py
├── cgvd_wrapper.py             # Main gym.Wrapper (step/reset override)
├── sam3_segmenter.py           # SAM 3 interface (Sam3Processor/Sam3Model)
├── instruction_parser.py       # Target + Anchor extraction
└── spectral_abstraction.py     # Gaussian blur compositing
```

### Evaluation Scripts

| Script | Description |
|--------|-------------|
| `run_bridge_eval.sh` | Run evaluation **with CGVD** (blur background, debug output) |
| `run_bridge_eval_baseline.sh` | Run evaluation **without CGVD** (baseline for comparison) |
| `run_bridge_eval_multi.sh` | Run evaluation on multiple tasks |

**SimplerEnv-Specific APIs (MUST USE):**
```python
# Image extraction - DO NOT use obs['image']
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
image = get_image_from_maniskill2_obs_dict(env, obs)

# Instruction - DO NOT use obs['mission']
instruction = env.unwrapped.get_language_instruction()
```

## Files to Modify

| File | Change |
|------|--------|
| `scripts/try_checkpoint_in_simpler.py` | Wrap env with `CGVDWrapper` when `--use_cgvd` flag |
| `pyproject.toml` | Add SAM 3 dependency |

**NOTE:** SimplerAdapter is NOT modified - wrapper intercepts obs before adapter sees it.

## Key Implementation Details

### 1. CGVDWrapper Class (Main Entry Point)
```python
class CGVDWrapper(gym.Wrapper):
    def __init__(self, env, update_freq=1, blur_sigma=15, presence_threshold=0.15):
        super().__init__(env)
        self.segmenter = SAM3Segmenter()
        self.parser = InstructionParser()

        self.update_freq = update_freq      # Per-frame for smooth tracking
        self.blur_sigma = blur_sigma        # Configurable, default 15
        self.presence_threshold = presence_threshold  # Detection confidence

        self.cached_mask = None
        self.frame_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._apply_cgvd(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cached_mask = None
        self.frame_count = 0
        obs = self._apply_cgvd(obs)
        return obs, info

    def _apply_cgvd(self, obs):
        # Get image using SimplerEnv utility
        image = get_image_from_maniskill2_obs_dict(self.env, obs)
        instruction = self.env.unwrapped.get_language_instruction()

        # Stage 1: Parse → target, anchor
        target, anchor = self.parser.parse(instruction)

        # Stage 2: Segment at 1Hz (with robot arm ALWAYS included)
        if self.frame_count % self.update_freq == 0 or self.cached_mask is None:
            concepts = self._build_concept_prompt(target, anchor)
            self.cached_mask = self.segmenter.segment(image, concepts)
        self.frame_count += 1

        # Stage 3: Spectral abstraction
        distilled = self._apply_blur_composite(image, self.cached_mask)

        # Write back (in-place modification)
        # ... implementation depends on obs structure
        return obs

    def _build_concept_prompt(self, target, anchor):
        # ALWAYS include robot arm!
        concepts = [target]
        if anchor:
            concepts.append(anchor)
        concepts.extend(["robot arm", "robot gripper"])
        return ". ".join(concepts)
```

### 2. Instruction Parser (Target + Anchor)
```python
def parse(self, instruction: str) -> Tuple[str, Optional[str]]:
    # Short-term hack for known tasks
    text = instruction.lower()
    if "apple" in text and "basket" in text:
        return ("apple", "wicker basket")
    elif "apple" in text and "drawer" in text:
        return ("apple", "drawer")
    elif "spoon" in text and "towel" in text:
        return ("spoon", "towel")
    # ... more task-specific patterns
    # Fallback: target only, no anchor
    return (self._extract_noun(text), None)
```

### 3. Spectral Abstraction (Blur Composite)
```python
def _apply_blur_composite(self, image, mask):
    # Blur background
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=self.blur_sigma)

    # Composite: foreground sharp, background blurred
    mask_3d = mask[..., np.newaxis].astype(np.float32)
    distilled = image * mask_3d + blurred * (1 - mask_3d)

    return distilled.astype(np.uint8)
```

## Evaluation Protocol

### Progressive Clutter Until Failure
Instead of fixed clutter levels, progressively add distractors until task failure:

```
Episode 1: 0 distractors → Success ✓
Episode 2: 1 distractor  → Success ✓
Episode 3: 2 distractors → Success ✓
...
Episode N: K distractors → Failure ✗
→ Breaking Point = K-1 objects
```

**Protocol:**
1. Start with clean scene (0 distractors)
2. Add 1 random distractor object per episode
3. Run N trials at each clutter level
4. Record success rate at each level
5. Find "breaking point" where SR drops below threshold (e.g., 50%)

### Metrics
- **Breaking Point**: # distractors before failure (higher = more robust)
- **Clutter Robustness Curve**: SR vs # distractors (area under curve)
- **Gating Lift**: Breaking point with gating / Breaking point baseline
- **Gating Detection Recall**: Target object detected / Total episodes
- **Inference Overhead**: Additional latency from gating (ms)

### Tasks
**Bridge (WidowX):** spoon_on_towel, carrot_on_plate, stack_cube, eggplant_in_basket
**Fractal (Google Robot):** pick_coke, move_near, open_drawer, close_drawer

## Ablation Studies

1. **Segmentation Model**: SAM 3 vs SAM 2 + Grounding DINO vs SAM 2 + OWL-ViT
2. **Gating Mode**: Strict vs Moderate vs Relaxed
3. **Background Type**: Gray vs Black vs Blur
4. **Confidence Threshold**: 0.2 - 0.5
5. **Robot Inclusion**: With/without robot arm in mask
6. **Prompt Engineering**: Direct instruction vs parsed concepts

## Implementation Order

### Phase 1: Core CGVD Framework
1. Create `src/cgvd/` directory structure
2. Implement `SAM3Segmenter` (Sam3Processor/Sam3Model + hallucination check)
3. Implement `InstructionParser` (target + anchor extraction)
4. Implement `spectral_abstraction.py` (Gaussian blur composite)
5. Implement `CGVDWrapper` (gym.Wrapper with step/reset)

### Phase 2: Integration
6. Add `--use_cgvd` flag to `try_checkpoint_in_simpler.py`
7. Add SAM 3 to `pyproject.toml`
8. Test on single task (widowx_spoon_on_towel)

### Phase 3: Evaluation Infrastructure
9. Create clutter generator for SimplerEnv
10. Create comprehensive evaluation script
11. Set up metrics logging and visualization

### Phase 4: Experiments
12. Run baseline (no CGVD) across clutter levels
13. Run CGVD configurations
14. Run ablation matrix
15. Measure computational overhead

## Verification

### Quick Start with Convenience Scripts

```bash
# Run baseline evaluation (no CGVD)
./run_bridge_eval_baseline.sh

# Run CGVD evaluation (with background blur)
./run_bridge_eval.sh
```

Compare the output videos to see the visual difference between baseline and CGVD.

### Manual Testing

1. **Single episode test with CGVD**:
   ```bash
   python scripts/try_checkpoint_in_simpler.py \
       --task widowx_spoon_on_towel \
       --checkpoint_path checkpoints/bridge_beta.pt \
       --use_cgvd \
       --cgvd_blur_sigma 15 \
       --recording
   ```

2. **Visual inspection**: Check recorded video to verify:
   - Target object (spoon) is sharp
   - Robot arm/gripper is sharp
   - Background is blurred
   - No hallucinated masks

3. **Compare baseline vs CGVD**:
   ```bash
   # Without CGVD
   python scripts/try_checkpoint_in_simpler.py --task widowx_spoon_on_towel ...

   # With CGVD
   python scripts/try_checkpoint_in_simpler.py --task widowx_spoon_on_towel --use_cgvd ...
   ```

## Dependencies to Add

```toml
# pyproject.toml additions
transformers >= 5.0.0      # Sam3Processor, Sam3Model (requires main branch until released)
opencv-python >= 4.8.0     # For spectral abstraction (Gaussian blur)
```

**SAM 3 Installation:**
```bash
# SAM3 requires transformers >= 5.0.0 (currently in main branch, not yet released)
pip install git+https://github.com/huggingface/transformers.git@main

# Requires HF_TOKEN for gated model access
export HF_TOKEN=your_huggingface_token
```

**SAM 3 API Usage:**
```python
from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to(device)

# Inference with text prompt
inputs = processor(images=image, text="apple. basket. robot arm", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Extract masks with hallucination check
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.4,           # Confidence threshold (hallucination check)
    mask_threshold=0.5,      # Mask binarization threshold
    target_sizes=[(h, w)],   # Original image size
)

# Results contain: masks, boxes, scores
masks = results[0]["masks"]  # List of binary masks
```

## Expected Results

### Breaking Point Comparison (# distractors before failure)

| Model | Baseline | + Gating | Lift |
|-------|----------|----------|------|
| Pi0 | ~5 | ~12 | 2.4x |
| OpenVLA | ~4 | ~10 | 2.5x |
| GR00T | ~6 | ~14 | 2.3x |
| Gemini Robotics | ~7 | ~15 | 2.1x |

### Key Paper Claims

1. **Model-agnostic**: Works across 4 diverse VLA architectures without retraining
2. **Significant robustness gain**: 2-3x improvement in clutter tolerance
3. **Minimal overhead**: ~30ms additional latency (SAM 3 on H200)
4. **Plug-and-play**: Simple preprocessing module, no architecture changes
5. **Unified pipeline**: Single foundation model (SAM 3) for detection + segmentation

## Implementation Priority

1. **Week 1-2**: Core gating framework (SAM 3 + background neutralization)
2. **Week 3**: Pi0 integration + initial experiments
3. **Week 4**: OpenVLA integration
4. **Week 5**: GR00T + Gemini Robotics API wrappers
5. **Week 6-7**: Full experiments + ablations
6. **Week 8**: Paper writing + visualizations
