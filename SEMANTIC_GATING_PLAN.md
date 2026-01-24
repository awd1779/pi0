# Semantic Gating Framework for VLA Models - RSS 2026

## Overview

Implement a model-agnostic "Semantic Gating" framework to address background-induced feature dilution in VLA models operating in cluttered environments. The framework uses vision foundation models to isolate task-relevant objects based on language instructions, projects them into canonical visual space with background neutralization.

**Key Decisions:**
- **Detection**: Grounding DINO (text → bounding boxes)
- **Segmentation**: SAM 2 (boxes → precise masks)
- **Eval Platform**: SimplerEnv simulation
- **VLA Models**: Pi0, OpenVLA, GR00T, Gemini Robotics
- **Clutter Protocol**: Progressive addition until failure

## Architecture

```
Raw Image + Instruction
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              SEMANTIC GATING MODULE                          │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Grounding    │    │    SAM 2     │    │  Background  │   │
│  │ DINO         │───▶│  Segmenter   │───▶│ Neutralizer  │   │
│  │ (Detection)  │    │  (Masking)   │    │  (Gating)    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         ▲                                                    │
│         │                                                    │
│  ┌──────────────┐                                           │
│  │ Instruction  │ "pick coke can" → ["coke can", "robot"]   │
│  │ Parser       │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
    Gated Image (clean background, task-relevant objects only)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│         VLA MODEL (model-agnostic - no retraining)          │
│  Pi0 / OpenVLA / GR00T / Gemini Robotics                    │
└─────────────────────────────────────────────────────────────┘
```

## VLA Model Integration

| Model | Type | Integration |
|-------|------|-------------|
| **Pi0** | Flow matching | This repo - direct integration |
| **OpenVLA** | Discrete tokens | Clone repo, add gating adapter |
| **GR00T** | NVIDIA foundation | API wrapper (needs access) |
| **Gemini Robotics** | Google | API wrapper (needs access) |

## Files to Create

```
src/gating/
├── __init__.py
├── semantic_gater.py           # Main orchestrator
├── detection/
│   ├── __init__.py
│   └── grounding_dino.py       # Text-guided detection
├── segmentation/
│   ├── __init__.py
│   └── sam_wrapper.py          # SAM mask generation
├── neutralization/
│   ├── __init__.py
│   └── background.py           # Background replacement
└── utils/
    ├── __init__.py
    ├── prompt_engineering.py   # Instruction → detection prompts
    └── mask_ops.py             # Mask composition
```

## Files to Modify

| File | Change |
|------|--------|
| `src/agent/env_adapter/simpler.py` | Add gating call in `preprocess()` (lines 53-71) |
| `src/agent/env_adapter/base.py` | Add gating config interface |
| `config/eval/bridge.yaml` | Add `semantic_gating` config block |
| `scripts/try_checkpoint_in_simpler.py` | Add `--use_gating` CLI flag |

## Key Implementation Details

### 1. SemanticGater Class (Main Entry Point)
```python
class SemanticGater:
    def gate(self, image: np.ndarray, instruction: str) -> np.ndarray:
        # 1. Parse instruction → detection prompts
        # 2. Detect objects with Grounding DINO
        # 3. Segment with SAM
        # 4. Compose mask (target + robot + workspace)
        # 5. Neutralize background
        return gated_image
```

### 2. Gating Modes
- **Strict**: Only target object visible
- **Moderate**: Target + robot arm + gripper (recommended)
- **Relaxed**: Target + robot + immediate workspace

### 3. Background Options
- `gray_128`: Neutral gray (recommended)
- `black`: Pure black
- `blur`: Blurred original background
- `domain_avg`: Domain-specific average color

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

1. **Detection Model**: DINO-Base vs DINO-Tiny vs OWL-ViT
2. **Segmentation**: SAM-Base vs SAM-Large vs Box-only
3. **Gating Mode**: Strict vs Moderate vs Relaxed
4. **Background Type**: Gray vs Black vs Blur
5. **Detection Threshold**: 0.2 - 0.5
6. **Robot Inclusion**: With/without robot arm

## Implementation Order

### Phase 1: Core Framework
1. Create `src/gating/` directory structure
2. Implement `GroundingDINODetector` wrapper
3. Implement `SAMSegmenter` wrapper
4. Implement `BackgroundNeutralizer`
5. Implement `InstructionParser`
6. Implement main `SemanticGater` orchestrator

### Phase 2: Integration
7. Modify `SimplerAdapter.preprocess()` to optionally apply gating
8. Add config schema to `bridge.yaml`
9. Add CLI flags to evaluation scripts
10. Test on single task (widowx_spoon_on_towel)

### Phase 3: Evaluation Infrastructure
11. Create clutter generator for SimplerEnv
12. Create comprehensive evaluation script
13. Set up metrics logging and visualization

### Phase 4: Experiments
14. Run baseline (no gating) across all clutter levels
15. Run all gating configurations
16. Run ablation matrix
17. Measure computational overhead

## Verification

1. **Unit test gating module**:
   ```bash
   python -m pytest tests/test_semantic_gater.py
   ```

2. **Single episode test**:
   ```bash
   python scripts/try_checkpoint_in_simpler.py \
       --task widowx_spoon_on_towel \
       --checkpoint_path checkpoints/bridge_beta.pt \
       --use_gating --gating_mode moderate \
       --recording
   ```

3. **Full evaluation**:
   ```bash
   python scripts/eval_semantic_gating.py \
       --tasks widowx_spoon_on_towel,widowx_carrot_on_plate \
       --clutter_levels 0,1,2,3,4 \
       --gating_modes none,strict,moderate,relaxed
   ```

## Dependencies to Add

```toml
# pyproject.toml additions
transformers >= 4.47.1      # For Grounding DINO
sam-2                       # SAM 2 from Meta (pip install sam-2)
# OR from source: git clone https://github.com/facebookresearch/sam2

# For other VLA models:
openvla                     # OpenVLA (if testing)
```

**SAM 2 Installation:**
```bash
pip install sam-2
# Or from source for latest:
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e .
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
3. **Minimal overhead**: ~100ms additional latency (Grounding DINO + SAM 2)
4. **Plug-and-play**: Simple preprocessing module, no architecture changes

## Implementation Priority

1. **Week 1-2**: Core gating framework (Grounding DINO + SAM 2 + neutralization)
2. **Week 3**: Pi0 integration + initial experiments
3. **Week 4**: OpenVLA integration
4. **Week 5**: GR00T + Gemini Robotics API wrappers
5. **Week 6-7**: Full experiments + ablations
6. **Week 8**: Paper writing + visualizations
