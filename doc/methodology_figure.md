# CGVD Pipeline Figure — Design Specification

Design spec for `figures/pipeline.pdf`, referenced by `paper/method.tex` as `Fig.~\ref{fig:pipeline}`.

## Figure Layout

Horizontal left-to-right pipeline, full `\textwidth` (`figure*` environment), ~7in wide × ~3in tall.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  CGVD Pipeline Overview                                                         │
│                                                                                 │
│  ┌──────────┐    ┌──────────────────────┐    ┌──────────┐    ┌───────────────┐  │
│  │Instruction│    │  Dual-Mask           │    │Safe-Set  │    │  Inpaint &    │  │
│  │ Parsing   │───▶│  Segmentation       │───▶│Subtraction│───▶│  Composite   │  │
│  │           │    │  (SAM3)              │    │  D ∧ ¬S   │    │  (LaMa)      │  │
│  └──────────┘    └──────────────────────┘    └──────────┘    └───────────────┘  │
│       │                  │    │                    │                  │           │
│  "put spoon         ┌────┴────┐              Final mask         Distilled       │
│   on towel"         ▼         ▼              (yellow)          observation      │
│       │          Distractor  Safe-set                          ō_t → VLA        │
│  target=spoon    mask (red)  mask (green)                                       │
│  anchor=towel                                                                   │
│                                                                                 │
│  ── Warmup phase (frames 0-5): accumulate masks via union ──                    │
│  ── Post-warmup: cached masks, only robot tracked live ──                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Design Spec

**Overall structure**: Left-to-right flow, 4 stages corresponding to method.tex §3.2–§3.6.

### Stage 1 — Instruction Parsing (§3.2 Concept-Gated Decomposition)

- Show example instruction in a speech bubble or text box: `"put the spoon on the towel"`
- Arrow splitting into two concept lists:
  - Safe-set S = {spoon, towel, robot} (green label)
  - Distractor set D = {fork, spatula, knife, ...} (red label)
- Small icon or text showing this is deterministic (no GPT-4o)

### Stage 2 — Dual-Mask Segmentation (§3.3)

- Show a sample camera frame with cluttered scene
- Two parallel arrows from the image:
  - Top path → SAM3 with distractor prompts → red mask overlay (M_dist)
  - Bottom path → SAM3 with safe-set prompts → green mask overlay (M_safe)
- Emphasize **independence** with a visual separator between paths
- Callout boxes:
  - "Strict threshold (τ=0.3)" on distractor path
  - "Permissive threshold (τ=0.15)" on safe-set path
- Optional inset: warmup timeline showing frames 0→5 with union accumulation

### Stage 2.5 — Cross-Validation (§3.4, optional inset)

- Small inset diagram showing:
  - Spatula detected as both "spatula" (distractor) AND "spoon" (safe-set)
  - Genuineness score: g(s_i) = σ_safe − σ_dist
  - Arrow showing false positive removed from safe-set
- Render as a dashed-border callout attached to Stage 2

### Stage 3 — Safe-Set Subtraction (§3.5)

- Visual equation with mask thumbnails:
  - M_dist (red) ∧ ¬ dilate(M_safe, r) (green with dilation buffer) = M_final (yellow)
- Highlight the **safety guarantee**: even if target in M_dist, subtraction removes it
- Show dilation buffer as a slightly expanded green border

### Stage 4 — Inpaint & Composite (§3.6)

- Two sub-steps:
  1. LaMa inpainting: M_final regions filled with realistic background
  2. Feathered compositing: cached inpainted bg + live frame → distilled ō_t
- Show before (cluttered) and after (clean) observation side by side
- Arrow from output to generic "VLA π_θ" box → action a_t
- Indicate caching with a small cache/refresh icon

### Bottom Banner (§3.7 Automatic Fallback)

- Thin strip at bottom: "If target not detected during warmup → pass-through (monotonicity guarantee)"

## Color Scheme

| Element | Color | Hex |
|---------|-------|-----|
| Safe-set masks | Green | #4CAF50 |
| Distractor masks | Red | #F44336 |
| Final inpainting mask | Yellow/Amber | #FFC107 |
| Pipeline arrows | Dark grey | #424242 |
| Background boxes | Light grey | #F5F5F5 |
| Accent (guarantees) | Blue | #2196F3 |

All colors chosen for print-friendliness and colorblind accessibility (avoid red-green only distinctions — use shape/pattern as redundant channel).

## Caption

> CGVD pipeline overview. The language instruction gates which visual concepts are protected (safe-set) and which are removed (distractors). Safe-set subtraction ensures the target is architecturally excluded from the inpainting mask, regardless of segmentation errors.

## Production Notes

- **Tool**: TikZ, Inkscape, or Figma — TikZ recommended for camera-ready consistency
- **Size**: `\textwidth` wide (~7in), ~3in tall → `figure*` environment
- **Resolution**: Vector PDF preferred
- **Example frames**: Can be generated by running an episode with debug output (`cgvd_debug/` directory already saves 6-column visualizations)
