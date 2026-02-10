# Optimization: Score-Aware Cross-Validation for CGVD Safe-Set

## Motivation

With cross-validation disabled, CGVD already performs well above baseline with 10 semantic distractors. However, spatulas falsely detected as "spoon" remain in the safe-set, causing **over-protection** — those spatulas are shielded from inpainting even though they're distractors. Re-enabling cross-validation with a smarter guard could remove more distractors and further improve SR.

## Root Cause

The old guard iterates in arbitrary dict order, decrementing `concept_counts` as it removes instances. When only 1 remains, it stops — but the survivor is whichever happened to be last in iteration order, NOT the most genuine one.

With 10 distractors and ~7 "spoon" detections per frame, the real spoon can be removed early while a spatula survives as the "last instance."

## Solution: Genuineness Scoring

Replace the broken guard with a **two-pass approach**:

### Pass 1: Score all target instances

For each "spoon" instance, compute:

```
genuineness = safe_score("spoon") - max_overlapping_dist_score("spatula")
```

This measures how much more "spoon-like" than "spatula-like" the object is:

| Object | safe_score ("spoon") | dist_score ("spatula") | genuineness | Action |
|--------|---------------------|----------------------|-------------|--------|
| Real spoon | ~0.60 | ~0.35 | **+0.25** | KEEP |
| Spatula A | ~0.50 | ~0.92 | **-0.42** | Remove |
| Spatula B | ~0.55 | ~0.88 | **-0.33** | Remove |

### Pass 2: Keep only the most genuine

- Remove all target instances with negative genuineness (they're distractors)
- **Always** keep the instance with highest genuineness (the real target)
- Anchor instances (towel) are never filtered

### Safety fallback

If ALL target instances have negative genuineness (SAM3 very confused), keep the least-negative one. The target-specific safety valve (Fix 2) provides the final backstop by disabling CGVD entirely when the target is never detected.

## Why This Works

The real spoon has positive genuineness because it genuinely IS a spoon — it scores higher as "spoon" than as "spatula". Spatulas have negative genuineness because they're actually spatulas — they score higher as "spatula" than as "spoon".

The "always keep most genuine" rule guarantees the real spoon is never removed, regardless of iteration order.

## Implementation

**File:** `src/cgvd/cgvd_wrapper.py`

1. Rewrite `_cross_validate_safeset` method (lines 265-343) with genuineness scoring
2. Re-enable the cross-validation call in `_apply_cgvd` (replace the "DISABLED" comment at lines 578-583)
3. Use `logical_and` with binary threshold for mask subtraction (avoids soft-value bug)

## Verification

```bash
python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path checkpoints/bridge_beta.pt \
    --use_bf16 --num_episodes 5 \
    --distractors ycb_030_fork:0.55 ycb_033_spatula:0.50 ycb_032_knife:0.50 \
        ycb_030_fork:0.55 ycb_033_spatula:0.50 ycb_032_knife:0.50 \
        ycb_030_fork:0.55 ycb_033_spatula:0.50 ycb_032_knife:0.50 ycb_037_scissors:0.45 \
    --use_cgvd --cgvd_save_debug --cgvd_verbose \
    --output_dir logs/cgvd_fix_test
```

Check logs for:
- `Cross-val: KEEPING 'spoon_X'` — exactly 1 instance kept per frame (the real spoon)
- `Cross-val: removing 'spoon_Y'` — spatula false positives removed with negative genuineness
- Spoon visible in debug images column 5 ("Distilled")
- More spatulas removed vs the "disabled" approach
