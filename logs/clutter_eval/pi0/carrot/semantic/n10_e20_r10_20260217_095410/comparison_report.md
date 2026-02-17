# Paired Evaluation Report

**Generated:** 2026-02-17 17:37:04

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Pi0 |
| Task | `widowx_carrot_on_plate` |
| Category | semantic |
| Num Distractors | 10 |
| Episodes per run | 20 |
| Number of runs | 10 |
| Seeds | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Checkpoint | `/app/open-pi-zero/checkpoints/bridge_beta.pt` |

### Distractors
- `rc_corn_0:0.07`
- `rc_corn_3:0.07`
- `rc_corn_4:0.07`
- `rc_cucumber_0:0.07`
- `rc_cucumber_3:0.07`
- `rc_cucumber_5:0.07`
- `rc_eggplant_0:0.07`
- `rc_eggplant_2:0.07`
- `rc_garlic_0:0.07`
- `rc_onion_0:0.07`
- `rc_onion_1:0.07`
- `rc_bell_pepper_0:0.07`
- `rc_bell_pepper_1:0.07`
- `rc_potato_0:0.07`
- `rc_potato_5:0.07`
- `rc_tomato_0:0.07`
- `rc_tomato_6:0.07`
- `rc_squash_0:0.07`
- `rc_squash_16:0.07`
- `rc_sweet_potato_2:0.07`

## Results by Run

| Run | Seed | Baseline SR | CGVD SR | Δ SR | Baseline h-SR | CGVD h-SR | Δ h-SR |
|-----|------|-------------|---------|------|---------------|-----------|--------|
| 1 | 0 | 55.0% | 45.0% | -10.0% | 50.0% | 45.0% | -5.0% |
| 2 | 1 | 50.0% | 25.0% | -25.0% | 45.0% | 25.0% | -20.0% |
| 3 | 2 | 45.0% | 25.0% | -20.0% | 45.0% | 25.0% | -20.0% |
| 4 | 3 | 60.0% | 50.0% | -10.0% | 55.0% | 50.0% | -5.0% |
| 5 | 4 | 65.0% | 60.0% | -5.0% | 60.0% | 60.0% | +0.0% |
| 6 | 5 | 55.0% | 40.0% | -15.0% | 55.0% | 40.0% | -15.0% |
| 7 | 6 | 55.0% | 45.0% | -10.0% | 55.0% | 45.0% | -10.0% |
| 8 | 7 | 70.0% | 45.0% | -25.0% | 70.0% | 45.0% | -25.0% |
| 9 | 8 | 50.0% | 50.0% | +0.0% | 50.0% | 50.0% | +0.0% |
| 10 | 9 | 55.0% | 50.0% | -5.0% | 50.0% | 35.0% | -15.0% |

## Summary Statistics

### Success Rate (SR)

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 56.0% | 7.0% | 45.0% | 70.0% |
| CGVD | 43.5% | 10.5% | 25.0% | 60.0% |

**Average SR Improvement: -12.5%**

### Hard Success Rate (h-SR) - Success without collisions

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 53.5% | 7.1% | 45.0% | 70.0% |
| CGVD | 42.0% | 10.5% | 25.0% | 60.0% |

**Average h-SR Improvement: -11.5%**

## Collision Analysis

| Method | Episodes with Collision | Collision Rate |
|--------|------------------------|----------------|
| Baseline | 25/200 | 12.5% |
| CGVD | 29/200 | 14.5% |

## Failure Mode Analysis

| Failure Mode | Baseline | CGVD |
|--------------|----------|------|
| success | 112 | 87 |
| never_reached | 9 | 24 |
| missed_grasp | 60 | 64 |
| dropped | 19 | 25 |

## CGVD Latency Analysis

| Component | Mean (s) | Std (s) | Min (s) | Max (s) |
|-----------|----------|---------|---------|--------|
| Total CGVD | 17.561 | 0.073 | 17.367 | 17.756 |
| SAM3 | 15.871 | 0.056 | 15.653 | 15.991 |
| LaMa | 3.466 | 0.036 | 3.395 | 3.628 |

## Per-Episode Details

### Run 1 (Seed: 0)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 1 | success | dropped |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✗ | ✗ | ✗ | 8 | 0 | success | dropped |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 3 | success | missed_grasp |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 23 | missed_grasp | missed_grasp |
| 15 | ✗ | ✓ | ✗ | ✓ | 33 | 0 | missed_grasp | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 20 | dropped | dropped |
| 17 | ✓ | ✗ | ✓ | ✗ | 0 | 6 | success | missed_grasp |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 20 | ✗ | ✓ | ✗ | ✓ | 6 | 0 | missed_grasp | success |

### Run 2 (Seed: 1)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | dropped |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 2 | 0 | never_reached | never_reached |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 7 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | never_reached |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 2 | missed_grasp | never_reached |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 15 | ✓ | ✓ | ✗ | ✓ | 7 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 38 | 0 | missed_grasp | never_reached |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 14 | success | dropped |
| 19 | ✗ | ✗ | ✗ | ✗ | 1 | 0 | missed_grasp | missed_grasp |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |

### Run 3 (Seed: 2)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 27 | missed_grasp | missed_grasp |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 80 | 14 | missed_grasp | missed_grasp |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 5 | missed_grasp | missed_grasp |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 80 | success | dropped |
| 14 | ✗ | ✗ | ✗ | ✗ | 3 | 0 | missed_grasp | missed_grasp |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 17 | ✗ | ✗ | ✗ | ✗ | 19 | 32 | missed_grasp | missed_grasp |
| 18 | ✗ | ✗ | ✗ | ✗ | 0 | 2 | dropped | dropped |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |

### Run 4 (Seed: 3)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 60 | 0 | never_reached | dropped |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 12 | ✓ | ✓ | ✗ | ✓ | 1 | 0 | success | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 3 | missed_grasp | dropped |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |

### Run 5 (Seed: 4)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | dropped |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 4 | success | missed_grasp |
| 12 | ✗ | ✗ | ✗ | ✗ | 12 | 16 | never_reached | never_reached |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 5 | 0 | missed_grasp | missed_grasp |
| 17 | ✓ | ✓ | ✗ | ✓ | 1 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |

### Run 6 (Seed: 5)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 26 | 0 | missed_grasp | missed_grasp |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 10 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 14 | 0 | missed_grasp | never_reached |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 2 | 0 | missed_grasp | missed_grasp |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |

### Run 7 (Seed: 6)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 2 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 21 | 5 | missed_grasp | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 2 | dropped | missed_grasp |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | never_reached |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |

### Run 8 (Seed: 7)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 4 | 0 | dropped | never_reached |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 5 | 0 | missed_grasp | missed_grasp |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 9 (Seed: 8)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 2 | success | missed_grasp |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 50 | success | missed_grasp |
| 6 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 10 | never_reached | missed_grasp |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 9 | success | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 1 | 0 | missed_grasp | missed_grasp |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |

### Run 10 (Seed: 9)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 3 | success | never_reached |
| 2 | ✓ | ✓ | ✓ | ✗ | 0 | 1 | success | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 4 | ✓ | ✓ | ✗ | ✗ | 3 | 3 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✓ | ✓ | ✗ | 0 | 2 | success | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 25 | 3 | missed_grasp | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 10 | ✗ | ✗ | ✗ | ✗ | 18 | 0 | missed_grasp | missed_grasp |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 2 | missed_grasp | missed_grasp |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 8 | success | missed_grasp |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |

