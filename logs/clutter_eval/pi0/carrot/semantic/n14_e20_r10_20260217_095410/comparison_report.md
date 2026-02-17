# Paired Evaluation Report

**Generated:** 2026-02-17 19:36:38

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Pi0 |
| Task | `widowx_carrot_on_plate` |
| Category | semantic |
| Num Distractors | 14 |
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
| 1 | 0 | 60.0% | 40.0% | -20.0% | 60.0% | 40.0% | -20.0% |
| 2 | 1 | 45.0% | 45.0% | +0.0% | 45.0% | 40.0% | -5.0% |
| 3 | 2 | 50.0% | 55.0% | +5.0% | 35.0% | 50.0% | +15.0% |
| 4 | 3 | 55.0% | 40.0% | -15.0% | 55.0% | 30.0% | -25.0% |
| 5 | 4 | 70.0% | 65.0% | -5.0% | 60.0% | 60.0% | +0.0% |
| 6 | 5 | 60.0% | 45.0% | -15.0% | 55.0% | 35.0% | -20.0% |
| 7 | 6 | 60.0% | 50.0% | -10.0% | 50.0% | 45.0% | -5.0% |
| 8 | 7 | 60.0% | 45.0% | -15.0% | 50.0% | 45.0% | -5.0% |
| 9 | 8 | 65.0% | 35.0% | -30.0% | 55.0% | 25.0% | -30.0% |
| 10 | 9 | 60.0% | 40.0% | -20.0% | 55.0% | 30.0% | -25.0% |

## Summary Statistics

### Success Rate (SR)

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 58.5% | 6.7% | 45.0% | 70.0% |
| CGVD | 46.0% | 8.3% | 35.0% | 65.0% |

**Average SR Improvement: -12.5%**

### Hard Success Rate (h-SR) - Success without collisions

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 52.0% | 7.1% | 35.0% | 60.0% |
| CGVD | 40.0% | 10.0% | 25.0% | 60.0% |

**Average h-SR Improvement: -12.0%**

## Collision Analysis

| Method | Episodes with Collision | Collision Rate |
|--------|------------------------|----------------|
| Baseline | 48/200 | 24.0% |
| CGVD | 47/200 | 23.5% |

## Failure Mode Analysis

| Failure Mode | Baseline | CGVD |
|--------------|----------|------|
| success | 117 | 92 |
| never_reached | 13 | 27 |
| missed_grasp | 51 | 62 |
| dropped | 19 | 19 |

## CGVD Latency Analysis

| Component | Mean (s) | Std (s) | Min (s) | Max (s) |
|-----------|----------|---------|---------|--------|
| Total CGVD | 17.667 | 0.076 | 17.406 | 17.851 |
| SAM3 | 15.892 | 0.056 | 15.672 | 16.003 |
| LaMa | 3.461 | 0.025 | 3.402 | 3.582 |

## Per-Episode Details

### Run 1 (Seed: 0)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 2 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 4 | ✗ | ✗ | ✗ | ✗ | 0 | 4 | dropped | missed_grasp |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 6 | success | missed_grasp |
| 6 | ✗ | ✗ | ✗ | ✗ | 2 | 43 | dropped | missed_grasp |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✓ | ✗ | ✓ | 17 | 0 | missed_grasp | success |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | dropped |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 7 | 8 | dropped | missed_grasp |

### Run 2 (Seed: 1)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 1 | 28 | never_reached | dropped |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✓ | ✓ | ✗ | 0 | 4 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 3 | 2 | missed_grasp | never_reached |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 19 | dropped | missed_grasp |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 15 | dropped | missed_grasp |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 29 | success | missed_grasp |
| 19 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | missed_grasp | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 3 (Seed: 2)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 12 | success | never_reached |
| 4 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 36 | 0 | missed_grasp | success |
| 7 | ✗ | ✓ | ✗ | ✗ | 0 | 2 | missed_grasp | success |
| 8 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 9 | ✓ | ✓ | ✗ | ✓ | 1 | 0 | success | success |
| 10 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✗ | ✗ | ✗ | 9 | 17 | success | missed_grasp |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 4 | success | missed_grasp |
| 14 | ✗ | ✗ | ✗ | ✗ | 1 | 1 | missed_grasp | missed_grasp |
| 15 | ✓ | ✓ | ✗ | ✓ | 2 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 1 | 1 | dropped | missed_grasp |
| 17 | ✗ | ✗ | ✗ | ✗ | 1 | 7 | never_reached | missed_grasp |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 1 | success | dropped |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |

### Run 4 (Seed: 3)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✗ | ✓ | ✗ | ✗ | 8 | 4 | never_reached | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✓ | ✗ | ✗ | 11 | 33 | missed_grasp | success |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 1 | success | missed_grasp |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 1 | 2 | never_reached | missed_grasp |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 17 | ✗ | ✗ | ✗ | ✗ | 12 | 26 | never_reached | missed_grasp |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | dropped |

### Run 5 (Seed: 4)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | never_reached |
| 3 | ✓ | ✓ | ✗ | ✓ | 2 | 0 | success | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 2 | success | missed_grasp |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✓ | ✗ | ✓ | 30 | 0 | missed_grasp | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 4 | success | dropped |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 6 | 0 | missed_grasp | missed_grasp |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✓ | ✓ | ✓ | ✗ | 0 | 7 | success | success |
| 20 | ✓ | ✓ | ✗ | ✓ | 7 | 0 | success | success |

### Run 6 (Seed: 5)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 2 | ✓ | ✓ | ✗ | ✗ | 47 | 13 | success | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 1 | 0 | missed_grasp | missed_grasp |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 3 | dropped | dropped |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✓ | ✗ | ✗ | 10 | 2 | missed_grasp | success |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 34 | success | never_reached |
| 17 | ✗ | ✗ | ✗ | ✗ | 43 | 42 | never_reached | missed_grasp |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 7 (Seed: 6)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 14 | 0 | missed_grasp | missed_grasp |
| 7 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 29 | 51 | missed_grasp | missed_grasp |
| 9 | ✓ | ✓ | ✗ | ✗ | 1 | 12 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 4 | missed_grasp | dropped |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 1 | success | dropped |
| 14 | ✗ | ✗ | ✗ | ✗ | 55 | 6 | missed_grasp | missed_grasp |
| 15 | ✓ | ✓ | ✗ | ✓ | 2 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 7 | 0 | missed_grasp | never_reached |
| 19 | ✗ | ✓ | ✗ | ✓ | 3 | 0 | missed_grasp | success |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |

### Run 8 (Seed: 7)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 12 | ✗ | ✓ | ✗ | ✓ | 39 | 0 | missed_grasp | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 14 | ✗ | ✓ | ✗ | ✓ | 23 | 0 | missed_grasp | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 54 | 0 | missed_grasp | missed_grasp |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 17 | ✗ | ✗ | ✗ | ✗ | 21 | 0 | missed_grasp | never_reached |
| 18 | ✓ | ✓ | ✗ | ✓ | 2 | 0 | success | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 20 | ✓ | ✓ | ✗ | ✓ | 21 | 0 | success | success |

### Run 9 (Seed: 8)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✗ | 0 | 5 | success | success |
| 2 | ✗ | ✗ | ✗ | ✗ | 10 | 0 | missed_grasp | missed_grasp |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 9 | ✓ | ✓ | ✗ | ✓ | 1 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 22 | 0 | missed_grasp | never_reached |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 12 | missed_grasp | never_reached |
| 13 | ✓ | ✓ | ✗ | ✗ | 7 | 28 | success | success |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 16 | ✗ | ✗ | ✗ | ✗ | 21 | 0 | missed_grasp | never_reached |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |

### Run 10 (Seed: 9)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 17 | success | dropped |
| 7 | ✗ | ✗ | ✗ | ✗ | 11 | 10 | missed_grasp | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 8 | success | missed_grasp |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 11 | ✗ | ✗ | ✗ | ✗ | 2 | 0 | missed_grasp | never_reached |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 4 | missed_grasp | dropped |
| 14 | ✓ | ✓ | ✓ | ✗ | 0 | 1 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 6 | dropped | missed_grasp |
| 16 | ✗ | ✓ | ✗ | ✗ | 6 | 7 | dropped | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 27 | 0 | never_reached | success |
| 18 | ✓ | ✓ | ✗ | ✓ | 3 | 0 | success | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 3 | missed_grasp | missed_grasp |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |

