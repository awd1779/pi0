# Paired Evaluation Report

**Generated:** 2026-02-17 21:38:19

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Pi0 |
| Task | `widowx_carrot_on_plate` |
| Category | semantic |
| Num Distractors | 18 |
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
| 1 | 0 | 40.0% | 60.0% | +20.0% | 35.0% | 45.0% | +10.0% |
| 2 | 1 | 50.0% | 60.0% | +10.0% | 35.0% | 45.0% | +10.0% |
| 3 | 2 | 50.0% | 45.0% | -5.0% | 40.0% | 30.0% | -10.0% |
| 4 | 3 | 50.0% | 60.0% | +10.0% | 25.0% | 50.0% | +25.0% |
| 5 | 4 | 60.0% | 45.0% | -15.0% | 55.0% | 40.0% | -15.0% |
| 6 | 5 | 75.0% | 45.0% | -30.0% | 60.0% | 40.0% | -20.0% |
| 7 | 6 | 50.0% | 45.0% | -5.0% | 45.0% | 40.0% | -5.0% |
| 8 | 7 | 45.0% | 55.0% | +10.0% | 35.0% | 40.0% | +5.0% |
| 9 | 8 | 50.0% | 50.0% | +0.0% | 30.0% | 30.0% | +0.0% |
| 10 | 9 | 45.0% | 45.0% | +0.0% | 45.0% | 40.0% | -5.0% |

## Summary Statistics

### Success Rate (SR)

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 51.5% | 9.2% | 40.0% | 75.0% |
| CGVD | 51.0% | 6.6% | 45.0% | 60.0% |

**Average SR Improvement: -0.5%**

### Hard Success Rate (h-SR) - Success without collisions

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 40.5% | 10.4% | 25.0% | 60.0% |
| CGVD | 40.0% | 5.9% | 30.0% | 50.0% |

**Average h-SR Improvement: -0.5%**

## Collision Analysis

| Method | Episodes with Collision | Collision Rate |
|--------|------------------------|----------------|
| Baseline | 67/200 | 33.5% |
| CGVD | 58/200 | 29.0% |

## Failure Mode Analysis

| Failure Mode | Baseline | CGVD |
|--------------|----------|------|
| success | 103 | 102 |
| never_reached | 14 | 21 |
| missed_grasp | 55 | 59 |
| dropped | 28 | 18 |

## CGVD Latency Analysis

| Component | Mean (s) | Std (s) | Min (s) | Max (s) |
|-----------|----------|---------|---------|--------|
| Total CGVD | 17.741 | 0.075 | 17.482 | 17.967 |
| SAM3 | 15.908 | 0.060 | 15.689 | 16.060 |
| LaMa | 3.455 | 0.025 | 3.405 | 3.578 |

## Per-Episode Details

### Run 1 (Seed: 0)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 43 | 0 | dropped | never_reached |
| 5 | ✓ | ✓ | ✓ | ✗ | 0 | 4 | success | success |
| 6 | ✗ | ✓ | ✗ | ✗ | 14 | 6 | dropped | success |
| 7 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 5 | 7 | missed_grasp | missed_grasp |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✗ | ✓ | ✗ | ✗ | 0 | 9 | missed_grasp | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 15 | ✗ | ✓ | ✗ | ✓ | 18 | 0 | dropped | success |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 145 | success | missed_grasp |
| 17 | ✗ | ✗ | ✗ | ✗ | 34 | 0 | missed_grasp | missed_grasp |
| 18 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 19 | ✓ | ✓ | ✗ | ✓ | 10 | 0 | success | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 4 | dropped | missed_grasp |

### Run 2 (Seed: 1)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 8 | 0 | never_reached | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 4 | ✓ | ✓ | ✗ | ✗ | 10 | 1 | success | success |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 30 | 28 | missed_grasp | missed_grasp |
| 10 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | never_reached | success |
| 11 | ✓ | ✗ | ✗ | ✗ | 7 | 0 | success | missed_grasp |
| 12 | ✗ | ✓ | ✗ | ✓ | 11 | 0 | dropped | success |
| 13 | ✗ | ✓ | ✗ | ✗ | 0 | 25 | missed_grasp | success |
| 14 | ✗ | ✓ | ✗ | ✓ | 7 | 0 | missed_grasp | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 32 | 11 | missed_grasp | missed_grasp |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 13 | missed_grasp | missed_grasp |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 25 | 5 | missed_grasp | never_reached |
| 20 | ✓ | ✓ | ✗ | ✗ | 1 | 5 | success | success |

### Run 3 (Seed: 2)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 8 | 31 | dropped | missed_grasp |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 2 | 0 | missed_grasp | missed_grasp |
| 7 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 8 | ✓ | ✗ | ✗ | ✗ | 6 | 0 | success | never_reached |
| 9 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | dropped |
| 11 | ✓ | ✓ | ✓ | ✗ | 0 | 1 | success | success |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 13 | ✓ | ✓ | ✗ | ✗ | 1 | 3 | success | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 7 | 0 | dropped | missed_grasp |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 16 | ✗ | ✗ | ✗ | ✗ | 25 | 9 | never_reached | missed_grasp |
| 17 | ✗ | ✓ | ✗ | ✗ | 80 | 29 | never_reached | success |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 2 | success | missed_grasp |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 19 | success | missed_grasp |

### Run 4 (Seed: 3)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 11 | 0 | dropped | success |
| 2 | ✓ | ✓ | ✗ | ✓ | 5 | 0 | success | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 2 | dropped | never_reached |
| 4 | ✓ | ✓ | ✓ | ✗ | 0 | 1 | success | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 113 | 21 | dropped | dropped |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 5 | missed_grasp | dropped |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✓ | ✓ | ✗ | ✓ | 11 | 0 | success | success |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 12 | ✓ | ✓ | ✗ | ✓ | 8 | 0 | success | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 36 | 0 | missed_grasp | success |
| 14 | ✓ | ✓ | ✗ | ✓ | 1 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 16 | ✗ | ✓ | ✗ | ✓ | 18 | 0 | never_reached | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 29 | missed_grasp | missed_grasp |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 20 | ✓ | ✓ | ✗ | ✗ | 1 | 1 | success | success |

### Run 5 (Seed: 4)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 11 | 0 | missed_grasp | success |
| 2 | ✗ | ✓ | ✗ | ✗ | 0 | 5 | missed_grasp | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 1 | 0 | missed_grasp | missed_grasp |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 7 | ✗ | ✗ | ✗ | ✗ | 5 | 0 | dropped | never_reached |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 31 | success | never_reached |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 24 | success | missed_grasp |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✓ | ✗ | ✗ | ✗ | 10 | 5 | success | never_reached |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |

### Run 6 (Seed: 5)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✓ | ✗ | ✗ | 0 | 2 | missed_grasp | success |
| 4 | ✓ | ✗ | ✓ | ✗ | 0 | 15 | success | dropped |
| 5 | ✗ | ✗ | ✗ | ✗ | 9 | 0 | dropped | missed_grasp |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✓ | ✗ | ✗ | ✗ | 10 | 4 | success | dropped |
| 11 | ✓ | ✓ | ✗ | ✓ | 16 | 0 | success | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 1 | never_reached | missed_grasp |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 17 | ✓ | ✗ | ✓ | ✗ | 0 | 6 | success | missed_grasp |
| 18 | ✓ | ✗ | ✗ | ✗ | 6 | 0 | success | never_reached |
| 19 | ✗ | ✓ | ✗ | ✓ | 8 | 0 | missed_grasp | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 7 (Seed: 6)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 2 | ✓ | ✓ | ✗ | ✓ | 3 | 0 | success | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 16 | 0 | missed_grasp | missed_grasp |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 3 | missed_grasp | missed_grasp |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 24 | 31 | missed_grasp | missed_grasp |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 30 | success | dropped |
| 10 | ✗ | ✗ | ✗ | ✗ | 28 | 18 | missed_grasp | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 13 | ✓ | ✓ | ✓ | ✗ | 0 | 5 | success | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 21 | 0 | missed_grasp | never_reached |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 38 | 3 | missed_grasp | dropped |

### Run 8 (Seed: 7)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 4 | missed_grasp | missed_grasp |
| 2 | ✗ | ✗ | ✗ | ✗ | 11 | 0 | never_reached | missed_grasp |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 4 | ✓ | ✓ | ✗ | ✓ | 2 | 0 | success | success |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 6 | ✗ | ✗ | ✗ | ✗ | 5 | 0 | dropped | never_reached |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 8 | ✓ | ✓ | ✓ | ✗ | 0 | 11 | success | success |
| 9 | ✗ | ✓ | ✗ | ✗ | 0 | 4 | dropped | success |
| 10 | ✓ | ✓ | ✗ | ✓ | 5 | 0 | success | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 43 | 0 | missed_grasp | success |
| 12 | ✓ | ✓ | ✓ | ✗ | 0 | 1 | success | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 12 | 0 | missed_grasp | missed_grasp |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 9 (Seed: 8)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✗ | ✓ | 7 | 0 | success | success |
| 2 | ✓ | ✗ | ✓ | ✗ | 0 | 23 | success | dropped |
| 3 | ✗ | ✓ | ✗ | ✓ | 23 | 0 | never_reached | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | dropped |
| 5 | ✓ | ✓ | ✓ | ✗ | 0 | 1 | success | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 7 | ✓ | ✓ | ✗ | ✗ | 10 | 6 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 129 | 10 | missed_grasp | missed_grasp |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 13 | 0 | missed_grasp | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✗ | 0 | 1 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 7 | missed_grasp | missed_grasp |
| 13 | ✓ | ✗ | ✗ | ✗ | 12 | 4 | success | missed_grasp |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 16 | ✗ | ✗ | ✗ | ✗ | 4 | 0 | dropped | never_reached |
| 17 | ✗ | ✓ | ✗ | ✗ | 12 | 2 | missed_grasp | success |
| 18 | ✓ | ✓ | ✗ | ✓ | 10 | 0 | success | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |

### Run 10 (Seed: 9)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 4 | ✗ | ✓ | ✗ | ✓ | 3 | 0 | missed_grasp | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | dropped |
| 9 | ✗ | ✗ | ✗ | ✗ | 1 | 2 | dropped | dropped |
| 10 | ✗ | ✗ | ✗ | ✗ | 12 | 16 | missed_grasp | missed_grasp |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 14 | missed_grasp | never_reached |
| 12 | ✓ | ✓ | ✓ | ✗ | 0 | 3 | success | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 14 | ✗ | ✓ | ✗ | ✓ | 1 | 0 | dropped | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 16 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | missed_grasp | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 13 | success | never_reached |

