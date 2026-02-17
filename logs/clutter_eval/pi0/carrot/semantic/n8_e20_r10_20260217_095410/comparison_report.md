# Paired Evaluation Report

**Generated:** 2026-02-17 15:39:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Pi0 |
| Task | `widowx_carrot_on_plate` |
| Category | semantic |
| Num Distractors | 8 |
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
| 1 | 0 | 70.0% | 45.0% | -25.0% | 60.0% | 40.0% | -20.0% |
| 2 | 1 | 60.0% | 40.0% | -20.0% | 55.0% | 35.0% | -20.0% |
| 3 | 2 | 55.0% | 40.0% | -15.0% | 50.0% | 40.0% | -10.0% |
| 4 | 3 | 45.0% | 45.0% | +0.0% | 40.0% | 45.0% | +5.0% |
| 5 | 4 | 65.0% | 45.0% | -20.0% | 65.0% | 45.0% | -20.0% |
| 6 | 5 | 70.0% | 40.0% | -30.0% | 70.0% | 35.0% | -35.0% |
| 7 | 6 | 60.0% | 55.0% | -5.0% | 55.0% | 55.0% | +0.0% |
| 8 | 7 | 45.0% | 55.0% | +10.0% | 45.0% | 50.0% | +5.0% |
| 9 | 8 | 55.0% | 55.0% | +0.0% | 55.0% | 50.0% | -5.0% |
| 10 | 9 | 70.0% | 30.0% | -40.0% | 65.0% | 30.0% | -35.0% |

## Summary Statistics

### Success Rate (SR)

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 59.5% | 9.1% | 45.0% | 70.0% |
| CGVD | 45.0% | 7.7% | 30.0% | 55.0% |

**Average SR Improvement: -14.5%**

### Hard Success Rate (h-SR) - Success without collisions

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 56.0% | 8.9% | 40.0% | 70.0% |
| CGVD | 42.5% | 7.5% | 30.0% | 55.0% |

**Average h-SR Improvement: -13.5%**

## Collision Analysis

| Method | Episodes with Collision | Collision Rate |
|--------|------------------------|----------------|
| Baseline | 25/200 | 12.5% |
| CGVD | 25/200 | 12.5% |

## Failure Mode Analysis

| Failure Mode | Baseline | CGVD |
|--------------|----------|------|
| success | 119 | 90 |
| never_reached | 10 | 28 |
| missed_grasp | 56 | 62 |
| dropped | 15 | 20 |

## CGVD Latency Analysis

| Component | Mean (s) | Std (s) | Min (s) | Max (s) |
|-----------|----------|---------|---------|--------|
| Total CGVD | 17.513 | 0.079 | 17.231 | 17.697 |
| SAM3 | 15.884 | 0.065 | 15.667 | 16.061 |
| LaMa | 3.457 | 0.025 | 3.397 | 3.613 |

## Per-Episode Details

### Run 1 (Seed: 0)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✓ | ✗ | ✓ | 7 | 0 | success | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 17 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 19 | ✓ | ✓ | ✗ | ✗ | 2 | 6 | success | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |

### Run 2 (Seed: 1)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 6 | 0 | missed_grasp | dropped |
| 2 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 4 | never_reached | missed_grasp |
| 4 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 5 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 6 | ✓ | ✗ | ✗ | ✗ | 3 | 1 | success | never_reached |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✓ | ✗ | ✓ | 6 | 0 | missed_grasp | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 9 | success | missed_grasp |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 13 | success | missed_grasp |
| 16 | ✗ | ✗ | ✗ | ✗ | 10 | 51 | missed_grasp | never_reached |
| 17 | ✗ | ✗ | ✗ | ✗ | 87 | 0 | missed_grasp | never_reached |
| 18 | ✓ | ✓ | ✓ | ✗ | 0 | 1 | success | success |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 3 (Seed: 2)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 4 | ✗ | ✗ | ✗ | ✗ | 0 | 2 | missed_grasp | missed_grasp |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 6 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 21 | missed_grasp | missed_grasp |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 9 | ✗ | ✗ | ✗ | ✗ | 1 | 0 | never_reached | never_reached |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✗ | ✗ | ✗ | 1 | 0 | success | dropped |

### Run 4 (Seed: 3)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 11 | missed_grasp | missed_grasp |
| 2 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 3 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 8 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 36 | 0 | missed_grasp | missed_grasp |
| 12 | ✓ | ✓ | ✗ | ✓ | 1 | 0 | success | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 33 | 20 | missed_grasp | dropped |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 17 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |

### Run 5 (Seed: 4)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | never_reached |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 7 | success | missed_grasp |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |

### Run 6 (Seed: 5)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✓ | ✓ | ✓ | ✗ | 0 | 3 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 7 | 0 | missed_grasp | missed_grasp |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |

### Run 7 (Seed: 6)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✗ | ✓ | 1 | 0 | success | success |
| 2 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 5 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 1 | 35 | missed_grasp | never_reached |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 21 | dropped | missed_grasp |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | never_reached |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |

### Run 8 (Seed: 7)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 5 | 0 | dropped | success |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 2 | success | missed_grasp |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 10 | success | never_reached |
| 12 | ✗ | ✓ | ✗ | ✓ | 10 | 0 | missed_grasp | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 14 | ✗ | ✓ | ✗ | ✓ | 33 | 0 | never_reached | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 10 | 0 | missed_grasp | missed_grasp |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✗ | ✓ | ✗ | ✗ | 95 | 1 | never_reached | success |

### Run 9 (Seed: 8)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 2 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | dropped |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 1 | 1 | dropped | missed_grasp |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✓ | ✗ | ✓ | 27 | 0 | missed_grasp | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 14 | 0 | dropped | never_reached |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✓ | ✓ | ✓ | ✗ | 0 | 6 | success | success |

### Run 10 (Seed: 9)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 2 | ✓ | ✗ | ✗ | ✗ | 14 | 7 | success | never_reached |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 6 | success | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 3 | missed_grasp | dropped |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 15 | ✗ | ✗ | ✗ | ✗ | 86 | 0 | missed_grasp | missed_grasp |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 17 | ✓ | ✗ | ✓ | ✗ | 0 | 18 | success | missed_grasp |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 1 | success | dropped |

