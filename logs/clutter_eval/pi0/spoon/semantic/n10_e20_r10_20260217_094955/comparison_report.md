# Paired Evaluation Report

**Generated:** 2026-02-17 17:38:15

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Pi0 |
| Task | `widowx_spoon_on_towel` |
| Category | semantic |
| Num Distractors | 10 |
| Episodes per run | 20 |
| Number of runs | 10 |
| Seeds | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Checkpoint | `/app/open-pi-zero/checkpoints/bridge_beta.pt` |

### Distractors
- `rc_fork_0:0.1`
- `rc_fork_1:0.1`
- `rc_fork_4:0.1`
- `rc_fork_7:0.1`
- `rc_fork_11:0.1`
- `rc_knife_0:0.1`
- `rc_knife_7:0.1`
- `rc_knife_20:0.1`
- `rc_knife_23:0.1`
- `rc_knife_25:0.1`
- `rc_spatula_1:0.1`
- `rc_spatula_4:0.1`
- `rc_spatula_8:0.1`
- `rc_spatula_11:0.1`
- `rc_spatula_3:0.1`
- `rc_scissors_0:0.1`
- `rc_scissors_6:0.1`
- `rc_scissors_12:0.1`
- `rc_scissors_4:0.1`
- `rc_scissors_8:0.1`

## Results by Run

| Run | Seed | Baseline SR | CGVD SR | Δ SR | Baseline h-SR | CGVD h-SR | Δ h-SR |
|-----|------|-------------|---------|------|---------------|-----------|--------|
| 1 | 0 | 40.0% | 80.0% | +40.0% | 40.0% | 80.0% | +40.0% |
| 2 | 1 | 50.0% | 60.0% | +10.0% | 45.0% | 60.0% | +15.0% |
| 3 | 2 | 55.0% | 85.0% | +30.0% | 55.0% | 85.0% | +30.0% |
| 4 | 3 | 35.0% | 60.0% | +25.0% | 25.0% | 60.0% | +35.0% |
| 5 | 4 | 75.0% | 80.0% | +5.0% | 55.0% | 80.0% | +25.0% |
| 6 | 5 | 45.0% | 65.0% | +20.0% | 35.0% | 65.0% | +30.0% |
| 7 | 6 | 50.0% | 60.0% | +10.0% | 50.0% | 60.0% | +10.0% |
| 8 | 7 | 45.0% | 70.0% | +25.0% | 35.0% | 70.0% | +35.0% |
| 9 | 8 | 40.0% | 65.0% | +25.0% | 40.0% | 65.0% | +25.0% |
| 10 | 9 | 60.0% | 85.0% | +25.0% | 45.0% | 85.0% | +40.0% |

## Summary Statistics

### Success Rate (SR)

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 49.5% | 11.1% | 35.0% | 75.0% |
| CGVD | 71.0% | 9.9% | 60.0% | 85.0% |

**Average SR Improvement: +21.5%**

### Hard Success Rate (h-SR) - Success without collisions

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 42.5% | 9.0% | 25.0% | 55.0% |
| CGVD | 71.0% | 9.9% | 60.0% | 85.0% |

**Average h-SR Improvement: +28.5%**

## Collision Analysis

| Method | Episodes with Collision | Collision Rate |
|--------|------------------------|----------------|
| Baseline | 53/200 | 26.5% |
| CGVD | 15/200 | 7.5% |

## Failure Mode Analysis

| Failure Mode | Baseline | CGVD |
|--------------|----------|------|
| success | 99 | 142 |
| never_reached | 63 | 31 |
| missed_grasp | 33 | 23 |
| dropped | 5 | 4 |

## CGVD Latency Analysis

| Component | Mean (s) | Std (s) | Min (s) | Max (s) |
|-----------|----------|---------|---------|--------|
| Total CGVD | 17.779 | 0.058 | 17.603 | 17.987 |
| SAM3 | 16.261 | 0.054 | 16.103 | 16.462 |
| LaMa | 3.229 | 0.038 | 3.176 | 3.408 |

## Per-Episode Details

### Run 1 (Seed: 0)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 41 | 0 | never_reached | never_reached |
| 2 | ✗ | ✓ | ✗ | ✓ | 14 | 0 | never_reached | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 27 | 0 | never_reached | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 51 | 0 | never_reached | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 7 | ✗ | ✓ | ✗ | ✓ | 24 | 0 | never_reached | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |

### Run 2 (Seed: 1)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 54 | 0 | never_reached | missed_grasp |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 24 | 0 | never_reached | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 17 | never_reached | never_reached |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 12 | ✓ | ✗ | ✗ | ✗ | 1 | 0 | success | missed_grasp |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | dropped |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 12 | never_reached | never_reached |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 3 (Seed: 2)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 14 | 0 | never_reached | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 65 | 0 | never_reached | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 40 | 0 | never_reached | success |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | dropped | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 1 | success | never_reached |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |

### Run 4 (Seed: 3)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 11 | 0 | never_reached | missed_grasp |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 55 | 0 | never_reached | success |
| 5 | ✓ | ✓ | ✗ | ✓ | 19 | 0 | success | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✓ | ✗ | ✓ | 2 | 0 | missed_grasp | success |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | dropped |
| 18 | ✓ | ✗ | ✗ | ✗ | 6 | 5 | success | never_reached |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 5 (Seed: 4)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 14 | 0 | never_reached | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✓ | ✗ | ✓ | 10 | 0 | success | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 15 | success | never_reached |
| 10 | ✓ | ✗ | ✗ | ✗ | 19 | 0 | success | missed_grasp |
| 11 | ✓ | ✓ | ✗ | ✓ | 5 | 0 | success | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 2 | never_reached | never_reached |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 17 | ✓ | ✓ | ✗ | ✓ | 11 | 0 | success | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 6 (Seed: 5)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 6 | 0 | never_reached | missed_grasp |
| 2 | ✓ | ✓ | ✗ | ✓ | 5 | 0 | success | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 22 | 0 | never_reached | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 23 | 0 | never_reached | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✓ | ✗ | ✓ | 3 | 0 | success | success |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 31 | success | never_reached |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 14 | never_reached | never_reached |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 2 | missed_grasp | never_reached |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 17 | success | never_reached |
| 17 | ✗ | ✓ | ✗ | ✓ | 3 | 0 | missed_grasp | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 8 | 0 | never_reached | success |

### Run 7 (Seed: 6)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 11 | 0 | missed_grasp | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 51 | 0 | never_reached | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 15 | 0 | never_reached | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 19 | success | never_reached |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 5 | success | never_reached |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 19 | ✗ | ✓ | ✗ | ✓ | 39 | 0 | never_reached | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 55 | 0 | never_reached | missed_grasp |

### Run 8 (Seed: 7)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 7 | 0 | missed_grasp | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✓ | ✗ | ✓ | 2 | 0 | success | success |
| 8 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 9 | ✓ | ✓ | ✗ | ✓ | 61 | 0 | success | success |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 36 | success | never_reached |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 20 | never_reached | never_reached |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 18 | ✗ | ✓ | ✗ | ✓ | 12 | 0 | never_reached | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 8 | 0 | never_reached | missed_grasp |
| 20 | ✗ | ✓ | ✗ | ✓ | 5 | 0 | never_reached | success |

### Run 9 (Seed: 8)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 41 | 0 | never_reached | success |
| 2 | ✗ | ✗ | ✗ | ✗ | 11 | 0 | never_reached | missed_grasp |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 5 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | never_reached |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 12 | 0 | never_reached | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 4 | 0 | never_reached | missed_grasp |
| 19 | ✗ | ✓ | ✗ | ✓ | 62 | 0 | never_reached | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 17 | 0 | never_reached | success |

### Run 10 (Seed: 9)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 33 | 0 | never_reached | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✓ | ✗ | ✓ | 16 | 0 | success | success |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 75 | success | never_reached |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 12 | ✓ | ✓ | ✗ | ✓ | 3 | 0 | success | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 14 | ✓ | ✓ | ✗ | ✓ | 5 | 0 | success | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 16 | 0 | never_reached | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 21 | 0 | never_reached | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 22 | 0 | never_reached | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

