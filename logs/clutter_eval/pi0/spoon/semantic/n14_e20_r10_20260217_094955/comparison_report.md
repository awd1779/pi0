# Paired Evaluation Report

**Generated:** 2026-02-17 19:38:39

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Pi0 |
| Task | `widowx_spoon_on_towel` |
| Category | semantic |
| Num Distractors | 14 |
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
| 1 | 0 | 30.0% | 60.0% | +30.0% | 30.0% | 60.0% | +30.0% |
| 2 | 1 | 40.0% | 60.0% | +20.0% | 40.0% | 60.0% | +20.0% |
| 3 | 2 | 35.0% | 75.0% | +40.0% | 30.0% | 70.0% | +40.0% |
| 4 | 3 | 40.0% | 85.0% | +45.0% | 40.0% | 85.0% | +45.0% |
| 5 | 4 | 45.0% | 80.0% | +35.0% | 40.0% | 80.0% | +40.0% |
| 6 | 5 | 60.0% | 60.0% | +0.0% | 60.0% | 55.0% | -5.0% |
| 7 | 6 | 60.0% | 50.0% | -10.0% | 60.0% | 50.0% | -10.0% |
| 8 | 7 | 50.0% | 85.0% | +35.0% | 45.0% | 85.0% | +40.0% |
| 9 | 8 | 45.0% | 50.0% | +5.0% | 40.0% | 50.0% | +10.0% |
| 10 | 9 | 50.0% | 75.0% | +25.0% | 45.0% | 75.0% | +30.0% |

## Summary Statistics

### Success Rate (SR)

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 45.5% | 9.3% | 30.0% | 60.0% |
| CGVD | 68.0% | 12.9% | 50.0% | 85.0% |

**Average SR Improvement: +22.5%**

### Hard Success Rate (h-SR) - Success without collisions

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 43.0% | 9.8% | 30.0% | 60.0% |
| CGVD | 67.0% | 13.1% | 50.0% | 85.0% |

**Average h-SR Improvement: +24.0%**

## Collision Analysis

| Method | Episodes with Collision | Collision Rate |
|--------|------------------------|----------------|
| Baseline | 58/200 | 29.0% |
| CGVD | 27/200 | 13.5% |

## Failure Mode Analysis

| Failure Mode | Baseline | CGVD |
|--------------|----------|------|
| success | 91 | 136 |
| never_reached | 61 | 37 |
| missed_grasp | 41 | 23 |
| dropped | 7 | 4 |

## CGVD Latency Analysis

| Component | Mean (s) | Std (s) | Min (s) | Max (s) |
|-----------|----------|---------|---------|--------|
| Total CGVD | 17.828 | 0.082 | 17.639 | 18.147 |
| SAM3 | 16.298 | 0.076 | 16.089 | 16.620 |
| LaMa | 3.224 | 0.030 | 3.174 | 3.334 |

## Per-Episode Details

### Run 1 (Seed: 0)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 33 | 0 | never_reached | dropped |
| 2 | ✗ | ✓ | ✗ | ✓ | 22 | 0 | never_reached | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 12 | 0 | never_reached | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 16 | missed_grasp | never_reached |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 3 | 0 | never_reached | missed_grasp |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 2 | 0 | missed_grasp | dropped |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 4 | success | never_reached |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 49 | missed_grasp | dropped |
| 15 | ✗ | ✓ | ✗ | ✓ | 10 | 0 | missed_grasp | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |

### Run 2 (Seed: 1)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 70 | dropped | never_reached |
| 2 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | never_reached | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 1 | 0 | missed_grasp | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 24 | 19 | never_reached | never_reached |
| 7 | ✗ | ✗ | ✗ | ✗ | 19 | 0 | never_reached | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 10 | ✗ | ✓ | ✗ | ✓ | 11 | 0 | never_reached | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 19 | 0 | never_reached | missed_grasp |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 17 | success | never_reached |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 9 | success | never_reached |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 3 (Seed: 2)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✗ | 60 | 2 | never_reached | success |
| 2 | ✓ | ✗ | ✓ | ✗ | 0 | 5 | success | never_reached |
| 3 | ✗ | ✓ | ✗ | ✓ | 32 | 0 | never_reached | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 2 | 0 | never_reached | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 19 | 0 | never_reached | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 60 | 0 | never_reached | success |
| 7 | ✗ | ✓ | ✗ | ✓ | 34 | 0 | never_reached | success |
| 8 | ✗ | ✓ | ✗ | ✓ | 16 | 0 | never_reached | success |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✓ | ✗ | ✓ | 3 | 0 | never_reached | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 19 | ✓ | ✓ | ✗ | ✓ | 3 | 0 | success | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 3 | never_reached | missed_grasp |

### Run 4 (Seed: 3)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 57 | 0 | never_reached | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | never_reached | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 21 | 0 | never_reached | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 8 | 0 | missed_grasp | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 21 | 0 | never_reached | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 16 | 0 | missed_grasp | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 5 (Seed: 4)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 133 | 0 | never_reached | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 23 | 0 | never_reached | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 22 | 0 | never_reached | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 19 | 0 | never_reached | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 44 | 0 | never_reached | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 18 | missed_grasp | never_reached |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✓ | ✗ | ✗ | ✗ | 24 | 0 | success | never_reached |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 111 | missed_grasp | never_reached |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 6 (Seed: 5)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 7 | 0 | never_reached | never_reached |
| 2 | ✗ | ✓ | ✗ | ✓ | 54 | 0 | never_reached | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 8 | 0 | never_reached | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 7 | ✗ | ✓ | ✗ | ✓ | 12 | 0 | missed_grasp | success |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 47 | success | never_reached |
| 9 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 17 | success | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✓ | ✓ | ✗ | 0 | 18 | success | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 19 | never_reached | never_reached |
| 20 | ✗ | ✓ | ✗ | ✓ | 45 | 0 | never_reached | success |

### Run 7 (Seed: 6)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 2 | ✗ | ✓ | ✗ | ✓ | 14 | 0 | dropped | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 10 | success | never_reached |
| 8 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 23 | never_reached | never_reached |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 19 | success | never_reached |
| 19 | ✗ | ✗ | ✗ | ✗ | 25 | 14 | never_reached | never_reached |
| 20 | ✗ | ✓ | ✗ | ✓ | 13 | 0 | never_reached | success |

### Run 8 (Seed: 7)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 21 | 0 | never_reached | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 7 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 14 | ✓ | ✓ | ✗ | ✓ | 5 | 0 | success | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 41 | 81 | never_reached | never_reached |
| 19 | ✗ | ✗ | ✗ | ✗ | 34 | 0 | never_reached | missed_grasp |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |

### Run 9 (Seed: 8)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 26 | 0 | never_reached | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 1 | 0 | missed_grasp | never_reached |
| 5 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 6 | ✓ | ✗ | ✗ | ✗ | 1 | 0 | success | missed_grasp |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 20 | missed_grasp | never_reached |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 10 | success | never_reached |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 22 | missed_grasp | never_reached |
| 17 | ✗ | ✓ | ✗ | ✓ | 15 | 0 | never_reached | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 30 | 0 | never_reached | missed_grasp |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 13 | success | never_reached |
| 20 | ✗ | ✓ | ✗ | ✓ | 45 | 0 | never_reached | success |

### Run 10 (Seed: 9)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 24 | 0 | never_reached | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 121 | 0 | never_reached | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✗ | ✓ | 4 | 0 | success | success |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 28 | success | never_reached |
| 7 | ✗ | ✓ | ✗ | ✓ | 10 | 0 | dropped | success |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 19 | success | never_reached |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 56 | 0 | never_reached | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 41 | 0 | never_reached | missed_grasp |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 31 | 0 | never_reached | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

