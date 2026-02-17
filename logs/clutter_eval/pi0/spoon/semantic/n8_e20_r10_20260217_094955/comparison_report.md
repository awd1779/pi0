# Paired Evaluation Report

**Generated:** 2026-02-17 15:39:16

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Pi0 |
| Task | `widowx_spoon_on_towel` |
| Category | semantic |
| Num Distractors | 8 |
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
| 1 | 0 | 35.0% | 80.0% | +45.0% | 35.0% | 80.0% | +45.0% |
| 2 | 1 | 45.0% | 70.0% | +25.0% | 40.0% | 70.0% | +30.0% |
| 3 | 2 | 10.0% | 60.0% | +50.0% | 10.0% | 60.0% | +50.0% |
| 4 | 3 | 70.0% | 65.0% | -5.0% | 70.0% | 60.0% | -10.0% |
| 5 | 4 | 50.0% | 80.0% | +30.0% | 50.0% | 80.0% | +30.0% |
| 6 | 5 | 50.0% | 65.0% | +15.0% | 40.0% | 60.0% | +20.0% |
| 7 | 6 | 50.0% | 70.0% | +20.0% | 50.0% | 70.0% | +20.0% |
| 8 | 7 | 45.0% | 70.0% | +25.0% | 45.0% | 70.0% | +25.0% |
| 9 | 8 | 50.0% | 65.0% | +15.0% | 50.0% | 65.0% | +15.0% |
| 10 | 9 | 45.0% | 55.0% | +10.0% | 40.0% | 55.0% | +15.0% |

## Summary Statistics

### Success Rate (SR)

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 45.0% | 14.3% | 10.0% | 70.0% |
| CGVD | 68.0% | 7.5% | 55.0% | 80.0% |

**Average SR Improvement: +23.0%**

### Hard Success Rate (h-SR) - Success without collisions

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 43.0% | 14.4% | 10.0% | 70.0% |
| CGVD | 67.0% | 8.1% | 55.0% | 80.0% |

**Average h-SR Improvement: +24.0%**

## Collision Analysis

| Method | Episodes with Collision | Collision Rate |
|--------|------------------------|----------------|
| Baseline | 50/200 | 25.0% |
| CGVD | 19/200 | 9.5% |

## Failure Mode Analysis

| Failure Mode | Baseline | CGVD |
|--------------|----------|------|
| success | 90 | 136 |
| never_reached | 64 | 28 |
| missed_grasp | 43 | 30 |
| dropped | 3 | 6 |

## CGVD Latency Analysis

| Component | Mean (s) | Std (s) | Min (s) | Max (s) |
|-----------|----------|---------|---------|--------|
| Total CGVD | 17.755 | 0.061 | 17.595 | 17.930 |
| SAM3 | 16.249 | 0.056 | 16.083 | 16.415 |
| LaMa | 3.229 | 0.037 | 3.176 | 3.348 |

## Per-Episode Details

### Run 1 (Seed: 0)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 96 | 0 | never_reached | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 14 | 0 | never_reached | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 30 | 0 | missed_grasp | success |
| 4 | ✓ | ✗ | ✓ | ✗ | 0 | 5 | success | never_reached |
| 5 | ✗ | ✓ | ✗ | ✓ | 3 | 0 | never_reached | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 20 | 0 | never_reached | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 6 | 0 | missed_grasp | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 6 | 0 | never_reached | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 22 | 0 | never_reached | missed_grasp |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 4 | success | never_reached |

### Run 2 (Seed: 1)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 13 | 0 | never_reached | missed_grasp |
| 2 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | never_reached | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 6 | 0 | missed_grasp | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 1 | 0 | never_reached | success |
| 6 | ✓ | ✓ | ✗ | ✓ | 5 | 0 | success | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 15 | success | never_reached |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 3 (Seed: 2)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 34 | 0 | never_reached | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 12 | 0 | never_reached | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 27 | 0 | never_reached | never_reached |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 41 | 0 | never_reached | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 11 | 0 | never_reached | never_reached |
| 7 | ✗ | ✗ | ✗ | ✗ | 24 | 9 | never_reached | never_reached |
| 8 | ✗ | ✓ | ✗ | ✓ | 11 | 0 | never_reached | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 15 | 0 | never_reached | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 7 | never_reached | never_reached |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 59 | 0 | never_reached | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 14 | 0 | missed_grasp | success |

### Run 4 (Seed: 3)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 2 | ✗ | ✓ | ✗ | ✓ | 20 | 0 | never_reached | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 50 | 0 | never_reached | never_reached |
| 4 | ✓ | ✓ | ✓ | ✗ | 0 | 1 | success | success |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 23 | success | never_reached |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 14 | 0 | missed_grasp | never_reached |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 11 | success | never_reached |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 5 (Seed: 4)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 12 | 0 | never_reached | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 19 | 0 | never_reached | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | missed_grasp | success |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 9 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 11 | success | never_reached |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |

### Run 6 (Seed: 5)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 19 | 0 | never_reached | missed_grasp |
| 2 | ✓ | ✓ | ✗ | ✓ | 11 | 0 | success | success |
| 3 | ✓ | ✓ | ✗ | ✓ | 6 | 0 | success | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 9 | ✗ | ✗ | ✗ | ✗ | 8 | 0 | missed_grasp | dropped |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 18 | success | never_reached |
| 13 | ✗ | ✓ | ✗ | ✗ | 0 | 8 | never_reached | success |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 20 | never_reached | never_reached |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 20 | ✗ | ✓ | ✗ | ✓ | 26 | 0 | never_reached | success |

### Run 7 (Seed: 6)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 7 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 8 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 14 | dropped | never_reached |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 12 | missed_grasp | never_reached |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 19 | ✗ | ✓ | ✗ | ✓ | 20 | 0 | never_reached | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 15 | 0 | never_reached | missed_grasp |

### Run 8 (Seed: 7)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 12 | 0 | never_reached | missed_grasp |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 6 | missed_grasp | missed_grasp |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 18 | ✗ | ✓ | ✗ | ✓ | 13 | 0 | never_reached | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 44 | 0 | never_reached | missed_grasp |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 9 (Seed: 8)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 39 | 0 | never_reached | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 27 | 0 | never_reached | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 4 | ✗ | ✗ | ✗ | ✗ | 8 | 0 | missed_grasp | missed_grasp |
| 5 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 5 | success | missed_grasp |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 9 | missed_grasp | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 17 | ✗ | ✓ | ✗ | ✓ | 38 | 0 | never_reached | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | never_reached | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 32 | 0 | never_reached | success |

### Run 10 (Seed: 9)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 23 | 0 | never_reached | missed_grasp |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 11 | 0 | never_reached | missed_grasp |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✓ | ✓ | ✗ | ✓ | 9 | 0 | success | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 2 | success | never_reached |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 16 | ✗ | ✓ | ✗ | ✓ | 44 | 0 | never_reached | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 18 | ✗ | ✓ | ✗ | ✓ | 26 | 0 | never_reached | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 8 | never_reached | never_reached |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |

