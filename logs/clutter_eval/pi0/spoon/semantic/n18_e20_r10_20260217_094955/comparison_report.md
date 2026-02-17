# Paired Evaluation Report

**Generated:** 2026-02-17 21:40:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Pi0 |
| Task | `widowx_spoon_on_towel` |
| Category | semantic |
| Num Distractors | 18 |
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
| 1 | 0 | 45.0% | 60.0% | +15.0% | 45.0% | 60.0% | +15.0% |
| 2 | 1 | 45.0% | 70.0% | +25.0% | 45.0% | 70.0% | +25.0% |
| 3 | 2 | 35.0% | 70.0% | +35.0% | 30.0% | 70.0% | +40.0% |
| 4 | 3 | 35.0% | 60.0% | +25.0% | 30.0% | 60.0% | +30.0% |
| 5 | 4 | 50.0% | 60.0% | +10.0% | 40.0% | 60.0% | +20.0% |
| 6 | 5 | 45.0% | 75.0% | +30.0% | 40.0% | 75.0% | +35.0% |
| 7 | 6 | 50.0% | 60.0% | +10.0% | 45.0% | 55.0% | +10.0% |
| 8 | 7 | 50.0% | 75.0% | +25.0% | 45.0% | 75.0% | +30.0% |
| 9 | 8 | 50.0% | 65.0% | +15.0% | 40.0% | 65.0% | +25.0% |
| 10 | 9 | 45.0% | 70.0% | +25.0% | 35.0% | 70.0% | +35.0% |

## Summary Statistics

### Success Rate (SR)

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 45.0% | 5.5% | 35.0% | 50.0% |
| CGVD | 66.5% | 5.9% | 60.0% | 75.0% |

**Average SR Improvement: +21.5%**

### Hard Success Rate (h-SR) - Success without collisions

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 39.5% | 5.7% | 30.0% | 45.0% |
| CGVD | 66.0% | 6.6% | 55.0% | 75.0% |

**Average h-SR Improvement: +26.5%**

## Collision Analysis

| Method | Episodes with Collision | Collision Rate |
|--------|------------------------|----------------|
| Baseline | 58/200 | 29.0% |
| CGVD | 26/200 | 13.0% |

## Failure Mode Analysis

| Failure Mode | Baseline | CGVD |
|--------------|----------|------|
| success | 90 | 133 |
| never_reached | 64 | 29 |
| missed_grasp | 40 | 34 |
| dropped | 6 | 4 |

## CGVD Latency Analysis

| Component | Mean (s) | Std (s) | Min (s) | Max (s) |
|-----------|----------|---------|---------|--------|
| Total CGVD | 17.796 | 0.065 | 17.644 | 17.988 |
| SAM3 | 16.256 | 0.058 | 16.087 | 16.463 |
| LaMa | 3.218 | 0.043 | 3.171 | 3.670 |

## Per-Episode Details

### Run 1 (Seed: 0)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 29 | 0 | never_reached | dropped |
| 2 | ✗ | ✓ | ✗ | ✓ | 24 | 0 | never_reached | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 74 | 0 | never_reached | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 42 | 0 | never_reached | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 16 | success | never_reached |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 30 | 25 | missed_grasp | never_reached |
| 11 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | never_reached | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 38 | never_reached | never_reached |

### Run 2 (Seed: 1)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 5 | 0 | never_reached | missed_grasp |
| 2 | ✗ | ✓ | ✗ | ✓ | 34 | 0 | never_reached | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 13 | 0 | never_reached | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 15 | 0 | never_reached | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 68 | 0 | never_reached | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 39 | missed_grasp | never_reached |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 23 | missed_grasp | never_reached |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 17 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 3 (Seed: 2)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 12 | 0 | never_reached | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 89 | 0 | never_reached | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 18 | 0 | never_reached | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 14 | 206 | never_reached | never_reached |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 8 | ✗ | ✓ | ✗ | ✓ | 69 | 0 | never_reached | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 12 | 0 | dropped | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 97 | never_reached | never_reached |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 12 | 0 | missed_grasp | dropped |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✗ | ✗ | ✗ | 1 | 0 | success | missed_grasp |

### Run 4 (Seed: 3)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 12 | 0 | never_reached | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 46 | 0 | never_reached | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 62 | 0 | never_reached | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 9 | ✗ | ✗ | ✗ | ✗ | 4 | 0 | missed_grasp | missed_grasp |
| 10 | ✗ | ✗ | ✗ | ✗ | 16 | 0 | never_reached | missed_grasp |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 18 | ✓ | ✓ | ✗ | ✓ | 46 | 0 | success | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |

### Run 5 (Seed: 4)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 16 | 0 | never_reached | success |
| 3 | ✓ | ✗ | ✗ | ✗ | 20 | 0 | success | missed_grasp |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 26 | 68 | never_reached | never_reached |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 7 | never_reached | never_reached |
| 7 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 10 | ✓ | ✗ | ✗ | ✗ | 12 | 0 | success | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 16 | success | never_reached |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 6 (Seed: 5)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 25 | 0 | never_reached | missed_grasp |
| 2 | ✗ | ✓ | ✗ | ✓ | 161 | 0 | never_reached | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 2 | 0 | missed_grasp | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 58 | missed_grasp | never_reached |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 18 | success | never_reached |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✓ | ✗ | ✓ | 10 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 13 | 0 | never_reached | success |

### Run 7 (Seed: 6)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✗ | ✗ | 2 | 3 | success | never_reached |
| 2 | ✗ | ✓ | ✗ | ✗ | 0 | 5 | missed_grasp | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 3 | 0 | never_reached | missed_grasp |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 39 | success | never_reached |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 1 | never_reached | missed_grasp |
| 7 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 8 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 34 | success | never_reached |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 12 | success | never_reached |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 9 | 0 | never_reached | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 27 | 0 | never_reached | missed_grasp |

### Run 8 (Seed: 7)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 28 | 0 | never_reached | missed_grasp |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✗ | ✗ | ✗ | 1 | 0 | success | never_reached |
| 18 | ✗ | ✓ | ✗ | ✓ | 13 | 0 | missed_grasp | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 7 | 0 | never_reached | missed_grasp |
| 20 | ✗ | ✓ | ✗ | ✓ | 10 | 0 | never_reached | success |

### Run 9 (Seed: 8)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 50 | 0 | never_reached | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 10 | 0 | never_reached | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✗ | ✓ | ✗ | 0 | 50 | success | never_reached |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 159 | never_reached | never_reached |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✗ | ✗ | ✗ | 10 | 34 | success | never_reached |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 2 | success | never_reached |
| 10 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 13 | ✓ | ✓ | ✗ | ✓ | 8 | 0 | success | success |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 80 | success | never_reached |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 56 | success | never_reached |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 22 | 0 | never_reached | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 60 | 0 | never_reached | missed_grasp |
| 19 | ✗ | ✓ | ✗ | ✓ | 26 | 0 | never_reached | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |

### Run 10 (Seed: 9)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 3 | 0 | missed_grasp | missed_grasp |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 5 | ✓ | ✗ | ✗ | ✗ | 22 | 0 | success | missed_grasp |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 10 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 10 | never_reached | missed_grasp |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✓ | ✗ | ✓ | 7 | 0 | success | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 103 | 0 | never_reached | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 8 | 16 | never_reached | missed_grasp |
| 18 | ✗ | ✓ | ✗ | ✓ | 51 | 0 | never_reached | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 12 | 0 | never_reached | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 62 | 0 | never_reached | success |

