# Paired Evaluation Report

**Generated:** 2026-02-17 19:35:15

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Pi0 |
| Task | `widowx_carrot_on_plate` |
| Category | control |
| Num Distractors | 14 |
| Episodes per run | 20 |
| Number of runs | 10 |
| Seeds | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Checkpoint | `/app/open-pi-zero/checkpoints/bridge_beta.pt` |

### Distractors
- `rc_fork_0:0.07`
- `rc_fork_1:0.07`
- `rc_fork_2:0.07`
- `rc_knife_0:0.07`
- `rc_knife_1:0.07`
- `rc_knife_2:0.07`
- `rc_spoon_0:0.07`
- `rc_spoon_1:0.07`
- `rc_spoon_2:0.07`
- `rc_spatula_0:0.07`
- `rc_spatula_1:0.07`
- `rc_ladle_0:0.07`
- `rc_ladle_1:0.07`
- `rc_scissors_0:0.07`
- `rc_bowl_0:0.07`
- `rc_bowl_1:0.07`
- `rc_mug_0:0.07`
- `rc_mug_1:0.07`
- `rc_plate_0:0.07`
- `rc_plate_1:0.07`

## Results by Run

| Run | Seed | Baseline SR | CGVD SR | Δ SR | Baseline h-SR | CGVD h-SR | Δ h-SR |
|-----|------|-------------|---------|------|---------------|-----------|--------|
| 1 | 0 | 35.0% | 55.0% | +20.0% | 35.0% | 55.0% | +20.0% |
| 2 | 1 | 30.0% | 60.0% | +30.0% | 30.0% | 55.0% | +25.0% |
| 3 | 2 | 55.0% | 45.0% | -10.0% | 55.0% | 45.0% | -10.0% |
| 4 | 3 | 55.0% | 50.0% | -5.0% | 55.0% | 50.0% | -5.0% |
| 5 | 4 | 65.0% | 50.0% | -15.0% | 65.0% | 50.0% | -15.0% |
| 6 | 5 | 45.0% | 55.0% | +10.0% | 45.0% | 50.0% | +5.0% |
| 7 | 6 | 55.0% | 60.0% | +5.0% | 55.0% | 60.0% | +5.0% |
| 8 | 7 | 65.0% | 45.0% | -20.0% | 65.0% | 45.0% | -20.0% |
| 9 | 8 | 70.0% | 50.0% | -20.0% | 60.0% | 50.0% | -10.0% |
| 10 | 9 | 60.0% | 50.0% | -10.0% | 55.0% | 45.0% | -10.0% |

## Summary Statistics

### Success Rate (SR)

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 53.5% | 12.5% | 30.0% | 70.0% |
| CGVD | 52.0% | 5.1% | 45.0% | 60.0% |

**Average SR Improvement: -1.5%**

### Hard Success Rate (h-SR) - Success without collisions

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 52.0% | 11.2% | 30.0% | 65.0% |
| CGVD | 50.5% | 4.7% | 45.0% | 60.0% |

**Average h-SR Improvement: -1.5%**

## Collision Analysis

| Method | Episodes with Collision | Collision Rate |
|--------|------------------------|----------------|
| Baseline | 23/200 | 11.5% |
| CGVD | 8/200 | 4.0% |

## Failure Mode Analysis

| Failure Mode | Baseline | CGVD |
|--------------|----------|------|
| success | 107 | 104 |
| never_reached | 6 | 20 |
| missed_grasp | 74 | 59 |
| dropped | 13 | 17 |

## CGVD Latency Analysis

| Component | Mean (s) | Std (s) | Min (s) | Max (s) |
|-----------|----------|---------|---------|--------|
| Total CGVD | 17.814 | 0.133 | 17.542 | 18.603 |
| SAM3 | 16.077 | 0.111 | 15.785 | 16.726 |
| LaMa | 3.498 | 0.032 | 3.433 | 3.642 |

## Per-Episode Details

### Run 1 (Seed: 0)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 7 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | dropped | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 7 | dropped | never_reached |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 8 | 0 | missed_grasp | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 21 | 0 | missed_grasp | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 27 | 0 | missed_grasp | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |

### Run 2 (Seed: 1)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✗ | 0 | 14 | missed_grasp | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 37 | 0 | missed_grasp | dropped |
| 15 | ✗ | ✗ | ✗ | ✗ | 3 | 17 | missed_grasp | missed_grasp |
| 16 | ✗ | ✓ | ✗ | ✓ | 17 | 0 | missed_grasp | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | missed_grasp | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 3 (Seed: 2)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 4 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 11 | 1 | missed_grasp | missed_grasp |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 14 | 0 | dropped | missed_grasp |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 4 (Seed: 3)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 12 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | missed_grasp | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 14 | ✗ | ✓ | ✗ | ✓ | 36 | 0 | missed_grasp | success |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |

### Run 5 (Seed: 4)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 1 | 0 | missed_grasp | dropped |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 3 | missed_grasp | missed_grasp |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 2 | 0 | missed_grasp | never_reached |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |

### Run 6 (Seed: 5)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 0 | 6 | missed_grasp | missed_grasp |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 11 | 0 | missed_grasp | success |
| 12 | ✓ | ✓ | ✓ | ✗ | 0 | 3 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |

### Run 7 (Seed: 6)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 9 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 22 | 0 | missed_grasp | never_reached |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 8 (Seed: 7)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 15 | 0 | missed_grasp | never_reached |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | dropped |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 9 (Seed: 8)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 4 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✗ | ✗ | ✗ | 7 | 0 | success | missed_grasp |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 51 | 0 | missed_grasp | never_reached |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 17 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 18 | ✓ | ✗ | ✗ | ✗ | 5 | 0 | success | missed_grasp |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |

### Run 10 (Seed: 9)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✗ | ✓ | 2 | 0 | success | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 7 | ✗ | ✗ | ✗ | ✗ | 24 | 0 | dropped | missed_grasp |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✗ | 0 | 1 | missed_grasp | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 2 | 0 | missed_grasp | dropped |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

