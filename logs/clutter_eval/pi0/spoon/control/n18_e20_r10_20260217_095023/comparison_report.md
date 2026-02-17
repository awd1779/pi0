# Paired Evaluation Report

**Generated:** 2026-02-17 22:08:12

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Pi0 |
| Task | `widowx_spoon_on_towel` |
| Category | control |
| Num Distractors | 18 |
| Episodes per run | 20 |
| Number of runs | 10 |
| Seeds | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Checkpoint | `/app/open-pi-zero/checkpoints/bridge_beta.pt` |

### Distractors
- `rc_apple_7:0.06`
- `rc_banana_20:0.06`
- `rc_orange_1:0.06`
- `rc_lemon_4:0.06`
- `rc_mango_12:0.06`
- `rc_tomato_6:0.06`
- `rc_potato_5:0.06`
- `rc_cucumber_3:0.06`
- `rc_corn_0:0.06`
- `rc_bowl_1:0.06`
- `rc_mug_15:0.06`
- `rc_plate_1:0.06`
- `rc_bread_11:0.06`
- `rc_egg_6:0.06`
- `rc_chocolate_7:0.06`
- `rc_canned_food_3:0.06`
- `rc_sponge_0:0.06`
- `rc_candle_1:0.06`
- `rc_wine_11:0.06`
- `rc_milk_5:0.06`

## Results by Run

| Run | Seed | Baseline SR | CGVD SR | Δ SR | Baseline h-SR | CGVD h-SR | Δ h-SR |
|-----|------|-------------|---------|------|---------------|-----------|--------|
| 1 | 0 | 65.0% | 85.0% | +20.0% | 55.0% | 80.0% | +25.0% |
| 2 | 1 | 65.0% | 75.0% | +10.0% | 55.0% | 60.0% | +5.0% |
| 3 | 2 | 65.0% | 40.0% | -25.0% | 60.0% | 40.0% | -20.0% |
| 4 | 3 | 50.0% | 65.0% | +15.0% | 45.0% | 55.0% | +10.0% |
| 5 | 4 | 50.0% | 55.0% | +5.0% | 45.0% | 45.0% | +0.0% |
| 6 | 5 | 55.0% | 60.0% | +5.0% | 55.0% | 55.0% | +0.0% |
| 7 | 6 | 60.0% | 60.0% | +0.0% | 60.0% | 50.0% | -10.0% |
| 8 | 7 | 55.0% | 60.0% | +5.0% | 50.0% | 55.0% | +5.0% |
| 9 | 8 | 45.0% | 40.0% | -5.0% | 40.0% | 30.0% | -10.0% |
| 10 | 9 | 65.0% | 65.0% | +0.0% | 50.0% | 55.0% | +5.0% |

## Summary Statistics

### Success Rate (SR)

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 57.5% | 7.2% | 45.0% | 65.0% |
| CGVD | 60.5% | 13.1% | 40.0% | 85.0% |

**Average SR Improvement: +3.0%**

### Hard Success Rate (h-SR) - Success without collisions

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 51.5% | 6.3% | 40.0% | 60.0% |
| CGVD | 52.5% | 12.5% | 30.0% | 80.0% |

**Average h-SR Improvement: +1.0%**

## Collision Analysis

| Method | Episodes with Collision | Collision Rate |
|--------|------------------------|----------------|
| Baseline | 51/200 | 25.5% |
| CGVD | 43/200 | 21.5% |

## Failure Mode Analysis

| Failure Mode | Baseline | CGVD |
|--------------|----------|------|
| success | 115 | 121 |
| never_reached | 29 | 26 |
| missed_grasp | 50 | 46 |
| dropped | 6 | 7 |

## CGVD Latency Analysis

| Component | Mean (s) | Std (s) | Min (s) | Max (s) |
|-----------|----------|---------|---------|--------|
| Total CGVD | 18.638 | 0.077 | 18.410 | 18.873 |
| SAM3 | 16.340 | 0.065 | 16.135 | 16.531 |
| LaMa | 3.519 | 0.055 | 3.457 | 4.204 |

## Per-Episode Details

### Run 1 (Seed: 0)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 3 | ✗ | ✗ | ✗ | ✗ | 6 | 0 | missed_grasp | never_reached |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✓ | ✗ | ✓ | 5 | 0 | success | success |
| 9 | ✓ | ✓ | ✗ | ✓ | 2 | 0 | success | success |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✗ | 0 | 5 | success | success |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |

### Run 2 (Seed: 1)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 75 | 1 | never_reached | missed_grasp |
| 2 | ✓ | ✓ | ✓ | ✗ | 0 | 2 | success | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✗ | ✗ | ✗ | 13 | 10 | success | missed_grasp |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 14 | 0 | missed_grasp | missed_grasp |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 166 | 0 | never_reached | never_reached |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 18 | ✓ | ✓ | ✗ | ✗ | 5 | 5 | success | success |
| 19 | ✗ | ✓ | ✗ | ✗ | 0 | 8 | missed_grasp | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 3 (Seed: 2)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 2 | ✗ | ✗ | ✗ | ✗ | 2 | 0 | missed_grasp | missed_grasp |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 13 | success | missed_grasp |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 32 | success | never_reached |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 8 | success | missed_grasp |
| 10 | ✗ | ✗ | ✗ | ✗ | 20 | 0 | never_reached | missed_grasp |
| 11 | ✓ | ✓ | ✗ | ✓ | 48 | 0 | success | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✓ | ✗ | ✓ | 2 | 0 | dropped | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 150 | never_reached | never_reached |
| 17 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 18 | ✗ | ✗ | ✗ | ✗ | 6 | 2 | dropped | missed_grasp |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |

### Run 4 (Seed: 3)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✗ | 0 | 1 | missed_grasp | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 4 | 0 | missed_grasp | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✓ | ✗ | ✗ | 13 | 21 | success | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 6 | ✗ | ✓ | ✗ | ✓ | 11 | 0 | missed_grasp | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 31 | 0 | missed_grasp | never_reached |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 12 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 10 | 0 | missed_grasp | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 2 | never_reached | missed_grasp |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 19 | ✗ | ✗ | ✗ | ✗ | 1 | 5 | missed_grasp | missed_grasp |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |

### Run 5 (Seed: 4)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 17 | 38 | never_reached | never_reached |
| 2 | ✓ | ✓ | ✗ | ✓ | 3 | 0 | success | success |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 3 | success | missed_grasp |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 48 | 0 | never_reached | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 8 | 31 | never_reached | missed_grasp |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 18 | success | never_reached |
| 8 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 3 | success | never_reached |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 11 | ✗ | ✓ | ✗ | ✗ | 74 | 2 | never_reached | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 15 | ✓ | ✓ | ✓ | ✗ | 0 | 7 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 11 | 0 | missed_grasp | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 11 | never_reached | missed_grasp |

### Run 6 (Seed: 5)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 3 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 4 | ✓ | ✓ | ✓ | ✗ | 0 | 1 | success | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 20 | missed_grasp | dropped |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 29 | 10 | missed_grasp | missed_grasp |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 58 | missed_grasp | never_reached |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 12 | 0 | never_reached | success |

### Run 7 (Seed: 6)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 2 | ✗ | ✗ | ✗ | ✗ | 0 | 12 | missed_grasp | missed_grasp |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 14 | missed_grasp | never_reached |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✓ | ✗ | ✗ | 5 | 21 | dropped | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 55 | 0 | missed_grasp | missed_grasp |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✗ | 0 | 19 | missed_grasp | success |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 20 | ✗ | ✗ | ✗ | ✗ | 11 | 23 | missed_grasp | never_reached |

### Run 8 (Seed: 7)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 29 | 0 | missed_grasp | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 7 | ✗ | ✓ | ✗ | ✓ | 33 | 0 | missed_grasp | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 23 | 11 | missed_grasp | never_reached |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 14 | 0 | missed_grasp | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 18 | ✓ | ✗ | ✗ | ✗ | 20 | 2 | success | missed_grasp |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 8 | success | missed_grasp |
| 20 | ✗ | ✓ | ✗ | ✗ | 37 | 2 | never_reached | success |

### Run 9 (Seed: 8)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 39 | 0 | never_reached | success |
| 2 | ✗ | ✗ | ✗ | ✗ | 94 | 7 | never_reached | missed_grasp |
| 3 | ✗ | ✓ | ✗ | ✗ | 57 | 7 | never_reached | success |
| 4 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 5 | ✓ | ✓ | ✓ | ✗ | 0 | 1 | success | success |
| 6 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 7 | ✓ | ✗ | ✓ | ✗ | 0 | 2 | success | never_reached |
| 8 | ✗ | ✓ | ✗ | ✓ | 3 | 0 | missed_grasp | success |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 11 | ✓ | ✓ | ✗ | ✓ | 3 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 14 | 0 | missed_grasp | missed_grasp |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 17 | ✗ | ✓ | ✗ | ✓ | 49 | 0 | never_reached | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 2 | 0 | never_reached | never_reached |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |

### Run 10 (Seed: 9)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 4 | missed_grasp | missed_grasp |
| 2 | ✗ | ✓ | ✗ | ✓ | 8 | 0 | missed_grasp | success |
| 3 | ✗ | ✓ | ✗ | ✗ | 58 | 1 | never_reached | success |
| 4 | ✓ | ✓ | ✗ | ✓ | 6 | 0 | success | success |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✓ | ✗ | ✗ | 22 | 2 | missed_grasp | success |
| 12 | ✓ | ✗ | ✗ | ✗ | 12 | 0 | success | missed_grasp |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 14 | ✗ | ✓ | ✗ | ✓ | 6 | 0 | missed_grasp | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | never_reached |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✗ | ✗ | ✗ | 7 | 0 | success | missed_grasp |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

