# Paired Evaluation Report

**Generated:** 2026-02-17 20:00:04

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Pi0 |
| Task | `widowx_spoon_on_towel` |
| Category | control |
| Num Distractors | 14 |
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
| 1 | 0 | 55.0% | 70.0% | +15.0% | 55.0% | 70.0% | +15.0% |
| 2 | 1 | 50.0% | 70.0% | +20.0% | 45.0% | 65.0% | +20.0% |
| 3 | 2 | 65.0% | 75.0% | +10.0% | 60.0% | 65.0% | +5.0% |
| 4 | 3 | 50.0% | 70.0% | +20.0% | 50.0% | 65.0% | +15.0% |
| 5 | 4 | 65.0% | 85.0% | +20.0% | 60.0% | 85.0% | +25.0% |
| 6 | 5 | 65.0% | 60.0% | -5.0% | 65.0% | 60.0% | -5.0% |
| 7 | 6 | 50.0% | 60.0% | +10.0% | 50.0% | 60.0% | +10.0% |
| 8 | 7 | 60.0% | 70.0% | +10.0% | 50.0% | 65.0% | +15.0% |
| 9 | 8 | 70.0% | 60.0% | -10.0% | 60.0% | 50.0% | -10.0% |
| 10 | 9 | 50.0% | 80.0% | +30.0% | 40.0% | 80.0% | +40.0% |

## Summary Statistics

### Success Rate (SR)

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 58.0% | 7.5% | 50.0% | 70.0% |
| CGVD | 70.0% | 8.1% | 60.0% | 85.0% |

**Average SR Improvement: +12.0%**

### Hard Success Rate (h-SR) - Success without collisions

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | 53.5% | 7.4% | 40.0% | 65.0% |
| CGVD | 66.5% | 9.5% | 50.0% | 85.0% |

**Average h-SR Improvement: +13.0%**

## Collision Analysis

| Method | Episodes with Collision | Collision Rate |
|--------|------------------------|----------------|
| Baseline | 33/200 | 16.5% |
| CGVD | 20/200 | 10.0% |

## Failure Mode Analysis

| Failure Mode | Baseline | CGVD |
|--------------|----------|------|
| success | 116 | 140 |
| never_reached | 44 | 20 |
| missed_grasp | 36 | 35 |
| dropped | 4 | 5 |

## CGVD Latency Analysis

| Component | Mean (s) | Std (s) | Min (s) | Max (s) |
|-----------|----------|---------|---------|--------|
| Total CGVD | 18.442 | 0.094 | 18.163 | 18.816 |
| SAM3 | 16.371 | 0.085 | 16.098 | 16.634 |
| LaMa | 3.522 | 0.029 | 3.453 | 3.664 |

## Per-Episode Details

### Run 1 (Seed: 0)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 79 | 212 | never_reached | never_reached |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 5 | 0 | never_reached | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 22 | 0 | never_reached | missed_grasp |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | dropped | missed_grasp |
| 11 | ✗ | ✓ | ✗ | ✓ | 5 | 0 | never_reached | success |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |

### Run 2 (Seed: 1)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 34 | 0 | never_reached | missed_grasp |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 51 | 0 | never_reached | success |
| 5 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 6 | ✓ | ✓ | ✗ | ✗ | 71 | 41 | success | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 15 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 20 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |

### Run 3 (Seed: 2)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✗ | 55 | 1 | never_reached | success |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✗ | ✓ | ✗ | ✗ | 0 | 30 | missed_grasp | success |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 12 | ✓ | ✓ | ✗ | ✓ | 5 | 0 | success | success |
| 13 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |

### Run 4 (Seed: 3)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 2 | ✗ | ✗ | ✗ | ✗ | 17 | 8 | never_reached | never_reached |
| 3 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 8 | ✓ | ✗ | ✓ | ✗ | 0 | 54 | success | never_reached |
| 9 | ✗ | ✓ | ✗ | ✓ | 15 | 0 | never_reached | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 16 | ✓ | ✓ | ✓ | ✗ | 0 | 25 | success | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 5 (Seed: 4)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✓ | ✗ | ✓ | 9 | 0 | success | success |
| 5 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 21 | success | missed_grasp |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 9 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |

### Run 6 (Seed: 5)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 32 | 0 | never_reached | never_reached |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 48 | 0 | never_reached | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 7 | ✗ | ✗ | ✗ | ✗ | 34 | 0 | never_reached | dropped |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 2 | 0 | missed_grasp | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 1 | never_reached | never_reached |
| 16 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 17 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✓ | ✗ | ✓ | ✗ | 0 | 8 | success | missed_grasp |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |

### Run 7 (Seed: 6)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 2 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | dropped | success |
| 4 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 5 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | never_reached |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 9 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 10 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 11 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 12 | ✓ | ✗ | ✓ | ✗ | 0 | 54 | success | never_reached |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 20 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |

### Run 8 (Seed: 7)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 5 | ✓ | ✓ | ✗ | ✓ | 13 | 0 | success | success |
| 6 | ✓ | ✓ | ✗ | ✓ | 4 | 0 | success | success |
| 7 | ✗ | ✓ | ✗ | ✗ | 0 | 1 | missed_grasp | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✗ | ✗ | ✗ | 24 | 18 | missed_grasp | missed_grasp |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | dropped |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 13 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | never_reached |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 16 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 17 | ✗ | ✗ | ✗ | ✗ | 0 | 11 | missed_grasp | missed_grasp |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | never_reached | missed_grasp |
| 20 | ✗ | ✓ | ✗ | ✓ | 15 | 0 | missed_grasp | success |

### Run 9 (Seed: 8)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 303 | 31 | never_reached | never_reached |
| 2 | ✗ | ✗ | ✗ | ✗ | 30 | 0 | never_reached | missed_grasp |
| 3 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 4 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 5 | ✓ | ✓ | ✗ | ✓ | 36 | 0 | success | success |
| 6 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 7 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 8 | ✗ | ✓ | ✗ | ✓ | 17 | 0 | never_reached | success |
| 9 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 10 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 11 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 12 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | missed_grasp |
| 13 | ✓ | ✓ | ✗ | ✗ | 14 | 15 | success | success |
| 14 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 15 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 16 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 17 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 18 | ✗ | ✗ | ✗ | ✗ | 13 | 30 | never_reached | never_reached |
| 19 | ✓ | ✓ | ✓ | ✗ | 0 | 5 | success | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

### Run 10 (Seed: 9)

| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |
|---------|------|------|--------|--------|--------|--------|-----------|----------|
| 1 | ✗ | ✗ | ✗ | ✗ | 26 | 8 | never_reached | never_reached |
| 2 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 3 | ✗ | ✓ | ✗ | ✓ | 18 | 0 | never_reached | success |
| 4 | ✓ | ✓ | ✗ | ✓ | 9 | 0 | success | success |
| 5 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 6 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 7 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | missed_grasp | success |
| 8 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 9 | ✗ | ✓ | ✗ | ✓ | 0 | 0 | never_reached | success |
| 10 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 11 | ✓ | ✗ | ✓ | ✗ | 0 | 0 | success | missed_grasp |
| 12 | ✓ | ✓ | ✗ | ✓ | 23 | 0 | success | success |
| 13 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 14 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 15 | ✗ | ✗ | ✗ | ✗ | 0 | 0 | missed_grasp | dropped |
| 16 | ✗ | ✗ | ✗ | ✗ | 51 | 22 | never_reached | never_reached |
| 17 | ✗ | ✓ | ✗ | ✓ | 35 | 0 | missed_grasp | success |
| 18 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |
| 19 | ✗ | ✓ | ✗ | ✓ | 40 | 0 | never_reached | success |
| 20 | ✓ | ✓ | ✓ | ✓ | 0 | 0 | success | success |

