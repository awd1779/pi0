# GR00T Sweep Evaluation Commands

## Quick Test (2 runs, 5 episodes, 2 distractor counts)

```bash
./scripts/clutter_eval/run_category_sweep_fast_groot.sh -e 5 -r 2 --counts 0,3
```

## Full Sweep (same conditions as Pi0)

```bash
./scripts/clutter_eval/run_category_sweep_fast_groot.sh
```

Defaults: 21 episodes, 10 runs, categories=semantic,visual,control, counts=0,1,3,5,7,9

## Full Sweep with CGVD

Terminal 1 - start SAM3 server:

```bash
conda activate sam3 && python scripts/sam3_server.py
```

Terminal 2 - run sweep:

```bash
./scripts/clutter_eval/run_category_sweep_fast_groot.sh
```

## Custom Task

```bash
./scripts/clutter_eval/run_category_sweep_fast_groot.sh --task widowx_spoon_on_towel
```

```bash
./scripts/clutter_eval/run_category_sweep_fast_groot.sh --task widowx_put_eggplant_in_basket
```

```bash
./scripts/clutter_eval/run_category_sweep_fast_groot.sh --task google_robot_pick_horizontal_coke_can
```

## Dry Run (preview without executing)

```bash
./scripts/clutter_eval/run_category_sweep_fast_groot.sh --dry-run
```

## With Video Recording

```bash
./scripts/clutter_eval/run_category_sweep_fast_groot.sh --recording
```

## Results

GR00T results: `logs/clutter_eval/gr00t/`

Pi0 results: `logs/clutter_eval/pi0/`

Both produce the same CSV/JSON format for direct comparison.
