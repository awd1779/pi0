#!/usr/bin/env python3
"""CGVD-only batch evaluation script (skips baseline).

Identical to batch_eval.py but skips the baseline phase entirely,
cutting evaluation time roughly in half when you only need CGVD results.

Usage:
    python batch_eval_cgvd_only.py --task widowx_spoon_on_towel \
        --categories semantic --distractor_counts 5 \
        --episodes 21 --runs 3
"""

import argparse
import gc
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from batch_eval import (
    BatchEvaluator,
    ConfigResult,
    EvalConfig,
    generate_configs,
    load_distractors_from_file,
    print_sweep_summary,
    save_sweep_results,
)


class CGVDOnlyEvaluator(BatchEvaluator):
    """BatchEvaluator that only runs CGVD (no baseline)."""

    def run_configuration(self, config: EvalConfig, config_output_dir: Optional[str] = None) -> ConfigResult:
        """Run a single configuration (CGVD only, no baseline)."""
        import random
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        print(f"\n--- Config: {config.task} | {config.category} | {config.num_distractors} distractors | seed={config.seed} ---")

        run_dir = None
        if config_output_dir:
            run_dir = os.path.join(config_output_dir, f"run_{config.run_index}")
            os.makedirs(run_dir, exist_ok=True)
            self._save_config_file(config, config_output_dir)

        cgvd_results = []
        cgvd_log_lines = []

        # Run CGVD only
        if config.cgvd_distractor_names:
            cgvd_output = os.path.join(run_dir, "cgvd") if run_dir else None
            if cgvd_output:
                os.makedirs(cgvd_output, exist_ok=True)

            print(f"  [CGVD] Running {config.num_episodes} episodes...")
            env = self._create_env(config, use_cgvd=True, output_dir=cgvd_output)
            for ep_idx in range(config.num_episodes):
                result = self._run_episode(
                    env, ep_idx, config.seed,
                    mode="cgvd",
                    output_dir=cgvd_output,
                    task=config.task,
                    use_cgvd=True,
                )
                cgvd_results.append(result)
                status = "SUCCESS" if result.success else "FAILED"
                log_line = f"Episode {ep_idx+1}: {status} (steps={result.steps}, time={result.episode_time:.2f}s, collisions={result.collision_count}, failure={result.failure_mode}, cgvd={result.cgvd_time:.2f}s)"
                cgvd_log_lines.append(log_line)
                print(f"    {log_line}")
            env.close()
        else:
            print(f"  [CGVD] Skipped (no distractors)")

        # Save log
        if run_dir and config.cgvd_distractor_names:
            self._save_log_file(cgvd_log_lines, cgvd_results, run_dir, "cgvd", config)

        torch.cuda.empty_cache()
        gc.collect()

        cgvd_successes = sum(1 for r in cgvd_results if r.success)

        result = ConfigResult(
            config=config,
            baseline_successes=0,
            baseline_total=0,
            cgvd_successes=cgvd_successes,
            cgvd_total=len(cgvd_results),
            baseline_results=[],
            cgvd_results=cgvd_results,
        )

        print(f"  Results: CGVD={result.cgvd_rate:.1f}%")
        return result


def main():
    parser = argparse.ArgumentParser(description="CGVD-only batch evaluation (skips baseline)")

    # Task configuration
    parser.add_argument("--task", type=str, default="widowx_carrot_on_plate")
    parser.add_argument("--checkpoint_path", type=str,
                       default="/home/ubuntu/open-pi-zero/checkpoints/bridge_beta.pt")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--use_torch_compile", action="store_true")

    # Sweep configuration
    parser.add_argument("--categories", type=str, nargs="+",
                       default=["semantic", "visual", "control"])
    parser.add_argument("--distractor_counts", type=int, nargs="+",
                       default=[0, 1, 3, 5, 7, 9])
    parser.add_argument("--episodes", type=int, default=21)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=0)

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")

    # Recording and debug
    parser.add_argument("--recording", action="store_true")
    parser.add_argument("--cgvd_save_debug", action="store_true")
    parser.add_argument("--cgvd_verbose", action="store_true")
    parser.add_argument("--save_attention", action="store_true")

    # Distractor randomization
    parser.add_argument("--randomize_distractors", action="store_true")

    # CGVD thresholds
    parser.add_argument("--cgvd_safe_threshold", type=float, default=0.3)
    parser.add_argument("--cgvd_robot_threshold", type=float, default=0.3)
    parser.add_argument("--cgvd_distractor_threshold", type=float, default=0.20)
    parser.add_argument("--robot_seg_on_original", action="store_true")

    args = parser.parse_args()

    configs = generate_configs(args)

    print("=" * 70)
    print("CGVD-ONLY BATCH EVALUATION")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Categories: {args.categories}")
    print(f"Distractor counts: {args.distractor_counts}")
    print(f"Episodes per config: {args.episodes}")
    print(f"Runs per config: {args.runs}")
    print(f"Total configurations: {len(configs)}")
    print(f"Total episodes: {len(configs) * args.episodes}")  # No x2 — CGVD only
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would run the following configurations:")
        for i, c in enumerate(configs[:10]):
            print(f"  {i+1}. {c.task} | {c.category} | {c.num_distractors} dist | seed={c.seed}")
        if len(configs) > 10:
            print(f"  ... and {len(configs) - 10} more")
        return

    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"logs/clutter_eval/batch_cgvd_only/{args.task}/{timestamp}"

    evaluator = CGVDOnlyEvaluator(
        checkpoint_path=args.checkpoint_path,
        device=args.gpu_id,
        use_bf16=args.use_bf16,
        use_torch_compile=args.use_torch_compile,
        recording=args.recording,
        cgvd_save_debug=args.cgvd_save_debug,
        cgvd_verbose=args.cgvd_verbose,
        cgvd_safe_threshold=args.cgvd_safe_threshold,
        cgvd_robot_threshold=args.cgvd_robot_threshold,
        cgvd_distractor_threshold=args.cgvd_distractor_threshold,
        cgvd_robot_seg_on_original=args.robot_seg_on_original,
        save_attention=args.save_attention,
    )

    results = list(evaluator.run_sweep(configs, args.output_dir))

    save_sweep_results(results, args.output_dir)
    print_sweep_summary(results)

    print(f"\nModel load time: {evaluator.model_load_time:.2f}s (loaded ONCE for {len(configs)} configs)")
    total_episodes = sum(len(r.cgvd_results) for r in results)
    print(f"Total CGVD episodes run: {total_episodes}")


if __name__ == "__main__":
    main()
