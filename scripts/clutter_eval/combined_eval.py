#!/usr/bin/env python3
"""Combined baseline + CGVD evaluation script.

This script runs both baseline and CGVD episodes in a single process with one model
load, reducing initialization overhead by 50% compared to running them separately.

Usage:
    python combined_eval.py --task widowx_spoon_on_towel --num_episodes 10 --seed 42

The script:
1. Loads the model ONCE
2. For each seed/run:
   - Creates environment with distractors
   - Runs baseline episodes (no CGVD wrapper)
   - Runs CGVD episodes (with CGVD wrapper)
3. Reports comparative results
"""

import argparse
import gc
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import hydra
import imageio
import numpy as np
import simpler_env
import torch
from omegaconf import OmegaConf

from src.model.vla.pizero import PiZeroInference
from src.utils.monitor import log_allocated_gpu_memory


@dataclass
class EpisodeResult:
    """Result from a single episode."""
    success: bool
    steps: int
    inference_time: float
    episode_time: float
    episode_id: int


@dataclass
class EvalResult:
    """Aggregated results from evaluation."""
    mode: str  # "baseline" or "cgvd"
    success_rate: float
    num_successes: int
    num_episodes: int
    avg_steps: float
    avg_inference_time: float
    avg_episode_time: float
    episode_results: List[EpisodeResult]


def load_checkpoint(model, path):
    """Load checkpoint to CPU first, then GPU."""
    data = torch.load(path, weights_only=True, map_location="cpu")
    data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
    model.load_state_dict(data["model"], strict=True)
    print(f"Loaded model from {path}")


def create_base_env(args):
    """Create base environment with distractors (but no CGVD wrapper)."""
    env = simpler_env.make(args.task)

    if args.distractors:
        from src.cgvd.distractor_wrapper import DistractorWrapper
        env = DistractorWrapper(
            env, args.distractors,
            distractor_scale=args.distractor_scale,
            external_asset_scale=args.external_asset_scale,
            num_distractors=args.num_distractors,
            randomize_per_episode=args.randomize_distractors,
        )
        if args.distractor_scale:
            scale_info = f"all={args.distractor_scale}"
        else:
            ext_scale = args.external_asset_scale if args.external_asset_scale else 0.1
            scale_info = f"rc_*/ycb_*={ext_scale}, others=1.0"
        print(f"[Distractors] Added: {args.distractors} ({scale_info})")

    return env


def wrap_with_cgvd(env, args):
    """Wrap environment with CGVD."""
    from src.cgvd import CGVDWrapper

    if args.output_dir and args.output_dir != ".":
        debug_dir = os.path.join(args.output_dir, "cgvd", "cgvd_debug")
    else:
        debug_dir = os.path.join("cgvd_debug", args.task)

    return CGVDWrapper(
        env,
        update_freq=args.cgvd_update_freq,
        presence_threshold=args.cgvd_presence_threshold,
        use_mock_segmenter=args.cgvd_use_mock,
        include_robot=True,
        verbose=args.cgvd_verbose,
        save_debug_images=args.cgvd_save_debug,
        debug_dir=debug_dir,
        distractor_names=args.cgvd_distractor_names,
        cache_distractor_once=True,
        robot_presence_threshold=args.cgvd_robot_threshold,
        distractor_presence_threshold=args.cgvd_distractor_threshold,
        disable_safeset=getattr(args, 'cgvd_disable_safeset', False),
        disable_inpaint=getattr(args, 'cgvd_disable_inpaint', False),
    )


def run_episode(
    args,
    env,
    env_adapter,
    model,
    cfg,
    device,
    dtype,
    episode_idx: int,
    mode: str,
) -> EpisodeResult:
    """Run a single episode and return result."""
    episode_start = time.time()
    env_adapter.reset()

    episode_id = (args.seed + episode_idx) % 24
    env_reset_options = {
        "obj_init_options": {"episode_id": episode_id}
    }
    obs, reset_info = env.reset(options=env_reset_options)
    instruction = env.unwrapped.get_language_instruction()

    video_writer = None
    video_path = None
    if args.recording:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        mode_dir = os.path.join(args.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        video_path = os.path.join(mode_dir, f"try_{args.task}_{mode}_ep{episode_idx}.mp4")
        video_writer = imageio.get_writer(video_path)

    print(f"\n--- [{mode.upper()}] Episode {episode_idx + 1}/{args.num_episodes} (id={episode_id}) ---")
    print(f"Instruction: {instruction}")

    cnt_step = 0
    inference_times = []
    success = False

    while True:
        inputs = env_adapter.preprocess(env, obs, instruction)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(
                inputs["attention_mask"], dtype=dtype
            )
        )
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
            causal_mask
        )
        inputs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"].to(dtype),
            "image_text_proprio_mask": image_text_proprio_mask,
            "action_mask": action_mask,
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": inputs["proprios"].to(dtype),
        }
        inputs = {k: v.to(device) for k, v in inputs.items()}
        start_inference_time = time.time()
        with torch.inference_mode():
            actions = model(**inputs)
        if cnt_step > 0:
            inference_times.append(time.time() - start_inference_time)
        env_actions = env_adapter.postprocess(actions[0].float().cpu().numpy())

        for env_action in env_actions[: cfg.act_steps]:
            obs, reward, success, truncated, info = env.step(env_action)
            cnt_step += 1
            if truncated:
                break

        if video_writer is not None:
            video_writer.append_data(env_adapter.get_video_frame(env, obs))

        new_instruction = env.unwrapped.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction

        if truncated:
            if video_writer is not None:
                video_writer.close()
                if video_path:
                    result_suffix = "SUCCESS" if success else "FAILED"
                    new_video_path = video_path.replace(".mp4", f"_{result_suffix}.mp4")
                    os.rename(video_path, new_video_path)
                    print(f"Saved: {new_video_path}")
            break

    episode_time = time.time() - episode_start
    avg_inference = np.mean(inference_times) if inference_times else 0
    result_str = "SUCCESS" if success else "FAILED"
    print(f"[{mode.upper()}] Episode {episode_idx + 1}: {result_str} (steps={cnt_step}, time={episode_time:.2f}s)")

    return EpisodeResult(
        success=success,
        steps=cnt_step,
        inference_time=avg_inference,
        episode_time=episode_time,
        episode_id=episode_id,
    )


def run_evaluation(
    args,
    env,
    env_adapter,
    model,
    cfg,
    device,
    dtype,
    mode: str,
) -> EvalResult:
    """Run all episodes for a given mode (baseline or cgvd)."""
    results = []

    for ep_idx in range(args.num_episodes):
        result = run_episode(
            args, env, env_adapter, model, cfg, device, dtype, ep_idx, mode
        )
        results.append(result)

        if args.clear_cuda_cache:
            torch.cuda.empty_cache()
            gc.collect()

    num_successes = sum(1 for r in results if r.success)
    success_rate = num_successes / len(results) * 100

    return EvalResult(
        mode=mode,
        success_rate=success_rate,
        num_successes=num_successes,
        num_episodes=len(results),
        avg_steps=np.mean([r.steps for r in results]),
        avg_inference_time=np.mean([r.inference_time for r in results]),
        avg_episode_time=np.mean([r.episode_time for r in results]),
        episode_results=results,
    )


def print_comparison(baseline: EvalResult, cgvd: EvalResult):
    """Print comparison between baseline and CGVD results."""
    improvement = cgvd.success_rate - baseline.success_rate
    sign = "+" if improvement >= 0 else ""

    print("\n" + "=" * 60)
    print("COMBINED EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\n{'Mode':<12} {'Success Rate':>15} {'Successes':>12} {'Avg Steps':>12}")
    print("-" * 60)
    print(f"{'Baseline':<12} {baseline.success_rate:>14.1f}% {baseline.num_successes:>12} {baseline.avg_steps:>12.1f}")
    print(f"{'CGVD':<12} {cgvd.success_rate:>14.1f}% {cgvd.num_successes:>12} {cgvd.avg_steps:>12.1f}")
    print("-" * 60)
    print(f"{'Improvement':<12} {sign}{improvement:>14.1f}%")
    print("=" * 60)

    # Per-episode comparison
    print("\nPer-Episode Comparison:")
    print(f"{'Episode':<10} {'Baseline':>12} {'CGVD':>12} {'Match':>10}")
    print("-" * 50)
    for i, (b, c) in enumerate(zip(baseline.episode_results, cgvd.episode_results)):
        b_str = "SUCCESS" if b.success else "FAILED"
        c_str = "SUCCESS" if c.success else "FAILED"
        match = "=" if b.success == c.success else ("+" if c.success else "-")
        print(f"{i+1:<10} {b_str:>12} {c_str:>12} {match:>10}")
    print("=" * 60)


def save_results(args, baseline: EvalResult, cgvd: EvalResult):
    """Save results to CSV files."""
    if not args.output_dir or args.output_dir == ".":
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Per-episode results
    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, "w") as f:
        f.write("episode,episode_id,baseline_success,cgvd_success,baseline_steps,cgvd_steps,baseline_time,cgvd_time\n")
        for i, (b, c) in enumerate(zip(baseline.episode_results, cgvd.episode_results)):
            f.write(f"{i},{b.episode_id},{int(b.success)},{int(c.success)},{b.steps},{c.steps},{b.episode_time:.2f},{c.episode_time:.2f}\n")

    # Summary
    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, "w") as f:
        f.write("mode,success_rate,num_successes,num_episodes,avg_steps,avg_episode_time\n")
        f.write(f"baseline,{baseline.success_rate:.1f},{baseline.num_successes},{baseline.num_episodes},{baseline.avg_steps:.1f},{baseline.avg_episode_time:.2f}\n")
        f.write(f"cgvd,{cgvd.success_rate:.1f},{cgvd.num_successes},{cgvd.num_episodes},{cgvd.avg_steps:.1f},{cgvd.avg_episode_time:.2f}\n")

    print(f"\nResults saved to: {args.output_dir}")


def main(args):
    """Main evaluation function."""
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device setup
    device = torch.device(f"cuda:{args.gpu_id}")
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32

    # Load config
    if "fractal" in args.checkpoint_path:
        cfg = OmegaConf.load("config/eval/fractal_apple.yaml")
    elif "bridge" in args.checkpoint_path:
        cfg = OmegaConf.load("config/eval/bridge.yaml")
    else:
        raise ValueError(f"Unknown checkpoint type: {args.checkpoint_path}")

    # Load model ONCE for both baseline and CGVD
    print("\n" + "=" * 60)
    print("LOADING MODEL (shared for baseline + CGVD)")
    print("=" * 60)
    model_load_start = time.time()
    model = PiZeroInference(cfg, use_ddp=False)
    load_checkpoint(model, args.checkpoint_path)
    model.freeze_all_weights()
    model.to(dtype)
    model.to(device)

    if args.use_torch_compile:
        model = torch.compile(model, mode="default")

    model.eval()
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")
    log_allocated_gpu_memory(None, "loading model", args.gpu_id)

    # Create env adapter (shared)
    env_adapter = hydra.utils.instantiate(cfg.env.adapter)

    # Auto-derive CGVD distractor names
    if args.distractors and not args.cgvd_distractor_names:
        derived_names = []
        for asset_id in args.distractors:
            asset_id_clean = asset_id.split(":")[0]
            parts = asset_id_clean.split("_")
            if len(parts) >= 3:
                if parts[0] == "ycb":
                    name = " ".join(parts[2:])
                else:
                    name = " ".join(parts[1:-1])
                if name and name not in derived_names:
                    derived_names.append(name)
            elif len(parts) == 2:
                name = parts[0]
                if name and name not in derived_names:
                    derived_names.append(name)
        if derived_names:
            args.cgvd_distractor_names = derived_names
            print(f"[CGVD] Auto-derived distractor names: {derived_names}")

    # ========================================
    # PHASE 1: Baseline Evaluation
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE EVALUATION")
    print("=" * 60)

    baseline_env = create_base_env(args)
    baseline_result = run_evaluation(
        args, baseline_env, env_adapter, model, cfg, device, dtype, "baseline"
    )
    baseline_env.close()

    # ========================================
    # PHASE 2: CGVD Evaluation
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 2: CGVD EVALUATION")
    print("=" * 60)

    # Create fresh env and wrap with CGVD
    cgvd_env = create_base_env(args)
    cgvd_env = wrap_with_cgvd(cgvd_env, args)
    cgvd_result = run_evaluation(
        args, cgvd_env, env_adapter, model, cfg, device, dtype, "cgvd"
    )
    cgvd_env.close()

    # ========================================
    # Results
    # ========================================
    print_comparison(baseline_result, cgvd_result)
    save_results(args, baseline_result, cgvd_result)

    # Print timing info
    print(f"\nModel load time: {model_load_time:.2f}s (loaded ONCE)")
    print(f"Total baseline time: {sum(r.episode_time for r in baseline_result.episode_results):.2f}s")
    print(f"Total CGVD time: {sum(r.episode_time for r in cgvd_result.episode_results):.2f}s")

    # Return for shell script parsing
    print(f"\nSUCCESS RATE: {baseline_result.success_rate:.1f}%")  # For baseline log parsing
    return baseline_result, cgvd_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined baseline + CGVD evaluation")

    # Task configuration
    parser.add_argument("--task", type=str, default="widowx_spoon_on_towel",
                       choices=[
                           "widowx_carrot_on_plate",
                           "widowx_banana_on_plate",
                           "widowx_put_eggplant_in_basket",
                           "widowx_spoon_on_towel",
                           "widowx_stack_cube",
                           "google_robot_pick_horizontal_coke_can",
                           "google_robot_pick_vertical_coke_can",
                           "google_robot_pick_standing_coke_can",
                           "google_robot_move_near_v0",
                           "google_robot_open_drawer",
                           "google_robot_close_drawer",
                           "google_robot_place_apple_in_closed_top_drawer",
                       ])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_torch_compile", action="store_true")
    parser.add_argument("--clear_cuda_cache", action="store_true")
    parser.add_argument("--recording", action="store_true")
    parser.add_argument("--output_dir", type=str, default=".")

    # Distractor configuration
    parser.add_argument("--distractors", type=str, nargs="*", default=[])
    parser.add_argument("--distractor_scale", type=float, default=None)
    parser.add_argument("--external_asset_scale", type=float, default=None)
    parser.add_argument("--num_distractors", type=int, default=None)
    parser.add_argument("--randomize_distractors", action="store_true")

    # CGVD configuration
    parser.add_argument("--cgvd_update_freq", type=int, default=1)
    parser.add_argument("--cgvd_presence_threshold", type=float, default=0.4)
    parser.add_argument("--cgvd_use_mock", action="store_true")
    parser.add_argument("--cgvd_verbose", action="store_true")
    parser.add_argument("--cgvd_save_debug", action="store_true")
    parser.add_argument("--cgvd_distractor_names", type=str, nargs="*", default=[])
    parser.add_argument("--cgvd_robot_threshold", type=float, default=0.05)
    parser.add_argument("--cgvd_distractor_threshold", type=float, default=0.3)
    parser.add_argument("--cgvd_disable_safeset", action="store_true")
    parser.add_argument("--cgvd_disable_inpaint", action="store_true")

    args = parser.parse_args()

    # Validate checkpoint vs task
    if "google_robot" in args.task:
        assert "fractal" in args.checkpoint_path, "Google robot tasks require fractal checkpoint"
    if "widowx" in args.task:
        assert "bridge" in args.checkpoint_path, "WidowX tasks require bridge checkpoint"

    main(args)
