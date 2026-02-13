#!/usr/bin/env python3
"""Batch evaluation script for running full configuration sweeps.

This script loads the model ONCE and runs through ALL configurations, providing
massive speedup compared to spawning separate processes for each configuration.

Estimated speedup: 95% reduction in model loading time
- Old: 360 separate processes × 5-30s model load = 30-180 minutes
- New: 1 model load × 5-30s = 5-30 seconds

Usage:
    # Run full sweep
    python batch_eval.py --configs configs.json

    # Run with specific configurations
    python batch_eval.py --task widowx_carrot_on_plate \
        --categories semantic visual control \
        --distractor_counts 0 1 3 5 7 9 \
        --episodes 21 --runs 10

    # Dry run to see what would be executed
    python batch_eval.py --task widowx_carrot_on_plate --dry_run
"""

import argparse
import gc
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import hydra
import imageio
import numpy as np
import simpler_env
import torch
from omegaconf import OmegaConf

from src.model.vla.pizero import PiZeroInference
from src.utils.monitor import log_allocated_gpu_memory


# Global settings for recording (set by BatchEvaluator)
_RECORDING_ENABLED = False
_CGVD_SAVE_DEBUG = False
_CGVD_VERBOSE = False


@dataclass
class EvalConfig:
    """Configuration for a single evaluation run."""
    task: str
    category: str
    num_distractors: int
    seed: int
    num_episodes: int
    run_index: int
    distractors: List[str] = field(default_factory=list)
    cgvd_distractor_names: List[str] = field(default_factory=list)
    randomize_distractors: bool = False  # Randomize distractors per episode
    distractor_pool_size: int = 0  # Size of pool to sample from (0 = use all)


@dataclass
class EpisodeResult:
    """Result from a single episode."""
    success: bool
    steps: int
    inference_time: float
    episode_time: float
    episode_id: int
    # New metrics
    collision_count: int = 0
    collision_frames: List[int] = field(default_factory=list)
    failure_mode: str = ""  # "success", "never_reached", "missed_grasp", "dropped"
    cgvd_time: float = 0.0  # Total CGVD time for episode
    sam3_time: float = 0.0  # Total SAM3 segmentation time
    lama_time: float = 0.0  # Total LaMa inpainting time

    @property
    def hard_success(self) -> bool:
        """Success without any collisions (h-SR metric)."""
        return self.success and self.collision_count == 0


@dataclass
class ConfigResult:
    """Results from a single configuration (both baseline and CGVD)."""
    config: EvalConfig
    baseline_successes: int
    baseline_total: int
    cgvd_successes: int
    cgvd_total: int
    baseline_results: List[EpisodeResult] = field(default_factory=list)
    cgvd_results: List[EpisodeResult] = field(default_factory=list)

    @property
    def baseline_rate(self) -> float:
        return self.baseline_successes / self.baseline_total * 100 if self.baseline_total > 0 else 0

    @property
    def cgvd_rate(self) -> float:
        return self.cgvd_successes / self.cgvd_total * 100 if self.cgvd_total > 0 else 0

    @property
    def improvement(self) -> float:
        return self.cgvd_rate - self.baseline_rate

    @property
    def baseline_hard_success_rate(self) -> float:
        """Hard success rate: success without any collisions."""
        hard = sum(1 for r in self.baseline_results if r.hard_success)
        return hard / self.baseline_total * 100 if self.baseline_total > 0 else 0

    @property
    def cgvd_hard_success_rate(self) -> float:
        """Hard success rate: success without any collisions."""
        hard = sum(1 for r in self.cgvd_results if r.hard_success)
        return hard / self.cgvd_total * 100 if self.cgvd_total > 0 else 0

    @property
    def hard_improvement(self) -> float:
        """Improvement in hard success rate."""
        return self.cgvd_hard_success_rate - self.baseline_hard_success_rate


class BatchEvaluator:
    """Evaluator that loads model once and runs multiple configurations."""

    def __init__(
        self,
        checkpoint_path: str,
        device: int = 0,
        use_bf16: bool = True,
        use_torch_compile: bool = False,
        recording: bool = False,
        cgvd_save_debug: bool = False,
        cgvd_verbose: bool = False,
        cgvd_safe_threshold: float = 0.6,
        cgvd_robot_threshold: float = 0.3,
        cgvd_distractor_threshold: float = 0.20,
        save_attention: bool = False,
    ):
        """Initialize batch evaluator with shared model.

        Args:
            checkpoint_path: Path to model checkpoint
            device: GPU device ID
            use_bf16: Use bfloat16 precision
            use_torch_compile: Use torch.compile for optimization
            recording: Save video recordings of episodes
            cgvd_save_debug: Save CGVD debug images
            cgvd_verbose: Print verbose CGVD output
            cgvd_safe_threshold: Threshold for safe-set (target/anchor) detection
            cgvd_robot_threshold: Threshold for robot arm detection
            cgvd_distractor_threshold: Threshold for distractor detection
            save_attention: Save attention map visualizations for each episode
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(f"cuda:{device}")
        self.dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.gpu_id = device
        self.recording = recording
        self.cgvd_save_debug = cgvd_save_debug
        self.cgvd_verbose = cgvd_verbose
        self.cgvd_safe_threshold = cgvd_safe_threshold
        self.cgvd_robot_threshold = cgvd_robot_threshold
        self.cgvd_distractor_threshold = cgvd_distractor_threshold
        self.save_attention = save_attention

        # Set globals for recording
        global _RECORDING_ENABLED, _CGVD_SAVE_DEBUG, _CGVD_VERBOSE
        _RECORDING_ENABLED = recording
        _CGVD_SAVE_DEBUG = cgvd_save_debug
        _CGVD_VERBOSE = cgvd_verbose

        if recording or save_attention:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load config based on checkpoint type
        if "fractal" in checkpoint_path:
            self.cfg = OmegaConf.load("config/eval/fractal_apple.yaml")
        elif "bridge" in checkpoint_path:
            self.cfg = OmegaConf.load("config/eval/bridge.yaml")
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_path}")

        # Load model ONCE
        print("\n" + "=" * 70)
        print("BATCH EVALUATOR: Loading model (shared across all configurations)")
        print("=" * 70)
        model_load_start = time.time()

        self.model = PiZeroInference(self.cfg, use_ddp=False)
        self._load_checkpoint()
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)

        if use_torch_compile:
            self.model = torch.compile(self.model, mode="default")

        self.model.eval()
        self.model_load_time = time.time() - model_load_start
        print(f"Model loaded in {self.model_load_time:.2f}s")
        log_allocated_gpu_memory(None, "loading model", self.gpu_id)

        # Create attention capture if enabled
        self.attention_capture = None
        if save_attention:
            from src.utils.attention_capture import AttentionCapture
            self.attention_capture = AttentionCapture(self.model)

        # Create env adapter (shared)
        self.env_adapter = hydra.utils.instantiate(self.cfg.env.adapter)

    def _load_checkpoint(self):
        """Load checkpoint weights."""
        data = torch.load(self.checkpoint_path, weights_only=True, map_location="cpu")
        data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
        self.model.load_state_dict(data["model"], strict=True)
        print(f"Loaded model from {self.checkpoint_path}")

    def _create_env(self, config: EvalConfig, use_cgvd: bool = False, output_dir: Optional[str] = None):
        """Create environment for a configuration."""
        env = simpler_env.make(config.task)

        # Add distractors if specified AND count > 0
        # (num_distractors=0 means no distractors, skip wrapper entirely)
        if config.distractors and config.num_distractors > 0:
            from src.cgvd.distractor_wrapper import DistractorWrapper

            # If randomizing, pass full pool and let wrapper sample per episode
            # Otherwise, distractors list is already trimmed to num_distractors
            num_to_sample = config.num_distractors if config.randomize_distractors else None

            env = DistractorWrapper(
                env,
                config.distractors,
                distractor_scale=None,
                external_asset_scale=0.1,
                num_distractors=num_to_sample,
                randomize_per_episode=config.randomize_distractors,
            )

        # Optionally wrap with CGVD
        if use_cgvd and config.cgvd_distractor_names:
            from src.cgvd import CGVDWrapper

            # Set up debug directory for CGVD
            debug_dir = "cgvd_debug"
            if output_dir and self.cgvd_save_debug:
                debug_dir = os.path.join(output_dir, "cgvd_debug")

            env = CGVDWrapper(
                env,
                update_freq=1,
                presence_threshold=self.cgvd_safe_threshold,
                use_mock_segmenter=False,
                include_robot=True,
                verbose=self.cgvd_verbose,
                save_debug_images=self.cgvd_save_debug,
                debug_dir=debug_dir,
                distractor_names=config.cgvd_distractor_names,
                cache_distractor_once=True,
                robot_presence_threshold=self.cgvd_robot_threshold,
                distractor_presence_threshold=self.cgvd_distractor_threshold,
            )

        return env

    def _run_episode(
        self,
        env,
        episode_idx: int,
        seed: int,
        mode: str = "baseline",
        output_dir: Optional[str] = None,
        task: str = "",
        category: str = "",
        use_cgvd: bool = False,
    ) -> EpisodeResult:
        """Run a single episode."""
        from src.cgvd import CollisionTracker, GraspAnalyzer

        episode_start = time.time()
        self.env_adapter.reset()

        episode_id = (seed + episode_idx) % 24
        env_reset_options = {"obj_init_options": {"episode_id": episode_id}}
        obs, _ = env.reset(options=env_reset_options)
        instruction = env.unwrapped.get_language_instruction()

        # Initialize collision tracker (only if distractors are present)
        collision_tracker = None
        has_distractors = hasattr(env, 'distractor_objs') or (
            hasattr(env, 'env') and hasattr(env.env, 'distractor_objs')
        )
        if has_distractors:
            collision_tracker = CollisionTracker(env)

        # Initialize grasp analyzer
        grasp_analyzer = GraspAnalyzer(env)
        grasp_analyzer.on_reset(obs)

        # Set up video recording
        # Videos go directly in output_dir (e.g., run_0/baseline/try_task_baseline_ep0.mp4)
        video_writer = None
        video_path = None
        if self.recording and output_dir:
            video_path = os.path.join(output_dir, f"try_{task}_{category}_{mode}_ep{episode_idx}.mp4")
            video_writer = imageio.get_writer(video_path, fps=10)

        cnt_step = 0
        inference_times = []
        success = False

        while True:
            inputs = self.env_adapter.preprocess(env, obs, instruction)
            causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
                self.model.build_causal_mask_and_position_ids(
                    inputs["attention_mask"], dtype=self.dtype
                )
            )
            image_text_proprio_mask, action_mask = self.model.split_full_mask_into_submasks(causal_mask)
            inputs = {
                "input_ids": inputs["input_ids"],
                "pixel_values": inputs["pixel_values"].to(self.dtype),
                "image_text_proprio_mask": image_text_proprio_mask,
                "action_mask": action_mask,
                "vlm_position_ids": vlm_position_ids,
                "proprio_position_ids": proprio_position_ids,
                "action_position_ids": action_position_ids,
                "proprios": inputs["proprios"].to(self.dtype),
            }
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            start_inference = time.time()
            with torch.inference_mode():
                actions = self.model(**inputs)
            if cnt_step > 0:
                inference_times.append(time.time() - start_inference)

            # Save attention map on first inference step
            if cnt_step == 0 and self.attention_capture is not None and output_dir:
                # Get the original image for visualization
                pixel_values = inputs["pixel_values"].cpu().float()  # Convert bf16 -> float32
                # pixel_values is [B, C, H, W] normalized, convert to [H, W, C] uint8
                img = pixel_values[0].permute(1, 2, 0).numpy()
                img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                attn_path = os.path.join(output_dir, f"attention_{task}_{category}_{mode}_ep{episode_idx}.png")
                saved = self.attention_capture.save_attention_map(img, attn_path)
                if saved:
                    print(f"  [Attention] Saved: {attn_path}")
                else:
                    print(f"  [Attention] Failed to save (no attention captured). Hooks: {len(self.attention_capture.hooks)}")
                self.attention_capture.clear()

            env_actions = self.env_adapter.postprocess(actions[0].float().cpu().numpy())

            for env_action in env_actions[: self.cfg.act_steps]:
                obs, reward, success, truncated, info = env.step(env_action)
                cnt_step += 1

                # Track collisions
                if collision_tracker is not None:
                    collision_tracker.check_collisions(cnt_step)

                # Track grasp state
                grasp_analyzer.on_step(obs, env_action, cnt_step)

                # Save frame to video
                if video_writer is not None:
                    video_writer.append_data(self.env_adapter.get_video_frame(env, obs))

                if truncated:
                    break

            new_instruction = env.unwrapped.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction

            if truncated:
                break

        # Close video and rename with result
        if video_writer is not None:
            video_writer.close()
            if video_path:
                result_suffix = "SUCCESS" if success else "FAILED"
                new_video_path = video_path.replace(".mp4", f"_{result_suffix}.mp4")
                os.rename(video_path, new_video_path)

        episode_time = time.time() - episode_start
        avg_inference = np.mean(inference_times) if inference_times else 0

        # Get collision stats
        collision_count = 0
        collision_frames = []
        if collision_tracker is not None:
            collision_stats = collision_tracker.get_stats()
            collision_count = collision_stats["collision_count"]
            collision_frames = collision_stats["collision_frames"]

        # Get grasp failure classification
        failure_mode = grasp_analyzer.classify_failure(success, obs)

        # Get CGVD timing stats (only for CGVD mode)
        cgvd_time = 0.0
        sam3_time = 0.0
        lama_time = 0.0
        if use_cgvd:
            # Try to get timing from CGVDWrapper
            cgvd_wrapper = self._get_cgvd_wrapper(env)
            if cgvd_wrapper is not None:
                timing_stats = cgvd_wrapper.get_timing_stats()
                cgvd_time = timing_stats.get("total_cgvd_time", 0.0)
                sam3_time = timing_stats.get("total_sam3_time", 0.0)
                lama_time = timing_stats.get("total_lama_time", 0.0)

        return EpisodeResult(
            success=success,
            steps=cnt_step,
            inference_time=avg_inference,
            episode_time=episode_time,
            episode_id=episode_id,
            collision_count=collision_count,
            collision_frames=collision_frames,
            failure_mode=failure_mode,
            cgvd_time=cgvd_time,
            sam3_time=sam3_time,
            lama_time=lama_time,
        )

    def _get_cgvd_wrapper(self, env):
        """Extract CGVDWrapper from potentially nested environment.

        Args:
            env: Environment (possibly wrapped)

        Returns:
            CGVDWrapper instance or None if not found
        """
        from src.cgvd import CGVDWrapper

        current = env
        while current is not None:
            if isinstance(current, CGVDWrapper):
                return current
            if hasattr(current, 'env'):
                current = current.env
            else:
                break
        return None

    def run_configuration(self, config: EvalConfig, config_output_dir: Optional[str] = None) -> ConfigResult:
        """Run a single configuration (both baseline and CGVD).

        Args:
            config: Evaluation configuration
            config_output_dir: Pre-computed output directory for this config group

        Returns:
            ConfigResult with baseline and CGVD results
        """
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        print(f"\n--- Config: {config.task} | {config.category} | {config.num_distractors} distractors | seed={config.seed} ---")

        # Create run directory within the config directory
        # Structure: config_output_dir/run_{idx}/
        run_dir = None
        if config_output_dir:
            run_dir = os.path.join(config_output_dir, f"run_{config.run_index}")
            os.makedirs(run_dir, exist_ok=True)

            # Save config.txt (only once per config group)
            self._save_config_file(config, config_output_dir)

        baseline_results = []
        cgvd_results = []
        baseline_log_lines = []
        cgvd_log_lines = []

        # Run baseline
        baseline_output = os.path.join(run_dir, "baseline") if run_dir else None
        if baseline_output:
            os.makedirs(baseline_output, exist_ok=True)

        print(f"  [Baseline] Running {config.num_episodes} episodes...")
        env = self._create_env(config, use_cgvd=False, output_dir=baseline_output)
        for ep_idx in range(config.num_episodes):
            result = self._run_episode(
                env, ep_idx, config.seed,
                mode="baseline",
                output_dir=baseline_output,
                task=config.task,
                category=config.category,
                use_cgvd=False,
            )
            baseline_results.append(result)
            status = "SUCCESS" if result.success else "FAILED"
            log_line = f"Episode {ep_idx+1}: {status} (steps={result.steps}, time={result.episode_time:.2f}s, collisions={result.collision_count}, failure={result.failure_mode})"
            baseline_log_lines.append(log_line)
            print(f"    {log_line}")
        env.close()

        # Run CGVD (only if distractors specified)
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
                    category=config.category,
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
            cgvd_results = baseline_results  # Copy baseline results if no distractors

        # Save log files
        if run_dir:
            self._save_log_file(baseline_log_lines, baseline_results, run_dir, "baseline", config)
            if config.cgvd_distractor_names:
                self._save_log_file(cgvd_log_lines, cgvd_results, run_dir, "cgvd", config)

        # Clean up
        torch.cuda.empty_cache()
        gc.collect()

        baseline_successes = sum(1 for r in baseline_results if r.success)
        cgvd_successes = sum(1 for r in cgvd_results if r.success)

        result = ConfigResult(
            config=config,
            baseline_successes=baseline_successes,
            baseline_total=len(baseline_results),
            cgvd_successes=cgvd_successes,
            cgvd_total=len(cgvd_results),
            baseline_results=baseline_results,
            cgvd_results=cgvd_results,
        )

        print(f"  Results: Baseline={result.baseline_rate:.1f}%, CGVD={result.cgvd_rate:.1f}%, Δ={result.improvement:+.1f}%")

        return result

    def _get_task_short(self, task: str) -> str:
        """Extract short task name from full task name."""
        task_map = {
            "widowx_carrot_on_plate": "carrot",
            "widowx_banana_on_plate": "banana",
            "widowx_put_eggplant_in_basket": "eggplant",
            "widowx_spoon_on_towel": "spoon",
            "widowx_stack_cube": "cube",
        }
        if task in task_map:
            return task_map[task]
        # Fallback: extract from task name
        for keyword in ["spoon", "carrot", "eggplant", "banana", "cube"]:
            if keyword in task:
                return keyword
        return task.replace("widowx_", "").replace("google_robot_", "")

    def _save_config_file(self, config: EvalConfig, output_dir: str):
        """Save config.txt matching run_paired_eval.sh format."""
        config_path = os.path.join(output_dir, "config.txt")
        if os.path.exists(config_path):
            return  # Don't overwrite existing config

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(f"MODEL=pi0\n")
            f.write(f"TASK={config.task}\n")
            f.write(f"TASK_SHORT={self._get_task_short(config.task)}\n")
            f.write(f"CHECKPOINT={self.checkpoint_path}\n")
            f.write(f"CATEGORY={config.category}\n")
            f.write(f"NUM_DISTRACTORS={config.num_distractors}\n")
            f.write(f"RANDOMIZE_DISTRACTORS={config.randomize_distractors}\n")
            f.write(f"NUM_EPISODES={config.num_episodes}\n")
            f.write(f"SEED={config.seed}\n")
            f.write(f"DISTRACTORS={' '.join(config.distractors)}\n")

    def _save_log_file(self, log_lines: List[str], results: List[EpisodeResult],
                       run_dir: str, mode: str, config: EvalConfig):
        """Save log file matching run_paired_eval.sh format."""
        log_path = os.path.join(run_dir, f"{mode}.log")
        num_success = sum(1 for r in results if r.success)
        success_rate = num_success / len(results) * 100 if results else 0

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Task: {config.task}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Seed: {config.seed}\n")
            f.write(f"Episodes: {config.num_episodes}\n")
            f.write("-" * 50 + "\n")
            for line in log_lines:
                f.write(line + "\n")
            f.write("-" * 50 + "\n")
            f.write(f"SUCCESS RATE: {success_rate:.1f}%\n")


    def _save_aggregated_report(self, results: List[ConfigResult], output_dir: str):
        """Save aggregated comparison_report.md, results.csv, summary.csv for all runs."""
        if not results:
            return

        config = results[0].config

        # Calculate aggregate statistics
        baseline_rates = [r.baseline_rate for r in results]
        cgvd_rates = [r.cgvd_rate for r in results]
        improvements = [r.improvement for r in results]

        baseline_mean = np.mean(baseline_rates)
        baseline_std = np.std(baseline_rates)
        cgvd_mean = np.mean(cgvd_rates)
        cgvd_std = np.std(cgvd_rates)
        avg_improvement = np.mean(improvements)

        # Save comparison_report.md
        report_path = os.path.join(output_dir, "comparison_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Paired Evaluation Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Configuration\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            f.write(f"| Model | Pi0 |\n")
            f.write(f"| Task | `{config.task}` |\n")
            f.write(f"| Category | {config.category} |\n")
            f.write(f"| Num Distractors | {config.num_distractors} |\n")
            f.write(f"| Episodes per run | {config.num_episodes} |\n")
            f.write(f"| Number of runs | {len(results)} |\n")
            f.write(f"| Seeds | {', '.join(str(r.config.seed) for r in results)} |\n")
            f.write(f"| Checkpoint | `{self.checkpoint_path}` |\n\n")

            f.write("### Distractors\n")
            for d in config.distractors[:config.num_distractors] if not config.randomize_distractors else config.distractors:
                f.write(f"- `{d}`\n")
            f.write("\n")

            f.write("## Results by Run\n\n")
            f.write("| Run | Seed | Baseline SR | CGVD SR | Δ SR | Baseline h-SR | CGVD h-SR | Δ h-SR |\n")
            f.write("|-----|------|-------------|---------|------|---------------|-----------|--------|\n")
            for i, r in enumerate(results):
                sign = "+" if r.improvement >= 0 else ""
                h_sign = "+" if r.hard_improvement >= 0 else ""
                f.write(f"| {i+1} | {r.config.seed} | {r.baseline_rate:.1f}% | {r.cgvd_rate:.1f}% | {sign}{r.improvement:.1f}% | {r.baseline_hard_success_rate:.1f}% | {r.cgvd_hard_success_rate:.1f}% | {h_sign}{r.hard_improvement:.1f}% |\n")
            f.write("\n")

            f.write("## Summary Statistics\n\n")
            f.write("### Success Rate (SR)\n\n")
            f.write("| Method | Mean | Std | Min | Max |\n")
            f.write("|--------|------|-----|-----|-----|\n")
            f.write(f"| Baseline | {baseline_mean:.1f}% | {baseline_std:.1f}% | {min(baseline_rates):.1f}% | {max(baseline_rates):.1f}% |\n")
            f.write(f"| CGVD | {cgvd_mean:.1f}% | {cgvd_std:.1f}% | {min(cgvd_rates):.1f}% | {max(cgvd_rates):.1f}% |\n\n")

            sign = "+" if avg_improvement >= 0 else ""
            f.write(f"**Average SR Improvement: {sign}{avg_improvement:.1f}%**\n\n")

            # Hard Success Rate statistics
            baseline_hard_rates = [r.baseline_hard_success_rate for r in results]
            cgvd_hard_rates = [r.cgvd_hard_success_rate for r in results]
            baseline_hard_mean = np.mean(baseline_hard_rates)
            baseline_hard_std = np.std(baseline_hard_rates)
            cgvd_hard_mean = np.mean(cgvd_hard_rates)
            cgvd_hard_std = np.std(cgvd_hard_rates)
            avg_hard_improvement = cgvd_hard_mean - baseline_hard_mean

            f.write("### Hard Success Rate (h-SR) - Success without collisions\n\n")
            f.write("| Method | Mean | Std | Min | Max |\n")
            f.write("|--------|------|-----|-----|-----|\n")
            f.write(f"| Baseline | {baseline_hard_mean:.1f}% | {baseline_hard_std:.1f}% | {min(baseline_hard_rates):.1f}% | {max(baseline_hard_rates):.1f}% |\n")
            f.write(f"| CGVD | {cgvd_hard_mean:.1f}% | {cgvd_hard_std:.1f}% | {min(cgvd_hard_rates):.1f}% | {max(cgvd_hard_rates):.1f}% |\n\n")

            h_sign = "+" if avg_hard_improvement >= 0 else ""
            f.write(f"**Average h-SR Improvement: {h_sign}{avg_hard_improvement:.1f}%**\n\n")

            # New metrics: Collision Rate
            f.write("## Collision Analysis\n\n")
            baseline_total_eps = sum(len(r.baseline_results) for r in results)
            cgvd_total_eps = sum(len(r.cgvd_results) for r in results)
            baseline_collision_eps = sum(1 for r in results for e in r.baseline_results if e.collision_count > 0)
            cgvd_collision_eps = sum(1 for r in results for e in r.cgvd_results if e.collision_count > 0)
            baseline_collision_rate = baseline_collision_eps / max(1, baseline_total_eps) * 100
            cgvd_collision_rate = cgvd_collision_eps / max(1, cgvd_total_eps) * 100

            f.write("| Method | Episodes with Collision | Collision Rate |\n")
            f.write("|--------|------------------------|----------------|\n")
            f.write(f"| Baseline | {baseline_collision_eps}/{baseline_total_eps} | {baseline_collision_rate:.1f}% |\n")
            f.write(f"| CGVD | {cgvd_collision_eps}/{cgvd_total_eps} | {cgvd_collision_rate:.1f}% |\n\n")

            # New metrics: Failure Mode Breakdown
            f.write("## Failure Mode Analysis\n\n")
            baseline_modes = {"success": 0, "never_reached": 0, "missed_grasp": 0, "dropped": 0}
            cgvd_modes = {"success": 0, "never_reached": 0, "missed_grasp": 0, "dropped": 0}
            for r in results:
                for e in r.baseline_results:
                    if e.failure_mode in baseline_modes:
                        baseline_modes[e.failure_mode] += 1
                for e in r.cgvd_results:
                    if e.failure_mode in cgvd_modes:
                        cgvd_modes[e.failure_mode] += 1

            f.write("| Failure Mode | Baseline | CGVD |\n")
            f.write("|--------------|----------|------|\n")
            for mode in ["success", "never_reached", "missed_grasp", "dropped"]:
                f.write(f"| {mode} | {baseline_modes[mode]} | {cgvd_modes[mode]} |\n")
            f.write("\n")

            # New metrics: CGVD Latency
            cgvd_times = [e.cgvd_time for r in results for e in r.cgvd_results if e.cgvd_time > 0]
            sam3_times = [e.sam3_time for r in results for e in r.cgvd_results if e.sam3_time > 0]
            lama_times = [e.lama_time for r in results for e in r.cgvd_results if e.lama_time > 0]

            if cgvd_times:
                f.write("## CGVD Latency Analysis\n\n")
                f.write("| Component | Mean (s) | Std (s) | Min (s) | Max (s) |\n")
                f.write("|-----------|----------|---------|---------|--------|\n")
                f.write(f"| Total CGVD | {np.mean(cgvd_times):.3f} | {np.std(cgvd_times):.3f} | {min(cgvd_times):.3f} | {max(cgvd_times):.3f} |\n")
                if sam3_times:
                    f.write(f"| SAM3 | {np.mean(sam3_times):.3f} | {np.std(sam3_times):.3f} | {min(sam3_times):.3f} | {max(sam3_times):.3f} |\n")
                if lama_times:
                    f.write(f"| LaMa | {np.mean(lama_times):.3f} | {np.std(lama_times):.3f} | {min(lama_times):.3f} | {max(lama_times):.3f} |\n")
                f.write("\n")

            f.write("## Per-Episode Details\n\n")
            for i, r in enumerate(results):
                f.write(f"### Run {i+1} (Seed: {r.config.seed})\n\n")
                f.write("| Episode | B-SR | C-SR | B-h-SR | C-h-SR | B-Coll | C-Coll | B-Failure | C-Failure |\n")
                f.write("|---------|------|------|--------|--------|--------|--------|-----------|----------|\n")
                for j, (b, c) in enumerate(zip(r.baseline_results, r.cgvd_results)):
                    b_str = "✓" if b.success else "✗"
                    c_str = "✓" if c.success else "✗"
                    b_hard_str = "✓" if b.hard_success else "✗"
                    c_hard_str = "✓" if c.hard_success else "✗"
                    f.write(f"| {j+1} | {b_str} | {c_str} | {b_hard_str} | {c_hard_str} | {b.collision_count} | {c.collision_count} | {b.failure_mode} | {c.failure_mode} |\n")
                f.write("\n")

        # Save results.csv (per-episode) with new metrics
        results_path = os.path.join(output_dir, "results.csv")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("run,seed,episode,episode_id,baseline_success,cgvd_success,"
                    "baseline_hard_success,cgvd_hard_success,baseline_time,cgvd_time,"
                    "baseline_collisions,cgvd_collisions,baseline_failure_mode,cgvd_failure_mode,"
                    "cgvd_pipeline_time,sam3_time,lama_time\n")
            for r in results:
                for i, (b, c) in enumerate(zip(r.baseline_results, r.cgvd_results)):
                    f.write(f"{r.config.run_index},{r.config.seed},{i},{b.episode_id},"
                           f"{int(b.success)},{int(c.success)},"
                           f"{int(b.hard_success)},{int(c.hard_success)},{b.episode_time:.2f},{c.episode_time:.2f},"
                           f"{b.collision_count},{c.collision_count},{b.failure_mode},{c.failure_mode},"
                           f"{c.cgvd_time:.3f},{c.sam3_time:.3f},{c.lama_time:.3f}\n")

        # Save summary.csv (per-run) with new metrics
        summary_path = os.path.join(output_dir, "summary.csv")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("run,seed,baseline_success_rate,cgvd_success_rate,improvement,"
                    "baseline_hard_success_rate,cgvd_hard_success_rate,hard_improvement,"
                    "baseline_collision_rate,cgvd_collision_rate,avg_cgvd_time\n")
            for r in results:
                baseline_collision_rate = sum(1 for e in r.baseline_results if e.collision_count > 0) / max(1, len(r.baseline_results)) * 100
                cgvd_collision_rate = sum(1 for e in r.cgvd_results if e.collision_count > 0) / max(1, len(r.cgvd_results)) * 100
                avg_cgvd_time = np.mean([e.cgvd_time for e in r.cgvd_results if e.cgvd_time > 0]) if any(e.cgvd_time > 0 for e in r.cgvd_results) else 0.0
                f.write(f"{r.config.run_index},{r.config.seed},{r.baseline_rate:.1f},"
                       f"{r.cgvd_rate:.1f},{r.improvement:.1f},"
                       f"{r.baseline_hard_success_rate:.1f},{r.cgvd_hard_success_rate:.1f},{r.hard_improvement:.1f},"
                       f"{baseline_collision_rate:.1f},{cgvd_collision_rate:.1f},{avg_cgvd_time:.3f}\n")

    def run_sweep(
        self,
        configs: List[EvalConfig],
        output_dir: Optional[str] = None,
    ) -> Generator[ConfigResult, None, None]:
        """Run sweep through all configurations.

        Args:
            configs: List of configurations to run
            output_dir: Optional directory to save results

        Yields:
            ConfigResult for each configuration
        """
        total = len(configs)
        start_time = time.time()

        # Pre-compute output directories for each (category, num_distractors) combo
        # so all runs share the same timestamp
        config_dirs = {}
        if output_dir:
            for config in configs:
                key = (config.category, config.num_distractors)
                if key not in config_dirs:
                    task_short = self._get_task_short(config.task)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    # Count runs for this config
                    num_runs = sum(1 for c in configs if (c.category, c.num_distractors) == key)
                    config_dirs[key] = os.path.join(
                        output_dir,
                        task_short,
                        config.category,
                        f"n{config.num_distractors}_e{config.num_episodes}_r{num_runs}_{timestamp}",
                    )
                    # Small delay to ensure unique timestamps
                    time.sleep(0.01)

        # Track results per config group for aggregated reports
        config_results: Dict[Tuple[str, int], List[ConfigResult]] = {}

        for i, config in enumerate(configs):
            print(f"\n{'='*70}")
            print(f"Configuration {i+1}/{total}")
            print(f"{'='*70}")

            # Get pre-computed output directory
            key = (config.category, config.num_distractors)
            config_output_dir = config_dirs.get(key) if output_dir else None

            result = self.run_configuration(config, config_output_dir)
            yield result

            # Track for aggregated reports
            if key not in config_results:
                config_results[key] = []
            config_results[key].append(result)

            # Save aggregated reports when all runs for this config are done
            if config_output_dir:
                expected_runs = sum(1 for c in configs if (c.category, c.num_distractors) == key)
                if len(config_results[key]) == expected_runs:
                    self._save_aggregated_report(config_results[key], config_output_dir)

            # Progress update
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{total} ({elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m remaining)")


def load_distractors_from_file(filepath: str, category: str, num_distractors: int) -> Tuple[List[str], List[str]]:
    """Load distractors from file.

    Args:
        filepath: Path to distractors file (or base path without category suffix)
        category: Distractor category (semantic, visual, control)
        num_distractors: Number of distractors to use (0 = all)

    Returns:
        Tuple of (distractor asset IDs, CGVD distractor names)
    """
    # Try category-specific file first
    base_path = Path(filepath)
    if base_path.exists():
        file_to_use = base_path
    else:
        # Try with category suffix
        categorized = base_path.parent / f"{base_path.stem}_{category}.txt"
        if categorized.exists():
            file_to_use = categorized
        else:
            print(f"Warning: No distractors file found for {filepath} / {category}")
            return [], []

    # Read distractors
    distractors = []
    with open(file_to_use) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                distractors.append(line)

    # Limit if specified
    if num_distractors > 0:
        distractors = distractors[:num_distractors]

    # Derive CGVD names
    cgvd_names = []
    for asset_id in distractors:
        asset_id_clean = asset_id.split(":")[0]
        parts = asset_id_clean.split("_")
        if len(parts) >= 3:
            if parts[0] == "ycb":
                name = " ".join(parts[2:])
            else:
                name = " ".join(parts[1:-1])
            if name and name not in cgvd_names:
                cgvd_names.append(name)
        elif len(parts) == 2:
            name = parts[0]
            if name and name not in cgvd_names:
                cgvd_names.append(name)

    return distractors, cgvd_names


def generate_configs(args) -> List[EvalConfig]:
    """Generate list of configurations from command-line args.

    Order: count → category → run
    This ensures all runs for a given distractor count complete before moving to the next count.
    """
    configs = []

    # Determine task short name for distractor file lookup
    task_to_base = {
        "widowx_carrot_on_plate": "carrot",
        "widowx_banana_on_plate": "banana",
        "widowx_put_eggplant_in_basket": "eggplant",
        "widowx_spoon_on_towel": "spoon",
        "widowx_stack_cube": "cube",
    }
    task_base = task_to_base.get(args.task, args.task.split("_")[1] if "_" in args.task else args.task)

    # Order: count → category → run
    # Completes all work for count=0 before moving to count=2, etc.
    for num_dist in args.distractor_counts:
        for category in args.categories:
            # Load distractors for this category/count
            distractor_file = f"scripts/clutter_eval/distractors/distractors_{task_base}_{category}.txt"

            if args.randomize_distractors:
                # Load full pool (pass 0 to get all), will sample num_dist per episode
                distractors, cgvd_names = load_distractors_from_file(distractor_file, category, 0)
            else:
                # Load only the first num_dist distractors
                distractors, cgvd_names = load_distractors_from_file(distractor_file, category, num_dist)

            for run_idx in range(args.runs):
                seed = args.start_seed + run_idx
                configs.append(EvalConfig(
                    task=args.task,
                    category=category,
                    num_distractors=num_dist,
                    seed=seed,
                    num_episodes=args.episodes,
                    run_index=run_idx,
                    distractors=distractors,
                    cgvd_distractor_names=cgvd_names if num_dist > 0 else [],
                    randomize_distractors=args.randomize_distractors,
                    distractor_pool_size=len(distractors),
                ))

    return configs


def save_sweep_results(results: List[ConfigResult], output_dir: str):
    """Save sweep results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Summary CSV with new metrics
    summary_path = os.path.join(output_dir, "sweep_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("task,category,num_distractors,run,seed,baseline_rate,cgvd_rate,improvement,"
                "baseline_hard_sr,cgvd_hard_sr,hard_improvement,"
                "baseline_collisions,cgvd_collisions,baseline_never_reached,baseline_missed_grasp,baseline_dropped,"
                "cgvd_never_reached,cgvd_missed_grasp,cgvd_dropped,avg_cgvd_time,avg_sam3_time,avg_lama_time\n")
        for r in results:
            # Calculate collision rates
            baseline_collisions = sum(1 for e in r.baseline_results if e.collision_count > 0)
            cgvd_collisions = sum(1 for e in r.cgvd_results if e.collision_count > 0)

            # Count failure modes
            baseline_never_reached = sum(1 for e in r.baseline_results if e.failure_mode == "never_reached")
            baseline_missed_grasp = sum(1 for e in r.baseline_results if e.failure_mode == "missed_grasp")
            baseline_dropped = sum(1 for e in r.baseline_results if e.failure_mode == "dropped")
            cgvd_never_reached = sum(1 for e in r.cgvd_results if e.failure_mode == "never_reached")
            cgvd_missed_grasp = sum(1 for e in r.cgvd_results if e.failure_mode == "missed_grasp")
            cgvd_dropped = sum(1 for e in r.cgvd_results if e.failure_mode == "dropped")

            # Calculate average CGVD timing
            cgvd_times = [e.cgvd_time for e in r.cgvd_results if e.cgvd_time > 0]
            sam3_times = [e.sam3_time for e in r.cgvd_results if e.sam3_time > 0]
            lama_times = [e.lama_time for e in r.cgvd_results if e.lama_time > 0]
            avg_cgvd = np.mean(cgvd_times) if cgvd_times else 0.0
            avg_sam3 = np.mean(sam3_times) if sam3_times else 0.0
            avg_lama = np.mean(lama_times) if lama_times else 0.0

            f.write(f"{r.config.task},{r.config.category},{r.config.num_distractors},"
                   f"{r.config.run_index},{r.config.seed},"
                   f"{r.baseline_rate:.1f},{r.cgvd_rate:.1f},{r.improvement:.1f},"
                   f"{r.baseline_hard_success_rate:.1f},{r.cgvd_hard_success_rate:.1f},{r.hard_improvement:.1f},"
                   f"{baseline_collisions},{cgvd_collisions},"
                   f"{baseline_never_reached},{baseline_missed_grasp},{baseline_dropped},"
                   f"{cgvd_never_reached},{cgvd_missed_grasp},{cgvd_dropped},"
                   f"{avg_cgvd:.3f},{avg_sam3:.3f},{avg_lama:.3f}\n")

    # Detailed JSON with new metrics
    json_path = os.path.join(output_dir, "sweep_results.json")
    json_results = []
    for r in results:
        json_results.append({
            "config": {
                "task": r.config.task,
                "category": r.config.category,
                "num_distractors": r.config.num_distractors,
                "seed": r.config.seed,
                "run_index": r.config.run_index,
                "num_episodes": r.config.num_episodes,
            },
            "baseline": {
                "success_rate": r.baseline_rate,
                "hard_success_rate": r.baseline_hard_success_rate,
                "successes": r.baseline_successes,
                "hard_successes": sum(1 for e in r.baseline_results if e.hard_success),
                "total": r.baseline_total,
                "collision_rate": sum(1 for e in r.baseline_results if e.collision_count > 0) / max(1, len(r.baseline_results)) * 100,
                "failure_modes": {
                    "success": sum(1 for e in r.baseline_results if e.failure_mode == "success"),
                    "never_reached": sum(1 for e in r.baseline_results if e.failure_mode == "never_reached"),
                    "missed_grasp": sum(1 for e in r.baseline_results if e.failure_mode == "missed_grasp"),
                    "dropped": sum(1 for e in r.baseline_results if e.failure_mode == "dropped"),
                },
                "episodes": [{
                    "success": e.success,
                    "hard_success": e.hard_success,
                    "steps": e.steps,
                    "time": e.episode_time,
                    "collision_count": e.collision_count,
                    "failure_mode": e.failure_mode,
                } for e in r.baseline_results]
            },
            "cgvd": {
                "success_rate": r.cgvd_rate,
                "hard_success_rate": r.cgvd_hard_success_rate,
                "successes": r.cgvd_successes,
                "hard_successes": sum(1 for e in r.cgvd_results if e.hard_success),
                "total": r.cgvd_total,
                "collision_rate": sum(1 for e in r.cgvd_results if e.collision_count > 0) / max(1, len(r.cgvd_results)) * 100,
                "failure_modes": {
                    "success": sum(1 for e in r.cgvd_results if e.failure_mode == "success"),
                    "never_reached": sum(1 for e in r.cgvd_results if e.failure_mode == "never_reached"),
                    "missed_grasp": sum(1 for e in r.cgvd_results if e.failure_mode == "missed_grasp"),
                    "dropped": sum(1 for e in r.cgvd_results if e.failure_mode == "dropped"),
                },
                "avg_cgvd_time": np.mean([e.cgvd_time for e in r.cgvd_results if e.cgvd_time > 0]) if any(e.cgvd_time > 0 for e in r.cgvd_results) else 0.0,
                "avg_sam3_time": np.mean([e.sam3_time for e in r.cgvd_results if e.sam3_time > 0]) if any(e.sam3_time > 0 for e in r.cgvd_results) else 0.0,
                "avg_lama_time": np.mean([e.lama_time for e in r.cgvd_results if e.lama_time > 0]) if any(e.lama_time > 0 for e in r.cgvd_results) else 0.0,
                "episodes": [{
                    "success": e.success,
                    "hard_success": e.hard_success,
                    "steps": e.steps,
                    "time": e.episode_time,
                    "collision_count": e.collision_count,
                    "failure_mode": e.failure_mode,
                    "cgvd_time": e.cgvd_time,
                    "sam3_time": e.sam3_time,
                    "lama_time": e.lama_time,
                } for e in r.cgvd_results]
            },
            "improvement": r.improvement,
            "hard_improvement": r.hard_improvement,
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - {summary_path}")
    print(f"  - {json_path}")


def print_sweep_summary(results: List[ConfigResult]):
    """Print summary of sweep results."""
    print("\n" + "=" * 120)
    print("SWEEP SUMMARY")
    print("=" * 120)

    # Group by category and num_distractors
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (r.config.category, r.config.num_distractors)
        grouped[key].append(r)

    print(f"\n{'Category':<12} {'Dist':<6} {'Runs':<6} {'Baseline SR':<12} {'CGVD SR':<12} {'Δ SR':<8} {'h-SR(B)':<8} {'h-SR(C)':<8} {'Δ h-SR':<8} {'CGVD-T':<8}")
    print("-" * 120)

    for (category, num_dist), group in sorted(grouped.items()):
        baseline_rates = [r.baseline_rate for r in group]
        cgvd_rates = [r.cgvd_rate for r in group]
        improvements = [r.improvement for r in group]

        baseline_mean = np.mean(baseline_rates)
        baseline_std = np.std(baseline_rates)
        cgvd_mean = np.mean(cgvd_rates)
        cgvd_std = np.std(cgvd_rates)
        imp_mean = np.mean(improvements)

        # Hard success rates
        baseline_hard_rates = [r.baseline_hard_success_rate for r in group]
        cgvd_hard_rates = [r.cgvd_hard_success_rate for r in group]
        baseline_hard_mean = np.mean(baseline_hard_rates)
        cgvd_hard_mean = np.mean(cgvd_hard_rates)
        hard_imp_mean = cgvd_hard_mean - baseline_hard_mean

        # Average CGVD time
        cgvd_times = [e.cgvd_time for r in group for e in r.cgvd_results if e.cgvd_time > 0]
        avg_cgvd_time = np.mean(cgvd_times) if cgvd_times else 0.0

        print(f"{category:<12} {num_dist:<6} {len(group):<6} "
              f"{baseline_mean:>5.1f}±{baseline_std:<5.1f} "
              f"{cgvd_mean:>5.1f}±{cgvd_std:<5.1f} "
              f"{imp_mean:>+6.1f}% "
              f"{baseline_hard_mean:>6.1f}% "
              f"{cgvd_hard_mean:>6.1f}% "
              f"{hard_imp_mean:>+6.1f}% "
              f"{avg_cgvd_time:>6.2f}s")

    print("=" * 120)

    # Print failure mode summary
    print("\nFAILURE MODE BREAKDOWN")
    print("-" * 60)
    baseline_modes = {"success": 0, "never_reached": 0, "missed_grasp": 0, "dropped": 0}
    cgvd_modes = {"success": 0, "never_reached": 0, "missed_grasp": 0, "dropped": 0}
    for r in results:
        for e in r.baseline_results:
            if e.failure_mode in baseline_modes:
                baseline_modes[e.failure_mode] += 1
        for e in r.cgvd_results:
            if e.failure_mode in cgvd_modes:
                cgvd_modes[e.failure_mode] += 1

    print(f"{'Mode':<15} {'Baseline':<10} {'CGVD':<10}")
    print("-" * 35)
    for mode in ["success", "never_reached", "missed_grasp", "dropped"]:
        print(f"{mode:<15} {baseline_modes[mode]:<10} {cgvd_modes[mode]:<10}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for configuration sweeps")

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
    parser.add_argument("--dry_run", action="store_true",
                       help="Print configurations without running")

    # Recording and debug
    parser.add_argument("--recording", action="store_true",
                       help="Save video recordings of each episode")
    parser.add_argument("--cgvd_save_debug", action="store_true",
                       help="Save CGVD debug images (original/mask/distilled)")
    parser.add_argument("--cgvd_verbose", action="store_true",
                       help="Print verbose CGVD output")
    parser.add_argument("--save_attention", action="store_true",
                       help="Save attention map visualizations for each episode")

    # Distractor randomization
    parser.add_argument("--randomize_distractors", action="store_true",
                       help="Randomly sample distractors from pool each episode")

    # CGVD thresholds
    parser.add_argument("--cgvd_safe_threshold", type=float, default=0.6,
                       help="Threshold for safe-set (target/anchor) detection (default: 0.6)")
    parser.add_argument("--cgvd_robot_threshold", type=float, default=0.3,
                       help="Threshold for robot arm detection (default: 0.3)")
    parser.add_argument("--cgvd_distractor_threshold", type=float, default=0.20,
                       help="Threshold for distractor detection (default: 0.20)")

    args = parser.parse_args()

    # Generate configurations
    configs = generate_configs(args)

    print("=" * 70)
    print("BATCH EVALUATION CONFIGURATION")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Categories: {args.categories}")
    print(f"Distractor counts: {args.distractor_counts}")
    print(f"Episodes per config: {args.episodes}")
    print(f"Runs per config: {args.runs}")
    print(f"Total configurations: {len(configs)}")
    print(f"Total episodes: {len(configs) * args.episodes * 2}")  # x2 for baseline + CGVD
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would run the following configurations:")
        for i, c in enumerate(configs[:10]):  # Show first 10
            print(f"  {i+1}. {c.task} | {c.category} | {c.num_distractors} dist | seed={c.seed}")
        if len(configs) > 10:
            print(f"  ... and {len(configs) - 10} more")
        return

    # Determine output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"logs/clutter_eval/batch/{args.task}/{timestamp}"

    # Run batch evaluation
    evaluator = BatchEvaluator(
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
        save_attention=args.save_attention,
    )

    results = list(evaluator.run_sweep(configs, args.output_dir))

    # Save and summarize
    save_sweep_results(results, args.output_dir)
    print_sweep_summary(results)

    # Final timing
    print(f"\nModel load time: {evaluator.model_load_time:.2f}s (loaded ONCE for {len(configs)} configs)")
    total_episodes = sum(len(r.baseline_results) + len(r.cgvd_results) for r in results)
    print(f"Total episodes run: {total_episodes}")


if __name__ == "__main__":
    main()
