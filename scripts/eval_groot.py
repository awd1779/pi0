"""
GR00T N1.6 Evaluation Script for SimplerEnv.

This script evaluates GR00T on SimplerEnv tasks. It should be run in the
'groot' conda environment which has Isaac-GR00T installed.

Usage:
    conda activate groot
    python scripts/eval_groot.py \
        --task widowx_carrot_on_plate \
        --num_episodes 10 \
        --use_cgvd

Environment Setup:
    conda create -n groot python=3.10 -y
    conda activate groot
    git clone https://github.com/NVIDIA/Isaac-GR00T.git ~/Isaac-GR00T
    cd ~/Isaac-GR00T && pip install -e .
    pip install -e ~/allenzren_SimplerEnv
    pip install -e ~/allenzren_SimplerEnv/ManiSkill2_real2sim
    pip install -e ~/open-pi-zero
"""

# ============================================================================
# CRITICAL: Monkey-patch transformers.image_utils BEFORE any other imports
# The Eagle processor requires VideoInput which doesn't exist in transformers 4.53.0
# ============================================================================
import os
from typing import List, Union

# First, patch the cached Eagle processor files if they exist
def _patch_eagle_files():
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/Eagle-Block2A-2B-v2"
    )
    if not os.path.exists(cache_dir):
        return

    # Patch 1: Fix VideoInput import in processing_eagle3_vl.py
    proc_path = os.path.join(cache_dir, "processing_eagle3_vl.py")
    if os.path.exists(proc_path):
        with open(proc_path, 'r') as f:
            content = f.read()
        if "# VideoInput is not available in transformers" not in content:
            old_import = "from transformers.image_utils import ImageInput, VideoInput, get_image_size, to_numpy_array"
            new_import = """from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
# VideoInput is not available in transformers 4.53.0, define it locally
from typing import List, Union
import numpy as np
import torch
from PIL import Image
VideoInput = Union[List[Image.Image], List[np.ndarray], List[torch.Tensor]]"""
            if old_import in content:
                content = content.replace(old_import, new_import)
                with open(proc_path, 'w') as f:
                    f.write(content)
                print("[GR00T] Patched processing_eagle3_vl.py to fix VideoInput import")

    # Patch 2: Disable fast image processor (requires newer transformers)
    fast_path = os.path.join(cache_dir, "image_processing_eagle3_vl_fast.py")
    if os.path.exists(fast_path):
        with open(fast_path, 'r') as f:
            content = f.read()
        if "This file is disabled" not in content:
            new_content = '''# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
#
# This file is disabled because it requires a newer version of transformers.
# The slow image processor will be used instead.

raise ImportError(
    "Eagle3_VLImageProcessorFast requires transformers >= 4.57.0. "
    "Falling back to slow image processor."
)
'''
            with open(fast_path, 'w') as f:
                f.write(new_content)
            print("[GR00T] Disabled fast image processor (requires newer transformers)")

_patch_eagle_files()

# Also add VideoInput to transformers.image_utils in case it gets imported elsewhere
try:
    import numpy as np
    import torch
    from PIL import Image
    import transformers.image_utils as _image_utils
    if not hasattr(_image_utils, 'VideoInput'):
        _image_utils.VideoInput = Union[List[Image.Image], List[np.ndarray], List[torch.Tensor]]
except ImportError:
    pass
# ============================================================================

import gc
import random
import time

import imageio
import simpler_env


from src.model.vla.groot import GR00TInference
from src.agent.env_adapter.groot_simpler import (
    GR00TBridgeSimplerAdapter,
    GR00TFractalSimplerAdapter,
)


def wrap_env_with_cgvd(env, args):
    """Wrap environment with CGVD wrapper if enabled."""
    if not args.use_cgvd:
        return env

    from src.cgvd import CGVDWrapper

    print(f"[CGVD] Wrapping environment with CGVD wrapper (LaMa inpainting)")
    print(f"[CGVD]   update_freq={args.cgvd_update_freq}")
    print(f"[CGVD]   presence_threshold={args.cgvd_presence_threshold}")
    print(f"[CGVD]   use_mock_segmenter={args.cgvd_use_mock}")
    if args.cgvd_distractor_names:
        print(f"[CGVD]   distractor_names={args.cgvd_distractor_names}")
    else:
        print(f"[CGVD]   No distractors specified, CGVD will pass through unchanged")
    if getattr(args, 'cgvd_disable_safeset', False):
        print(f"[CGVD]   ABLATION: Safe-set protection DISABLED")
    if getattr(args, 'cgvd_disable_inpaint', False):
        print(f"[CGVD]   ABLATION: Inpainting DISABLED (mean-color fill)")

    # Use output_dir for debug images if specified
    if args.output_dir and args.output_dir != ".":
        debug_dir = os.path.join(args.output_dir, "cgvd_debug")
    else:
        debug_dir = os.path.join("cgvd_debug", args.task)

    return CGVDWrapper(
        env,
        update_freq=args.cgvd_update_freq,
        presence_threshold=args.cgvd_presence_threshold,
        use_mock_segmenter=args.cgvd_use_mock,
        use_server_segmenter=args.cgvd_use_server,
        include_robot=True,
        verbose=args.cgvd_verbose,
        save_debug_images=args.cgvd_save_debug,
        debug_dir=debug_dir,
        distractor_names=args.cgvd_distractor_names,
        cache_distractor_once=True,
        robot_presence_threshold=args.cgvd_robot_threshold,
        distractor_presence_threshold=args.cgvd_distractor_threshold,
        # Ablation flags
        disable_safeset=getattr(args, 'cgvd_disable_safeset', False),
        disable_inpaint=getattr(args, 'cgvd_disable_inpaint', False),
    )


def get_embodiment_from_task(task: str) -> str:
    """Determine embodiment type from task name."""
    if task.startswith("widowx"):
        return "bridge"
    elif task.startswith("google_robot"):
        return "fractal"
    else:
        raise ValueError(f"Unknown task type: {task}")


def create_env_adapter(embodiment: str, args):
    """Create the appropriate environment adapter."""
    if embodiment == "bridge":
        return GR00TBridgeSimplerAdapter(image_size=(256, 256))
    elif embodiment == "fractal":
        return GR00TFractalSimplerAdapter(image_size=(256, 320))
    else:
        raise ValueError(f"Unknown embodiment: {embodiment}")


def run_episode(args, env, env_adapter, model, episode_idx, act_steps=4):
    """Run a single episode and return (success, steps, inference_times, episode_time)."""
    episode_start = time.time()
    env_adapter.reset()
    model.reset()  # Reset GR00T policy state between episodes

    # Use different episode_id for each run to get variety
    episode_id = (args.seed + episode_idx) % 21
    env_reset_options = {
        "obj_init_options": {"episode_id": episode_id},
    }
    obs, reset_info = env.reset(options=env_reset_options)
    instruction = env.unwrapped.get_language_instruction()

    video_writer = None
    video_path = None
    if args.recording:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.makedirs(args.output_dir, exist_ok=True)
        suffix = "_cgvd" if args.use_cgvd else "_baseline"
        video_path = os.path.join(args.output_dir, f"groot_{args.task}{suffix}_ep{episode_idx}.mp4")
        video_writer = imageio.get_writer(video_path)

    print(f"\n--- Episode {episode_idx + 1}/{args.num_episodes} (id={episode_id}) ---")
    print(f"Instruction: {instruction}")

    cnt_step = 0
    inference_times = []
    success = False

    while True:
        # Preprocess observation
        inputs = env_adapter.preprocess(env, obs, instruction)

        # Run GR00T inference
        start_inference_time = time.time()
        actions = model.forward(
            images=inputs["image"],
            state=inputs["state"],
            instruction=inputs["instruction"],
        )
        if cnt_step > 0:
            inference_times.append(time.time() - start_inference_time)

        # Postprocess actions
        env_actions = env_adapter.postprocess(actions)

        # Execute action steps
        for env_action in env_actions[:act_steps]:
            obs, reward, success, truncated, info = env.step(env_action)
            cnt_step += 1
            if truncated:
                break

        # Save frame for video
        if video_writer is not None:
            video_writer.append_data(env_adapter.get_video_frame(env, obs))

        # Update instruction if changed (long-horizon tasks)
        new_instruction = env.unwrapped.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction

        # Check termination
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
    result_str = "SUCCESS" if success else "FAILED"
    print(f"Episode {episode_idx + 1}: {result_str} (steps={cnt_step}, time={episode_time:.2f}s)")

    return success, cnt_step, inference_times, episode_time


def main(args):
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine embodiment from task
    embodiment = get_embodiment_from_task(args.task)

    # Auto-select model path based on embodiment if not specified
    if args.model_path is None:
        if embodiment == "bridge":
            args.model_path = "nvidia/GR00T-N1.6-bridge"
        elif embodiment == "fractal":
            args.model_path = "nvidia/GR00T-N1.6-fractal"
        else:
            raise ValueError(f"Cannot auto-select model for embodiment: {embodiment}")

    print(f"[GR00T] Task: {args.task}")
    print(f"[GR00T] Embodiment: {embodiment}")

    # Device setup
    device = f"cuda:{args.gpu_id}"
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    print(f"[GR00T] Using device: {device}, dtype: {dtype}")

    # Load GR00T model
    model = GR00TInference(
        model_path=args.model_path,
        embodiment=embodiment,
        device=device,
        dtype=dtype,
    )

    # Create SimplerEnv
    env = simpler_env.make(args.task)

    # Add distractors if specified
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

    # Optionally wrap with CGVD
    env = wrap_env_with_cgvd(env, args)

    # Create environment adapter
    env_adapter = create_env_adapter(embodiment, args)

    # Run episodes
    successes = []
    all_steps = []
    all_inference_times = []
    all_episode_times = []

    for ep_idx in range(args.num_episodes):
        success, steps, inference_times, episode_time = run_episode(
            args, env, env_adapter, model, ep_idx, act_steps=args.act_steps
        )
        successes.append(success)
        all_steps.append(steps)
        all_inference_times.extend(inference_times)
        all_episode_times.append(episode_time)

        # CUDA memory tracking
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated(args.gpu_id) / 1e9
            reserved_gb = torch.cuda.memory_reserved(args.gpu_id) / 1e9
            print(f"[Memory] Episode {ep_idx + 1}: allocated={allocated_gb:.2f}GB, reserved={reserved_gb:.2f}GB")

        # Optional CUDA cache clearing
        if args.clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            if args.num_episodes > 1:
                print(f"[Memory] Cleared CUDA cache")

    # Print summary
    num_success = sum(successes)
    success_rate = num_success / args.num_episodes * 100

    print("\n\n" + "=" * 50)
    print("EVALUATION SUMMARY (GR00T)")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Task: {args.task}")
    print(f"Embodiment: {embodiment}")
    print(f"CGVD: {'enabled' if args.use_cgvd else 'disabled'}")
    if args.use_cgvd:
        print(f"  blur_sigma={args.cgvd_blur_sigma}")
    print("-" * 50)
    print(f"Episodes: {args.num_episodes}")
    print(f"Successes: {num_success}")
    print(f"SUCCESS RATE: {success_rate:.1f}%")
    print("-" * 50)
    print(f"Results: {['S' if s else 'F' for s in successes]}")
    print(f"Avg steps per episode: {np.mean(all_steps):.1f}")
    if all_inference_times:
        print(f"Avg inference time: {np.mean(all_inference_times):.3f}s")
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_reserved(args.gpu_id) / 1024 ** 3:.2f} GB")
    print("-" * 50)
    print("TIMING PROFILE")
    print(f"Episode times: {[f'{t:.1f}s' for t in all_episode_times]}")
    print(f"Avg episode time: {np.mean(all_episode_times):.2f}s")
    if len(all_episode_times) > 1:
        time_drift = all_episode_times[-1] - all_episode_times[0]
        drift_pct = (time_drift / all_episode_times[0]) * 100 if all_episode_times[0] > 0 else 0
        print(f"Time drift (last - first): {time_drift:+.2f}s ({drift_pct:+.1f}%)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GR00T evaluation on SimplerEnv")
    parser.add_argument(
        "--task",
        type=str,
        default="widowx_carrot_on_plate",
        choices=[
            # Bridge/WidowX tasks
            "widowx_carrot_on_plate",
            "widowx_put_eggplant_in_basket",
            "widowx_spoon_on_towel",
            "widowx_stack_cube",
            # Fractal/Google Robot tasks
            "google_robot_pick_horizontal_coke_can",
            "google_robot_pick_vertical_coke_can",
            "google_robot_pick_standing_coke_can",
            "google_robot_move_near_v0",
            "google_robot_open_drawer",
            "google_robot_close_drawer",
            "google_robot_place_apple_in_closed_top_drawer",
        ],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="GR00T model path (default: auto-select based on task)",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--act_steps", type=int, default=1, help="Number of action steps to execute per inference (default: 1, matching NVIDIA benchmark)")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument(
        "--clear_cuda_cache",
        action="store_true",
        help="Clear CUDA cache between episodes",
    )
    parser.add_argument("--recording", action="store_true")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output videos")

    # Distractor arguments
    parser.add_argument(
        "--distractors",
        type=str,
        nargs="*",
        default=[],
        help="Distractor object IDs to add (e.g., apple orange sponge)",
    )
    parser.add_argument(
        "--distractor_scale",
        type=float,
        default=None,
        help="Scale multiplier for ALL distractor objects",
    )
    parser.add_argument(
        "--external_asset_scale",
        type=float,
        default=None,
        help="Scale multiplier for rc_* and ycb_* objects only (default: 0.1)",
    )
    parser.add_argument(
        "--num_distractors",
        type=int,
        default=None,
        help="Number of distractors to sample per episode (requires --randomize_distractors)"
    )
    parser.add_argument(
        "--randomize_distractors",
        action="store_true",
        help="Randomly sample distractors from pool each episode"
    )

    # CGVD arguments
    parser.add_argument(
        "--use_cgvd",
        action="store_true",
        help="Enable Concept-Gated Visual Distillation",
    )
    parser.add_argument(
        "--cgvd_blur_sigma",
        type=float,
        default=15.0,
        help="Gaussian blur sigma for CGVD background (default: 15.0)",
    )
    parser.add_argument(
        "--cgvd_update_freq",
        type=int,
        default=10,
        help="Frames between CGVD mask updates (default: 10 = 1Hz at 10fps)",
    )
    parser.add_argument(
        "--cgvd_presence_threshold",
        type=float,
        default=0.4,
        help="SAM3 confidence threshold for CGVD (default: 0.4)",
    )
    parser.add_argument(
        "--cgvd_use_mock",
        action="store_true",
        help="Use mock segmenter for CGVD testing",
    )
    parser.add_argument(
        "--cgvd_use_server",
        action="store_true",
        help="Use SAM3 server (for environments with transformers version conflicts)",
    )
    parser.add_argument(
        "--cgvd_feather_edges",
        action="store_true",
        help="Apply edge feathering to CGVD mask transitions",
    )
    parser.add_argument(
        "--cgvd_verbose",
        action="store_true",
        help="Print CGVD debug information",
    )
    parser.add_argument(
        "--cgvd_save_debug",
        action="store_true",
        help="Save debug images showing original/mask/distilled",
    )
    parser.add_argument(
        "--cgvd_distractor_names",
        type=str,
        nargs="*",
        default=[],
        help="Object names to blur as distractors",
    )
    parser.add_argument(
        "--cgvd_darken_strength",
        type=float,
        default=0.0,
        help="Blend blurred distractors toward background (0=pure blur, 1=solid bg)",
    )
    parser.add_argument(
        "--cgvd_robot_threshold",
        type=float,
        default=0.05,
        help="Threshold for robot arm detection (default 0.05, very permissive)",
    )
    parser.add_argument(
        "--cgvd_distractor_threshold",
        type=float,
        default=0.3,
        help="Threshold for distractor detection (default 0.3, stricter to avoid false positives)",
    )
    # Ablation flags for CGVD component studies
    parser.add_argument(
        "--cgvd_disable_safeset",
        action="store_true",
        help="Ablation: Disable safe-set protection (mask distractors without protecting target/anchor)",
    )
    parser.add_argument(
        "--cgvd_disable_inpaint",
        action="store_true",
        help="Ablation: Disable LaMa inpainting (use mean-color fill instead)",
    )

    args = parser.parse_args()

    # Auto-derive cgvd_distractor_names from distractors if not explicitly provided
    if args.distractors and args.use_cgvd and not args.cgvd_distractor_names:
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
            print(f"[CGVD] Auto-derived distractor names from assets: {derived_names}")

    main(args)
