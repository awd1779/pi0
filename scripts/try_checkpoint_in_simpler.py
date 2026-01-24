import os
import random
import time

import hydra
import imageio
import numpy as np
import simpler_env
import torch
from omegaconf import OmegaConf

from src.model.vla.pizero import PiZeroInference
from src.utils.monitor import log_allocated_gpu_memory, log_execution_time


def wrap_env_with_cgvd(env, args):
    """Wrap environment with CGVD wrapper if enabled."""
    if not args.use_cgvd:
        return env

    from src.cgvd import CGVDWrapper

    print(f"[CGVD] Wrapping environment with CGVD wrapper")
    print(f"[CGVD]   blur_sigma={args.cgvd_blur_sigma}")
    print(f"[CGVD]   update_freq={args.cgvd_update_freq}")
    print(f"[CGVD]   presence_threshold={args.cgvd_presence_threshold}")
    print(f"[CGVD]   use_mock_segmenter={args.cgvd_use_mock}")

    return CGVDWrapper(
        env,
        update_freq=args.cgvd_update_freq,
        blur_sigma=args.cgvd_blur_sigma,
        presence_threshold=args.cgvd_presence_threshold,
        use_mock_segmenter=args.cgvd_use_mock,
        feather_edges=args.cgvd_feather_edges,
        include_robot=True,  # Always include robot arm
        verbose=args.cgvd_verbose,
        save_debug_images=args.cgvd_save_debug,
    )


@log_execution_time()
def load_checkpoint(model, path):
    """load to cpu first, then move to gpu"""
    data = torch.load(path, weights_only=True, map_location="cpu")
    # remove "_orig_mod." prefix if saved model was compiled
    data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
    model.load_state_dict(data["model"], strict=True)
    print(f"Loaded model from {path}")


def run_episode(args, env, env_adapter, model, cfg, device, dtype, episode_idx):
    """Run a single episode and return (success, steps, inference_times)."""
    env_adapter.reset()

    # Use different episode_id for each run to get variety
    episode_id = (args.seed + episode_idx) % 21
    env_reset_options = {}
    env_reset_options["obj_init_options"] = {
        "episode_id": episode_id,  # this determines the obj inits in bridge
    }
    obs, reset_info = env.reset(options=env_reset_options)
    instruction = env.unwrapped.get_language_instruction()

    video_writer = None
    if args.recording:
        os.environ["TOKENIZERS_PARALLELISM"] = (
            "false"  # avoid tokenizer forking warning about deadlock
        )
        os.makedirs(args.output_dir, exist_ok=True)
        suffix = "_cgvd" if args.use_cgvd else "_baseline"
        video_path = os.path.join(args.output_dir, f"try_{args.task}{suffix}_ep{episode_idx}.mp4")
        video_writer = imageio.get_writer(video_path)

    print(f"\n--- Episode {episode_idx + 1}/{args.num_episodes} (id={episode_id}) ---")
    print(f"Instruction: {instruction}")

    cnt_step = 0
    inference_times = []
    success = False

    while True:
        # infer action chunk
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
        with torch.inference_mode():  # speeds up
            actions = model(**inputs)
        if cnt_step > 0:
            inference_times.append(time.time() - start_inference_time)
        env_actions = env_adapter.postprocess(actions[0].float().cpu().numpy())

        # environment step
        for env_action in env_actions[: cfg.act_steps]:
            obs, reward, success, truncated, info = env.step(env_action)
            cnt_step += 1
            if truncated:
                break

        # save frame
        if video_writer is not None:
            video_writer.append_data(env_adapter.get_video_frame(env, obs))

        # update instruction in long horizon tasks, e.g., pick apple ---> put in top drawer
        new_instruction = env.unwrapped.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction

        # original octo eval only done when timeout, i.e., not upon success
        if truncated:
            if video_writer is not None:
                video_writer.close()
            break

    result_str = "SUCCESS" if success else "FAILED"
    print(f"Episode {episode_idx + 1}: {result_str} (steps={cnt_step})")

    return success, cnt_step, inference_times


def main(args):
    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # devices
    device = torch.device(f"cuda:{args.gpu_id}")

    # load default configs
    if "fractal" in args.checkpoint_path:
        cfg = OmegaConf.load(
            "config/eval/fractal_apple.yaml"
        )  # doesn't matter which task
    if "bridge" in args.checkpoint_path:
        cfg = OmegaConf.load("config/eval/bridge.yaml")

    # model
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    model = PiZeroInference(cfg, use_ddp=False)
    load_checkpoint(model, args.checkpoint_path)
    model.freeze_all_weights()
    model.to(dtype)
    model.to(device)
    if (
        args.use_torch_compile
    ):  # model being compiled in the first batch which takes some time
        model = torch.compile(
            model,
            mode="default",  # "reduce-overhead; max-autotune(-no-cudagraphs)
            # backend="inductor", # default: inductor; cudagraphs
        )
    # modes: https://pytorch.org/docs/main/generated/torch.compile.html
    # backends: https://pytorch.org/docs/stable/torch.compiler.html
    model.eval()
    print(f"Using cuda device: {device} dtype: {dtype}")
    log_allocated_gpu_memory(None, "loading model", args.gpu_id)

    # simpler env
    env = simpler_env.make(args.task)

    # Add distractors if specified
    if args.distractors:
        from src.cgvd.distractor_wrapper import DistractorWrapper
        env = DistractorWrapper(env, args.distractors)
        print(f"[Distractors] Added: {args.distractors}")

    # optionally wrap with CGVD (after distractors so it sees the cluttered scene)
    env = wrap_env_with_cgvd(env, args)

    # env specifics
    env_adapter = hydra.utils.instantiate(cfg.env.adapter)

    # run episodes
    successes = []
    all_steps = []
    all_inference_times = []

    for ep_idx in range(args.num_episodes):
        success, steps, inference_times = run_episode(
            args, env, env_adapter, model, cfg, device, dtype, ep_idx
        )
        successes.append(success)
        all_steps.append(steps)
        all_inference_times.extend(inference_times)

    # summary
    num_success = sum(successes)
    success_rate = num_success / args.num_episodes * 100

    print("\n\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Task: {args.task}")
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
    print(f"Peak VRAM: {torch.cuda.max_memory_reserved(args.gpu_id) / 1024 ** 3:.2f} GB")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="google_robot_pick_horizontal_coke_can",
        choices=[
            "widowx_carrot_on_plate",
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
        ],
    )
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_torch_compile", action="store_true")
    parser.add_argument("--recording", action="store_true")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output videos")
    parser.add_argument(
        "--distractors",
        type=str,
        nargs="*",
        default=[],
        help="Distractor object IDs to add (e.g., apple orange sponge)"
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
        help="Use mock segmenter for CGVD testing (no SAM3 required)",
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
        help="Save debug images showing original/mask/distilled (to cgvd_debug/)",
    )

    args = parser.parse_args()

    # check task
    if "google_robot" in args.task:
        assert "fractal" in args.checkpoint_path
    if "widowx" in args.task:
        assert "bridge" in args.checkpoint_path

    main(args)
