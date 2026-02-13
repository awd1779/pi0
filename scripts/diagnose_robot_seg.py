#!/usr/bin/env python3
"""Diagnose robot segmentation: visualize SAM3 input/output per frame.

Runs a multi-step episode with random actions to move the robot,
then saves per-frame diagnostic images showing:
  1. Original frame (input to CGVD)
  2. Image fed to SAM3 for robot segmentation (original vs last_distilled)
  3. Robot mask from SAM3
  4. Cached distractor mask
  5. Final compositing mask (cached_mask)
  6. Distilled output

This helps verify whether robot_seg_on_original=True fixes the 1-frame lag.

Usage:
    python scripts/diagnose_robot_seg.py --steps 20 --output_dir output/robot_seg_diag
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def mask_to_rgb(mask, color=(0, 255, 0)):
    """Convert single-channel mask to RGB overlay."""
    vis = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    binary = (mask > 0.5).astype(np.uint8)
    vis[:, :, 0] = binary * color[0]
    vis[:, :, 1] = binary * color[1]
    vis[:, :, 2] = binary * color[2]
    return vis


def mask_overlay(image, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay mask on image with transparency."""
    out = image.copy()
    binary = mask > 0.5
    overlay = np.zeros_like(image)
    overlay[binary] = color
    out[binary] = (
        (1 - alpha) * out[binary].astype(np.float32)
        + alpha * overlay[binary].astype(np.float32)
    ).astype(np.uint8)
    return out


def load_distractors(task_base, category="semantic", num_distractors=5):
    """Load distractors from file."""
    filepath = f"scripts/clutter_eval/distractors/distractors_{task_base}_{category}.txt"
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found")
        return [], []

    distractors = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                distractors.append(line)

    if num_distractors > 0:
        distractors = distractors[:num_distractors]

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


def get_camera_name(env):
    unwrapped = env.unwrapped
    if "widowx" in unwrapped.robot_uid:
        return "3rd_view_camera"
    elif "google_robot" in unwrapped.robot_uid:
        return "overhead_camera"
    else:
        raise ValueError(f"Unknown robot type: {unwrapped.robot_uid}")


def extract_image(obs, camera_name):
    return obs["image"][camera_name]["rgb"].copy()


def main():
    parser = argparse.ArgumentParser(description="Diagnose robot segmentation pipeline")
    parser.add_argument("--task", default="widowx_carrot_on_plate")
    parser.add_argument("--category", default="semantic")
    parser.add_argument("--num_distractors", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of env steps to run (robot moves)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--output_dir", default="output/robot_seg_diag")
    parser.add_argument("--robot_seg_on_original", type=int, default=1,
                        help="1=segment robot on original frame (new default), 0=on distilled (old)")
    args = parser.parse_args()

    task_base_map = {
        "widowx_carrot_on_plate": "carrot",
        "widowx_banana_on_plate": "banana",
        "widowx_put_eggplant_in_basket": "eggplant",
        "widowx_spoon_on_towel": "spoon",
        "widowx_stack_cube": "cube",
    }
    task_base = task_base_map.get(args.task, args.task)
    distractors, cgvd_names = load_distractors(task_base, args.category, args.num_distractors)
    print(f"Distractors: {distractors}")
    print(f"CGVD names: {cgvd_names}")

    if not distractors:
        sys.exit(1)

    robot_seg_on_original = bool(args.robot_seg_on_original)
    print(f"robot_seg_on_original = {robot_seg_on_original}")

    import simpler_env
    from src.cgvd import CGVDWrapper
    from src.cgvd.distractor_wrapper import DistractorWrapper

    env = simpler_env.make(args.task)
    env = DistractorWrapper(
        env, distractors,
        distractor_scale=None, external_asset_scale=0.1, num_distractors=None,
    )
    env = CGVDWrapper(
        env,
        update_freq=1,
        presence_threshold=0.6,
        use_mock_segmenter=False,
        include_robot=True,
        verbose=True,
        save_debug_images=False,
        distractor_names=cgvd_names,
        cache_distractor_once=True,
        robot_presence_threshold=0.3,
        distractor_presence_threshold=0.20,
        robot_seg_on_original=robot_seg_on_original,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Reset
    print(f"\nResetting env (episode_id={args.episode_id})...")
    obs, _ = env.reset(options={"obj_init_options": {"episode_id": args.episode_id}})
    camera_name = get_camera_name(env)
    action_space = env.action_space

    font = cv2.FONT_HERSHEY_SIMPLEX

    for step_i in range(args.steps):
        # Take a random action to move the robot
        action = action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # Extract the distilled image from obs
        distilled = extract_image(obs, camera_name)

        # Access CGVD internals for diagnostics
        # The original frame was used internally — we can't get it back directly
        # but we CAN access the masks and the segmentation source info
        robot_mask = env.last_robot_mask
        cached_mask = env.cached_mask
        cached_distractor = env.cached_distractor_mask
        cached_safe = env.cached_safe_mask
        last_distilled = env.last_distilled_image  # This is now set to current frame's output

        # Build diagnostic image: 6 columns
        h, w = distilled.shape[:2]
        label_h = 40
        n_cols = 6
        canvas = np.zeros((h + label_h, w * n_cols, 3), dtype=np.uint8)

        # Col 0: Distilled output
        canvas[label_h:, 0:w] = distilled
        cv2.putText(canvas, "Distilled Output", (10, 15), font, 0.45, (255, 255, 255), 1)
        cv2.putText(canvas, f"Step {step_i}", (10, 30), font, 0.35, (200, 200, 200), 1)

        # Col 1: Robot mask (from SAM3)
        if robot_mask is not None:
            robot_vis = mask_to_rgb(robot_mask, color=(0, 255, 0))
            coverage = robot_mask.sum() / robot_mask.size * 100
            canvas[label_h:, w:2*w] = robot_vis
            cv2.putText(canvas, f"Robot Mask ({coverage:.1f}%)", (w + 10, 15), font, 0.45, (0, 255, 0), 1)
            src_label = "src=ORIGINAL" if robot_seg_on_original else "src=DISTILLED(t-1)"
            color = (0, 255, 0) if robot_seg_on_original else (0, 165, 255)
            cv2.putText(canvas, src_label, (w + 10, 30), font, 0.35, color, 1)
        else:
            cv2.putText(canvas, "Robot Mask: None", (w + 10, 15), font, 0.45, (128, 128, 128), 1)

        # Col 2: Robot mask overlaid on distilled (shows alignment)
        if robot_mask is not None:
            overlay = mask_overlay(distilled, robot_mask, color=(0, 255, 0), alpha=0.5)
            canvas[label_h:, 2*w:3*w] = overlay
            cv2.putText(canvas, "Robot on Distilled", (2*w + 10, 15), font, 0.45, (0, 255, 0), 1)
        else:
            canvas[label_h:, 2*w:3*w] = distilled
            cv2.putText(canvas, "Robot on Distilled: N/A", (2*w + 10, 15), font, 0.45, (128, 128, 128), 1)

        # Col 3: Cached distractor mask
        if cached_distractor is not None:
            dist_vis = mask_to_rgb(cached_distractor, color=(255, 0, 0))
            d_cov = cached_distractor.sum() / cached_distractor.size * 100
            canvas[label_h:, 3*w:4*w] = dist_vis
            cv2.putText(canvas, f"Distractor Mask ({d_cov:.1f}%)", (3*w + 10, 15), font, 0.45, (255, 100, 100), 1)
        else:
            cv2.putText(canvas, "Distractor: warmup", (3*w + 10, 15), font, 0.45, (128, 128, 128), 1)

        # Col 4: Safe-set mask (target + anchor, no robot)
        if cached_safe is not None:
            safe_vis = mask_to_rgb(cached_safe, color=(0, 128, 255))
            s_cov = cached_safe.sum() / cached_safe.size * 100
            canvas[label_h:, 4*w:5*w] = safe_vis
            cv2.putText(canvas, f"Safe Set ({s_cov:.1f}%)", (4*w + 10, 15), font, 0.45, (100, 180, 255), 1)
        else:
            cv2.putText(canvas, "Safe Set: warmup", (4*w + 10, 15), font, 0.45, (128, 128, 128), 1)

        # Col 5: Final cached_mask (D AND NOT S)
        if cached_mask is not None:
            final_vis = mask_to_rgb(cached_mask, color=(255, 255, 0))
            f_cov = cached_mask.sum() / cached_mask.size * 100
            canvas[label_h:, 5*w:6*w] = final_vis
            cv2.putText(canvas, f"Final Mask ({f_cov:.1f}%)", (5*w + 10, 15), font, 0.45, (255, 255, 0), 1)
        else:
            cv2.putText(canvas, "Final: warmup", (5*w + 10, 15), font, 0.45, (128, 128, 128), 1)

        # Save
        out_path = os.path.join(args.output_dir, f"step_{step_i:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"  Saved {out_path}")

        if done or truncated:
            print(f"Episode ended at step {step_i}")
            break

    env.close()

    # Also stitch into a video for easy scrubbing
    frames = sorted(Path(args.output_dir).glob("step_*.png"))
    if frames:
        sample = cv2.imread(str(frames[0]))
        h_v, w_v = sample.shape[:2]
        video_path = os.path.join(args.output_dir, "diagnostic.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 5, (w_v, h_v))
        for f in frames:
            writer.write(cv2.imread(str(f)))
        writer.release()
        print(f"\nVideo: {video_path}")

    print(f"\nDone. Diagnostic frames in {args.output_dir}/")


if __name__ == "__main__":
    main()
