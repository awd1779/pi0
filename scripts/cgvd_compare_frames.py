#!/usr/bin/env python3
"""Compare CGVD-distilled frames vs clean (no-distractor) frames.

Runs the same episode twice with the same seed:
1. Clean: no distractors loaded — saves raw frames (ground truth)
2. CGVD: distractors + CGVD wrapper — saves distilled frames

Outputs pixel-level comparison images and metrics (MAE, SSIM).

Usage:
    python scripts/cgvd_compare_frames.py \
        --task widowx_carrot_on_plate \
        --category semantic \
        --num_distractors 5 \
        --episodes 3 \
        --output_dir output/cgvd_compare
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_distractors_from_file(filepath, category, num_distractors):
    """Load distractors from file (self-contained, avoids batch_eval imports).

    Args:
        filepath: Path to distractors file (or base path without category suffix)
        category: Distractor category (semantic, visual, control)
        num_distractors: Number of distractors to use (0 = all)

    Returns:
        Tuple of (distractor asset IDs, CGVD distractor names)
    """
    base_path = Path(filepath)
    if base_path.exists():
        file_to_use = base_path
    else:
        categorized = base_path.parent / f"{base_path.stem}_{category}.txt"
        if categorized.exists():
            file_to_use = categorized
        else:
            print(f"Warning: No distractors file found for {filepath} / {category}")
            return [], []

    distractors = []
    with open(file_to_use) as f:
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


# Map task names to distractor file base names
TASK_TO_BASE = {
    "widowx_carrot_on_plate": "carrot",
    "widowx_banana_on_plate": "banana",
    "widowx_put_eggplant_in_basket": "eggplant",
    "widowx_spoon_on_towel": "spoon",
    "widowx_stack_cube": "cube",
}


def get_camera_name(env):
    """Determine camera name from robot type."""
    unwrapped = env.unwrapped
    if "google_robot" in unwrapped.robot_uid:
        return "overhead_camera"
    elif "widowx" in unwrapped.robot_uid:
        return "3rd_view_camera"
    else:
        raise ValueError(f"Unknown robot type: {unwrapped.robot_uid}")


def extract_image(obs, camera_name):
    """Extract RGB image from observation dict."""
    return obs["image"][camera_name]["rgb"].copy()


def compute_metrics(clean, cgvd):
    """Compute comparison metrics between two images.

    Args:
        clean: (H, W, 3) uint8 clean image
        cgvd: (H, W, 3) uint8 CGVD-distilled image

    Returns:
        dict with mae, ssim, per-channel mae
    """
    clean_f = clean.astype(np.float32)
    cgvd_f = cgvd.astype(np.float32)

    # Mean Absolute Error (0-255 scale)
    mae = np.mean(np.abs(clean_f - cgvd_f))

    # SSIM (computed on grayscale)
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
    cgvd_gray = cv2.cvtColor(cgvd, cv2.COLOR_RGB2GRAY)
    ssim_val = ssim(clean_gray, cgvd_gray)

    # Per-channel MAE
    mae_r = np.mean(np.abs(clean_f[:, :, 0] - cgvd_f[:, :, 0]))
    mae_g = np.mean(np.abs(clean_f[:, :, 1] - cgvd_f[:, :, 1]))
    mae_b = np.mean(np.abs(clean_f[:, :, 2] - cgvd_f[:, :, 2]))

    # Peak error
    peak_err = np.max(np.abs(clean_f - cgvd_f))

    # Fraction of pixels with error > threshold
    pixel_diff = np.max(np.abs(clean_f - cgvd_f), axis=2)
    frac_above_5 = np.mean(pixel_diff > 5)
    frac_above_10 = np.mean(pixel_diff > 10)
    frac_above_25 = np.mean(pixel_diff > 25)

    return {
        "mae": mae,
        "ssim": ssim_val,
        "mae_r": mae_r,
        "mae_g": mae_g,
        "mae_b": mae_b,
        "peak_error": peak_err,
        "frac_above_5": frac_above_5,
        "frac_above_10": frac_above_10,
        "frac_above_25": frac_above_25,
    }


def make_comparison_image(clean, cgvd, diff_amplified):
    """Create side-by-side comparison: Clean | CGVD | Difference heatmap.

    Args:
        clean: (H, W, 3) uint8
        cgvd: (H, W, 3) uint8
        diff_amplified: (H, W, 3) uint8 amplified difference

    Returns:
        (H, W*3, 3) uint8 comparison image
    """
    h, w = clean.shape[:2]

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_h = 30
    canvas = np.zeros((h + label_h, w * 3, 3), dtype=np.uint8)

    # Labels
    cv2.putText(canvas, "Clean (no distractors)", (10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "CGVD Distilled", (w + 10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "Difference (5x amplified)", (2 * w + 10, 20), font, 0.5, (255, 255, 255), 1)

    # Images (convert RGB to BGR for label drawing, then back)
    canvas[label_h:, :w] = clean
    canvas[label_h:, w:2*w] = cgvd
    canvas[label_h:, 2*w:] = diff_amplified

    return canvas


def make_difference_heatmap(clean, cgvd, amplify=5.0):
    """Create amplified difference heatmap.

    Args:
        clean: (H, W, 3) uint8
        cgvd: (H, W, 3) uint8
        amplify: Amplification factor for visibility

    Returns:
        (H, W, 3) uint8 heatmap (colorized)
    """
    diff = np.abs(clean.astype(np.float32) - cgvd.astype(np.float32))
    # Max across channels for heatmap
    diff_gray = np.max(diff, axis=2)
    # Amplify and clip
    diff_amplified = np.clip(diff_gray * amplify, 0, 255).astype(np.uint8)
    # Apply colormap (TURBO gives good perceptual gradient)
    heatmap = cv2.applyColorMap(diff_amplified, cv2.COLORMAP_TURBO)
    # Convert BGR (OpenCV colormap) to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def run_clean_episode(task, seed, episode_idx):
    """Run a clean episode (no distractors) and capture first frame.

    Returns:
        (image, camera_name) or (None, None) on failure
    """
    import simpler_env
    env = simpler_env.make(task)
    try:
        episode_id = (seed + episode_idx) % 24
        obs, _ = env.reset(options={"obj_init_options": {"episode_id": episode_id}})
        camera_name = get_camera_name(env)
        image = extract_image(obs, camera_name)
        return image, camera_name
    finally:
        env.close()


def run_cgvd_episode(task, seed, episode_idx, distractors, cgvd_names,
                     cgvd_save_debug=False, debug_dir="cgvd_debug",
                     robot_seg_on_original=False):
    """Run a CGVD episode (distractors + CGVD) and capture first distilled frame.

    Args:
        robot_seg_on_original: If True, segment robot on original frame instead
            of distilled frame. A/B test for robot segmentation source.

    Returns:
        (distilled_image, camera_name) or (None, None) on failure
    """
    import simpler_env
    from src.cgvd import CGVDWrapper
    from src.cgvd.distractor_wrapper import DistractorWrapper

    env = simpler_env.make(task)
    env = DistractorWrapper(
        env,
        distractors,
        distractor_scale=None,
        external_asset_scale=0.1,
        num_distractors=None,
    )
    env = CGVDWrapper(
        env,
        update_freq=1,
        presence_threshold=0.6,
        use_mock_segmenter=False,
        include_robot=True,
        verbose=False,
        save_debug_images=cgvd_save_debug,
        debug_dir=debug_dir,
        distractor_names=cgvd_names,
        cache_distractor_once=True,
        robot_presence_threshold=0.3,
        distractor_presence_threshold=0.20,
        robot_seg_on_original=robot_seg_on_original,
    )
    try:
        episode_id = (seed + episode_idx) % 24
        obs, _ = env.reset(options={"obj_init_options": {"episode_id": episode_id}})
        camera_name = get_camera_name(env)
        image = extract_image(obs, camera_name)
        return image, camera_name
    finally:
        env.close()


def make_ab_comparison_image(clean, distilled, original, amplify=5.0):
    """Create 5-column comparison: Clean | Distilled | Original | Diff(D) | Diff(O).

    Args:
        clean: (H, W, 3) uint8 clean image (no distractors)
        distilled: (H, W, 3) uint8 CGVD with robot seg on distilled
        original: (H, W, 3) uint8 CGVD with robot seg on original
        amplify: Amplification factor for difference heatmaps

    Returns:
        (H+label_h, W*5, 3) uint8 comparison image
    """
    h, w = clean.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_h = 30

    heatmap_d = make_difference_heatmap(clean, distilled, amplify=amplify)
    heatmap_o = make_difference_heatmap(clean, original, amplify=amplify)

    canvas = np.zeros((h + label_h, w * 5, 3), dtype=np.uint8)

    cv2.putText(canvas, "Clean", (10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "CGVD (distilled seg)", (w + 10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "CGVD (original seg)", (2 * w + 10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "Diff: distilled", (3 * w + 10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "Diff: original", (4 * w + 10, 20), font, 0.5, (255, 255, 255), 1)

    canvas[label_h:, :w] = clean
    canvas[label_h:, w:2*w] = distilled
    canvas[label_h:, 2*w:3*w] = original
    canvas[label_h:, 3*w:4*w] = heatmap_d
    canvas[label_h:, 4*w:] = heatmap_o

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Compare CGVD-distilled frames vs clean (no-distractor) frames"
    )
    parser.add_argument("--task", default="widowx_carrot_on_plate",
                        choices=list(TASK_TO_BASE.keys()),
                        help="SimplerEnv task name")
    parser.add_argument("--category", default="semantic",
                        choices=["semantic", "visual", "control"],
                        help="Distractor category")
    parser.add_argument("--num_distractors", type=int, default=5,
                        help="Number of distractors to load")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to compare")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--output_dir", default="output/cgvd_compare",
                        help="Directory for output images and metrics")
    parser.add_argument("--cgvd_save_debug", action="store_true",
                        help="Save CGVD debug images (6-column layout)")
    parser.add_argument("--amplify", type=float, default=5.0,
                        help="Difference amplification factor for heatmap")
    parser.add_argument("--robot_seg_on_original", action="store_true",
                        help="Also run variant with robot segmented on original frame (A/B test)")
    args = parser.parse_args()

    # Load distractors
    task_base = TASK_TO_BASE[args.task]
    distractor_file = f"scripts/clutter_eval/distractors/distractors_{task_base}_{args.category}.txt"
    distractors, cgvd_names = load_distractors_from_file(
        distractor_file, args.category, args.num_distractors
    )
    print(f"Loaded {len(distractors)} distractors: {distractors}")
    print(f"CGVD distractor names: {cgvd_names}")

    if not distractors:
        print("ERROR: No distractors loaded. Check distractor file path.")
        sys.exit(1)

    run_ab = args.robot_seg_on_original
    variant_label = "3-variant (clean/distilled/original)" if run_ab else "2-variant (clean/distilled)"
    print(f"Mode: {variant_label}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect metrics across episodes
    all_metrics_distilled = []
    all_metrics_original = []

    for ep_idx in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx} (seed={args.seed}, episode_id={(args.seed + ep_idx) % 24})")
        print(f"{'='*60}")

        ep_dir = os.path.join(args.output_dir, f"episode_{ep_idx:03d}")
        os.makedirs(ep_dir, exist_ok=True)

        # 1. Run clean episode
        print("  Running clean episode (no distractors)...")
        t0 = time.time()
        clean_img, camera = run_clean_episode(args.task, args.seed, ep_idx)
        print(f"  Clean episode done in {time.time() - t0:.1f}s")

        if clean_img is None:
            print("  ERROR: Failed to get clean frame, skipping episode")
            continue

        # 2. Run CGVD episode (distilled robot seg — current behavior)
        print("  Running CGVD episode (distilled robot seg)...")
        t0 = time.time()
        debug_dir_d = os.path.join(ep_dir, "cgvd_debug_distilled") if args.cgvd_save_debug else "cgvd_debug"
        cgvd_distilled_img, _ = run_cgvd_episode(
            args.task, args.seed, ep_idx,
            distractors, cgvd_names,
            cgvd_save_debug=args.cgvd_save_debug,
            debug_dir=debug_dir_d,
            robot_seg_on_original=False,
        )
        print(f"  CGVD (distilled) done in {time.time() - t0:.1f}s")

        if cgvd_distilled_img is None:
            print("  ERROR: Failed to get CGVD distilled frame, skipping episode")
            continue

        # 3. Compute metrics for distilled variant
        metrics_d = compute_metrics(clean_img, cgvd_distilled_img)
        metrics_d["episode"] = ep_idx
        metrics_d["episode_id"] = (args.seed + ep_idx) % 24
        metrics_d["variant"] = "distilled"
        all_metrics_distilled.append(metrics_d)

        print(f"  [Distilled] MAE: {metrics_d['mae']:.2f} | SSIM: {metrics_d['ssim']:.4f} | "
              f"Peak: {metrics_d['peak_error']:.0f}")
        print(f"  [Distilled] Pixels >5: {metrics_d['frac_above_5']*100:.1f}% | "
              f">10: {metrics_d['frac_above_10']*100:.1f}% | "
              f">25: {metrics_d['frac_above_25']*100:.1f}%")

        # Save distilled images
        cv2.imwrite(os.path.join(ep_dir, "clean_frame.png"),
                     cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(ep_dir, "cgvd_distilled_frame.png"),
                     cv2.cvtColor(cgvd_distilled_img, cv2.COLOR_RGB2BGR))

        heatmap_d = make_difference_heatmap(clean_img, cgvd_distilled_img, amplify=args.amplify)
        cv2.imwrite(os.path.join(ep_dir, "diff_heatmap_distilled.png"),
                     cv2.cvtColor(heatmap_d, cv2.COLOR_RGB2BGR))

        comparison_d = make_comparison_image(clean_img, cgvd_distilled_img, heatmap_d)
        cv2.imwrite(os.path.join(ep_dir, "comparison_distilled.png"),
                     cv2.cvtColor(comparison_d, cv2.COLOR_RGB2BGR))

        # 4. Optionally run original variant
        cgvd_original_img = None
        if run_ab:
            print("  Running CGVD episode (original robot seg)...")
            t0 = time.time()
            debug_dir_o = os.path.join(ep_dir, "cgvd_debug_original") if args.cgvd_save_debug else "cgvd_debug"
            cgvd_original_img, _ = run_cgvd_episode(
                args.task, args.seed, ep_idx,
                distractors, cgvd_names,
                cgvd_save_debug=args.cgvd_save_debug,
                debug_dir=debug_dir_o,
                robot_seg_on_original=True,
            )
            print(f"  CGVD (original) done in {time.time() - t0:.1f}s")

            if cgvd_original_img is None:
                print("  ERROR: Failed to get CGVD original frame")
            else:
                metrics_o = compute_metrics(clean_img, cgvd_original_img)
                metrics_o["episode"] = ep_idx
                metrics_o["episode_id"] = (args.seed + ep_idx) % 24
                metrics_o["variant"] = "original"
                all_metrics_original.append(metrics_o)

                print(f"  [Original]  MAE: {metrics_o['mae']:.2f} | SSIM: {metrics_o['ssim']:.4f} | "
                      f"Peak: {metrics_o['peak_error']:.0f}")
                print(f"  [Original]  Pixels >5: {metrics_o['frac_above_5']*100:.1f}% | "
                      f">10: {metrics_o['frac_above_10']*100:.1f}% | "
                      f">25: {metrics_o['frac_above_25']*100:.1f}%")

                # Save original variant images
                cv2.imwrite(os.path.join(ep_dir, "cgvd_original_frame.png"),
                             cv2.cvtColor(cgvd_original_img, cv2.COLOR_RGB2BGR))

                heatmap_o = make_difference_heatmap(clean_img, cgvd_original_img, amplify=args.amplify)
                cv2.imwrite(os.path.join(ep_dir, "diff_heatmap_original.png"),
                             cv2.cvtColor(heatmap_o, cv2.COLOR_RGB2BGR))

                comparison_o = make_comparison_image(clean_img, cgvd_original_img, heatmap_o)
                cv2.imwrite(os.path.join(ep_dir, "comparison_original.png"),
                             cv2.cvtColor(comparison_o, cv2.COLOR_RGB2BGR))

                # Direct distilled-vs-original comparison (isolates robot seg difference)
                diff_do = make_difference_heatmap(cgvd_distilled_img, cgvd_original_img, amplify=args.amplify)
                cv2.imwrite(os.path.join(ep_dir, "diff_distilled_vs_original.png"),
                             cv2.cvtColor(diff_do, cv2.COLOR_RGB2BGR))

                # 5-column A/B comparison
                ab_comparison = make_ab_comparison_image(
                    clean_img, cgvd_distilled_img, cgvd_original_img, amplify=args.amplify
                )
                cv2.imwrite(os.path.join(ep_dir, "comparison_ab.png"),
                             cv2.cvtColor(ab_comparison, cv2.COLOR_RGB2BGR))

                # Metrics delta
                delta_mae = metrics_o["mae"] - metrics_d["mae"]
                delta_ssim = metrics_o["ssim"] - metrics_d["ssim"]
                sign_mae = "+" if delta_mae >= 0 else ""
                sign_ssim = "+" if delta_ssim >= 0 else ""
                print(f"  [Delta O-D] MAE: {sign_mae}{delta_mae:.2f} | SSIM: {sign_ssim}{delta_ssim:.4f}")

        print(f"  Saved to {ep_dir}/")

    # Write metrics CSV
    if all_metrics_distilled:
        all_metrics = all_metrics_distilled + all_metrics_original
        csv_path = os.path.join(args.output_dir, "metrics.csv")
        fieldnames = ["episode", "episode_id", "variant", "mae", "ssim", "mae_r", "mae_g", "mae_b",
                       "peak_error", "frac_above_5", "frac_above_10", "frac_above_25"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in all_metrics:
                writer.writerow(m)

        # Write summary
        summary_path = os.path.join(args.output_dir, "summary.txt")

        def _avg(metrics_list, key):
            return np.mean([m[key] for m in metrics_list])

        summary_lines = [
            f"CGVD Frame Comparison Summary",
            f"{'='*50}",
            f"Task: {args.task}",
            f"Category: {args.category}",
            f"Num distractors: {args.num_distractors}",
            f"Episodes compared: {len(all_metrics_distilled)}",
            f"Seed: {args.seed}",
            f"Robot seg A/B: {'enabled' if run_ab else 'disabled'}",
            f"",
            f"Distilled (robot seg on distilled frame):",
            f"  MAE:  {_avg(all_metrics_distilled, 'mae'):.2f} / 255",
            f"  SSIM: {_avg(all_metrics_distilled, 'ssim'):.4f}",
            f"  Pixels with error >5:  {_avg(all_metrics_distilled, 'frac_above_5')*100:.1f}%",
            f"  Pixels with error >10: {_avg(all_metrics_distilled, 'frac_above_10')*100:.1f}%",
            f"  Pixels with error >25: {_avg(all_metrics_distilled, 'frac_above_25')*100:.1f}%",
        ]

        if all_metrics_original:
            summary_lines += [
                f"",
                f"Original (robot seg on original frame):",
                f"  MAE:  {_avg(all_metrics_original, 'mae'):.2f} / 255",
                f"  SSIM: {_avg(all_metrics_original, 'ssim'):.4f}",
                f"  Pixels with error >5:  {_avg(all_metrics_original, 'frac_above_5')*100:.1f}%",
                f"  Pixels with error >10: {_avg(all_metrics_original, 'frac_above_10')*100:.1f}%",
                f"  Pixels with error >25: {_avg(all_metrics_original, 'frac_above_25')*100:.1f}%",
                f"",
                f"Delta (Original - Distilled):",
                f"  MAE:  {_avg(all_metrics_original, 'mae') - _avg(all_metrics_distilled, 'mae'):+.2f}",
                f"  SSIM: {_avg(all_metrics_original, 'ssim') - _avg(all_metrics_distilled, 'ssim'):+.4f}",
            ]

        summary_lines += [f"", f"Per-episode:"]
        for m in all_metrics_distilled:
            line = (f"  Episode {m['episode']} (id={m['episode_id']}) [distilled]: "
                    f"MAE={m['mae']:.2f}, SSIM={m['ssim']:.4f}, "
                    f">10px={m['frac_above_10']*100:.1f}%")
            summary_lines.append(line)
        for m in all_metrics_original:
            line = (f"  Episode {m['episode']} (id={m['episode_id']}) [original]:  "
                    f"MAE={m['mae']:.2f}, SSIM={m['ssim']:.4f}, "
                    f">10px={m['frac_above_10']*100:.1f}%")
            summary_lines.append(line)

        summary_text = "\n".join(summary_lines) + "\n"
        with open(summary_path, "w") as f:
            f.write(summary_text)

        print(f"\n{'='*60}")
        print(summary_text)
        print(f"Results saved to {args.output_dir}/")
    else:
        print("\nNo episodes completed successfully.")


if __name__ == "__main__":
    main()
