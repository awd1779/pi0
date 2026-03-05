"""Standalone CGVD processing for real-world images.

Usage:
    cd /home/ubuntu/open-pi-zero
    python scripts/cgvd_standalone.py \
        --input_dir ~/Downloads/Imgs \
        --output_dir ~/Downloads/Imgs/cgvd_output \
        --target "banana" \
        --distractors "carrot. corn. bell pepper. pumpkin. chili pepper. apple. cucumber"
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cgvd.sam3_segmenter import SAM3Segmenter
from src.cgvd.lama_inpainter import LamaInpainter


def process_image(
    image_path: str,
    segmenter: SAM3Segmenter,
    inpainter: LamaInpainter,
    target_concepts: str,
    distractor_concepts: str,
    robot_concepts: str = "",
    safe_dilation: int = 5,
    lama_dilation: int = 11,
    blend_sigma: float = 3.0,
    presence_threshold: float = 0.15,
    distractor_threshold: float = 0.20,
):
    """Process a single image through CGVD pipeline.

    Returns:
        Tuple of (original, distilled, distractor_mask, safe_mask, inpainted_bg)
    """
    # Load image
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print(f"\n--- Processing {os.path.basename(image_path)} ---")
    print(f"  Image size: {img_rgb.shape[1]}x{img_rgb.shape[0]}")

    # Step 1: Segment distractors
    print(f"  Segmenting distractors: {distractor_concepts}")
    t0 = time.time()
    distractor_mask = segmenter.segment(
        img_rgb, distractor_concepts, presence_threshold=distractor_threshold
    )
    distractor_scores = segmenter.last_scores.copy()
    print(f"  Distractor scores: {distractor_scores}")
    print(f"  Distractor mask coverage: {distractor_mask.sum() / distractor_mask.size * 100:.1f}%")
    print(f"  SAM3 distractor time: {time.time() - t0:.2f}s")

    # Step 2: Segment safe-set (target)
    print(f"  Segmenting safe-set: {target_concepts}")
    t0 = time.time()
    safe_mask = segmenter.segment(
        img_rgb, target_concepts, presence_threshold=presence_threshold
    )
    safe_scores = segmenter.last_scores.copy()
    print(f"  Safe scores: {safe_scores}")
    print(f"  Safe mask coverage: {safe_mask.sum() / safe_mask.size * 100:.1f}%")
    print(f"  SAM3 safe time: {time.time() - t0:.2f}s")

    # Step 2b: Segment robot arm (separate from target safe mask)
    robot_mask_binary = np.zeros_like(safe_mask)
    if robot_concepts:
        print(f"  Segmenting robot: {robot_concepts}")
        t0 = time.time()
        robot_mask = segmenter.segment(
            img_rgb, robot_concepts, presence_threshold=0.05
        )
        robot_scores = segmenter.last_scores.copy()
        print(f"  Robot scores: {robot_scores}")
        print(f"  Robot mask coverage: {robot_mask.sum() / robot_mask.size * 100:.1f}%")
        print(f"  SAM3 robot time: {time.time() - t0:.2f}s")
        robot_mask_binary = (robot_mask > 0.5).astype(np.float32)

    # Step 3: Dilate target and robot separately, then combine
    # Target gets full dilation (max(safe_dilation, lama_dilation)) to protect from inpaint bleed
    # Robot gets only safe_dilation (small buffer, no lama_dilation bloat)
    target_dilation = max(safe_dilation, lama_dilation)
    robot_dilation = safe_dilation  # small buffer only

    target_dilated = (safe_mask > 0.5).astype(np.uint8)
    if target_dilation > 0:
        kernel = np.ones((target_dilation, target_dilation), np.uint8)
        target_dilated = cv2.dilate(target_dilated, kernel, iterations=1)

    robot_dilated = robot_mask_binary.astype(np.uint8)
    if robot_dilation > 0:
        kernel = np.ones((robot_dilation, robot_dilation), np.uint8)
        robot_dilated = cv2.dilate(robot_dilated, kernel, iterations=1)

    safe_dilated = np.maximum(target_dilated, robot_dilated).astype(np.float32)
    print(f"  Safe mask (target dilated={target_dilation}px + robot dilated={robot_dilation}px): "
          f"{safe_dilated.sum() / safe_dilated.size * 100:.1f}%")

    # Save undilated distractor for compositing mask (before lama_dilation)
    undilated_distractor = (distractor_mask > 0.5).astype(np.float32)

    # Step 3.5: Dilate distractor mask BEFORE safe-set subtraction (matches real CGVD)
    if lama_dilation > 0:
        dilation_kernel = np.ones((lama_dilation, lama_dilation), np.uint8)
        distractor_mask = cv2.dilate(
            (distractor_mask > 0.5).astype(np.uint8), dilation_kernel, iterations=1
        ).astype(np.float32)
        print(f"  Distractor mask after lama_dilation={lama_dilation}: "
              f"{distractor_mask.sum() / distractor_mask.size * 100:.1f}%")

    # Step 4: Inpaint mask = dilated distractor AND NOT safe
    inpaint_mask = np.logical_and(
        distractor_mask > 0.5, safe_dilated < 0.5
    ).astype(np.float32)

    # Compositing mask = use dilated inpaint mask so shadows are covered too
    # (In sim, undilated works because there are no shadows. Real images need dilation.)
    compositing_mask = inpaint_mask

    print(f"  Inpaint mask coverage: {inpaint_mask.sum() / inpaint_mask.size * 100:.1f}%")
    print(f"  Compositing mask coverage: {compositing_mask.sum() / compositing_mask.size * 100:.1f}%")

    # Step 5: LaMa inpaint using dilated mask (covers shadows/edges)
    print("  Running LaMa inpainting...")
    t0 = time.time()
    inpainted = inpainter.inpaint(img_rgb, inpaint_mask, dilate_mask=0)
    print(f"  LaMa time: {time.time() - t0:.2f}s")

    # Step 6: Composite using dilated compositing mask (covers shadows in real images)
    distilled = composite(img_rgb, inpainted, compositing_mask, blend_sigma, safe_dilated)

    return img_rgb, distilled, undilated_distractor, safe_mask, inpaint_mask, inpainted


def composite(image, inpainted, mask, blend_sigma, safe_mask):
    """Feathered compositing (simplified version of CGVD._composite)."""
    h, w = image.shape[:2]

    # Binary masks
    binary_dist = (mask > 0.5).astype(np.float32)
    binary_safe = (safe_mask > 0.5).astype(np.float32)

    # Feathered alpha from distractor mask
    ksize = int(6 * blend_sigma + 1) | 1  # ensure odd
    alpha = cv2.GaussianBlur(binary_dist, (ksize, ksize), blend_sigma)

    # Mechanism 2: clamp alpha outside binary target region
    alpha = np.where(binary_safe > 0.5, 0.0, alpha)

    # Blend
    alpha_3 = alpha[:, :, np.newaxis]
    result = (1.0 - alpha_3) * image.astype(np.float32) + alpha_3 * inpainted.astype(np.float32)

    return np.clip(result, 0, 255).astype(np.uint8)


def save_results(output_dir, basename, original, distilled, dist_mask, safe_mask, final_mask, inpainted):
    """Save before/after and diagnostic images."""
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(basename)[0]

    # Before / After
    cv2.imwrite(
        os.path.join(output_dir, f"{name}_before.jpg"),
        cv2.cvtColor(original, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{name}_after.jpg"),
        cv2.cvtColor(distilled, cv2.COLOR_RGB2BGR),
    )

    # Side by side
    side_by_side = np.hstack([original, distilled])
    cv2.imwrite(
        os.path.join(output_dir, f"{name}_comparison.jpg"),
        cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR),
    )

    # Diagnostic masks
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    cv2.imwrite(os.path.join(diag_dir, f"{name}_dist_mask.png"), (dist_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(diag_dir, f"{name}_safe_mask.png"), (safe_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(diag_dir, f"{name}_final_mask.png"), (final_mask * 255).astype(np.uint8))
    cv2.imwrite(
        os.path.join(diag_dir, f"{name}_inpainted.jpg"),
        cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR),
    )

    # Overlay: red = distractor, green = safe, on original
    overlay = original.copy()
    overlay[dist_mask > 0.5] = (overlay[dist_mask > 0.5] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    overlay[safe_mask > 0.5] = (overlay[safe_mask > 0.5] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    cv2.imwrite(
        os.path.join(diag_dir, f"{name}_overlay.jpg"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )


def main():
    parser = argparse.ArgumentParser(description="Standalone CGVD processing")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: input_dir/cgvd_output)")
    parser.add_argument("--target", type=str, required=True, help="Target object (safe-set), e.g. 'banana'")
    parser.add_argument("--distractors", type=str, required=True,
                        help="Distractor concepts (dot-separated), e.g. 'carrot. corn. pepper'")
    parser.add_argument("--include_robot", action="store_true", default=True,
                        help="Include robot arm in safe-set (default: True)")
    parser.add_argument("--robot_concepts", type=str, default="robot arm. gripper",
                        help="Robot concepts for safe-set (dot-separated)")
    parser.add_argument("--safe_dilation", type=int, default=5)
    parser.add_argument("--lama_dilation", type=int, default=11)
    parser.add_argument("--blend_sigma", type=float, default=3.0)
    parser.add_argument("--presence_threshold", type=float, default=0.15)
    parser.add_argument("--distractor_threshold", type=float, default=0.20)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "cgvd_output")

    # Initialize models
    print("Loading SAM3...")
    segmenter = SAM3Segmenter(presence_threshold=args.presence_threshold)

    print("Loading LaMa...")
    inpainter = LamaInpainter(device="cuda")
    inpainter._load_model()

    # Find images
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    print(f"\nFound {len(images)} images in {args.input_dir}")

    # Process each
    for img_file in images:
        img_path = os.path.join(args.input_dir, img_file)
        robot_concepts = args.robot_concepts if args.include_robot else ""
        original, distilled, dist_mask, safe_mask, final_mask, inpainted = process_image(
            img_path, segmenter, inpainter,
            target_concepts=args.target,
            distractor_concepts=args.distractors,
            robot_concepts=robot_concepts,
            safe_dilation=args.safe_dilation,
            lama_dilation=args.lama_dilation,
            blend_sigma=args.blend_sigma,
            presence_threshold=args.presence_threshold,
            distractor_threshold=args.distractor_threshold,
        )
        save_results(args.output_dir, img_file, original, distilled, dist_mask, safe_mask, final_mask, inpainted)
        print(f"  Saved to {args.output_dir}")

    print(f"\nDone! Results in {args.output_dir}")


if __name__ == "__main__":
    main()
