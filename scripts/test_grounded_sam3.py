#!/usr/bin/env python3
"""Compare SAM3 text-only vs Grounded SAM3 (GDINO + SAM3 box-prompted) segmentation.

Usage:
    python scripts/test_grounded_sam3.py \
        --image path/to/frame.png \
        --concepts "spoon. fork. towel"

Optional flags:
    --threshold        Presence threshold (default 0.15)
    --gdino-model      tiny | base (default tiny)
    --gdino-box-threshold   GDINO box confidence threshold (default 0.25)
    --gdino-text-threshold  GDINO text confidence threshold (default 0.25)
    --save-dir         Output directory (default grounded_sam3_debug)
    --skip-sam3        Skip SAM3 text-only baseline (faster)
"""

import argparse
import os
import sys

import cv2
import numpy as np


def _draw_masks_overlay(image: np.ndarray, masks_dict: dict, alpha: float = 0.4) -> np.ndarray:
    """Draw colored mask overlays on image."""
    overlay = image.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 255, 0), (255, 128, 0),
    ]
    for i, (name, mask) in enumerate(masks_dict.items()):
        color = colors[i % len(colors)]
        binary = (mask > 0.5).astype(np.uint8)
        for c in range(3):
            overlay[:, :, c] = np.where(
                binary,
                (overlay[:, :, c] * (1 - alpha) + color[c] * alpha).astype(np.uint8),
                overlay[:, :, c],
            )
    return overlay


def _draw_boxes(image: np.ndarray, detections: dict, concept_list: list) -> np.ndarray:
    """Draw GDINO bounding boxes on image."""
    overlay = image.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ]
    for i, concept in enumerate(concept_list):
        color = colors[i % len(colors)]
        for box, score in detections.get(concept, []):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{concept}: {score:.2f}"
            cv2.putText(overlay, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return overlay


def _max_score_for_concept(scores: dict, concept: str) -> float:
    """Get max score for a concept, checking indexed variants (concept_0, concept_1, ...)."""
    best = scores.get(concept, 0.0)
    for key, val in scores.items():
        if key.startswith(concept + "_"):
            best = max(best, val)
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Compare SAM3 text-only vs GDINO+SAM3 box-prompted segmentation",
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--concepts", required=True,
                        help="Dot-separated concepts (e.g., 'spoon. fork. towel')")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Presence threshold (default 0.15)")
    parser.add_argument("--gdino-model", choices=["tiny", "base"], default="base",
                        help="Grounding DINO model size (default base)")
    parser.add_argument("--gdino-box-threshold", type=float, default=0.25,
                        help="GDINO box confidence threshold (default 0.25)")
    parser.add_argument("--gdino-text-threshold", type=float, default=0.25,
                        help="GDINO text confidence threshold (default 0.25)")
    parser.add_argument("--save-dir", default="grounded_sam3_debug",
                        help="Output directory (default grounded_sam3_debug)")
    parser.add_argument("--skip-sam3", action="store_true",
                        help="Skip SAM3 text-only baseline (faster)")
    args = parser.parse_args()

    # Add project root to path
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    sys.path.insert(0, project_root)

    from src.cgvd.sam3_segmenter import SAM3Segmenter
    from src.cgvd.grounded_sam3_segmenter import GroundedSAM3Segmenter

    os.makedirs(args.save_dir, exist_ok=True)

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: cannot read {args.image}")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    print(f"Image: {args.image} ({w}x{h})")
    print(f"Concepts: {args.concepts}")
    print(f"Threshold: {args.threshold}")
    print()

    concept_list = [c.strip() for c in args.concepts.split(".") if c.strip()]

    # ---- SAM3 text-only ----
    sam3_scores: dict = {}
    sam3_masks: dict = {}
    sam3_time = 0.0

    if not args.skip_sam3:
        print("=" * 60)
        print("Running SAM3 (text-only)...")
        sam3 = SAM3Segmenter(presence_threshold=args.threshold)
        sam3.segment(image_rgb, args.concepts, presence_threshold=args.threshold)
        sam3_time = sam3.last_segment_time
        sam3_scores = dict(sam3.last_scores)
        sam3_masks = dict(sam3.last_individual_masks)
        print(f"  Time: {sam3_time:.3f}s")
        for name in sorted(sam3_scores):
            score = sam3_scores[name]
            mask = sam3_masks.get(name)
            px = int(mask.sum()) if mask is not None else 0
            print(f"  {name}: score={score:.4f}, pixels={px}")
        print()

    # ---- GDINO + SAM3 ----
    print("=" * 60)
    gdino_model_id = {
        "tiny": "IDEA-Research/grounding-dino-tiny",
        "base": "IDEA-Research/grounding-dino-base",
    }[args.gdino_model]

    print(f"Running GroundedSAM3 (gdino={args.gdino_model})...")
    gsam3 = GroundedSAM3Segmenter(
        gdino_model_name=gdino_model_id,
        gdino_box_threshold=args.gdino_box_threshold,
        gdino_text_threshold=args.gdino_text_threshold,
        presence_threshold=args.threshold,
    )
    gsam3.segment(image_rgb, args.concepts, presence_threshold=args.threshold)
    gsam3_time = gsam3.last_segment_time
    gsam3_scores = dict(gsam3.last_scores)
    gsam3_masks = dict(gsam3.last_individual_masks)
    gdino_detections = gsam3.last_gdino_detections

    print(f"  Time: {gsam3_time:.3f}s")
    print("  GDINO detections:")
    for concept in concept_list:
        dets = gdino_detections.get(concept, [])
        print(f"    {concept}: {len(dets)} box(es)")
        for box, score in dets:
            bw, bh = box[2] - box[0], box[3] - box[1]
            print(f"      [{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}] "
                  f"({bw:.0f}x{bh:.0f}) score={score:.4f}")
    print("  SAM3 results:")
    for name in sorted(gsam3_scores):
        score = gsam3_scores[name]
        mask = gsam3_masks.get(name)
        px = int(mask.sum()) if mask is not None else 0
        print(f"    {name}: score={score:.4f}, pixels={px}")
    print()

    # ---- Comparison table ----
    print("=" * 60)
    print("COMPARISON")
    header = f"{'Concept':<30} {'SAM3-text':>12} {'GDINO+SAM3':>12} {'GDINO boxes':>12}"
    print(header)
    print("-" * len(header))
    for concept in concept_list:
        s3 = _max_score_for_concept(sam3_scores, concept)
        gs = _max_score_for_concept(gsam3_scores, concept)
        n_boxes = len(gdino_detections.get(concept, []))
        s3_str = f"{s3:.4f}" if not args.skip_sam3 else "(skip)"
        print(f"{concept:<30} {s3_str:>12} {gs:>12.4f} {n_boxes:>12d}")

    if not args.skip_sam3:
        print(f"\n{'Timing':<30} {sam3_time:>12.3f}s {gsam3_time:>12.3f}s")
    else:
        print(f"\n{'Timing':<30} {'(skipped)':>12} {gsam3_time:>12.3f}s")
    print()

    # ---- Visualization ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    n_panels = 4 if not args.skip_sam3 else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    idx = 0

    # Panel: Original
    axes[idx].imshow(image_rgb)
    axes[idx].set_title("Original")
    axes[idx].axis("off")
    idx += 1

    # Panel: SAM3 text-only masks
    if not args.skip_sam3:
        sam3_overlay = _draw_masks_overlay(image_rgb, sam3_masks)
        axes[idx].imshow(sam3_overlay)
        axes[idx].set_title(f"SAM3 text-only ({sam3_time:.2f}s)")
        axes[idx].axis("off")
        idx += 1

    # Panel: GDINO boxes
    boxes_overlay = _draw_boxes(image_rgb, gdino_detections, concept_list)
    axes[idx].imshow(boxes_overlay)
    axes[idx].set_title("GDINO boxes")
    axes[idx].axis("off")
    idx += 1

    # Panel: GDINO+SAM3 masks
    gsam3_overlay = _draw_masks_overlay(image_rgb, gsam3_masks)
    axes[idx].imshow(gsam3_overlay)
    axes[idx].set_title(f"GDINO+SAM3 ({gsam3_time:.2f}s)")
    axes[idx].axis("off")

    plt.tight_layout()
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(args.save_dir, f"{base_name}_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
