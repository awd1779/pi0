"""Test SAM3 with alternative phrasings for the green spoon."""
import numpy as np
from PIL import Image
import cv2, os

BASE = "logs/clutter_eval/pi0/spoon/semantic/n12_e10_r1_20260216_041245/run_0"
frame_path = f"{BASE}/baseline/frames/episode_000/frame_0000.png"
image = Image.open(frame_path).convert("RGB")
img_np = np.array(image)

from src.cgvd.sam3_segmenter import SAM3Segmenter

seg = SAM3Segmenter(model_name="facebook/sam3", presence_threshold=0.1)
seg._lazy_init()

out_dir = "cgvd_debug/sam3_test"
os.makedirs(out_dir, exist_ok=True)

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]

def save_vis(img, concept, instance_masks, out_path, threshold_label=""):
    vis = img.copy()
    sorted_inst = sorted(instance_masks, key=lambda x: x[1], reverse=True)
    for i, (mask, score) in enumerate(sorted_inst[:8]):
        color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        binary = (mask > 0.5).astype(np.uint8)
        pixels = int(binary.sum())
        if pixels < 10:
            continue
        vis[binary > 0] = (vis[binary > 0] * 0.5 + color * 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color.astype(int).tolist(), 2)
        ys, xs = np.where(binary > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            label = f"#{i}: {score:.3f} ({pixels}px)"
            cv2.putText(vis, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(vis, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color.astype(int).tolist(), 1)
    n = len(sorted_inst)
    best = sorted_inst[0][1] if sorted_inst else 0
    title = f'"{concept}" {threshold_label} best={best:.4f}, n={n}'
    cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
    cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

concepts = [
    "spoon with green handle",
    "spoon with a green handle",
    "green handled spoon",
    "green spoon",
    "spoon",
]

for concept in concepts:
    for thresh in [0.01, 0.4]:
        mask, score, instance_masks = seg._segment_single_concept(
            image, concept, presence_threshold=thresh
        )
        n = len(instance_masks)
        top3 = sorted([s for _, s in instance_masks], reverse=True)[:3]
        top3_str = ", ".join(f"{s:.4f}" for s in top3)
        print(f"  t={thresh}  '{concept:30s}' -> best={score:.4f}, n={n:3d}, top3=[{top3_str}]")

        fname = concept.replace(" ", "_")
        save_vis(img_np, concept, instance_masks,
                 f"{out_dir}/{fname}_t{thresh:.2f}.png", f"(t={thresh})")

print(f"\nSaved to {out_dir}/")
