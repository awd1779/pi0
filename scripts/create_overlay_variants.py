#!/usr/bin/env python3
"""Create overlay variants from bridge_real_eval_1.png using texture replacement.

Segments the image into table vs wall, then replaces BOTH regions with
downloaded material textures (marble, concrete, tiles, etc.).
Preserves original lighting cues by blending in the luminance channel.

Textures from ambientcg.com (CC0 license), stored in scripts/textures/.

Usage:
    python scripts/create_overlay_variants.py
"""

import cv2
import numpy as np
from pathlib import Path

# Paths
INPAINTING_DIR = (
    Path.home()
    / "allenzren_SimplerEnv"
    / "ManiSkill2_real2sim"
    / "data"
    / "real_inpainting"
)
BASE_IMAGE = INPAINTING_DIR / "bridge_real_eval_1.png"
TEXTURE_DIR = Path(__file__).resolve().parent / "textures"

# 6 variant definitions: (filename, table_texture, wall_texture, lighting_blend)
VARIANTS = [
    # Variant 2: white marble table + white plaster wall
    (
        "bridge_real_eval_variant_2.png",
        TEXTURE_DIR / "marble.jpg",
        TEXTURE_DIR / "wall_plaster.jpg",
        0.15,
    ),
    # Variant 3: dark concrete table + brick wall
    (
        "bridge_real_eval_variant_3.png",
        TEXTURE_DIR / "concrete.jpg",
        TEXTURE_DIR / "wall_brick.jpg",
        0.15,
    ),
    # Variant 4: brushed steel table + dark wood wall
    (
        "bridge_real_eval_variant_4.png",
        TEXTURE_DIR / "metal.jpg",
        TEXTURE_DIR / "wall_darkwood.jpg",
        0.15,
    ),
    # Variant 5: dark walnut table + warm plaster wall
    (
        "bridge_real_eval_variant_5.png",
        TEXTURE_DIR / "warmwood.jpg",
        TEXTURE_DIR / "wall_plaster.jpg",
        0.15,
    ),
    # Variant 6: geometric mosaic table + neutral painted wall
    (
        "bridge_real_eval_variant_6.png",
        TEXTURE_DIR / "mosaic.jpg",
        TEXTURE_DIR / "wall_paintedplaster.jpg",
        0.10,
    ),
    # Variant 7: blue azulejo tile table + white rough plaster wall
    (
        "bridge_real_eval_variant_7.png",
        TEXTURE_DIR / "azulejo.jpg",
        TEXTURE_DIR / "wall_plaster_rough.jpg",
        0.10,
    ),
]


def segment_table_wall(img_bgr):
    """Segment table (warm wood) vs wall (cool grey) using HSV thresholds.

    Returns:
        table_mask: float32 array (0-1), 1 = table pixel
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    # Table: saturation > 50 (wood has color; wall is grey)
    table_mask = (sat > 50).astype(np.float32)

    # Smooth the mask boundary to avoid harsh seams
    table_mask = cv2.GaussianBlur(table_mask, (15, 15), 0)
    table_mask = np.clip(table_mask, 0, 1)

    return table_mask


def get_luminance(img_bgr):
    """Extract normalized luminance (0-1) from BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray


def load_texture(path, target_h, target_w):
    """Load and resize a texture image."""
    tex = cv2.imread(str(path))
    if tex is None:
        raise FileNotFoundError(f"Cannot read texture: {path}")
    return cv2.resize(tex, (target_w, target_h)).astype(np.float32)


def apply_lighting(texture, orig_luminance, blend):
    """Blend original lighting into texture to preserve shadows/highlights."""
    if blend <= 0:
        return texture
    mean_lum = orig_luminance.mean()
    if mean_lum > 0:
        lum_factor = orig_luminance / mean_lum
    else:
        lum_factor = np.ones_like(orig_luminance)
    lum_factor_3 = lum_factor[:, :, np.newaxis]
    result = texture * (1.0 - blend + blend * lum_factor_3)
    return np.clip(result, 0, 255)


def create_variant(img_bgr, table_mask, table_path, wall_path, lighting_blend):
    """Create a variant by replacing both table and wall textures."""
    h, w = img_bgr.shape[:2]
    orig_lum = get_luminance(img_bgr)

    # Prepare table texture
    table_tex = load_texture(table_path, h, w)
    table_tex = apply_lighting(table_tex, orig_lum, lighting_blend)

    # Prepare wall texture
    wall_tex = load_texture(wall_path, h, w)
    wall_tex = apply_lighting(wall_tex, orig_lum, lighting_blend)

    # Composite: table region gets table texture, wall region gets wall texture
    mask3 = table_mask[:, :, np.newaxis]
    result = wall_tex * (1.0 - mask3) + table_tex * mask3

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    if not BASE_IMAGE.exists():
        raise FileNotFoundError(f"Base image not found: {BASE_IMAGE}")

    img = cv2.imread(str(BASE_IMAGE))
    print(f"Base image: {BASE_IMAGE} ({img.shape[1]}x{img.shape[0]})")

    # Segment table vs wall
    table_mask = segment_table_wall(img)
    table_pct = table_mask.mean() * 100
    print(f"Table mask: {table_pct:.1f}% of pixels")

    # Generate variants
    for filename, table_path, wall_path, lighting_blend in VARIANTS:
        out_path = INPAINTING_DIR / filename
        variant = create_variant(img, table_mask, table_path, wall_path, lighting_blend)
        cv2.imwrite(str(out_path), variant)
        print(f"  Written: {out_path.name}  (table={table_path.stem}, wall={wall_path.stem})")

    print(f"\nDone! Generated {len(VARIANTS)} variants in {INPAINTING_DIR}")


if __name__ == "__main__":
    main()
