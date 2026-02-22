#!/usr/bin/env python3
"""Create color-variant bridge spoons from the original green/yellow spoon.

Each variant reuses the same mesh (textured.dae) and collision (collision.obj),
only the texture (Image_0.003.jpg) is recolored via HSV manipulation.

Usage:
    python scripts/create_spoon_color_variants.py
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

# Paths
SIMPLER_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = None  # Auto-detected below
JSON_PATH = None

# Try common locations for SimplerEnv assets
_CANDIDATES = [
    SIMPLER_ROOT / "ManiSkill2_real2sim" / "data" / "custom",
    Path.home() / "allenzren_SimplerEnv" / "ManiSkill2_real2sim" / "data" / "custom",
]

for _c in _CANDIDATES:
    if (_c / "models" / "bridge_spoon_generated_modified").is_dir():
        MODELS_DIR = _c / "models"
        JSON_PATH = _c / "info_bridge_custom_v0.json"
        break

if MODELS_DIR is None:
    raise FileNotFoundError(
        "Cannot find bridge_spoon_generated_modified model directory. "
        f"Searched: {[str(c / 'models') for c in _CANDIDATES]}"
    )

SOURCE_MODEL = "bridge_spoon_generated_modified"
TEXTURE_NAME = "Image_0.003.jpg"

# Color variants to create: name -> recolor params
# The original texture is green/yellow (~hue 30-50 in OpenCV's 0-180 range)
# OpenCV HSV: H=0-180, S=0-255, V=0-255
VARIANTS = {
    "bridge_spoon_red":     {"target_hue": 0,   "sat_scale": 1.0, "val_scale": 1.0},
    "bridge_spoon_orange":  {"target_hue": 10,  "sat_scale": 1.0, "val_scale": 1.0},
    "bridge_spoon_yellow":  {"target_hue": 25,  "sat_scale": 1.0, "val_scale": 1.0},
    "bridge_spoon_cyan":    {"target_hue": 75,  "sat_scale": 1.0, "val_scale": 1.0},
    "bridge_spoon_purple":  {"target_hue": 130, "sat_scale": 1.0, "val_scale": 1.0},
    "bridge_spoon_pink":    {"target_hue": 150, "sat_scale": 0.7, "val_scale": 1.1},
    "bridge_spoon_white":   {"target_hue": None, "sat_scale": 0.1, "val_scale": 1.3},
    "bridge_spoon_black":   {"target_hue": None, "sat_scale": 0.15, "val_scale": 0.3},
}

# JSON entry template (identical to bridge_spoon_generated_modified)
SPOON_ENTRY = {
    "bbox": {
        "min": [-0.07, -0.0173, -0.013021],
        "max": [0.07, 0.0173, 0.013021],
    },
    "scales": [1.0],
    "density": 1200,
}


def recolor_texture(img_bgr, target_hue=None, sat_scale=1.0, val_scale=1.0):
    """Recolor a texture image by shifting hue and adjusting saturation/value.

    Args:
        img_bgr: Input BGR image (OpenCV format).
        target_hue: Target hue in OpenCV range (0-180). None to keep original hue.
        sat_scale: Saturation multiplier (0.0 = grayscale, 1.0 = unchanged).
        val_scale: Value/brightness multiplier.

    Returns:
        Recolored BGR image.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    if target_hue is not None:
        # Set all hues to the target (uniform recolor)
        hsv[:, :, 0] = target_hue

    # Scale saturation and value
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def create_variant(variant_name, params):
    """Create a single color variant directory with recolored texture."""
    src_dir = MODELS_DIR / SOURCE_MODEL
    dst_dir = MODELS_DIR / variant_name

    if dst_dir.exists():
        print(f"  [skip] {variant_name} already exists at {dst_dir}")
        return False

    dst_dir.mkdir(parents=True)

    # Copy mesh and collision (identical)
    shutil.copy2(src_dir / "textured.dae", dst_dir / "textured.dae")
    shutil.copy2(src_dir / "collision.obj", dst_dir / "collision.obj")

    # Recolor texture
    img = cv2.imread(str(src_dir / TEXTURE_NAME))
    if img is None:
        raise FileNotFoundError(f"Cannot read texture: {src_dir / TEXTURE_NAME}")

    recolored = recolor_texture(img, **params)
    cv2.imwrite(str(dst_dir / TEXTURE_NAME), recolored)

    print(f"  [created] {variant_name} -> {dst_dir}")
    return True


def update_json(variant_names):
    """Add new variant entries to info_bridge_custom_v0.json."""
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    added = []
    for name in variant_names:
        if name not in data:
            data[name] = SPOON_ENTRY.copy()
            added.append(name)
        else:
            print(f"  [skip] {name} already in JSON")

    if added:
        with open(JSON_PATH, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"  [updated] JSON with {len(added)} new entries: {added}")
    else:
        print("  [skip] No new JSON entries needed")


def main():
    print(f"Source model: {MODELS_DIR / SOURCE_MODEL}")
    print(f"JSON: {JSON_PATH}")
    print()

    # Create color variant directories
    print("Creating color variants:")
    created = []
    for name, params in VARIANTS.items():
        if create_variant(name, params):
            created.append(name)
    print()

    # Update JSON
    print("Updating model database:")
    update_json(list(VARIANTS.keys()))
    print()

    # Summary
    print(f"Done! Created {len(created)} new variant(s).")
    if created:
        print("Verify textures visually:")
        for name in created:
            print(f"  {MODELS_DIR / name / TEXTURE_NAME}")


if __name__ == "__main__":
    main()
