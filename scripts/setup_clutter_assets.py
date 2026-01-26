#!/usr/bin/env python3
"""
Setup script to download and convert YCB and RoboCasa assets for SimplerEnv clutter testing.

This script:
1. Downloads YCB and RoboCasa datasets via ManiSkill
2. Converts assets to SimplerEnv format (collision.obj + textured.obj/dae)
3. Updates info_bridge_custom_v0.json with asset metadata

Usage:
    python scripts/setup_clutter_assets.py [--skip-download] [--simplenv-path PATH]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_simplerenv_data_dir():
    """Find SimplerEnv custom data directory."""
    # Common locations to check (allenzren first - that's what uv uses)
    candidates = [
        Path.home() / "allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom",
        Path.home() / "SimplerEnv-OpenVLA/ManiSkill2_real2sim/data/custom",
        Path.home() / "SimplerEnv/ManiSkill2_real2sim/data/custom",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def find_maniskill_asset_dir():
    """Find ManiSkill asset directory."""
    return Path.home() / ".maniskill/data"


def download_assets(conda_env="maniskill"):
    """Download YCB and RoboCasa datasets via ManiSkill."""
    print("=" * 60)
    print("Downloading YCB and RoboCasa datasets...")
    print("=" * 60)

    datasets = ["ycb", "RoboCasa"]

    for dataset in datasets:
        print(f"\n[Download] {dataset}...")
        cmd = f"source ~/miniconda3/bin/activate {conda_env} && python -m mani_skill.utils.download_asset {dataset} -y"
        result = subprocess.run(cmd, shell=True, executable="/bin/bash")
        if result.returncode != 0:
            print(f"[Warning] Failed to download {dataset}, continuing...")
        else:
            print(f"[OK] {dataset} downloaded successfully")


def get_ycb_objects(maniskill_dir):
    """Get list of YCB objects from ManiSkill."""
    ycb_info_path = maniskill_dir / "assets/mani_skill2_ycb/info_pick_v0.json"
    if not ycb_info_path.exists():
        print(f"[Warning] YCB info file not found: {ycb_info_path}")
        return {}

    with open(ycb_info_path) as f:
        return json.load(f)


def get_robocasa_objects(maniskill_dir):
    """Get list of RoboCasa objects from downloaded assets."""
    robocasa_dir = maniskill_dir / "scene_datasets/robocasa_dataset/assets/objects"
    objects = {}

    if not robocasa_dir.exists():
        print(f"[Warning] RoboCasa directory not found: {robocasa_dir}")
        return objects

    # RoboCasa has objaverse subdirectory with category/model_name structure
    for source in ["objaverse", "aigen_objs"]:
        source_dir = robocasa_dir / source
        if not source_dir.exists():
            continue

        for category_dir in source_dir.iterdir():
            if not category_dir.is_dir():
                continue
            if category_dir.name.startswith("__"):  # Skip __MACOSX
                continue

            for model_dir in category_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                # RoboCasa format: collision/ and visual/ subdirectories
                collision_dir = model_dir / "collision"
                visual_dir = model_dir / "visual"

                has_collision = collision_dir.exists() and any(collision_dir.glob("*.obj"))
                has_visual = visual_dir.exists() and any(visual_dir.glob("*.obj"))

                if has_collision or has_visual:
                    model_id = model_dir.name
                    objects[model_id] = {
                        "source": source,
                        "category": category_dir.name,
                        "path": str(model_dir),
                    }

    return objects


def convert_ycb_asset(model_id, model_info, ycb_models_dir, output_dir):
    """Convert a single YCB asset to SimplerEnv format."""
    src_dir = ycb_models_dir / model_id
    dst_dir = output_dir / f"ycb_{model_id}"

    if not src_dir.exists():
        print(f"  [Skip] Source not found: {src_dir}")
        return None

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy collision mesh (convert .ply to .obj if needed)
    collision_ply = src_dir / "collision.ply"
    collision_obj = src_dir / "collision.obj"

    if collision_obj.exists():
        shutil.copy2(collision_obj, dst_dir / "collision.obj")
    elif collision_ply.exists():
        # Just copy the .ply for now - SimplerEnv can handle both
        shutil.copy2(collision_ply, dst_dir / "collision.ply")
        # Also create a symlink as .obj
        try:
            (dst_dir / "collision.obj").symlink_to("collision.ply")
        except:
            shutil.copy2(collision_ply, dst_dir / "collision.obj")

    # Copy visual mesh and textures
    for f in src_dir.iterdir():
        if f.suffix in [".obj", ".mtl", ".png", ".jpg", ".dae"]:
            if f.name != "collision.obj":
                shutil.copy2(f, dst_dir / f.name)

    # Return metadata for info file
    return {
        "bbox": model_info.get("bbox", {"min": [-0.05, -0.05, -0.05], "max": [0.05, 0.05, 0.05]}),
        "scales": model_info.get("scales", [1.0]),
        "density": model_info.get("density", 1000),
    }


def convert_robocasa_asset(model_id, model_info, output_dir):
    """Convert a single RoboCasa asset to SimplerEnv format."""
    src_dir = Path(model_info["path"])
    dst_dir = output_dir / f"rc_{model_id}"

    if not src_dir.exists():
        print(f"  [Skip] Source not found: {src_dir}")
        return None

    dst_dir.mkdir(parents=True, exist_ok=True)

    # RoboCasa format has collision/ and visual/ subdirectories
    collision_dir = src_dir / "collision"
    visual_dir = src_dir / "visual"

    # Copy collision mesh (use first one if multiple parts)
    if collision_dir.exists():
        collision_files = sorted(collision_dir.glob("*.obj"))
        if collision_files:
            # Use model_normalized_collision_0.obj as primary collision mesh
            shutil.copy2(collision_files[0], dst_dir / "collision.obj")

    # Copy visual mesh and textures
    if visual_dir.exists():
        # Copy the main visual mesh (model_normalized_0.obj)
        visual_files = sorted(visual_dir.glob("model_normalized*.obj"))
        if visual_files:
            shutil.copy2(visual_files[0], dst_dir / "textured.obj")

        # Copy material file
        mtl_files = list(visual_dir.glob("*.mtl"))
        for mtl in mtl_files:
            shutil.copy2(mtl, dst_dir / mtl.name)

        # Copy textures (png, jpg, jpeg)
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for tex in visual_dir.glob(ext):
                shutil.copy2(tex, dst_dir / tex.name)

    # Estimate bounding box (will be refined when object is loaded)
    # These are reasonable defaults for kitchen objects
    category = model_info.get("category", "")

    # Category-specific size estimates
    size_estimates = {
        "fork": {"min": [-0.08, -0.015, -0.005], "max": [0.08, 0.015, 0.005]},
        "knife": {"min": [-0.10, -0.015, -0.005], "max": [0.10, 0.015, 0.005]},
        "spoon": {"min": [-0.07, -0.02, -0.01], "max": [0.07, 0.02, 0.01]},
        "spatula": {"min": [-0.12, -0.03, -0.01], "max": [0.12, 0.03, 0.01]},
        "ladle": {"min": [-0.10, -0.04, -0.04], "max": [0.10, 0.04, 0.04]},
        "apple": {"min": [-0.04, -0.04, -0.04], "max": [0.04, 0.04, 0.04]},
        "banana": {"min": [-0.08, -0.02, -0.02], "max": [0.08, 0.02, 0.02]},
        "orange": {"min": [-0.04, -0.04, -0.04], "max": [0.04, 0.04, 0.04]},
        "carrot": {"min": [-0.08, -0.015, -0.015], "max": [0.08, 0.015, 0.015]},
        "potato": {"min": [-0.04, -0.03, -0.03], "max": [0.04, 0.03, 0.03]},
        "tomato": {"min": [-0.03, -0.03, -0.03], "max": [0.03, 0.03, 0.03]},
        "lemon": {"min": [-0.03, -0.025, -0.025], "max": [0.03, 0.025, 0.025]},
        "bowl": {"min": [-0.06, -0.06, -0.03], "max": [0.06, 0.06, 0.03]},
        "cup": {"min": [-0.04, -0.04, -0.05], "max": [0.04, 0.04, 0.05]},
        "mug": {"min": [-0.045, -0.045, -0.05], "max": [0.045, 0.045, 0.05]},
        "plate": {"min": [-0.08, -0.08, -0.01], "max": [0.08, 0.08, 0.01]},
    }

    bbox = size_estimates.get(category, {"min": [-0.05, -0.05, -0.05], "max": [0.05, 0.05, 0.05]})

    # Density based on category
    density_map = {
        "fork": 2500,  # Metal
        "knife": 2500,
        "spoon": 2500,
        "spatula": 800,
        "ladle": 2000,
        "apple": 800,
        "banana": 600,
        "orange": 900,
        "carrot": 700,
        "potato": 1100,
        "tomato": 950,
        "lemon": 850,
        "bowl": 1000,
        "cup": 1000,
        "mug": 1200,
        "plate": 1000,
    }

    return {
        "bbox": bbox,
        "scales": [1.0],
        "density": density_map.get(category, 1000),
    }


def update_info_json(simplerenv_dir, new_assets):
    """Update info_bridge_custom_v0.json with new asset metadata."""
    info_path = simplerenv_dir / "info_bridge_custom_v0.json"

    # Load existing data
    if info_path.exists():
        with open(info_path) as f:
            info_data = json.load(f)
    else:
        info_data = {}

    # Add new assets
    info_data.update(new_assets)

    # Write back
    with open(info_path, "w") as f:
        json.dump(info_data, f, indent=2)

    print(f"[OK] Updated {info_path} with {len(new_assets)} new assets")


def main():
    parser = argparse.ArgumentParser(description="Setup YCB and RoboCasa assets for SimplerEnv clutter testing")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading assets")
    parser.add_argument("--simplerenv-path", type=str, help="Path to SimplerEnv custom data directory")
    parser.add_argument("--conda-env", type=str, default="maniskill", help="Conda environment name")
    parser.add_argument("--max-objects", type=int, default=50, help="Max objects per dataset to convert")
    args = parser.parse_args()

    # Find directories
    if args.simplerenv_path:
        simplerenv_dir = Path(args.simplerenv_path)
    else:
        simplerenv_dir = find_simplerenv_data_dir()

    if simplerenv_dir is None or not simplerenv_dir.exists():
        print("[Error] Could not find SimplerEnv data directory.")
        print("Please specify with --simplerenv-path")
        sys.exit(1)

    maniskill_dir = find_maniskill_asset_dir()

    print(f"SimplerEnv data dir: {simplerenv_dir}")
    print(f"ManiSkill asset dir: {maniskill_dir}")

    # Step 1: Download assets
    if not args.skip_download:
        download_assets(args.conda_env)

    # Step 2: Get available objects
    print("\n" + "=" * 60)
    print("Finding available objects...")
    print("=" * 60)

    ycb_objects = get_ycb_objects(maniskill_dir)
    robocasa_objects = get_robocasa_objects(maniskill_dir)

    print(f"Found {len(ycb_objects)} YCB objects")
    print(f"Found {len(robocasa_objects)} RoboCasa objects")

    # Step 3: Convert assets
    print("\n" + "=" * 60)
    print("Converting assets to SimplerEnv format...")
    print("=" * 60)

    models_dir = simplerenv_dir / "models"
    new_assets = {}

    # Convert YCB objects
    ycb_models_dir = maniskill_dir / "assets/mani_skill2_ycb/models"
    count = 0
    for model_id, model_info in ycb_objects.items():
        if count >= args.max_objects:
            break

        print(f"[YCB] Converting {model_id}...")
        asset_id = f"ycb_{model_id}"
        metadata = convert_ycb_asset(model_id, model_info, ycb_models_dir, models_dir)
        if metadata:
            new_assets[asset_id] = metadata
            count += 1

    print(f"Converted {count} YCB objects")

    # Convert RoboCasa objects (prioritize utensils and common kitchen objects)
    priority_categories = [
        "fork", "knife", "spoon", "spatula", "ladle",  # Utensils first!
        "apple", "banana", "orange", "lemon", "carrot", "tomato", "potato",  # Fruits/vegetables
        "bowl", "cup", "mug", "plate",  # Containers
    ]

    count = 0
    converted_categories = set()

    # First pass: prioritized categories
    for category in priority_categories:
        if count >= args.max_objects:
            break

        for model_id, model_info in robocasa_objects.items():
            if model_info.get("category") == category:
                print(f"[RoboCasa] Converting {model_id} ({category})...")
                asset_id = f"rc_{model_id}"
                metadata = convert_robocasa_asset(model_id, model_info, models_dir)
                if metadata:
                    new_assets[asset_id] = metadata
                    converted_categories.add(category)
                    count += 1
                    break  # One per category for now

    # Second pass: remaining objects
    for model_id, model_info in robocasa_objects.items():
        if count >= args.max_objects:
            break

        category = model_info.get("category", "")
        if category in converted_categories:
            continue

        print(f"[RoboCasa] Converting {model_id} ({category})...")
        asset_id = f"rc_{model_id}"
        metadata = convert_robocasa_asset(model_id, model_info, models_dir)
        if metadata:
            new_assets[asset_id] = metadata
            converted_categories.add(category)
            count += 1

    print(f"Converted {count} RoboCasa objects")

    # Step 4: Update info JSON
    if new_assets:
        update_info_json(simplerenv_dir, new_assets)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"\nTotal new assets: {len(new_assets)}")
    print(f"Assets directory: {models_dir}")
    print(f"\nTo list available objects:")
    print("  python scripts/list_available_objects.py")
    print(f"\nTo test with distractors:")
    print("  python scripts/try_checkpoint_in_simpler.py \\")
    print("      --task widowx_spoon_on_towel \\")
    print("      --distractors rc_fork_0 rc_knife_0 ycb_011_banana \\")
    print("      --num_episodes 3 --recording --use_bf16")


if __name__ == "__main__":
    main()
