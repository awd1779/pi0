#!/usr/bin/env python3
"""
Convert ALL utensil objects from RoboCasa to SimplerEnv format.

This script:
1. Converts all forks, knives, spatulas, ladles (NOT spoons) from RoboCasa
2. Properly merges multiple OBJ files into one
3. Copies textures and preserves material references
4. Updates info_bridge_custom_v0.json

Usage:
    python scripts/convert_all_utensils.py [--dry-run] [--include-spoons]
"""

import argparse
import json
import shutil
from pathlib import Path


def find_simplerenv_data_dir():
    """Find SimplerEnv custom data directory."""
    candidates = [
        Path.home() / "allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom",
        Path.home() / "SimplerEnv-OpenVLA/ManiSkill2_real2sim/data/custom",
        Path.home() / "SimplerEnv/ManiSkill2_real2sim/data/custom",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_robocasa_dir():
    """Find RoboCasa objects directory."""
    return Path.home() / ".maniskill/data/scene_datasets/robocasa_dataset/assets/objects/objaverse"


def merge_obj_files(obj_files, output_path):
    """
    Merge multiple OBJ files into a single file.

    Handles:
    - Vertex position offsets (v)
    - Texture coordinate offsets (vt)
    - Vertex normal offsets (vn)
    - Material library references (mtllib)
    - Material usage (usemtl)

    Returns:
        Tuple of (num_vertices, num_faces)
    """
    all_vertices = []
    all_texcoords = []
    all_normals = []
    all_faces = []
    mtllib = None
    usemtl = None

    vertex_offset = 0
    texcoord_offset = 0
    normal_offset = 0

    for obj_file in sorted(obj_files):
        vertices = []
        texcoords = []
        normals = []
        faces = []

        with open(obj_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('mtllib ') and mtllib is None:
                    mtllib = line
                elif line.startswith('usemtl ') and usemtl is None:
                    usemtl = line
                elif line.startswith('v '):
                    vertices.append(line)
                elif line.startswith('vt '):
                    texcoords.append(line)
                elif line.startswith('vn '):
                    normals.append(line)
                elif line.startswith('f '):
                    parts = line.split()
                    new_parts = ['f']
                    for part in parts[1:]:
                        indices = part.split('/')
                        new_vertex_idx = int(indices[0]) + vertex_offset

                        if len(indices) == 1:
                            new_parts.append(str(new_vertex_idx))
                        elif len(indices) == 2:
                            new_vt_idx = int(indices[1]) + texcoord_offset if indices[1] else ''
                            new_parts.append(f"{new_vertex_idx}/{new_vt_idx}")
                        else:
                            new_vt_idx = int(indices[1]) + texcoord_offset if indices[1] else ''
                            new_vn_idx = int(indices[2]) + normal_offset if indices[2] else ''
                            new_parts.append(f"{new_vertex_idx}/{new_vt_idx}/{new_vn_idx}")

                    faces.append(' '.join(new_parts))

        all_vertices.extend(vertices)
        all_texcoords.extend(texcoords)
        all_normals.extend(normals)
        all_faces.extend(faces)

        vertex_offset += len(vertices)
        texcoord_offset += len(texcoords)
        normal_offset += len(normals)

    with open(output_path, 'w') as f:
        f.write(f"# Merged from {len(obj_files)} OBJ files\n")
        if mtllib:
            f.write(mtllib + '\n')
        if usemtl:
            f.write(usemtl + '\n')
        for v in all_vertices:
            f.write(v + '\n')
        for vt in all_texcoords:
            f.write(vt + '\n')
        for vn in all_normals:
            f.write(vn + '\n')
        for face in all_faces:
            f.write(face + '\n')

    return len(all_vertices), len(all_faces)


def convert_robocasa_object(category, model_name, src_dir, output_dir, dry_run=False):
    """
    Convert a single RoboCasa object to SimplerEnv format.

    Args:
        category: Object category (fork, knife, etc.)
        model_name: Model directory name (fork_0, knife_1, etc.)
        src_dir: Source directory path
        output_dir: Output models directory
        dry_run: If True, don't actually copy files

    Returns:
        Tuple of (asset_id, metadata) or (None, None) if failed
    """
    visual_dir = src_dir / "visual"
    collision_dir = src_dir / "collision"

    if not visual_dir.exists():
        print(f"  [Skip] No visual directory: {src_dir}")
        return None, None

    asset_id = f"rc_{model_name}"
    dst_dir = output_dir / asset_id

    if dry_run:
        print(f"  [DRY-RUN] Would convert {model_name} -> {asset_id}")
        # Still return metadata for counting
        return asset_id, get_metadata_for_category(category)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. Merge collision meshes
    if collision_dir.exists():
        collision_files = sorted(collision_dir.glob("*.obj"))
        if collision_files:
            if len(collision_files) == 1:
                shutil.copy2(collision_files[0], dst_dir / "collision.obj")
            else:
                num_verts, num_faces = merge_obj_files(collision_files, dst_dir / "collision.obj")
                print(f"    Merged {len(collision_files)} collision parts -> {num_verts}v, {num_faces}f")

    # 2. Merge visual meshes
    visual_files = sorted(visual_dir.glob("model_normalized*.obj"))
    if visual_files:
        if len(visual_files) == 1:
            shutil.copy2(visual_files[0], dst_dir / "textured.obj")
        else:
            num_verts, num_faces = merge_obj_files(visual_files, dst_dir / "textured.obj")
            print(f"    Merged {len(visual_files)} visual parts -> {num_verts}v, {num_faces}f")
    else:
        print(f"  [Skip] No visual OBJ files: {visual_dir}")
        return None, None

    # 3. Copy material file
    mtl_files = list(visual_dir.glob("*.mtl"))
    for mtl in mtl_files:
        shutil.copy2(mtl, dst_dir / mtl.name)

    # 4. Copy ALL texture files (png, jpg, jpeg)
    texture_count = 0
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
        for tex in visual_dir.glob(ext):
            shutil.copy2(tex, dst_dir / tex.name)
            texture_count += 1

    if texture_count > 0:
        print(f"    Copied {texture_count} texture files")

    return asset_id, get_metadata_for_category(category)


def get_metadata_for_category(category):
    """Get bbox and density metadata for a category."""
    size_estimates = {
        # Utensils
        "fork": {"min": [-0.08, -0.015, -0.005], "max": [0.08, 0.015, 0.005]},
        "knife": {"min": [-0.10, -0.015, -0.005], "max": [0.10, 0.015, 0.005]},
        "spoon": {"min": [-0.07, -0.02, -0.01], "max": [0.07, 0.02, 0.01]},
        "spatula": {"min": [-0.12, -0.03, -0.01], "max": [0.12, 0.03, 0.01]},
        "ladle": {"min": [-0.10, -0.04, -0.04], "max": [0.10, 0.04, 0.04]},
        "scissors": {"min": [-0.08, -0.03, -0.01], "max": [0.08, 0.03, 0.01]},
        # Containers
        "bowl": {"min": [-0.08, -0.08, -0.04], "max": [0.08, 0.08, 0.04]},
        "mug": {"min": [-0.05, -0.05, -0.05], "max": [0.05, 0.05, 0.05]},
        "plate": {"min": [-0.10, -0.10, -0.02], "max": [0.10, 0.10, 0.02]},
        # Vegetables
        "corn": {"min": [-0.08, -0.02, -0.02], "max": [0.08, 0.02, 0.02]},
        "cucumber": {"min": [-0.10, -0.02, -0.02], "max": [0.10, 0.02, 0.02]},
        "eggplant": {"min": [-0.10, -0.04, -0.04], "max": [0.10, 0.04, 0.04]},
        "garlic": {"min": [-0.03, -0.03, -0.03], "max": [0.03, 0.03, 0.03]},
        "onion": {"min": [-0.04, -0.04, -0.04], "max": [0.04, 0.04, 0.04]},
        "bell_pepper": {"min": [-0.04, -0.04, -0.05], "max": [0.04, 0.04, 0.05]},
        "potato": {"min": [-0.05, -0.03, -0.03], "max": [0.05, 0.03, 0.03]},
        "tomato": {"min": [-0.04, -0.04, -0.04], "max": [0.04, 0.04, 0.04]},
        "broccoli": {"min": [-0.05, -0.05, -0.05], "max": [0.05, 0.05, 0.05]},
        "squash": {"min": [-0.08, -0.04, -0.04], "max": [0.08, 0.04, 0.04]},
        "sweet_potato": {"min": [-0.06, -0.03, -0.03], "max": [0.06, 0.03, 0.03]},
    }

    density_map = {
        # Utensils (metal)
        "fork": 2500,
        "knife": 2500,
        "spoon": 2500,
        "spatula": 800,
        "ladle": 2000,
        "scissors": 2500,
        # Containers (ceramic/glass)
        "bowl": 2000,
        "mug": 2000,
        "plate": 2000,
        # Vegetables (organic)
        "corn": 800,
        "cucumber": 900,
        "eggplant": 700,
        "garlic": 1000,
        "onion": 900,
        "bell_pepper": 600,
        "potato": 1100,
        "tomato": 950,
        "broccoli": 500,
        "squash": 800,
        "sweet_potato": 1050,
    }

    return {
        "bbox": size_estimates.get(category, {"min": [-0.05, -0.05, -0.05], "max": [0.05, 0.05, 0.05]}),
        "scales": [1.0],
        "density": density_map.get(category, 1000),
    }


def update_info_json(simplerenv_dir, new_assets):
    """Update info_bridge_custom_v0.json with new asset metadata."""
    info_path = simplerenv_dir / "info_bridge_custom_v0.json"

    if info_path.exists():
        with open(info_path) as f:
            info_data = json.load(f)
    else:
        info_data = {}

    info_data.update(new_assets)

    with open(info_path, "w") as f:
        json.dump(info_data, f, indent=2)

    print(f"[OK] Updated {info_path} with {len(new_assets)} assets")


def main():
    parser = argparse.ArgumentParser(description="Convert all RoboCasa utensils to SimplerEnv format")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be converted without doing it")
    parser.add_argument("--include-spoons", action="store_true", help="Also convert spoons (excluded by default)")
    parser.add_argument("--categories", type=str, nargs="+",
                        default=["fork", "knife", "spatula", "ladle"],
                        help="Categories to convert (default: fork knife spatula ladle)")
    args = parser.parse_args()

    # Add spoons if requested
    if args.include_spoons and "spoon" not in args.categories:
        args.categories.append("spoon")

    # Find directories
    simplerenv_dir = find_simplerenv_data_dir()
    robocasa_dir = find_robocasa_dir()

    if simplerenv_dir is None:
        print("[Error] Could not find SimplerEnv data directory")
        return

    if not robocasa_dir.exists():
        print(f"[Error] RoboCasa directory not found: {robocasa_dir}")
        print("Run: python -m mani_skill.utils.download_asset RoboCasa -y")
        return

    print("=" * 60)
    print("CONVERTING ROBOCASA UTENSILS TO SIMPLERENV FORMAT")
    print("=" * 60)
    print(f"SimplerEnv dir: {simplerenv_dir}")
    print(f"RoboCasa dir: {robocasa_dir}")
    print(f"Categories: {args.categories}")
    print(f"Dry run: {args.dry_run}")
    print()

    models_dir = simplerenv_dir / "models"
    new_assets = {}

    for category in args.categories:
        category_dir = robocasa_dir / category
        if not category_dir.exists():
            print(f"[Warning] Category directory not found: {category_dir}")
            continue

        # Get all model directories
        model_dirs = sorted([d for d in category_dir.iterdir() if d.is_dir()])
        print(f"\n[{category.upper()}] Found {len(model_dirs)} objects")

        for model_dir in model_dirs:
            model_name = model_dir.name
            print(f"  Converting {model_name}...")

            asset_id, metadata = convert_robocasa_object(
                category, model_name, model_dir, models_dir, dry_run=args.dry_run
            )

            if asset_id and metadata:
                new_assets[asset_id] = metadata

    # Update info JSON
    if new_assets and not args.dry_run:
        update_info_json(simplerenv_dir, new_assets)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total objects converted: {len(new_assets)}")

    for category in args.categories:
        count = sum(1 for k in new_assets if k.startswith(f"rc_{category}_"))
        print(f"  {category}: {count}")

    if args.dry_run:
        print("\n[DRY-RUN] No files were modified. Run without --dry-run to convert.")
    else:
        print(f"\nObjects saved to: {models_dir}")
        print("\nTo verify, run:")
        print("  uv run python scripts/list_available_objects.py --verbose | grep rc_")


if __name__ == "__main__":
    main()
