#!/usr/bin/env python3
"""
Check integrity of distractor assets for SimplerEnv.

This script validates:
1. Collision mesh files exist and are parseable
2. Z-bounds of meshes (detect extreme offsets like RoboCasa objects)
3. Empty or corrupted files
4. Missing textures or materials

Usage:
    python scripts/check_asset_integrity.py [--verbose] [--fix]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


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


def parse_obj_file(obj_path: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Parse OBJ file and extract vertices.

    Returns:
        Tuple of (vertices array, error message or None)
    """
    if not obj_path.exists():
        return None, f"File not found: {obj_path}"

    if obj_path.stat().st_size == 0:
        return None, f"Empty file: {obj_path}"

    vertices = []
    try:
        with open(obj_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            vertices.append([x, y, z])
                        except ValueError as e:
                            return None, f"Parse error at line {line_num}: {e}"
    except Exception as e:
        return None, f"Read error: {e}"

    if not vertices:
        return None, "No vertices found in OBJ file"

    return np.array(vertices), None


def compute_mesh_bounds(vertices: np.ndarray) -> Dict:
    """Compute bounding box statistics from vertices."""
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2
    extent = maxs - mins

    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "center": center.tolist(),
        "extent": extent.tolist(),
        "z_offset": center[2],  # How far from origin the Z center is
        "height": extent[2],
    }


def check_model(model_dir: Path, verbose: bool = False) -> Dict:
    """Check a single model directory for issues."""
    model_id = model_dir.name
    issues = []
    info = {"model_id": model_id, "path": str(model_dir)}

    # Check for collision mesh
    collision_obj = model_dir / "collision.obj"
    collision_ply = model_dir / "collision.ply"

    if collision_obj.exists():
        vertices, error = parse_obj_file(collision_obj)
        if error:
            issues.append(f"Collision mesh error: {error}")
        elif vertices is not None:
            bounds = compute_mesh_bounds(vertices)
            info["collision_bounds"] = bounds
            info["num_collision_vertices"] = len(vertices)

            # Flag extreme Z offsets (>5cm from origin)
            if abs(bounds["z_offset"]) > 0.05:
                issues.append(f"Z-offset warning: center at Z={bounds['z_offset']:.3f}m (mesh not centered)")

            # Flag very large or very small objects
            if bounds["height"] > 1.0:
                issues.append(f"Very large object: height={bounds['height']:.2f}m")
            elif bounds["height"] < 0.001:
                issues.append(f"Very small object: height={bounds['height']:.4f}m")
    elif collision_ply.exists():
        info["collision_format"] = "ply"
        # PLY parsing is more complex, just note it exists
        if collision_ply.stat().st_size == 0:
            issues.append("Empty collision.ply file")
    else:
        issues.append("No collision mesh found (collision.obj or collision.ply)")

    # Check for visual mesh
    visual_files = list(model_dir.glob("textured.*")) + list(model_dir.glob("*.dae")) + list(model_dir.glob("*.glb"))
    if not visual_files:
        # Check for any .obj that's not collision
        obj_files = [f for f in model_dir.glob("*.obj") if f.name != "collision.obj"]
        if obj_files:
            visual_files = obj_files

    if visual_files:
        info["visual_files"] = [f.name for f in visual_files]
        for vf in visual_files:
            if vf.stat().st_size == 0:
                issues.append(f"Empty visual file: {vf.name}")
    else:
        issues.append("No visual mesh found")

    # Check for textures
    texture_exts = ["*.png", "*.jpg", "*.jpeg"]
    textures = []
    for ext in texture_exts:
        textures.extend(model_dir.glob(ext))
    info["textures"] = [t.name for t in textures]

    # Check for material files
    mtl_files = list(model_dir.glob("*.mtl"))
    if mtl_files:
        info["materials"] = [m.name for m in mtl_files]
        # Check if referenced textures exist
        for mtl in mtl_files:
            try:
                with open(mtl, 'r') as f:
                    for line in f:
                        if line.strip().startswith("map_"):
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                tex_name = parts[-1]
                                tex_path = model_dir / tex_name
                                if not tex_path.exists():
                                    issues.append(f"Missing texture: {tex_name} (referenced in {mtl.name})")
            except Exception as e:
                issues.append(f"Error reading {mtl.name}: {e}")

    info["issues"] = issues
    info["is_valid"] = len(issues) == 0 or all("warning" in i.lower() for i in issues)

    return info


def main():
    parser = argparse.ArgumentParser(description="Check asset integrity")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")
    parser.add_argument("--filter", type=str, help="Filter models by prefix (e.g., 'rc_', 'ycb_')")
    parser.add_argument("--show-bounds", action="store_true", help="Show Z-bounds for all objects")
    parser.add_argument("--only-issues", action="store_true", help="Only show models with issues")
    args = parser.parse_args()

    simplerenv_dir = find_simplerenv_data_dir()
    if simplerenv_dir is None:
        print("[Error] Could not find SimplerEnv data directory")
        return

    models_dir = simplerenv_dir / "models"
    if not models_dir.exists():
        print(f"[Error] Models directory not found: {models_dir}")
        return

    print("=" * 70)
    print("ASSET INTEGRITY CHECK")
    print("=" * 70)
    print(f"Models directory: {models_dir}")
    print()

    # Get all model directories
    model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])

    if args.filter:
        model_dirs = [d for d in model_dirs if d.name.startswith(args.filter)]

    print(f"Checking {len(model_dirs)} models...")
    print()

    # Check each model
    results = []
    issues_count = 0
    z_offset_issues = []

    for model_dir in model_dirs:
        info = check_model(model_dir, args.verbose)
        results.append(info)

        if info["issues"]:
            issues_count += 1

            # Track Z-offset issues separately
            for issue in info["issues"]:
                if "Z-offset" in issue:
                    z_offset_issues.append(info)

        if args.only_issues and not info["issues"]:
            continue

        if args.verbose or info["issues"]:
            status = "[OK]" if info["is_valid"] else "[ISSUE]"
            print(f"{status} {info['model_id']}")

            if args.show_bounds and "collision_bounds" in info:
                b = info["collision_bounds"]
                print(f"    Bounds: Z=[{b['min'][2]:.3f}, {b['max'][2]:.3f}], center=({b['center'][0]:.3f}, {b['center'][1]:.3f}, {b['center'][2]:.3f})")

            for issue in info["issues"]:
                print(f"    - {issue}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total models: {len(results)}")
    print(f"Models with issues: {issues_count}")
    print(f"Models with Z-offset: {len(z_offset_issues)}")

    if z_offset_issues:
        print()
        print("-" * 70)
        print("OBJECTS WITH SIGNIFICANT Z-OFFSET (mesh not centered at origin)")
        print("-" * 70)
        print("These objects may spawn inside the table if not handled correctly.")
        print()

        # Sort by Z-offset magnitude
        z_offset_issues.sort(key=lambda x: abs(x.get("collision_bounds", {}).get("z_offset", 0)), reverse=True)

        for info in z_offset_issues[:20]:  # Show top 20
            b = info.get("collision_bounds", {})
            z_off = b.get("z_offset", 0)
            height = b.get("height", 0)
            print(f"  {info['model_id']:40s}  Z-offset: {z_off:+.3f}m  height: {height:.3f}m")

    # Check for specific problematic patterns
    print()
    print("-" * 70)
    print("CORRUPTION CHECK")
    print("-" * 70)

    empty_files = [r for r in results if any("Empty" in i for i in r["issues"])]
    missing_collision = [r for r in results if any("No collision" in i for i in r["issues"])]
    missing_visual = [r for r in results if any("No visual" in i for i in r["issues"])]
    parse_errors = [r for r in results if any("Parse error" in i or "Read error" in i for i in r["issues"])]

    if empty_files:
        print(f"Empty files: {len(empty_files)}")
        for r in empty_files[:5]:
            print(f"  - {r['model_id']}")
    else:
        print("Empty files: None")

    if missing_collision:
        print(f"Missing collision mesh: {len(missing_collision)}")
        for r in missing_collision[:5]:
            print(f"  - {r['model_id']}")
    else:
        print("Missing collision mesh: None")

    if parse_errors:
        print(f"Parse/read errors: {len(parse_errors)}")
        for r in parse_errors[:5]:
            print(f"  - {r['model_id']}: {r['issues']}")
    else:
        print("Parse/read errors: None")

    if not (empty_files or missing_collision or parse_errors):
        print()
        print("[OK] No corruption detected in asset files!")
    else:
        print()
        print("[WARNING] Some assets may need repair or re-download")


if __name__ == "__main__":
    main()
