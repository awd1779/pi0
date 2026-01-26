#!/usr/bin/env python3
"""
List all available distractor objects for SimplerEnv clutter testing.

This script scans the SimplerEnv asset directories and info JSON files
to show all object IDs that can be used with the --distractors flag.

Usage:
    python scripts/list_available_objects.py [--simplerenv-path PATH] [--verbose]
"""

import argparse
import json
from pathlib import Path


def find_simplerenv_data_dir():
    """Find SimplerEnv custom data directory."""
    # allenzren first - that's what uv uses
    candidates = [
        Path.home() / "allenzren_SimplerEnv/ManiSkill2_real2sim/data/custom",
        Path.home() / "SimplerEnv-OpenVLA/ManiSkill2_real2sim/data/custom",
        Path.home() / "SimplerEnv/ManiSkill2_real2sim/data/custom",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_info_json(simplerenv_dir):
    """Load info_bridge_custom_v0.json."""
    info_path = simplerenv_dir / "info_bridge_custom_v0.json"
    if info_path.exists():
        with open(info_path) as f:
            return json.load(f)
    return {}


def scan_models_dir(simplerenv_dir):
    """Scan models directory for available objects."""
    models_dir = simplerenv_dir / "models"
    objects = {}

    if not models_dir.exists():
        return objects

    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_id = model_dir.name

        # Check for required files
        has_collision = (model_dir / "collision.obj").exists() or (model_dir / "collision.ply").exists()
        has_visual = any((model_dir / f"textured{ext}").exists() for ext in [".obj", ".dae", ".glb"])

        if has_collision or has_visual:
            # Categorize by prefix
            if model_id.startswith("ycb_"):
                source = "YCB"
            elif model_id.startswith("rc_"):
                source = "RoboCasa"
            elif model_id.startswith("bridge_"):
                source = "Bridge"
            else:
                source = "Other"

            objects[model_id] = {
                "source": source,
                "has_collision": has_collision,
                "has_visual": has_visual,
            }

    return objects


def categorize_objects(objects):
    """Categorize objects by type."""
    categories = {
        "utensils": [],
        "fruits": [],
        "vegetables": [],
        "containers": [],
        "cans": [],
        "cubes": [],
        "other": [],
    }

    utensil_keywords = ["fork", "knife", "spoon", "spatula", "ladle", "whisk"]
    fruit_keywords = ["apple", "banana", "orange", "lemon", "lime", "peach", "pear", "mango", "kiwi", "grape"]
    vegetable_keywords = ["carrot", "corn", "cucumber", "eggplant", "garlic", "onion", "pepper", "potato", "tomato", "broccoli"]
    container_keywords = ["bowl", "cup", "mug", "plate", "pot", "pan", "pitcher", "jar", "bottle"]
    can_keywords = ["can", "coke", "pepsi", "sprite", "fanta", "redbull", "7up"]
    cube_keywords = ["cube"]

    for obj_id in objects:
        obj_lower = obj_id.lower()

        if any(kw in obj_lower for kw in utensil_keywords):
            categories["utensils"].append(obj_id)
        elif any(kw in obj_lower for kw in fruit_keywords):
            categories["fruits"].append(obj_id)
        elif any(kw in obj_lower for kw in vegetable_keywords):
            categories["vegetables"].append(obj_id)
        elif any(kw in obj_lower for kw in container_keywords):
            categories["containers"].append(obj_id)
        elif any(kw in obj_lower for kw in can_keywords):
            categories["cans"].append(obj_id)
        elif any(kw in obj_lower for kw in cube_keywords):
            categories["cubes"].append(obj_id)
        else:
            categories["other"].append(obj_id)

    return categories


def main():
    parser = argparse.ArgumentParser(description="List available distractor objects")
    parser.add_argument("--simplerenv-path", type=str, help="Path to SimplerEnv custom data directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Find SimplerEnv directory
    if args.simplerenv_path:
        simplerenv_dir = Path(args.simplerenv_path)
    else:
        simplerenv_dir = find_simplerenv_data_dir()

    if simplerenv_dir is None or not simplerenv_dir.exists():
        print("[Error] Could not find SimplerEnv data directory.")
        print("Please specify with --simplerenv-path")
        return

    # Load objects from info JSON and scan models directory
    info_objects = load_info_json(simplerenv_dir)
    dir_objects = scan_models_dir(simplerenv_dir)

    # Merge both sources
    all_objects = set(info_objects.keys()) | set(dir_objects.keys())

    if args.json:
        import json as json_module
        print(json_module.dumps(sorted(all_objects), indent=2))
        return

    # Categorize objects
    categories = categorize_objects(all_objects)

    # Print results
    print("=" * 70)
    print("AVAILABLE DISTRACTOR OBJECTS")
    print("=" * 70)
    print(f"SimplerEnv data dir: {simplerenv_dir}")
    print(f"Total objects: {len(all_objects)}")
    print()

    # Count by source
    sources = {"YCB": 0, "RoboCasa": 0, "Bridge": 0, "Other": 0}
    for obj_id in all_objects:
        if obj_id.startswith("ycb_"):
            sources["YCB"] += 1
        elif obj_id.startswith("rc_"):
            sources["RoboCasa"] += 1
        elif obj_id.startswith("bridge_"):
            sources["Bridge"] += 1
        else:
            sources["Other"] += 1

    print("By source:")
    for source, count in sources.items():
        if count > 0:
            print(f"  {source}: {count}")
    print()

    # Print by category
    print("-" * 70)
    print("BY CATEGORY")
    print("-" * 70)

    category_order = ["utensils", "fruits", "vegetables", "containers", "cans", "cubes", "other"]
    category_emoji = {
        "utensils": "🍴",
        "fruits": "🍎",
        "vegetables": "🥕",
        "containers": "🥣",
        "cans": "🥫",
        "cubes": "🧊",
        "other": "📦",
    }

    for cat in category_order:
        items = sorted(categories[cat])
        if items:
            emoji = category_emoji.get(cat, "")
            print(f"\n{emoji} {cat.upper()} ({len(items)}):")
            if args.verbose:
                for item in items:
                    source = "YCB" if item.startswith("ycb_") else "RoboCasa" if item.startswith("rc_") else "Bridge"
                    print(f"    {item} [{source}]")
            else:
                # Print in columns
                col_width = 35
                for i in range(0, len(items), 2):
                    row = items[i:i+2]
                    print("    " + "  ".join(f"{x:<{col_width}}" for x in row))

    # Print usage examples
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)

    print("\n# Test with kitchen utensils:")
    utensils = categories["utensils"][:3] if categories["utensils"] else ["rc_fork_0", "rc_knife_0", "rc_spoon_0"]
    print(f"python scripts/try_checkpoint_in_simpler.py \\")
    print(f"    --task widowx_spoon_on_towel \\")
    print(f"    --distractors {' '.join(utensils)} \\")
    print(f"    --num_episodes 5 --recording --use_bf16")

    print("\n# Test with fruits and vegetables:")
    fruits_vegs = (categories["fruits"][:2] + categories["vegetables"][:2])[:4]
    if not fruits_vegs:
        fruits_vegs = ["apple", "orange", "eggplant", "bridge_carrot_generated_modified"]
    print(f"python scripts/try_checkpoint_in_simpler.py \\")
    print(f"    --task widowx_carrot_on_plate \\")
    print(f"    --distractors {' '.join(fruits_vegs)} \\")
    print(f"    --num_episodes 5 --recording --use_bf16")

    print("\n# Heavy clutter test (6 objects):")
    heavy_clutter = (categories["utensils"][:2] + categories["fruits"][:2] + categories["cubes"][:2])[:6]
    if len(heavy_clutter) < 6:
        heavy_clutter = ["green_cube_3cm", "yellow_cube_3cm", "eggplant", "apple", "orange", "sponge"]
    print(f"python scripts/try_checkpoint_in_simpler.py \\")
    print(f"    --task widowx_spoon_on_towel \\")
    print(f"    --distractors {' '.join(heavy_clutter)} \\")
    print(f"    --num_episodes 5 --recording --use_bf16")


if __name__ == "__main__":
    main()
