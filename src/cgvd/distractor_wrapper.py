import numpy as np
import sapien.core as sapien
from pathlib import Path


# ============================================================================
# Available Objects for Clutter Testing
# ============================================================================
# These lists are for REFERENCE - use with --distractors flag in evaluation
# Run `python scripts/list_available_objects.py` to see what's actually available
# Run `python scripts/setup_clutter_assets.py` to download more assets
# ============================================================================

# YCB Dataset Objects (standard robotics benchmark)
# Download: python -m mani_skill.utils.download_asset ycb -y
# Converted objects will have "ycb_" prefix
YCB_OBJECTS = [
    # Cans and bottles
    "ycb_002_master_chef_can",
    "ycb_003_cracker_box",
    "ycb_004_sugar_box",
    "ycb_005_tomato_soup_can",
    "ycb_006_mustard_bottle",
    "ycb_007_tuna_fish_can",
    "ycb_008_pudding_box",
    "ycb_009_gelatin_box",
    "ycb_010_potted_meat_can",
    # Fruits
    "ycb_011_banana",
    "ycb_012_strawberry",
    "ycb_013_apple",
    "ycb_014_lemon",
    "ycb_015_peach",
    "ycb_016_pear",
    "ycb_017_orange",
    "ycb_018_plum",
    # Containers
    "ycb_024_bowl",
    "ycb_025_mug",
    "ycb_029_plate",
    # Other
    "ycb_019_pitcher_base",
    "ycb_021_bleach_cleanser",
    "ycb_035_power_drill",
    "ycb_036_wood_block",
    "ycb_037_scissors",
    "ycb_040_large_marker",
    "ycb_051_large_clamp",
    "ycb_052_extra_large_clamp",
    "ycb_061_foam_brick",
]

# RoboCasa Kitchen Objects (has utensils!)
# Download: python -m mani_skill.utils.download_asset RoboCasa -y
# Converted objects will have "rc_" prefix
ROBOCASA_OBJECTS = {
    # UTENSILS (key for VLA testing!)
    "utensils": [
        "rc_fork_0", "rc_fork_1", "rc_fork_2",
        "rc_knife_0", "rc_knife_1", "rc_knife_2",
        "rc_spoon_0", "rc_spoon_1", "rc_spoon_2",
        "rc_spatula_0", "rc_spatula_1",
        "rc_ladle_0", "rc_ladle_1",
        "rc_whisk_0",
    ],
    # Fruits
    "fruits": [
        "rc_apple_0", "rc_apple_1",
        "rc_banana_0", "rc_banana_1",
        "rc_orange_0", "rc_orange_1",
        "rc_lemon_0", "rc_lemon_1",
        "rc_lime_0",
        "rc_peach_0",
        "rc_pear_0",
        "rc_mango_0",
        "rc_kiwi_0",
        "rc_strawberry_0",
        "rc_grapes_0",
    ],
    # Vegetables
    "vegetables": [
        "rc_carrot_0", "rc_carrot_1",
        "rc_corn_0",
        "rc_cucumber_0",
        "rc_eggplant_0",
        "rc_garlic_0",
        "rc_onion_0",
        "rc_bell_pepper_0",
        "rc_potato_0",
        "rc_tomato_0",
        "rc_broccoli_0",
        "rc_mushroom_0",
    ],
    # Containers
    "containers": [
        "rc_bowl_0", "rc_bowl_1",
        "rc_cup_0", "rc_cup_1",
        "rc_mug_0", "rc_mug_1",
        "rc_plate_0", "rc_plate_1",
        "rc_pot_0",
        "rc_pan_0",
        "rc_pitcher_0",
    ],
    # Packaged foods
    "packaged_foods": [
        "rc_can_0", "rc_canned_food_0",
        "rc_boxed_food_0",
        "rc_cereal_0",
        "rc_jam_0",
        "rc_ketchup_0",
        "rc_yogurt_0",
    ],
}


class DistractorWrapper:
    """Wrapper to add distractor objects to SimplerEnv Bridge environments."""

    # Bridge environment objects (for widowx tasks)
    BRIDGE_DISTRACTORS = [
        "eggplant", "green_cube_3cm", "yellow_cube_3cm",
        "bridge_carrot_generated_modified", "bridge_spoon_generated_modified",
        "bridge_spoon_blue",  # Blue color variant of spoon
        "bridge_plate_objaverse", "sink",
    ]

    # Google Robot environment objects (for google_robot tasks)
    GOOGLE_ROBOT_DISTRACTORS = [
        "apple", "orange", "sponge", "blue_plastic_bottle", "eggplant",
        "opened_coke_can", "opened_pepsi_can", "opened_sprite_can",
        "bridge_carrot_generated_modified", "green_cube_3cm", "yellow_cube_3cm",
        "opened_fanta_can", "opened_redbull_can", "opened_7up_can",
    ]

    # Default scale multiplier for external dataset objects (they tend to be oversized)
    DEFAULT_EXTERNAL_ASSET_SCALE = 0.1  # 10% of original size for rc_* and ycb_* objects

    def __init__(self, env, distractor_ids, distractor_scale=None, external_asset_scale=None):
        """Initialize distractor wrapper.

        Args:
            env: The SimplerEnv environment to wrap
            distractor_ids: List of distractor object IDs to add
            distractor_scale: Optional scale multiplier for ALL distractors (0.0-1.0).
                            If set, overrides all other scale settings.
            external_asset_scale: Optional scale multiplier for rc_* and ycb_* objects only.
                            If None, uses DEFAULT_EXTERNAL_ASSET_SCALE (0.1).
                            Built-in objects (green_cube, eggplant, etc.) are unaffected.
        """
        self.env = env
        self.distractor_ids = distractor_ids
        self.distractor_scale = distractor_scale
        self.external_asset_scale = external_asset_scale if external_asset_scale is not None else self.DEFAULT_EXTERNAL_ASSET_SCALE
        self.distractor_objs = []
        self._distractors_loaded = False

    def _load_distractors(self):
        """Load distractor objects into the scene."""
        if self._distractors_loaded:
            return

        base_env = self.env.unwrapped
        scene = base_env._scene
        asset_root = base_env.asset_root
        model_db = base_env.model_db

        for model_id in self.distractor_ids:
            if model_id not in model_db:
                print(f"[Distractor] Warning: '{model_id}' not in model_db, skipping")
                print(f"[Distractor] Available objects: {list(model_db.keys())}")
                continue

            density = model_db[model_id].get("density", 1000)

            # Determine scale: use explicit distractor_scale, or model_db scale, or default
            if self.distractor_scale is not None:
                scale = self.distractor_scale
            else:
                # Use scale from model_db if available
                model_scales = model_db[model_id].get("scales", [1.0])
                scale = model_scales[0] if model_scales else 1.0

                # Apply scale reduction for external dataset objects (they tend to be oversized)
                # Built-in objects (green_cube, eggplant, bridge_*, etc.) keep their original scale
                if model_id.startswith("rc_") or model_id.startswith("ycb_"):
                    scale *= self.external_asset_scale

            obj = base_env._build_actor_helper(
                model_id, scene,
                scale=scale,
                density=density,
                physical_material=scene.create_physical_material(0.5, 0.5, 0.0),
                root_dir=asset_root,
            )
            obj.name = f"distractor_{model_id}"
            self.distractor_objs.append(obj)
            print(f"[Distractor] Loaded: {model_id} (scale={scale:.2f})")

        self._distractors_loaded = True
        print(f"[Distractor] Successfully loaded {len(self.distractor_objs)} distractor(s)")

    def _position_distractors(self, rng):
        """Position distractors randomly on table, avoiding task objects and robot."""
        base_env = self.env.unwrapped
        table_height = 0.87  # Bridge table height

        # Safety bubble around task objects - distractors must stay outside this radius
        SAFETY_BUBBLE_RADIUS = 0.08  # 8cm radius around each task object

        # Get task object positions to create safety bubbles
        safety_bubbles = []  # List of (x, y, radius)

        for attr in ['episode_target_obj', 'episode_source_obj']:
            if hasattr(base_env, attr):
                obj = getattr(base_env, attr)
                if obj is not None:
                    pos = obj.pose.p
                    safety_bubbles.append((pos[0], pos[1], SAFETY_BUBBLE_RADIUS))
                    print(f"[Distractor] Safety bubble around {attr}: center=({pos[0]:.3f}, {pos[1]:.3f}), radius={SAFETY_BUBBLE_RADIUS}m")

        # Table bounds - AVOID ROBOT WORKSPACE
        # Robot base is at x≈0, gripper extends to x≈-0.10
        # Safe table area is x < -0.12 (away from robot)
        # Table extends back to about x = -0.30
        TABLE_X_MIN, TABLE_X_MAX = -0.28, -0.12  # Keep away from robot (x > -0.12)
        TABLE_Y_MIN, TABLE_Y_MAX = -0.14, 0.14

        # 6 safe positions away from robot workspace
        edge_positions = [
            (-0.26, -0.12),   # Back-left
            (-0.26, 0.00),    # Back-center
            (-0.26, 0.12),    # Back-right
            (-0.20, -0.12),   # Mid-left
            (-0.20, 0.12),    # Mid-right
            (-0.14, 0.00),    # Front-center (but still away from robot)
        ]

        # Track placed distractor positions to avoid stacking
        placed_positions = []

        for i, obj in enumerate(self.distractor_objs):
            max_attempts = 50
            valid = False
            x, y = 0, 0

            for attempt in range(max_attempts):
                if i < len(edge_positions) and attempt == 0:
                    # Try edge position first with small jitter
                    x, y = edge_positions[i]
                    x += rng.uniform(-0.02, 0.02)
                    y += rng.uniform(-0.02, 0.02)
                    # Clamp to valid bounds
                    x = max(TABLE_X_MIN, min(TABLE_X_MAX, x))
                    y = max(TABLE_Y_MIN, min(TABLE_Y_MAX, y))
                else:
                    # Random position in valid table area
                    x = rng.uniform(TABLE_X_MIN, TABLE_X_MAX)
                    y = rng.uniform(TABLE_Y_MIN, TABLE_Y_MAX)

                # Check against task object safety bubbles (circular distance)
                valid = True
                for bx, by, radius in safety_bubbles:
                    dist = np.sqrt((x - bx)**2 + (y - by)**2)
                    if dist < radius:
                        valid = False
                        break

                # Also check against already-placed distractors (10cm minimum spacing)
                if valid:
                    for px, py in placed_positions:
                        dist = np.sqrt((x - px)**2 + (y - py)**2)
                        if dist < 0.10:
                            valid = False
                            break

                if valid:
                    break

            # Only place if valid position found
            if not valid:
                print(f"[Distractor] WARNING: Could not find valid position for {obj.name} after {max_attempts} attempts, skipping")
                # Move object far away (off-table) so it doesn't interfere
                obj.set_pose(sapien.Pose([10, 10, -10], [1, 0, 0, 0]))
                continue

            # Record this position
            placed_positions.append((x, y))

            z = table_height + 0.5
            obj.set_pose(sapien.Pose([x, y, z], [1, 0, 0, 0]))
            print(f"[Distractor] Positioned {obj.name} at ({x:.3f}, {y:.3f}, {z:.3f})")

        return safety_bubbles

    def _count_visible_distractors(self, initial_positions=None):
        """Count how many distractors are still on/above the table after physics.

        Args:
            initial_positions: List of (x, y, z) tuples of initial spawn positions.
                               If provided, also checks for objects stuck to robot.
        """
        table_height = 0.87
        count = 0
        for i, obj in enumerate(self.distractor_objs):
            pos = obj.pose.p
            x, y, z = pos[0], pos[1], pos[2]

            # Check if fell off table
            if z < table_height - 0.1:
                print(f"[Distractor] WARNING: {obj.name} fell off table (z={z:.3f})")
                continue

            # Check if in robot workspace (x > -0.10 means likely stuck to robot)
            if x > -0.10:
                print(f"[Distractor] WARNING: {obj.name} in robot workspace (x={x:.3f})")
                continue

            # Check if object moved significantly from initial spawn position
            # (objects that get stuck often don't fall to table properly)
            if initial_positions and i < len(initial_positions):
                init_x, init_y, init_z = initial_positions[i]
                if abs(z - init_z) < 0.1:  # Didn't fall much - might be stuck
                    print(f"[Distractor] WARNING: {obj.name} may be stuck (z barely changed: {init_z:.3f} -> {z:.3f})")
                    continue

            count += 1
            print(f"[Distractor] OK: {obj.name} at ({x:.3f}, {y:.3f}, {z:.3f})")

        return count

    def _remove_bubble_violators(self, safety_bubbles):
        """Remove distractors that ended up inside safety bubbles after physics."""
        base_env = self.env.unwrapped
        removed = []
        for obj in self.distractor_objs[:]:  # Copy list to allow removal
            pos = obj.pose.p
            for bx, by, radius in safety_bubbles:
                dist = np.sqrt((pos[0] - bx)**2 + (pos[1] - by)**2)
                if dist < radius:
                    print(f"[Distractor] REMOVING {obj.name} - inside safety bubble (dist={dist:.3f} < {radius})")
                    base_env._scene.remove_actor(obj)
                    self.distractor_objs.remove(obj)
                    removed.append(obj.name)
                    break
        return removed

    def _remove_touching_distractors(self, min_dist=0.05):
        """Remove distractors that are too close to each other after physics settling.

        When two objects are too close, removes the one that was added later (higher index).
        """
        base_env = self.env.unwrapped
        removed = []

        # Build list of (obj, position) for remaining distractors
        obj_positions = [(obj, obj.pose.p) for obj in self.distractor_objs]

        # Check each pair
        to_remove = set()
        for i, (obj_i, pos_i) in enumerate(obj_positions):
            if obj_i.name in to_remove:
                continue
            for j, (obj_j, pos_j) in enumerate(obj_positions[i+1:], start=i+1):
                if obj_j.name in to_remove:
                    continue
                dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                if dist < min_dist:
                    # Remove the later object (higher index)
                    print(f"[Distractor] REMOVING {obj_j.name} - too close to {obj_i.name} (dist={dist:.3f} < {min_dist})")
                    to_remove.add(obj_j.name)

        # Actually remove the objects
        for obj in self.distractor_objs[:]:
            if obj.name in to_remove:
                base_env._scene.remove_actor(obj)
                self.distractor_objs.remove(obj)
                removed.append(obj.name)

        return removed

    def reset(self, **kwargs):
        # Remove old distractor actors from scene (scene persists across resets)
        base_env = self.env.unwrapped
        if hasattr(base_env, '_scene') and base_env._scene is not None:
            for obj in self.distractor_objs:
                try:
                    base_env._scene.remove_actor(obj)
                    print(f"[Distractor] Removed old actor: {obj.name}")
                except Exception as e:
                    print(f"[Distractor] Could not remove {obj.name}: {e}")

        # Reset state
        self._distractors_loaded = False
        self.distractor_objs = []

        obs, info = self.env.reset(**kwargs)

        # Load distractors into the new scene
        self._load_distractors()

        # Position them randomly (spawns 0.5m above table)
        rng = np.random.RandomState(kwargs.get("options", {}).get("obj_init_options", {}).get("episode_id", 0))
        safety_bubbles = self._position_distractors(rng)

        # Record initial positions before physics
        initial_positions = []
        for obj in self.distractor_objs:
            pos = obj.pose.p
            initial_positions.append((pos[0], pos[1], pos[2]))

        # Let distractors settle with physics (matching SimplerEnv's multi-phase approach)
        base_env = self.env.unwrapped
        sim_freq = getattr(base_env, 'sim_freq', 500)

        # Phase 1: Lock rotation (prevent tipping), settle 0.5s
        for obj in self.distractor_objs:
            obj.lock_motion(0, 0, 0, 1, 1, 0)  # lock x,y rotation, allow z rotation
        for _ in range(int(sim_freq * 0.5)):
            base_env._scene.step()

        # Phase 2: Unlock, reset velocities, settle 0.5s more
        for obj in self.distractor_objs:
            obj.lock_motion(0, 0, 0, 0, 0, 0)  # unlock all
            obj.set_pose(obj.pose)  # explicit set to prevent sleep
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        for _ in range(int(sim_freq * 0.5)):
            base_env._scene.step()

        # Phase 3: Check if still moving, settle more if needed
        total_lin_vel = sum(np.linalg.norm(obj.velocity) for obj in self.distractor_objs)
        total_ang_vel = sum(np.linalg.norm(obj.angular_velocity) for obj in self.distractor_objs)
        if total_lin_vel > 1e-3 or total_ang_vel > 1e-2:
            for _ in range(int(sim_freq * 1.0)):  # extra 1.0s
                base_env._scene.step()

        # Remove distractors that drifted into safety bubbles during physics
        removed = self._remove_bubble_violators(safety_bubbles)
        if removed:
            print(f"[Distractor] Removed {len(removed)} objects that violated safety bubbles: {removed}")

        # Remove distractors that ended up too close to each other after settling
        removed_touching = self._remove_touching_distractors(min_dist=0.05)
        if removed_touching:
            print(f"[Distractor] Removed {len(removed_touching)} objects that were touching: {removed_touching}")

        # Log how many distractors are visible after settling
        visible_count = self._count_visible_distractors(initial_positions)
        print(f"[Distractor] After settling: {visible_count}/{len(self.distractor_objs)} distractors on table")

        # Re-capture observation now that distractors are in scene
        # base_env.get_obs() returns raw obs with "Color" key, but gym wrappers
        # (RGBDObservationWrapper) transform it to "rgb". Apply same transformation.
        raw_obs = base_env.get_obs()
        obs = self._transform_observation(raw_obs)

        return obs, info

    def _transform_observation(self, obs):
        """Transform raw observation to match gym wrapper format.

        Converts "Color" (float [0,1]) to "rgb" (uint8 [0,255]) to match
        what RGBDObservationWrapper does in the wrapper chain.
        """
        if "image" not in obs:
            return obs

        for cam_uid, cam_images in obs["image"].items():
            if "Color" in cam_images:
                # Convert Color (float RGBA) to rgb (uint8 RGB)
                color = cam_images["Color"]
                rgb = color[..., :3]  # Drop alpha channel
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                cam_images["rgb"] = rgb
                del cam_images["Color"]
            if "Position" in cam_images:
                # Convert Position to depth
                position = cam_images["Position"]
                depth = -position[..., [2]]  # Z component, negated
                cam_images["depth"] = depth
                del cam_images["Position"]

        return obs

    def step(self, action):
        return self.env.step(action)

    def __getattr__(self, name):
        return getattr(self.env, name)
