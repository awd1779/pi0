import numpy as np
import sapien.core as sapien
from pathlib import Path


def get_actor_xy_radius(actor):
    """Compute XY bounding radius for an actor from its collision shapes.

    Returns the radius of the smallest circle (centered at actor origin)
    that contains the actor's XY footprint.
    """
    max_radius = 0.0

    for shape in actor.get_collision_shapes():
        geom = shape.geometry

        if isinstance(geom, sapien.BoxGeometry):
            half = geom.half_lengths
            # XY diagonal / 2
            radius = np.sqrt(half[0]**2 + half[1]**2)

        elif isinstance(geom, sapien.SphereGeometry):
            radius = geom.radius

        elif isinstance(geom, sapien.ConvexMeshGeometry):
            verts = np.array(geom.vertices) * np.array(geom.scale)
            # XY extent from vertices
            xy_min = verts[:, :2].min(axis=0)
            xy_max = verts[:, :2].max(axis=0)
            extent = xy_max - xy_min
            radius = np.sqrt(extent[0]**2 + extent[1]**2) / 2

        elif isinstance(geom, sapien.CapsuleGeometry):
            # Capsule: radius + half_length in one direction
            radius = geom.radius + geom.half_length

        else:
            # Fallback for unknown geometry types
            radius = 0.05

        max_radius = max(max_radius, radius)

    return max_radius


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
            distractor_ids: List of distractor object IDs to add. Supports per-object scales
                            using format "object_id:scale" (e.g., "rc_fork_11:0.5").
            distractor_scale: Optional scale multiplier for ALL distractors (0.0-1.0).
                            If set, overrides all other scale settings.
            external_asset_scale: Optional scale multiplier for rc_* and ycb_* objects only.
                            If None, uses DEFAULT_EXTERNAL_ASSET_SCALE (0.1).
                            Built-in objects (green_cube, eggplant, etc.) are unaffected.
                            Utensils (fork, knife, spoon, etc.) default to 1.0.
        """
        self.env = env
        # Parse distractor_ids for per-object scales (format: "object_id:scale")
        self.distractor_ids = []
        self.per_object_scales = {}  # object_id -> scale
        for item in distractor_ids:
            if ":" in item:
                obj_id, scale_str = item.rsplit(":", 1)
                try:
                    self.per_object_scales[obj_id] = float(scale_str)
                    self.distractor_ids.append(obj_id)
                except ValueError:
                    # Not a valid scale, treat whole thing as object ID
                    self.distractor_ids.append(item)
            else:
                self.distractor_ids.append(item)

        self.distractor_scale = distractor_scale
        self.external_asset_scale = external_asset_scale if external_asset_scale is not None else self.DEFAULT_EXTERNAL_ASSET_SCALE
        self.distractor_objs = []
        self.distractor_radii = []  # XY bounding radius for each distractor
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

            # Check if this is a utensil (which shouldn't be scaled down by default)
            is_utensil = any(u in model_id.lower() for u in ["fork", "knife", "spoon", "spatula", "ladle", "whisk"])

            # Determine scale with priority:
            # 1. Per-object scale (e.g., "rc_fork_11:0.5")
            # 2. Global distractor_scale (applies to all)
            # 3. Default logic (utensils=1.0, external=0.1, built-in=1.0)
            scale_source = "default"
            if model_id in self.per_object_scales:
                scale = self.per_object_scales[model_id]
                scale_source = "per-object"
            elif self.distractor_scale is not None:
                scale = self.distractor_scale
                scale_source = "global"
            else:
                # Use scale from model_db if available
                model_scales = model_db[model_id].get("scales", [1.0])
                scale = model_scales[0] if model_scales else 1.0

                # Apply scale reduction for external dataset objects (they tend to be oversized)
                # EXCEPT utensils which are already correctly sized
                # Built-in objects (green_cube, eggplant, bridge_*, etc.) keep their original scale
                is_external = model_id.startswith("rc_") or model_id.startswith("ycb_")
                if is_external and not is_utensil:
                    scale *= self.external_asset_scale
                    scale_source = "external"
                elif is_utensil:
                    scale_source = "utensil"

            obj = base_env._build_actor_helper(
                model_id, scene,
                scale=scale,
                density=density,
                physical_material=scene.create_physical_material(0.5, 0.5, 0.0),
                root_dir=asset_root,
            )
            obj.name = f"distractor_{model_id}"
            self.distractor_objs.append(obj)

            # Compute XY bounding radius from collision geometry
            xy_radius = get_actor_xy_radius(obj)
            self.distractor_radii.append(xy_radius)
            print(f"[Distractor] Loaded: {model_id} (scale={scale:.3f}, {scale_source}, radius={xy_radius:.3f}m)")

        self._distractors_loaded = True
        print(f"[Distractor] Successfully loaded {len(self.distractor_objs)} distractor(s)")

    def _position_distractors(self, rng):
        """Position distractors on table/sink using grid-based placement.

        Creates a grid over the surface, filters out cells that overlap
        with task objects, then places each distractor in a random available cell.

        For eggplant task (sink environment), uses sink basin bounds instead of table.
        """
        base_env = self.env.unwrapped

        # Detect if this is the sink task (eggplant in basket)
        instruction = ""
        if hasattr(base_env, 'get_language_instruction'):
            instruction = base_env.get_language_instruction()
        is_sink_task = "eggplant" in instruction.lower() and "basket" in instruction.lower()

        # Sink basin bounds (for eggplant task) - from collision mesh analysis
        # Basin interior world coords where objects can rest
        SINK_X_MIN, SINK_X_MAX = -0.276, -0.045
        SINK_Y_MIN, SINK_Y_MAX = -0.052, 0.303
        SINK_Z = 0.88  # Basin floor height

        # Table bounds (for other tasks) - from bridge_table_1_v1 collision mesh
        # Actual table surface: X: -0.35 to 0.01, Y: -0.30 to 0.30
        # Robot at X=0.147, keep margin from robot workspace
        TABLE_X_MIN, TABLE_X_MAX = -0.35, -0.02
        TABLE_Y_MIN, TABLE_Y_MAX = -0.28, 0.28
        TABLE_Z = 0.87  # Table height

        # Select bounds based on task
        if is_sink_task:
            X_MIN, X_MAX = SINK_X_MIN, SINK_X_MAX
            Y_MIN, Y_MAX = SINK_Y_MIN, SINK_Y_MAX
            surface_height = SINK_Z
            print(f"[Distractor] Detected SINK task: placing distractors in basin")
        else:
            X_MIN, X_MAX = TABLE_X_MIN, TABLE_X_MAX
            Y_MIN, Y_MAX = TABLE_Y_MIN, TABLE_Y_MAX
            surface_height = TABLE_Z
            print(f"[Distractor] Detected TABLE task: placing distractors on table")

        # Store surface height for use by other methods
        self._surface_height = surface_height
        self._is_sink_task = is_sink_task

        # Safety bubble parameters
        SAFETY_PADDING = 0.0  # No padding - distractors can touch task object bounding boxes
        FALLBACK_RADIUS = 0.06  # 6cm fallback if no bounding box available

        # Get task object positions for safety bubbles (and centroid calculation)
        # For sink task, we skip safety bubbles but still need positions for centroid
        safety_bubbles = []  # List of (x, y, radius)
        task_object_positions = []  # For centroid calculation
        obj_bbox_attrs = {
            'episode_source_obj': 'episode_source_obj_bbox_world',
            'episode_target_obj': 'episode_target_obj_bbox_world',
        }

        for obj_attr, bbox_attr in obj_bbox_attrs.items():
            if hasattr(base_env, obj_attr):
                obj = getattr(base_env, obj_attr)
                if obj is not None:
                    pos = obj.pose.p
                    task_object_positions.append((pos[0], pos[1]))

                    # Skip safety bubbles for sink task (limited space)
                    if is_sink_task:
                        print(f"[Distractor] Task object: {obj_attr} at ({pos[0]:.3f}, {pos[1]:.3f}) - no safety bubble (sink task)")
                        continue

                    if hasattr(base_env, bbox_attr):
                        bbox = getattr(base_env, bbox_attr)
                        if bbox is not None:
                            bbox_radius = np.sqrt(bbox[0]**2 + bbox[1]**2) / 2
                            radius = bbox_radius + SAFETY_PADDING
                        else:
                            radius = FALLBACK_RADIUS
                    else:
                        radius = FALLBACK_RADIUS
                    safety_bubbles.append((pos[0], pos[1], radius))
                    print(f"[Distractor] Safety bubble: {obj_attr} at ({pos[0]:.3f}, {pos[1]:.3f}), r={radius:.3f}m")

        EDGE_MARGIN = 0.02  # Keep object centers 2cm from edges

        # Compute centroid of task objects for grid centering
        if task_object_positions:
            centroid_x = np.mean([p[0] for p in task_object_positions])
            centroid_y = np.mean([p[1] for p in task_object_positions])
        else:
            # Fallback to surface center if no task objects
            centroid_x = (X_MIN + X_MAX) / 2
            centroid_y = (Y_MIN + Y_MAX) / 2
        print(f"[Distractor] Task centroid: ({centroid_x:.3f}, {centroid_y:.3f})")

        # Grid centered around task objects
        # Use smaller grid for sink (limited space)
        if is_sink_task:
            GRID_RADIUS_X = 0.10  # 10cm in each X direction from centroid
            GRID_RADIUS_Y = 0.15  # 15cm in each Y direction from centroid
        else:
            GRID_RADIUS_X = 0.15  # 15cm in each X direction from centroid
            GRID_RADIUS_Y = 0.20  # 20cm in each Y direction from centroid

        grid_x_min = max(X_MIN, centroid_x - GRID_RADIUS_X)
        grid_x_max = min(X_MAX, centroid_x + GRID_RADIUS_X)
        grid_y_min = max(Y_MIN, centroid_y - GRID_RADIUS_Y)
        grid_y_max = min(Y_MAX, centroid_y + GRID_RADIUS_Y)

        print(f"[Distractor] Centered grid: X:[{grid_x_min:.3f}, {grid_x_max:.3f}], Y:[{grid_y_min:.3f}, {grid_y_max:.3f}]")

        # Grid parameters (4x4=16 cells)
        GRID_COLS = 4
        GRID_ROWS = 4
        cell_width = (grid_x_max - grid_x_min - 2 * EDGE_MARGIN) / GRID_COLS
        cell_height = (grid_y_max - grid_y_min - 2 * EDGE_MARGIN) / GRID_ROWS

        print(f"[Distractor] Grid: {GRID_COLS}x{GRID_ROWS}, cell size: {cell_width:.3f}x{cell_height:.3f}m")

        # Build list of grid cells as (center_x, center_y)
        grid_cells = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                cx = grid_x_min + EDGE_MARGIN + (col + 0.5) * cell_width
                cy = grid_y_min + EDGE_MARGIN + (row + 0.5) * cell_height
                grid_cells.append((cx, cy))

        # Filter out cells that overlap with safety bubbles
        available_cells = []
        for cx, cy in grid_cells:
            in_bubble = False
            for bx, by, radius in safety_bubbles:
                # Check if cell center is inside bubble (just use bubble radius, no cell margin)
                dist = np.sqrt((cx - bx)**2 + (cy - by)**2)
                if dist < radius:
                    in_bubble = True
                    break
            if not in_bubble:
                available_cells.append((cx, cy))

        print(f"[Distractor] Available cells: {len(available_cells)}/{len(grid_cells)} (after removing safety zones)")

        # Shuffle available cells for random assignment
        available_cells = list(available_cells)  # Make a copy
        rng.shuffle(available_cells)

        # Place each distractor in a cell
        for i, obj in enumerate(self.distractor_objs):
            my_radius = self.distractor_radii[i]

            if i < len(available_cells):
                # Use a cell - add jitter within cell
                cx, cy = available_cells[i]
                jitter_x = rng.uniform(-cell_width * 0.3, cell_width * 0.3)
                jitter_y = rng.uniform(-cell_height * 0.3, cell_height * 0.3)
                x = cx + jitter_x
                y = cy + jitter_y
                cell_idx = i
            else:
                # No cells left - place in top row (Y+) with spacing, staying within centered grid
                print(f"[Distractor] WARNING: No cell for {obj.name}, using overflow position")
                overflow_idx = i - len(available_cells)
                # Spread overflow objects across X range in Y+ region of centered grid
                x = grid_x_min + EDGE_MARGIN + 0.03 + (overflow_idx % 4) * 0.05
                y = grid_y_max - EDGE_MARGIN - 0.03 - (overflow_idx // 4) * 0.05
                # Clamp to centered grid bounds
                x = max(grid_x_min + EDGE_MARGIN, min(x, grid_x_max - EDGE_MARGIN))
                y = max(grid_y_min + EDGE_MARGIN, min(y, grid_y_max - EDGE_MARGIN))
                cell_idx = -1

            # Spawn above surface (table or sink basin)
            z = surface_height + 0.10

            # Random rotation
            angle = rng.uniform(0, 2 * np.pi)
            quat = [np.cos(angle/2), 0, 0, np.sin(angle/2)]

            obj.set_pose(sapien.Pose([x, y, z], quat))
            print(f"[Distractor] Positioned {obj.name} at ({x:.3f}, {y:.3f}, {z:.3f}), cell={cell_idx}, rot={np.degrees(angle):.0f}°")


        # Return safety bubbles and grid bounds for use by relocation methods
        grid_bounds = (grid_x_min, grid_x_max, grid_y_min, grid_y_max)
        return safety_bubbles, grid_bounds

    def _count_visible_distractors(self, initial_positions=None):
        """Count how many distractors are still on/above the surface after physics.

        Args:
            initial_positions: List of (x, y, z) tuples of initial spawn positions.
                               If provided, also checks for objects stuck to robot.
        """
        # Use stored surface height from _position_distractors, or default to table
        surface_height = getattr(self, '_surface_height', 0.87)
        count = 0
        for i, obj in enumerate(self.distractor_objs):
            pos = obj.pose.p
            x, y, z = pos[0], pos[1], pos[2]

            # Check if fell off surface
            if z < surface_height - 0.1:
                print(f"[Distractor] WARNING: {obj.name} fell off surface (z={z:.3f})")
                continue

            # Check if in robot workspace (x > -0.08 means likely stuck to robot)
            # Table bounds go up to x=-0.09, so use -0.08 as threshold
            if x > -0.08:
                print(f"[Distractor] WARNING: {obj.name} in robot workspace (x={x:.3f})")
                continue

            # Check if object moved significantly from initial spawn position
            # Objects spawn 0.08m above table, so expect ~0.06-0.08m fall
            # Use 0.04m threshold to catch truly stuck objects (didn't fall at all)
            if initial_positions and i < len(initial_positions):
                init_x, init_y, init_z = initial_positions[i]
                if abs(z - init_z) < 0.04:  # Didn't fall much - might be stuck
                    print(f"[Distractor] WARNING: {obj.name} may be stuck (z barely changed: {init_z:.3f} -> {z:.3f})")
                    continue

            count += 1
            print(f"[Distractor] OK: {obj.name} at ({x:.3f}, {y:.3f}, {z:.3f})")

        return count

    def _relocate_bubble_violators(self, safety_bubbles, rng, grid_bounds):
        """Relocate distractors that ended up inside safety bubbles after physics.

        Instead of removing objects, tries to find a new valid position and respawn them.
        Only removes if no valid position can be found after max attempts.
        """
        base_env = self.env.unwrapped
        # Use stored surface height from _position_distractors, or default to table
        surface_height = getattr(self, '_surface_height', 0.87)

        # Use centered grid bounds from _position_distractors
        grid_x_min, grid_x_max, grid_y_min, grid_y_max = grid_bounds

        relocated = []
        removed = []

        # Get current positions of all distractors for spacing check
        def get_other_positions(exclude_obj):
            return [(o.pose.p[0], o.pose.p[1]) for o in self.distractor_objs if o != exclude_obj]

        for obj in self.distractor_objs[:]:  # Copy list to allow removal
            pos = obj.pose.p
            in_bubble = False

            for bx, by, radius in safety_bubbles:
                dist = np.sqrt((pos[0] - bx)**2 + (pos[1] - by)**2)
                if dist < radius:
                    in_bubble = True
                    print(f"[Distractor] {obj.name} inside safety bubble (dist={dist:.3f} < {radius}), relocating...")
                    break

            if not in_bubble:
                continue

            # Try to find a new valid position
            max_attempts = 30
            found_position = False

            for attempt in range(max_attempts):
                # Random position in centered grid area
                new_x = rng.uniform(grid_x_min, grid_x_max)
                new_y = rng.uniform(grid_y_min, grid_y_max)

                # Check against safety bubbles
                valid = True
                for bx, by, radius in safety_bubbles:
                    dist = np.sqrt((new_x - bx)**2 + (new_y - by)**2)
                    if dist < radius:
                        valid = False
                        break

                # Check against other distractors (8cm minimum spacing)
                if valid:
                    for px, py in get_other_positions(obj):
                        dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                        if dist < 0.08:
                            valid = False
                            break

                if valid:
                    # Found valid position - relocate object
                    new_z = surface_height + 0.02  # Slightly above surface
                    obj.set_pose(sapien.Pose([new_x, new_y, new_z], [1, 0, 0, 0]))
                    obj.set_velocity(np.zeros(3))
                    obj.set_angular_velocity(np.zeros(3))
                    print(f"[Distractor] RELOCATED {obj.name} to ({new_x:.3f}, {new_y:.3f}, {new_z:.3f})")
                    relocated.append(obj.name)
                    found_position = True
                    break

            if not found_position:
                # No valid position found - leave in place instead of removing
                print(f"[Distractor] KEEPING {obj.name} in place - no valid position found after {max_attempts} attempts")

        return relocated, removed

    def _relocate_touching_distractors(self, safety_bubbles, rng, grid_bounds, min_dist=0.05):
        """Relocate distractors that are too close to each other after physics settling.

        When two objects are too close, relocates the one that was added later (higher index).
        Only removes if no valid position can be found.
        """
        base_env = self.env.unwrapped
        # Use stored surface height from _position_distractors, or default to table
        surface_height = getattr(self, '_surface_height', 0.87)

        # Use centered grid bounds from _position_distractors
        grid_x_min, grid_x_max, grid_y_min, grid_y_max = grid_bounds

        relocated = []
        removed = []

        # Build list of (obj, position) for remaining distractors
        obj_positions = [(obj, obj.pose.p) for obj in self.distractor_objs]

        # Find objects that need relocation
        to_relocate = []
        for i, (obj_i, pos_i) in enumerate(obj_positions):
            for j, (obj_j, pos_j) in enumerate(obj_positions[i+1:], start=i+1):
                dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                if dist < min_dist:
                    # Relocate the later object (higher index)
                    if obj_j not in to_relocate:
                        print(f"[Distractor] {obj_j.name} too close to {obj_i.name} (dist={dist:.3f} < {min_dist}), relocating...")
                        to_relocate.append(obj_j)

        # Get positions of objects that DON'T need relocation
        def get_stable_positions():
            return [(o.pose.p[0], o.pose.p[1]) for o in self.distractor_objs if o not in to_relocate]

        # Try to relocate each object
        for obj in to_relocate:
            max_attempts = 30
            found_position = False

            for attempt in range(max_attempts):
                new_x = rng.uniform(grid_x_min, grid_x_max)
                new_y = rng.uniform(grid_y_min, grid_y_max)

                # Check against safety bubbles
                valid = True
                for bx, by, radius in safety_bubbles:
                    dist = np.sqrt((new_x - bx)**2 + (new_y - by)**2)
                    if dist < radius:
                        valid = False
                        break

                # Check against stable objects and already-relocated objects
                if valid:
                    for px, py in get_stable_positions():
                        dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                        if dist < 0.08:
                            valid = False
                            break

                # Also check against already relocated objects
                if valid:
                    for rel_name in relocated:
                        rel_obj = next((o for o in self.distractor_objs if o.name == rel_name), None)
                        if rel_obj:
                            rel_pos = rel_obj.pose.p
                            dist = np.sqrt((new_x - rel_pos[0])**2 + (new_y - rel_pos[1])**2)
                            if dist < 0.08:
                                valid = False
                                break

                if valid:
                    new_z = surface_height + 0.02
                    obj.set_pose(sapien.Pose([new_x, new_y, new_z], [1, 0, 0, 0]))
                    obj.set_velocity(np.zeros(3))
                    obj.set_angular_velocity(np.zeros(3))
                    print(f"[Distractor] RELOCATED {obj.name} to ({new_x:.3f}, {new_y:.3f}, {new_z:.3f})")
                    relocated.append(obj.name)
                    found_position = True
                    break

            if not found_position:
                # No valid position found - leave in place instead of removing
                print(f"[Distractor] KEEPING {obj.name} in place - no valid position found after {max_attempts} attempts")

        return relocated, removed

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
        self.distractor_radii = []

        obs, info = self.env.reset(**kwargs)

        # Load distractors into the new scene
        self._load_distractors()

        # Position them randomly (spawns 0.5m above table)
        rng = np.random.RandomState(kwargs.get("options", {}).get("obj_init_options", {}).get("episode_id", 0))
        safety_bubbles, grid_bounds = self._position_distractors(rng)

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

        # Relocate distractors that drifted into safety bubbles during physics
        relocated, removed = self._relocate_bubble_violators(safety_bubbles, rng, grid_bounds)
        if relocated:
            print(f"[Distractor] Relocated {len(relocated)} objects that were in safety bubbles: {relocated}")
        if removed:
            print(f"[Distractor] Removed {len(removed)} objects (no valid position found): {removed}")

        # Note: Not relocating distractors that are close to each other - physics handles this naturally

        # Quick physics settle after any relocations (0.3s)
        if relocated:
            print(f"[Distractor] Settling relocated objects...")
            for _ in range(int(sim_freq * 0.3)):
                base_env._scene.step()

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
