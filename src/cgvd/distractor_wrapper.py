from typing import Tuple

import numpy as np
import sapien.core as sapien
from pathlib import Path


def _get_shape_local_pose(shape):
    """Get the local pose of a collision shape, handling SAPIEN API differences."""
    if hasattr(shape, 'local_pose'):
        return shape.local_pose
    if hasattr(shape, 'get_local_pose'):
        return shape.get_local_pose()
    # Identity pose fallback
    return sapien.Pose()


def _transform_points_by_pose(points, pose):
    """Transform points (N,3) by a sapien Pose (rotation + translation)."""
    mat = pose.to_transformation_matrix()
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    return points @ rot.T + trans


def get_actor_xy_radius(actor):
    """Compute XY bounding radius for an actor from its collision shapes.

    Each collision sub-shape may have a local_pose offset (SAPIEN centers
    decomposed convex pieces at their centroid). We must transform vertices
    into actor space before computing the overall XY footprint.

    Returns the radius of the smallest circle (centered at actor origin)
    that contains the actor's XY footprint.
    """
    all_xy_points = []

    for shape in actor.get_collision_shapes():
        geom = shape.geometry
        local_pose = _get_shape_local_pose(shape)

        if isinstance(geom, sapien.ConvexMeshGeometry):
            verts = np.array(geom.vertices) * np.array(geom.scale)
            # Transform to actor space via local_pose
            verts_actor = _transform_points_by_pose(verts, local_pose)
            all_xy_points.append(verts_actor[:, :2])

        elif isinstance(geom, sapien.BoxGeometry):
            half = geom.half_lengths
            corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            corners_actor = _transform_points_by_pose(corners, local_pose)
            all_xy_points.append(corners_actor[:, :2])

        elif isinstance(geom, sapien.SphereGeometry):
            r = geom.radius
            # Sphere center in actor space + radius
            center = _transform_points_by_pose(np.zeros((1, 3)), local_pose)[0]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    all_xy_points.append([center[0] + sx * r, center[1] + sy * r])

        elif isinstance(geom, sapien.CapsuleGeometry):
            r = geom.radius + geom.half_length
            center = _transform_points_by_pose(np.zeros((1, 3)), local_pose)[0]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    all_xy_points.append([center[0] + sx * r, center[1] + sy * r])

        else:
            all_xy_points.append([[-0.05, -0.05], [0.05, 0.05]])

    if not all_xy_points:
        return 0.05  # Fallback

    combined = np.vstack(all_xy_points)
    # Max distance from actor origin to any vertex in XY
    distances = np.sqrt(combined[:, 0]**2 + combined[:, 1]**2)
    return float(distances.max())


def get_actor_z_bounds(actor) -> Tuple[float, float]:
    """Get the min/max Z coordinates of an actor's collision geometry.

    Transforms vertices by each shape's local_pose to get actor-relative
    coordinates, then finds the global Z min/max.

    Returns:
        Tuple of (z_min, z_max) relative to actor origin
    """
    all_z = []

    for shape in actor.get_collision_shapes():
        geom = shape.geometry
        local_pose = _get_shape_local_pose(shape)

        if isinstance(geom, sapien.ConvexMeshGeometry):
            verts = np.array(geom.vertices) * np.array(geom.scale)
            verts_actor = _transform_points_by_pose(verts, local_pose)
            all_z.append(verts_actor[:, 2])

        elif isinstance(geom, sapien.BoxGeometry):
            half = geom.half_lengths
            corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            corners_actor = _transform_points_by_pose(corners, local_pose)
            all_z.append(corners_actor[:, 2])

        elif isinstance(geom, sapien.SphereGeometry):
            center = _transform_points_by_pose(np.zeros((1, 3)), local_pose)[0]
            r = geom.radius
            all_z.append(np.array([center[2] - r, center[2] + r]))

        elif isinstance(geom, sapien.CapsuleGeometry):
            center = _transform_points_by_pose(np.zeros((1, 3)), local_pose)[0]
            r = geom.radius + geom.half_length
            all_z.append(np.array([center[2] - r, center[2] + r]))

    if not all_z:
        return (-0.05, 0.05)  # Fallback
    combined = np.concatenate(all_z)
    return (float(combined.min()), float(combined.max()))


def get_actor_all_vertices(actor):
    """Collect all collision geometry vertices into a single (N, 3) array in actor-local space.

    Iterates collision shapes, transforms by each shape's local_pose, and
    concatenates into one array.  Used for AABB analysis (e.g. elongation
    detection) before any world-space pose is applied.
    """
    all_verts = []

    for shape in actor.get_collision_shapes():
        geom = shape.geometry
        local_pose = _get_shape_local_pose(shape)

        if isinstance(geom, sapien.ConvexMeshGeometry):
            verts = np.array(geom.vertices) * np.array(geom.scale)
            verts_actor = _transform_points_by_pose(verts, local_pose)
            all_verts.append(verts_actor)

        elif isinstance(geom, sapien.BoxGeometry):
            half = geom.half_lengths
            corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            corners_actor = _transform_points_by_pose(corners, local_pose)
            all_verts.append(corners_actor)

        elif isinstance(geom, sapien.SphereGeometry):
            r = geom.radius
            center = _transform_points_by_pose(np.zeros((1, 3)), local_pose)[0]
            # Approximate sphere with 6 axis-aligned extremes
            extremes = np.array([
                [center[0] + r, center[1], center[2]],
                [center[0] - r, center[1], center[2]],
                [center[0], center[1] + r, center[2]],
                [center[0], center[1] - r, center[2]],
                [center[0], center[1], center[2] + r],
                [center[0], center[1], center[2] - r],
            ])
            all_verts.append(extremes)

        elif isinstance(geom, sapien.CapsuleGeometry):
            r = geom.radius + geom.half_length
            center = _transform_points_by_pose(np.zeros((1, 3)), local_pose)[0]
            extremes = np.array([
                [center[0] + r, center[1], center[2]],
                [center[0] - r, center[1], center[2]],
                [center[0], center[1] + r, center[2]],
                [center[0], center[1] - r, center[2]],
                [center[0], center[1], center[2] + r],
                [center[0], center[1], center[2] - r],
            ])
            all_verts.append(extremes)

    if not all_verts:
        return np.zeros((0, 3))
    return np.vstack(all_verts)


def compute_lay_flat_quaternion(verts):
    """Return a quaternion (WXYZ) that lays an elongated object flat.

    Computes AABB extents from *verts* (N, 3).  If the Z extent exceeds
    1.5x the larger of X/Y extents the object is considered "standing" and
    a 90-degree rotation around the Y axis is returned (maps local Z → -X,
    laying the long axis horizontal).

    Otherwise returns identity (no correction needed).
    """
    if len(verts) == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])

    dx = verts[:, 0].max() - verts[:, 0].min()
    dy = verts[:, 1].max() - verts[:, 1].min()
    dz = verts[:, 2].max() - verts[:, 2].min()

    if dz > 1.5 * max(dx, dy):
        # 90° around Y: cos(45°) ≈ 0.7071068, sin(45°) ≈ 0.7071068
        return np.array([0.7071068, 0.0, 0.7071068, 0.0])

    return np.array([1.0, 0.0, 0.0, 0.0])


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

    def _log(self, msg):
        """Print and buffer a log message for later file output."""
        print(msg)
        self._log_lines.append(msg)

    def __init__(self, env, distractor_ids, distractor_scale=None, external_asset_scale=None,
                 num_distractors=None, randomize_per_episode=False):
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
            num_distractors: Number of distractors to sample per episode when randomize_per_episode=True.
                            If None or >= len(pool), uses all distractors.
            randomize_per_episode: If True, randomly sample num_distractors from pool each episode.
                            If False, uses all distractors (or first num_distractors if specified).
        """
        self.env = env
        # Parse distractor_ids for per-object scales (format: "object_id:scale")
        self._all_distractor_ids = []  # Full pool for randomization
        self.per_object_scales = {}  # object_id -> scale
        for item in distractor_ids:
            if ":" in item:
                obj_id, scale_str = item.rsplit(":", 1)
                try:
                    self.per_object_scales[obj_id] = float(scale_str)
                    self._all_distractor_ids.append(obj_id)
                except ValueError:
                    # Not a valid scale, treat whole thing as object ID
                    self._all_distractor_ids.append(item)
            else:
                self._all_distractor_ids.append(item)

        # Store pool for randomization
        self.distractor_pool = self._all_distractor_ids.copy()
        self.num_distractors = num_distractors
        self.randomize_per_episode = randomize_per_episode

        # If not randomizing, apply num_distractors limit now (take first N)
        if not randomize_per_episode and num_distractors and num_distractors < len(self._all_distractor_ids):
            self.distractor_ids = self._all_distractor_ids[:num_distractors]
        else:
            self.distractor_ids = self._all_distractor_ids.copy()

        self.distractor_scale = distractor_scale
        self.external_asset_scale = external_asset_scale if external_asset_scale is not None else self.DEFAULT_EXTERNAL_ASSET_SCALE
        self.distractor_objs = []
        self.distractor_radii = []  # XY bounding radius for each distractor
        self.distractor_z_bounds = []  # Z bounds (z_min, z_max) for each distractor
        self.distractor_base_quats = []  # Lay-flat quaternion per distractor
        self.distractor_spawn_quats = []  # Final spawn quaternion (lay-flat + random yaw)
        self._distractors_loaded = False
        self._log_lines = []  # Buffered log lines, flushed to file after each reset

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
                self._log(f"[Distractor] Warning: '{model_id}' not in model_db, skipping")
                self._log(f"[Distractor] Available objects: {list(model_db.keys())}")
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

            # Compute lay-flat orientation for elongated objects
            verts = get_actor_all_vertices(obj)
            base_q = compute_lay_flat_quaternion(verts)
            self.distractor_base_quats.append(base_q)

            # Recompute bounds with the lay-flat rotation applied
            rot_mat = sapien.Pose(q=base_q).to_transformation_matrix()[:3, :3]
            rot_verts = verts @ rot_mat.T if len(verts) > 0 else verts
            if len(rot_verts) > 0:
                xy_dists = np.sqrt(rot_verts[:, 0]**2 + rot_verts[:, 1]**2)
                xy_radius = float(xy_dists.max())
                z_min = float(rot_verts[:, 2].min())
                z_max = float(rot_verts[:, 2].max())
            else:
                xy_radius = 0.05
                z_min, z_max = -0.05, 0.05
            self.distractor_radii.append(xy_radius)
            self.distractor_z_bounds.append((z_min, z_max))

            is_rotated = not np.allclose(base_q, [1, 0, 0, 0])
            self._log(f"[Distractor] Loaded: {model_id} (scale={scale:.3f}, {scale_source}, radius={xy_radius:.3f}m, z_bounds=({z_min:.3f}, {z_max:.3f}), lay_flat={'YES' if is_rotated else 'no'})")

        self._distractors_loaded = True
        self._log(f"[Distractor] Successfully loaded {len(self.distractor_objs)} distractor(s)")

    def _position_distractors(self, rng):
        """Position distractors on table/sink using grid-based placement.

        Divides the placement area into a fixed grid of 6cm x 6cm cells.
        One object per cell guarantees no overlap regardless of radius accuracy.
        Cells overlapping task-object safety bubbles are excluded.

        For eggplant task (sink environment), uses sink basin bounds instead of table.
        """
        base_env = self.env.unwrapped

        # Detect if this is the sink task (eggplant in basket)
        instruction = ""
        if hasattr(base_env, 'get_language_instruction'):
            instruction = base_env.get_language_instruction()
        is_sink_task = "eggplant" in instruction.lower() and "basket" in instruction.lower()

        # Sink basin bounds (for eggplant task) - from collision mesh analysis
        SINK_MARGIN_LEFT = 0.02
        SINK_MARGIN_RIGHT = 0.01
        SINK_MARGIN_Y = 0.02
        SINK_X_MIN, SINK_X_MAX = -0.276 + SINK_MARGIN_LEFT, -0.045 - SINK_MARGIN_RIGHT
        SINK_Y_MIN, SINK_Y_MAX = -0.052 + SINK_MARGIN_Y, 0.303 - SINK_MARGIN_Y
        SINK_Z = 0.88

        # Table bounds - use FULL table area for placement
        # Table surface: X: -0.35 to 0.01, Y: -0.30 to 0.30
        # Keep 3cm from edges, and stay away from robot (X > -0.05)
        TABLE_EDGE_BUFFER = 0.03
        TABLE_X_MIN, TABLE_X_MAX = -0.35 + TABLE_EDGE_BUFFER, -0.05  # Stay away from robot
        TABLE_Y_MIN, TABLE_Y_MAX = -0.28 + TABLE_EDGE_BUFFER, 0.28 - TABLE_EDGE_BUFFER
        TABLE_Z = 0.87

        # Select bounds based on task
        if is_sink_task:
            X_MIN, X_MAX = SINK_X_MIN, SINK_X_MAX
            Y_MIN, Y_MAX = SINK_Y_MIN, SINK_Y_MAX
            surface_height = SINK_Z
            self._log(f"[Distractor] Detected SINK task: placing distractors in basin")
        else:
            X_MIN, X_MAX = TABLE_X_MIN, TABLE_X_MAX
            Y_MIN, Y_MAX = TABLE_Y_MIN, TABLE_Y_MAX
            surface_height = TABLE_Z
            self._log(f"[Distractor] Detected TABLE task: placing distractors on table")
            self._log(f"[Distractor] Placement area: X:[{X_MIN:.3f}, {X_MAX:.3f}], Y:[{Y_MIN:.3f}, {Y_MAX:.3f}]")

        # Store surface height for use by other methods
        self._surface_height = surface_height
        self._is_sink_task = is_sink_task

        # Safety bubble parameters
        # Keep padding around task objects so distractors don't block them
        SAFETY_PADDING = 0.02  # 2cm padding (grid cells provide inherent 3cm buffer + XY-locked settling)
        FALLBACK_RADIUS = 0.08

        # Get task object positions for safety bubbles
        safety_bubbles = []  # List of (x, y, radius)
        obj_bbox_attrs = {
            'episode_source_obj': 'episode_source_obj_bbox_world',
            'episode_target_obj': 'episode_target_obj_bbox_world',
        }

        for obj_attr, bbox_attr in obj_bbox_attrs.items():
            if hasattr(base_env, obj_attr):
                obj = getattr(base_env, obj_attr)
                if obj is not None:
                    pos = obj.pose.p

                    # For sink task: skip safety bubble for source (eggplant)
                    if is_sink_task and obj_attr == 'episode_source_obj':
                        self._log(f"[Distractor] Task object: {obj_attr} at ({pos[0]:.3f}, {pos[1]:.3f}) - no safety bubble")
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

                    if is_sink_task and obj_attr == 'episode_target_obj':
                        radius = max(radius, 0.08)

                    safety_bubbles.append((pos[0], pos[1], radius))
                    self._log(f"[Distractor] Safety bubble: {obj_attr} at ({pos[0]:.3f}, {pos[1]:.3f}), r={radius:.3f}m")

        # --- Grid-based placement ---
        # Fixed grid over placement area: one object per cell guarantees no overlap
        CELL_SIZE = 0.06  # 6cm x 6cm cells (fits largest utensil ~4cm with 2cm margin)
        JITTER = 0.01     # Up to 1cm random offset within cell for natural appearance

        area_w = X_MAX - X_MIN
        area_h = Y_MAX - Y_MIN
        n_cols = max(1, int(area_w / CELL_SIZE))
        n_rows = max(1, int(area_h / CELL_SIZE))

        # Center grid within placement area (absorb leftover margin evenly)
        margin_x = (area_w - n_cols * CELL_SIZE) / 2
        margin_y = (area_h - n_rows * CELL_SIZE) / 2
        grid_x0 = X_MIN + margin_x
        grid_y0 = Y_MIN + margin_y

        self._log(f"[Distractor] Grid: {n_cols}x{n_rows} = {n_cols * n_rows} cells "
              f"(cell={CELL_SIZE*100:.0f}cm, area={area_w:.3f}x{area_h:.3f}m)")

        # Build list of cell centers
        all_cells = []
        for col in range(n_cols):
            for row in range(n_rows):
                cx = grid_x0 + (col + 0.5) * CELL_SIZE
                cy = grid_y0 + (row + 0.5) * CELL_SIZE
                all_cells.append((cx, cy))

        # Mark cells that overlap safety bubbles as unavailable (circle-AABB test)
        available_cells = []
        for cx, cy in all_cells:
            cell_x_min = cx - CELL_SIZE / 2
            cell_x_max = cx + CELL_SIZE / 2
            cell_y_min = cy - CELL_SIZE / 2
            cell_y_max = cy + CELL_SIZE / 2

            blocked = False
            for bx, by, bradius in safety_bubbles:
                # Closest point on AABB to circle center
                nearest_x = np.clip(bx, cell_x_min, cell_x_max)
                nearest_y = np.clip(by, cell_y_min, cell_y_max)
                dist = np.sqrt((bx - nearest_x)**2 + (by - nearest_y)**2)
                if dist < bradius:
                    blocked = True
                    break

            if not blocked:
                available_cells.append((cx, cy))

        self._log(f"[Distractor] Available cells: {len(available_cells)}/{len(all_cells)}")

        # Per-distractor radius-aware maximin assignment
        # Sort distractors by radius (largest first) so big objects get first
        # pick of the farthest cells — prevents large objects (plate, eggplant)
        # from being assigned to cells where their edge overlaps task objects.
        placement_order = sorted(range(len(self.distractor_objs)),
                                 key=lambda i: self.distractor_radii[i], reverse=True)

        assigned = {}  # obj_idx -> (cx, cy) or None
        remaining = list(available_cells)

        for obj_idx in placement_order:
            r = self.distractor_radii[obj_idx]

            # Filter cells safe for this distractor's size:
            # distance from cell center to bubble center >= bubble_radius + distractor_radius
            valid = [i for i, (cx, cy) in enumerate(remaining)
                     if all(np.sqrt((cx - bx)**2 + (cy - by)**2) >= br + r
                            for bx, by, br in safety_bubbles)]

            if not valid:
                # Fallback: try any remaining cell (ignore distractor radius,
                # just keep cell center outside bubble). This allows the
                # distractor's edge to encroach slightly but keeps its center away.
                if remaining:
                    chosen = rng.randint(len(remaining))
                    assigned[obj_idx] = remaining.pop(chosen)
                    self._log(f"[Distractor] {self.distractor_objs[obj_idx].name} (r={r:.3f}) — relaxed placement (no radius-safe cell)")
                    continue
                # Truly no cells left
                assigned[obj_idx] = None
                continue

            prev_cells = [c for c in assigned.values() if c is not None]
            if not prev_cells:
                # First distractor: random pick from valid cells
                chosen = valid[rng.randint(len(valid))]
            else:
                # Maximin: maximize min-distance to already-assigned cells
                best_d, best = -1.0, []
                for i in valid:
                    cx, cy = remaining[i]
                    min_d = min(np.sqrt((cx - px)**2 + (cy - py)**2)
                               for px, py in prev_cells)
                    if min_d > best_d + 1e-9:
                        best_d, best = min_d, [i]
                    elif abs(min_d - best_d) < 1e-9:
                        best.append(i)
                chosen = best[rng.randint(len(best))]

            assigned[obj_idx] = remaining.pop(chosen)

        # Compute spawn quaternions: base lay-flat + random yaw for visual variety
        self.distractor_spawn_quats = []
        for obj_idx in range(len(self.distractor_objs)):
            yaw = rng.uniform(0, 2 * np.pi)
            yaw_q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])  # WXYZ
            base_q = self.distractor_base_quats[obj_idx]
            spawn_q = (sapien.Pose(q=yaw_q) * sapien.Pose(q=base_q)).q
            self.distractor_spawn_quats.append(spawn_q)

        # Place each distractor at its assigned cell
        for place_idx, (obj_idx, obj) in enumerate(zip(range(len(self.distractor_objs)), self.distractor_objs)):
            z_min, z_max = self.distractor_z_bounds[obj_idx]
            cell = assigned.get(obj_idx)

            if cell is not None:
                cx, cy = cell
                x = cx + rng.uniform(-JITTER, JITTER)
                y = cy + rng.uniform(-JITTER, JITTER)
            else:
                # OVERFLOW — no cells left at all; hide off-scene
                obj.set_pose(sapien.Pose([0, 0, -5], [1, 0, 0, 0]))
                self._log(f"[Distractor] OVERFLOW: {obj.name} (r={self.distractor_radii[obj_idx]:.3f}) — hidden off-scene")
                continue  # skip the normal pose-set below

            # Compute spawn height using mesh Z bounds
            if is_sink_task:
                # For sink: stagger heights to prevent mid-air collisions during falling
                stagger_offset = place_idx * 0.03  # 3cm higher for each subsequent object
                z = surface_height + 0.15 - z_min + stagger_offset
            else:
                # For table: 2cm clearance above table surface
                z = surface_height + 0.02 - z_min

            obj.set_pose(sapien.Pose([x, y, z], self.distractor_spawn_quats[obj_idx]))
            self._log(f"[Distractor] Positioned {obj.name} at ({x:.3f}, {y:.3f}, {z:.3f}), r={self.distractor_radii[obj_idx]:.3f}")

        # Return safety bubbles and placement bounds for use by relocation methods
        grid_bounds = (X_MIN, X_MAX, Y_MIN, Y_MAX)
        return safety_bubbles, grid_bounds

    def _fix_clipped_objects(self):
        """Fix objects that clipped through the surface during physics settling.

        Small objects can penetrate collision meshes. This repositions any object
        that ended up below the surface height.
        """
        surface_height = getattr(self, '_surface_height', 0.87)
        fixed_count = 0

        for obj in self.distractor_objs:
            pos = obj.pose.p
            if pos[2] < surface_height:
                # Object clipped into/through the surface - reposition it
                new_z = surface_height + 0.05  # 5cm above surface
                obj.set_pose(sapien.Pose([pos[0], pos[1], new_z], obj.pose.q))
                obj.set_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))
                self._log(f"[Distractor] FIXED: {obj.name} clipped through surface (z={pos[2]:.3f} -> {new_z:.3f})")
                fixed_count += 1

        return fixed_count

    def _count_visible_distractors(self, initial_positions=None):
        """Count how many distractors are still on/above the surface after physics.

        Simply checks if objects are within the table/sink bounds and above surface.
        Objects placed off-table (due to no valid position) will be outside bounds.
        """
        surface_height = getattr(self, '_surface_height', 0.87)
        is_sink_task = getattr(self, '_is_sink_task', False)

        # Define valid placement bounds (same as in _position_distractors)
        if is_sink_task:
            X_MIN, X_MAX = -0.256, -0.055
            Y_MIN, Y_MAX = -0.032, 0.283
        else:
            X_MIN, X_MAX = -0.35, -0.04  # Table bounds with margin
            Y_MIN, Y_MAX = -0.30, 0.30

        count = 0
        for obj in self.distractor_objs:
            pos = obj.pose.p
            x, y, z = pos[0], pos[1], pos[2]

            # Check if within XY bounds (objects placed off-table have x ~ -0.82)
            if x < X_MIN or x > X_MAX or y < Y_MIN or y > Y_MAX:
                self._log(f"[Distractor] OFF-TABLE: {obj.name} at ({x:.3f}, {y:.3f}, {z:.3f})")
                continue

            # Check if fell below surface
            if z < surface_height - 0.15:
                self._log(f"[Distractor] FELL: {obj.name} at ({x:.3f}, {y:.3f}, {z:.3f})")
                continue

            count += 1
            self._log(f"[Distractor] ON-TABLE: {obj.name} at ({x:.3f}, {y:.3f}, {z:.3f})")

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
                    self._log(f"[Distractor] {obj.name} inside safety bubble (dist={dist:.3f} < {radius}), relocating...")
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
                    obj_idx = self.distractor_objs.index(obj)
                    obj.set_pose(sapien.Pose([new_x, new_y, new_z], self.distractor_spawn_quats[obj_idx]))
                    obj.set_velocity(np.zeros(3))
                    obj.set_angular_velocity(np.zeros(3))
                    self._log(f"[Distractor] RELOCATED {obj.name} to ({new_x:.3f}, {new_y:.3f}, {new_z:.3f})")
                    relocated.append(obj.name)
                    found_position = True
                    break

            if not found_position:
                # No valid position found - leave in place instead of removing
                self._log(f"[Distractor] KEEPING {obj.name} in place - no valid position found after {max_attempts} attempts")

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
                        self._log(f"[Distractor] {obj_j.name} too close to {obj_i.name} (dist={dist:.3f} < {min_dist}), relocating...")
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
                    obj_idx = self.distractor_objs.index(obj)
                    obj.set_pose(sapien.Pose([new_x, new_y, new_z], self.distractor_spawn_quats[obj_idx]))
                    obj.set_velocity(np.zeros(3))
                    obj.set_angular_velocity(np.zeros(3))
                    self._log(f"[Distractor] RELOCATED {obj.name} to ({new_x:.3f}, {new_y:.3f}, {new_z:.3f})")
                    relocated.append(obj.name)
                    found_position = True
                    break

            if not found_position:
                # No valid position found - leave in place instead of removing
                self._log(f"[Distractor] KEEPING {obj.name} in place - no valid position found after {max_attempts} attempts")

        return relocated, removed

    def reset(self, **kwargs):
        # Clear log buffer for this episode
        self._log_lines = []

        # Remove old distractor actors from scene (scene persists across resets)
        base_env = self.env.unwrapped
        if hasattr(base_env, '_scene') and base_env._scene is not None:
            for obj in self.distractor_objs:
                try:
                    base_env._scene.remove_actor(obj)
                    self._log(f"[Distractor] Removed old actor: {obj.name}")
                except Exception as e:
                    self._log(f"[Distractor] Could not remove {obj.name}: {e}")

        # Reset state
        self._distractors_loaded = False
        self.distractor_objs = []
        self.distractor_radii = []
        self.distractor_z_bounds = []
        self.distractor_base_quats = []
        self.distractor_spawn_quats = []

        # Sample distractors for this episode if randomization is enabled
        if self.randomize_per_episode and self.num_distractors is not None:
            episode_id = kwargs.get("options", {}).get("obj_init_options", {}).get("episode_id", 0)
            rng = np.random.RandomState(episode_id)
            n_sample = min(self.num_distractors, len(self.distractor_pool))

            # Sample without replacement
            indices = rng.choice(len(self.distractor_pool), size=n_sample, replace=False)
            self.distractor_ids = [self.distractor_pool[i] for i in indices]
            self._log(f"[Distractor] Episode {episode_id}: sampled {self.distractor_ids}")

        obs, info = self.env.reset(**kwargs)

        # Load distractors into the new scene
        self._load_distractors()

        # Position them randomly (spawns 0.5m above table)
        rng = np.random.RandomState(kwargs.get("options", {}).get("obj_init_options", {}).get("episode_id", 0))
        safety_bubbles, grid_bounds = self._position_distractors(rng)

        # Lock task objects so distractor physics can't push them
        task_objs_locked = []
        for attr in ['episode_source_obj', 'episode_target_obj']:
            if hasattr(base_env, attr):
                obj = getattr(base_env, attr)
                if obj is not None:
                    obj.lock_motion(1, 1, 1, 1, 1, 1)  # Lock all 6 DOF
                    task_objs_locked.append(obj)
                    pos = obj.pose.p
                    self._log(f"[Distractor] Locked {attr} at ({pos[0]:.3f}, {pos[1]:.3f})")

        # Record initial positions before physics
        initial_positions = []
        for obj in self.distractor_objs:
            pos = obj.pose.p
            initial_positions.append((pos[0], pos[1], pos[2]))

        # Let distractors settle with physics (matching SimplerEnv's multi-phase approach)
        base_env = self.env.unwrapped
        sim_freq = getattr(base_env, 'sim_freq', 500)

        # Settling times differ by task: longer for sink (staggered drops), shorter for table
        # Use 5 seconds total settling to ensure objects are completely still
        is_sink_task = getattr(self, '_is_sink_task', False)
        settle_phase1 = 2.0 if is_sink_task else 2.5
        settle_phase2 = 2.0 if is_sink_task else 2.5

        # Phase 1: Lock XY translation + XY rotation so objects fall straight down
        # to their grid cells without pushing each other sideways off the table
        for obj in self.distractor_objs:
            obj.lock_motion(1, 1, 0, 1, 1, 0)  # lock XY translation + XY rotation (fall straight down)
        for _ in range(int(sim_freq * settle_phase1)):
            base_env._scene.step()

        # Phase 2: Unlock, reset velocities, settle more
        for obj in self.distractor_objs:
            obj.lock_motion(0, 0, 0, 0, 0, 0)  # unlock all
            obj.set_pose(obj.pose)  # explicit set to prevent sleep
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        for _ in range(int(sim_freq * settle_phase2)):
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
            self._log(f"[Distractor] Relocated {len(relocated)} objects that were in safety bubbles: {relocated}")
        if removed:
            self._log(f"[Distractor] Removed {len(removed)} objects (no valid position found): {removed}")

        # Note: Not relocating distractors that are close to each other - physics handles this naturally

        # Quick physics settle after any relocations (0.3s)
        if relocated:
            self._log(f"[Distractor] Settling relocated objects...")
            for _ in range(int(sim_freq * 0.3)):
                base_env._scene.step()

        # Fix any objects that clipped through the surface (common with small objects)
        fixed_count = self._fix_clipped_objects()
        if fixed_count > 0:
            self._log(f"[Distractor] Fixed {fixed_count} objects that clipped through surface")
            # Brief settle after fixing
            for _ in range(int(sim_freq * 0.1)):
                base_env._scene.step()

        # Log how many distractors are visible after settling
        visible_count = self._count_visible_distractors(initial_positions)
        self._log(f"[Distractor] After settling: {visible_count}/{len(self.distractor_objs)} distractors on table")

        # Unlock task objects now that distractors have settled
        for obj in task_objs_locked:
            obj.lock_motion(0, 0, 0, 0, 0, 0)

        # Verify task objects haven't moved (belt-and-suspenders sanity check)
        for attr in ['episode_source_obj', 'episode_target_obj']:
            if hasattr(base_env, attr):
                obj = getattr(base_env, attr)
                if obj is not None:
                    pos = obj.pose.p
                    self._log(f"[Distractor] Task object {attr} final pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        # Final settle: ensure all distractors are at rest before locking.
        # _fix_clipped_objects places objects 5cm above surface, and
        # _relocate_bubble_violators teleports them — both need time to land.
        for _ in range(int(sim_freq * 3.0)):
            base_env._scene.step()

        # Zero residual velocities and lock all 6 DOF for the entire episode
        for obj in self.distractor_objs:
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
            obj.lock_motion(1, 1, 1, 1, 1, 1)

        # Write placement log to file for debugging
        log_dir = Path("cgvd_debug")
        log_dir.mkdir(exist_ok=True)
        episode_id = kwargs.get("options", {}).get("obj_init_options", {}).get("episode_id", 0)
        log_path = log_dir / f"distractor_placement_ep{episode_id}.log"
        log_path.write_text("\n".join(self._log_lines) + "\n", encoding="utf-8")

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
