import numpy as np
import sapien.core as sapien
from pathlib import Path


class DistractorWrapper:
    """Wrapper to add distractor objects to SimplerEnv Bridge environments."""

    AVAILABLE_DISTRACTORS = [
        "apple", "orange", "sponge", "blue_plastic_bottle", "eggplant",
        "opened_coke_can", "opened_pepsi_can", "opened_sprite_can",
        "bridge_carrot_generated_modified", "green_cube_3cm", "yellow_cube_3cm",
        "opened_fanta_can", "opened_redbull_can", "opened_7up_can",
    ]

    def __init__(self, env, distractor_ids):
        self.env = env
        self.distractor_ids = distractor_ids
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
                print(f"[Distractor] Warning: {model_id} not in model_db, skipping")
                continue

            density = model_db[model_id].get("density", 1000)
            obj = base_env._build_actor_helper(
                model_id, scene,
                scale=1.0,
                density=density,
                physical_material=scene.create_physical_material(0.5, 0.5, 0.0),
                root_dir=asset_root,
            )
            obj.name = f"distractor_{model_id}"
            self.distractor_objs.append(obj)

        self._distractors_loaded = True

    def _position_distractors(self, rng):
        """Position distractors randomly on table, avoiding main objects."""
        table_height = 0.87  # Bridge table height

        for i, obj in enumerate(self.distractor_objs):
            # Random position on table (avoiding center where main objects are)
            while True:
                x = rng.uniform(-0.35, -0.05)
                y = rng.uniform(-0.15, 0.25)
                # Check not too close to center
                if abs(x + 0.16) > 0.12 or abs(y) > 0.12:
                    break

            z = table_height + 0.05
            quat = [1, 0, 0, 0]  # Upright
            obj.set_pose(sapien.Pose([x, y, z], quat))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Load distractors on first reset
        self._load_distractors()

        # Position them randomly
        rng = np.random.RandomState(kwargs.get("options", {}).get("obj_init_options", {}).get("episode_id", 0))
        self._position_distractors(rng)

        return obs, info

    def step(self, action):
        return self.env.step(action)

    def __getattr__(self, name):
        return getattr(self.env, name)
