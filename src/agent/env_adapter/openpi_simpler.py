"""
OpenPI-specific environment adapters for SimplerEnv.

These adapters handle preprocessing (obs -> model input) and postprocessing
(model output -> env action) for OpenPI Pi0 models on Bridge and Fractal tasks.

OpenPI uses a cross-embodiment format that differs from task-specific models.
The adapters here convert SimplerEnv observations to OpenPI format and
OpenPI actions back to SimplerEnv's expected action format.
"""

import json
from typing import Tuple

import cv2
import numpy as np
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from src.agent.env_adapter.base import BaseEnvAdapter
from src.utils.geometry import euler2axangle, mat2euler, quat2mat


class OpenPIBridgeSimplerAdapter(BaseEnvAdapter):
    """
    OpenPI adapter for Bridge/WidowX tasks in SimplerEnv.

    OpenPI expects DROID-like observation format:
        - observation/exterior_image_1_left: [H, W, 3] uint8
        - observation/wrist_image_left: [H, W, 3] uint8 (optional)
        - observation/joint_position: [6] or [7] float
        - observation/gripper_position: [1] float
        - prompt: str

    Actions from OpenPI: [horizon, 8] (xyz + euler + gripper + terminate)
    SimplerEnv expects: [7] (xyz + axis-angle + gripper)
    """

    def __init__(
        self,
        dataset_statistics_path: str = None,
        image_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.image_size = tuple(image_size)

        # Load normalization statistics if provided
        self.dataset_statistics = None
        if dataset_statistics_path:
            with open(dataset_statistics_path, "r") as f:
                self.dataset_statistics = json.load(f)

        # EE pose in Bridge data was relative to a top-down pose
        self.default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )

    def reset(self):
        """Reset adapter state between episodes."""
        pass

    def preprocess(
        self,
        env,
        obs: dict,
        instruction: str,
    ) -> dict:
        """
        Preprocess observation for OpenPI.

        Returns dict with:
            - image: [H, W, 3] uint8 RGB (resized to 224x224)
            - state: [7] float32 proprioception (xyz + euler + gripper)
            - instruction: str
        """
        # Get and resize image
        image = get_image_from_maniskill2_obs_dict(env, obs)  # [H, W, 3]
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Get raw proprioception
        raw_proprio = self._get_raw_proprio(obs)

        return {
            "image": image.astype(np.uint8),
            "state": raw_proprio.astype(np.float32),
            "instruction": instruction,
        }

    def _get_raw_proprio(self, obs: dict) -> np.ndarray:
        """Extract raw proprioception from observation."""
        proprio = obs["agent"]["eef_pos"]
        # Convert ee rotation to the frame of top-down
        rm_bridge = quat2mat(proprio[3:7])
        rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7]

        return np.concatenate([
            proprio[:3],
            rpy_bridge_converted,
            [gripper_openness],
        ])

    def postprocess(
        self,
        actions: np.ndarray,
    ) -> np.ndarray:
        """
        Postprocess OpenPI actions for SimplerEnv.

        Args:
            actions: [horizon, D] where D is typically 7 or 8

        Returns:
            env_actions: [horizon, 7] for SimplerEnv (xyz + axis-angle + gripper)
        """
        # Handle different action dimensions from OpenPI
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        # OpenPI may output [xyz, euler, gripper, terminate] or just [xyz, euler, gripper]
        # We need to extract the relevant 7 dimensions
        action_dim = actions.shape[-1]

        env_actions = np.zeros((len(actions), 7))
        for idx, raw_action in enumerate(actions):
            # Position deltas
            xyz = raw_action[:3]

            # Rotation (euler) - convert to axis-angle
            if action_dim >= 6:
                roll, pitch, yaw = raw_action[3:6]
                action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            else:
                # No rotation info, assume zero rotation
                action_rotation_ax = np.array([0, 0, 1])
                action_rotation_angle = 0.0

            # Gripper action
            if action_dim >= 7:
                gripper = raw_action[6]
                # OpenPI may use different gripper conventions
                # Assume [-1, 1] range, convert to SimplerEnv convention
                if gripper > 0.5:
                    action_gripper = 1.0  # open
                elif gripper < -0.5:
                    action_gripper = -1.0  # close
                else:
                    # Threshold for binary gripper
                    action_gripper = 1.0 if gripper > 0 else -1.0
            else:
                action_gripper = 0.0  # neutral

            env_actions[idx] = np.concatenate([
                xyz,
                action_rotation_ax * action_rotation_angle,
                [action_gripper],
            ])

        return env_actions

    def get_video_frame(self, env, obs: dict) -> np.ndarray:
        """Get frame for video recording."""
        return get_image_from_maniskill2_obs_dict(env, obs)


class OpenPIFractalSimplerAdapter(BaseEnvAdapter):
    """
    OpenPI adapter for Fractal/Google Robot tasks in SimplerEnv.

    Similar to Bridge adapter but handles Fractal-specific proprioception format
    which uses quaternions instead of euler angles.

    Proprioception: 8D [xyz + quat (xyzw) + gripper_closedness]
    Action: 7D [xyz + euler + gripper] -> converted to axis-angle
    """

    def __init__(
        self,
        dataset_statistics_path: str = None,
        image_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.image_size = tuple(image_size)

        # Load normalization statistics if provided
        self.dataset_statistics = None
        if dataset_statistics_path:
            with open(dataset_statistics_path, "r") as f:
                self.dataset_statistics = json.load(f)

        # Sticky gripper state
        self.sticky_gripper_num_repeat = 15

    def reset(self):
        """Reset adapter state between episodes."""
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0

    def preprocess(
        self,
        env,
        obs: dict,
        instruction: str,
    ) -> dict:
        """
        Preprocess observation for OpenPI.

        Returns dict with:
            - image: [H, W, 3] uint8 RGB
            - state: [8] float32 proprioception (xyz + quat + gripper)
            - instruction: str
        """
        # Get and resize image
        image = get_image_from_maniskill2_obs_dict(env, obs)  # [H, W, 3]
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Get raw proprioception
        raw_proprio = self._get_raw_proprio(obs)

        return {
            "image": image.astype(np.uint8),
            "state": raw_proprio.astype(np.float32),
            "instruction": instruction,
        }

    def _get_raw_proprio(self, obs: dict) -> np.ndarray:
        """Extract raw proprioception from observation."""
        # Convert wxyz quat from simpler to xyzw used in fractal
        quat_xyzw = np.roll(obs["agent"]["eef_pos"][3:7], -1)
        gripper_width = obs["agent"]["eef_pos"][7]  # 0=close, 1=open
        gripper_closedness = 1 - gripper_width  # Fractal uses closedness

        return np.concatenate([
            obs["agent"]["eef_pos"][:3],
            quat_xyzw,
            [gripper_closedness],
        ])

    def postprocess(
        self,
        actions: np.ndarray,
    ) -> np.ndarray:
        """
        Postprocess OpenPI actions for SimplerEnv.

        Args:
            actions: [horizon, D] where D is typically 7 or 8

        Returns:
            env_actions: [horizon, 7] for SimplerEnv (xyz + axis-angle + gripper)
        """
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        action_dim = actions.shape[-1]
        env_actions = np.zeros((len(actions), 7))

        for idx, raw_action in enumerate(actions):
            # Position deltas
            xyz = raw_action[:3]

            # Rotation (euler) - convert to axis-angle
            if action_dim >= 6:
                roll, pitch, yaw = raw_action[3:6]
                action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            else:
                action_rotation_ax = np.array([0, 0, 1])
                action_rotation_angle = 0.0

            # Process gripper with sticky mechanism
            if action_dim >= 7:
                gripper = raw_action[6]
                action_gripper = self._postprocess_gripper(gripper)
            else:
                action_gripper = 0.0

            env_actions[idx] = np.concatenate([
                xyz,
                action_rotation_ax * action_rotation_angle,
                [action_gripper],
            ])

        return env_actions

    def _postprocess_gripper(self, action: float) -> float:
        """
        Process gripper action with sticky mechanism.

        OpenPI may output in different ranges. We convert to SimplerEnv: -1=open, 1=close.
        """
        # Assume action is in [-1, 1] or [0, 1]
        if action > 1.0 or action < -1.0:
            # Clip to valid range
            action = np.clip(action, -1.0, 1.0)

        # Convert to relative gripper action
        relative_gripper_action = action

        # Switch to sticky closing
        if np.abs(relative_gripper_action) > 0.5 and not self.sticky_action_is_on:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action

        # Apply sticky action
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action

        # Reset after max repeats
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0

        return relative_gripper_action

    def get_video_frame(self, env, obs: dict) -> np.ndarray:
        """Get frame for video recording."""
        return get_image_from_maniskill2_obs_dict(env, obs)
