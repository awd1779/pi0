"""
GR00T-specific environment adapters for SimplerEnv.

These adapters handle preprocessing (obs -> model input) and postprocessing
(model output -> env action) for GR00T on Bridge and Fractal tasks.

Key difference from Pi0 adapters:
- GR00T's processor handles normalization internally, so we pass raw values
- GR00T's decode_action already unnormalizes outputs, so we only convert formats
"""

from typing import Tuple

import cv2
import numpy as np
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from src.agent.env_adapter.base import BaseEnvAdapter
from src.utils.geometry import euler2axangle, mat2euler, quat2mat


class GR00TBridgeSimplerAdapter(BaseEnvAdapter):
    """
    GR00T adapter for Bridge/WidowX tasks in SimplerEnv.

    Proprioception: 7D [xyz + euler (rpy) + gripper_openness]
    Action: 7D [xyz + euler + gripper] - already unnormalized by GR00T
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        **kwargs,  # Accept but ignore unused params for compatibility
    ):
        super().__init__()
        self.image_size = tuple(image_size)

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
        Preprocess observation for GR00T.

        GR00T's processor handles normalization internally, so we pass raw values.

        Returns dict with:
            - image: [H, W, 3] uint8 RGB
            - state: [7] float32 proprioception (raw, unnormalized)
            - instruction: str
        """
        # Get and resize image
        image = get_image_from_maniskill2_obs_dict(env, obs)  # [H, W, 3]
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Get raw proprioception (GR00T processor handles normalization)
        raw_proprio = self._get_raw_proprio(obs)

        return {
            "image": image,
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
        Postprocess GR00T actions for SimplerEnv.

        GR00T's decode_action already unnormalizes actions, so we only need to:
        1. Convert euler angles to axis-angle format
        2. Convert gripper from [0,1] to [-1,1]

        Args:
            actions: [horizon, 7] already unnormalized by GR00T (xyz + euler + gripper)
                     where gripper is in [0, 1] (0=close, 1=open)

        Returns:
            env_actions: [horizon, 7] for SimplerEnv (xyz + axis-angle + gripper)
        """
        # Actions are already in physical units from GR00T's decode_action
        # Just convert euler to axis-angle format for SimplerEnv
        env_actions = np.zeros((len(actions), 7))
        for idx, action in enumerate(actions):
            roll, pitch, yaw = action[3:6]
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)

            # Gripper: GR00T outputs [0, 1] (0=close, 1=open)
            # Convert to SimplerEnv: -1=close, 1=open
            gripper = action[-1]
            action_gripper = 2.0 * (gripper > 0.5) - 1.0

            env_actions[idx] = np.concatenate([
                action[:3],
                action_rotation_ax * action_rotation_angle,
                [action_gripper],
            ])

        return env_actions

    def get_video_frame(self, env, obs: dict) -> np.ndarray:
        """Get frame for video recording."""
        return get_image_from_maniskill2_obs_dict(env, obs)


class GR00TFractalSimplerAdapter(BaseEnvAdapter):
    """
    GR00T adapter for Fractal/Google Robot tasks in SimplerEnv.

    Proprioception: 8D [xyz + quat (xyzw) + gripper_closedness]
    Action: 7D [xyz + euler + gripper] - already unnormalized by GR00T
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        **kwargs,  # Accept but ignore unused params for compatibility
    ):
        super().__init__()
        self.image_size = tuple(image_size)

        # Sticky gripper state
        self.sticky_gripper_num_repeat = 15
        self.reset()

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
        Preprocess observation for GR00T.

        GR00T's processor handles normalization internally, so we pass raw values.

        Returns dict with:
            - image: [H, W, 3] uint8 RGB
            - state: [8] float32 proprioception (raw, unnormalized)
            - instruction: str
        """
        # Get and resize image
        image = get_image_from_maniskill2_obs_dict(env, obs)  # [H, W, 3]
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Get raw proprioception (GR00T processor handles normalization)
        raw_proprio = self._get_raw_proprio(obs)

        return {
            "image": image,
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
        Postprocess GR00T actions for SimplerEnv.

        GR00T's decode_action already unnormalizes actions, so we only need to:
        1. Convert euler angles to axis-angle format
        2. Apply sticky gripper mechanism

        Args:
            actions: [horizon, 7] already unnormalized by GR00T (xyz + euler + gripper)
                     where gripper is in [0, 1] (0=close, 1=open)

        Returns:
            env_actions: [horizon, 7] for SimplerEnv (xyz + axis-angle + gripper)
        """
        # Actions are already in physical units from GR00T's decode_action
        # Just convert euler to axis-angle format for SimplerEnv
        env_actions = np.zeros((len(actions), 7))
        for idx, action in enumerate(actions):
            roll, pitch, yaw = action[3:6]
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)

            # Process gripper with sticky mechanism
            gripper = action[-1]
            action_gripper = self._postprocess_gripper(gripper)

            env_actions[idx] = np.concatenate([
                action[:3],
                action_rotation_ax * action_rotation_angle,
                [action_gripper],
            ])

        return env_actions

    def _postprocess_gripper(self, action: float) -> float:
        """
        Process gripper action with sticky mechanism.

        Trained with [0, 1] (0=close, 1=open).
        Convert to SimplerEnv: -1=open, 1=close.
        """
        action = (action * 2) - 1  # [0, 1] -> [-1, 1] where -1=close, 1=open
        relative_gripper_action = -action  # Flip for SimplerEnv convention

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
