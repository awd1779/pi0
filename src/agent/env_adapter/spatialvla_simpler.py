"""
SpatialVLA-specific environment adapter for SimplerEnv.

This adapter handles preprocessing (obs -> model input) and postprocessing
(model output -> env action) for SpatialVLA on Bridge tasks.

Same as OpenVLA adapter: pure vision-language (no proprioception),
euler->axis-angle conversion, gripper binarization. SpatialVLA returns
action chunks [T, 7] instead of single actions.
"""

from typing import Tuple

import cv2
import numpy as np
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from src.agent.env_adapter.base import BaseEnvAdapter
from src.utils.geometry import euler2axangle


class SpatialVLABridgeSimplerAdapter(BaseEnvAdapter):
    """
    SpatialVLA adapter for Bridge/WidowX tasks in SimplerEnv.

    No proprioception input (pure vision-language).
    Action: [T, 7] chunks of [xyz + euler + gripper] - already unnormalized.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        **kwargs,
    ):
        super().__init__()
        self.image_size = tuple(image_size)

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
        Preprocess observation for SpatialVLA.

        Returns dict with:
            - image: [H, W, 3] uint8 RGB resized to 224x224
            - instruction: str
        """
        image = get_image_from_maniskill2_obs_dict(env.unwrapped, obs)
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_AREA,
        )

        return {
            "image": image,
            "instruction": instruction,
        }

    def postprocess(
        self,
        actions: np.ndarray,
    ) -> np.ndarray:
        """
        Postprocess SpatialVLA actions for SimplerEnv.

        Converts euler angles to axis-angle and binarizes gripper.

        Args:
            actions: [N, 7] already unnormalized (xyz + euler + gripper)

        Returns:
            env_actions: [N, 7] for SimplerEnv (xyz + axis-angle + gripper)
        """
        env_actions = np.zeros((len(actions), 7))
        for idx, action in enumerate(actions):
            roll, pitch, yaw = action[3:6]
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)

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
        return get_image_from_maniskill2_obs_dict(env.unwrapped, obs)
