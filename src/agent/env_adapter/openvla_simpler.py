"""
OpenVLA-specific environment adapter for SimplerEnv.

This adapter handles preprocessing (obs -> model input) and postprocessing
(model output -> env action) for OpenVLA on Bridge tasks.

Key difference from GR00T adapter:
- OpenVLA is a pure vision-language model: NO proprioception input
- OpenVLA outputs a single action per inference call (no action chunking)
- Action format is the same as GR00T Bridge: euler + gripper [0,1]
"""

from typing import Tuple

import cv2
import numpy as np
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from src.agent.env_adapter.base import BaseEnvAdapter
from src.utils.geometry import euler2axangle


class OpenVLABridgeSimplerAdapter(BaseEnvAdapter):
    """
    OpenVLA adapter for Bridge/WidowX tasks in SimplerEnv.

    No proprioception input (pure vision-language).
    Action: 7D [xyz + euler + gripper] - already unnormalized by predict_action.
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
        Preprocess observation for OpenVLA.

        OpenVLA takes only an image and text instruction (no proprioception).

        Returns dict with:
            - image: [H, W, 3] uint8 RGB
            - instruction: str
        """
        image = get_image_from_maniskill2_obs_dict(env, obs)
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
        Postprocess OpenVLA actions for SimplerEnv.

        OpenVLA's predict_action already unnormalizes actions, so we only need to:
        1. Convert euler angles to axis-angle format
        2. Convert gripper from [0,1] to [-1,1] (binarized)

        Args:
            actions: [N, 7] already unnormalized (xyz + euler + gripper)
                     where gripper is in [0, 1] (0=close, 1=open)

        Returns:
            env_actions: [N, 7] for SimplerEnv (xyz + axis-angle + gripper)
        """
        env_actions = np.zeros((len(actions), 7))
        for idx, action in enumerate(actions):
            roll, pitch, yaw = action[3:6]
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)

            # Gripper: OpenVLA outputs [0, 1] (0=close, 1=open)
            # Binarize and convert to SimplerEnv: -1=close, 1=open
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
