"""
GR00T N1.6 Model Wrapper for SimplerEnv evaluation.

This module wraps NVIDIA's GR00T-N1.6-3B model for use with SimplerEnv tasks.
It requires the Isaac-GR00T package to be installed in a separate conda environment.

Usage:
    model = GR00TInference(
        model_path="nvidia/GR00T-N1.6-3B",
        embodiment="bridge",
        device="cuda:0",
    )
    actions = model.forward(images, state, instruction)
"""

from typing import Optional

import numpy as np
import torch


class GR00TInference:
    """Wrapper around GR00T-N1.6-3B for SimplerEnv evaluation."""

    # Embodiment tag enum names for different robot platforms
    # These map to gr00t.data.embodiment_tags.EmbodimentTag enum values
    EMBODIMENT_TAG_NAMES = {
        "bridge": "OXE_WIDOWX",    # WidowX robot from Open-X-Embodiment
        "fractal": "OXE_GOOGLE",   # Google robot from Open-X-Embodiment
    }

    # State dimensions for each embodiment
    STATE_DIMS = {
        "bridge": 7,  # xyz + euler + gripper
        "fractal": 8,  # xyz + quat + gripper
    }

    # Action dimensions for each embodiment
    ACTION_DIMS = {
        "bridge": 7,  # xyz + euler + gripper
        "fractal": 7,  # xyz + euler + gripper
    }

    def __init__(
        self,
        model_path: str = "nvidia/GR00T-N1.6-3B",
        embodiment: str = "bridge",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        action_horizon: int = 16,
    ):
        """
        Initialize the GR00T model wrapper.

        Args:
            model_path: HuggingFace model path or local path
            embodiment: Robot embodiment ("bridge" or "fractal")
            device: Device to run on (e.g., "cuda:0")
            dtype: Data type for model (unused, GR00T uses bfloat16 internally)
            action_horizon: Number of action steps to predict
        """
        self.model_path = model_path
        self.embodiment = embodiment
        self.device = device
        self.dtype = dtype
        self.action_horizon = action_horizon

        if embodiment not in self.EMBODIMENT_TAG_NAMES:
            raise ValueError(
                f"Unknown embodiment: {embodiment}. "
                f"Supported: {list(self.EMBODIMENT_TAG_NAMES.keys())}"
            )

        self.embodiment_tag_name = self.EMBODIMENT_TAG_NAMES[embodiment]
        self.state_dim = self.STATE_DIMS[embodiment]
        self.action_dim = self.ACTION_DIMS[embodiment]

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the GR00T model from HuggingFace or local path."""
        try:
            from gr00t.policy.gr00t_policy import Gr00tPolicy
            from gr00t.data.embodiment_tags import EmbodimentTag
        except ImportError:
            raise ImportError(
                "Isaac-GR00T not installed. Please install with:\n"
                "  git clone https://github.com/NVIDIA/Isaac-GR00T.git\n"
                "  cd Isaac-GR00T && pip install -e ."
            )

        # Get the embodiment tag enum
        self.embodiment_tag = getattr(EmbodimentTag, self.embodiment_tag_name)

        print(f"[GR00T] Loading model from {self.model_path}")
        print(f"[GR00T] Embodiment: {self.embodiment} -> {self.embodiment_tag_name}")

        self.policy = Gr00tPolicy(
            embodiment_tag=self.embodiment_tag,
            model_path=self.model_path,
            device=self.device,
        )

        # Get modality config to understand expected keys
        self.modality_config = self.policy.modality_configs

        print(f"[GR00T] Model loaded successfully on {self.device}")
        print(f"[GR00T] Video keys: {self.modality_config['video'].modality_keys}")
        print(f"[GR00T] State keys: {self.modality_config['state'].modality_keys}")
        print(f"[GR00T] Language key: {self.policy.language_key}")

    def forward(
        self,
        images: np.ndarray,
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """
        Run inference to get action predictions.

        Args:
            images: RGB images [H, W, 3] uint8 (single image)
            state: Robot proprioception state [state_dim] float32
                   Bridge: [7] = xyz + rpy + gripper
                   Fractal: [8] = xyz + quat(xyzw) + gripper
            instruction: Language instruction string

        Returns:
            actions: Predicted action chunk [horizon, action_dim] unnormalized
        """
        # Get expected keys from modality config
        video_key = self.modality_config["video"].modality_keys[0]
        state_keys = self.modality_config["state"].modality_keys
        language_key = self.policy.language_key

        # Prepare observation dict for GR00T
        # Expected format:
        #   video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
        #   state: dict[str, np.ndarray[np.float32, (B, T, D)]] - each key is a separate array
        #   language: dict[str, list[list[str]]] (shape B, T)

        # Add batch and temporal dimensions
        # images: [H, W, 3] -> [1, 1, H, W, 3]
        video_data = images[np.newaxis, np.newaxis, ...].astype(np.uint8)

        # Split state into separate keys as expected by GR00T
        # Each state key should be [B, T, 1] for scalar values
        state_dict = self._split_state_to_keys(state, state_keys)

        # language: str -> [[str]]
        language_data = [[instruction]]

        obs = {
            "video": {video_key: video_data},
            "state": state_dict,
            "language": {language_key: language_data},
        }

        # Run inference
        # get_action returns (action_dict, info_dict) tuple
        action_chunk, _info = self.policy.get_action(obs)

        # Concatenate action keys into single array
        # For OXE embodiments: x, y, z, roll, pitch, yaw, gripper
        actions = self._concat_action_keys(action_chunk)

        return actions

    def _split_state_to_keys(
        self,
        state: np.ndarray,
        state_keys: list,
    ) -> dict:
        """
        Split state array into dict with separate keys for GR00T.

        For OXE_WIDOWX: x, y, z, roll, pitch, yaw, pad, gripper (8 keys)
        For OXE_GOOGLE: x, y, z, rx, ry, rz, rw, gripper (8 keys)

        Args:
            state: [state_dim] float32 array
            state_keys: list of state key names from modality config

        Returns:
            dict mapping state keys to [1, 1, 1] arrays
        """
        state_dict = {}

        # For Bridge/WidowX: state has 7 elements but 8 keys (pad is added)
        # For Fractal/Google: state has 8 elements and 8 keys
        if self.embodiment == "bridge":
            # State: xyz(3) + rpy(3) + gripper(1) = 7
            # Keys: x, y, z, roll, pitch, yaw, pad, gripper = 8
            mapping = {
                "x": state[0:1],
                "y": state[1:2],
                "z": state[2:3],
                "roll": state[3:4],
                "pitch": state[4:5],
                "yaw": state[5:6],
                "pad": np.array([0.0], dtype=np.float32),  # Dummy padding
                "gripper": state[6:7],
            }
        else:  # fractal
            # State: xyz(3) + quat_xyzw(4) + gripper(1) = 8
            # Keys: x, y, z, rx, ry, rz, rw, gripper = 8
            mapping = {
                "x": state[0:1],
                "y": state[1:2],
                "z": state[2:3],
                "rx": state[3:4],
                "ry": state[4:5],
                "rz": state[5:6],
                "rw": state[6:7],
                "gripper": state[7:8],
            }

        # Build state dict with proper shape [B, T, D] = [1, 1, 1]
        for key in state_keys:
            if key in mapping:
                state_dict[key] = mapping[key][np.newaxis, np.newaxis, :].astype(np.float32)
            else:
                raise KeyError(f"Unknown state key: {key}")

        return state_dict

    def _concat_action_keys(self, action_chunk: dict) -> np.ndarray:
        """
        Concatenate action keys into single array.

        For OXE embodiments, action keys are: x, y, z, roll, pitch, yaw, gripper

        Args:
            action_chunk: dict mapping action keys to [B, T, 1] arrays

        Returns:
            actions: [T, 7] numpy array
        """
        action_keys = self.modality_config["action"].modality_keys
        # Expected order: x, y, z, roll, pitch, yaw, gripper

        action_arrays = []
        for key in action_keys:
            arr = action_chunk[key]
            # Remove batch dimension: [B, T, 1] -> [T, 1]
            if arr.ndim == 3:
                arr = arr[0]
            action_arrays.append(arr)

        # Concatenate along last dimension: [T, 7]
        actions = np.concatenate(action_arrays, axis=-1)
        return actions

    def reset(self):
        """Reset policy state between episodes."""
        self.policy.reset()

    def get_action_chunk(
        self,
        images: np.ndarray,
        state: np.ndarray,
        instruction: str,
        num_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get action chunk, optionally truncated to num_steps.

        Args:
            images: RGB images [H, W, 3] uint8
            state: Robot proprioception [state_dim]
            instruction: Language instruction
            num_steps: Number of steps to return (default: all)

        Returns:
            actions: [num_steps, action_dim] normalized to [-1, 1]
        """
        actions = self.forward(images, state, instruction)

        if num_steps is not None:
            actions = actions[:num_steps]

        return actions
