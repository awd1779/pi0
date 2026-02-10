"""
OpenPI Pi0 Model Wrapper for SimplerEnv evaluation.

This module wraps Physical Intelligence's Pi0 base model for use with SimplerEnv tasks.
It requires the openpi package to be installed in a separate conda environment.

The pi0_base model is trained on 10k+ hours of diverse robot data including Bridge/OXE,
making it useful for testing CGVD on a pre-trained cross-embodiment model.

Usage:
    model = OpenPIInference(
        model_name="pi0_widowx",  # Use WidowX config for SimplerEnv Bridge tasks
        device="cuda:0",
    )
    actions = model.forward(images, state, instruction)

Environment Setup:
    conda create -n openpi python=3.11 -y
    conda activate openpi
    cd /home/ubuntu && git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
    cd openpi
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
"""

from typing import Optional

import numpy as np


class OpenPIInference:
    """Wrapper around OpenPI Pi0 models for SimplerEnv evaluation."""

    # Inference-ready models (config and checkpoint match)
    SUPPORTED_MODELS = [
        "pi0_widowx",       # WidowX/Bridge with pi0_base checkpoint (for SimplerEnv)
        "pi0_fractal",      # Fractal/Google Robot with pi0_base checkpoint (for SimplerEnv)
        "pi0_fast_droid",   # Cross-embodiment (DROID), fast inference
        "pi05_droid",       # Cross-embodiment (DROID), better language following
        "pi0_fast_libero",  # LIBERO simulation
        "pi05_libero",      # LIBERO simulation
        "pi0_aloha",        # ALOHA bimanual
    ]

    # Map config names to their checkpoint directories
    # Some configs use different checkpoints than their name suggests
    CHECKPOINT_MAP = {
        "pi0_widowx": "pi0_base",   # WidowX config uses pi0_base checkpoint
        "pi0_fractal": "pi0_base",  # Fractal config uses pi0_base checkpoint
    }

    def __init__(
        self,
        model_name: str = "pi0_widowx",
        device: str = "cuda:0",
        action_horizon: int = 16,
    ):
        """
        Initialize the OpenPI model wrapper.

        Args:
            model_name: OpenPI model name (e.g., "pi0_widowx", "pi0_fast_droid")
            device: Device to run on (e.g., "cuda:0")
            action_horizon: Number of action steps to predict
        """
        self.model_name = model_name
        self.device = device
        self.action_horizon = action_horizon

        if model_name not in self.SUPPORTED_MODELS:
            print(f"[OpenPI] Warning: {model_name} not in known models: {self.SUPPORTED_MODELS}")

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the OpenPI model from checkpoint."""
        try:
            from openpi.training import config as openpi_config
            from openpi.policies import policy_config
            from openpi.shared import download
        except ImportError:
            raise ImportError(
                "OpenPI not installed. Please install with:\n"
                "  cd /home/ubuntu && git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git\n"
                "  cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ."
            )

        print(f"[OpenPI] Loading model: {self.model_name}")

        # Get model configuration
        config = openpi_config.get_config(self.model_name)

        # Determine checkpoint directory (may differ from config name)
        checkpoint_name = self.CHECKPOINT_MAP.get(self.model_name, self.model_name)

        # Download checkpoint if needed (from Google Cloud Storage)
        checkpoint_dir = download.maybe_download(
            f"gs://openpi-assets/checkpoints/{checkpoint_name}"
        )

        print(f"[OpenPI] Config: {self.model_name}, Checkpoint: {checkpoint_name}")
        print(f"[OpenPI] Checkpoint dir: {checkpoint_dir}")

        # Create the policy
        self.policy = policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            pytorch_device=self.device,
        )

        print(f"[OpenPI] Model loaded successfully on {self.device}")

    def forward(
        self,
        image: np.ndarray,
        state: np.ndarray,
        instruction: str,
        wrist_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run inference to get action predictions.

        Args:
            image: RGB image [H, W, 3] uint8 (exterior/third-person view)
            state: Robot proprioception state [state_dim] float32
            instruction: Language instruction string
            wrist_image: Optional wrist camera image [H, W, 3] uint8 (unused for WidowX)

        Returns:
            actions: Predicted action chunk [horizon, action_dim]
        """
        # Ensure image is uint8 and HWC format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Ensure HWC format (224, 224, 3)
        if image.shape[0] == 3 and image.shape[-1] != 3:
            # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))

        # Build observation dict in WidowX format
        # The WidowXInputs transform will handle padding for missing cameras
        obs = {
            "observation/image": image,
            "observation/state": state.astype(np.float32),
            "prompt": instruction,
        }

        # Run inference
        result = self.policy.infer(obs)

        # Extract actions from result
        actions = result.get("actions", result.get("action", None))

        if actions is None:
            # Try to find action key
            for key in result.keys():
                if "action" in key.lower():
                    actions = result[key]
                    break

        if actions is None:
            raise ValueError(f"Could not find actions in result: {result.keys()}")

        # Ensure numpy array
        if hasattr(actions, "numpy"):
            actions = actions.numpy()

        # Ensure 2D: [horizon, action_dim]
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        return actions

    def reset(self):
        """Reset policy state between episodes."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def get_action_chunk(
        self,
        image: np.ndarray,
        state: np.ndarray,
        instruction: str,
        num_steps: Optional[int] = None,
        wrist_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get action chunk, optionally truncated to num_steps.

        Args:
            image: RGB image [H, W, 3] uint8
            state: Robot proprioception [state_dim]
            instruction: Language instruction
            num_steps: Number of steps to return (default: all)
            wrist_image: Optional wrist camera image (unused for WidowX)

        Returns:
            actions: [num_steps, action_dim]
        """
        actions = self.forward(image, state, instruction, wrist_image=wrist_image)

        if num_steps is not None:
            actions = actions[:num_steps]

        return actions

    def get_model_for_hooks(self):
        """
        Get the underlying model for attention hooks.

        Returns the internal model object that can be used with
        Pi0AttentionVisualizer for attention extraction.

        Returns:
            model: The underlying PyTorch model, or None if not accessible
        """
        # Try common attribute names used by OpenPI policies
        model_attrs = ['model', '_model', 'net', '_net', 'network']

        for attr in model_attrs:
            if hasattr(self.policy, attr):
                return getattr(self.policy, attr)

        # If not found, try to find any nn.Module in the policy
        try:
            import torch.nn as nn
            for name in dir(self.policy):
                obj = getattr(self.policy, name, None)
                if isinstance(obj, nn.Module):
                    return obj
        except Exception:
            pass

        return None

    def get_policy_info(self) -> dict:
        """
        Get information about the policy structure for debugging.

        Returns:
            dict: Information about policy attributes and structure
        """
        info = {
            'policy_type': type(self.policy).__name__,
            'policy_module': type(self.policy).__module__,
            'attributes': [],
            'nn_modules': [],
        }

        for name in dir(self.policy):
            if not name.startswith('_'):
                try:
                    obj = getattr(self.policy, name)
                    if callable(obj):
                        info['attributes'].append(f"{name}() - method")
                    else:
                        obj_type = type(obj).__name__
                        info['attributes'].append(f"{name}: {obj_type}")

                        # Check if it's an nn.Module
                        try:
                            import torch.nn as nn
                            if isinstance(obj, nn.Module):
                                info['nn_modules'].append(name)
                        except ImportError:
                            pass
                except Exception:
                    info['attributes'].append(f"{name}: <error accessing>")

        return info
