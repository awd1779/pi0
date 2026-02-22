"""
SpatialVLA Model Wrapper for SimplerEnv evaluation.

This module wraps the SpatialVLA-4B model for use with SimplerEnv tasks.
SpatialVLA uses a PaLiGemma2-3B backbone, 224x224 images, and supports
action chunking (4 future actions per inference call) with image history.

It requires transformers>=4.47.0 in a separate conda environment.

Usage:
    model = SpatialVLAInference(
        model_path="IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge",
        unnorm_key="bridge_orig/1.0.0",
        device="cuda:0",
    )
    actions = model.forward(image, instruction)  # [T, 7]
"""

from collections import deque

import numpy as np


class SpatialVLAInference:
    """Wrapper around SpatialVLA-4B for SimplerEnv evaluation."""

    def __init__(
        self,
        model_path: str = "IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge",
        unnorm_key: str = "bridge_orig/1.0.0",
        device: str = "cuda:0",
    ):
        self.model_path = model_path
        self.unnorm_key = unnorm_key
        self.device = device

        self._load_model()

    def _load_model(self):
        """Load the SpatialVLA model and processor from HuggingFace."""
        import torch
        from transformers import AutoModel, AutoProcessor

        print(f"[SpatialVLA] Loading model from {self.model_path}")
        print(f"[SpatialVLA] Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)

        # Image history buffer for temporal context
        self.obs_interval = self.processor.obs_delta
        num_obs_steps = self.processor.num_obs_steps
        maxlen = (num_obs_steps - 1) * self.obs_interval + 1
        self.image_history = deque(maxlen=maxlen)

        print(f"[SpatialVLA] obs_interval={self.obs_interval}, num_obs_steps={num_obs_steps}, history_len={maxlen}")
        print(f"[SpatialVLA] Model loaded successfully")
        if torch.cuda.is_available():
            device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            allocated_gb = torch.cuda.memory_allocated(device_idx) / 1e9
            print(f"[SpatialVLA] GPU memory allocated: {allocated_gb:.2f} GB")

    def forward(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """
        Run inference to get an action chunk prediction.

        Args:
            image: RGB image [H, W, 3] uint8
            instruction: Language instruction string

        Returns:
            actions: [T, 7] numpy array of action chunks
                     (xyz + euler + gripper), already unnormalized.
        """
        from PIL import Image

        pil_image = Image.fromarray(image)
        self.image_history.append(pil_image)

        # Sample images at obs_interval spacing from history
        images = list(self.image_history)[::self.obs_interval]

        inputs = self.processor(
            images=images,
            text=instruction,
            unnorm_key=self.unnorm_key,
            return_tensors="pt",
            do_normalize=False,
        ).to(self.model.device)

        generation_outputs = self.model.predict_action(inputs)
        decoded = self.processor.decode_actions(
            generation_outputs,
            unnorm_key=self.unnorm_key,
        )
        actions = decoded["actions"]  # [T, 7] numpy array

        return np.asarray(actions, dtype=np.float32)

    def reset(self):
        """Reset policy state between episodes. Clears image history."""
        self.image_history.clear()
