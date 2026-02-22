"""
OpenVLA Model Wrapper for SimplerEnv evaluation.

This module wraps the OpenVLA-7B model for use with SimplerEnv tasks.
It requires the openvla package to be installed in a separate conda environment.

OpenVLA is a pure vision-language model (no proprioception input) that outputs
a single 7D action per inference call (no action chunking).

Usage:
    model = OpenVLAInference(
        model_path="openvla/openvla-7b",
        unnorm_key="bridge_orig",
        device="cuda:0",
    )
    action = model.forward(image, instruction)
"""

import numpy as np


class OpenVLAInference:
    """Wrapper around OpenVLA-7B for SimplerEnv evaluation."""

    def __init__(
        self,
        model_path: str = "openvla/openvla-7b",
        unnorm_key: str = "bridge_orig",
        device: str = "cuda:0",
        use_bf16: bool = True,
        load_in_8bit: bool = False,
    ):
        """
        Initialize the OpenVLA model wrapper.

        Args:
            model_path: HuggingFace model path or local path
            unnorm_key: Unnormalization key for action decoding (dataset-specific)
            device: Device to run on (e.g., "cuda:0")
            use_bf16: Use bfloat16 precision
            load_in_8bit: Load model in 8-bit quantization (reduces VRAM)
        """
        self.model_path = model_path
        self.unnorm_key = unnorm_key
        self.device = device
        self.use_bf16 = use_bf16
        self.load_in_8bit = load_in_8bit

        self._load_model()

    def _load_model(self):
        """Load the OpenVLA model from HuggingFace or local path."""
        import torch

        try:
            from prismatic.extern.hf import get_vla_model_and_tokenizer
        except ImportError:
            get_vla_model_and_tokenizer = None

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers not installed. Please install with:\n"
                "  pip install transformers>=4.40.0"
            )

        print(f"[OpenVLA] Loading model from {self.model_path}")
        print(f"[OpenVLA] Device: {self.device}, bf16: {self.use_bf16}, 8bit: {self.load_in_8bit}")

        # Register OpenVLA's custom Auto classes if available
        if get_vla_model_and_tokenizer is not None:
            # This registers the custom model class with AutoModelForVision2Seq
            pass

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Load model (matches SimplerEnv-OpenVLA reference)
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            if self.use_bf16:
                model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            **model_kwargs,
        )

        if not self.load_in_8bit:
            self.model = self.model.to(self.device)

        print(f"[OpenVLA] Model loaded successfully")
        if torch.cuda.is_available():
            device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            allocated_gb = torch.cuda.memory_allocated(device_idx) / 1e9
            print(f"[OpenVLA] GPU memory allocated: {allocated_gb:.2f} GB")

    def forward(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """
        Run inference to get a single action prediction.

        Args:
            image: RGB image [H, W, 3] uint8
            instruction: Language instruction string

        Returns:
            action: 7D action array [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
                    Already unnormalized by predict_action.
        """
        from PIL import Image

        # Convert numpy image to PIL
        pil_image = Image.fromarray(image)

        # Pass task description directly as prompt
        # (matches SimplerEnv-OpenVLA reference implementation)
        prompt = instruction

        # Tokenize inputs
        inputs = self.processor(prompt, pil_image).to(
            self.model.device, dtype=self.model.dtype
        )

        # Get action prediction (already unnormalized)
        action = self.model.predict_action(
            **inputs,
            unnorm_key=self.unnorm_key,
            do_sample=False,
        )

        # action is a numpy array of shape [7]
        return np.asarray(action, dtype=np.float32)

    def reset(self):
        """Reset policy state between episodes. No-op for OpenVLA (no recurrent state)."""
        pass
