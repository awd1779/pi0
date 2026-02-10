"""LaMa-based inpainting for CGVD distractor removal."""

import time
from typing import Optional

import numpy as np


# Module-level singleton for sharing across CGVDWrapper instances
_lama_singleton: Optional["LamaInpainter"] = None


def get_lama_inpainter(device: str = "cuda") -> "LamaInpainter":
    """Get or create a singleton LaMa inpainter.

    This function returns a shared instance to avoid redundant model loading
    when multiple CGVDWrapper instances are created (e.g., in batch evaluation).

    Args:
        device: Device to run on ("cuda" or "cpu"), only used on first call

    Returns:
        Shared LamaInpainter instance
    """
    global _lama_singleton

    if _lama_singleton is None:
        _lama_singleton = LamaInpainter(device=device)
        # Force lazy initialization to load the model now
        _lama_singleton._load_model()
        print("[LaMa] Created singleton LamaInpainter")
    return _lama_singleton


def clear_lama_singleton():
    """Clear the singleton instance (useful for testing or memory cleanup)."""
    global _lama_singleton
    _lama_singleton = None


class LamaInpainter:
    """Removes masked regions using LaMa inpainting."""

    def __init__(self, device: str = "cuda"):
        """Initialize LaMa model.

        Args:
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device
        self._model = None

        # Timing instrumentation
        self.last_inpaint_time: float = 0.0

    def _load_model(self):
        """Lazy load LaMa model on first use."""
        if self._model is None:
            from simple_lama_inpainting import SimpleLama

            self._model = SimpleLama(device=self.device)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        dilate_mask: int = 11,
    ) -> np.ndarray:
        """Remove masked regions via inpainting.

        Args:
            image: RGB image [H, W, 3] uint8
            mask: Binary mask where 1 = inpaint (remove), 0 = keep
            dilate_mask: Pixels to dilate mask (helps cover shadows/edges)

        Returns:
            Inpainted image [H, W, 3] uint8
        """
        start_time = time.time()

        self._load_model()

        # Convert mask to uint8 (LaMa expects 0-255)
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Optional: dilate mask slightly for cleaner edges
        if dilate_mask > 0:
            import cv2

            kernel = np.ones((dilate_mask, dilate_mask), np.uint8)
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)

        # Run LaMa inpainting
        result = self._model(image, mask_uint8)

        self.last_inpaint_time = time.time() - start_time

        return np.array(result)
