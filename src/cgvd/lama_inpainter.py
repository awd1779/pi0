"""LaMa-based inpainting for CGVD distractor removal."""

import numpy as np


class LamaInpainter:
    """Removes masked regions using LaMa inpainting."""

    def __init__(self, device: str = "cuda"):
        """Initialize LaMa model.

        Args:
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy load LaMa model on first use."""
        if self._model is None:
            from simple_lama_inpainting import SimpleLama

            self._model = SimpleLama(device=self.device)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        dilate_mask: int = 3,
    ) -> np.ndarray:
        """Remove masked regions via inpainting.

        Args:
            image: RGB image [H, W, 3] uint8
            mask: Binary mask where 1 = inpaint (remove), 0 = keep
            dilate_mask: Pixels to dilate mask (helps clean edges)

        Returns:
            Inpainted image [H, W, 3] uint8
        """
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

        return np.array(result)
