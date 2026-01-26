"""Spectral Visual Abstraction for background neutralization."""

import cv2
import numpy as np


class SpectralAbstraction:
    """Applies spectral abstraction (Gaussian blur) to background regions.

    The distilled image preserves sharp details for foreground (masked) regions
    while blurring background regions to reduce visual clutter.

    Formula: I_distilled = M * I_raw + (1-M) * GaussianBlur(I_raw)
    """

    def __init__(self, sigma: float = 15.0):
        """Initialize spectral abstraction.

        Args:
            sigma: Standard deviation for Gaussian blur (default 15.0)
        """
        self.sigma = sigma

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply spectral abstraction to an image.

        Args:
            image: Input RGB image, shape (H, W, 3), dtype uint8
            mask: Binary mask where 1 = foreground (keep sharp), 0 = background (blur)
                  Shape (H, W), dtype bool or float

        Returns:
            Distilled image with sharp foreground and blurred background,
            shape (H, W, 3), dtype uint8
        """
        # Validate inputs
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3), got shape {image.shape}")

        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask (H,W), got shape {mask.shape}")

        if mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match image shape {image.shape[:2]}"
            )

        # Apply Gaussian blur to entire image
        # ksize=(0,0) means kernel size is computed from sigma
        blurred = cv2.GaussianBlur(
            image, ksize=(0, 0), sigmaX=self.sigma, sigmaY=self.sigma
        )

        # Expand mask to 3 channels for broadcasting
        mask_3d = mask[..., np.newaxis].astype(np.float32)

        # Composite: foreground sharp, background blurred
        # I_distilled = M * I_raw + (1-M) * Blur(I_raw)
        distilled = image.astype(np.float32) * mask_3d + blurred.astype(
            np.float32
        ) * (1 - mask_3d)

        return distilled.astype(np.uint8)

    def apply_with_edge_feathering(
        self, image: np.ndarray, mask: np.ndarray, feather_radius: int = 5
    ) -> np.ndarray:
        """Apply spectral abstraction with smooth edge transitions.

        This variant applies a slight blur to the mask edges to avoid
        hard boundaries between sharp and blurred regions.

        Args:
            image: Input RGB image, shape (H, W, 3), dtype uint8
            mask: Binary mask, shape (H, W)
            feather_radius: Radius for edge feathering (default 5)

        Returns:
            Distilled image with feathered edges, shape (H, W, 3), dtype uint8
        """
        # Feather the mask edges
        mask_float = mask.astype(np.float32)
        if feather_radius > 0:
            mask_float = cv2.GaussianBlur(
                mask_float, ksize=(0, 0), sigmaX=feather_radius, sigmaY=feather_radius
            )

        # Apply blur to image
        blurred = cv2.GaussianBlur(
            image, ksize=(0, 0), sigmaX=self.sigma, sigmaY=self.sigma
        )

        # Composite with feathered mask
        mask_3d = mask_float[..., np.newaxis]
        distilled = image.astype(np.float32) * mask_3d + blurred.astype(
            np.float32
        ) * (1 - mask_3d)

        return distilled.astype(np.uint8)

    def apply_to_masked_regions(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Blur ONLY the masked regions, keep everything else sharp.

        This is the INVERSE of apply() - here mask=1 means BLUR (distractor),
        not KEEP. Used for distractor-only blurring mode.

        Formula: I_distilled = (1-M) * I_raw + M * GaussianBlur(I_raw)

        Args:
            image: Input RGB image, shape (H, W, 3), dtype uint8
            mask: Binary mask where 1 = blur (distractor), 0 = keep sharp
                  Shape (H, W), dtype bool or float

        Returns:
            Distilled image with blurred distractors and sharp background,
            shape (H, W, 3), dtype uint8
        """
        # Validate inputs
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3), got shape {image.shape}")

        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask (H,W), got shape {mask.shape}")

        if mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match image shape {image.shape[:2]}"
            )

        # Apply Gaussian blur to entire image
        blurred = cv2.GaussianBlur(
            image, ksize=(0, 0), sigmaX=self.sigma, sigmaY=self.sigma
        )

        # Expand mask to 3 channels for broadcasting
        mask_3d = mask[..., np.newaxis].astype(np.float32)

        # INVERTED composite: blur where mask=1, keep sharp where mask=0
        # I_distilled = (1-M) * I_raw + M * Blur(I_raw)
        distilled = (
            image.astype(np.float32) * (1 - mask_3d)
            + blurred.astype(np.float32) * mask_3d
        )

        return distilled.astype(np.uint8)