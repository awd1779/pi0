"""CGVD Wrapper - Main gym.Wrapper for Concept-Gated Visual Distillation."""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from src.cgvd.instruction_parser import InstructionParser
from src.cgvd.sam3_segmenter import SAM3Segmenter, create_segmenter
from src.cgvd.spectral_abstraction import SpectralAbstraction


class CGVDWrapper(gym.Wrapper):
    """Concept-Gated Visual Distillation wrapper for SimplerEnv.

    This wrapper intercepts observations from the environment and applies
    visual distillation to reduce clutter while preserving task-relevant regions.

    Pipeline:
    1. Extract image and instruction from observation
    2. Parse instruction to identify target and anchor objects
    3. Segment image using SAM3 to find relevant regions (at 1Hz)
    4. Apply spectral abstraction (blur) to background regions
    5. Write distilled image back to observation

    The robot arm and gripper are ALWAYS included in the foreground mask
    to prevent proprioception alignment issues.
    """

    def __init__(
        self,
        env: gym.Env,
        update_freq: int = 1,
        blur_sigma: float = 15.0,
        presence_threshold: float = 0.15,
        use_mock_segmenter: bool = False,
        segmenter_model: str = "facebook/sam3",
        feather_edges: bool = False,
        feather_radius: int = 5,
        include_robot: bool = True,
        verbose: bool = False,
        save_debug_images: bool = False,
        debug_dir: str = "cgvd_debug",
        distractor_names: Optional[List[str]] = None,
    ):
        """Initialize CGVD wrapper.

        Args:
            env: SimplerEnv environment to wrap
            update_freq: Frames between SAM3 segmentation updates (default 1 = every frame)
            blur_sigma: Gaussian blur sigma for background (default 15.0)
            presence_threshold: Minimum confidence to accept SAM3 mask (default 0.15)
            use_mock_segmenter: Use mock segmenter for testing (default False)
            segmenter_model: SAM3 model identifier (default "facebook/sam3")
            feather_edges: Apply edge feathering to mask transitions (default False)
            feather_radius: Radius for edge feathering (default 5)
            include_robot: Always include robot arm/gripper in mask (default True)
            verbose: Print debug information (default False)
            save_debug_images: Save original/mask/distilled images for debugging (default False)
            debug_dir: Directory to save debug images (default "cgvd_debug")
            distractor_names: List of distractor object names to blur (e.g., ["fork", "knife"]).
                            If provided, switches to distractor-only blurring mode.
        """
        super().__init__(env)

        # Configuration
        self.update_freq = update_freq
        self.blur_sigma = blur_sigma
        self.presence_threshold = presence_threshold
        self.feather_edges = feather_edges
        self.feather_radius = feather_radius
        self.include_robot = include_robot
        self.verbose = verbose
        self.save_debug_images = save_debug_images
        self.debug_dir = debug_dir
        self.distractor_names = distractor_names or []

        # Create debug directory if needed
        if self.save_debug_images:
            os.makedirs(self.debug_dir, exist_ok=True)

        # Initialize components
        self.segmenter = create_segmenter(
            use_mock=use_mock_segmenter,
            model_name=segmenter_model,
            presence_threshold=presence_threshold,
        )
        self.parser = InstructionParser()
        self.abstraction = SpectralAbstraction(sigma=blur_sigma)

        # State
        self.cached_mask: Optional[np.ndarray] = None
        self.frame_count: int = 0
        self.current_instruction: Optional[str] = None
        self.current_target: Optional[str] = None
        self.current_anchor: Optional[str] = None
        self.episode_count: int = 0
        self.episode_debug_dir: Optional[str] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and clear cached state.

        Args:
            seed: Random seed for environment reset
            options: Additional reset options

        Returns:
            Tuple of (observation, info) with distilled observation
        """
        # Reset environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Clear cached state
        self.cached_mask = None
        self.frame_count = 0
        self.current_instruction = None
        self.current_target = None
        self.current_anchor = None

        # Create episode-specific debug directory
        if self.save_debug_images:
            self.episode_debug_dir = os.path.join(
                self.debug_dir, f"episode_{self.episode_count:03d}"
            )
            os.makedirs(self.episode_debug_dir, exist_ok=True)
        self.episode_count += 1

        # Apply CGVD to initial observation
        obs = self._apply_cgvd(obs)

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take environment step and apply CGVD to observation.

        Args:
            action: Action to take in environment

        Returns:
            Tuple of (obs, reward, terminated, truncated, info) with distilled observation
        """
        # Take environment step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply CGVD to observation
        obs = self._apply_cgvd(obs)

        return obs, reward, terminated, truncated, info

    def _get_image_and_camera(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """Extract image and camera name from observation.

        Determines the correct camera based on robot type to ensure we read
        and write to the same camera that the VLA uses.

        Args:
            obs: Observation dict from environment

        Returns:
            Tuple of (image array, camera name)
        """
        unwrapped = self.env.unwrapped

        # Determine camera name based on robot type (same logic as simpler_env)
        if "google_robot" in unwrapped.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in unwrapped.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            # Fallback: use first camera with rgb
            camera_name = None
            for name in obs.get("image", {}):
                if "rgb" in obs["image"][name]:
                    camera_name = name
                    break
            if camera_name is None:
                raise ValueError("Could not determine camera name from observation")

        image = obs["image"][camera_name]["rgb"]
        return image, camera_name

    def _apply_cgvd(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Concept-Gated Visual Distillation to observation.

        Args:
            obs: Raw observation from environment

        Returns:
            Modified observation with distilled image
        """
        # Stage 0: Extract image from SimplerEnv
        # IMPORTANT: Get camera name to ensure we write back to the same camera
        image, camera_name = self._get_image_and_camera(obs)

        if self.distractor_names:
            # DISTRACTOR MODE: Segment and blur only specified distractor objects
            # Build concept prompt from distractor names
            concepts = ". ".join(self.distractor_names)

            if self.frame_count % self.update_freq == 0 or self.cached_mask is None:
                self.cached_mask = self.segmenter.segment(image, concepts)
                if self.verbose:
                    mask_coverage = self.cached_mask.sum() / self.cached_mask.size * 100
                    print(
                        f"[CGVD] Frame {self.frame_count}: Distractor mask ({concepts}), "
                        f"coverage={mask_coverage:.1f}%"
                    )

            self.frame_count += 1

            # Apply blur to ONLY distractor regions (mask=1 means blur)
            distilled = self.abstraction.apply_to_masked_regions(image, self.cached_mask)

        else:
            # LEGACY MODE: Segment foreground, blur background
            instruction = self.env.unwrapped.get_language_instruction()

            # Check if instruction changed (for multi-stage tasks)
            if instruction != self.current_instruction:
                self.current_instruction = instruction
                self.current_target, self.current_anchor = self.parser.parse(instruction)
                # Force mask update on instruction change
                self.cached_mask = None
                if self.verbose:
                    print(
                        f"[CGVD] New instruction: '{instruction}' -> "
                        f"target='{self.current_target}', anchor='{self.current_anchor}'"
                    )

            # Build concept prompt (target + anchor + robot)
            concepts = self.parser.build_concept_prompt(
                self.current_target,
                self.current_anchor,
                include_robot=self.include_robot,
            )

            # Segment at update_freq or on cache miss
            if self.frame_count % self.update_freq == 0 or self.cached_mask is None:
                self.cached_mask = self.segmenter.segment(image, concepts)
                if self.verbose:
                    mask_coverage = self.cached_mask.sum() / self.cached_mask.size * 100
                    print(
                        f"[CGVD] Frame {self.frame_count}: Updated mask, "
                        f"coverage={mask_coverage:.1f}%"
                    )

            self.frame_count += 1

            # Apply spectral abstraction (blur background, keep foreground sharp)
            if self.feather_edges:
                distilled = self.abstraction.apply_with_edge_feathering(
                    image, self.cached_mask, feather_radius=self.feather_radius
                )
            else:
                distilled = self.abstraction.apply(image, self.cached_mask)

        # Write distilled image back to observation
        # IMPORTANT: Write to the SAME camera we read from (camera_name)
        obs = self._write_image_to_obs(obs, distilled, camera_name)

        # Save debug images if enabled
        if self.save_debug_images:
            self._save_debug_images(image, self.cached_mask, distilled)

        return obs

    def _save_debug_images(
        self, original: np.ndarray, mask: np.ndarray, distilled: np.ndarray
    ):
        """Save debug images showing original, mask, and distilled output."""
        frame_num = self.frame_count - 1  # Already incremented

        # Create side-by-side comparison
        h, w = original.shape[:2]

        # Convert mask to visualization (grayscale -> RGB)
        mask_vis = (mask * 255).astype(np.uint8)
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)

        # Stack horizontally: original | mask | distilled
        comparison = np.hstack([original, mask_vis, distilled])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
        # Show mask label based on mode
        if self.distractor_names:
            mask_label = f"Distractors: {', '.join(self.distractor_names[:3])}"
            if len(self.distractor_names) > 3:
                mask_label += "..."
            mask_label += " (white=blur)"
        else:
            mask_label = "Foreground (white=keep)"
        cv2.putText(comparison, mask_label, (w + 10, 30), font, 0.5, (255, 255, 255), 2)
        cv2.putText(comparison, "Distilled (VLA input)", (2 * w + 10, 30), font, 0.7, (255, 255, 255), 2)

        # Add confidence scores if available
        if hasattr(self.segmenter, 'last_scores') and self.segmenter.last_scores:
            y_offset = 55
            for concept, score in self.segmenter.last_scores.items():
                # Color code: green if detected (>threshold), red if not
                color = (0, 255, 0) if score >= self.presence_threshold else (255, 100, 100)
                score_text = f"{concept}: {score:.2f}"
                cv2.putText(comparison, score_text, (w + 10, y_offset), font, 0.4, color, 1)
                y_offset += 18

        # Add instruction text
        if self.current_instruction:
            cv2.putText(
                comparison,
                f"Instruction: {self.current_instruction}",
                (10, h - 10),
                font,
                0.5,
                (255, 255, 0),
                1,
            )

        # Save as BGR for OpenCV
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(self.episode_debug_dir, f"frame_{frame_num:04d}.png"), comparison_bgr
        )

    def _write_image_to_obs(
        self, obs: Dict[str, Any], image: np.ndarray, camera_name: str
    ) -> Dict[str, Any]:
        """Write distilled image back into observation dict.

        IMPORTANT: Writes to the SPECIFIC camera that was read from, not just
        the first camera in the dict. This ensures the VLA sees the distilled
        image from the correct camera.

        Args:
            obs: Original observation dict
            image: Distilled image to write back
            camera_name: Name of camera to write to (must match what was read)

        Returns:
            Modified observation dict
        """
        # SimplerEnv uses ManiSkill2 observation format
        # Write to the SPECIFIC camera we read from
        if "image" in obs and camera_name in obs["image"]:
            if "rgb" in obs["image"][camera_name]:
                obs["image"][camera_name]["rgb"] = image
            else:
                # Camera exists but no rgb key - add it
                obs["image"][camera_name]["rgb"] = image

        # Also check for direct image key (some SimplerEnv versions)
        elif "rgb" in obs:
            obs["rgb"] = image

        # Handle other potential formats
        elif "pixels" in obs:
            obs["pixels"] = image

        return obs

    def get_current_mask(self) -> Optional[np.ndarray]:
        """Get the current cached segmentation mask.

        Useful for visualization and debugging.

        Returns:
            Current mask or None if not yet computed
        """
        return self.cached_mask

    def get_mask_stats(self) -> Dict[str, Any]:
        """Get statistics about current mask for debugging.

        Returns:
            Dict with mask statistics
        """
        if self.cached_mask is None:
            return {"initialized": False}

        return {
            "initialized": True,
            "frame_count": self.frame_count,
            "mask_coverage": float(self.cached_mask.sum() / self.cached_mask.size),
            "mask_shape": self.cached_mask.shape,
            "current_target": self.current_target,
            "current_anchor": self.current_anchor,
            "current_instruction": self.current_instruction,
        }
