"""CGVD Wrapper - Main gym.Wrapper for Concept-Gated Visual Distillation."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from src.cgvd.instruction_parser import InstructionParser
from src.cgvd.sam3_segmenter import SAM3Segmenter, create_segmenter


class CGVDWrapper(gym.Wrapper):
    """Concept-Gated Visual Distillation wrapper for SimplerEnv.

    This wrapper intercepts observations from the environment and applies
    visual distillation to remove distractors while preserving task-relevant regions.

    Pipeline:
    1. Extract image and instruction from observation
    2. Parse instruction to identify target and anchor objects
    3. Segment distractors and safe-set using SAM3
    4. Compute final mask: distractors AND (NOT safe-set)
    5. Apply LaMa inpainting to remove distractors seamlessly
    6. Write distilled image back to observation

    The robot arm and gripper are ALWAYS included in the safe-set mask
    to prevent proprioception alignment issues.
    """

    def __init__(
        self,
        env: gym.Env,
        update_freq: int = 1,
        presence_threshold: float = 0.15,
        use_mock_segmenter: bool = False,
        use_server_segmenter: bool = False,
        segmenter_model: str = "facebook/sam3",
        include_robot: bool = True,
        verbose: bool = False,
        save_debug_images: bool = False,
        debug_dir: str = "cgvd_debug",
        distractor_names: Optional[List[str]] = None,
        cache_distractor_once: bool = False,
        robot_presence_threshold: float = 0.05,
        distractor_presence_threshold: float = 0.3,
        # Ablation parameters
        disable_safeset: bool = False,
        disable_inpaint: bool = False,
        # Legacy parameters (ignored, kept for backwards compatibility)
        use_inpaint: bool = True,
        **kwargs,
    ):
        """Initialize CGVD wrapper.

        Args:
            env: SimplerEnv environment to wrap
            update_freq: Frames between SAM3 segmentation updates (default 1 = every frame)
            presence_threshold: Threshold for safe-set (target/anchor) detection (default 0.15)
            use_mock_segmenter: Use mock segmenter for testing (default False)
            use_server_segmenter: Use SAM3 server for environments with transformers conflicts (default False)
            segmenter_model: SAM3 model identifier (default "facebook/sam3")
            include_robot: Always include robot arm/gripper in mask (default True)
            verbose: Print debug information (default False)
            save_debug_images: Save original/mask/distilled images for debugging (default False)
            debug_dir: Directory to save debug images (default "cgvd_debug")
            distractor_names: List of distractor object names to remove via inpainting.
            cache_distractor_once: Only compute distractor mask on first frame, reuse for episode.
                                   Trades accuracy for speed. Mask becomes stale as objects move.
            robot_presence_threshold: Threshold for robot arm detection (default 0.05).
                                      Robot is segmented separately with this lower threshold.
            distractor_presence_threshold: Threshold for distractor detection (default 0.3).
                                           Higher threshold reduces false positives.
            disable_safeset: Ablation flag - if True, skip safe-set subtraction (mask distractors
                             directly without protecting target/anchor). Default False.
            disable_inpaint: Ablation flag - if True, use mean-color fill instead of LaMa
                             inpainting. Default False.
            use_inpaint: Legacy parameter (ignored, use disable_inpaint instead).
            **kwargs: Additional legacy parameters (ignored).
        """
        super().__init__(env)

        # Configuration
        self.update_freq = update_freq
        self.presence_threshold = presence_threshold
        self.include_robot = include_robot
        self.verbose = verbose
        self.save_debug_images = save_debug_images
        self.debug_dir = debug_dir
        self.distractor_names = distractor_names or []
        self.cache_distractor_once = cache_distractor_once
        self.robot_presence_threshold = robot_presence_threshold
        self.distractor_presence_threshold = distractor_presence_threshold

        # Ablation flags
        self.disable_safeset = disable_safeset
        self.disable_inpaint = disable_inpaint

        # Initialize LaMa inpainter (unless disabled for ablation)
        if not self.disable_inpaint:
            from src.cgvd.lama_inpainter import LamaInpainter
            self.inpainter = LamaInpainter(device="cuda")
        else:
            self.inpainter = None
            if self.verbose:
                print("[CGVD] Ablation: Inpainting DISABLED (using mean-color fill)")

        # Create debug directory if needed
        if self.save_debug_images:
            os.makedirs(self.debug_dir, exist_ok=True)

        # Initialize components
        self.segmenter = create_segmenter(
            use_mock=use_mock_segmenter,
            use_server=use_server_segmenter,
            model_name=segmenter_model,
            presence_threshold=presence_threshold,
        )
        self.parser = InstructionParser()

        # State - masks
        self.cached_mask: Optional[np.ndarray] = None
        self.cached_distractor_mask: Optional[np.ndarray] = None  # Raw distractor mask before subtraction
        self.cached_safe_mask: Optional[np.ndarray] = None  # Safe set mask (target + anchor + robot)

        # State - confidence scores for debug display
        self.distractor_scores: Dict[str, float] = {}
        self.safe_scores: Dict[str, float] = {}

        # State - frame tracking
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
        self.cached_distractor_mask = None
        self.cached_safe_mask = None
        self.distractor_scores = {}
        self.safe_scores = {}
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

        if not self.distractor_names:
            # No distractors specified - return image unchanged
            if self.verbose and self.frame_count == 0:
                print("[CGVD] No distractors specified, passing through unchanged")
            self.frame_count += 1
            return obs

        # DISTRACTOR MODE with Safe-Set Protection
        # This mode segments distractors, then SUBTRACTS the safe set (target + anchor + robot)
        # to guarantee the target object is NEVER blurred, even if SAM3 confuses it with a distractor

        # Step 1: Get target/anchor from instruction (needed for safe set)
        instruction = self.env.unwrapped.get_language_instruction()
        if instruction != self.current_instruction:
            self.current_instruction = instruction
            self.current_target, self.current_anchor = self.parser.parse(instruction)
            if self.verbose:
                print(
                    f"[CGVD] Instruction: '{instruction}' -> "
                    f"target='{self.current_target}', anchor='{self.current_anchor}'"
                )

        # Step 2: Distractor mask - per-frame with STRICT threshold (0.3)
        # Stricter threshold reduces false positives (carrot detected as banana)
        seg_start = time.time()
        distractor_concepts = ". ".join(self.distractor_names)

        # Determine if we should recompute distractor mask
        should_recompute = (
            self.cached_distractor_mask is None or  # First frame
            (not self.cache_distractor_once and self.frame_count % self.update_freq == 0)
        )
        if should_recompute:
            distractor_mask = self.segmenter.segment(
                image, distractor_concepts, presence_threshold=self.distractor_presence_threshold
            )
            self.cached_distractor_mask = distractor_mask
            self.distractor_scores = self.segmenter.last_scores.copy()
            if self.verbose:
                print(f"[CGVD] Computed distractor mask (frame {self.frame_count}, threshold={self.distractor_presence_threshold})")
        else:
            distractor_mask = self.cached_distractor_mask

        # Step 3: Safe-set mask (skip if ablation flag set)
        # Target/anchor are cached once (stationary), robot is tracked every frame (moving)

        if self.disable_safeset:
            # Ablation: Skip safe-set protection entirely
            safe_mask = np.zeros_like(distractor_mask)
            self.cached_safe_mask = safe_mask  # Store for debug visualization
            if self.verbose and self.frame_count == 0:
                print("[CGVD] Ablation: Safe-set DISABLED (no target/anchor protection)")
        else:
            # Step 3a: Target + anchor mask - cached once on first frame
            if self.cached_safe_mask is None:
                safe_concepts = self.parser.build_concept_prompt(
                    self.current_target,
                    self.current_anchor,
                    include_robot=False,  # Robot tracked separately
                )
                raw_target_mask = self.segmenter.segment(
                    image, safe_concepts, presence_threshold=self.presence_threshold
                )
                self.safe_scores = self.segmenter.last_scores.copy()

                # Subtract distractor mask from safe-set to prevent false protection
                # This fixes semantic confusion where e.g. spatula/ladle get detected as "spoon"
                # NOTE: Reuse distractor_mask instead of calling segment() again for consistency
                if self.distractor_names and distractor_mask is not None:
                    # Remove any overlap: safe = target AND (NOT distractors)
                    self.cached_safe_mask = np.logical_and(
                        raw_target_mask > 0.5, distractor_mask < 0.5
                    ).astype(np.float32)
                    if self.verbose:
                        raw_cov = raw_target_mask.sum() / raw_target_mask.size * 100
                        dist_cov = distractor_mask.sum() / distractor_mask.size * 100
                        final_cov = self.cached_safe_mask.sum() / self.cached_safe_mask.size * 100
                        print(f"[CGVD] Safe-set: raw={raw_cov:.1f}%, distractor={dist_cov:.1f}%, final={final_cov:.1f}%")
                else:
                    self.cached_safe_mask = raw_target_mask
                    if self.verbose:
                        print(f"[CGVD] Cached target+anchor mask (frame {self.frame_count})")

            # Step 3b: Robot mask - tracked every frame
            # NOTE: Disabled for performance testing. Robot arm unlikely to be detected
            # as a distractor (fork/knife/etc), so safe-set subtraction may be unnecessary.
            # Uncomment to re-enable robot protection.
            # if self.include_robot:
            #     robot_concepts = "robot arm. robot gripper"
            #     robot_mask = self.segmenter.segment(
            #         image, robot_concepts, presence_threshold=self.robot_presence_threshold
            #     )
            #     self.safe_scores.update(self.segmenter.last_scores)
            #     # Combine cached target+anchor with fresh robot mask
            #     safe_mask = np.maximum(self.cached_safe_mask, robot_mask)
            #     if self.verbose:
            #         print(f"[CGVD] Robot mask (threshold={self.robot_presence_threshold}): {robot_mask.sum() / robot_mask.size * 100:.1f}%")
            # else:
            #     safe_mask = self.cached_safe_mask
            safe_mask = self.cached_safe_mask  # Skip robot tracking for now

        # Step 4: SUBTRACT safe set from distractors
        # final_mask = distractor AND (NOT safe)
        # This ensures target is NEVER blurred even if SAM3 confuses it with a distractor
        self.cached_mask = np.logical_and(
            distractor_mask > 0.5, safe_mask < 0.5
        ).astype(np.float32)

        seg_time = time.time() - seg_start
        if self.verbose:
            d_cov = distractor_mask.sum() / distractor_mask.size * 100
            s_cov = safe_mask.sum() / safe_mask.size * 100
            f_cov = self.cached_mask.sum() / self.cached_mask.size * 100
            print(
                f"[CGVD] Distractor: {d_cov:.1f}%, Safe: {s_cov:.1f}%, Final: {f_cov:.1f}%, seg_time={seg_time:.3f}s"
            )

        self.frame_count += 1

        # Apply visual distillation to remove distractors
        if self.disable_inpaint:
            # Ablation: Mean-color fill instead of inpainting
            distilled = self._apply_mean_fill(image, self.cached_mask)
        else:
            # Full CGVD: LaMa inpainting (per-frame)
            distilled = self.inpainter.inpaint(image, self.cached_mask)

        # Write distilled image back to observation
        # IMPORTANT: Write to the SAME camera we read from (camera_name)
        obs = self._write_image_to_obs(obs, distilled, camera_name)

        # Save debug images if enabled
        if self.save_debug_images:
            self._save_debug_images(image, self.cached_mask, distilled)

        return obs

    def _apply_mean_fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mean-color fill to masked regions (ablation alternative to inpainting).

        Args:
            image: Input image (H, W, 3) RGB
            mask: Binary mask (H, W) where 1 = fill region

        Returns:
            Image with masked regions filled with mean color of visible regions
        """
        mask_binary = mask > 0.5
        if not mask_binary.any():
            return image.copy()

        # Compute mean color of non-masked regions
        visible_mask = ~mask_binary
        if visible_mask.any():
            mean_color = image[visible_mask].mean(axis=0).astype(np.uint8)
        else:
            mean_color = np.array([128, 128, 128], dtype=np.uint8)

        # Fill masked regions
        result = image.copy()
        result[mask_binary] = mean_color
        return result

    def _save_debug_images(
        self, original: np.ndarray, mask: np.ndarray, distilled: np.ndarray
    ):
        """Save debug images showing original, masks, and distilled output.

        In distractor mode with safe-set protection, shows 5 columns:
        - Original | Distractors | Safe Set | Final (D-S) | Distilled

        In legacy mode, shows 3 columns:
        - Original | Foreground | Distilled
        """
        frame_num = self.frame_count - 1  # Already incremented
        h, w = original.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.distractor_names and self.cached_distractor_mask is not None:
            # 5-column layout for distractor mode with safe-set protection
            dist_vis = (self.cached_distractor_mask * 255).astype(np.uint8)
            dist_vis = cv2.cvtColor(dist_vis, cv2.COLOR_GRAY2RGB)

            safe_vis = (self.cached_safe_mask * 255).astype(np.uint8)
            safe_vis = cv2.cvtColor(safe_vis, cv2.COLOR_GRAY2RGB)

            final_vis = (mask * 255).astype(np.uint8)
            final_vis = cv2.cvtColor(final_vis, cv2.COLOR_GRAY2RGB)

            comparison = np.hstack([original, dist_vis, safe_vis, final_vis, distilled])

            # Labels with color coding
            cv2.putText(comparison, "Original", (10, 30), font, 0.5, (255, 255, 255), 1)
            cv2.putText(
                comparison, "Distractors", (w + 10, 30), font, 0.5, (255, 255, 255), 1
            )
            cv2.putText(
                comparison, "Safe Set", (2 * w + 10, 30), font, 0.5, (0, 255, 0), 1
            )
            cv2.putText(
                comparison, "Final (D-S)", (3 * w + 10, 30), font, 0.5, (255, 255, 0), 1
            )
            cv2.putText(
                comparison, "Distilled", (4 * w + 10, 30), font, 0.5, (255, 255, 255), 1
            )

            # Add coverage percentages
            d_cov = self.cached_distractor_mask.sum() / self.cached_distractor_mask.size * 100
            s_cov = self.cached_safe_mask.sum() / self.cached_safe_mask.size * 100
            f_cov = mask.sum() / mask.size * 100
            cv2.putText(
                comparison, f"{d_cov:.1f}%", (w + 10, 50), font, 0.4, (200, 200, 200), 1
            )
            cv2.putText(
                comparison, f"{s_cov:.1f}%", (2 * w + 10, 50), font, 0.4, (0, 200, 0), 1
            )
            cv2.putText(
                comparison, f"{f_cov:.1f}%", (3 * w + 10, 50), font, 0.4, (200, 200, 0), 1
            )

            # Show per-concept confidence scores under Distractors column
            y_offset = 70
            for concept, score in self.distractor_scores.items():
                color = (0, 255, 0) if score >= 0.3 else (255, 100, 100)
                cv2.putText(comparison, f"{concept}: {score:.2f}", (w + 10, y_offset), font, 0.35, color, 1)
                y_offset += 15

            # Show per-concept confidence scores under Safe Set column
            y_offset = 70
            for concept, score in self.safe_scores.items():
                color = (0, 255, 0) if score >= 0.15 else (255, 100, 100)
                cv2.putText(comparison, f"{concept}: {score:.2f}", (2 * w + 10, y_offset), font, 0.35, color, 1)
                y_offset += 15

            # Add target/anchor info
            if self.current_target:
                cv2.putText(
                    comparison,
                    f"Target: {self.current_target}",
                    (2 * w + 10, h - 25),
                    font,
                    0.4,
                    (0, 255, 0),
                    1,
                )
            if self.current_anchor:
                cv2.putText(
                    comparison,
                    f"Anchor: {self.current_anchor}",
                    (2 * w + 10, h - 10),
                    font,
                    0.4,
                    (0, 200, 0),
                    1,
                )

        else:
            # Original 3-column layout for legacy mode
            mask_vis = (mask * 255).astype(np.uint8)
            mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)

            comparison = np.hstack([original, mask_vis, distilled])

            # Add labels
            cv2.putText(comparison, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(
                comparison, "Foreground (white=keep)", (w + 10, 30), font, 0.5, (255, 255, 255), 2
            )
            cv2.putText(
                comparison, "Distilled (VLA input)", (2 * w + 10, 30), font, 0.7, (255, 255, 255), 2
            )

            # Add confidence scores if available
            if hasattr(self.segmenter, "last_scores") and self.segmenter.last_scores:
                y_offset = 55
                for concept, score in self.segmenter.last_scores.items():
                    # Color code: green if detected (>threshold), red if not
                    color = (
                        (0, 255, 0)
                        if score >= self.presence_threshold
                        else (255, 100, 100)
                    )
                    score_text = f"{concept}: {score:.2f}"
                    cv2.putText(
                        comparison, score_text, (w + 10, y_offset), font, 0.4, color, 1
                    )
                    y_offset += 18

        # Add instruction text (common to both modes)
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
            os.path.join(self.episode_debug_dir, f"frame_{frame_num:04d}.png"),
            comparison_bgr,
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
