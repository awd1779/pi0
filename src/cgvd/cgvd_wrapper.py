"""CGVD Wrapper - Main gym.Wrapper for Concept-Gated Visual Distillation."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from src.cgvd.instruction_parser import InstructionParser
from src.cgvd.sam3_segmenter import SAM3Segmenter, create_segmenter, get_sam3_segmenter


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
        cache_distractor_once: bool = True,
        robot_presence_threshold: float = 0.05,
        distractor_presence_threshold: float = 0.3,
        safeset_warmup_frames: int = 5,
        deferred_detection_frames: int = 10,
        # Compositing parameters
        blend_sigma: float = 3.0,
        lama_dilation: int = 11,
        safe_dilation: int = 5,
        cache_refresh_interval: int = 0,
        # Distractor IoU suppression
        distractor_iou_threshold: float = 0.15,
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
            safeset_warmup_frames: Number of frames to accumulate safe-set detections (default 5).
                                   During warm-up, safe-set is recomputed each frame and unioned
                                   with previous detections. This handles robot arm occlusion on
                                   early frames by allowing the target to be detected when visible.
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
        self.safeset_warmup_frames = safeset_warmup_frames
        self.deferred_detection_frames = deferred_detection_frames
        self.distractor_iou_threshold = distractor_iou_threshold

        # Compositing parameters
        self.blend_sigma = blend_sigma
        self.lama_dilation = lama_dilation
        self.safe_dilation = safe_dilation
        self._reinforce_size = self.safe_dilation + 2 * int(np.ceil(self.blend_sigma))
        self.cache_refresh_interval = cache_refresh_interval

        # Ablation flags
        self.disable_safeset = disable_safeset
        self.disable_inpaint = disable_inpaint

        # Initialize LaMa inpainter using singleton (unless disabled for ablation)
        # Using singleton avoids redundant model loading when multiple CGVDWrapper
        # instances are created (e.g., in batch evaluation)
        if not self.disable_inpaint:
            from src.cgvd.lama_inpainter import get_lama_inpainter
            self.inpainter = get_lama_inpainter(device="cuda")
        else:
            self.inpainter = None
            if self.verbose:
                print("[CGVD] Ablation: Inpainting DISABLED (using mean-color fill)")

        # Create debug directory if needed
        if self.save_debug_images:
            os.makedirs(self.debug_dir, exist_ok=True)
            self.log_file = open(os.path.join(self.debug_dir, "cgvd_log.txt"), "w")
        else:
            self.log_file = None

        # Initialize components using singleton for segmenter
        # This avoids redundant model loading across CGVDWrapper instances
        self.segmenter = get_sam3_segmenter(
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
        self.cached_inpainted_image: Optional[np.ndarray] = None  # Cached inpainted background for compositing
        self.last_robot_mask: Optional[np.ndarray] = None  # Store robot mask for compositing
        self.cached_robot_mask: Optional[np.ndarray] = None  # Accumulated robot mask from warmup
        self.last_distilled_image: Optional[np.ndarray] = None  # Previous frame's distilled output for robot detection
        self.current_safe_mask: Optional[np.ndarray] = None  # Per-frame safe mask (target + anchor + robot)
        self._target_detected_in_warmup: bool = False  # Whether the target specifically was detected during warmup
        self._safe_mask_votes: Optional[np.ndarray] = None  # Per-pixel detection count during warmup

        # State - confidence scores and individual masks for debug display
        self.distractor_scores: Dict[str, float] = {}
        self.safe_scores: Dict[str, float] = {}
        self.distractor_individual_masks: Dict[str, np.ndarray] = {}
        self.safe_individual_masks: Dict[str, np.ndarray] = {}

        # State - frame tracking
        self.frame_count: int = 0
        self.current_instruction: Optional[str] = None
        self.current_target: Optional[str] = None
        self.current_anchor: Optional[str] = None
        self.episode_count: int = 0
        self.episode_debug_dir: Optional[str] = None

        # Timing instrumentation
        self.last_cgvd_time: float = 0.0
        self.last_sam3_time: float = 0.0
        self.last_lama_time: float = 0.0
        self.total_cgvd_time: float = 0.0
        self.total_sam3_time: float = 0.0
        self.total_lama_time: float = 0.0

        # Compensate TimeLimit for warmup steps so VLA gets the full step budget.
        if self.distractor_names and self.safeset_warmup_frames > 0:
            import gymnasium
            current = self.env
            while hasattr(current, 'env'):
                if isinstance(current, gymnasium.wrappers.TimeLimit):
                    current._max_episode_steps += self.safeset_warmup_frames
                    if self.verbose:
                        print(f"[CGVD] Adjusted TimeLimit: +{self.safeset_warmup_frames} "
                              f"warmup steps (new max: {current._max_episode_steps})")
                    break
                current = current.env

    def _log(self, msg: str):
        """Log message to console (if verbose) and to file (if save_debug_images)."""
        if self.verbose:
            print(msg)
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()

    def _filter_overlapping_detections(
        self,
        masks: Dict[str, np.ndarray],
        scores: Dict[str, float],
        overlap_threshold: float = 0.5,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Keep only the highest confidence instance per concept.

        When SAM3 detects multiple instances (e.g., 'spoon_0', 'spoon_1'),
        keep only the one with highest confidence for each base concept.
        This filters out misidentified objects (e.g., spatula detected as spoon
        but with lower confidence).

        Args:
            masks: Dict mapping concept name to mask array (e.g., 'spoon_0', 'spoon_1', 'towel')
            scores: Dict mapping concept name to confidence score
            overlap_threshold: Unused (kept for API compatibility)

        Returns:
            Filtered (masks, scores) dicts with only top-scoring instance per concept
        """
        if len(masks) <= 1:
            return masks, scores

        # Group by base concept name (strip _0, _1, etc.)
        from collections import defaultdict
        concept_groups = defaultdict(list)
        for name, score in scores.items():
            # Extract base name (e.g., 'spoon_0' -> 'spoon', 'towel' -> 'towel')
            base_name = name.rsplit('_', 1)[0] if '_' in name and name.rsplit('_', 1)[1].isdigit() else name
            concept_groups[base_name].append((name, score, masks[name]))

        kept_masks = {}
        kept_scores = {}

        for base_name, instances in concept_groups.items():
            # Sort by score descending, keep top one
            instances.sort(key=lambda x: x[1], reverse=True)
            top_name, top_score, top_mask = instances[0]

            kept_masks[top_name] = top_mask
            kept_scores[top_name] = top_score

            # Log filtered instances
            for name, score, _ in instances[1:]:
                self._log(f"[CGVD] Filtering '{name}' (score={score:.3f}) - keeping '{top_name}' (score={top_score:.3f})")

        return kept_masks, kept_scores

    def _suppress_overlapping_distractors(
        self,
        distractor_masks: Dict[str, np.ndarray],
        distractor_scores: Dict[str, float],
        safe_mask: np.ndarray,
        iou_threshold: float = 0.15,
    ) -> Dict[str, np.ndarray]:
        """Suppress distractor detections that overlap significantly with the safe-set."""
        filtered = {}
        for concept, mask in distractor_masks.items():
            if mask.sum() == 0:
                continue
            intersection = np.logical_and(mask > 0.5, safe_mask > 0.5).sum()
            mask_area = (mask > 0.5).sum()
            if mask_area == 0:
                continue
            overlap = intersection / mask_area
            if overlap > iou_threshold:
                self._log(f"[CGVD] Suppressing distractor '{concept}' "
                          f"(score={distractor_scores.get(concept, 0):.3f}, "
                          f"overlap={overlap:.3f} > {iou_threshold})")
            else:
                filtered[concept] = mask
        return filtered

    def _cross_validate_safeset(
        self,
        safe_masks: Dict[str, np.ndarray],
        safe_scores: Dict[str, float],
        distractor_masks: Dict[str, np.ndarray],
        distractor_scores: Dict[str, float],
    ) -> Optional[np.ndarray]:
        """Score-aware cross-validation for safe-set instances.

        For each TARGET instance, compute genuineness = safe_score - max_overlapping_dist_score.
        Remove instances with negative genuineness (they're distractors detected as target).
        Always keep the most genuine instance (highest genuineness score).
        Anchor instances are never filtered.

        Args:
            safe_masks: Individual safe-set masks {concept_name: mask}
            safe_scores: Confidence scores for safe-set detections
            distractor_masks: Individual distractor masks {concept_name: mask}
            distractor_scores: Confidence scores for distractor detections

        Returns:
            Combined mask of false-positive safe regions to subtract (H, W) float32,
            or None if no false positives found.
        """
        if not safe_masks or not distractor_masks:
            return None

        # Compute genuineness for each target instance
        target_genuineness = {}  # {safe_name: (genuineness, safe_mask, max_dist_name, max_dist_score)}

        for safe_name, safe_mask in safe_masks.items():
            base = safe_name.rsplit('_', 1)[0] if '_' in safe_name and safe_name.rsplit('_', 1)[1].isdigit() else safe_name

            # Only filter target instances, skip anchor
            if base != self.current_target:
                continue

            safe_score = safe_scores.get(safe_name, 0)
            safe_area = (safe_mask > 0.5).sum()
            if safe_area == 0:
                continue

            # Find max overlapping distractor score
            max_dist_score = 0.0
            max_dist_name = None
            for dist_name, dist_mask in distractor_masks.items():
                intersection = np.logical_and(safe_mask > 0.5, dist_mask > 0.5).sum()
                union = np.logical_or(safe_mask > 0.5, dist_mask > 0.5).sum()
                if union == 0:
                    continue
                iou = intersection / union
                if iou > 0.3:
                    dist_score = distractor_scores.get(dist_name, 0)
                    if dist_score > max_dist_score:
                        max_dist_score = dist_score
                        max_dist_name = dist_name

            genuineness = safe_score - max_dist_score
            target_genuineness[safe_name] = (genuineness, safe_mask, max_dist_name, max_dist_score)
            self._log(f"[CGVD] Cross-val: '{safe_name}' (score={safe_score:.3f}) "
                      f"genuineness={genuineness:.3f} "
                      f"(best distractor overlap: '{max_dist_name}', score={max_dist_score:.3f})")

        if not target_genuineness:
            return None

        # Find the most genuine instance — ALWAYS keep this one
        best_name = max(target_genuineness, key=lambda k: target_genuineness[k][0])

        # Build false-positive mask from non-genuine instances
        false_positive_mask = None
        for name, (genuineness, mask, dist_name, dist_score) in target_genuineness.items():
            if name == best_name:
                self._log(f"[CGVD] Cross-val: KEEPING '{name}' (genuineness={genuineness:.3f})")
                continue
            if genuineness < 0:  # More "distractor" than "target"
                self._log(f"[CGVD] Cross-val: removing '{name}' (genuineness={genuineness:.3f})")
                if false_positive_mask is None:
                    false_positive_mask = mask.copy()
                else:
                    false_positive_mask = np.maximum(false_positive_mask, mask)

        return false_positive_mask

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
        self.cached_inpainted_image = None
        self.last_robot_mask = None
        self.cached_robot_mask = None
        self.last_distilled_image = None
        self.current_safe_mask = None
        self._target_detected_in_warmup = False
        self._safe_mask_votes = None
        self.distractor_scores = {}
        self.safe_scores = {}
        self.frame_count = 0
        self.current_instruction = None
        self.current_target = None
        self.current_anchor = None

        # Reset timing accumulators
        self.last_cgvd_time = 0.0
        self.last_sam3_time = 0.0
        self.last_lama_time = 0.0
        self.total_cgvd_time = 0.0
        self.total_sam3_time = 0.0
        self.total_lama_time = 0.0

        # Create episode-specific debug directory
        if self.save_debug_images:
            self.episode_debug_dir = os.path.join(
                self.debug_dir, f"episode_{self.episode_count:03d}"
            )
            os.makedirs(self.episode_debug_dir, exist_ok=True)
        self.episode_count += 1

        # Internal warmup: step the env with no-op actions while accumulating
        # SAM3 masks. The VLA never sees these frames — it only receives the
        # first fully-distilled image after warmup completes.
        # No-op (np.zeros(7)) = hold position, no rotation, gripper unchanged.
        if self.distractor_names and self.safeset_warmup_frames > 0:
            for i in range(self.safeset_warmup_frames):
                self._apply_cgvd(obs)  # accumulate masks, skip compositing
                obs, _, _, _, _ = self.env.step(np.zeros(7))  # no-op step

        # First post-warmup frame: inpaint + composite (or just apply if no distractors)
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
        cgvd_start = time.time()

        # Stage 0: Extract image from SimplerEnv
        # IMPORTANT: Get camera name to ensure we write back to the same camera
        image, camera_name = self._get_image_and_camera(obs)

        if not self.distractor_names:
            # No distractors specified - return image unchanged
            if self.verbose and self.frame_count == 0:
                print("[CGVD] No distractors specified, passing through unchanged")
            self.frame_count += 1
            self.last_cgvd_time = time.time() - cgvd_start
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

        # Step 2: Distractor mask - accumulate during warm-up period
        # Stricter threshold reduces false positives (carrot detected as banana)
        seg_start = time.time()
        distractor_concepts = ". ".join(self.distractor_names)

        # Determine if we should recompute distractor mask
        in_warmup = self.frame_count < self.safeset_warmup_frames
        should_recompute = (
            self.cached_distractor_mask is None or  # First frame
            in_warmup or  # Warm-up period: accumulate detections
            (not self.cache_distractor_once and self.frame_count % self.update_freq == 0)
        )
        if should_recompute:
            raw_distractor_mask = self.segmenter.segment(
                image, distractor_concepts, presence_threshold=self.distractor_presence_threshold
            )
            self.distractor_scores = self.segmenter.last_scores.copy()
            self.distractor_individual_masks = self.segmenter.last_individual_masks.copy()

            # Log distractor scores
            scores_str = ", ".join(f"{k}={v:.3f}" for k, v in self.distractor_scores.items())
            self._log(f"[CGVD] Frame {self.frame_count} Distractor scores: {scores_str}")

            if self.cached_distractor_mask is None:
                # First frame: initialize
                self.cached_distractor_mask = raw_distractor_mask
            elif in_warmup:
                # Warm-up frames: accumulate (union of all detections)
                self.cached_distractor_mask = np.maximum(self.cached_distractor_mask, raw_distractor_mask)
            else:
                # Post-warmup recompute (when cache_distractor_once=False)
                self.cached_distractor_mask = raw_distractor_mask

            if self.verbose:
                cov = self.cached_distractor_mask.sum() / self.cached_distractor_mask.size * 100
                status = "accumulating" if in_warmup else "frozen"
                print(f"[CGVD] Distractor mask: {cov:.1f}% (frame {self.frame_count}, {status})")

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
            # Step 3a: Target + anchor mask - accumulate during warm-up period
            in_warmup = self.frame_count < self.safeset_warmup_frames
            in_deferred = (
                not self._target_detected_in_warmup
                and self.frame_count >= self.safeset_warmup_frames
                and self.frame_count < self.safeset_warmup_frames + self.deferred_detection_frames
            )

            if self.cached_safe_mask is None or in_warmup or in_deferred:
                safe_concepts = self.parser.build_concept_prompt(
                    self.current_target,
                    self.current_anchor,
                    include_robot=False,  # Robot tracked separately
                )
                raw_target_mask = self.segmenter.segment(
                    image, safe_concepts, presence_threshold=self.presence_threshold
                )
                self.safe_scores = self.segmenter.last_scores.copy()
                self.safe_individual_masks = self.segmenter.last_individual_masks.copy()

                # Keep only top-scoring instance per concept (e.g., highest-scoring "spoon").
                # The real target typically scores higher than false positives (e.g., spatula
                # misdetected as "spoon" with 0.72 vs real spoon at 0.87). Top-1 filtering
                # prevents the false positive from entering the safe-set, which would create
                # a hole in the distractor mask where the spatula should be inpainted away.
                # Combined with majority voting across warmup frames, this is robust even
                # if top-1 occasionally picks the wrong instance on a single frame.
                filtered_masks, filtered_scores = self._filter_overlapping_detections(
                    self.safe_individual_masks, self.safe_scores
                )
                raw_target_mask = np.zeros_like(raw_target_mask)
                for m in filtered_masks.values():
                    raw_target_mask = np.maximum(raw_target_mask, m)
                self.safe_individual_masks = filtered_masks
                self.safe_scores = filtered_scores

                # Always log scores to file (if enabled), print only if verbose
                scores_str = ", ".join(f"{k}={v:.3f}" for k, v in self.safe_scores.items())
                self._log(f"[CGVD] Frame {self.frame_count} Safe-set scores (filtered): {scores_str}")

                # Track whether the target specifically was detected during warmup.
                # Keys can be "spoon" (single) or "spoon_0", "spoon_1" (multi-instance).
                if not self._target_detected_in_warmup and self.current_target:
                    for key, score in self.safe_scores.items():
                        base = key.rsplit('_', 1)[0] if '_' in key and key.rsplit('_', 1)[1].isdigit() else key
                        if base == self.current_target and score >= self.presence_threshold:
                            self._target_detected_in_warmup = True
                            phase = "warmup" if in_warmup else "deferred detection"
                            self._log(f"[CGVD] Target '{self.current_target}' detected in {phase} (key='{key}', score={score:.3f})")
                            break

                if self.cached_safe_mask is None:
                    # First frame: initialize
                    self.cached_safe_mask = raw_target_mask
                    self._safe_mask_votes = (raw_target_mask > 0.5).astype(np.float32)
                else:
                    # Warm-up frames: accumulate (union of all detections)
                    self.cached_safe_mask = np.maximum(self.cached_safe_mask, raw_target_mask)
                    if self._safe_mask_votes is None:
                        self._safe_mask_votes = (raw_target_mask > 0.5).astype(np.float32)
                    else:
                        self._safe_mask_votes += (raw_target_mask > 0.5).astype(np.float32)

                # Cross-validate for logging only (keep diagnostic info).
                # Top-1 filtering handles false positive rejection.
                fp_mask = self._cross_validate_safeset(
                    self.safe_individual_masks, self.safe_scores,
                    self.distractor_individual_masks, self.distractor_scores,
                )

                # On last warmup frame, log vote statistics for diagnostics.
                # Safe-set uses union accumulation (np.maximum) — robust to SAM3
                # detection gaps. Top-1 filtering handles false positive rejection.
                is_last_warmup = in_warmup and (self.frame_count == self.safeset_warmup_frames - 1)
                if is_last_warmup and self._safe_mask_votes is not None:
                    union_pixels = int((self.cached_safe_mask > 0.5).sum())
                    max_votes = int(self._safe_mask_votes.max())
                    self._log(f"[CGVD] Safe-set finalized: {union_pixels} pixels "
                              f"(union of {self.safeset_warmup_frames} frames, max_votes={max_votes})")

                if self.verbose:
                    cov = self.cached_safe_mask.sum() / self.cached_safe_mask.size * 100
                    status = "accumulating" if in_warmup else ("deferred" if in_deferred else "frozen")
                    print(f"[CGVD] Safe-set mask: {cov:.1f}% (frame {self.frame_count}, {status})")

            # Step 3b: Robot mask - tracked every frame
            if self.include_robot:
                robot_concepts = "robot arm. robot gripper"
                robot_image = self.last_distilled_image if self.last_distilled_image is not None else image
                robot_mask = self.segmenter.segment(
                    robot_image, robot_concepts, presence_threshold=self.robot_presence_threshold
                )
                self.safe_scores.update(self.segmenter.last_scores)
                self.last_robot_mask = robot_mask  # Store for compositing
                if in_warmup:
                    if self.cached_robot_mask is None:
                        self.cached_robot_mask = robot_mask.copy()
                    else:
                        self.cached_robot_mask = np.maximum(self.cached_robot_mask, robot_mask)
                # Combine cached target+anchor with fresh robot mask
                safe_mask = np.maximum(self.cached_safe_mask, robot_mask)
                if self.verbose:
                    print(f"[CGVD] Robot mask (threshold={self.robot_presence_threshold}): {robot_mask.sum() / robot_mask.size * 100:.1f}%")
            else:
                safe_mask = self.cached_safe_mask

            # Note: Pre-suppression of overlapping distractors removed.
            # The final mask formula (distractor AND NOT safe) at Step 4 already
            # gates distractors by the safe-set. Pre-suppression was removing
            # distractors from the mask before they reached the final gating,
            # causing them to show through untouched.

        self.current_safe_mask = safe_mask

        # Dilate safe-set mask to create protective buffer for Step 4 gating.
        # Uses cached_safe_mask (target+anchor only) instead of safe_mask
        # (which includes the robot). This makes cached_mask stable across
        # frames — robot visibility is handled in _composite() via re-enforcement.
        if self.safe_dilation > 0:
            safe_dilation_kernel = np.ones((self.safe_dilation, self.safe_dilation), np.uint8)
            safe_mask_for_gating = cv2.dilate(
                (self.cached_safe_mask > 0.5).astype(np.uint8), safe_dilation_kernel, iterations=1
            ).astype(np.float32)
        else:
            safe_mask_for_gating = (self.cached_safe_mask > 0.5).astype(np.float32)

        # Step 3.5: Dilate distractor mask BEFORE safe-set subtraction
        # This ensures dilation is gated by the safe-set (task objects never bleed in)
        if self.lama_dilation > 0 and not self.disable_inpaint:
            dilation_kernel = np.ones((self.lama_dilation, self.lama_dilation), np.uint8)
            distractor_mask = cv2.dilate(
                (distractor_mask > 0.5).astype(np.uint8), dilation_kernel, iterations=1
            ).astype(np.float32)

        # Step 4: SUBTRACT safe set from (dilated) distractors
        # final_mask = distractor AND (NOT safe)
        # This ensures target is NEVER inpainted even if SAM3 confuses it with a distractor
        # Uses dilated safe mask to provide protective buffer around target edges
        self.cached_mask = np.logical_and(
            distractor_mask > 0.5, safe_mask_for_gating < 0.5
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

        # During warmup: masks are accumulated above, but skip compositing.
        # The VLA never sees warmup frames — they are consumed internally by
        # reset() which steps the env with no-op actions.
        if self.frame_count <= self.safeset_warmup_frames:
            self.last_cgvd_time = time.time() - cgvd_start
            self.total_cgvd_time += self.last_cgvd_time
            return obs

        # Deferred target detection: if the target wasn't detected during warmup
        # (likely occluded by robot gripper), continue compositing anyway and keep
        # trying to detect it in post-warmup frames. This is safe because:
        # - SAM3 never segmented the occluded target as a distractor → not in cached_mask
        # - The target shows through the composite naturally at mask=0 pixels
        # - When the robot moves and reveals the target, deferred detection adds it
        #   to cached_safe_mask for explicit protection
        if (self.frame_count == self.safeset_warmup_frames + 1
                and not self._target_detected_in_warmup):
            print(
                f"[CGVD] WARNING: Target '{self.current_target}' not detected during warmup "
                f"(likely occluded by robot). Continuing compositing with deferred target detection "
                f"for {self.deferred_detection_frames} frames."
            )

        # Apply visual distillation to remove distractors
        if self.disable_inpaint:
            # Ablation: Mean-color fill instead of inpainting
            distilled = self._apply_mean_fill(image, self.cached_mask)
        elif self.cached_inpainted_image is None:
            # First frame: clean-plate: inpaint distractors + robot + safe-set
            cache_mask = self._build_inpaint_mask()
            self.cached_inpainted_image = self.inpainter.inpaint(image, cache_mask, dilate_mask=0)
            # For frame 0: composite non-distractor regions from current frame
            # This shows current frame everywhere EXCEPT distractor regions (which show inpainted background)
            distilled = self._composite(image, self.cached_inpainted_image, self.cached_mask)
            if self.verbose:
                print(f"[CGVD] Cached inpainted background (frame {self.frame_count})")
        else:
            # Periodic refresh of cached background (disabled by default,
            # cached_mask is stable so refresh only causes visual jumps)
            if (self.cache_refresh_interval > 0 and
                    self.frame_count % self.cache_refresh_interval == 0):
                cache_mask = self._build_inpaint_mask()
                self.cached_inpainted_image = self.inpainter.inpaint(image, cache_mask, dilate_mask=0)
                if self.verbose:
                    print(f"[CGVD] Cache refresh at frame {self.frame_count}")

            # Composite non-distractor regions from current frame
            # - Distractor regions: show cached inpainted background (distractors removed)
            # - Non-distractor regions: show current frame (robot + target move naturally)
            distilled = self._composite(image, self.cached_inpainted_image, self.cached_mask)
            if self.verbose:
                print(f"[CGVD] Using cached inpainting with scene composite (frame {self.frame_count})")

        # Write distilled image back to observation
        # IMPORTANT: Write to the SAME camera we read from (camera_name)
        obs = self._write_image_to_obs(obs, distilled, camera_name)
        self.last_distilled_image = distilled.copy()

        # Save debug images if enabled
        if self.save_debug_images:
            self._save_debug_images(image, self.cached_mask, distilled)

        # Update timing stats
        self.last_cgvd_time = time.time() - cgvd_start
        self.total_cgvd_time += self.last_cgvd_time

        # Get component timing from segmenter and inpainter
        if hasattr(self.segmenter, 'last_segment_time'):
            self.last_sam3_time = self.segmenter.last_segment_time
            self.total_sam3_time += self.last_sam3_time
        if self.inpainter is not None and hasattr(self.inpainter, 'last_inpaint_time'):
            self.last_lama_time = self.inpainter.last_inpaint_time
            self.total_lama_time += self.last_lama_time

        return obs

    def _composite(self, image: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Composite inpainted background with live frame using feathered blending.

        Args:
            image: Live camera frame (H, W, 3) uint8
            inpainted: Cached inpainted background (H, W, 3) uint8
            mask: Binary mask (H, W) float32 where 1 = distractor region

        Returns:
            Composited image (H, W, 3) uint8
        """
        if self.blend_sigma > 0:
            feathered = cv2.GaussianBlur(mask, (0, 0), sigmaX=self.blend_sigma, sigmaY=self.blend_sigma)

            safe = self.current_safe_mask if self.current_safe_mask is not None else self.cached_safe_mask

            # Defense-in-depth: binarize safe mask so all gating uses clean 0/1 values.
            # Prevents soft SAM3 values from leaking through Mechanism 2 or re-enforcement.
            if safe is not None:
                safe = (safe > 0.5).astype(np.float32)

            # Binary target/anchor mask — immune to soft-value leakage from SAM3.
            # Stationary objects with crisp boundaries get hard 0/1 protection.
            binary_target = (
                (self.cached_safe_mask > 0.5).astype(np.float32)
                if self.cached_safe_mask is not None
                else np.zeros(feathered.shape, dtype=np.float32)
            )

            # Dilate binary_target beyond cached_mask boundary by ~2σ of
            # blend_sigma so GaussianBlur feathered values are negligible (<2%)
            # at the re-enforcement edge, eliminating table-color outline.
            if self.safe_dilation > 0 and binary_target.max() > 0:
                _target_kern = np.ones((self._reinforce_size, self._reinforce_size), np.uint8)
                binary_target = cv2.dilate(
                    binary_target.astype(np.uint8), _target_kern, iterations=1
                ).astype(np.float32)

            # Mechanism 2: Clamp feathered at distractor pixels outside target.
            # With robot decoupled from cached_mask, we only gate by binary_target
            # (not safe, which includes robot). This ensures distractor pixels
            # always show inpainted background regardless of robot position.
            # Binarize cached_distractor_mask so soft SAM3 edge values (0.6-0.7)
            # clamp feathered to exactly 1.0, not a partial value that leaks
            # 30-40% of the live frame (with distractor) through.
            if self.cached_distractor_mask is not None:
                binary_distractor = (self.cached_distractor_mask > 0.5).astype(np.float32)
                if binary_target is not None:
                    non_safe_distractor = binary_distractor * (1.0 - binary_target)
                else:
                    non_safe_distractor = binary_distractor
                feathered = np.maximum(feathered, non_safe_distractor)

            # Re-enforce safe-set pixels so they always show the live frame.
            # Binarize robot contribution (>0.5) to prevent SAM3 boundary fuzz
            # from partially zeroing feathered at nearby distractor pixels.
            # This gives the robot a crisp boundary that cleanly punches through
            # distractor regions without creating an indeterminate halo.
            if safe is not None:
                robot_contrib = np.clip(safe - binary_target, 0.0, 1.0)
                robot_binary = (robot_contrib > 0.5).astype(np.float32)
                # Dilate robot re-enforcement beyond cached_mask boundary by ~2σ of
                # blend_sigma so GaussianBlur feathered values are negligible (<2%)
                # at the re-enforcement edge, eliminating robot-color halo.
                if self.safe_dilation > 0 and robot_binary.max() > 0:
                    _robot_kern = np.ones((self._reinforce_size, self._reinforce_size), np.uint8)
                    robot_binary_dilated = cv2.dilate(
                        robot_binary.astype(np.uint8), _robot_kern, iterations=1
                    ).astype(np.float32)
                else:
                    robot_binary_dilated = robot_binary

                # In distractor zones: use raw binarized robot mask.
                # Previously used eroded mask to strip SAM3 boundary false positives,
                # but the ~2px erosion gap lets Mechanism 2 force feathered=1.0 at the
                # robot boundary, showing clean plate (table texture) instead of the
                # live robot — creating a visible halo. Raw binarized mask (thresholded
                # at >0.5) is sufficient to filter low-confidence SAM3 boundary pixels.
                if self.cached_distractor_mask is not None:
                    distractor_zone = (self.cached_distractor_mask > 0.5).astype(np.float32)
                    non_distractor = 1.0 - distractor_zone

                    # Outside distractors: dilated (halo elimination)
                    # Inside distractors: raw binarized (no erosion gap)
                    robot_binary_dilated = (
                        robot_binary_dilated * non_distractor
                        + robot_binary * distractor_zone
                    )

                reinforce_mask = np.maximum(binary_target, robot_binary_dilated)
                feathered = feathered * (1.0 - reinforce_mask)

            feathered_3d = feathered[..., None]
            return (feathered_3d * inpainted.astype(np.float32) +
                    (1.0 - feathered_3d) * image.astype(np.float32)).astype(np.uint8)
        else:
            # Hard compositing (original behavior, sigma=0)
            mask_3d = mask[..., None] > 0.5
            return np.where(mask_3d, inpainted, image)

    def _build_inpaint_mask(self) -> np.ndarray:
        """Build inpainting mask: distractors + robot.

        Does NOT include safe-set (target/anchor) — keeping them in the
        inpainted background ensures the spoon is visible even when
        feathered blending is imperfect near distractors.  Accepting a
        minor ghost at the old position after pickup is far less harmful
        than the spoon disappearing during approach.
        """
        mask = self.cached_mask.copy()
        if self.include_robot and self.last_robot_mask is not None:
            robot = self.cached_robot_mask if self.cached_robot_mask is not None else self.last_robot_mask
            # Dilate robot mask to cover GaussianBlur spread (~2σ beyond
            # cached_mask boundary) for the clean plate. Without this, the
            # gap between undilated robot mask and dilated-safe hole in
            # cached_mask leaves un-inpainted pixels that retain stale
            # robot arm color after the arm moves away.
            if self.safe_dilation > 0:
                kern = np.ones((self._reinforce_size, self._reinforce_size), np.uint8)
                robot = cv2.dilate(
                    (robot > 0.5).astype(np.uint8), kern, iterations=1
                ).astype(np.float32)
            mask = np.maximum(mask, robot)
        return mask

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

            # Seam heatmap: absolute pixel difference between distilled and original
            seam_diff = np.abs(distilled.astype(np.float32) - original.astype(np.float32)).mean(axis=2)
            seam_vis = (np.clip(seam_diff * 3, 0, 255)).astype(np.uint8)
            seam_vis_rgb = cv2.applyColorMap(seam_vis, cv2.COLORMAP_JET)
            seam_vis_rgb = cv2.cvtColor(seam_vis_rgb, cv2.COLOR_BGR2RGB)

            # Draw compositing boundary contours on Distilled column
            distilled_annotated = distilled.copy()
            contours, _ = cv2.findContours(
                (mask > 0.5).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(distilled_annotated, contours, -1, (255, 0, 0), 1)

            comparison = np.hstack([original, dist_vis, safe_vis, final_vis, distilled_annotated, seam_vis_rgb])

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
            cv2.putText(
                comparison, "Seam Diff", (5 * w + 10, 30), font, 0.5, (255, 255, 255), 1
            )

            # Warn if target object not detected in safe-set
            if self.current_target:
                target_detected = any(
                    self.current_target in k and v >= self.presence_threshold
                    for k, v in self.safe_scores.items()
                )
                if not target_detected:
                    cv2.putText(
                        comparison, "WARNING: TARGET NOT IN SAFE SET",
                        (10, h - 40), font, 0.6, (0, 0, 255), 2,
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

            # Overlay labels on mask regions for Distractors column
            for concept, score in self.distractor_scores.items():
                if concept in self.distractor_individual_masks:
                    mask = self.distractor_individual_masks[concept]
                    if mask.sum() > 0:
                        # Find centroid of mask
                        ys, xs = np.where(mask > 0.5)
                        cx, cy = int(xs.mean()), int(ys.mean())
                        # Draw marker and label on Distractors column (offset by w)
                        color = (0, 255, 0) if score >= 0.3 else (0, 0, 255)
                        label = f"{concept}: {score:.2f}"
                        # Draw circle at centroid
                        cv2.circle(comparison, (w + cx, cy), 5, color, -1)
                        # Draw text with background for visibility
                        (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
                        cv2.rectangle(comparison, (w + cx - 2, cy - th - 4), (w + cx + tw + 2, cy + 4), (0, 0, 0), -1)
                        cv2.putText(comparison, label, (w + cx, cy), font, 0.4, color, 1)

            # Overlay labels on mask regions for Safe Set column
            for concept, score in self.safe_scores.items():
                if concept in self.safe_individual_masks:
                    mask = self.safe_individual_masks[concept]
                    if mask.sum() > 0:
                        # Find centroid of mask
                        ys, xs = np.where(mask > 0.5)
                        cx, cy = int(xs.mean()), int(ys.mean())
                        # Draw marker and label on Safe Set column (offset by 2*w)
                        color = (0, 255, 0) if score >= 0.15 else (0, 0, 255)
                        label = f"{concept}: {score:.2f}"
                        # Draw circle at centroid
                        cv2.circle(comparison, (2 * w + cx, cy), 5, color, -1)
                        # Draw text with background for visibility
                        (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
                        cv2.rectangle(comparison, (2 * w + cx - 2, cy - th - 4), (2 * w + cx + tw + 2, cy + 4), (0, 0, 0), -1)
                        cv2.putText(comparison, label, (2 * w + cx, cy), font, 0.4, color, 1)

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

    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics for CGVD pipeline.

        Returns:
            Dict with timing data:
            - last_cgvd_time: Time for most recent CGVD call
            - last_sam3_time: Time for most recent SAM3 segmentation
            - last_lama_time: Time for most recent LaMa inpainting
            - total_cgvd_time: Accumulated CGVD time this episode
            - total_sam3_time: Accumulated SAM3 time this episode
            - total_lama_time: Accumulated LaMa time this episode
            - avg_cgvd_time: Average CGVD time per frame
            - avg_sam3_time: Average SAM3 time per frame
            - avg_lama_time: Average LaMa time per frame
        """
        frame_count = max(1, self.frame_count)
        return {
            "last_cgvd_time": self.last_cgvd_time,
            "last_sam3_time": self.last_sam3_time,
            "last_lama_time": self.last_lama_time,
            "total_cgvd_time": self.total_cgvd_time,
            "total_sam3_time": self.total_sam3_time,
            "total_lama_time": self.total_lama_time,
            "avg_cgvd_time": self.total_cgvd_time / frame_count,
            "avg_sam3_time": self.total_sam3_time / frame_count,
            "avg_lama_time": self.total_lama_time / frame_count,
        }
