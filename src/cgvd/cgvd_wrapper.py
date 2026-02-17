"""CGVD Wrapper - Main gym.Wrapper for Concept-Gated Visual Distillation."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from src.cgvd.instruction_parser import InstructionParser
from src.cgvd.sam3_segmenter import SAM3Segmenter, get_sam3_segmenter


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
        safeset_warmup_frames: int = 1,
        # Compositing parameters
        blend_sigma: float = 3.0,
        lama_dilation: int = 11,
        safe_dilation: int = 5,
        cache_refresh_interval: int = 0,
        # Safe-set robustness parameters
        genuineness_margin: float = -0.1,
        iou_gate_threshold: float = 0.15,
        iou_gate_start_frame: int = 2,
        min_component_pixels: int = 50,
        overlap_penalty_cap: float = 0.7,
        # Legacy (kept so callers don't break)
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

        if kwargs:
            print(f"[CGVD] WARNING: Unknown parameters ignored: {list(kwargs.keys())}")

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
        # Safe-set robustness parameters
        self.genuineness_margin = genuineness_margin
        self.iou_gate_threshold = iou_gate_threshold
        self.iou_gate_start_frame = iou_gate_start_frame
        self.min_component_pixels = min_component_pixels
        self.overlap_penalty_cap = overlap_penalty_cap
        # Compositing parameters
        self.blend_sigma = blend_sigma
        self.lama_dilation = lama_dilation
        self.safe_dilation = safe_dilation
        self._step4_safe_dilation = max(self.safe_dilation, self.lama_dilation)
        self._reinforce_size = self._step4_safe_dilation + 3 * int(np.ceil(self.blend_sigma))
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
        self.cached_compositing_mask: Optional[np.ndarray] = None  # Undilated distractor mask for compositing (no lama_dilation)
        self.cached_distractor_mask: Optional[np.ndarray] = None  # Raw distractor mask before subtraction
        self.cached_safe_mask: Optional[np.ndarray] = None  # Combined safe set (derived: max(target, anchor))
        self.cached_target_mask: Optional[np.ndarray] = None  # Target only (e.g., spoon)
        self.cached_anchor_mask: Optional[np.ndarray] = None  # Anchor only (e.g., towel)
        self.cached_inpainted_image: Optional[np.ndarray] = None  # Cached inpainted background for compositing
        self.last_robot_mask: Optional[np.ndarray] = None  # Store robot mask for compositing
        self.cached_robot_mask: Optional[np.ndarray] = None  # Accumulated robot mask from warmup
        self.current_safe_mask: Optional[np.ndarray] = None  # Per-frame safe mask (target + anchor + robot)
        self._target_votes: Optional[np.ndarray] = None  # Per-pixel vote count for target
        self._anchor_votes: Optional[np.ndarray] = None  # Per-pixel vote count for anchor
        self._instance_genuineness: Dict[str, float] = {}  # Per-instance genuineness scores from cross-val

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


    def _log(self, msg: str):
        """Log message to console (if verbose) and to file (if save_debug_images)."""
        if self.verbose:
            print(msg)
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()

    def _hide_robot(self):
        """Hide robot visual meshes so SAM3 sees unoccluded scene."""
        try:
            robot = self.env.unwrapped.agent.robot
            for link in robot.get_links():
                link.hide_visual()
        except Exception as e:
            self._log(f"[CGVD] Warning: could not hide robot: {e}")

    def _show_robot(self):
        """Restore robot visual meshes."""
        try:
            robot = self.env.unwrapped.agent.robot
            for link in robot.get_links():
                link.unhide_visual()
        except Exception as e:
            self._log(f"[CGVD] Warning: could not show robot: {e}")

    def _render_robot_free_image(self, camera_name: str) -> np.ndarray:
        """Re-render scene with robot hidden to get unoccluded view for SAM3.

        Used during warmup so SAM3 sees the real target unoccluded by the
        robot arm, preventing the spatula (fully visible) from winning top-1.

        Args:
            camera_name: Name of camera to render from (e.g. "overhead_camera")

        Returns:
            Robot-free image as uint8 RGB (H, W, 3)
        """
        self._hide_robot()
        try:
            scene = self.env.unwrapped._scene
            scene.update_render()
            camera = self.env.unwrapped._cameras[camera_name]
            camera.take_picture()
            images = camera.get_images()
            color_rgba = images["Color"]  # float32 [0,1] (H, W, 4)
            return (np.clip(color_rgba[..., :3], 0, 1) * 255).astype(np.uint8)
        finally:
            self._show_robot()
            self.env.unwrapped._scene.update_render()

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

    def _cross_validate_safeset(
        self,
        safe_masks: Dict[str, np.ndarray],
        safe_scores: Dict[str, float],
        distractor_masks: Dict[str, np.ndarray],
        distractor_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Score-aware cross-validation for safe-set instances (compute-only).

        For each TARGET instance, compute genuineness = safe_score - max_overlapping_dist_score.
        Returns genuineness scores for use in Layer 3 scoring — does NOT remove any instances.
        Anchor instances are skipped (never scored).

        Args:
            safe_masks: Individual safe-set masks {concept_name: mask}
            safe_scores: Confidence scores for safe-set detections
            distractor_masks: Individual distractor masks {concept_name: mask}
            distractor_scores: Confidence scores for distractor detections

        Returns:
            Dict mapping instance name to genuineness score {name: float}.
            Empty dict if no target instances or no distractors.
        """
        if not safe_masks or not distractor_masks:
            return {}

        genuineness_scores = {}

        for safe_name, safe_mask in safe_masks.items():
            base = safe_name.rsplit('_', 1)[0] if '_' in safe_name and safe_name.rsplit('_', 1)[1].isdigit() else safe_name

            # Only score target instances, skip anchor
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
            genuineness_scores[safe_name] = genuineness
            self._log(f"[CGVD] Cross-val: '{safe_name}' (score={safe_score:.3f}) "
                      f"genuineness={genuineness:.3f} "
                      f"(best distractor overlap: '{max_dist_name}', score={max_dist_score:.3f})")

        return genuineness_scores

    def _accumulate_target(self, target_mask: np.ndarray):
        """Accumulate a target detection with IoU spatial gating (Layer 2).

        Args:
            target_mask: Binary mask for this frame's target detection
        """
        new_binary = (target_mask > 0.5)
        if new_binary.sum() < self.min_component_pixels:
            return  # Too small to be meaningful

        if self.cached_target_mask is None:
            # First detection: initialize anchor
            self.cached_target_mask = target_mask.copy()
            self._target_votes = new_binary.astype(np.float32)
            self._log("[CGVD] Layer 2: target anchor initialized")
            return

        # IoU gating: skip on early frames
        early_frame = self.frame_count < self.iou_gate_start_frame

        if early_frame:
            # Accumulate unconditionally
            self.cached_target_mask = np.maximum(self.cached_target_mask, target_mask)
            if self._target_votes is None:
                self._target_votes = new_binary.astype(np.float32)
            else:
                self._target_votes += new_binary.astype(np.float32)
            self._log(f"[CGVD] Layer 2: accumulated (ungated, frame={self.frame_count})")
            return

        # Compute IoU against TARGET-ONLY accumulated mask
        existing_binary = (self.cached_target_mask > 0.5)
        intersection = np.logical_and(new_binary, existing_binary).sum()
        union_area = np.logical_or(new_binary, existing_binary).sum()
        iou = float(intersection) / max(int(union_area), 1)

        if iou > self.iou_gate_threshold:
            self.cached_target_mask = np.maximum(self.cached_target_mask, target_mask)
            self._target_votes += new_binary.astype(np.float32)
            self._log(f"[CGVD] Layer 2: accumulated (IoU={iou:.3f})")
        else:
            self._log(f"[CGVD] Layer 2: REJECTED (IoU={iou:.3f} < {self.iou_gate_threshold})")

    def _cleanup_target_mask(self):
        """Post-warmup cleanup: keep only the best target component (Layer 3).

        Runs on cached_target_mask ONLY (not anchor). Uses connected component
        analysis to separate spatially disjoint detections, then scores each by
        detection consistency and distractor contamination.
        """
        if self.cached_target_mask is None or self._target_votes is None:
            return

        binary = (self.cached_target_mask > 0.5).astype(np.uint8)

        # Use 4-connectivity to reduce false merges from diagonally-adjacent masks
        num_labels, labels = cv2.connectedComponents(binary, connectivity=4)

        if num_labels <= 2:  # background + 1 component = nothing to clean
            self._log(f"[CGVD] Layer 3: {num_labels - 1} component(s), no cleanup needed")
            return

        best_label = -1
        best_score = -1.0

        for label_id in range(1, num_labels):
            component = (labels == label_id)
            pixel_count = int(component.sum())

            if pixel_count < self.min_component_pixels:
                continue

            # Average votes: how consistently was this region detected?
            avg_votes = float(self._target_votes[component].mean())

            # Distractor overlap: fraction of component covered by distractor mask
            if self.cached_distractor_mask is not None:
                dist_overlap = float(np.logical_and(
                    component, self.cached_distractor_mask > 0.5
                ).sum()) / max(pixel_count, 1)
            else:
                dist_overlap = 0.0

            # Find best genuineness among instances overlapping this component
            best_genuineness = 0.0
            best_genuine_name = None
            for inst_name, inst_mask in self.safe_individual_masks.items():
                base = inst_name.rsplit('_', 1)[0] if '_' in inst_name and inst_name.rsplit('_', 1)[1].isdigit() else inst_name
                if base != self.current_target:
                    continue
                # Check if this instance overlaps the component
                inst_binary = (inst_mask > 0.5)
                overlap = np.logical_and(component, inst_binary).sum()
                if overlap > 0:
                    g = self._instance_genuineness.get(inst_name, 0.0)
                    if best_genuine_name is None or g > best_genuineness:
                        best_genuineness = g
                        best_genuine_name = inst_name

            # Score: consistency * (1 - distractor contamination) * genuineness factor
            # genuineness modulates the score: positive genuineness boosts, negative penalizes
            overlap_penalty = min(dist_overlap, self.overlap_penalty_cap)
            genuineness_factor = 1.0 + best_genuineness  # genuineness_weight=1.0 baked in
            score = avg_votes * (1.0 - overlap_penalty) * genuineness_factor

            self._log(f"[CGVD] Layer 3: component {label_id}: "
                      f"pixels={pixel_count}, avg_votes={avg_votes:.1f}, "
                      f"dist_overlap={dist_overlap:.2f}, genuineness={best_genuineness:.3f} "
                      f"({best_genuine_name}), score={score:.3f}")

            if score > best_score:
                best_score = score
                best_label = label_id

        if best_label > 0:
            kept = (labels == best_label).astype(np.float32)
            removed_pixels = int(binary.sum() - kept.sum())
            if removed_pixels > 0:
                self._log(f"[CGVD] Layer 3: keeping component {best_label} "
                          f"(score={best_score:.3f}), removed {removed_pixels} pixels")
                self.cached_target_mask = kept

    def _recompute_cached_safe_mask(self, shape: Tuple[int, ...]):
        """Recompute combined safe mask from per-concept masks.

        Args:
            shape: (H, W) shape for zero initialization
        """
        h, w = shape[:2]
        target = self.cached_target_mask if self.cached_target_mask is not None else np.zeros((h, w), dtype=np.float32)
        anchor = self.cached_anchor_mask if self.cached_anchor_mask is not None else np.zeros((h, w), dtype=np.float32)
        self.cached_safe_mask = np.maximum(target, anchor)

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

        # Update distractor names to match actually-spawned objects
        # (when randomize_per_episode=True, the spawned set changes each episode)
        try:
            spawned_names = self.env.get_cgvd_concept_names()
            if spawned_names:
                self.distractor_names = spawned_names
                if self.verbose:
                    print(f"[CGVD] Updated distractor names from spawned objects: {spawned_names}")
        except AttributeError:
            pass  # No DistractorWrapper in chain, keep static names

        # Clear cached state
        self.cached_mask = None
        self.cached_compositing_mask = None
        self.cached_distractor_mask = None
        self.cached_safe_mask = None
        self.cached_target_mask = None
        self.cached_anchor_mask = None
        self.cached_inpainted_image = None
        self.last_robot_mask = None
        self.cached_robot_mask = None
        self.current_safe_mask = None
        self._target_votes = None
        self._anchor_votes = None
        self._instance_genuineness = {}
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

        # Internal warmup: accumulate SAM3 masks without stepping physics.
        # The scene is already settled (distractor wrapper handles physics),
        # so env.step() is unnecessary and can cause the robot's PD controller
        # to push objects (e.g. spoon near gripper) out of position.
        if self.distractor_names and self.safeset_warmup_frames > 0:
            for i in range(self.safeset_warmup_frames):
                self._apply_cgvd(obs)  # accumulate masks, skip compositing

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

        # During warmup, render robot-free image so SAM3 sees distractors
        # unoccluded by the robot arm.  Reused for safe-set query below.
        if in_warmup:
            safe_query_image = self._render_robot_free_image(camera_name)
            self._log("[CGVD] Using robot-free image for distractor and safe-set queries")
        else:
            safe_query_image = image
        should_recompute = (
            self.cached_distractor_mask is None or  # First frame
            in_warmup or  # Warm-up period: accumulate detections
            (not self.cache_distractor_once and self.frame_count % self.update_freq == 0)
        )
        if should_recompute:
            raw_distractor_mask = self.segmenter.segment(
                safe_query_image, distractor_concepts, presence_threshold=self.distractor_presence_threshold
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
            if self.cached_safe_mask is None or in_warmup:
                safe_concepts = self.parser.build_concept_prompt(
                    self.current_target,
                    self.current_anchor,
                    include_robot=False,  # Robot tracked separately
                )

                # safe_query_image was set earlier: robot-free during warmup,
                # raw observation otherwise.

                raw_target_mask = self.segmenter.segment(
                    safe_query_image, safe_concepts, presence_threshold=self.presence_threshold
                )
                self.safe_scores = self.segmenter.last_scores.copy()
                self.safe_individual_masks = self.segmenter.last_individual_masks.copy()

                # Cross-validate: compute genuineness scores (no removal).
                # Scores are stored for Layer 3 connected-component scoring.
                genuineness_scores = self._cross_validate_safeset(
                    self.safe_individual_masks, self.safe_scores,
                    self.distractor_individual_masks, self.distractor_scores,
                )
                self._instance_genuineness.update(genuineness_scores)

                scores_str = ", ".join(f"{k}={v:.3f}" for k, v in self.safe_scores.items())
                self._log(f"[CGVD] Frame {self.frame_count} Safe-set scores: {scores_str}")

                # Split by concept and accumulate ALL instances per-concept
                for name, mask in self.safe_individual_masks.items():
                    base = name.rsplit('_', 1)[0] if '_' in name and name.rsplit('_', 1)[1].isdigit() else name
                    if base == self.current_target:
                        self._accumulate_target(mask)
                    else:
                        # Anchor: accumulate unconditionally (never filtered)
                        if self.cached_anchor_mask is None:
                            self.cached_anchor_mask = mask.copy()
                            self._anchor_votes = (mask > 0.5).astype(np.float32)
                        else:
                            self.cached_anchor_mask = np.maximum(self.cached_anchor_mask, mask)
                            self._anchor_votes += (mask > 0.5).astype(np.float32)

                # Recompute combined safe mask from per-concept masks
                self._recompute_cached_safe_mask(raw_target_mask.shape)

                # On last warmup frame: run Layer 3 cleanup and log stats
                is_last_warmup = in_warmup and (self.frame_count == self.safeset_warmup_frames - 1)
                if is_last_warmup:
                    if self._target_votes is not None:
                        union_pixels = int((self.cached_target_mask > 0.5).sum()) if self.cached_target_mask is not None else 0
                        max_votes = int(self._target_votes.max())
                        self._log(f"[CGVD] Target mask before cleanup: {union_pixels} pixels "
                                  f"(union of {self.safeset_warmup_frames} frames, max_votes={max_votes})")
                    self._cleanup_target_mask()
                    self._recompute_cached_safe_mask(raw_target_mask.shape)

                if self.verbose:
                    cov = self.cached_safe_mask.sum() / self.cached_safe_mask.size * 100 if self.cached_safe_mask is not None else 0
                    status = "accumulating" if in_warmup else "frozen"
                    print(f"[CGVD] Safe-set mask: {cov:.1f}% (frame {self.frame_count}, {status})")

            # Step 3b: Robot mask - tracked every frame
            if self.include_robot:
                robot_concepts = "robot arm. robot gripper"
                robot_image = image
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

        self.current_safe_mask = safe_mask

        # Dilate safe-set mask to create protective buffer for Step 4 gating.
        # Uses cached_safe_mask (target+anchor only) instead of safe_mask
        # (which includes the robot). This makes cached_mask stable across
        # frames — robot visibility is handled in _composite() via re-enforcement.
        if self._step4_safe_dilation > 0:
            safe_dilation_kernel = np.ones((self._step4_safe_dilation, self._step4_safe_dilation), np.uint8)
            safe_mask_for_gating = cv2.dilate(
                (self.cached_safe_mask > 0.5).astype(np.uint8), safe_dilation_kernel, iterations=1
            ).astype(np.float32)
        else:
            safe_mask_for_gating = (self.cached_safe_mask > 0.5).astype(np.float32)

        # Save undilated distractor for compositing mask (before lama_dilation)
        undilated_distractor = (distractor_mask > 0.5).astype(np.float32)

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

        # Compositing mask: undilated distractor AND NOT safe
        # Uses the pre-dilation distractor mask so GaussianBlur transition
        # starts at the actual distractor boundary, not lama_dilation px beyond it.
        # Mechanism 2 (raw cached_distractor_mask) handles the core distractor pixels.
        self.cached_compositing_mask = np.logical_and(
            undilated_distractor > 0.5, safe_mask_for_gating < 0.5
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
            # On the LAST warmup frame, pre-compute the clean plate using the
            # real image (with robot) + full inpaint mask (distractors + robot).
            # We use the real image rather than the robot-free render because
            # SAPIEN's IBL renderer recomputes lighting when the robot is hidden,
            # causing a global color shift that creates visible seams at
            # distractor boundaries during compositing.  The robot-free image
            # is still used for SAM3 queries where color accuracy is irrelevant.
            if (self.frame_count == self.safeset_warmup_frames
                    and not self.disable_inpaint
                    and self.inpainter is not None):
                cache_mask = self._build_inpaint_mask()
                self.cached_inpainted_image = self.inpainter.inpaint(
                    image, cache_mask, dilate_mask=0
                )
                self.last_lama_time = self.inpainter.last_inpaint_time
                self.total_lama_time += self.last_lama_time
                self._log("[CGVD] Pre-computed clean plate from real image (last warmup frame)")
            if self.save_debug_images:
                self._save_debug_images(image, self.cached_mask, safe_query_image,
                                        query_image=safe_query_image)
            self.last_cgvd_time = time.time() - cgvd_start
            self.total_cgvd_time += self.last_cgvd_time
            return obs

        # Apply visual distillation to remove distractors
        # Clean plate is pre-computed during warmup from the real image.
        if self.disable_inpaint:
            # Ablation: Mean-color fill instead of inpainting
            distilled = self._apply_mean_fill(image, self.cached_mask)
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
            distilled = self._composite(image, self.cached_inpainted_image, self.cached_compositing_mask)
            if self.verbose:
                print(f"[CGVD] Using cached inpainting with scene composite (frame {self.frame_count})")

        # Write distilled image back to observation
        # IMPORTANT: Write to the SAME camera we read from (camera_name)
        obs = self._write_image_to_obs(obs, distilled, camera_name)

        # Save debug images if enabled
        if self.save_debug_images:
            self._save_debug_images(image, self.cached_mask, distilled)

        # Update timing stats
        self.last_cgvd_time = time.time() - cgvd_start
        self.total_cgvd_time += self.last_cgvd_time

        # Get component timing from segmenter and inpainter
        self.last_sam3_time = self.segmenter.last_segment_time
        self.total_sam3_time += self.last_sam3_time
        if self.inpainter is not None:
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
            # safe (binarized) includes target+anchor+robot from SAM3; binary_target
            # (dilated) provides extra halo protection for the stationary target/anchor.
            # Robot uses raw SAM3 mask — re-enforcement shows the live frame directly,
            # so the 1-2px SAM3 boundary under-segmentation is imperceptible.
            if safe is not None:
                reinforce_mask = np.maximum(safe, binary_target)
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
        self,
        original: np.ndarray,
        mask: np.ndarray,
        distilled: np.ndarray,
        query_image: Optional[np.ndarray] = None,
    ):
        """Save 4-panel debug image: Original | Distractors | Safe-set | VLA Input.

        Panel 1 (Original): Raw camera frame (with robot).
        Panel 2 (Distractor Detections): RED overlays on the image SAM3 queried.
            During warmup this is the robot-free image. Thick contour = accumulated mask.
        Panel 3 (Safe-set Detections): GREEN overlays on the image SAM3 queried.
            BLUE overlay for robot mask. Thick contour = accumulated mask.
        Panel 4 (VLA Input): Final composited image sent to the policy.

        Args:
            original: Live camera frame (with robot)
            mask: Final mask (cached_mask) after Step 4
            distilled: Composited output (or robot-free image during warmup)
            query_image: Image that SAM3 actually queried (robot-free during warmup).
                If None, falls back to original.
        """
        frame_num = self.frame_count - 1  # Already incremented
        h, w = original.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        in_warmup = frame_num < self.safeset_warmup_frames
        overlay_alpha = 0.4

        # Base image for detection overlay panels: use the image SAM3 actually saw
        sam3_base = query_image if query_image is not None else original

        if self.distractor_names and self.cached_distractor_mask is not None:
            # ── Panel 1: Original (live frame with robot) ──
            panel1 = original.copy()

            # ── Panel 2: Distractor Detections (overlaid on SAM3 query image) ──
            panel2 = sam3_base.copy()

            for name, ind_mask in self.distractor_individual_masks.items():
                score = self.distractor_scores.get(name, 0)
                mask_bin = (ind_mask > 0.5).astype(np.uint8)
                if mask_bin.sum() < 10:
                    continue
                where = mask_bin > 0
                panel2[where] = (
                    panel2[where].astype(np.float32) * (1 - overlay_alpha)
                    + np.array([255, 60, 60], dtype=np.float32) * overlay_alpha
                ).astype(np.uint8)
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel2, contours, -1, (255, 0, 0), 1)
                ys, xs = np.where(where)
                cx, cy = int(xs.mean()), int(ys.mean())
                label = f"{name}:{score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, font, 0.35, 1)
                cv2.rectangle(panel2, (cx - 2, cy - th - 3), (cx + tw + 2, cy + 3), (0, 0, 0), -1)
                cv2.putText(panel2, label, (cx, cy), font, 0.35, (255, 120, 120), 1)

            # Accumulated distractor contour (thick)
            if (self.cached_distractor_mask > 0.5).any():
                contours, _ = cv2.findContours(
                    (self.cached_distractor_mask > 0.5).astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(panel2, contours, -1, (255, 0, 0), 2)

            # ── Panel 3: Safe-set Detections (overlaid on SAM3 query image) ──
            panel3 = sam3_base.copy()

            # Target/anchor individual masks — GREEN overlay
            for name, ind_mask in self.safe_individual_masks.items():
                score = self.safe_scores.get(name, 0)
                mask_bin = (ind_mask > 0.5).astype(np.uint8)
                if mask_bin.sum() < 10:
                    continue
                where = mask_bin > 0
                panel3[where] = (
                    panel3[where].astype(np.float32) * (1 - overlay_alpha)
                    + np.array([60, 255, 60], dtype=np.float32) * overlay_alpha
                ).astype(np.uint8)
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel3, contours, -1, (0, 255, 0), 1)
                ys, xs = np.where(where)
                cx, cy = int(xs.mean()), int(ys.mean())
                label = f"{name}:{score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, font, 0.35, 1)
                cv2.rectangle(panel3, (cx - 2, cy - th - 3), (cx + tw + 2, cy + 3), (0, 0, 0), -1)
                cv2.putText(panel3, label, (cx, cy), font, 0.35, (120, 255, 120), 1)

            # Robot mask — BLUE overlay (separate SAM3 query, shown on safe-set panel)
            if self.last_robot_mask is not None:
                robot_bin = (self.last_robot_mask > 0.5).astype(np.uint8)
                if robot_bin.sum() > 10:
                    where = robot_bin > 0
                    panel3[where] = (
                        panel3[where].astype(np.float32) * 0.7
                        + np.array([80, 130, 255], dtype=np.float32) * 0.3
                    ).astype(np.uint8)
                    contours, _ = cv2.findContours(robot_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(panel3, contours, -1, (80, 130, 255), 1)

            # Accumulated safe-set contour (thick)
            if self.cached_safe_mask is not None and (self.cached_safe_mask > 0.5).any():
                contours, _ = cv2.findContours(
                    (self.cached_safe_mask > 0.5).astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(panel3, contours, -1, (0, 255, 0), 2)

            # ── Panel 4: VLA Input ──
            panel4 = distilled.copy()
            # Draw compositing boundary
            if (mask > 0.5).any():
                contours, _ = cv2.findContours(
                    (mask > 0.5).astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(panel4, contours, -1, (255, 0, 0), 1)

            # ── Assemble ──
            comparison = np.hstack([panel1, panel2, panel3, panel4])

            # Panel titles
            cv2.putText(comparison, f"Original (frame {frame_num})", (5, 20), font, 0.5, (255, 255, 255), 1)
            if query_image is not None:
                cv2.putText(comparison, "Distractor Query (robot-free)", (w + 5, 20), font, 0.45, (255, 120, 120), 1)
                cv2.putText(comparison, "Safe-set Query (robot-free)", (2 * w + 5, 20), font, 0.45, (120, 255, 120), 1)
            else:
                cv2.putText(comparison, "Distractor Query", (w + 5, 20), font, 0.5, (255, 120, 120), 1)
                cv2.putText(comparison, "Safe-set Query", (2 * w + 5, 20), font, 0.5, (120, 255, 120), 1)
            if in_warmup:
                cv2.putText(comparison, "Warmup (not sent to VLA)", (3 * w + 5, 20), font, 0.5, (255, 255, 0), 1)
            else:
                cv2.putText(comparison, "VLA Input", (3 * w + 5, 20), font, 0.5, (255, 255, 255), 1)

            # Coverage stats on panel 2
            d_cov = self.cached_distractor_mask.sum() / self.cached_distractor_mask.size * 100
            cv2.putText(comparison, f"coverage: {d_cov:.1f}%", (w + 5, 38), font, 0.3, (200, 160, 160), 1)

            # Target/anchor + coverage on panel 3
            y_info = 38
            if self.current_target:
                cv2.putText(comparison, f"target: {self.current_target}", (2 * w + 5, y_info), font, 0.35, (0, 255, 0), 1)
                y_info += 15
            if self.current_anchor:
                cv2.putText(comparison, f"anchor: {self.current_anchor}", (2 * w + 5, y_info), font, 0.35, (0, 200, 0), 1)
                y_info += 15
            s_cov = self.cached_safe_mask.sum() / self.cached_safe_mask.size * 100 if self.cached_safe_mask is not None else 0
            cv2.putText(comparison, f"coverage: {s_cov:.1f}%  BLUE=robot", (2 * w + 5, y_info), font, 0.3, (160, 200, 160), 1)

            # Warning if target not in safe-set
            if self.current_target:
                target_detected = any(
                    self.current_target in k and v >= self.presence_threshold
                    for k, v in self.safe_scores.items()
                )
                if not target_detected:
                    cv2.putText(comparison, "TARGET NOT IN SAFE SET",
                                (5, h - 30), font, 0.6, (255, 0, 0), 2)

            # Instruction across bottom of panel 1
            if self.current_instruction:
                cv2.putText(comparison, self.current_instruction, (5, h - 8), font, 0.35, (255, 255, 0), 1)

        else:
            # No distractors — simple 2-panel
            comparison = np.hstack([original, distilled])
            cv2.putText(comparison, "Original", (5, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(comparison, "VLA Input", (w + 5, 20), font, 0.5, (255, 255, 255), 1)

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
