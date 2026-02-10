"""Grasp failure analysis for CGVD evaluation metrics.

Categorizes episode failures into:
- success: Task completed successfully
- never_reached: Target object position unchanged from start
- missed_grasp: Gripper closed near target but object not grasped
- dropped: Object was moving with gripper, then fell
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np


class GraspAnalyzer:
    """Analyzes grasp outcomes to categorize failure modes.

    Uses object position tracking and gripper state to infer what went wrong
    when a task fails. This provides more granular feedback than binary success.
    """

    # Thresholds for failure mode detection
    POSITION_CHANGE_THRESHOLD = 0.02  # 2cm - object must move this much to count as "reached"
    NEAR_TARGET_THRESHOLD = 0.08  # 8cm - gripper must be within this distance to count as "near"
    GRASP_HEIGHT_THRESHOLD = 0.05  # 5cm - object must lift this high to count as "grasped"
    DROP_HEIGHT_THRESHOLD = 0.03  # 3cm - if object falls this much after grasp, it was "dropped"

    def __init__(self, env):
        """Initialize grasp analyzer.

        Args:
            env: SimplerEnv environment (can be wrapped)
        """
        self.env = env

        # Episode state
        self.initial_target_pos: Optional[np.ndarray] = None
        self.max_target_height: float = 0.0
        self.was_grasped: bool = False
        self.grasp_frame: Optional[int] = None
        self.gripper_closed_near_target: bool = False
        self.gripper_close_frame: Optional[int] = None

        # Track position history for detailed analysis
        self.target_positions: list = []
        self.gripper_positions: list = []

    def reset(self):
        """Reset analyzer state for new episode."""
        self.initial_target_pos = None
        self.max_target_height = 0.0
        self.was_grasped = False
        self.grasp_frame = None
        self.gripper_closed_near_target = False
        self.gripper_close_frame = None
        self.target_positions = []
        self.gripper_positions = []

    def on_reset(self, obs: Dict[str, Any]):
        """Called after environment reset to capture initial state.

        Args:
            obs: Initial observation from environment
        """
        self.reset()
        self.initial_target_pos = self._get_target_position()
        if self.initial_target_pos is not None:
            self.max_target_height = self.initial_target_pos[2]
            self.target_positions.append(self.initial_target_pos.copy())

    def _get_target_position(self) -> Optional[np.ndarray]:
        """Get current target object position from environment.

        Returns:
            Target object position as [x, y, z] or None if not available
        """
        try:
            base_env = self.env.unwrapped
            # SimplerEnv stores target object as episode_source_obj
            if hasattr(base_env, 'episode_source_obj') and base_env.episode_source_obj is not None:
                return np.array(base_env.episode_source_obj.pose.p)
            return None
        except Exception:
            return None

    def _get_gripper_position(self) -> Optional[np.ndarray]:
        """Get current gripper position from environment.

        Returns:
            Gripper TCP position as [x, y, z] or None if not available
        """
        try:
            base_env = self.env.unwrapped
            # Try to get TCP position from agent
            if hasattr(base_env, 'agent'):
                agent = base_env.agent
                if hasattr(agent, 'tcp') and agent.tcp is not None:
                    return np.array(agent.tcp.pose.p)
                # Fallback: use end effector link
                if hasattr(agent, 'robot'):
                    ee_link = agent.robot.get_links()[-1]  # Last link is typically EE
                    return np.array(ee_link.pose.p)
            return None
        except Exception:
            return None

    def _get_gripper_state(self, action: np.ndarray) -> float:
        """Extract gripper command from action.

        Args:
            action: Action array from policy

        Returns:
            Gripper command value (negative = close, positive = open)
        """
        try:
            # Standard format: action[6] is gripper for 7-DOF arm
            if len(action) >= 7:
                return float(action[6])
            return 0.0
        except Exception:
            return 0.0

    def on_step(self, obs: Dict[str, Any], action: np.ndarray, step_num: int):
        """Called after each environment step to update tracking.

        Args:
            obs: Observation from environment
            action: Action taken
            step_num: Current step number
        """
        target_pos = self._get_target_position()
        gripper_pos = self._get_gripper_position()
        gripper_cmd = self._get_gripper_state(action)

        if target_pos is not None:
            self.target_positions.append(target_pos.copy())
            self.max_target_height = max(self.max_target_height, target_pos[2])

            # Check if object has been grasped (lifted above initial height)
            if self.initial_target_pos is not None:
                height_delta = target_pos[2] - self.initial_target_pos[2]
                if height_delta > self.GRASP_HEIGHT_THRESHOLD and not self.was_grasped:
                    self.was_grasped = True
                    self.grasp_frame = step_num

        if gripper_pos is not None:
            self.gripper_positions.append(gripper_pos.copy())

            # Check if gripper closed near target
            if target_pos is not None and gripper_cmd < 0:  # Negative = closing
                dist_to_target = np.linalg.norm(gripper_pos - target_pos)
                if dist_to_target < self.NEAR_TARGET_THRESHOLD:
                    if not self.gripper_closed_near_target:
                        self.gripper_closed_near_target = True
                        self.gripper_close_frame = step_num

    def classify_failure(self, success: bool, final_obs: Optional[Dict] = None) -> str:
        """Classify the episode outcome.

        Args:
            success: Whether the task succeeded (from env info)
            final_obs: Final observation (optional, unused currently)

        Returns:
            Failure mode string: "success", "never_reached", "missed_grasp", or "dropped"
        """
        if success:
            return "success"

        # Check if target moved at all
        if not self._target_moved():
            return "never_reached"

        # Check if object was picked up then dropped
        if self.was_grasped:
            final_pos = self._get_target_position()
            if final_pos is not None and self.initial_target_pos is not None:
                # Object was lifted but is now near initial height = dropped
                current_height = final_pos[2]
                initial_height = self.initial_target_pos[2]
                height_delta = current_height - initial_height

                # If object is close to initial height and we know it was lifted, it was dropped
                if height_delta < self.DROP_HEIGHT_THRESHOLD:
                    return "dropped"
                # Object is still elevated - might be in wrong location
                return "dropped"

        # Gripper closed near target but didn't grasp
        if self.gripper_closed_near_target:
            return "missed_grasp"

        # Default: robot got close but didn't complete grasp
        return "never_reached"

    def _target_moved(self) -> bool:
        """Check if target object moved significantly from initial position.

        Returns:
            True if target moved more than threshold from start
        """
        if self.initial_target_pos is None or len(self.target_positions) < 2:
            return False

        # Check maximum displacement from initial position
        max_displacement = 0.0
        for pos in self.target_positions:
            displacement = np.linalg.norm(pos[:2] - self.initial_target_pos[:2])  # XY only
            max_displacement = max(max_displacement, displacement)

        return max_displacement > self.POSITION_CHANGE_THRESHOLD

    def get_stats(self) -> Dict:
        """Get grasp analysis statistics.

        Returns:
            Dict with analysis data:
            - was_grasped: Whether object was ever lifted
            - grasp_frame: Frame when grasp was detected
            - gripper_closed_near_target: Whether gripper closed near object
            - target_moved: Whether target moved from initial position
            - max_height_delta: Maximum height object was lifted
        """
        max_height_delta = 0.0
        if self.initial_target_pos is not None:
            max_height_delta = self.max_target_height - self.initial_target_pos[2]

        return {
            "was_grasped": self.was_grasped,
            "grasp_frame": self.grasp_frame,
            "gripper_closed_near_target": self.gripper_closed_near_target,
            "gripper_close_frame": self.gripper_close_frame,
            "target_moved": self._target_moved(),
            "max_height_delta": max_height_delta,
            "num_target_observations": len(self.target_positions),
        }
