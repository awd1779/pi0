"""Collision tracking for CGVD evaluation metrics.

Tracks collisions between robot gripper and distractor objects using SAPIEN's
contact detection API.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class CollisionTracker:
    """Tracks collisions between robot gripper and distractor objects.

    Uses SAPIEN's scene.get_contacts() API to detect when the robot's
    gripper links touch any distractor object.
    """

    # Common patterns for gripper link names across robot models
    GRIPPER_LINK_PATTERNS = [
        "gripper",
        "finger",
        "hand",
        "left_finger",
        "right_finger",
    ]

    def __init__(self, env, distractor_names: Optional[List[str]] = None):
        """Initialize collision tracker.

        Args:
            env: SimplerEnv environment (can be wrapped)
            distractor_names: List of distractor actor name patterns to track.
                            If None, tracks all actors with "distractor_" prefix.
        """
        self.env = env
        self._scene = env.unwrapped._scene

        # Store gripper links directly (for name-based matching in contacts)
        self._gripper_links: Set = set()
        self._gripper_link_names: Set[str] = set()
        self._identify_gripper_links()

        # Store distractor actors directly
        self._distractor_actors: Set = set()
        self._distractor_actor_names: Set[str] = set()
        self._distractor_names = distractor_names or []
        self._identify_distractor_actors()

        # Collision tracking state
        self.collision_count: int = 0
        self.collision_frames: List[int] = []
        self._collided_distractors: Set[str] = set()

    def _identify_gripper_links(self):
        """Identify gripper links by name patterns from the robot articulation."""
        try:
            robot = self.env.unwrapped.agent.robot
            for link in robot.get_links():
                link_name = link.name.lower()
                for pattern in self.GRIPPER_LINK_PATTERNS:
                    if pattern in link_name:
                        self._gripper_links.add(link)
                        self._gripper_link_names.add(link.name)
                        break
            if self._gripper_link_names:
                print(f"[CollisionTracker] Found gripper links: {self._gripper_link_names}")
        except Exception as e:
            print(f"[CollisionTracker] Warning: Could not identify gripper links: {e}")

    def _identify_distractor_actors(self):
        """Identify distractor actors in the scene."""
        try:
            for actor in self._scene.get_all_actors():
                actor_name = actor.name
                # Match distractor_ prefix (from DistractorWrapper)
                if actor_name.startswith("distractor_"):
                    self._distractor_actors.add(actor)
                    self._distractor_actor_names.add(actor_name)
                # Also match explicit distractor names if provided
                elif self._distractor_names:
                    for name in self._distractor_names:
                        if name in actor_name:
                            self._distractor_actors.add(actor)
                            self._distractor_actor_names.add(actor_name)
                            break
            if self._distractor_actor_names:
                print(f"[CollisionTracker] Found distractor actors: {self._distractor_actor_names}")
        except Exception as e:
            print(f"[CollisionTracker] Warning: Could not identify distractor actors: {e}")

    def reset(self):
        """Reset collision tracking state for new episode."""
        self.collision_count = 0
        self.collision_frames = []
        self._collided_distractors = set()
        # Re-identify distractor actors (they may have changed after env reset)
        self._distractor_actors = set()
        self._distractor_actor_names = set()
        self._identify_distractor_actors()

    def _is_gripper_distractor_contact(self, contact) -> Tuple[bool, Optional[str]]:
        """Check if contact is between gripper and distractor.

        Args:
            contact: SAPIEN contact object

        Returns:
            Tuple of (is_collision, distractor_name) where distractor_name is
            the name of the collided distractor or None if no collision.
        """
        try:
            # Get actors from contact - try different SAPIEN API versions
            actor0 = getattr(contact, 'actor0', None) or getattr(contact, 'actors', [None, None])[0]
            actor1 = getattr(contact, 'actor1', None) or getattr(contact, 'actors', [None, None])[1]

            if actor0 is None or actor1 is None:
                return False, None

            # Get names safely
            name0 = getattr(actor0, 'name', '') or ''
            name1 = getattr(actor1, 'name', '') or ''

            # Check if one is gripper and other is distractor using name matching
            is_gripper0 = name0 in self._gripper_link_names or any(p in name0.lower() for p in self.GRIPPER_LINK_PATTERNS)
            is_gripper1 = name1 in self._gripper_link_names or any(p in name1.lower() for p in self.GRIPPER_LINK_PATTERNS)
            is_distractor0 = name0 in self._distractor_actor_names or name0.startswith("distractor_")
            is_distractor1 = name1 in self._distractor_actor_names or name1.startswith("distractor_")

            if is_gripper0 and is_distractor1:
                return True, name1
            elif is_gripper1 and is_distractor0:
                return True, name0

            return False, None

        except Exception:
            return False, None

    def check_collisions(self, step_num: int) -> bool:
        """Check for gripper-distractor collisions at current timestep.

        Args:
            step_num: Current simulation step number

        Returns:
            True if any collision detected this frame, False otherwise
        """
        # Skip if we couldn't identify gripper or distractor components
        if not self._gripper_link_names and not self._gripper_links:
            return False
        if not self._distractor_actor_names and not self._distractor_actors:
            return False

        try:
            contacts = self._scene.get_contacts()
            collision_detected = False

            for contact in contacts:
                is_collision, distractor_name = self._is_gripper_distractor_contact(contact)
                if is_collision:
                    self.collision_count += 1
                    self.collision_frames.append(step_num)
                    if distractor_name:
                        self._collided_distractors.add(distractor_name)
                    collision_detected = True

            return collision_detected

        except Exception as e:
            # Silently handle errors - contacts API may not be available
            return False

    def get_stats(self) -> Dict:
        """Get collision statistics.

        Returns:
            Dict with collision statistics:
            - collision_count: Total number of collision frames
            - collision_frames: List of frame numbers where collisions occurred
            - unique_distractors_hit: Number of unique distractors touched
            - collided_distractor_names: Set of distractor names that were touched
        """
        return {
            "collision_count": self.collision_count,
            "collision_frames": self.collision_frames.copy(),
            "unique_distractors_hit": len(self._collided_distractors),
            "collided_distractor_names": self._collided_distractors.copy(),
        }

    @property
    def had_collision(self) -> bool:
        """Whether any collision occurred during the episode."""
        return self.collision_count > 0
