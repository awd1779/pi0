"""Instruction Parser for extracting target and anchor objects from language instructions."""

import re
from typing import Optional, Tuple


class InstructionParser:
    """Parses language instructions to extract target and anchor objects.

    The parser uses pattern matching to identify:
    - Target: The object being manipulated (e.g., "apple", "spoon")
    - Anchor: The destination or reference object (e.g., "basket", "towel")
    """

    # Known task patterns for SimplerEnv tasks
    TASK_PATTERNS = [
        # Bridge tasks
        (r"spoon.*towel", ("spoon", "towel")),
        (r"carrot.*plate", ("carrot", "plate")),
        (r"eggplant.*basket", ("eggplant", "wicker basket")),
        (r"stack.*cube", ("cube", "cube")),
        # Fractal/Google Robot tasks
        (r"pick.*coke", ("coke can", None)),
        (r"pick.*can", ("coke can", None)),
        (r"move.*near", ("object", "target location")),
        (r"open.*drawer", ("drawer handle", "drawer")),
        (r"close.*drawer", ("drawer handle", "drawer")),
        (r"apple.*drawer", ("apple", "drawer")),
        (r"place.*apple.*basket", ("apple", "basket")),
        (r"pick.*apple", ("apple", None)),
    ]

    # Common action verbs to strip from noun extraction
    ACTION_VERBS = [
        "pick",
        "place",
        "put",
        "move",
        "stack",
        "open",
        "close",
        "grasp",
        "grab",
        "lift",
        "drop",
    ]

    # Prepositions indicating anchor objects
    ANCHOR_PREPOSITIONS = ["on", "in", "into", "onto", "near", "beside", "next to"]

    def __init__(self):
        pass

    def parse(self, instruction: str) -> Tuple[str, Optional[str]]:
        """Parse instruction to extract target and anchor objects.

        Args:
            instruction: Natural language instruction (e.g., "pick apple and place in basket")

        Returns:
            Tuple of (target_object, anchor_object). Anchor may be None.
        """
        text = instruction.lower().strip()

        # Try known task patterns first
        for pattern, (target, anchor) in self.TASK_PATTERNS:
            if re.search(pattern, text):
                return (target, anchor)

        # Fallback: extract nouns heuristically
        target = self._extract_target(text)
        anchor = self._extract_anchor(text)

        return (target, anchor)

    def _extract_target(self, text: str) -> str:
        """Extract the target object (thing being manipulated)."""
        # Remove action verbs and find first noun-like word
        words = text.split()
        filtered = []
        skip_next = False

        for i, word in enumerate(words):
            # Skip articles
            if word in ["a", "an", "the"]:
                continue
            # Skip action verbs
            if word in self.ACTION_VERBS:
                continue
            # Skip prepositions and their following words
            if word in self.ANCHOR_PREPOSITIONS:
                skip_next = True
                continue
            if skip_next:
                skip_next = False
                continue
            # Clean punctuation
            clean_word = re.sub(r"[^\w\s]", "", word)
            if clean_word and len(clean_word) > 1:
                filtered.append(clean_word)

        # Return first meaningful word as target
        if filtered:
            return filtered[0]

        # Ultimate fallback
        return "object"

    def _extract_anchor(self, text: str) -> Optional[str]:
        """Extract the anchor object (destination/reference)."""
        # Look for patterns like "in basket", "on plate", etc.
        for prep in self.ANCHOR_PREPOSITIONS:
            pattern = rf"{prep}\s+(?:the\s+)?(\w+)"
            match = re.search(pattern, text)
            if match:
                anchor = match.group(1)
                # Skip if it's a verb or too short
                if anchor not in self.ACTION_VERBS and len(anchor) > 2:
                    return anchor

        return None

    def build_concept_prompt(
        self, target: str, anchor: Optional[str], include_robot: bool = True
    ) -> str:
        """Build a SAM3-compatible concept prompt.

        Args:
            target: Target object name
            anchor: Anchor object name (optional)
            include_robot: Whether to include robot arm/gripper (default True)

        Returns:
            Dot-separated concept string for SAM3
        """
        concepts = [target]
        if anchor:
            concepts.append(anchor)
        if include_robot:
            concepts.extend(["robot arm", "robot gripper"])

        return ". ".join(concepts)
