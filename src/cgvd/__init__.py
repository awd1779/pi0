"""Concept-Gated Visual Distillation (CGVD) Framework.

A model-agnostic perception preprocessing module that "cleans" visual observations
before they reach robotic policies to prevent feature dilution in cluttered environments.
"""

from src.cgvd.cgvd_wrapper import CGVDWrapper
from src.cgvd.collision_tracker import CollisionTracker
from src.cgvd.grasp_analyzer import GraspAnalyzer
from src.cgvd.instruction_parser import InstructionParser
from src.cgvd.lama_inpainter import (
    LamaInpainter,
    clear_lama_singleton,
    get_lama_inpainter,
)
from src.cgvd.sam3_segmenter import (
    SAM3Segmenter,
    clear_sam3_singleton,
    get_sam3_segmenter,
)
__all__ = [
    "CGVDWrapper",
    "SAM3Segmenter",
    "InstructionParser",
    # Evaluation metrics
    "CollisionTracker",
    "GraspAnalyzer",
    # Singleton getters for efficient model reuse
    "get_sam3_segmenter",
    "get_lama_inpainter",
    "clear_sam3_singleton",
    "clear_lama_singleton",
    "LamaInpainter",
]
