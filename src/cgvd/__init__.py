"""Concept-Gated Visual Distillation (CGVD) Framework.

A model-agnostic perception preprocessing module that "cleans" visual observations
before they reach robotic policies to prevent feature dilution in cluttered environments.
"""

from src.cgvd.cgvd_wrapper import CGVDWrapper
from src.cgvd.instruction_parser import InstructionParser
from src.cgvd.sam3_segmenter import SAM3Segmenter
from src.cgvd.spectral_abstraction import SpectralAbstraction

__all__ = [
    "CGVDWrapper",
    "SAM3Segmenter",
    "InstructionParser",
    "SpectralAbstraction",
]
