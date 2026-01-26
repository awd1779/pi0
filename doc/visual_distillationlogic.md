"""
visual_distillation.py

This module implements the "Safe-Subtraction" pipeline for Concept-Gated Visual Distillation (CGVD).
It generates a binary mask used to spectrally abstract (blur) semantic distractors while preserving
the 'Support Manifold' (Table) and 'Task-Relevant Entities' (Target/Anchor).

Logic:
1. Broad Net: Detect EVERYTHING matching generic attributes (Geometry/Material/Affordance).
2. Size Filter: Discard massive masks (Table/Background) and tiny noise.
3. Safe Set: Detect Target and Anchor explicitly.
4. Subtraction: Distractor = (Broad Net) - (Safe Set).
"""

import numpy as np
import cv2
from concept_fusion_anchors import get_all_anchors

def generate_distillation_mask(image, sam_model, target_prompts):
    """
    Generates a binary mask where 1 = Distractor (Blur), 0 = Keep Sharp.

    Args:
        image (np.array): Input RGB image.
        sam_model: Wrapper for SAM/SAM 3 model with a .predict(prompts, image) method.
        target_prompts (list): Specific semantic targets to keep (e.g., ["Spoon", "Blue Towel"]).

    Returns:
        np.array: Binary mask (uint8) of the same (H, W) as input image.
    """
    height, width = image.shape[:2]
    image_area = height * width

    # --- Step 1: The Broad Net (Attribute Union) ---
    # Get all generic attribute prompts (ConceptFusion Protocol)
    anchor_prompts = get_all_anchors()

    # Run SAM on all generic anchors to get candidate masks
    # Assumption: sam_model.predict returns a list of binary masks [N, H, W]
    candidate_masks, _, _ = sam_model.predict(anchor_prompts, image)

    # Initialize the "All Matter" mask
    all_matter_mask = np.zeros((height, width), dtype=bool)

    # CRITICAL - Size Filtering (The "Table Guard")
    # We iterate through every detected candidate to filter out the environment support.
    for mask in candidate_masks:
        mask_area = np.sum(mask)
        
        # Rule A: The Table Guard
        # If mask covers > 40% of image, it is the support manifold (Table/Floor).
        # We MUST NOT blur this, or Visual Odometry fails.
        if mask_area > (0.40 * image_area):
            continue 

        # Rule B: Noise Filter
        # Ignore single-pixel noise or tiny artifacts.
        if mask_area < (0.005 * image_area):
            continue

        # If it passes filters, it is valid "Matter" (potential clutter)
        all_matter_mask = np.logical_or(all_matter_mask, mask)

    # --- Step 2: The Safe Set (Target Protection) ---
    # We must explicitly find the things we want to interact with.
    # We also hardcode the Robot Body to ensure we don't blur the arm/gripper.
    safe_prompts = target_prompts + ["Robot Arm", "Gripper", "Robotic Hand"]
    
    # Run SAM on safe prompts
    safe_masks_list, _, _ = sam_model.predict(safe_prompts, image)
    
    # Create the Union of all safe items
    if len(safe_masks_list) > 0:
        safe_mask = np.max(np.array(safe_masks_list), axis=0)
    else:
        safe_mask = np.zeros((height, width), dtype=bool)

    # --- Step 3: The Semantic Subtraction ---
    # Distractor = (Matter) AND (NOT Safe)
    # This logic ensures that if the "Blue Towel" was found by 'fabric' in Step 1,
    # it is removed from the blur mask here because it exists in the safe_mask.
    # The "Red Towel" (Decoy), however, remains in the blur mask.
    distractor_mask = np.logical_and(all_matter_mask, ~safe_mask)

    return distractor_mask.astype(np.uint8)

def apply_spectral_abstraction(image, distractor_mask, kernel_size=(51, 51), sigma=0):
    """
    Helper utility to apply the blur based on the generated mask.
    """
    # Create heavily blurred version of the image
    blurred_img = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Composite: Start with original (Sharp Table/Targets)
    final_image = image.copy()
    
    # Apply blur only where mask is True
    # Ensure mask is boolean for indexing
    mask_bool = distractor_mask > 0
    final_image[mask_bool] = blurred_img[mask_bool]
    
    return final_image
