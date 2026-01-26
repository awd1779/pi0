"""
concept_fusion_anchors.py

This module defines the "Multimodal Attribute Protocol" for open-set visual perception.
Instead of relying on rigid class labels (e.g., 'YCB-005_tomato_soup_can'), we probe the scene
using orthogonal feature axes identified in ConceptFusion (Jatavallabhula et al., RSS 2023).

This approach ensures robustness to unseen objects by retrieving them based on their
intrinsic physical properties (Geometry, Material, Affordance) rather than specific instance IDs.
"""

# The ConceptFusion Attribute Pool
# We probe the scene using three orthogonal axes of semantic definition.
CONCEPT_ANCHORS = {
    # AXIS 1: GEOMETRY (Shape-based retrieval)
    # ConceptFusion aligns 3D visual geometry with language embeddings.
    # These prompts catch objects based on their silhouette and volume.
    "geometry": [
        "box", "cube", "cuboid",
        "cylinder", "canister", "tube",
        "sphere", "round object",
        "flat object", "slab",
        "curved object"
    ],

    # AXIS 2: MATERIAL (Texture-based retrieval)
    # ConceptFusion uses texture cues to differentiate objects.
    # These prompts catch objects based on surface reflection and texture.
    "material": [
        "metal item", "shiny object", "aluminum",
        "plastic item", "clear plastic", "container",
        "cardboard", "paper",
        "ceramic", "porcelain",
        "glass item",
        "fabric", "cloth", "textile"  # Critical for catching Towels (even Decoys)
    ],

    # AXIS 3: AFFORDANCE (Function-based retrieval)
    # ConceptFusion reasons about what an object DOES (affordance).
    # These prompts catch objects based on their likely usage.
    "affordance": [
        "container", "receptacle", "vessel",
        "tool", "utensil", "instrument",
        "food", "snack", "ingredient",
        "debris", "scrap", "waste",
        "household object", "kitchen item"
    ]
}

def get_all_anchors():
    """
    Returns a flattened list of all attribute anchors to probe the scene efficiently.
    Used for batch inference with the VFM (Vision Foundation Model).
    """
    all_prompts = []
    for category in CONCEPT_ANCHORS.values():
        all_prompts.extend(category)
    
    # Remove duplicates just in case
    return list(set(all_prompts))
