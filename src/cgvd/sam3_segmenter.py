"""SAM 3 Segmenter for concept-driven visual grounding."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class SAM3Segmenter:
    """Segments images using SAM 3 with text prompts.

    Uses the Sam3Processor and Sam3Model from HuggingFace transformers
    to perform text-prompted instance segmentation.

    Note: SAM3 expects SINGLE concepts per query (e.g., "spoon", "towel").
    For multiple concepts, we query each separately and combine masks.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        presence_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ):
        """Initialize SAM3 segmenter.

        Args:
            model_name: HuggingFace model identifier for SAM3
            device: Device to run model on (default: auto-detect)
            dtype: Model dtype (default: float16 for efficiency)
            presence_threshold: Minimum confidence to accept a mask (hallucination check)
            mask_threshold: Threshold for binarizing predicted masks
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.presence_threshold = presence_threshold
        self.mask_threshold = mask_threshold

        self.processor = None
        self.model = None
        self._initialized = False
        self._vision_embeds_cache = None
        self.last_scores = {}  # Stores per-concept scores from last segment() call

    def _lazy_init(self):
        """Lazily initialize model and processor on first use."""
        if self._initialized:
            return

        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError:
            raise ImportError(
                "SAM3 requires transformers >= 5.0.0 (main branch). "
                "Install with: pip install git+https://github.com/huggingface/transformers.git@main"
            )

        # Get HuggingFace token for gated model access
        hf_token = os.environ.get("HF_TOKEN")

        self.processor = Sam3Processor.from_pretrained(self.model_name, token=hf_token)
        self.model = Sam3Model.from_pretrained(
            self.model_name, torch_dtype=self.dtype, token=hf_token
        )
        self.model.to(self.device)
        self.model.eval()
        self._initialized = True

    def _parse_concepts(self, concepts: str) -> List[str]:
        """Parse dot-separated concept string into list of individual concepts.

        Args:
            concepts: Dot-separated string like "spoon. towel. robot arm"

        Returns:
            List of individual concepts like ["spoon", "towel", "robot arm"]
        """
        # Split by dots and clean up
        parts = concepts.split(".")
        cleaned = [p.strip() for p in parts if p.strip()]
        return cleaned

    def _segment_single_concept(
        self,
        pil_image: Image.Image,
        concept: str,
        vision_embeds=None,
        original_sizes=None,
        presence_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Segment a single concept from an image.

        Args:
            pil_image: PIL Image
            concept: Single concept string (e.g., "spoon")
            vision_embeds: Pre-computed vision embeddings (optional)
            original_sizes: Original image sizes from processor
            presence_threshold: Override presence threshold (default: use instance threshold)

        Returns:
            Tuple of (mask, max_score) where mask is (H, W) float32
        """
        h, w = pil_image.size[1], pil_image.size[0]  # PIL is (W, H)

        # Use provided threshold or fall back to instance default
        threshold = presence_threshold if presence_threshold is not None else self.presence_threshold

        # If we have vision embeddings, use efficient multi-prompt inference
        if vision_embeds is not None:
            text_inputs = self.processor(text=concept, return_tensors="pt")
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

            with torch.no_grad():
                outputs = self.model(
                    vision_embeds=vision_embeds,
                    **text_inputs
                )
        else:
            # Full inference with both image and text
            inputs = self.processor(
                images=pil_image,
                text=concept,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            original_sizes = inputs.get("original_sizes")

            with torch.no_grad():
                outputs = self.model(**inputs)

        # Post-process to get instance masks
        target_sizes = original_sizes.tolist() if original_sizes is not None else [[h, w]]
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )

        # Combine masks for this concept
        combined_mask = np.zeros((h, w), dtype=np.float32)
        max_score = 0.0

        if results and len(results) > 0:
            result = results[0]
            if "masks" in result and len(result["masks"]) > 0:
                scores = result.get("scores", torch.ones(len(result["masks"])))

                for i in range(len(result["masks"])):
                    mask_tensor = result["masks"][i]
                    mask = mask_tensor.cpu().numpy().astype(np.float32)
                    score = float(scores[i].cpu()) if isinstance(scores[i], torch.Tensor) else float(scores[i])
                    max_score = max(max_score, score)

                    # Resize if needed
                    if mask.shape != (h, w):
                        import cv2
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    combined_mask = np.maximum(combined_mask, mask)

        return combined_mask, max_score

    def segment(
        self,
        image: np.ndarray,
        concepts: str,
        return_individual_masks: bool = False,
        presence_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Segment image based on text concepts.

        Args:
            image: Input RGB image, shape (H, W, 3), dtype uint8
            concepts: Dot-separated concept string (e.g., "apple. basket. robot arm")
            return_individual_masks: If True, return dict of individual masks per concept
            presence_threshold: Override presence threshold for this call (default: use instance threshold)

        Returns:
            Combined binary mask where 1 = any concept detected, 0 = background
            Shape (H, W), dtype float32

        Note:
            After calling segment(), you can access self.last_scores for per-concept
            confidence scores (dict mapping concept -> score).
        """
        self._lazy_init()

        # Use provided threshold or fall back to instance default
        threshold = presence_threshold if presence_threshold is not None else self.presence_threshold

        h, w = image.shape[:2]

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Parse concepts into individual queries
        concept_list = self._parse_concepts(concepts)
        print(f"[SAM3] Parsed concepts: {concept_list}")

        # Pre-compute vision embeddings for efficiency
        img_inputs = self.processor(images=pil_image, return_tensors="pt")
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
        original_sizes = img_inputs.get("original_sizes")

        with torch.no_grad():
            vision_embeds = self.model.get_vision_features(pixel_values=img_inputs["pixel_values"])

        # Query each concept and combine masks
        combined_mask = np.zeros((h, w), dtype=np.float32)
        individual_masks = {}
        self.last_scores = {}  # Store scores for external access

        for concept in concept_list:
            mask, score = self._segment_single_concept(
                pil_image, concept, vision_embeds, original_sizes, threshold,
            )
            print(f"[SAM3] Concept '{concept}': score={score:.3f}, threshold={threshold:.2f}, coverage={mask.sum() / mask.size * 100:.1f}%")

            individual_masks[concept] = {"mask": mask, "score": score}
            self.last_scores[concept] = score
            combined_mask = np.maximum(combined_mask, mask)

        # Binarize
        combined_mask = (combined_mask > 0.5).astype(np.float32)
        print(f"[SAM3] Combined mask coverage: {combined_mask.sum() / combined_mask.size * 100:.1f}%")

        if return_individual_masks:
            return combined_mask, individual_masks

        return combined_mask

    def segment_with_fallback(
        self,
        image: np.ndarray,
        concepts: str,
        fallback_to_full: bool = True,
    ) -> np.ndarray:
        """Segment with fallback behavior for failed detections.

        If no concepts are detected (hallucination check fails for all),
        this method can either return a blank mask or a full mask.

        Args:
            image: Input RGB image
            concepts: Dot-separated concept string
            fallback_to_full: If True, return full mask on detection failure
                             If False, return blank mask (safer for unknown scenes)

        Returns:
            Binary mask, shape (H, W)
        """
        mask = self.segment(image, concepts)

        # Check if we got any valid detections
        if mask.sum() < 100:  # Less than 100 pixels = likely failed
            h, w = image.shape[:2]
            if fallback_to_full:
                # Return full mask - don't blur anything
                return np.ones((h, w), dtype=np.float32)
            else:
                # Return blank mask - blur everything
                return np.zeros((h, w), dtype=np.float32)

        return mask


class MockSAM3Segmenter:
    """Mock SAM3 segmenter for testing without model dependencies.

    Returns a simple center-weighted mask that simulates detecting
    objects in the center of the frame (where robot workspace typically is).
    """

    def __init__(self, presence_threshold: float = 0.4, **kwargs):
        self.presence_threshold = presence_threshold
        self.last_scores = {}  # Mock scores

    def segment(
        self,
        image: np.ndarray,
        concepts: str,
        return_individual_masks: bool = False,
        presence_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Return a mock center-weighted mask.

        Args:
            image: Input RGB image
            concepts: Dot-separated concept string
            return_individual_masks: If True, return dict of individual masks
            presence_threshold: Override presence threshold (ignored in mock)
        """
        h, w = image.shape[:2]

        # Create center-weighted Gaussian mask
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        # Larger sigma = more of the image is "foreground"
        sigma = min(h, w) // 3

        mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        mask = (mask > 0.3).astype(np.float32)

        # Mock scores for each concept
        concept_list = [c.strip() for c in concepts.split(".") if c.strip()]
        self.last_scores = {c: 0.85 for c in concept_list}  # Mock high confidence

        if return_individual_masks:
            return mask, {}
        return mask

    def segment_with_fallback(
        self,
        image: np.ndarray,
        concepts: str,
        fallback_to_full: bool = True,
    ) -> np.ndarray:
        return self.segment(image, concepts)


class SAM3ClientSegmenter:
    """SAM3 client that communicates with a separate SAM3 server process.

    Use this when SAM3 cannot run in the same environment due to
    transformers version conflicts (e.g., with GR00T which needs 4.53.0).

    Start the server first:
        python scripts/sam3_server.py
    """

    def __init__(
        self,
        presence_threshold: float = 0.5,
        comm_dir: str = "/tmp/sam3_server",
        timeout: float = 30.0,
        **kwargs,
    ):
        self.presence_threshold = presence_threshold
        self.comm_dir = Path(comm_dir)
        self.timeout = timeout
        self.last_scores = {}
        self._check_server()

    def _check_server(self):
        """Check if SAM3 server is running."""
        ready_file = self.comm_dir / "ready"
        if not ready_file.exists():
            raise RuntimeError(
                "SAM3 server not running. Start it with:\n"
                "  conda activate <env-with-transformers-5.0>\n"
                "  python scripts/sam3_server.py &"
            )
        print("[SAM3 Client] Connected to SAM3 server")

    def _parse_concepts(self, concepts: str) -> List[str]:
        """Parse dot-separated concept string into list."""
        parts = concepts.split(".")
        return [p.strip() for p in parts if p.strip()]

    def segment(
        self,
        image: np.ndarray,
        concepts: str,
        presence_threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Segment image by sending request to SAM3 server."""
        import json
        import tempfile
        import time

        threshold = presence_threshold if presence_threshold is not None else self.presence_threshold
        concept_list = self._parse_concepts(concepts)

        if verbose:
            print(f"[SAM3 Client] Requesting segmentation: {concept_list}")

        # Save image to temp file
        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            image_path = f.name
            pil_image.save(image_path)

        try:
            # Write request
            request_file = self.comm_dir / "request.json"
            response_file = self.comm_dir / "response.npz"

            # Remove old response
            if response_file.exists():
                response_file.unlink()

            # Send request
            request = {
                'image_path': image_path,
                'concepts': concept_list,
                'threshold': threshold,
            }
            with open(request_file, 'w') as f:
                json.dump(request, f)

            # Wait for response
            start_time = time.time()
            while not response_file.exists():
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"SAM3 server timeout after {self.timeout}s")
                time.sleep(0.01)

            # Wait a bit for file to be fully written, then load with retries
            time.sleep(0.1)
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    data = np.load(response_file)
                    mask = data['mask']
                    break
                except EOFError:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                    else:
                        raise

            if 'error' in data.files and 'error' in data:
                print(f"[SAM3 Client] Server error: {data['error']}")

            if verbose:
                print(f"[SAM3 Client] Mask coverage: {mask.sum() / mask.size * 100:.1f}%")

            return mask

        finally:
            # Cleanup temp file
            os.unlink(image_path)

    def segment_with_robot(
        self,
        image: np.ndarray,
        concepts: str,
        fallback_to_full: bool = True,
    ) -> np.ndarray:
        return self.segment(image, concepts)


def create_segmenter(
    use_mock: bool = False,
    use_server: bool = False,
    **kwargs,
) -> SAM3Segmenter:
    """Factory function to create appropriate segmenter.

    Args:
        use_mock: If True, return mock segmenter for testing
        use_server: If True, use SAM3 client to connect to server
        **kwargs: Arguments passed to segmenter constructor

    Returns:
        SAM3Segmenter, SAM3ClientSegmenter, or MockSAM3Segmenter instance
    """
    if use_mock:
        return MockSAM3Segmenter(**kwargs)
    if use_server:
        return SAM3ClientSegmenter(**kwargs)
    return SAM3Segmenter(**kwargs)
