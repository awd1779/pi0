"""SAM 3 Segmenter for concept-driven visual grounding."""

import os
import time
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
        mask_threshold: float = 0.3,
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

        # Timing instrumentation
        self.last_segment_time: float = 0.0

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

        # Try local cache first to avoid network timeouts when model is
        # already downloaded.  Fall back to network download if needed.
        try:
            self.processor = Sam3Processor.from_pretrained(
                self.model_name, token=hf_token, local_files_only=True,
            )
            self.model = Sam3Model.from_pretrained(
                self.model_name, torch_dtype=self.dtype, token=hf_token,
                local_files_only=True,
            )
        except OSError:
            print("[SAM3] Model not in local cache, downloading from HuggingFace Hub...")
            self.processor = Sam3Processor.from_pretrained(
                self.model_name, token=hf_token,
            )
            self.model = Sam3Model.from_pretrained(
                self.model_name, torch_dtype=self.dtype, token=hf_token,
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

        # Collect per-instance masks
        combined_mask = np.zeros((h, w), dtype=np.float32)
        max_score = 0.0
        instance_masks = []  # List of (mask, score) for each instance

        if results and len(results) > 0:
            result = results[0]
            if "masks" in result and len(result["masks"]) > 0:
                if "scores" not in result:
                    raise KeyError("SAM3 post_process returned no 'scores' — check transformers version")
                scores = result["scores"]

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
                    instance_masks.append((mask.copy(), score))

        return combined_mask, max_score, instance_masks

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
        start_time = time.time()

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
        self.last_individual_masks = {}  # Store individual masks for debug visualization

        for concept in concept_list:
            mask, score, instance_masks = self._segment_single_concept(
                pil_image, concept, vision_embeds, original_sizes, threshold,
            )
            print(f"[SAM3] Concept '{concept}': score={score:.3f}, threshold={threshold:.2f}, coverage={mask.sum() / mask.size * 100:.1f}%, instances={len(instance_masks)}")

            individual_masks[concept] = {"mask": mask, "score": score}
            # Store per-instance masks and scores for debug visualization
            # Use consistent keys between masks and scores
            if len(instance_masks) == 0:
                # No detections - store empty with base name
                self.last_scores[concept] = score
                self.last_individual_masks[concept] = np.zeros_like(combined_mask)
            elif len(instance_masks) == 1:
                # Single instance - use base name
                self.last_scores[concept] = instance_masks[0][1]
                self.last_individual_masks[concept] = instance_masks[0][0]
            else:
                # Multiple instances - use indexed names only (no base name)
                for i, (inst_mask, inst_score) in enumerate(instance_masks):
                    self.last_individual_masks[f"{concept}_{i}"] = inst_mask
                    self.last_scores[f"{concept}_{i}"] = inst_score
            combined_mask = np.maximum(combined_mask, mask)

        # Binarize
        combined_mask = (combined_mask > 0.5).astype(np.float32)
        print(f"[SAM3] Combined mask coverage: {combined_mask.sum() / combined_mask.size * 100:.1f}%")

        self.last_segment_time = time.time() - start_time

        if return_individual_masks:
            return combined_mask, individual_masks

        return combined_mask



class MockSAM3Segmenter:
    """Mock SAM3 segmenter for testing without model dependencies.

    Returns a simple center-weighted mask that simulates detecting
    objects in the center of the frame (where robot workspace typically is).
    """

    def __init__(self, presence_threshold: float = 0.4, **kwargs):
        self.presence_threshold = presence_threshold
        self.last_scores = {}  # Mock scores
        self.last_individual_masks = {}  # Mock individual masks
        self.last_segment_time: float = 0.0  # Timing instrumentation

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
        start_time = time.time()

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

        self.last_segment_time = time.time() - start_time

        if return_individual_masks:
            return mask, {}
        return mask



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
        self.last_individual_masks = {}
        self.last_segment_time: float = 0.0  # Timing instrumentation
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

        start_time = time.time()

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
            wait_start = time.time()
            while not response_file.exists():
                if time.time() - wait_start > self.timeout:
                    raise TimeoutError(f"SAM3 server timeout after {self.timeout}s")
                time.sleep(0.01)

            # Wait a bit for file to be fully written, then load with retries
            time.sleep(0.1)
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    data = np.load(response_file)
                    break
                except EOFError:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                    else:
                        raise

            if 'error' in data.files:
                raise RuntimeError(f"SAM3 server error: {data['error']}")

            if 'mask' not in data.files:
                raise RuntimeError(f"SAM3 server response missing 'mask' key. Keys: {list(data.files)}")
            mask = data['mask']

            # Parse per-concept masks and scores from server response
            concept_list = self._parse_concepts(concepts)
            self.last_scores = {}
            self.last_individual_masks = {}
            for concept in concept_list:
                score_key = f'score_{concept}'
                mask_key = f'mask_{concept}'
                if score_key in data.files and mask_key in data.files:
                    self.last_scores[concept] = float(data[score_key])
                    self.last_individual_masks[concept] = data[mask_key].astype(np.float32)
                else:
                    # Fallback for old server that doesn't send per-concept data
                    self.last_scores[concept] = 1.0 if mask.any() else 0.0
                    self.last_individual_masks[concept] = mask.astype(np.float32)

            if verbose:
                print(f"[SAM3 Client] Mask coverage: {mask.sum() / mask.size * 100:.1f}%")
                for concept in concept_list:
                    score = self.last_scores.get(concept, 0.0)
                    cmask = self.last_individual_masks.get(concept)
                    cov = cmask.sum() / cmask.size * 100 if cmask is not None else 0.0
                    print(f"[SAM3 Client] Concept '{concept}': score={score:.3f}, coverage={cov:.1f}%")

            self.last_segment_time = time.time() - start_time

            return mask

        finally:
            # Cleanup temp file
            os.unlink(image_path)



# Module-level singleton instances for sharing across CGVDWrapper instances
_sam3_singleton: Optional[SAM3Segmenter] = None
_sam3_client_singleton: Optional[SAM3ClientSegmenter] = None
_mock_singleton: Optional[MockSAM3Segmenter] = None


def get_sam3_segmenter(
    use_mock: bool = False,
    use_server: bool = False,
    **kwargs,
) -> SAM3Segmenter:
    """Get or create a singleton SAM3 segmenter.

    This function returns a shared instance to avoid redundant model loading
    when multiple CGVDWrapper instances are created (e.g., in batch evaluation).

    The singleton is selected based on use_mock and use_server flags:
    - use_mock=True: Returns shared MockSAM3Segmenter
    - use_server=True: Returns shared SAM3ClientSegmenter
    - Otherwise: Returns shared SAM3Segmenter

    Args:
        use_mock: If True, return mock segmenter for testing
        use_server: If True, use SAM3 client to connect to server
        **kwargs: Arguments passed to segmenter constructor (only used on first call)

    Returns:
        Shared SAM3Segmenter, SAM3ClientSegmenter, or MockSAM3Segmenter instance
    """
    global _sam3_singleton, _sam3_client_singleton, _mock_singleton

    if use_mock:
        if _mock_singleton is None:
            _mock_singleton = MockSAM3Segmenter(**kwargs)
            print("[SAM3] Created singleton MockSAM3Segmenter")
        return _mock_singleton

    if use_server:
        if _sam3_client_singleton is None:
            _sam3_client_singleton = SAM3ClientSegmenter(**kwargs)
            print("[SAM3] Created singleton SAM3ClientSegmenter")
        return _sam3_client_singleton

    if _sam3_singleton is None:
        _sam3_singleton = SAM3Segmenter(**kwargs)
        print("[SAM3] Created singleton SAM3Segmenter")
    return _sam3_singleton


def clear_sam3_singleton():
    """Clear the singleton instances (useful for testing or memory cleanup)."""
    global _sam3_singleton, _sam3_client_singleton, _mock_singleton
    _sam3_singleton = None
    _sam3_client_singleton = None
    _mock_singleton = None


