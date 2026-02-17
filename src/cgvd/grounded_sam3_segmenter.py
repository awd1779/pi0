"""Grounded SAM3 Segmenter — Grounding DINO detection + SAM3 box-prompted segmentation.

Pipeline: GDINO(image, concepts) -> boxes -> SAM3(image, text + boxes) -> pixel masks

GDINO provides stable open-vocabulary detection with bounding boxes.
SAM3 uses the box as spatial guidance and text for semantic disambiguation.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class GroundedSAM3Segmenter:
    """Segments images using Grounding DINO + SAM3 box-prompted segmentation.

    Drop-in replacement for SAM3Segmenter with the same interface:
    - segment() returns combined binary mask
    - last_scores, last_individual_masks, last_segment_time populated after each call
    """

    def __init__(
        self,
        sam3_model_name: str = "facebook/sam3",
        gdino_model_name: str = "IDEA-Research/grounding-dino-base",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        presence_threshold: float = 0.5,
        mask_threshold: float = 0.3,
        gdino_box_threshold: float = 0.25,
        gdino_text_threshold: float = 0.25,
    ):
        self.sam3_model_name = sam3_model_name
        self.gdino_model_name = gdino_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.presence_threshold = presence_threshold
        self.mask_threshold = mask_threshold
        self.gdino_box_threshold = gdino_box_threshold
        self.gdino_text_threshold = gdino_text_threshold

        # Models (lazy init)
        self.sam3_processor = None
        self.sam3_model = None
        self.gdino_processor = None
        self.gdino_model = None
        self._initialized = False
        self._box_prompt_supported: Optional[bool] = None  # None = untested

        # GDINO context labels — when set, these extra concepts are included in
        # every GDINO query for better disambiguation, but only detections for
        # the *requested* concepts are returned.  Set by cgvd_wrapper.
        self.gdino_context_labels: List[str] = []

        # IoU threshold for box filtering — after SAM3 returns masks for a box,
        # discard masks whose overlap with the GDINO box is below this value.
        self.box_iou_threshold: float = 0.3

        # Interface compatibility with SAM3Segmenter
        self.last_scores: Dict[str, float] = {}
        self.last_individual_masks: Dict[str, np.ndarray] = {}
        self.last_segment_time: float = 0.0
        self.last_gdino_detections: Dict[str, list] = {}  # concept -> [(box, score)]

        # Per-instance GDINO scores — populated by segment() for cross-val
        self.last_gdino_instance_scores: Dict[str, float] = {}
        # Context-label detections — populated by segment() for cross-val
        self.last_context_detections: Dict[str, list] = {}  # ctx_label -> [(box, score)]

        # Debug data — populated by segment() for visualization
        self.last_all_detections: Dict[str, list] = {}  # ALL GDINO boxes (incl. context)
        self.last_suppressed: list = []  # [(concept, box, score, ctx_label, ctx_score)]
        self.last_concept_list: List[str] = []  # requested concepts for this call

    def _lazy_init(self):
        """Load GDINO and SAM3 models on first use."""
        if self._initialized:
            return

        hf_token = os.environ.get("HF_TOKEN")

        # --- Grounding DINO ---
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError:
            raise ImportError(
                "Grounding DINO requires transformers >= 4.36.0. "
                "Install with: pip install transformers>=4.36.0"
            )

        print(f"[GroundedSAM3] Loading GDINO: {self.gdino_model_name}")
        try:
            self.gdino_processor = AutoProcessor.from_pretrained(
                self.gdino_model_name, token=hf_token, local_files_only=True,
            )
            self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.gdino_model_name, torch_dtype=torch.float32, token=hf_token,
                local_files_only=True,
            )
        except OSError:
            print("[GroundedSAM3] GDINO not in cache, downloading...")
            self.gdino_processor = AutoProcessor.from_pretrained(
                self.gdino_model_name, token=hf_token,
            )
            self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.gdino_model_name, torch_dtype=torch.float32, token=hf_token,
            )
        self.gdino_model.to(self.device)
        self.gdino_model.eval()

        # --- SAM3 ---
        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError:
            raise ImportError(
                "SAM3 requires transformers >= 5.0.0 (main branch). "
                "Install with: pip install git+https://github.com/huggingface/transformers.git@main"
            )

        print(f"[GroundedSAM3] Loading SAM3: {self.sam3_model_name}")
        try:
            self.sam3_processor = Sam3Processor.from_pretrained(
                self.sam3_model_name, token=hf_token, local_files_only=True,
            )
            self.sam3_model = Sam3Model.from_pretrained(
                self.sam3_model_name, torch_dtype=self.dtype, token=hf_token,
                local_files_only=True,
            )
        except OSError:
            print("[GroundedSAM3] SAM3 not in cache, downloading...")
            self.sam3_processor = Sam3Processor.from_pretrained(
                self.sam3_model_name, token=hf_token,
            )
            self.sam3_model = Sam3Model.from_pretrained(
                self.sam3_model_name, torch_dtype=self.dtype, token=hf_token,
            )
        self.sam3_model.to(self.device)
        self.sam3_model.eval()

        self._initialized = True
        print("[GroundedSAM3] Both models loaded")

    def _parse_concepts(self, concepts: str) -> List[str]:
        """Parse dot-separated concept string into list."""
        parts = concepts.split(".")
        return [p.strip() for p in parts if p.strip()]

    def _run_gdino(
        self, pil_image: Image.Image, concept_list: List[str],
    ) -> Dict[str, list]:
        """Run Grounding DINO to get bounding boxes for each concept.

        Returns:
            Dict mapping concept -> list of (box_xyxy, score).
            box_xyxy is [x1, y1, x2, y2] in pixel coordinates.
        """
        # GDINO text format: concepts separated by ". " with trailing "."
        gdino_text = ". ".join(concept_list) + "."

        inputs = self.gdino_processor(
            images=pil_image, text=gdino_text, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.gdino_model(**inputs)

        h, w = pil_image.size[1], pil_image.size[0]
        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=self.gdino_box_threshold,
            text_threshold=self.gdino_text_threshold,
            target_sizes=[(h, w)],
        )

        concept_detections: Dict[str, list] = {c: [] for c in concept_list}

        if results and len(results) > 0:
            result = results[0]
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            labels = result.get("labels", [])

            for i, label in enumerate(labels):
                box = boxes[i].cpu().tolist() if isinstance(boxes[i], torch.Tensor) else list(boxes[i])
                score = float(scores[i].cpu()) if isinstance(scores[i], torch.Tensor) else float(scores[i])
                matched = self._match_label_to_concept(label, concept_list)
                if matched is not None:
                    concept_detections[matched].append((box, score))

        return concept_detections

    @staticmethod
    def _match_label_to_concept(label: str, concept_list: List[str]) -> Optional[str]:
        """Match a GDINO label to the best matching concept."""
        label_lower = label.lower().strip()

        # Exact match
        for c in concept_list:
            if label_lower == c.lower():
                return c

        # Substring match (label in concept or concept in label)
        for c in concept_list:
            if label_lower in c.lower() or c.lower() in label_lower:
                return c

        # Word overlap (at least one shared word)
        label_words = set(label_lower.split())
        best, best_overlap = None, 0
        for c in concept_list:
            overlap = len(label_words & set(c.lower().split()))
            if overlap > best_overlap:
                best_overlap = overlap
                best = c

        return best if best_overlap > 0 else None

    @staticmethod
    def _box_iou(box1: list, box2: list) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _deduplicate_boxes(
        self,
        concept_detections: Dict[str, list],
        iou_threshold: float = 0.5,
    ) -> Dict[str, list]:
        """Deduplicate boxes across concepts — keep only the highest-scoring label.

        When the same physical object is detected as both "spoon" (score=0.6)
        and "ladle" (score=0.9), this removes the lower-scoring duplicate so
        the object only appears under its best-matching concept.
        """
        # Flatten all detections into a list: [(concept, box, score, original_idx)]
        all_dets = []
        for concept, dets in concept_detections.items():
            for i, (box, score) in enumerate(dets):
                all_dets.append((concept, box, score, i))

        # Mark detections to remove: for each pair, if IoU > threshold and
        # they're from different concepts, suppress the lower-scoring one
        suppressed = set()  # indices into all_dets
        for i in range(len(all_dets)):
            if i in suppressed:
                continue
            c_i, box_i, score_i, _ = all_dets[i]
            for j in range(i + 1, len(all_dets)):
                if j in suppressed:
                    continue
                c_j, box_j, score_j, _ = all_dets[j]
                if c_i == c_j:
                    continue  # same concept, skip
                iou = self._box_iou(box_i, box_j)
                if iou > iou_threshold:
                    # Same physical object — keep higher score
                    if score_i >= score_j:
                        suppressed.add(j)
                        print(f"[GroundedSAM3] Dedup: '{c_j}':{score_j:.3f} "
                              f"suppressed by '{c_i}':{score_i:.3f} (IoU={iou:.2f})")
                    else:
                        suppressed.add(i)
                        print(f"[GroundedSAM3] Dedup: '{c_i}':{score_i:.3f} "
                              f"suppressed by '{c_j}':{score_j:.3f} (IoU={iou:.2f})")
                        break  # i is suppressed, no need to check more

        # Rebuild concept_detections without suppressed entries
        result: Dict[str, list] = {c: [] for c in concept_detections}
        for idx, (concept, box, score, _) in enumerate(all_dets):
            if idx not in suppressed:
                result[concept].append((box, score))
        return result

    def _suppress_by_context(
        self,
        concept_detections: Dict[str, list],
        all_detections: Dict[str, list],
        concept_list: List[str],
        iou_threshold: float = 0.5,
    ) -> Dict[str, list]:
        """Suppress detections that overlap a higher-scoring context-label box.

        For each requested-concept detection (e.g. "spoon" box), check if any
        context-label detection (e.g. "ladle" box) covers the same region with
        a higher GDINO score.  If so, the object is more likely the context
        label than the requested concept — suppress it.
        """
        context_labels = [c for c in all_detections if c not in concept_list]
        if not context_labels:
            return concept_detections

        self.last_suppressed = []

        for concept in concept_list:
            surviving = []
            for box, score in concept_detections[concept]:
                best_ctx_label = None
                best_ctx_score = 0.0
                for ctx_label in context_labels:
                    for ctx_box, ctx_score in all_detections[ctx_label]:
                        iou = self._box_iou(box, ctx_box)
                        if iou > iou_threshold and ctx_score > score and ctx_score > best_ctx_score:
                            best_ctx_score = ctx_score
                            best_ctx_label = ctx_label
                if best_ctx_label is not None:
                    self.last_suppressed.append(
                        (concept, box, score, best_ctx_label, best_ctx_score)
                    )
                    print(
                        f"[GroundedSAM3] Suppressed '{concept}' box "
                        f"(score={score:.3f}) — overlaps '{best_ctx_label}' "
                        f"(score={best_ctx_score:.3f})"
                    )
                else:
                    surviving.append((box, score))
            concept_detections[concept] = surviving

        return concept_detections

    def _segment_with_box(
        self,
        pil_image: Image.Image,
        concept: str,
        box: list,
        vision_embeds,
        original_sizes,
        threshold: float,
    ) -> Tuple[np.ndarray, list]:
        """Run SAM3 with text + box prompt for a single GDINO detection.

        Tries combined text+box prompt first. Falls back to text-only if the
        SAM3 processor does not support input_boxes.

        Returns:
            (combined_mask, instance_list) where instance_list is [(mask, score), ...]
        """
        h, w = pil_image.size[1], pil_image.size[0]
        outputs = None

        # --- Try text + box combined prompt ---
        if self._box_prompt_supported is not False:
            try:
                all_inputs = self.sam3_processor(
                    images=pil_image,
                    text=concept,
                    input_boxes=[[[box[0], box[1], box[2], box[3]]]],
                    return_tensors="pt",
                )
                # Remove image tensors — use cached vision_embeds instead.
                # Cast floating-point tensors to model dtype (fp16) to avoid
                # dtype mismatch with SAM3 weights.
                model_kwargs = {}
                skip_keys = {"pixel_values", "original_sizes", "reshaped_input_sizes"}
                for k, v in all_inputs.items():
                    if k not in skip_keys:
                        if isinstance(v, torch.Tensor):
                            if v.is_floating_point():
                                model_kwargs[k] = v.to(self.device, dtype=self.dtype)
                            else:
                                model_kwargs[k] = v.to(self.device)
                        else:
                            model_kwargs[k] = v

                with torch.no_grad():
                    outputs = self.sam3_model(vision_embeds=vision_embeds, **model_kwargs)

                if self._box_prompt_supported is None:
                    self._box_prompt_supported = True
                    print("[GroundedSAM3] SAM3 box+text prompt: supported")

            except (TypeError, RuntimeError) as e:
                if self._box_prompt_supported is None:
                    self._box_prompt_supported = False
                    print(f"[GroundedSAM3] SAM3 box prompt not supported ({e}), using text-only")
                outputs = None

        # --- Fallback: text-only with vision_embeds ---
        if outputs is None:
            text_inputs = self.sam3_processor(text=concept, return_tensors="pt")
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            with torch.no_grad():
                outputs = self.sam3_model(vision_embeds=vision_embeds, **text_inputs)

        # --- Post-process ---
        target_sizes = original_sizes.tolist() if original_sizes is not None else [[h, w]]
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )

        combined = np.zeros((h, w), dtype=np.float32)
        instances = []

        # Build binary mask from the GDINO box for IoU filtering
        box_mask = np.zeros((h, w), dtype=bool)
        y1, y2 = max(0, int(box[1])), min(h, int(box[3]))
        x1, x2 = max(0, int(box[0])), min(w, int(box[2]))
        box_mask[y1:y2, x1:x2] = True

        if results and len(results) > 0:
            result = results[0]
            if "masks" in result and len(result["masks"]) > 0:
                scores = result.get("scores", [])
                for i in range(len(result["masks"])):
                    mask = result["masks"][i].cpu().numpy().astype(np.float32)
                    score = float(scores[i].cpu()) if isinstance(scores[i], torch.Tensor) else float(scores[i])
                    if mask.shape != (h, w):
                        import cv2
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    # IoU-based box filtering: discard masks with low overlap
                    mask_binary = mask > 0.5
                    mask_area = float(mask_binary.sum())
                    if mask_area > 0:
                        intersection = float((mask_binary & box_mask).sum())
                        overlap = intersection / mask_area
                        if overlap < self.box_iou_threshold:
                            continue  # mask doesn't overlap the GDINO box enough

                    combined = np.maximum(combined, mask)
                    instances.append((mask.copy(), score))

        return combined, instances

    def segment(
        self,
        image: np.ndarray,
        concepts: str,
        return_individual_masks: bool = False,
        presence_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Segment image using GDINO detection + SAM3 box-prompted segmentation.

        Args:
            image: Input RGB image, shape (H, W, 3), dtype uint8
            concepts: Dot-separated concept string (e.g., "spoon. towel. robot arm")
            return_individual_masks: If True, return dict of individual masks per concept
            presence_threshold: Override presence threshold for this call

        Returns:
            Combined binary mask where 1 = any concept detected, 0 = background.
            Shape (H, W), dtype float32.
        """
        start_time = time.time()
        self._lazy_init()

        threshold = presence_threshold if presence_threshold is not None else self.presence_threshold
        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)
        concept_list = self._parse_concepts(concepts)

        # Step 1: GDINO detection (all concepts at once)
        t0 = time.time()
        all_detections = self._run_gdino(pil_image, concept_list)
        self.last_all_detections = {c: list(dets) for c, dets in all_detections.items()}
        self.last_concept_list = list(concept_list)
        self.last_suppressed = []
        # Deduplicate: if same box detected as multiple concepts, keep highest score
        concept_detections = self._deduplicate_boxes(all_detections)
        gdino_time = time.time() - t0
        self.last_gdino_detections = concept_detections

        # Step 2: SAM3 vision embeddings (computed once)
        t0 = time.time()
        img_inputs = self.sam3_processor(images=pil_image, return_tensors="pt")
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
        original_sizes = img_inputs.get("original_sizes")

        with torch.no_grad():
            vision_embeds = self.sam3_model.get_vision_features(
                pixel_values=img_inputs["pixel_values"],
            )
        embed_time = time.time() - t0

        # Step 3: SAM3 segmentation per concept using GDINO boxes
        combined_mask = np.zeros((h, w), dtype=np.float32)
        self.last_scores = {}
        self.last_individual_masks = {}
        self.last_gdino_instance_scores = {}
        individual_masks = {}

        for concept in concept_list:
            detections = concept_detections[concept]

            if not detections:
                # No GDINO detection for this concept
                self.last_scores[concept] = 0.0
                self.last_individual_masks[concept] = np.zeros((h, w), dtype=np.float32)
                individual_masks[concept] = {
                    "mask": np.zeros((h, w), dtype=np.float32),
                    "score": 0.0,
                }
                continue

            concept_mask = np.zeros((h, w), dtype=np.float32)
            all_instances = []  # [(mask, sam3_score, gdino_score)]

            for box, gdino_score in detections:
                mask, instances = self._segment_with_box(
                    pil_image, concept, box, vision_embeds, original_sizes, threshold,
                )
                for inst_mask, inst_score in instances:
                    all_instances.append((inst_mask, inst_score, gdino_score))
                concept_mask = np.maximum(concept_mask, mask)

            # Store per-instance results matching SAM3Segmenter key convention
            # Use GDINO box scores as detection confidence (not SAM3 segment scores)
            if len(all_instances) == 0:
                self.last_scores[concept] = 0.0
                self.last_individual_masks[concept] = np.zeros((h, w), dtype=np.float32)
            elif len(all_instances) == 1:
                self.last_scores[concept] = all_instances[0][2]  # gdino_score
                self.last_individual_masks[concept] = all_instances[0][0]
                self.last_gdino_instance_scores[concept] = all_instances[0][2]
            else:
                for i, (inst_mask, inst_score, g_score) in enumerate(all_instances):
                    self.last_scores[f"{concept}_{i}"] = g_score  # gdino_score
                    self.last_individual_masks[f"{concept}_{i}"] = inst_mask
                    self.last_gdino_instance_scores[f"{concept}_{i}"] = g_score

            max_score = max((g for _, _, g in all_instances), default=0.0)
            individual_masks[concept] = {"mask": concept_mask, "score": max_score}
            combined_mask = np.maximum(combined_mask, concept_mask)

        # Binarize
        combined_mask = (combined_mask > 0.5).astype(np.float32)
        self.last_segment_time = time.time() - start_time

        sam3_time = self.last_segment_time - gdino_time - embed_time
        det_str = ", ".join(
            f"{c}: {len(d)}" for c, d in concept_detections.items()
        )
        print(
            f"[GroundedSAM3] {self.last_segment_time:.3f}s "
            f"(gdino={gdino_time:.3f}s, embed={embed_time:.3f}s, sam3={sam3_time:.3f}s) "
            f"| detections: {det_str}"
        )

        if return_individual_masks:
            return combined_mask, individual_masks
        return combined_mask

    def segment_text_only(
        self,
        image: np.ndarray,
        concepts: str,
        presence_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Segment using SAM3 text-only (bypass GDINO).

        Useful for concepts like "robot arm" that GDINO handles poorly.
        Uses the same SAM3 model but queries each concept with text only,
        matching SAM3Segmenter behavior.
        """
        start_time = time.time()
        self._lazy_init()

        threshold = presence_threshold if presence_threshold is not None else self.presence_threshold
        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)
        concept_list = self._parse_concepts(concepts)

        # Vision embeddings (computed once)
        img_inputs = self.sam3_processor(images=pil_image, return_tensors="pt")
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
        original_sizes = img_inputs.get("original_sizes")

        with torch.no_grad():
            vision_embeds = self.sam3_model.get_vision_features(
                pixel_values=img_inputs["pixel_values"],
            )

        combined_mask = np.zeros((h, w), dtype=np.float32)
        self.last_scores = {}
        self.last_individual_masks = {}

        for concept in concept_list:
            text_inputs = self.sam3_processor(text=concept, return_tensors="pt")
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

            with torch.no_grad():
                outputs = self.sam3_model(vision_embeds=vision_embeds, **text_inputs)

            target_sizes = original_sizes.tolist() if original_sizes is not None else [[h, w]]
            results = self.sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=target_sizes,
            )

            concept_mask = np.zeros((h, w), dtype=np.float32)
            instances = []

            if results and len(results) > 0:
                result = results[0]
                if "masks" in result and len(result["masks"]) > 0:
                    scores = result.get("scores", [])
                    for i in range(len(result["masks"])):
                        mask = result["masks"][i].cpu().numpy().astype(np.float32)
                        score = float(scores[i].cpu()) if isinstance(scores[i], torch.Tensor) else float(scores[i])
                        if mask.shape != (h, w):
                            import cv2
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        concept_mask = np.maximum(concept_mask, mask)
                        instances.append((mask.copy(), score))

            if len(instances) == 0:
                self.last_scores[concept] = 0.0
                self.last_individual_masks[concept] = np.zeros((h, w), dtype=np.float32)
            elif len(instances) == 1:
                self.last_scores[concept] = instances[0][1]
                self.last_individual_masks[concept] = instances[0][0]
            else:
                for i, (inst_mask, inst_score) in enumerate(instances):
                    self.last_scores[f"{concept}_{i}"] = inst_score
                    self.last_individual_masks[f"{concept}_{i}"] = inst_mask

            combined_mask = np.maximum(combined_mask, concept_mask)

        combined_mask = (combined_mask > 0.5).astype(np.float32)
        self.last_segment_time = time.time() - start_time
        return combined_mask

    def render_debug_image(self, image: np.ndarray, label: str = "") -> np.ndarray:
        """Render a 3-panel debug image showing the GDINO+SAM3 pipeline.

        Panel 1 — GDINO Boxes: ALL bounding boxes from the GDINO query.
            - Green boxes: requested concepts that survived suppression
            - Red boxes: requested concepts suppressed by context
            - Yellow boxes: context-label detections
        Panel 2 — SAM3 Masks: colored instance masks from surviving boxes.
        Panel 3 — Combined: final binary mask overlaid on original image.

        Args:
            image: Original RGB image (H, W, 3) uint8.
            label: Optional label for the query type (e.g. "Safe-Set", "Distractor").

        Returns:
            RGB debug image (H, 3*W, 3) uint8.
        """
        import cv2

        h, w = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- Panel 1: GDINO Boxes ---
        p1 = image.copy()
        requested = set(self.last_concept_list)

        # Suppressed boxes (red, dashed via shorter segments)
        suppressed_boxes = {
            (c, tuple(box)): (ctx_label, ctx_score)
            for c, box, score, ctx_label, ctx_score in self.last_suppressed
        }

        for concept, detections in self.last_all_detections.items():
            for box, score in detections:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                key = (concept, tuple(box))

                if key in suppressed_boxes:
                    # Suppressed — red dashed
                    color = (255, 50, 50)
                    ctx_label, ctx_score = suppressed_boxes[key]
                    tag = f"{concept}:{score:.2f} X>{ctx_label}:{ctx_score:.2f}"
                    # Dashed box effect
                    for i in range(x1, x2, 8):
                        cv2.line(p1, (i, y1), (min(i + 4, x2), y1), color, 2)
                        cv2.line(p1, (i, y2), (min(i + 4, x2), y2), color, 2)
                    for i in range(y1, y2, 8):
                        cv2.line(p1, (x1, i), (x1, min(i + 4, y2)), color, 2)
                        cv2.line(p1, (x2, i), (x2, min(i + 4, y2)), color, 2)
                elif concept in requested:
                    # Surviving requested concept — green
                    color = (0, 255, 0)
                    tag = f"{concept}:{score:.2f}"
                    cv2.rectangle(p1, (x1, y1), (x2, y2), color, 2)
                else:
                    # Context label — yellow, thinner
                    color = (255, 255, 0)
                    tag = f"{concept}:{score:.2f}"
                    cv2.rectangle(p1, (x1, y1), (x2, y2), color, 1)

                # Label above box
                (tw, th), _ = cv2.getTextSize(tag, font, 0.3, 1)
                ty = max(y1 - 4, th + 2)
                cv2.rectangle(p1, (x1, ty - th - 2), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
                cv2.putText(p1, tag, (x1 + 2, ty), font, 0.3, color, 1)

        title = f"GDINO Boxes"
        if label:
            title += f" ({label})"
        cv2.putText(p1, title, (10, 25), font, 0.5, (255, 255, 255), 1)

        # Legend
        cv2.putText(p1, "green=kept  red=suppressed  yellow=context", (10, h - 10), font, 0.3, (200, 200, 200), 1)

        # --- Panel 2: SAM3 Masks ---
        p2 = image.copy()
        mask_colors = [
            (0, 255, 0), (0, 200, 255), (255, 128, 0), (255, 0, 255),
            (0, 255, 128), (128, 128, 255), (255, 255, 0), (0, 128, 255),
            (255, 0, 128), (128, 255, 0), (200, 100, 255), (100, 255, 200),
        ]
        labels_list = []
        for idx, (name, mask) in enumerate(self.last_individual_masks.items()):
            binary = mask > 0.5
            if binary.sum() == 0:
                continue
            color = mask_colors[idx % len(mask_colors)]
            overlay = p2.copy()
            overlay[binary] = color
            p2 = cv2.addWeighted(p2, 0.55, overlay, 0.45, 0)
            # Contour
            ctrs, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p2, ctrs, -1, color, 2)
            # Collect label info
            ys, xs = np.where(binary)
            cx, cy = int(xs.mean()), int(ys.mean())
            score = self.last_scores.get(name, 0.0)
            labels_list.append((name, score, cx, cy, color))

        for name, score, cx, cy, color in labels_list:
            tag = f"{name}:{score:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, font, 0.35, 1)
            cv2.rectangle(p2, (cx - 2, cy - th - 4), (cx + tw + 4, cy + 4), (0, 0, 0), -1)
            cv2.putText(p2, tag, (cx, cy), font, 0.35, color, 1)

        cv2.putText(p2, "SAM3 Masks", (10, 25), font, 0.5, (255, 255, 255), 1)

        # --- Panel 3: Combined binary mask overlay ---
        combined = np.zeros((h, w), dtype=np.float32)
        for mask in self.last_individual_masks.values():
            combined = np.maximum(combined, mask)
        combined_binary = (combined > 0.5).astype(np.uint8)

        p3 = image.copy()
        p3[combined_binary > 0] = (
            (p3[combined_binary > 0].astype(np.float32) * 0.4
             + np.array([0, 255, 0], dtype=np.float32) * 0.6)
        ).astype(np.uint8)
        ctrs, _ = cv2.findContours(combined_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p3, ctrs, -1, (0, 255, 0), 2)

        cov = combined_binary.sum() / combined_binary.size * 100
        cv2.putText(p3, f"Combined ({cov:.1f}%)", (10, 25), font, 0.5, (255, 255, 255), 1)

        return np.hstack([p1, p2, p3])


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
_grounded_sam3_singleton: Optional[GroundedSAM3Segmenter] = None


def get_grounded_sam3_segmenter(**kwargs) -> GroundedSAM3Segmenter:
    """Get or create a singleton GroundedSAM3Segmenter."""
    global _grounded_sam3_singleton
    if _grounded_sam3_singleton is None:
        _grounded_sam3_singleton = GroundedSAM3Segmenter(**kwargs)
        print("[GroundedSAM3] Created singleton GroundedSAM3Segmenter")
    return _grounded_sam3_singleton


def clear_grounded_sam3_singleton():
    """Clear the singleton instance."""
    global _grounded_sam3_singleton
    _grounded_sam3_singleton = None
