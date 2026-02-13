#!/usr/bin/env python3
"""
SAM3 Segmentation Server

Runs SAM3 in a separate process to avoid transformers version conflicts.
Communicates via a simple file-based protocol.

Usage:
    # In a separate terminal with transformers >= 5.0.0:
    python scripts/sam3_server.py

    # Or run in background:
    python scripts/sam3_server.py &
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    # Check transformers version
    import transformers
    version = transformers.__version__
    print(f"[SAM3 Server] Transformers version: {version}")

    major_version = int(version.split('.')[0])
    if major_version < 5:
        print(f"[SAM3 Server] ERROR: SAM3 requires transformers >= 5.0.0, got {version}")
        print("[SAM3 Server] Install with: pip install git+https://github.com/huggingface/transformers.git@main")
        sys.exit(1)

    from transformers import Sam3Model, Sam3Processor
    import torch

    # Initialize model
    print("[SAM3 Server] Loading SAM3 model...")
    model_name = "facebook/sam3"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Sam3Processor.from_pretrained(model_name)
    model = Sam3Model.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    print(f"[SAM3 Server] Model loaded on {device}")

    # Communication directory
    comm_dir = Path("/tmp/sam3_server")
    comm_dir.mkdir(exist_ok=True)
    request_file = comm_dir / "request.json"
    response_file = comm_dir / "response.npz"
    ready_file = comm_dir / "ready"

    # Signal ready
    ready_file.touch()
    print(f"[SAM3 Server] Ready. Listening for requests in {comm_dir}")

    while True:
        try:
            # Wait for request
            if request_file.exists():
                with open(request_file, 'r') as f:
                    request = json.load(f)

                # Remove request file to signal processing
                request_file.unlink()

                # Load image
                image_path = request['image_path']
                concepts = request['concepts']
                threshold = request.get('threshold', 0.5)

                from PIL import Image
                image = Image.open(image_path).convert('RGB')

                print(f"[SAM3 Server] Processing: {concepts}")

                # Run segmentation
                combined_mask = np.zeros((image.height, image.width), dtype=bool)
                concept_masks = {}   # {concept: mask}
                concept_scores = {}  # {concept: best_score}

                for concept in concepts:
                    inputs = processor(
                        images=image,
                        text=concept,
                        return_tensors="pt"
                    ).to(device)

                    original_sizes = inputs.get("original_sizes")

                    with torch.no_grad():
                        outputs = model(**inputs)

                    # Use post_process_instance_segmentation for SAM3
                    target_sizes = original_sizes.tolist() if original_sizes is not None else [[image.height, image.width]]
                    results = processor.post_process_instance_segmentation(
                        outputs,
                        threshold=threshold,
                        mask_threshold=0.3,
                        target_sizes=target_sizes,
                    )

                    concept_mask = np.zeros((image.height, image.width), dtype=bool)
                    best_score = 0.0

                    if results and len(results) > 0:
                        result = results[0]
                        if "masks" in result and len(result["masks"]) > 0:
                            scores = result.get("scores", torch.ones(len(result["masks"])))
                            for i, mask_tensor in enumerate(result["masks"]):
                                score = float(scores[i].cpu()) if isinstance(scores[i], torch.Tensor) else float(scores[i])
                                if score > threshold:
                                    mask_np = mask_tensor.cpu().numpy().astype(bool)
                                    if mask_np.ndim == 3:
                                        mask_np = mask_np[0]
                                    concept_mask |= mask_np
                                    best_score = max(best_score, score)
                                    combined_mask |= mask_np
                                    print(f"[SAM3 Server] Concept '{concept}': score={score:.3f}")

                    concept_masks[concept] = concept_mask
                    concept_scores[concept] = best_score

                # Save response with per-concept data
                save_dict = {'mask': combined_mask}
                for concept, cmask in concept_masks.items():
                    save_dict[f'mask_{concept}'] = cmask
                    save_dict[f'score_{concept}'] = np.array(concept_scores[concept])
                np.savez_compressed(response_file, **save_dict)
                print(f"[SAM3 Server] Done. Mask coverage: {combined_mask.sum() / combined_mask.size * 100:.1f}%")

            time.sleep(0.01)  # 10ms polling

        except KeyboardInterrupt:
            print("\n[SAM3 Server] Shutting down...")
            ready_file.unlink(missing_ok=True)
            break
        except Exception as e:
            print(f"[SAM3 Server] Error: {e}")
            # Save error response
            np.savez_compressed(response_file, mask=np.zeros((1, 1), dtype=bool), error=str(e))


if __name__ == "__main__":
    main()
