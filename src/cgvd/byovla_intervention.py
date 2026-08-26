"""BYOVLA reimplementation for the SimplerEnv + pi0 evaluation stack.

Faithful adaptation of the public BYOVLA reference implementation
(Hancock, Ren & Majumdar, ICRA 2025; github.com/irom-princeton/byovla) with
three documented substitutions, chosen to remove confounds against CGVD and
external-API dependencies:

  1. VLM (GPT-4o) -> the same closed distractor vocabulary D that CGVD
     receives (vocabulary-matched comparison; isolates the intervention
     mechanism from VLM quality and needs no API).
  2. GroundingDINO + SAM2 -> SAM 3 instance segmentation (shared with CGVD).
  3. Octo sample_actions with a fixed PRNGKey -> pi0 forward passes with a
     fixed torch seed (deterministic clean-vs-perturbed comparison, N=1;
     the reference used N=5 stochastic samples with a fixed key).

Kept faithful to the reference:
  - Per-object Gaussian-blur perturbation with random kernel size in [15, 30]
    (perturb_gaussian_blur), applied to the dilated object mask.
  - Sensitivity metric: delta = mean over the action chunk of
    sqrt(sum_d w_d * (a_clean - a_perturbed)^2) computed on UNNORMALIZED
    end-effector actions with w = [1,1,1,0,0,0,0] (translation only), and
    threshold thresh = 0.002 (2 mm), as in the reference main().
  - Only objects with delta >= thresh are inpainted (union of masks dilated
    by dilate_size = 10), via LaMa.
  - The intervention (segment -> probe -> inpaint) runs before EVERY policy
    inference call (every action chunk), as in the reference main loop.
  - The reference's warm_filter (Octo-specific color filter) is omitted.
"""
import time
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np


def perturb_gaussian_blur(image: np.ndarray, mask: np.ndarray, kernel_size: int = 25) -> np.ndarray:
    """Blur the masked region of the image (reference implementation semantics)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    m = (mask > 0.5).astype(np.float32)[..., None]
    out = image.astype(np.float32) * (1 - m) + blurred.astype(np.float32) * m
    return out.astype(np.uint8)


def dilate_mask(mask: np.ndarray, dilate_size: int) -> np.ndarray:
    if dilate_size <= 0:
        return (mask > 0.5).astype(np.uint8)
    kern = np.ones((dilate_size, dilate_size), np.uint8)
    return cv2.dilate((mask > 0.5).astype(np.uint8), kern, iterations=1)


class BYOVLAIntervention:
    """Run-time observation intervention (BYOVLA) around a frozen policy.

    Args:
        segmenter: SAM3Segmenter instance (shared with CGVD).
        inpainter: LamaInpainter instance (shared with CGVD).
        distractor_names: closed vocabulary D (stands in for the VLM output).
        distractor_threshold: SAM3 presence threshold for D (same as CGVD).
        thresh: sensitivity threshold on the translation-action delta (m).
        w: per-dimension weights on the action delta.
        dilate_size: inpainting mask dilation (reference: 10 px).
        probe_seed: torch seed fixed before every probe forward pass.
        rng: numpy RandomState for the random blur kernel.
    """

    def __init__(
        self,
        segmenter,
        inpainter,
        distractor_names: List[str],
        distractor_threshold: float = 0.20,
        thresh: float = 0.002,
        w: Optional[np.ndarray] = None,
        dilate_size: int = 10,
        probe_seed: int = 0,
        verbose: bool = False,
    ):
        self.segmenter = segmenter
        self.inpainter = inpainter
        self.distractor_names = list(distractor_names)
        self.distractor_threshold = distractor_threshold
        self.thresh = thresh
        self.w = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64)
        self.dilate_size = dilate_size
        self.probe_seed = probe_seed
        self.verbose = verbose
        self.rng = np.random.RandomState(probe_seed)
        # rolling stats (per episode; reset() clears)
        self.chunk_stats: List[Dict] = []

    def reset(self):
        self.chunk_stats = []

    def _delta(self, clean: np.ndarray, pert: np.ndarray) -> float:
        """Reference metric: mean over chunk of sqrt(sum_d w*(dA)^2)."""
        n = min(len(clean), len(pert))
        d = clean[:n] - pert[:n]                        # [n_steps, dims]
        dims = min(d.shape[1], len(self.w))
        dsq = np.square(d[:, :dims]) * self.w[:dims]
        return float(np.mean(np.sqrt(np.sum(dsq, axis=1))))

    def transform(
        self,
        image: np.ndarray,
        probe_fn: Callable[[np.ndarray, Optional[int]], np.ndarray],
    ) -> np.ndarray:
        """Segment D, probe per-object sensitivity, inpaint sensitive objects.

        probe_fn(image, seed) must run the frozen policy on `image` with the
        given fixed seed and return the UNNORMALIZED action chunk
        [n_steps, dims] (translation in meters first).
        """
        t_start = time.time()
        h, w_px = image.shape[:2]

        # --- Step 1 (VLM substitute) + segmentation ---
        t_seg = time.time()
        concepts = ". ".join(self.distractor_names)
        _ = self.segmenter.segment(
            image, concepts,
            return_individual_masks=True,
            presence_threshold=self.distractor_threshold,
        )
        individual = dict(getattr(self.segmenter, "last_individual_masks", {}) or {})
        seg_ms = (time.time() - t_seg) * 1000

        regions = [(name, m) for name, m in individual.items() if (m > 0.5).sum() > 0]

        # --- Step 2: sensitivity probe ---
        t_probe = time.time()
        clean_actions = probe_fn(image, self.probe_seed)
        deltas, sensitive = {}, []
        for name, mask in regions:
            kernel = int(self.rng.randint(15, 30))
            mask_d = dilate_mask(mask, 0)  # reference uses dilate_size=0 for probing
            pert_img = perturb_gaussian_blur(image, mask_d, kernel_size=kernel)
            pert_actions = probe_fn(pert_img, self.probe_seed)
            delta = self._delta(clean_actions, pert_actions)
            deltas[name] = delta
            if delta >= self.thresh:
                sensitive.append((name, mask))
        probe_ms = (time.time() - t_probe) * 1000

        # --- Step 3: inpaint sensitive objects ---
        t_inp = time.time()
        out = image
        if sensitive:
            union = np.zeros((h, w_px), dtype=np.uint8)
            for _, mask in sensitive:
                union = np.logical_or(union, dilate_mask(mask, self.dilate_size)).astype(np.uint8)
            out = self.inpainter.inpaint(image, union.astype(np.float32), dilate_mask=0)
        inpaint_ms = (time.time() - t_inp) * 1000

        total_ms = (time.time() - t_start) * 1000
        self.chunk_stats.append({
            "regions": len(regions), "sensitive": len(sensitive),
            "seg_ms": seg_ms, "probe_ms": probe_ms,
            "inpaint_ms": inpaint_ms, "total_ms": total_ms,
            "deltas": {k: round(v, 5) for k, v in deltas.items()},
        })
        print(f"[BYOVLA] regions={len(regions)} sensitive={len(sensitive)} "
              f"seg_ms={seg_ms:.0f} probe_ms={probe_ms:.0f} inpaint_ms={inpaint_ms:.0f} "
              f"total_ms={total_ms:.0f}")
        return out
