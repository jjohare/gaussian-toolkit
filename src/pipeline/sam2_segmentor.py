# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core SAM2 segmentation with automatic mask generation and video tracking.

Supports two modes:
  1. Automatic mask generation per frame (no prompts needed).
  2. Prompted video segmentation with consistent object IDs across frames.

Typical usage::

    seg = SAM2Segmentor.from_pretrained("facebook/sam2-hiera-large")
    masks, ids = seg.segment_video_auto(frame_dir)
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """Per-frame segmentation output.

    Attributes:
        frame_idx: Zero-based frame index.
        masks: Binary masks, shape ``(N, H, W)`` where *N* = object count.
        object_ids: Integer label for each mask, consistent across frames.
        scores: Confidence score per mask in ``[0, 1]``.
    """
    frame_idx: int
    masks: np.ndarray  # (N, H, W) bool
    object_ids: np.ndarray  # (N,) int
    scores: np.ndarray  # (N,) float


@dataclass
class PromptSpec:
    """A user-supplied prompt for a single object on a single frame.

    Provide *either* ``points``+``labels`` or ``box``, or both.
    """
    frame_idx: int
    object_id: int
    points: Optional[np.ndarray] = None  # (K, 2) xy
    labels: Optional[np.ndarray] = None  # (K,) 0=bg, 1=fg
    box: Optional[np.ndarray] = None  # (4,) xyxy


class SAM2Segmentor:
    """SAM2-based image and video segmentation engine.

    Parameters:
        model_id: HuggingFace model identifier, e.g. ``"facebook/sam2-hiera-large"``.
        device: PyTorch device string.
        points_per_side: Grid density for automatic mask generation.
        pred_iou_thresh: Minimum predicted IoU to keep a mask.
        stability_score_thresh: Minimum mask stability score.
        min_mask_region_area: Discard masks smaller than this pixel count.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.80,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
    ) -> None:
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area

        self._image_predictor: object | None = None
        self._video_predictor: object | None = None
        self._mask_generator: object | None = None

    # ------------------------------------------------------------------
    #  Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "facebook/sam2-hiera-large",
        device: str = "cuda",
        **kwargs,
    ) -> "SAM2Segmentor":
        """Convenience constructor matching the HF ``from_pretrained`` pattern."""
        return cls(model_id=model_id, device=device, **kwargs)

    # ------------------------------------------------------------------
    #  Lazy model loading
    # ------------------------------------------------------------------
    def _get_image_predictor(self):
        if self._image_predictor is None:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            logger.info("Loading SAM2 image predictor: %s", self.model_id)
            self._image_predictor = SAM2ImagePredictor.from_pretrained(
                self.model_id, device=self.device,
            )
        return self._image_predictor

    def _get_video_predictor(self):
        if self._video_predictor is None:
            from sam2.sam2_video_predictor import SAM2VideoPredictor

            logger.info("Loading SAM2 video predictor: %s", self.model_id)
            self._video_predictor = SAM2VideoPredictor.from_pretrained(
                self.model_id, device=self.device,
            )
        return self._video_predictor

    def _get_mask_generator(self):
        if self._mask_generator is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2

            predictor = self._get_image_predictor()
            self._mask_generator = SAM2AutomaticMaskGenerator(
                model=predictor.model,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                min_mask_region_area=self.min_mask_region_area,
            )
        return self._mask_generator

    # ------------------------------------------------------------------
    #  Automatic per-image mask generation
    # ------------------------------------------------------------------
    def generate_masks_single(self, image: np.ndarray) -> SegmentationResult:
        """Run automatic mask generation on a single RGB image.

        Args:
            image: ``(H, W, 3)`` uint8 RGB array.

        Returns:
            SegmentationResult with auto-assigned object IDs (1-indexed).
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB image, got shape {image.shape}")

        gen = self._get_mask_generator()
        raw_masks: List[dict] = gen.generate(image)

        if not raw_masks:
            h, w = image.shape[:2]
            return SegmentationResult(
                frame_idx=0,
                masks=np.empty((0, h, w), dtype=bool),
                object_ids=np.empty(0, dtype=np.int32),
                scores=np.empty(0, dtype=np.float32),
            )

        # Sort by area descending so the largest object gets ID 1
        raw_masks.sort(key=lambda m: m["area"], reverse=True)

        masks = np.stack([m["segmentation"].astype(bool) for m in raw_masks])
        scores = np.array([m["predicted_iou"] for m in raw_masks], dtype=np.float32)
        obj_ids = np.arange(1, len(raw_masks) + 1, dtype=np.int32)

        return SegmentationResult(
            frame_idx=0, masks=masks, object_ids=obj_ids, scores=scores,
        )

    # ------------------------------------------------------------------
    #  Automatic video segmentation (frame-by-frame with IoU tracking)
    # ------------------------------------------------------------------
    def segment_video_auto(
        self,
        frame_dir: str | Path,
        *,
        iou_threshold: float = 0.30,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
    ) -> List[SegmentationResult]:
        """Segment every frame independently, then associate IDs across frames
        via greedy IoU matching.

        Args:
            frame_dir: Directory containing video frames sorted by name.
            iou_threshold: Minimum IoU to consider two masks the same object.
            extensions: Accepted image file extensions.

        Returns:
            List of per-frame ``SegmentationResult`` with consistent object IDs.
        """
        import cv2

        frame_dir = Path(frame_dir)
        frame_paths = sorted(
            p for p in frame_dir.iterdir()
            if p.suffix.lower() in extensions
        )
        if not frame_paths:
            raise FileNotFoundError(f"No image files in {frame_dir}")

        logger.info("Auto-segmenting %d frames from %s", len(frame_paths), frame_dir)

        results: List[SegmentationResult] = []
        prev_masks: np.ndarray | None = None
        prev_ids: np.ndarray | None = None
        next_free_id = 1

        for idx, fp in enumerate(frame_paths):
            image = cv2.imread(str(fp))
            if image is None:
                logger.warning("Failed to read %s, skipping", fp)
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = self.generate_masks_single(image)

            if result.masks.shape[0] == 0:
                results.append(SegmentationResult(
                    frame_idx=idx,
                    masks=result.masks,
                    object_ids=result.object_ids,
                    scores=result.scores,
                ))
                prev_masks = None
                prev_ids = None
                continue

            # Track IDs across frames via IoU matching
            if prev_masks is not None and prev_masks.shape[0] > 0:
                matched_ids, next_free_id = self._match_masks_iou(
                    prev_masks, prev_ids, result.masks, next_free_id, iou_threshold,
                )
            else:
                n = result.masks.shape[0]
                matched_ids = np.arange(next_free_id, next_free_id + n, dtype=np.int32)
                next_free_id += n

            tracked = SegmentationResult(
                frame_idx=idx,
                masks=result.masks,
                object_ids=matched_ids,
                scores=result.scores,
            )
            results.append(tracked)
            prev_masks = result.masks
            prev_ids = matched_ids

        logger.info("Segmentation complete: %d frames, %d unique IDs",
                     len(results), next_free_id - 1)
        return results

    # ------------------------------------------------------------------
    #  Prompted video segmentation (SAM2 video predictor)
    # ------------------------------------------------------------------
    def segment_video_prompted(
        self,
        frame_dir: str | Path,
        prompts: List[PromptSpec],
        *,
        offload_to_cpu: bool = False,
    ) -> List[SegmentationResult]:
        """Segment a video using user-supplied point/box prompts.

        Uses the native SAM2 video predictor which propagates masks forward
        and backward from prompted keyframes.

        Args:
            frame_dir: Directory of JPEG frames (SAM2 reads from disk).
            prompts: One or more ``PromptSpec`` indicating objects to track.
            offload_to_cpu: Offload video state to CPU to save GPU memory.

        Returns:
            Per-frame ``SegmentationResult`` in frame order.
        """
        frame_dir = Path(frame_dir)
        predictor = self._get_video_predictor()

        inference_state = predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=offload_to_cpu,
            offload_state_to_cpu=offload_to_cpu,
        )

        # Register each prompt
        for p in prompts:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=p.frame_idx,
                obj_id=p.object_id,
                points=p.points,
                labels=p.labels,
                box=p.box,
            )

        # Propagate forward
        results_dict: Dict[int, Dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids_out, masks_out in predictor.propagate_in_video(
            inference_state
        ):
            # masks_out: dict mapping obj_id -> (1, H, W) logits
            results_dict[frame_idx] = {}
            for oid, mask_logits in zip(obj_ids_out, masks_out):
                results_dict[frame_idx][oid] = (mask_logits.squeeze(0).cpu().numpy() > 0.0)

        # Propagate backward from earliest prompted frame
        min_prompt_frame = min(p.frame_idx for p in prompts)
        if min_prompt_frame > 0:
            for frame_idx, obj_ids_out, masks_out in predictor.propagate_in_video(
                inference_state, reverse=True
            ):
                if frame_idx not in results_dict:
                    results_dict[frame_idx] = {}
                for oid, mask_logits in zip(obj_ids_out, masks_out):
                    if oid not in results_dict[frame_idx]:
                        results_dict[frame_idx][oid] = (
                            mask_logits.squeeze(0).cpu().numpy() > 0.0
                        )

        # Assemble into sorted SegmentationResult list
        results: List[SegmentationResult] = []
        for fidx in sorted(results_dict.keys()):
            frame_data = results_dict[fidx]
            if not frame_data:
                continue
            oids = sorted(frame_data.keys())
            masks = np.stack([frame_data[o] for o in oids])
            results.append(SegmentationResult(
                frame_idx=fidx,
                masks=masks,
                object_ids=np.array(oids, dtype=np.int32),
                scores=np.ones(len(oids), dtype=np.float32),
            ))

        predictor.reset_state(inference_state)
        return results

    # ------------------------------------------------------------------
    #  Prompted single-image segmentation
    # ------------------------------------------------------------------
    def segment_image_prompted(
        self,
        image: np.ndarray,
        *,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run SAM2 image prediction with point/box prompts.

        Args:
            image: ``(H, W, 3)`` uint8 RGB.
            points: ``(K, 2)`` xy coordinates.
            labels: ``(K,)`` 0=background, 1=foreground.
            box: ``(4,)`` xyxy bounding box.
            multimask: Return three quality levels instead of one.

        Returns:
            Tuple of ``(masks, scores, logits)`` where masks is
            ``(M, H, W)`` bool and M is 1 or 3.
        """
        predictor = self._get_image_predictor()
        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            multimask_output=multimask,
        )
        predictor.reset_predictor()
        return masks.astype(bool), scores, logits

    # ------------------------------------------------------------------
    #  Internal: greedy IoU matching for frame-to-frame ID tracking
    # ------------------------------------------------------------------
    @staticmethod
    def _match_masks_iou(
        prev_masks: np.ndarray,
        prev_ids: np.ndarray,
        curr_masks: np.ndarray,
        next_free_id: int,
        threshold: float,
    ) -> Tuple[np.ndarray, int]:
        """Match current masks to previous masks via IoU, greedy assignment.

        Returns matched ID array and updated next_free_id.
        """
        n_prev = prev_masks.shape[0]
        n_curr = curr_masks.shape[0]

        # Flatten masks for fast intersection/union computation
        prev_flat = prev_masks.reshape(n_prev, -1).astype(np.float32)
        curr_flat = curr_masks.reshape(n_curr, -1).astype(np.float32)

        # IoU matrix: (n_prev, n_curr)
        intersection = prev_flat @ curr_flat.T
        prev_areas = prev_flat.sum(axis=1, keepdims=True)
        curr_areas = curr_flat.sum(axis=1, keepdims=True)
        union = prev_areas + curr_areas.T - intersection
        iou = np.where(union > 0, intersection / union, 0.0)

        matched_ids = np.full(n_curr, -1, dtype=np.int32)
        used_prev = set()

        # Greedy: pick best IoU pairs in descending order
        flat_order = np.argsort(iou.ravel())[::-1]
        for flat_idx in flat_order:
            pi = int(flat_idx // n_curr)
            ci = int(flat_idx % n_curr)
            if iou[pi, ci] < threshold:
                break
            if pi in used_prev or matched_ids[ci] != -1:
                continue
            matched_ids[ci] = prev_ids[pi]
            used_prev.add(pi)

        # Assign new IDs to unmatched
        for ci in range(n_curr):
            if matched_ids[ci] == -1:
                matched_ids[ci] = next_free_id
                next_free_id += 1

        return matched_ids, next_free_id

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    def to(self, device: str) -> "SAM2Segmentor":
        """Move models to a different device (clears cached predictors)."""
        self.device = device
        self._image_predictor = None
        self._video_predictor = None
        self._mask_generator = None
        return self

    def unload(self) -> None:
        """Release GPU memory by discarding cached models."""
        self._image_predictor = None
        self._video_predictor = None
        self._mask_generator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
