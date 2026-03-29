#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test SAM3 segmentation pipeline.

Validates:
  1. SAM3 package installation and model loading.
  2. Text-prompted concept segmentation on a synthetic image.
  3. Multi-concept segmentation with unified result.
  4. Backward-compatible automatic mask generation (SAM2 fallback).
  5. Video concept segmentation (if frame directory provided).

Usage:
    python scripts/test_sam3_pipeline.py [--frames-dir /path/to/frames] [--device cuda]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("test_sam3")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(project_root))


def _create_test_image(height: int = 480, width: int = 640) -> np.ndarray:
    """Create a synthetic test image with colored rectangles on a gray background."""
    image = np.full((height, width, 3), 180, dtype=np.uint8)  # gray background

    # Red rectangle (simulating a painting)
    image[50:200, 100:300] = [220, 40, 40]

    # Blue rectangle (simulating furniture)
    image[250:400, 350:550] = [40, 40, 220]

    # Green rectangle (simulating a plant)
    image[100:250, 400:520] = [40, 180, 40]

    # Yellow rectangle (simulating a frame)
    image[300:380, 50:200] = [220, 200, 40]

    return image


def test_sam3_availability() -> bool:
    """Test 1: Check SAM3 package availability."""
    logger.info("--- Test 1: SAM3 availability ---")
    try:
        from pipeline.sam3_segmentor import SAM3Segmentor
        available = SAM3Segmentor.is_sam3_available()
        logger.info("SAM3 available: %s", available)

        if available:
            import sam3
            logger.info("SAM3 version: %s", sam3.__version__)
            from sam3.model_builder import build_sam3_image_model
            logger.info("SAM3 model builder importable: True")
        return available
    except Exception as exc:
        logger.error("SAM3 availability check failed: %s", exc)
        return False


def test_concept_segmentation(device: str) -> bool:
    """Test 2: Text-prompted concept segmentation on a synthetic image."""
    logger.info("--- Test 2: Concept segmentation ---")
    try:
        from pipeline.sam3_segmentor import SAM3Segmentor

        seg = SAM3Segmentor(device=device, confidence_threshold=0.3)
        image = _create_test_image()

        t0 = time.time()
        result = seg.segment_by_concept(image, "colored rectangles")
        dt = time.time() - t0

        logger.info(
            "Concept 'colored rectangles': %d masks found in %.2fs",
            result.masks.shape[0], dt,
        )
        logger.info("  Scores: %s", result.scores)
        logger.info("  Mask shapes: %s", result.masks.shape)

        if result.masks.shape[0] > 0:
            logger.info("  Boxes (xyxy): %s", result.boxes[:3])

        seg.unload()
        logger.info("Test 2 PASSED")
        return True
    except Exception as exc:
        logger.error("Test 2 FAILED: %s", exc, exc_info=True)
        return False


def test_multi_concept_segmentation(device: str) -> bool:
    """Test 3: Multi-concept segmentation with unified result."""
    logger.info("--- Test 3: Multi-concept segmentation ---")
    try:
        from pipeline.sam3_segmentor import SAM3Segmentor

        seg = SAM3Segmentor(device=device, confidence_threshold=0.3)
        image = _create_test_image()

        concepts = ["red objects", "blue objects", "green objects"]
        t0 = time.time()
        result, id_to_concept = seg.segment_by_concepts(image, concepts)
        dt = time.time() - t0

        logger.info(
            "Multi-concept: %d total masks in %.2fs",
            result.masks.shape[0], dt,
        )
        logger.info("  Object IDs: %s", result.object_ids)
        logger.info("  ID->Concept map: %s", id_to_concept)

        # Verify SegmentationResult format
        assert result.masks.dtype == bool, f"Expected bool masks, got {result.masks.dtype}"
        assert result.object_ids.dtype == np.int32
        assert result.scores.dtype == np.float32

        seg.unload()
        logger.info("Test 3 PASSED")
        return True
    except Exception as exc:
        logger.error("Test 3 FAILED: %s", exc, exc_info=True)
        return False


def test_sam2_fallback(device: str) -> bool:
    """Test 4: Backward-compatible automatic mask generation via SAM2."""
    logger.info("--- Test 4: SAM2 fallback (auto masks) ---")
    try:
        from pipeline.sam3_segmentor import SAM3Segmentor

        seg = SAM3Segmentor(device=device)
        image = _create_test_image()

        t0 = time.time()
        result = seg.generate_masks_single(image)
        dt = time.time() - t0

        logger.info(
            "Auto masks (SAM2 fallback): %d masks in %.2fs",
            result.masks.shape[0], dt,
        )
        logger.info("  Object IDs: %s", result.object_ids[:10])
        logger.info("  Top scores: %s", result.scores[:5])

        assert isinstance(result.masks, np.ndarray)
        assert result.masks.dtype == bool
        assert result.frame_idx == 0

        seg.unload()
        logger.info("Test 4 PASSED")
        return True
    except Exception as exc:
        logger.error("Test 4 FAILED: %s", exc, exc_info=True)
        return False


def test_video_concept_segmentation(device: str, frames_dir: str) -> bool:
    """Test 5: Video concept segmentation with SAM3."""
    logger.info("--- Test 5: Video concept segmentation ---")
    if not frames_dir:
        logger.info("No --frames-dir provided, skipping video test")
        return True

    try:
        from pipeline.sam3_segmentor import SAM3Segmentor

        seg = SAM3Segmentor(device=device, confidence_threshold=0.3)

        concepts = ["paintings", "furniture", "walls"]
        t0 = time.time()
        results = seg.segment_video_by_concepts(frames_dir, concepts)
        dt = time.time() - t0

        logger.info(
            "Video concept segmentation: %d frames in %.2fs",
            len(results), dt,
        )
        if results:
            first = results[0]
            logger.info(
                "  Frame 0: %d objects, IDs=%s",
                first.masks.shape[0], first.object_ids,
            )

        seg.unload()
        logger.info("Test 5 PASSED")
        return True
    except Exception as exc:
        logger.error("Test 5 FAILED: %s", exc, exc_info=True)
        return False


def test_mask_projector_compatibility(device: str) -> bool:
    """Test 6: Verify SAM3 output is compatible with MaskProjector."""
    logger.info("--- Test 6: MaskProjector compatibility ---")
    try:
        from pipeline.sam3_segmentor import SAM3Segmentor
        from pipeline.mask_projector import MaskProjector

        seg = SAM3Segmentor(device=device, confidence_threshold=0.3)
        image = _create_test_image()

        concepts = ["colored objects"]
        result, _ = seg.segment_by_concepts(image, concepts)

        # Test conversion to label maps
        label_maps = MaskProjector.segmentation_results_to_label_maps(
            [result], ["test_frame.jpg"],
        )

        if "test_frame.jpg" in label_maps:
            lm = label_maps["test_frame.jpg"]
            unique_labels = np.unique(lm)
            logger.info("  Label map shape: %s, unique labels: %s", lm.shape, unique_labels)
        else:
            logger.info("  No masks produced (empty result) -- still compatible")

        seg.unload()
        logger.info("Test 6 PASSED")
        return True
    except Exception as exc:
        logger.error("Test 6 FAILED: %s", exc, exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Test SAM3 segmentation pipeline")
    parser.add_argument("--frames-dir", type=str, default="",
                        help="Directory of video frames for video segmentation test")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SAM3 Pipeline Test Suite")
    logger.info("=" * 60)

    results = {}

    # Test 1: Availability (always run)
    results["availability"] = test_sam3_availability()

    sam3_ok = results["availability"]

    # Tests 2-3: SAM3 concept segmentation (only if SAM3 available)
    if sam3_ok:
        results["concept_seg"] = test_concept_segmentation(args.device)
        results["multi_concept"] = test_multi_concept_segmentation(args.device)
    else:
        logger.info("Skipping SAM3-specific tests (SAM3 not available)")
        results["concept_seg"] = None
        results["multi_concept"] = None

    # Test 4: SAM2 fallback (always test)
    results["sam2_fallback"] = test_sam2_fallback(args.device)

    # Test 5: Video concept segmentation (only if SAM3 available + frames provided)
    if sam3_ok:
        results["video_concept"] = test_video_concept_segmentation(args.device, args.frames_dir)
    else:
        results["video_concept"] = None

    # Test 6: MaskProjector compatibility
    if sam3_ok:
        results["projector_compat"] = test_mask_projector_compatibility(args.device)
    else:
        results["projector_compat"] = None

    # Summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    all_passed = True
    for name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASSED"
        else:
            status = "FAILED"
            all_passed = False
        logger.info("  %-25s %s", name, status)

    if all_passed:
        logger.info("All tests passed.")
        return 0
    else:
        logger.error("Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
