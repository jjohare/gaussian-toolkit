#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validation script for the SAM2 video segmentation pipeline.

Tests imports, data structures, frame quality metrics, mask projection
logic, and (optionally) SAM2 model loading when a GPU is available.

Run from the project root::

    python scripts/test_sam2_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports() -> None:
    """Verify all pipeline modules import cleanly."""
    from src.pipeline.sam2_segmentor import SAM2Segmentor, SegmentationResult, PromptSpec
    from src.pipeline.mask_projector import MaskProjector, ViewProjection
    from src.pipeline.frame_quality import FrameQualityAssessor, FrameQuality, Recommendation

    # Also test lazy imports via __init__
    from src.pipeline import (
        SAM2Segmentor as S1,
        MaskProjector as M1,
        FrameQualityAssessor as F1,
    )

    assert S1 is SAM2Segmentor
    assert M1 is MaskProjector
    assert F1 is FrameQualityAssessor
    print("[PASS] All imports successful")


def test_segmentation_result_dataclass() -> None:
    """Verify SegmentationResult holds numpy arrays correctly."""
    from src.pipeline.sam2_segmentor import SegmentationResult

    masks = np.random.rand(3, 64, 64) > 0.5
    ids = np.array([1, 2, 3], dtype=np.int32)
    scores = np.array([0.95, 0.88, 0.75], dtype=np.float32)

    result = SegmentationResult(frame_idx=0, masks=masks, object_ids=ids, scores=scores)
    assert result.masks.shape == (3, 64, 64)
    assert result.object_ids.shape == (3,)
    assert result.frame_idx == 0
    print("[PASS] SegmentationResult dataclass works")


def test_prompt_spec() -> None:
    """Verify PromptSpec creation with points and box."""
    from src.pipeline.sam2_segmentor import PromptSpec

    p = PromptSpec(
        frame_idx=5,
        object_id=1,
        points=np.array([[100, 200], [150, 250]], dtype=np.float32),
        labels=np.array([1, 1], dtype=np.int32),
    )
    assert p.frame_idx == 5
    assert p.points.shape == (2, 2)

    p_box = PromptSpec(
        frame_idx=0,
        object_id=2,
        box=np.array([10, 20, 300, 400], dtype=np.float32),
    )
    assert p_box.box.shape == (4,)
    print("[PASS] PromptSpec creation works")


def test_iou_matching() -> None:
    """Test the internal greedy IoU matching logic."""
    from src.pipeline.sam2_segmentor import SAM2Segmentor

    # Create two sets of masks with known overlap
    h, w = 100, 100
    prev = np.zeros((2, h, w), dtype=bool)
    prev[0, 10:50, 10:50] = True  # Object A: top-left square
    prev[1, 60:90, 60:90] = True  # Object B: bottom-right square

    curr = np.zeros((2, h, w), dtype=bool)
    curr[0, 12:52, 12:52] = True  # Shifted version of A
    curr[1, 62:92, 62:92] = True  # Shifted version of B

    prev_ids = np.array([10, 20], dtype=np.int32)

    matched, next_id = SAM2Segmentor._match_masks_iou(
        prev, prev_ids, curr, next_free_id=30, threshold=0.3,
    )

    # Both should match their predecessors due to high overlap
    assert matched[0] == 10, f"Expected 10, got {matched[0]}"
    assert matched[1] == 20, f"Expected 20, got {matched[1]}"
    print("[PASS] IoU matching assigns correct IDs")


def test_iou_matching_new_object() -> None:
    """Ensure unmatched masks get new IDs."""
    from src.pipeline.sam2_segmentor import SAM2Segmentor

    h, w = 100, 100
    prev = np.zeros((1, h, w), dtype=bool)
    prev[0, 10:50, 10:50] = True

    curr = np.zeros((2, h, w), dtype=bool)
    curr[0, 10:50, 10:50] = True  # Same as prev
    curr[1, 70:90, 70:90] = True  # Brand new object

    prev_ids = np.array([5], dtype=np.int32)
    matched, next_id = SAM2Segmentor._match_masks_iou(
        prev, prev_ids, curr, next_free_id=10, threshold=0.3,
    )

    assert matched[0] == 5
    assert matched[1] == 10  # New ID assigned
    assert next_id == 11
    print("[PASS] New objects get fresh IDs")


def test_view_projection() -> None:
    """Test 3D-to-2D projection with a simple pinhole camera."""
    from src.pipeline.mask_projector import ViewProjection

    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    R = np.eye(3)
    t = np.zeros(3)

    view = ViewProjection(
        image_name="test.jpg", R=R, t=t, K=K, width=640, height=480,
    )

    # A point at (0, 0, 10) should project to the principal point (320, 240)
    xyz = np.array([[0.0, 0.0, 10.0]])
    pixels, visible = view.project(xyz)

    assert visible[0], "Point should be visible"
    assert pixels[0, 0] == 320, f"Expected col=320, got {pixels[0, 0]}"
    assert pixels[0, 1] == 240, f"Expected row=240, got {pixels[0, 1]}"

    # A point behind the camera should not be visible
    xyz_behind = np.array([[0.0, 0.0, -5.0]])
    _, vis2 = view.project(xyz_behind)
    assert not vis2[0], "Point behind camera should not be visible"

    print("[PASS] ViewProjection projects correctly")


def test_mask_projector_assign_labels() -> None:
    """Test end-to-end label assignment with synthetic data."""
    from src.pipeline.mask_projector import MaskProjector, ViewProjection

    K = np.array([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0],
    ])

    # Two views: identity and slightly rotated
    views = [
        ViewProjection("frame_0.jpg", np.eye(3), np.zeros(3), K, 100, 100),
        ViewProjection("frame_1.jpg", np.eye(3), np.array([0.1, 0.0, 0.0]), K, 100, 100),
    ]

    projector = MaskProjector.__new__(MaskProjector)
    projector.views = views
    projector._image_name_to_view_idx = {v.image_name: i for i, v in enumerate(views)}

    # Create label maps: object 1 in top-left quadrant, object 2 in bottom-right
    label_map = np.zeros((100, 100), dtype=np.int32)
    label_map[:50, :50] = 1
    label_map[50:, 50:] = 2

    frame_masks = {
        "frame_0.jpg": label_map,
        "frame_1.jpg": label_map.copy(),
    }

    # 3D points: one that projects to top-left, one to bottom-right
    gaussians = np.array([
        [-0.4, -0.4, 1.0],  # Should project to approx (10, 10) -> label 1
        [0.4, 0.4, 1.0],    # Should project to approx (90, 90) -> label 2
    ])

    labels = projector.assign_labels(gaussians, frame_masks)
    assert labels[0] == 1, f"Expected label 1, got {labels[0]}"
    assert labels[1] == 2, f"Expected label 2, got {labels[1]}"
    print("[PASS] MaskProjector assigns correct labels")


def test_segmentation_results_to_label_maps() -> None:
    """Test conversion from SegmentationResult to integer label maps."""
    from src.pipeline.sam2_segmentor import SegmentationResult
    from src.pipeline.mask_projector import MaskProjector

    h, w = 64, 64
    masks = np.zeros((2, h, w), dtype=bool)
    masks[0, :32, :32] = True   # Object 1: top-left
    masks[1, 32:, 32:] = True   # Object 2: bottom-right

    result = SegmentationResult(
        frame_idx=0,
        masks=masks,
        object_ids=np.array([1, 2], dtype=np.int32),
        scores=np.array([0.9, 0.8], dtype=np.float32),
    )

    label_maps = MaskProjector.segmentation_results_to_label_maps(
        [result], ["frame_0.jpg"],
    )

    lm = label_maps["frame_0.jpg"]
    assert lm[0, 0] == 1
    assert lm[63, 63] == 2
    assert lm[0, 63] == 0  # Background
    print("[PASS] SegmentationResult -> label map conversion works")


def test_frame_quality_blur() -> None:
    """Test blur detection on synthetic sharp and blurry images."""
    from src.pipeline.frame_quality import FrameQualityAssessor
    import cv2

    assessor = FrameQualityAssessor(blur_threshold=100.0)

    # Sharp image: high-frequency checkerboard
    sharp = np.zeros((256, 256), dtype=np.uint8)
    sharp[::2, ::2] = 255
    sharp[1::2, 1::2] = 255
    sharp_score = assessor.compute_blur_score(sharp)

    # Blurry image: heavily blurred version
    blurry = cv2.GaussianBlur(sharp, (51, 51), 20)
    blurry_score = assessor.compute_blur_score(blurry)

    assert sharp_score > blurry_score, (
        f"Sharp ({sharp_score:.1f}) should score higher than blurry ({blurry_score:.1f})"
    )
    print(f"[PASS] Blur detection: sharp={sharp_score:.1f}, blurry={blurry_score:.1f}")


def test_frame_quality_exposure() -> None:
    """Test exposure analysis on dark and bright images."""
    from src.pipeline.frame_quality import FrameQualityAssessor

    assessor = FrameQualityAssessor()

    dark = np.full((100, 100), 10, dtype=np.uint8)
    bright = np.full((100, 100), 245, dtype=np.uint8)
    normal = np.full((100, 100), 128, dtype=np.uint8)

    dark_mean, _ = assessor.compute_exposure(dark)
    bright_mean, _ = assessor.compute_exposure(bright)
    normal_mean, _ = assessor.compute_exposure(normal)

    assert dark_mean < 0.1, f"Dark mean {dark_mean} should be < 0.1"
    assert bright_mean > 0.9, f"Bright mean {bright_mean} should be > 0.9"
    assert 0.4 < normal_mean < 0.6, f"Normal mean {normal_mean} out of range"
    print("[PASS] Exposure analysis correct")


def test_frame_quality_phash_duplicates() -> None:
    """Test perceptual hash duplicate detection."""
    from src.pipeline.frame_quality import FrameQualityAssessor

    assessor = FrameQualityAssessor(duplicate_hash_distance=10)

    # Two nearly identical images
    img_a = np.random.RandomState(42).randint(0, 256, (200, 200), dtype=np.uint8)
    img_b = img_a.copy()
    img_b[:5, :5] = 0  # Tiny modification

    # Completely different image
    img_c = np.random.RandomState(99).randint(0, 256, (200, 200), dtype=np.uint8)

    hash_a = assessor.compute_phash(img_a)
    hash_b = assessor.compute_phash(img_b)
    hash_c = assessor.compute_phash(img_c)

    dist_ab = assessor.hamming_distance(hash_a, hash_b)
    dist_ac = assessor.hamming_distance(hash_a, hash_c)

    assert dist_ab < dist_ac, (
        f"Similar images dist={dist_ab} should be less than different images dist={dist_ac}"
    )
    print(f"[PASS] pHash duplicates: similar={dist_ab}, different={dist_ac}")


def test_frame_quality_coverage() -> None:
    """Test coverage estimation on uniform vs textured images."""
    from src.pipeline.frame_quality import FrameQualityAssessor

    assessor = FrameQualityAssessor(coverage_block_size=32)

    # Uniform: zero coverage
    uniform = np.full((256, 256), 128, dtype=np.uint8)
    cov_uniform = assessor.compute_coverage(uniform)

    # Textured: high coverage
    textured = np.random.RandomState(42).randint(0, 256, (256, 256), dtype=np.uint8)
    cov_textured = assessor.compute_coverage(textured)

    assert cov_uniform < 0.1, f"Uniform coverage {cov_uniform} should be near 0"
    assert cov_textured > 0.5, f"Textured coverage {cov_textured} should be high"
    print(f"[PASS] Coverage: uniform={cov_uniform:.2f}, textured={cov_textured:.2f}")


def test_frame_quality_assess_directory() -> None:
    """End-to-end directory assessment with real files."""
    import cv2
    from src.pipeline.frame_quality import FrameQualityAssessor, Recommendation

    assessor = FrameQualityAssessor(blur_threshold=50.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)

        # Write 3 test images: sharp, blurry, duplicate of sharp
        sharp = np.random.RandomState(42).randint(0, 256, (200, 200, 3), dtype=np.uint8)
        blurry = cv2.GaussianBlur(sharp, (51, 51), 20)

        cv2.imwrite(str(td / "frame_000.jpg"), sharp)
        cv2.imwrite(str(td / "frame_001.jpg"), blurry)
        cv2.imwrite(str(td / "frame_002.jpg"), sharp)  # Duplicate of frame_000

        results = assessor.assess_directory(td)
        assert len(results) == 3

        # The third frame should be flagged as duplicate
        assert results[2].is_duplicate, "frame_002 should be flagged as duplicate"
        assert results[2].duplicate_of == results[0].path

        kept = assessor.filter_frames(results, include_marginal=False)
        print(f"[PASS] Directory assessment: {len(results)} frames, "
              f"{len(kept)} kept (strict)")


def test_sam2_segmentor_creation() -> None:
    """Verify SAM2Segmentor instantiation (no model download)."""
    from src.pipeline.sam2_segmentor import SAM2Segmentor

    seg = SAM2Segmentor(model_id="facebook/sam2-hiera-large", device="cpu")
    assert seg.model_id == "facebook/sam2-hiera-large"
    assert seg.device == "cpu"
    assert seg._image_predictor is None  # Not loaded yet

    seg2 = SAM2Segmentor.from_pretrained("facebook/sam2-hiera-large", device="cpu")
    assert seg2.model_id == seg.model_id

    seg.to("cpu")
    assert seg.device == "cpu"
    print("[PASS] SAM2Segmentor creation works (no model loaded)")


def main() -> None:
    """Run all validation tests."""
    tests = [
        test_imports,
        test_segmentation_result_dataclass,
        test_prompt_spec,
        test_iou_matching,
        test_iou_matching_new_object,
        test_view_projection,
        test_mask_projector_assign_labels,
        test_segmentation_results_to_label_maps,
        test_frame_quality_blur,
        test_frame_quality_exposure,
        test_frame_quality_phash_duplicates,
        test_frame_quality_coverage,
        test_frame_quality_assess_directory,
        test_sam2_segmentor_creation,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {test_fn.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
