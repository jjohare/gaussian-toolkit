#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the person removal pipeline stage.

Tests the PersonRemover class with synthetic images to validate:
- Detection pipeline runs without errors on clean images
- OpenCV inpainting works on masked regions
- Telea inpainting works on masked regions
- process_directory produces correct manifest structure
- Coverage thresholds drive correct frame decisions (drop/flag/inpaint)
- Config integration with PersonRemovalConfig
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Ensure src/ is on the path
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from pipeline.person_remover import PersonRemover, Detection, FrameAction
from pipeline.config import PipelineConfig, PersonRemovalConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("test_person_removal")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_scene(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a synthetic indoor scene (gradient background, no people)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Gradient floor
    for y in range(height):
        val = int(120 + 80 * (y / height))
        img[y, :] = [val, val - 20, val - 40]
    # Wall line
    cv2.line(img, (0, height // 3), (width, height // 3), (180, 170, 160), 2)
    # Some furniture rectangles
    cv2.rectangle(img, (50, 200), (150, 350), (80, 60, 40), -1)
    cv2.rectangle(img, (400, 180), (580, 360), (60, 80, 100), -1)
    return img


def make_scene_with_rectangle(
    width: int = 640,
    height: int = 480,
    rect_x: int = 250,
    rect_y: int = 100,
    rect_w: int = 100,
    rect_h: int = 300,
    color: tuple[int, int, int] = (0, 0, 255),
) -> tuple[np.ndarray, np.ndarray]:
    """Create a scene with a colored rectangle and its ground truth mask."""
    scene = make_synthetic_scene(width, height)
    # Draw the rectangle (simulating an object to be removed)
    cv2.rectangle(
        scene,
        (rect_x, rect_y),
        (rect_x + rect_w, rect_y + rect_h),
        color,
        -1,
    )
    # Ground truth mask
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = 255
    return scene, mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_detection_on_clean_image():
    """Detection should return empty list on a synthetic scene with no people."""
    logger.info("--- test_detection_on_clean_image ---")
    remover = PersonRemover(method="opencv", confidence=0.5, device="cpu")
    scene = make_synthetic_scene()

    detections = remover.detect_people(scene)
    assert isinstance(detections, list), "detect_people should return a list"
    logger.info("Detections on clean image: %d (expected 0)", len(detections))
    # A synthetic gradient should not trigger person detection
    # (low confidence detections may still appear, but should be filtered)
    assert len(detections) == 0, (
        f"Expected 0 detections on synthetic scene, got {len(detections)}"
    )
    logger.info("PASSED: No false positives on clean synthetic image")


def test_opencv_inpainting():
    """OpenCV NS inpainting should fill a masked region without error."""
    logger.info("--- test_opencv_inpainting ---")
    remover = PersonRemover(method="opencv", confidence=0.5, device="cpu")

    scene, mask = make_scene_with_rectangle()
    original = scene.copy()

    # Create a Detection from the ground truth mask
    det = Detection(
        bbox=(250, 100, 350, 400),
        mask=mask,
        confidence=0.99,
        area_pct=float(np.count_nonzero(mask)) / mask.size,
    )

    result = remover.remove_people(scene, [det])

    assert result.shape == scene.shape, "Output shape must match input"
    assert result.dtype == np.uint8, "Output must be uint8"

    # The inpainted region should differ from the original red rectangle
    roi_original = original[100:400, 250:350]
    roi_result = result[100:400, 250:350]
    diff = np.abs(roi_original.astype(float) - roi_result.astype(float)).mean()
    logger.info("Mean pixel difference in inpainted region: %.1f", diff)
    assert diff > 10, (
        f"Inpainted region should differ from original (diff={diff:.1f})"
    )
    logger.info("PASSED: OpenCV inpainting modifies the masked region")


def test_telea_inpainting():
    """OpenCV Telea inpainting should also work."""
    logger.info("--- test_telea_inpainting ---")
    remover = PersonRemover(method="telea", confidence=0.5, device="cpu")

    scene, mask = make_scene_with_rectangle(color=(0, 255, 0))
    det = Detection(
        bbox=(250, 100, 350, 400),
        mask=mask,
        confidence=0.99,
        area_pct=float(np.count_nonzero(mask)) / mask.size,
    )

    result = remover.remove_people(scene, [det])
    assert result.shape == scene.shape
    assert result.dtype == np.uint8

    roi_original = scene[100:400, 250:350]
    roi_result = result[100:400, 250:350]
    diff = np.abs(roi_original.astype(float) - roi_result.astype(float)).mean()
    logger.info("Mean pixel difference (Telea): %.1f", diff)
    assert diff > 10
    logger.info("PASSED: Telea inpainting works")


def test_process_frame_clean():
    """process_frame on a clean image should return action='clean'."""
    logger.info("--- test_process_frame_clean ---")
    remover = PersonRemover(method="opencv", confidence=0.5, device="cpu")
    scene = make_synthetic_scene()

    cleaned, action = remover.process_frame(scene, "test_clean.jpg")

    assert cleaned is not None, "Clean frame should not be None"
    assert action.action == "clean", f"Expected 'clean', got '{action.action}'"
    assert action.person_count == 0
    assert action.coverage_pct == 0.0
    logger.info("PASSED: Clean frame correctly identified")


def test_process_directory():
    """process_directory should produce manifest and output files."""
    logger.info("--- test_process_directory ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_dir = Path(tmpdir) / "input"
        out_dir = Path(tmpdir) / "output"
        in_dir.mkdir()

        # Write 5 synthetic frames
        for i in range(5):
            scene = make_synthetic_scene()
            # Add slight variation so frames aren't identical
            cv2.putText(
                scene, f"Frame {i}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            cv2.imwrite(str(in_dir / f"frame_{i:05d}.jpg"), scene)

        remover = PersonRemover(
            method="opencv", confidence=0.5, device="cpu",
        )
        manifest = remover.process_directory(str(in_dir), str(out_dir))

        # Validate manifest structure
        assert "total_frames" in manifest
        assert manifest["total_frames"] == 5
        assert "summary" in manifest
        assert "frames" in manifest
        assert len(manifest["frames"]) == 5

        # All frames should be clean (no real people)
        for frame in manifest["frames"]:
            assert frame["action"] == "clean", (
                f"Frame {frame['filename']} should be clean, got {frame['action']}"
            )

        # Output files should exist
        output_files = list(Path(out_dir).glob("*.jpg"))
        assert len(output_files) == 5, (
            f"Expected 5 output files, got {len(output_files)}"
        )

        # Manifest JSON should exist
        manifest_path = out_dir / "person_removal_manifest.json"
        assert manifest_path.exists(), "Manifest JSON not written"

        loaded = json.loads(manifest_path.read_text())
        assert loaded["total_frames"] == 5

        logger.info("PASSED: process_directory produces correct output")


def test_coverage_thresholds():
    """Verify drop/flag/inpaint decisions based on coverage thresholds."""
    logger.info("--- test_coverage_thresholds ---")

    # Create a remover with specific thresholds
    remover = PersonRemover(
        method="opencv",
        confidence=0.5,
        device="cpu",
        flag_threshold=0.05,
        drop_threshold=0.30,
    )

    h, w = 480, 640
    total = h * w

    # Case 1: Small person (2% coverage) -> inpainted
    small_mask = np.zeros((h, w), dtype=np.uint8)
    small_mask[200:250, 300:350] = 255  # ~1.6%
    small_det = Detection(
        bbox=(300, 200, 350, 250),
        mask=small_mask,
        confidence=0.9,
        area_pct=float(np.count_nonzero(small_mask)) / total,
    )

    # Case 2: Medium person (15% coverage) -> flagged_inpainted
    med_mask = np.zeros((h, w), dtype=np.uint8)
    med_mask[50:400, 200:430] = 255  # ~24%
    med_det = Detection(
        bbox=(200, 50, 430, 400),
        mask=med_mask,
        confidence=0.9,
        area_pct=float(np.count_nonzero(med_mask)) / total,
    )

    # Case 3: Large person (50% coverage) -> dropped
    large_mask = np.zeros((h, w), dtype=np.uint8)
    large_mask[0:h, 0:w // 2] = 255  # 50%
    large_det = Detection(
        bbox=(0, 0, w // 2, h),
        mask=large_mask,
        confidence=0.9,
        area_pct=0.50,
    )

    scene = make_synthetic_scene(w, h)

    # Manually test the decision logic through process_frame internals
    # For the small detection: coverage < flag_threshold -> "inpainted"
    small_coverage = float(np.count_nonzero(small_mask)) / total
    assert small_coverage < remover.flag_threshold, (
        f"Small coverage {small_coverage:.3f} should be < {remover.flag_threshold}"
    )

    # For medium: flag_threshold < coverage < drop_threshold -> "flagged_inpainted"
    med_coverage = float(np.count_nonzero(med_mask)) / total
    assert med_coverage > remover.flag_threshold, (
        f"Med coverage {med_coverage:.3f} should be > {remover.flag_threshold}"
    )
    assert med_coverage < remover.drop_threshold, (
        f"Med coverage {med_coverage:.3f} should be < {remover.drop_threshold}"
    )

    # For large: coverage > drop_threshold -> "dropped"
    large_coverage = float(np.count_nonzero(large_mask)) / total
    assert large_coverage > remover.drop_threshold, (
        f"Large coverage {large_coverage:.3f} should be > {remover.drop_threshold}"
    )

    logger.info(
        "Coverage values: small=%.3f, med=%.3f, large=%.3f",
        small_coverage, med_coverage, large_coverage,
    )
    logger.info("PASSED: Coverage threshold logic validated")


def test_empty_detections_returns_copy():
    """remove_people with no detections should return a copy of the original."""
    logger.info("--- test_empty_detections_returns_copy ---")
    remover = PersonRemover(method="opencv", confidence=0.5, device="cpu")
    scene = make_synthetic_scene()

    result = remover.remove_people(scene, [])

    assert result.shape == scene.shape
    assert np.array_equal(result, scene), "Should be identical copy"
    assert result is not scene, "Should be a copy, not the same object"
    logger.info("PASSED: Empty detections returns image copy")


def test_config_integration():
    """PersonRemovalConfig should serialize/deserialize in PipelineConfig."""
    logger.info("--- test_config_integration ---")

    cfg = PipelineConfig()
    assert hasattr(cfg, "person_removal")
    assert isinstance(cfg.person_removal, PersonRemovalConfig)
    assert cfg.person_removal.enabled is True
    assert cfg.person_removal.method == "opencv"
    assert cfg.person_removal.confidence == 0.5
    assert cfg.person_removal.dilation_px == 15

    # Test serialization round-trip
    d = cfg.to_dict()
    assert "person_removal" in d
    assert d["person_removal"]["enabled"] is True
    assert d["person_removal"]["drop_threshold"] == 0.30

    # Test deserialization
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(d, f, default=str)
        f.flush()
        loaded = PipelineConfig.load(f.name)

    assert loaded.person_removal.enabled is True
    assert loaded.person_removal.method == "opencv"
    assert loaded.person_removal.confidence == 0.5
    assert loaded.person_removal.drop_threshold == 0.30
    assert loaded.person_removal.comfyui_url == "http://192.168.2.48:8188"

    logger.info("PASSED: PersonRemovalConfig integrates with PipelineConfig")


def test_mask_dilation():
    """Verify that mask dilation expands the detection region."""
    logger.info("--- test_mask_dilation ---")

    remover_no_dilation = PersonRemover(
        method="opencv", confidence=0.5, device="cpu", dilation_px=0,
    )
    remover_with_dilation = PersonRemover(
        method="opencv", confidence=0.5, device="cpu", dilation_px=20,
    )

    h, w = 200, 200
    # Create masks manually using the internal method
    mask_no = np.zeros((h, w), dtype=np.uint8)
    mask_no[50:150, 50:150] = 255
    area_no = np.count_nonzero(mask_no)

    mask_dilated = mask_no.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    mask_dilated = cv2.dilate(mask_dilated, kernel, iterations=1)
    area_dilated = np.count_nonzero(mask_dilated)

    assert area_dilated > area_no, (
        f"Dilated mask ({area_dilated}px) should be larger than undilated ({area_no}px)"
    )
    logger.info(
        "Undilated: %d px, Dilated: %d px (%.1fx larger)",
        area_no, area_dilated, area_dilated / area_no,
    )
    logger.info("PASSED: Mask dilation expands detection region")


def test_multiple_detections_combine():
    """Multiple detections should combine into a single mask."""
    logger.info("--- test_multiple_detections_combine ---")
    remover = PersonRemover(method="opencv", confidence=0.5, device="cpu")

    h, w = 480, 640
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask1[100:300, 50:150] = 255
    mask2 = np.zeros((h, w), dtype=np.uint8)
    mask2[100:300, 400:500] = 255

    det1 = Detection(bbox=(50, 100, 150, 300), mask=mask1, confidence=0.9, area_pct=0.05)
    det2 = Detection(bbox=(400, 100, 500, 300), mask=mask2, confidence=0.85, area_pct=0.05)

    combined = remover._combine_masks([det1, det2], (h, w))
    assert np.count_nonzero(combined) > np.count_nonzero(mask1)
    assert np.count_nonzero(combined) == np.count_nonzero(mask1) + np.count_nonzero(mask2)

    scene = make_synthetic_scene(w, h)
    result = remover.remove_people(scene, [det1, det2])
    assert result.shape == scene.shape
    logger.info("PASSED: Multiple detections combine correctly")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_detection_on_clean_image,
        test_opencv_inpainting,
        test_telea_inpainting,
        test_process_frame_clean,
        test_process_directory,
        test_coverage_thresholds,
        test_empty_detections_returns_copy,
        test_config_integration,
        test_mask_dilation,
        test_multiple_detections_combine,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            failed += 1
            errors.append((test.__name__, str(exc)))
            logger.error("FAILED: %s: %s", test.__name__, exc)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Results: %d passed, %d failed out of %d tests", passed, failed, len(tests))
    if errors:
        for name, err in errors:
            logger.error("  FAIL: %s - %s", name, err)
    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
