# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Frame quality assessment for video-to-Gaussian pipelines.

Evaluates blur, exposure, duplicate detection, and spatial coverage to
filter out low-quality frames before reconstruction.

Typical usage::

    assessor = FrameQualityAssessor()
    report = assessor.assess_directory("/path/to/frames")
    good_frames = [r.path for r in report if r.recommendation == "keep"]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Recommendation(str, Enum):
    """Frame filtering recommendation."""
    KEEP = "keep"
    MARGINAL = "marginal"
    DROP = "drop"


@dataclass(slots=True)
class FrameQuality:
    """Quality assessment for a single frame.

    Attributes:
        path: Absolute path to the image file.
        blur_score: Laplacian variance (higher = sharper). Typical threshold ~100.
        exposure_mean: Mean normalised brightness in ``[0, 1]``.
        exposure_std: Standard deviation of brightness histogram.
        is_underexposed: True if the frame is too dark.
        is_overexposed: True if the frame is too bright.
        phash: 64-bit perceptual hash for duplicate detection.
        is_duplicate: True if this frame is a near-duplicate of an earlier one.
        duplicate_of: Path of the earlier frame this duplicates, if any.
        coverage_score: Fraction of image area with non-trivial gradient energy.
        recommendation: Aggregate keep/marginal/drop recommendation.
    """
    path: Path
    blur_score: float = 0.0
    exposure_mean: float = 0.0
    exposure_std: float = 0.0
    is_underexposed: bool = False
    is_overexposed: bool = False
    phash: int = 0
    is_duplicate: bool = False
    duplicate_of: Optional[Path] = None
    coverage_score: float = 0.0
    recommendation: Recommendation = Recommendation.KEEP


class FrameQualityAssessor:
    """Assess and filter video frames by quality metrics.

    Parameters:
        blur_threshold: Laplacian variance below which a frame is considered blurry.
        exposure_low: Mean brightness below this is underexposed.
        exposure_high: Mean brightness above this is overexposed.
        duplicate_hash_distance: Maximum Hamming distance to flag as duplicate.
        coverage_threshold: Minimum fraction of active gradient area.
        coverage_block_size: Block size for coverage grid estimation.
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        exposure_low: float = 0.10,
        exposure_high: float = 0.90,
        duplicate_hash_distance: int = 8,
        coverage_threshold: float = 0.20,
        coverage_block_size: int = 64,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.exposure_low = exposure_low
        self.exposure_high = exposure_high
        self.duplicate_hash_distance = duplicate_hash_distance
        self.coverage_threshold = coverage_threshold
        self.coverage_block_size = coverage_block_size

    # ------------------------------------------------------------------
    #  Individual metrics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_blur_score(gray: np.ndarray) -> float:
        """Compute Laplacian variance as a sharpness metric.

        A higher score indicates a sharper image.

        Args:
            gray: ``(H, W)`` uint8 grayscale image.

        Returns:
            Laplacian variance (float).
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def compute_exposure(gray: np.ndarray) -> Tuple[float, float]:
        """Compute mean and standard deviation of normalised brightness.

        Args:
            gray: ``(H, W)`` uint8 grayscale image.

        Returns:
            ``(mean, std)`` in ``[0, 1]``.
        """
        normed = gray.astype(np.float32) / 255.0
        return float(normed.mean()), float(normed.std())

    @staticmethod
    def compute_phash(gray: np.ndarray, hash_size: int = 8) -> int:
        """Compute a 64-bit perceptual hash (pHash) of a grayscale image.

        Uses DCT-based hashing: resize to ``(hash_size*4, hash_size*4)``,
        compute DCT, keep top-left ``hash_size x hash_size`` coefficients,
        threshold at the median.

        Args:
            gray: ``(H, W)`` uint8 grayscale image.
            hash_size: Side length of the hash grid (hash is ``hash_size^2`` bits).

        Returns:
            Integer perceptual hash.
        """
        resized = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
        dct_full = cv2.dct(resized.astype(np.float32))
        dct_low = dct_full[:hash_size, :hash_size]
        median = float(np.median(dct_low))
        bits = (dct_low > median).flatten()
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h

    @staticmethod
    def hamming_distance(a: int, b: int) -> int:
        """Count differing bits between two integers."""
        return bin(a ^ b).count("1")

    def compute_coverage(self, gray: np.ndarray) -> float:
        """Estimate spatial coverage as the fraction of image blocks with
        significant gradient energy.

        This filters out frames that are mostly uniform (e.g., blank walls,
        sky-only shots) which provide little value for reconstruction.

        Args:
            gray: ``(H, W)`` uint8 grayscale image.

        Returns:
            Coverage fraction in ``[0, 1]``.
        """
        bs = self.coverage_block_size
        h, w = gray.shape
        if h < bs or w < bs:
            # Image too small for block analysis; assume full coverage
            return 1.0

        # Compute gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)

        # Block-wise mean gradient
        n_rows = h // bs
        n_cols = w // bs
        active_blocks = 0
        total_blocks = n_rows * n_cols

        # Threshold: block is "active" if mean gradient > 10
        gradient_threshold = 10.0
        for r in range(n_rows):
            for c in range(n_cols):
                block = mag[r*bs:(r+1)*bs, c*bs:(c+1)*bs]
                if block.mean() > gradient_threshold:
                    active_blocks += 1

        return active_blocks / max(total_blocks, 1)

    # ------------------------------------------------------------------
    #  Single-frame assessment
    # ------------------------------------------------------------------
    def assess_frame(self, image_path: str | Path) -> FrameQuality:
        """Run all quality checks on a single frame.

        Args:
            image_path: Path to the image file.

        Returns:
            Populated ``FrameQuality`` instance.
        """
        path = Path(image_path)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = self.compute_blur_score(gray)
        exp_mean, exp_std = self.compute_exposure(gray)
        phash = self.compute_phash(gray)
        coverage = self.compute_coverage(gray)

        is_under = exp_mean < self.exposure_low
        is_over = exp_mean > self.exposure_high
        is_blurry = blur < self.blur_threshold
        is_low_coverage = coverage < self.coverage_threshold

        # Aggregate recommendation
        issues = sum([is_blurry, is_under, is_over, is_low_coverage])
        if issues >= 2:
            rec = Recommendation.DROP
        elif issues == 1:
            rec = Recommendation.MARGINAL
        else:
            rec = Recommendation.KEEP

        return FrameQuality(
            path=path,
            blur_score=blur,
            exposure_mean=exp_mean,
            exposure_std=exp_std,
            is_underexposed=is_under,
            is_overexposed=is_over,
            phash=phash,
            coverage_score=coverage,
            recommendation=rec,
        )

    # ------------------------------------------------------------------
    #  Batch assessment with duplicate detection
    # ------------------------------------------------------------------
    def assess_directory(
        self,
        frame_dir: str | Path,
        *,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
    ) -> List[FrameQuality]:
        """Assess all frames in a directory, including cross-frame duplicate
        detection.

        Args:
            frame_dir: Directory containing video frames.
            extensions: Accepted image file extensions.

        Returns:
            List of ``FrameQuality`` in filename-sorted order.
        """
        frame_dir = Path(frame_dir)
        paths = sorted(
            p for p in frame_dir.iterdir()
            if p.suffix.lower() in extensions
        )
        if not paths:
            raise FileNotFoundError(f"No image files in {frame_dir}")

        logger.info("Assessing %d frames in %s", len(paths), frame_dir)
        results: List[FrameQuality] = []
        seen_hashes: List[Tuple[int, Path]] = []  # (phash, path)

        for p in paths:
            try:
                fq = self.assess_frame(p)
            except FileNotFoundError:
                logger.warning("Skipping unreadable: %s", p)
                continue

            # Check for duplicates against all prior frames
            for prev_hash, prev_path in seen_hashes:
                dist = self.hamming_distance(fq.phash, prev_hash)
                if dist <= self.duplicate_hash_distance:
                    fq.is_duplicate = True
                    fq.duplicate_of = prev_path
                    if fq.recommendation == Recommendation.KEEP:
                        fq.recommendation = Recommendation.MARGINAL
                    break

            seen_hashes.append((fq.phash, fq.path))
            results.append(fq)

        n_keep = sum(1 for r in results if r.recommendation == Recommendation.KEEP)
        n_marg = sum(1 for r in results if r.recommendation == Recommendation.MARGINAL)
        n_drop = sum(1 for r in results if r.recommendation == Recommendation.DROP)
        logger.info("Quality assessment: %d keep, %d marginal, %d drop",
                     n_keep, n_marg, n_drop)
        return results

    def filter_frames(
        self,
        results: List[FrameQuality],
        *,
        include_marginal: bool = True,
    ) -> List[Path]:
        """Return paths of frames that pass the quality filter.

        Args:
            results: Output of ``assess_directory``.
            include_marginal: Whether to include marginal frames.

        Returns:
            Sorted list of file paths.
        """
        allowed = {Recommendation.KEEP}
        if include_marginal:
            allowed.add(Recommendation.MARGINAL)
        return sorted(r.path for r in results if r.recommendation in allowed)
