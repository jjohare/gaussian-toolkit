# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Quality assessment gates for each pipeline stage."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class QualityVerdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class QualityResult:
    """Outcome of a quality gate evaluation."""
    verdict: QualityVerdict
    gate_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict in (QualityVerdict.PASS, QualityVerdict.WARN)


# ---------------------------------------------------------------------------
# Input quality (pre-reconstruction)
# ---------------------------------------------------------------------------

@dataclass
class FrameStats:
    """Per-frame statistics from ingested video."""
    frame_count: int = 0
    blur_scores: list[float] = field(default_factory=list)
    exposure_values: list[float] = field(default_factory=list)
    resolutions: list[tuple[int, int]] = field(default_factory=list)
    coverage_score: float = 0.0


def assess_input_quality(stats: FrameStats, config: PipelineConfig) -> QualityResult:
    """Check extracted frames meet minimum requirements."""
    metrics: dict[str, float] = {}
    issues: list[str] = []

    # Frame count
    metrics["frame_count"] = float(stats.frame_count)
    if stats.frame_count < config.ingest.min_frames:
        issues.append(
            f"Too few frames: {stats.frame_count} < {config.ingest.min_frames}"
        )
    elif stats.frame_count > config.ingest.max_frames:
        issues.append(
            f"Too many frames: {stats.frame_count} > {config.ingest.max_frames} (will be slow)"
        )

    # Blur
    if stats.blur_scores:
        avg_blur = sum(stats.blur_scores) / len(stats.blur_scores)
        metrics["avg_blur_score"] = avg_blur
        blurry = sum(1 for s in stats.blur_scores if s < config.ingest.blur_threshold)
        metrics["blurry_frame_ratio"] = blurry / len(stats.blur_scores)
        if metrics["blurry_frame_ratio"] > 0.5:
            issues.append(
                f"{blurry}/{len(stats.blur_scores)} frames are blurry "
                f"(threshold={config.ingest.blur_threshold})"
            )

    # Exposure
    if stats.exposure_values:
        lo, hi = config.ingest.exposure_range
        under = sum(1 for e in stats.exposure_values if e < lo)
        over = sum(1 for e in stats.exposure_values if e > hi)
        total = len(stats.exposure_values)
        metrics["underexposed_ratio"] = under / total
        metrics["overexposed_ratio"] = over / total
        if (under + over) / total > 0.3:
            issues.append(
                f"Poor exposure: {under} under, {over} over out of {total}"
            )

    # Coverage
    metrics["coverage_score"] = stats.coverage_score
    if stats.coverage_score < 0.3:
        issues.append(f"Low scene coverage: {stats.coverage_score:.2f}")

    if not issues:
        return QualityResult(
            verdict=QualityVerdict.PASS,
            gate_name="input_quality",
            metrics=metrics,
            recommendation="Input quality is good.",
        )

    severity = len(issues)
    if severity >= 3 or (stats.frame_count < config.ingest.min_frames):
        return QualityResult(
            verdict=QualityVerdict.FAIL,
            gate_name="input_quality",
            metrics=metrics,
            details={"issues": issues},
            recommendation="Re-capture video with better lighting, stability, and coverage.",
        )

    return QualityResult(
        verdict=QualityVerdict.WARN,
        gate_name="input_quality",
        metrics=metrics,
        details={"issues": issues},
        recommendation="Proceeding with warnings: " + "; ".join(issues),
    )


# ---------------------------------------------------------------------------
# Training quality (quality gate 1)
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Metrics collected after Gaussian splatting training."""
    psnr: float = 0.0
    ssim: float = 0.0
    final_loss: float = float("inf")
    loss_history: list[float] = field(default_factory=list)
    iterations_completed: int = 0
    max_iterations: int = 0
    num_gaussians: int = 0


def _check_convergence(losses: list[float], window: int, threshold: float) -> bool:
    """Return True if loss has converged (plateau) over the last `window` values."""
    if len(losses) < window:
        return False
    tail = losses[-window:]
    if tail[0] == 0:
        return True
    relative_change = abs(tail[-1] - tail[0]) / (abs(tail[0]) + 1e-12)
    return relative_change < threshold


def assess_training_quality(metrics: TrainingMetrics, config: PipelineConfig) -> QualityResult:
    """Quality gate 1: is the trained Gaussian model good enough?"""
    result_metrics: dict[str, float] = {
        "psnr": metrics.psnr,
        "ssim": metrics.ssim,
        "final_loss": metrics.final_loss,
        "iterations": float(metrics.iterations_completed),
        "num_gaussians": float(metrics.num_gaussians),
    }
    issues: list[str] = []

    if metrics.psnr < config.quality.gate1_min_psnr:
        issues.append(
            f"PSNR {metrics.psnr:.2f} < {config.quality.gate1_min_psnr}"
        )

    if metrics.ssim < config.quality.gate1_min_ssim:
        issues.append(
            f"SSIM {metrics.ssim:.4f} < {config.quality.gate1_min_ssim}"
        )

    converged = _check_convergence(
        metrics.loss_history,
        config.training.convergence_window,
        config.training.convergence_threshold,
    )
    result_metrics["converged"] = 1.0 if converged else 0.0

    if not converged and metrics.iterations_completed >= metrics.max_iterations:
        issues.append("Training reached max iterations without convergence")

    if math.isinf(metrics.final_loss) or math.isnan(metrics.final_loss):
        issues.append("Training loss is inf/nan — diverged")

    if not issues:
        return QualityResult(
            verdict=QualityVerdict.PASS,
            gate_name="training_quality",
            metrics=result_metrics,
            recommendation="Training quality is satisfactory.",
        )

    if metrics.psnr < config.quality.gate1_min_psnr * 0.7:
        return QualityResult(
            verdict=QualityVerdict.FAIL,
            gate_name="training_quality",
            metrics=result_metrics,
            details={"issues": issues},
            recommendation="Increase iterations or improve input data.",
        )

    return QualityResult(
        verdict=QualityVerdict.WARN,
        gate_name="training_quality",
        metrics=result_metrics,
        details={"issues": issues},
        recommendation="Quality marginal: " + "; ".join(issues),
    )


# ---------------------------------------------------------------------------
# Mesh quality (quality gate 2 component)
# ---------------------------------------------------------------------------

@dataclass
class MeshMetrics:
    """Metrics for an extracted mesh."""
    vertex_count: int = 0
    face_count: int = 0
    is_watertight: bool = False
    normal_consistency: float = 0.0
    bounding_box_volume: float = 0.0
    object_label: str = ""


def assess_mesh_quality(metrics: MeshMetrics, config: PipelineConfig) -> QualityResult:
    """Evaluate an extracted mesh."""
    result_metrics: dict[str, float] = {
        "vertex_count": float(metrics.vertex_count),
        "face_count": float(metrics.face_count),
        "is_watertight": 1.0 if metrics.is_watertight else 0.0,
        "normal_consistency": metrics.normal_consistency,
        "bbox_volume": metrics.bounding_box_volume,
    }
    issues: list[str] = []

    if metrics.vertex_count < config.quality.gate2_min_mesh_vertices:
        issues.append(
            f"Too few vertices: {metrics.vertex_count} < {config.quality.gate2_min_mesh_vertices}"
        )

    if metrics.vertex_count > config.mesh.max_vertices:
        issues.append(
            f"Too many vertices: {metrics.vertex_count} > {config.mesh.max_vertices}"
        )

    if config.mesh.watertight_check and not metrics.is_watertight:
        issues.append("Mesh is not watertight")

    if metrics.normal_consistency < config.quality.gate2_normal_consistency:
        issues.append(
            f"Normal consistency {metrics.normal_consistency:.3f} "
            f"< {config.quality.gate2_normal_consistency}"
        )

    if not issues:
        return QualityResult(
            verdict=QualityVerdict.PASS,
            gate_name=f"mesh_quality:{metrics.object_label}",
            metrics=result_metrics,
            recommendation="Mesh quality is acceptable.",
        )

    has_critical = metrics.vertex_count < 50 or metrics.normal_consistency < 0.3
    return QualityResult(
        verdict=QualityVerdict.FAIL if has_critical else QualityVerdict.WARN,
        gate_name=f"mesh_quality:{metrics.object_label}",
        metrics=result_metrics,
        details={"issues": issues},
        recommendation="Re-extract mesh with adjusted parameters." if has_critical
        else "Marginal mesh: " + "; ".join(issues),
    )


# ---------------------------------------------------------------------------
# Round-trip quality (mesh → gaussian → render comparison)
# ---------------------------------------------------------------------------

@dataclass
class RoundTripMetrics:
    """Compare original Gaussian render to mesh-round-tripped render."""
    original_psnr: float = 0.0
    roundtrip_psnr: float = 0.0
    psnr_delta: float = 0.0
    ssim_delta: float = 0.0


def assess_roundtrip_quality(metrics: RoundTripMetrics, config: PipelineConfig) -> QualityResult:
    """Quality gate 2: does the mesh round-trip back to acceptable quality?"""
    result_metrics: dict[str, float] = {
        "original_psnr": metrics.original_psnr,
        "roundtrip_psnr": metrics.roundtrip_psnr,
        "psnr_delta": metrics.psnr_delta,
        "ssim_delta": metrics.ssim_delta,
    }

    if metrics.roundtrip_psnr >= config.quality.gate2_roundtrip_psnr:
        return QualityResult(
            verdict=QualityVerdict.PASS,
            gate_name="roundtrip_quality",
            metrics=result_metrics,
            recommendation="Round-trip fidelity is acceptable.",
        )

    if metrics.roundtrip_psnr < config.quality.gate2_roundtrip_psnr * 0.7:
        return QualityResult(
            verdict=QualityVerdict.FAIL,
            gate_name="roundtrip_quality",
            metrics=result_metrics,
            recommendation=(
                f"Round-trip PSNR {metrics.roundtrip_psnr:.2f} is too low. "
                "Mesh extraction may have lost significant detail."
            ),
        )

    return QualityResult(
        verdict=QualityVerdict.WARN,
        gate_name="roundtrip_quality",
        metrics=result_metrics,
        recommendation=(
            f"Round-trip PSNR {metrics.roundtrip_psnr:.2f} is marginal "
            f"(target: {config.quality.gate2_roundtrip_psnr})."
        ),
    )


# ---------------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------------

@dataclass
class FinalMetrics:
    """Metrics for the final assembled scene."""
    render_psnr: float = 0.0
    object_count: int = 0
    total_vertices: int = 0
    usd_file_size_mb: float = 0.0
    has_materials: bool = False


def assess_final_quality(metrics: FinalMetrics, config: PipelineConfig) -> QualityResult:
    """Final validation before marking pipeline DONE."""
    result_metrics: dict[str, float] = {
        "render_psnr": metrics.render_psnr,
        "object_count": float(metrics.object_count),
        "total_vertices": float(metrics.total_vertices),
        "usd_file_size_mb": metrics.usd_file_size_mb,
        "has_materials": 1.0 if metrics.has_materials else 0.0,
    }
    issues: list[str] = []

    if metrics.render_psnr < config.quality.final_min_psnr:
        issues.append(
            f"Final PSNR {metrics.render_psnr:.2f} < {config.quality.final_min_psnr}"
        )

    if metrics.object_count == 0:
        issues.append("No objects in final scene")

    if not issues:
        return QualityResult(
            verdict=QualityVerdict.PASS,
            gate_name="final_validation",
            metrics=result_metrics,
            recommendation="Pipeline output is complete and meets quality targets.",
        )

    return QualityResult(
        verdict=QualityVerdict.FAIL,
        gate_name="final_validation",
        metrics=result_metrics,
        details={"issues": issues},
        recommendation="Final output does not meet quality targets: " + "; ".join(issues),
    )
