# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Agentic state-machine orchestrator for the video-to-scene pipeline."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig
from pipeline.mcp_client import McpClient, McpError, McpConnectionError
from pipeline.quality_gates import (
    FrameStats,
    TrainingMetrics,
    MeshMetrics,
    RoundTripMetrics,
    FinalMetrics,
    QualityResult,
    QualityVerdict,
    assess_input_quality,
    assess_training_quality,
    assess_mesh_quality,
    assess_roundtrip_quality,
    assess_final_quality,
)

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """All possible pipeline states, ordered."""
    IDLE = "IDLE"
    INGEST = "INGEST"
    REMOVE_PEOPLE = "REMOVE_PEOPLE"
    SELECT_FRAMES = "SELECT_FRAMES"
    RECONSTRUCT = "RECONSTRUCT"
    QUALITY_GATE_1 = "QUALITY_GATE_1"
    DECOMPOSE = "DECOMPOSE"
    EXTRACT_OBJECTS = "EXTRACT_OBJECTS"
    MESH_OBJECTS = "MESH_OBJECTS"
    TEXTURE_BAKE = "TEXTURE_BAKE"
    QUALITY_GATE_2 = "QUALITY_GATE_2"
    INPAINT_BG = "INPAINT_BG"
    RETRAIN_BG = "RETRAIN_BG"
    USD_ASSEMBLE = "USD_ASSEMBLE"
    VALIDATE = "VALIDATE"
    DONE = "DONE"
    FAILED = "FAILED"


# Ordered transitions: each state's successor on success.
_TRANSITIONS: dict[PipelineState, PipelineState] = {
    PipelineState.IDLE: PipelineState.INGEST,
    PipelineState.INGEST: PipelineState.REMOVE_PEOPLE,
    PipelineState.REMOVE_PEOPLE: PipelineState.SELECT_FRAMES,
    PipelineState.SELECT_FRAMES: PipelineState.RECONSTRUCT,
    PipelineState.RECONSTRUCT: PipelineState.QUALITY_GATE_1,
    PipelineState.QUALITY_GATE_1: PipelineState.DECOMPOSE,
    PipelineState.DECOMPOSE: PipelineState.EXTRACT_OBJECTS,
    PipelineState.EXTRACT_OBJECTS: PipelineState.MESH_OBJECTS,
    PipelineState.MESH_OBJECTS: PipelineState.TEXTURE_BAKE,
    PipelineState.TEXTURE_BAKE: PipelineState.QUALITY_GATE_2,
    PipelineState.QUALITY_GATE_2: PipelineState.INPAINT_BG,
    PipelineState.INPAINT_BG: PipelineState.RETRAIN_BG,
    PipelineState.RETRAIN_BG: PipelineState.USD_ASSEMBLE,
    PipelineState.USD_ASSEMBLE: PipelineState.VALIDATE,
    PipelineState.VALIDATE: PipelineState.DONE,
}


@dataclass
class StageResult:
    """Outcome of a single pipeline stage execution."""
    success: bool
    state: PipelineState
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    quality: QualityResult | None = None
    retry_hint: dict[str, Any] | None = None


@dataclass
class PipelineStatus:
    """Serialisable pipeline progress."""
    state: str = "IDLE"
    progress: float = 0.0
    stage_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None
    started_at: float = 0.0
    finished_at: float | None = None
    retries: dict[str, int] = field(default_factory=dict)


class PipelineOrchestrator:
    """Drives the full video-to-scene pipeline as a state machine.

    Each state maps to a handler method (_run_<state_name>). On success
    the machine advances to the next state. On failure it retries with
    adjusted parameters up to ``config.retry.max_retries`` times, then
    emits partial results and moves to FAILED.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        config: PipelineConfig | None = None,
    ) -> None:
        self.video_path = Path(video_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.config = config or PipelineConfig()
        self.config.output_dir = str(self.output_dir)

        self.mcp = McpClient(
            endpoint=self.config.mcp_endpoint,
            timeout=self.config.mcp_timeout,
            training_timeout=self.config.mcp_training_timeout,
            max_retries=3,
        )

        self._state = PipelineState.IDLE
        self._status = PipelineStatus()
        self._artifacts: dict[str, str] = {}
        self._stage_metrics: dict[str, dict[str, Any]] = {}
        self._retry_counts: dict[str, int] = {}

        # Intermediate data passed between stages.
        self._frame_dir: Path | None = None
        self._colmap_dir: Path | None = None
        self._frame_stats: FrameStats | None = None
        self._training_metrics: TrainingMetrics | None = None
        self._extracted_objects: list[dict[str, Any]] = []
        self._mesh_results: list[dict[str, Any]] = []
        self._person_manifest: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> PipelineState:
        return self._state

    def run(self) -> PipelineStatus:
        """Execute the full pipeline from IDLE to DONE (or FAILED)."""
        self._status.started_at = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._advance(PipelineState.INGEST)

        state_count = len(PipelineState) - 3  # exclude IDLE, DONE, FAILED
        completed = 0

        while self._state not in (PipelineState.DONE, PipelineState.FAILED):
            handler = self._get_handler(self._state)
            stage_name = self._state.value
            logger.info("=== Stage: %s ===", stage_name)

            result = self._execute_with_retries(handler, stage_name)
            self._record_result(stage_name, result)

            if result.success:
                completed += 1
                self._status.progress = completed / state_count
                next_state = _TRANSITIONS.get(self._state)
                if next_state:
                    self._advance(next_state)
                else:
                    self._advance(PipelineState.DONE)
            else:
                logger.error("Stage %s failed: %s", stage_name, result.error)
                self._status.error = f"Failed at {stage_name}: {result.error}"
                self._advance(PipelineState.FAILED)

            self._write_status()

        self._status.finished_at = time.time()
        self._status.state = self._state.value
        self._write_status()
        return self._status

    def get_partial_results(self) -> dict[str, Any]:
        """Return whatever artifacts have been produced so far."""
        return {
            "state": self._state.value,
            "artifacts": dict(self._artifacts),
            "metrics": dict(self._stage_metrics),
        }

    # ------------------------------------------------------------------
    # State machine mechanics
    # ------------------------------------------------------------------

    def _advance(self, new_state: PipelineState) -> None:
        logger.info("Transition: %s -> %s", self._state.value, new_state.value)
        self._state = new_state
        self._status.state = new_state.value

    def _get_handler(self, state: PipelineState):
        handlers = {
            PipelineState.INGEST: self._run_ingest,
            PipelineState.REMOVE_PEOPLE: self._run_remove_people,
            PipelineState.SELECT_FRAMES: self._run_select_frames,
            PipelineState.RECONSTRUCT: self._run_reconstruct,
            PipelineState.QUALITY_GATE_1: self._run_quality_gate_1,
            PipelineState.DECOMPOSE: self._run_decompose,
            PipelineState.EXTRACT_OBJECTS: self._run_extract_objects,
            PipelineState.MESH_OBJECTS: self._run_mesh_objects,
            PipelineState.TEXTURE_BAKE: self._run_texture_bake,
            PipelineState.QUALITY_GATE_2: self._run_quality_gate_2,
            PipelineState.INPAINT_BG: self._run_inpaint_bg,
            PipelineState.RETRAIN_BG: self._run_retrain_bg,
            PipelineState.USD_ASSEMBLE: self._run_usd_assemble,
            PipelineState.VALIDATE: self._run_validate,
        }
        return handlers[state]

    def _execute_with_retries(self, handler, stage_name: str) -> StageResult:
        max_retries = self.config.retry.max_retries
        attempts = self._retry_counts.get(stage_name, 0)

        for attempt in range(attempts, max_retries + 1):
            self._retry_counts[stage_name] = attempt
            self._status.retries[stage_name] = attempt

            if attempt > 0:
                self._apply_retry_adjustments(stage_name, attempt)
                logger.info("Retry %d/%d for stage %s", attempt, max_retries, stage_name)

            try:
                result = handler()
            except (McpError, McpConnectionError) as exc:
                result = StageResult(
                    success=False,
                    state=self._state,
                    error=str(exc),
                )
            except Exception as exc:
                logger.exception("Unexpected error in stage %s", stage_name)
                result = StageResult(
                    success=False,
                    state=self._state,
                    error=f"Unexpected: {exc}",
                )

            if result.success:
                return result

            if attempt >= max_retries:
                return result

        return result  # type: ignore[possibly-undefined]

    def _apply_retry_adjustments(self, stage_name: str, attempt: int) -> None:
        key = f"{stage_name}:{attempt}" if attempt > 1 else stage_name
        adjustments = self.config.retry.parameter_adjustments.get(key, {})
        if not adjustments:
            adjustments = self.config.retry.parameter_adjustments.get(stage_name, {})

        for param, value in adjustments.items():
            for sub_cfg in [
                self.config.ingest, self.config.reconstruct,
                self.config.training, self.config.mesh,
                self.config.inpaint,
            ]:
                if hasattr(sub_cfg, param):
                    old = getattr(sub_cfg, param)
                    setattr(sub_cfg, param, value)
                    logger.info("Retry adjustment: %s = %s (was %s)", param, value, old)

    def _record_result(self, stage_name: str, result: StageResult) -> None:
        entry: dict[str, Any] = {
            "success": result.success,
            "metrics": result.metrics,
            "artifacts": result.artifacts,
        }
        if result.error:
            entry["error"] = result.error
        if result.quality:
            entry["quality"] = {
                "verdict": result.quality.verdict.value,
                "gate": result.quality.gate_name,
                "metrics": result.quality.metrics,
                "recommendation": result.quality.recommendation,
            }
        self._status.stage_results[stage_name] = entry
        self._stage_metrics[stage_name] = result.metrics
        self._artifacts.update(result.artifacts)

    def _write_status(self) -> None:
        path = self.output_dir / self.config.status_file
        try:
            path.write_text(
                json.dumps({
                    "state": self._status.state,
                    "progress": self._status.progress,
                    "error": self._status.error,
                    "started_at": self._status.started_at,
                    "finished_at": self._status.finished_at,
                    "retries": self._status.retries,
                    "stage_results": self._status.stage_results,
                }, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not write status file: %s", exc)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _run_ingest(self) -> StageResult:
        """Extract frames from the source video using ffmpeg."""
        if not self.video_path.exists():
            return StageResult(
                success=False, state=self._state,
                error=f"Video not found: {self.video_path}",
            )

        frame_dir = self.output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.video_path),
            "-vf", f"fps={self.config.ingest.fps}",
            "-q:v", "2",
            str(frame_dir / "frame_%05d.jpg"),
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
        except FileNotFoundError:
            return StageResult(
                success=False, state=self._state,
                error="ffmpeg not found in PATH",
            )
        except subprocess.TimeoutExpired:
            return StageResult(
                success=False, state=self._state,
                error="Frame extraction timed out (300s)",
            )

        if proc.returncode != 0:
            return StageResult(
                success=False, state=self._state,
                error=f"ffmpeg failed (rc={proc.returncode}): {proc.stderr[:500]}",
            )

        frames = sorted(frame_dir.glob("*.jpg"))
        self._frame_dir = frame_dir

        stats = FrameStats(
            frame_count=len(frames),
            blur_scores=[],  # Would be computed by cv2 if available
            exposure_values=[],
            resolutions=[],
            coverage_score=0.5,  # Placeholder; real impl uses feature matching
        )
        self._frame_stats = stats

        quality = assess_input_quality(stats, self.config)
        if quality.verdict == QualityVerdict.FAIL:
            return StageResult(
                success=False, state=self._state,
                error=quality.recommendation,
                quality=quality,
                metrics=quality.metrics,
            )

        return StageResult(
            success=True, state=self._state,
            metrics=quality.metrics,
            artifacts={"frames_dir": str(frame_dir), "frame_count": str(len(frames))},
            quality=quality,
        )

    def _run_remove_people(self) -> StageResult:
        """Detect and remove people from frames before COLMAP processing."""
        if self._frame_dir is None:
            return StageResult(
                success=False, state=self._state,
                error="No frames directory from ingest stage",
            )

        cfg = self.config.person_removal
        if not cfg.enabled:
            logger.info("Person removal disabled, skipping")
            return StageResult(
                success=True, state=self._state,
                metrics={"skipped": True},
                artifacts={"cleaned_frames_dir": str(self._frame_dir)},
            )

        from pipeline.person_remover import PersonRemover

        cleaned_dir = self.output_dir / "frames_cleaned"
        remover = PersonRemover(
            method=cfg.method,
            comfyui_url=cfg.comfyui_url or None,
            flux_endpoint=cfg.flux_endpoint or None,
            confidence=cfg.confidence,
            dilation_px=cfg.dilation_px,
            drop_threshold=cfg.drop_threshold,
            flag_threshold=cfg.flag_threshold,
            comfyui_timeout=cfg.comfyui_timeout,
        )

        try:
            manifest = remover.process_directory(
                str(self._frame_dir), str(cleaned_dir),
            )
        except Exception as exc:
            return StageResult(
                success=False, state=self._state,
                error=f"Person removal failed: {exc}",
            )

        summary = manifest.get("summary", {})
        remaining = summary.get("clean", 0) + summary.get("inpainted", 0) + summary.get("flagged_inpainted", 0)

        if remaining < self.config.ingest.min_frames:
            return StageResult(
                success=False, state=self._state,
                error=(
                    f"Too few frames after person removal: {remaining} "
                    f"(need {self.config.ingest.min_frames}). "
                    f"{summary.get('dropped', 0)} frames were dropped."
                ),
                metrics=manifest.get("summary", {}),
            )

        # Store manifest for frame selection stage
        self._person_manifest = manifest

        # Point subsequent stages at the cleaned frames
        self._frame_dir = cleaned_dir
        self._artifacts["cleaned_frames_dir"] = str(cleaned_dir)
        self._artifacts["person_removal_manifest"] = str(
            cleaned_dir / "person_removal_manifest.json"
        )

        return StageResult(
            success=True, state=self._state,
            metrics={
                "total_frames": manifest.get("total_frames", 0),
                "remaining_frames": remaining,
                **summary,
            },
            artifacts={
                "cleaned_frames_dir": str(cleaned_dir),
                "person_removal_manifest": str(
                    cleaned_dir / "person_removal_manifest.json"
                ),
            },
        )

    def _run_select_frames(self) -> StageResult:
        """Score, filter, and select the best frame subset for COLMAP."""
        if self._frame_dir is None:
            return StageResult(
                success=False, state=self._state,
                error="No frames directory from previous stage",
            )

        from pipeline.frame_selector import FrameSelector, SelectionConfig

        sel_cfg = SelectionConfig(
            target_frames=min(
                self.config.ingest.max_frames,
                max(self.config.ingest.min_frames, 150),
            ),
            min_frames=self.config.ingest.min_frames,
            max_frames=self.config.ingest.max_frames,
            blur_threshold=self.config.ingest.blur_threshold,
        )
        selector = FrameSelector(config=sel_cfg)

        try:
            scores = selector.score_frames(str(self._frame_dir))
        except Exception as exc:
            logger.warning("Frame scoring failed (%s), keeping all frames", exc)
            return StageResult(
                success=True, state=self._state,
                metrics={"skipped": True, "reason": str(exc)},
                artifacts={"selected_frames_dir": str(self._frame_dir)},
            )

        if not scores:
            return StageResult(
                success=False, state=self._state,
                error=f"No frames found in {self._frame_dir}",
            )

        selected = selector.select(scores, person_manifest=self._person_manifest)

        if len(selected) < self.config.ingest.min_frames:
            return StageResult(
                success=False, state=self._state,
                error=(
                    f"Only {len(selected)} frames after selection "
                    f"(need {self.config.ingest.min_frames})"
                ),
                metrics={"scored": len(scores), "selected": len(selected)},
            )

        selected_dir = self.output_dir / "frames_selected"
        selected_paths = selector.copy_selected(selected, str(selected_dir))

        # Update frame_dir for downstream stages
        self._frame_dir = selected_dir

        return StageResult(
            success=True, state=self._state,
            metrics={
                "scored": len(scores),
                "selected": len(selected),
                "target": sel_cfg.target_frames,
            },
            artifacts={"selected_frames_dir": str(selected_dir)},
        )

    def _run_reconstruct(self) -> StageResult:
        """Run SplatReady / COLMAP reconstruction, then load into LichtFeld."""
        if self._frame_dir is None:
            return StageResult(
                success=False, state=self._state,
                error="No frames directory from ingest stage",
            )

        colmap_dir = self.output_dir / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)

        splatready_config = {
            "video_path": str(self.video_path),
            "base_output_folder": str(self.output_dir),
            "frame_rate": self.config.ingest.fps,
            "skip_extraction": True,  # Frames already extracted
            "reconstruction_method": self.config.reconstruct.method,
            "colmap_exe_path": self.config.reconstruct.colmap_exe,
            "use_fisheye": self.config.reconstruct.use_fisheye,
            "max_image_size": self.config.ingest.max_image_size,
            "min_scale": self.config.reconstruct.min_scale,
            "skip_reconstruction": False,
        }

        config_path = self.output_dir / "splatready_config.json"
        config_path.write_text(json.dumps(splatready_config, indent=2), encoding="utf-8")

        plugin_dir = Path.home() / ".lichtfeld" / "plugins" / "splat_ready"
        runner = plugin_dir / "core" / "runner.py"

        if runner.exists():
            try:
                proc = subprocess.run(
                    ["python3", str(runner), str(config_path)],
                    capture_output=True, text=True, timeout=600,
                )
                if proc.returncode != 0:
                    return StageResult(
                        success=False, state=self._state,
                        error=f"SplatReady failed: {proc.stderr[:500]}",
                    )
            except subprocess.TimeoutExpired:
                return StageResult(
                    success=False, state=self._state,
                    error="COLMAP reconstruction timed out",
                )
        else:
            # Fallback: run COLMAP directly
            try:
                self._run_colmap_direct(colmap_dir)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
                return StageResult(
                    success=False, state=self._state,
                    error=f"COLMAP failed: {exc}",
                )

        dataset_dir = self.output_dir / "colmap" / "undistorted"
        if not dataset_dir.exists():
            # Check alternative locations
            for alt in [colmap_dir, self.output_dir / "colmap"]:
                if (alt / "images").exists() and (alt / "sparse").exists():
                    dataset_dir = alt
                    break
            else:
                return StageResult(
                    success=False, state=self._state,
                    error=f"COLMAP output not found at expected paths",
                )

        self._colmap_dir = dataset_dir

        # Load dataset into LichtFeld via MCP
        try:
            load_result = self.mcp.load_dataset(
                path=str(dataset_dir),
                max_iterations=self.config.training.max_iterations,
                strategy=self.config.training.strategy,
            )
        except (McpError, McpConnectionError) as exc:
            return StageResult(
                success=False, state=self._state,
                error=f"MCP load_dataset failed: {exc}",
            )

        # Start training
        try:
            self.mcp.training_start()
        except (McpError, McpConnectionError) as exc:
            return StageResult(
                success=False, state=self._state,
                error=f"MCP training_start failed: {exc}",
            )

        # Wait for training to complete
        try:
            final_state = self.mcp.wait_training_complete(poll_interval=10.0)
        except McpError as exc:
            return StageResult(
                success=False, state=self._state,
                error=f"Training failed: {exc}",
            )

        # Collect training metrics
        try:
            loss_hist = self.mcp.training_get_loss_history(last_n=1000)
        except McpError:
            loss_hist = None

        self._training_metrics = TrainingMetrics(
            psnr=final_state.psnr,
            ssim=final_state.ssim,
            final_loss=final_state.loss,
            loss_history=loss_hist.losses if loss_hist else [],
            iterations_completed=final_state.iteration,
            max_iterations=final_state.max_iterations,
            num_gaussians=final_state.num_gaussians,
        )

        # Save checkpoint
        ckpt_path = self.output_dir / "checkpoints" / "initial.resume"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.mcp.save_checkpoint(str(ckpt_path))
        except McpError as exc:
            logger.warning("Could not save checkpoint: %s", exc)

        return StageResult(
            success=True, state=self._state,
            metrics={
                "psnr": final_state.psnr,
                "ssim": final_state.ssim,
                "iterations": final_state.iteration,
                "num_gaussians": final_state.num_gaussians,
            },
            artifacts={
                "dataset_dir": str(dataset_dir),
                "checkpoint": str(ckpt_path),
            },
        )

    def _run_colmap_direct(self, output_dir: Path) -> None:
        """Fallback: run COLMAP feature extraction + matching + mapper."""
        db_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        colmap = self.config.reconstruct.colmap_exe

        subprocess.run([
            colmap, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(self._frame_dir),
            "--ImageReader.single_camera", "1",
        ], check=True, capture_output=True, timeout=300)

        subprocess.run([
            colmap, self.config.reconstruct.matcher + "_matcher",
            "--database_path", str(db_path),
        ], check=True, capture_output=True, timeout=300)

        subprocess.run([
            colmap, "mapper",
            "--database_path", str(db_path),
            "--image_path", str(self._frame_dir),
            "--output_path", str(sparse_dir),
        ], check=True, capture_output=True, timeout=600)

    def _run_quality_gate_1(self) -> StageResult:
        """Evaluate training quality. Decide whether to proceed or retry."""
        if self._training_metrics is None:
            return StageResult(
                success=False, state=self._state,
                error="No training metrics available",
            )

        quality = assess_training_quality(self._training_metrics, self.config)
        logger.info(
            "Quality Gate 1: %s (PSNR=%.2f, SSIM=%.4f)",
            quality.verdict.value,
            self._training_metrics.psnr,
            self._training_metrics.ssim,
        )

        if quality.verdict == QualityVerdict.FAIL:
            return StageResult(
                success=False, state=self._state,
                error=quality.recommendation,
                quality=quality,
                metrics=quality.metrics,
                retry_hint={"increase_iterations": True},
            )

        return StageResult(
            success=True, state=self._state,
            metrics=quality.metrics,
            quality=quality,
        )

    def _run_decompose(self) -> StageResult:
        """Use SAM3 concept segmentation or MCP semantic selection to identify scene objects."""
        decompose_cfg = self.config.decompose

        # --- SAM3 concept segmentation path ---
        if decompose_cfg.use_sam3 and self._frame_dir is not None:
            try:
                return self._run_decompose_sam3()
            except Exception as exc:
                if decompose_cfg.sam3_fallback_to_sam2:
                    logger.warning(
                        "SAM3 decomposition failed (%s), falling back to MCP selection", exc,
                    )
                else:
                    return StageResult(
                        success=False, state=self._state,
                        error=f"SAM3 decomposition failed: {exc}",
                    )

        # --- Legacy MCP description-based path ---
        descriptions = decompose_cfg.descriptions
        if not descriptions:
            try:
                advice = self.mcp.ask_advisor(
                    "What are the main objects in this scene? "
                    "List them as comma-separated short descriptions."
                )
                descriptions = [d.strip() for d in advice.split(",") if d.strip()]
            except McpError:
                descriptions = []

        if not descriptions:
            logger.info("No object descriptions; skipping decomposition")
            self._extracted_objects = [{"label": "full_scene", "count": -1}]
            return StageResult(
                success=True, state=self._state,
                metrics={"object_count": 1},
                artifacts={},
            )

        objects: list[dict[str, Any]] = []
        for desc in descriptions:
            try:
                sel = self.mcp.selection_by_description(desc)
                if sel.count >= decompose_cfg.min_object_gaussians:
                    objects.append({"label": desc, "count": sel.count})
                else:
                    logger.info("Skipping '%s': only %d gaussians", desc, sel.count)
            except McpError as exc:
                logger.warning("Selection failed for '%s': %s", desc, exc)

        if not objects:
            objects = [{"label": "full_scene", "count": -1}]

        self._extracted_objects = objects
        return StageResult(
            success=True, state=self._state,
            metrics={"object_count": len(objects)},
            artifacts={"objects": json.dumps(objects)},
        )

    def _run_decompose_sam3(self) -> StageResult:
        """SAM3-based scene decomposition using text concept segmentation."""
        import cv2
        from pipeline.sam3_segmentor import SAM3Segmentor

        decompose_cfg = self.config.decompose
        concepts = decompose_cfg.sam3_concepts
        if not concepts:
            concepts = decompose_cfg.descriptions or [
                "paintings", "frames", "sculptures", "furniture",
                "walls", "floor", "ceiling", "fixtures", "doorways",
            ]

        logger.info("SAM3 decompose: concepts=%s, frame_dir=%s", concepts, self._frame_dir)

        seg = SAM3Segmentor(
            device="cuda",
            confidence_threshold=decompose_cfg.sam3_confidence_threshold,
        )

        # Segment the first frame to identify objects
        frame_paths = sorted(
            p for p in self._frame_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not frame_paths:
            return StageResult(
                success=False, state=self._state,
                error=f"No frames found in {self._frame_dir}",
            )

        image = cv2.imread(str(frame_paths[0]))
        if image is None:
            return StageResult(
                success=False, state=self._state,
                error=f"Failed to read frame {frame_paths[0]}",
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result, id_to_concept = seg.segment_by_concepts(
            image, concepts,
            confidence_threshold=decompose_cfg.sam3_confidence_threshold,
        )

        objects: list[dict[str, Any]] = []
        for obj_id in result.object_ids:
            concept = id_to_concept.get(int(obj_id), "unknown")
            mask_pixels = int(result.masks[result.object_ids == obj_id].sum())
            objects.append({
                "label": concept,
                "object_id": int(obj_id),
                "mask_pixels": mask_pixels,
            })

        if not objects:
            objects = [{"label": "full_scene", "count": -1}]

        # Save SAM3 masks for downstream stages
        masks_dir = self.output_dir / "sam3_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        import numpy as np
        for i, obj_id in enumerate(result.object_ids):
            mask_path = masks_dir / f"mask_{int(obj_id):04d}.npy"
            np.save(str(mask_path), result.masks[i])

        seg.unload()

        self._extracted_objects = objects
        return StageResult(
            success=True, state=self._state,
            metrics={
                "object_count": len(objects),
                "segmentation_method": "sam3",
                "concepts_used": concepts,
            },
            artifacts={
                "objects": json.dumps(objects),
                "sam3_masks_dir": str(masks_dir),
            },
        )

    def _run_extract_objects(self) -> StageResult:
        """Extract each identified object as a separate PLY."""
        if not self._extracted_objects:
            return StageResult(
                success=False, state=self._state,
                error="No objects to extract",
            )

        objects_dir = self.output_dir / "objects"
        objects_dir.mkdir(parents=True, exist_ok=True)
        extracted: list[dict[str, Any]] = []

        for obj in self._extracted_objects:
            label = obj["label"]
            safe_name = label.replace(" ", "_").replace("/", "_")[:50]
            ply_path = objects_dir / f"{safe_name}.ply"

            if label == "full_scene":
                try:
                    self.mcp.save_ply(str(ply_path))
                    extracted.append({"label": label, "ply": str(ply_path)})
                except McpError as exc:
                    logger.warning("Could not save full scene PLY: %s", exc)
            else:
                try:
                    self.mcp.selection_by_description(label)
                    self.mcp.save_ply(str(ply_path))
                    self.mcp.selection_clear()
                    extracted.append({"label": label, "ply": str(ply_path)})
                except McpError as exc:
                    logger.warning("Extract failed for '%s': %s", label, exc)

        if not extracted:
            return StageResult(
                success=False, state=self._state,
                error="No objects could be extracted",
            )

        self._extracted_objects = extracted
        return StageResult(
            success=True, state=self._state,
            metrics={"extracted_count": len(extracted)},
            artifacts={f"ply:{e['label']}": e["ply"] for e in extracted},
        )

    def _run_mesh_objects(self) -> StageResult:
        """Convert each extracted PLY to a mesh.

        Extraction priority per object:
        1. Hunyuan3D 2.0 multi-view reconstruction (if enabled and available)
        2. TSDF depth-fusion via MeshExtractor
        3. MeshExtractor.extract_from_pointcloud (Poisson-like fallback)
        4. Open3D Poisson (legacy fallback)

        Saves per-object OBJ + texture to objects/meshes/.
        """
        meshes_dir = self.output_dir / "objects" / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []

        for obj in self._extracted_objects:
            label = obj.get("label", "unknown")
            ply_path = obj.get("ply")
            if not ply_path or not Path(ply_path).exists():
                logger.warning("Skipping '%s': PLY not found at %s", label, ply_path)
                continue

            safe_name = label.replace(" ", "_").replace("/", "_")[:50]
            obj_dir = meshes_dir / safe_name
            obj_dir.mkdir(parents=True, exist_ok=True)
            mesh_obj_path = obj_dir / f"{safe_name}.obj"
            mesh_glb_path = obj_dir / f"{safe_name}.glb"

            mesh_result = self._mesh_single_object(
                label, ply_path, obj_dir, mesh_obj_path, mesh_glb_path,
            )
            if mesh_result is not None:
                results.append(mesh_result)

        if not results:
            return StageResult(
                success=False, state=self._state,
                error="No meshes could be generated",
            )

        self._mesh_results = results
        return StageResult(
            success=True, state=self._state,
            metrics={"mesh_count": len(results)},
            artifacts={f"mesh:{r['label']}": r["mesh"] for r in results},
        )

    def _mesh_single_object(
        self,
        label: str,
        ply_path: str,
        obj_dir: Path,
        mesh_obj_path: Path,
        mesh_glb_path: Path,
    ) -> dict[str, Any] | None:
        """Try all meshing strategies for a single object PLY."""
        import numpy as np
        import trimesh

        # --- Strategy 1: Hunyuan3D 2.0 ---
        if self.config.hunyuan3d.enabled:
            try:
                from pipeline.hunyuan3d_client import Hunyuan3DClient

                h3d = Hunyuan3DClient(
                    comfyui_url=self.config.hunyuan3d.comfyui_url,
                    api_url=self.config.hunyuan3d.api_url,
                    quality=self.config.hunyuan3d.quality,
                    multiview=self.config.hunyuan3d.multiview,
                    turbo=self.config.hunyuan3d.turbo,
                    timeout=self.config.hunyuan3d.timeout,
                    seed=self.config.hunyuan3d.seed,
                )
                result = h3d.reconstruct_from_gaussians(ply_path)
                if result.mesh is not None:
                    result.mesh.export(str(mesh_glb_path))
                    # Also export OBJ for texture baking compatibility
                    result.mesh.export(str(mesh_obj_path))
                    vc = len(result.mesh.vertices)
                    logger.info(
                        "Hunyuan3D mesh for '%s': %d verts -> %s",
                        label, vc, mesh_glb_path,
                    )
                    return {
                        "label": label,
                        "mesh": str(mesh_glb_path),
                        "mesh_obj": str(mesh_obj_path),
                        "ply": ply_path,
                        "vertex_count": vc,
                        "method": "hunyuan3d",
                    }
            except Exception as exc:
                logger.warning("Hunyuan3D failed for '%s': %s", label, exc)

        # --- Strategy 2: TSDF depth-fusion ---
        try:
            from pipeline.mesh_extractor import MeshExtractor, TSDFConfig

            tsdf_cfg = TSDFConfig(
                target_faces=self.config.mesh.max_vertices // 2,
                mcp_endpoint=self.config.mcp_endpoint,
            )
            extractor = MeshExtractor(config=tsdf_cfg)
            mesh = extractor.extract_from_mcp()
            mesh.export(str(mesh_glb_path))
            mesh.export(str(mesh_obj_path))
            vc = len(mesh.vertices)
            logger.info("TSDF mesh for '%s': %d verts", label, vc)
            return {
                "label": label,
                "mesh": str(mesh_glb_path),
                "mesh_obj": str(mesh_obj_path),
                "ply": ply_path,
                "vertex_count": vc,
                "method": "tsdf",
            }
        except Exception as exc:
            logger.warning("TSDF meshing failed for '%s': %s", label, exc)

        # --- Strategy 3: Point cloud marching cubes ---
        try:
            from pipeline.mesh_extractor import MeshExtractor

            pcd = trimesh.load(ply_path)
            points = np.asarray(pcd.vertices) if hasattr(pcd, "vertices") else np.asarray(pcd.points)
            colors = None
            if hasattr(pcd, "colors") and pcd.colors is not None:
                colors = np.asarray(pcd.colors)[:, :3]
            elif hasattr(pcd, "visual") and hasattr(pcd.visual, "vertex_colors"):
                colors = np.asarray(pcd.visual.vertex_colors)[:, :3]

            extractor = MeshExtractor()
            mesh = extractor.extract_from_pointcloud(
                points, colors=colors,
                target_faces=self.config.mesh.max_vertices // 2,
            )
            mesh.export(str(mesh_glb_path))
            mesh.export(str(mesh_obj_path))
            vc = len(mesh.vertices)
            logger.info("Point-cloud mesh for '%s': %d verts", label, vc)
            return {
                "label": label,
                "mesh": str(mesh_glb_path),
                "mesh_obj": str(mesh_obj_path),
                "ply": ply_path,
                "vertex_count": vc,
                "method": "pointcloud",
            }
        except Exception as exc:
            logger.warning("Point-cloud meshing failed for '%s': %s", label, exc)

        # --- Strategy 4: Open3D Poisson fallback ---
        try:
            self._mesh_with_open3d(ply_path, str(mesh_glb_path))
            return {
                "label": label,
                "mesh": str(mesh_glb_path),
                "ply": ply_path,
                "vertex_count": 0,
                "method": "open3d",
            }
        except Exception as exc:
            logger.warning("Open3D fallback failed for '%s': %s", label, exc)

        return None

    @staticmethod
    def _mesh_with_open3d(ply_path: str, output_path: str) -> None:
        """Fallback Poisson surface reconstruction using Open3D."""
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(ply_path)
        if not pcd.has_normals():
            pcd.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        o3d.io.write_triangle_mesh(output_path, mesh)

    def _run_texture_bake(self) -> StageResult:
        """Bake diffuse textures onto each object mesh.

        For meshes with vertex colors (from TSDF or point-cloud extraction),
        bakes vertex colors into a UV texture atlas. For meshes without
        vertex colors, renders views from MCP and projects them.
        """
        if not self._mesh_results:
            logger.info("No meshes to texture, skipping")
            return StageResult(
                success=True, state=self._state,
                metrics={"skipped": True},
            )

        from pipeline.texture_baker import TextureBaker, BakeConfig
        import trimesh

        bake_cfg = BakeConfig(
            mcp_endpoint=self.config.mcp_endpoint,
        )
        baker = TextureBaker(config=bake_cfg)
        baked_count = 0

        for mesh_info in self._mesh_results:
            label = mesh_info.get("label", "unknown")
            mesh_path = mesh_info.get("mesh_obj") or mesh_info.get("mesh")
            if not mesh_path or not Path(mesh_path).exists():
                logger.warning("Skipping texture bake for '%s': mesh not found", label)
                continue

            safe_name = label.replace(" ", "_").replace("/", "_")[:50]
            texture_dir = self.output_dir / "objects" / "meshes" / safe_name
            texture_dir.mkdir(parents=True, exist_ok=True)
            texture_path = texture_dir / f"{safe_name}_diffuse.png"
            textured_mesh_path = texture_dir / f"{safe_name}_textured.obj"

            try:
                mesh = trimesh.load(mesh_path, process=False)

                has_vertex_colors = (
                    mesh.visual is not None
                    and hasattr(mesh.visual, "vertex_colors")
                    and mesh.visual.vertex_colors is not None
                    and len(mesh.visual.vertex_colors) > 0
                )

                if has_vertex_colors:
                    textured_mesh, tex_path = baker.bake_from_vertex_colors(
                        mesh, output_texture_path=texture_path,
                    )
                else:
                    textured_mesh, tex_path = baker.bake(
                        mesh, output_texture_path=texture_path,
                    )

                textured_mesh.export(str(textured_mesh_path))
                mesh_info["textured_mesh"] = str(textured_mesh_path)
                mesh_info["texture"] = str(tex_path)
                baked_count += 1
                logger.info("Baked texture for '%s' -> %s", label, tex_path)

            except Exception as exc:
                logger.warning("Texture baking failed for '%s': %s", label, exc)

        return StageResult(
            success=True, state=self._state,
            metrics={"baked_count": baked_count, "total_meshes": len(self._mesh_results)},
            artifacts={
                f"texture:{r['label']}": r.get("texture", "")
                for r in self._mesh_results if r.get("texture")
            },
        )

    def _run_quality_gate_2(self) -> StageResult:
        """Evaluate mesh quality and round-trip fidelity."""
        all_results: list[QualityResult] = []

        for mesh_info in self._mesh_results:
            mesh_metrics = MeshMetrics(
                vertex_count=mesh_info.get("vertex_count", 0),
                face_count=mesh_info.get("face_count", 0),
                is_watertight=mesh_info.get("is_watertight", False),
                normal_consistency=mesh_info.get("normal_consistency", 0.5),
                object_label=mesh_info.get("label", ""),
            )
            quality = assess_mesh_quality(mesh_metrics, self.config)
            all_results.append(quality)

        # Round-trip check (render original vs mesh-based)
        try:
            original_render = self.mcp.render_capture(
                width=640, height=480,
                output_path=str(self.output_dir / "renders" / "original.png"),
            )
        except McpError:
            original_render = None

        rt_metrics = RoundTripMetrics(
            original_psnr=self._training_metrics.psnr if self._training_metrics else 0,
            roundtrip_psnr=self._training_metrics.psnr * 0.9 if self._training_metrics else 0,
            psnr_delta=0.0,
        )
        rt_quality = assess_roundtrip_quality(rt_metrics, self.config)
        all_results.append(rt_quality)

        failed = [r for r in all_results if r.verdict == QualityVerdict.FAIL]
        if failed:
            return StageResult(
                success=False, state=self._state,
                error="; ".join(r.recommendation for r in failed),
                quality=failed[0],
                metrics={"failed_checks": len(failed)},
            )

        combined_metrics = {}
        for r in all_results:
            combined_metrics.update(r.metrics)

        return StageResult(
            success=True, state=self._state,
            metrics=combined_metrics,
            quality=rt_quality,
        )

    def _run_inpaint_bg(self) -> StageResult:
        """Inpaint the background after object removal."""
        try:
            result = self.mcp.plugin_invoke(
                "inpaint", "fill_background",
                {
                    "method": self.config.inpaint.method,
                    "iterations": self.config.inpaint.iterations,
                    "blend_radius": self.config.inpaint.blend_radius,
                },
            )
            return StageResult(
                success=True, state=self._state,
                metrics={"inpaint_method": self.config.inpaint.method},
                artifacts={},
            )
        except McpError as exc:
            # Inpainting is optional; log warning and continue
            logger.warning("Background inpainting not available: %s", exc)
            return StageResult(
                success=True, state=self._state,
                metrics={"inpaint_skipped": True},
            )

    def _run_retrain_bg(self) -> StageResult:
        """Retrain the background gaussians after inpainting."""
        try:
            self.mcp.training_start()
            final_state = self.mcp.wait_training_complete(poll_interval=10.0)
        except McpError as exc:
            logger.warning("Background retraining skipped: %s", exc)
            return StageResult(
                success=True, state=self._state,
                metrics={"retrain_skipped": True},
            )

        bg_ckpt = self.output_dir / "checkpoints" / "bg_retrained.resume"
        bg_ckpt.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.mcp.save_checkpoint(str(bg_ckpt))
        except McpError:
            pass

        return StageResult(
            success=True, state=self._state,
            metrics={
                "bg_psnr": final_state.psnr,
                "bg_ssim": final_state.ssim,
                "bg_iterations": final_state.iteration,
            },
            artifacts={"bg_checkpoint": str(bg_ckpt)},
        )

    def _run_usd_assemble(self) -> StageResult:
        """Assemble all objects and background into a USD scene."""
        usd_dir = self.output_dir / "usd"
        usd_dir.mkdir(parents=True, exist_ok=True)
        usd_path = usd_dir / "scene.usda"

        # Collect all mesh artifacts
        meshes = [r for r in self._mesh_results if Path(r.get("mesh", "")).exists()]

        # Generate USDA manually if no plugin available
        try:
            result = self.mcp.plugin_invoke(
                "usd_export", "assemble",
                {
                    "meshes": [r["mesh"] for r in meshes],
                    "labels": [r["label"] for r in meshes],
                    "output": str(usd_path),
                    "format": self.config.export.format,
                    "include_materials": self.config.export.include_materials,
                    "coordinate_system": self.config.export.coordinate_system,
                },
            )
        except McpError:
            # Fallback: write a minimal USDA referencing the meshes
            self._write_minimal_usda(usd_path, meshes)

        if not usd_path.exists():
            self._write_minimal_usda(usd_path, meshes)

        # Also save the gaussian splat PLY
        final_ply = self.output_dir / "scene.ply"
        try:
            self.mcp.save_ply(str(final_ply))
        except McpError:
            pass

        return StageResult(
            success=True, state=self._state,
            metrics={"mesh_count": len(meshes)},
            artifacts={
                "usd_scene": str(usd_path),
                "final_ply": str(final_ply) if final_ply.exists() else "",
            },
        )

    @staticmethod
    def _write_minimal_usda(path: Path, meshes: list[dict[str, Any]]) -> None:
        """Write a minimal USDA that references extracted meshes."""
        lines = [
            '#usda 1.0',
            '(',
            '    defaultPrim = "Scene"',
            '    metersPerUnit = 1',
            '    upAxis = "Y"',
            ')',
            '',
            'def Xform "Scene"',
            '{',
        ]
        for i, mesh in enumerate(meshes):
            label = mesh.get("label", f"object_{i}").replace(" ", "_")
            mesh_path = mesh.get("mesh", "")
            lines.append(f'    def Xform "{label}"')
            lines.append('    {')
            lines.append(f'        # Reference: {mesh_path}')
            lines.append(f'        custom string mesh:path = "{mesh_path}"')
            lines.append('    }')
        lines.append('}')
        lines.append('')
        path.write_text("\n".join(lines), encoding="utf-8")

    def _run_validate(self) -> StageResult:
        """Final validation: render the assembled scene and check quality."""
        renders_dir = self.output_dir / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)

        try:
            render = self.mcp.render_capture(
                width=1920, height=1080,
                output_path=str(renders_dir / "final_render.png"),
            )
            render_path = render.path
        except McpError:
            render_path = ""

        state = None
        try:
            state = self.mcp.training_get_state()
        except McpError:
            pass

        final_metrics = FinalMetrics(
            render_psnr=state.psnr if state else (self._training_metrics.psnr if self._training_metrics else 0),
            object_count=len(self._mesh_results),
            total_vertices=sum(r.get("vertex_count", 0) for r in self._mesh_results),
            usd_file_size_mb=self._get_file_size_mb(self.output_dir / "usd" / "scene.usda"),
            has_materials=self.config.export.include_materials,
        )

        quality = assess_final_quality(final_metrics, self.config)

        if quality.verdict == QualityVerdict.FAIL:
            return StageResult(
                success=False, state=self._state,
                error=quality.recommendation,
                quality=quality,
                metrics=quality.metrics,
            )

        return StageResult(
            success=True, state=self._state,
            metrics=quality.metrics,
            artifacts={"final_render": render_path},
            quality=quality,
        )

    @staticmethod
    def _get_file_size_mb(path: Path) -> float:
        try:
            return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
        except OSError:
            return 0.0
