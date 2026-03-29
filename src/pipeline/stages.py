# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent pipeline stages callable by Claude Code or scripts.

Each stage is a self-contained method that takes explicit inputs and returns
explicit outputs as a dict. There is no hidden state, no state machine, and
no automatic transitions. Claude Code (or any caller) decides what to run
next based on the results.

Usage from the terminal::

    from pipeline.stages import PipelineStages
    p = PipelineStages('/data/output/JOB_ID')
    result = p.ingest('/data/output/JOB_ID/input.mp4', fps=2.0)
    print(result)

All heavy imports (torch, cv2, trimesh, etc.) are deferred to the methods
that need them so that the module loads instantly.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage names (ordered) -- used by the web UI for display
# ---------------------------------------------------------------------------

STAGE_NAMES: list[str] = [
    "ingest",
    "remove_people",
    "select_frames",
    "reconstruct",
    "train",
    "segment",
    "extract_objects",
    "mesh_objects",
    "texture_bake",
    "assemble_usd",
    "validate",
]


# ---------------------------------------------------------------------------
# Result dataclass returned by every stage
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Outcome of a single pipeline stage execution."""
    success: bool
    stage: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Static helpers (carried over from old orchestrator)
# ---------------------------------------------------------------------------

def _load_ply_points(ply_path: str) -> tuple:
    """Load points and colors from a PLY file (handles both 3DGS and standard PLYs).

    Returns:
        Tuple of (points: np.ndarray Nx3, colors: np.ndarray Nx3 or None)
    """
    import numpy as np

    points = None
    colors = None

    # Try plyfile first (handles 3DGS PLYs with custom properties)
    try:
        from plyfile import PlyData
        ply = PlyData.read(ply_path)
        vertex = ply["vertex"]
        x = np.asarray(vertex["x"])
        y = np.asarray(vertex["y"])
        z = np.asarray(vertex["z"])
        points = np.column_stack([x, y, z])

        for r_name, g_name, b_name in [
            ("red", "green", "blue"),
            ("f_dc_0", "f_dc_1", "f_dc_2"),
        ]:
            if r_name in vertex.data.dtype.names:
                r = np.asarray(vertex[r_name])
                g = np.asarray(vertex[g_name])
                b = np.asarray(vertex[b_name])
                c = np.column_stack([r, g, b])
                if c.max() <= 1.0:
                    c = (c * 255).clip(0, 255).astype(np.uint8)
                elif c.max() > 255:
                    c = ((c * 0.2822 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                colors = c
                break

        valid = np.isfinite(points).all(axis=1)
        if not valid.all():
            points = points[valid]
            if colors is not None:
                colors = colors[valid]

        return points, colors
    except Exception:
        pass

    # Fallback: trimesh
    try:
        import trimesh
        loaded = trimesh.load(ply_path)
        if hasattr(loaded, "vertices") and loaded.vertices is not None:
            points = np.asarray(loaded.vertices)
        elif hasattr(loaded, "points"):
            points = np.asarray(loaded.points) if hasattr(loaded.points, '__len__') else None

        if points is not None:
            try:
                if hasattr(loaded, "visual") and hasattr(loaded.visual, "vertex_colors"):
                    vc = np.asarray(loaded.visual.vertex_colors)
                    if vc.ndim == 2 and vc.shape[1] >= 3:
                        colors = vc[:, :3]
            except Exception:
                pass

            valid = np.isfinite(points).all(axis=1)
            if not valid.all():
                points = points[valid]
                if colors is not None:
                    colors = colors[valid]

        return points, colors
    except Exception:
        pass

    return None, None


def _mesh_with_open3d(ply_path: str, output_path: str) -> None:
    """Fallback Poisson surface reconstruction using Open3D."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_normals():
        pcd.estimate_normals()
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    o3d.io.write_triangle_mesh(output_path, mesh)


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


def _find_usd_python() -> Path | None:
    """Locate a Python 3.11 interpreter with usd-core installed.

    Checks well-known venv paths and the system python3.11 in order
    of preference.
    """
    candidates = [
        Path("/opt/venv-usd/bin/python3"),
        Path.home() / "workspace" / "gaussians" / "LichtFeld-Studio" / ".venv-usd" / "bin" / "python3",
        Path("/usr/bin/python3.11"),
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            result = subprocess.run(
                [str(p), "-c", "from pxr import Usd; print('ok')"],
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "PYTHONPATH": ""},
            )
            if result.returncode == 0 and "ok" in result.stdout:
                return p
        except (OSError, subprocess.TimeoutExpired):
            continue
    return None


def _get_file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# PipelineStages -- the main class
# ---------------------------------------------------------------------------

class PipelineStages:
    """Individual pipeline stages callable by Claude Code or scripts.

    Each method is self-contained, takes explicit inputs, returns explicit
    outputs as a StageResult. No hidden state, no automatic transitions.
    Claude Code calls them in sequence, inspecting results between each call.
    """

    def __init__(self, job_dir: str, config: PipelineConfig | None = None) -> None:
        self.job_dir = Path(job_dir)
        self.config = config or PipelineConfig()
        self.config.output_dir = str(self.job_dir)

    # ------------------------------------------------------------------
    # Stage 1: Ingest
    # ------------------------------------------------------------------

    def ingest(self, video_path: str, fps: float = 2.0) -> StageResult:
        """Extract frames from video.

        Returns: {frames_dir, frame_count}
        """
        video = Path(video_path)
        if not video.exists():
            return StageResult(
                success=False, stage="ingest",
                error=f"Video not found: {video_path}",
            )

        frame_dir = self.job_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(frame_dir / "frame_%05d.jpg"),
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except FileNotFoundError:
            return StageResult(success=False, stage="ingest", error="ffmpeg not found in PATH")
        except subprocess.TimeoutExpired:
            return StageResult(success=False, stage="ingest", error="Frame extraction timed out (300s)")

        if proc.returncode != 0:
            return StageResult(
                success=False, stage="ingest",
                error=f"ffmpeg failed (rc={proc.returncode}): {proc.stderr[:500]}",
            )

        frames = sorted(frame_dir.glob("*.jpg"))
        return StageResult(
            success=True, stage="ingest",
            metrics={"frame_count": len(frames), "fps": fps},
            artifacts={"frames_dir": str(frame_dir), "frame_count": str(len(frames))},
        )

    # ------------------------------------------------------------------
    # Stage 2: Remove people
    # ------------------------------------------------------------------

    def remove_people(self, frames_dir: str) -> StageResult:
        """Detect and remove people from frames.

        Returns: {cleaned_dir, manifest}
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            return StageResult(
                success=False, stage="remove_people",
                error=f"Frames directory not found: {frames_dir}",
            )

        cfg = self.config.person_removal
        if not cfg.enabled:
            logger.info("Person removal disabled, skipping")
            return StageResult(
                success=True, stage="remove_people",
                metrics={"skipped": True},
                artifacts={"cleaned_frames_dir": frames_dir},
            )

        from pipeline.person_remover import PersonRemover

        cleaned_dir = self.job_dir / "frames_cleaned"
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
            manifest = remover.process_directory(frames_dir, str(cleaned_dir))
        except Exception as exc:
            return StageResult(
                success=False, stage="remove_people",
                error=f"Person removal failed: {exc}",
            )

        summary = manifest.get("summary", {})
        remaining = summary.get("clean", 0) + summary.get("inpainted", 0) + summary.get("flagged_inpainted", 0)

        if remaining < self.config.ingest.min_frames:
            return StageResult(
                success=False, stage="remove_people",
                error=f"Too few frames after person removal: {remaining} (need {self.config.ingest.min_frames})",
                metrics=summary,
            )

        manifest_path = cleaned_dir / "person_removal_manifest.json"
        return StageResult(
            success=True, stage="remove_people",
            metrics={"total_frames": manifest.get("total_frames", 0), "remaining_frames": remaining, **summary},
            artifacts={"cleaned_frames_dir": str(cleaned_dir), "manifest": str(manifest_path)},
        )

    # ------------------------------------------------------------------
    # Stage 3: Select frames
    # ------------------------------------------------------------------

    def select_frames(self, frames_dir: str, target: int = 150) -> StageResult:
        """Select best frames for COLMAP.

        Returns: {selected_dir, count}
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            return StageResult(
                success=False, stage="select_frames",
                error=f"Frames directory not found: {frames_dir}",
            )

        from pipeline.frame_selector import FrameSelector, SelectionConfig

        sel_cfg = SelectionConfig(
            target_frames=min(self.config.ingest.max_frames, max(self.config.ingest.min_frames, target)),
            min_frames=self.config.ingest.min_frames,
            max_frames=self.config.ingest.max_frames,
            blur_threshold=self.config.ingest.blur_threshold,
        )
        selector = FrameSelector(config=sel_cfg)

        try:
            scores = selector.score_frames(frames_dir)
        except Exception as exc:
            logger.warning("Frame scoring failed (%s), keeping all frames", exc)
            return StageResult(
                success=True, stage="select_frames",
                metrics={"skipped": True, "reason": str(exc)},
                artifacts={"selected_frames_dir": frames_dir},
            )

        if not scores:
            return StageResult(
                success=False, stage="select_frames",
                error=f"No frames found in {frames_dir}",
            )

        selected = selector.select(scores)

        if len(selected) < self.config.ingest.min_frames:
            return StageResult(
                success=False, stage="select_frames",
                error=f"Only {len(selected)} frames after selection (need {self.config.ingest.min_frames})",
                metrics={"scored": len(scores), "selected": len(selected)},
            )

        selected_dir = self.job_dir / "frames_selected"
        selector.copy_selected(selected, str(selected_dir))

        return StageResult(
            success=True, stage="select_frames",
            metrics={"scored": len(scores), "selected": len(selected), "target": sel_cfg.target_frames},
            artifacts={"selected_frames_dir": str(selected_dir), "count": str(len(selected))},
        )

    # ------------------------------------------------------------------
    # Stage 4: Reconstruct (COLMAP SfM)
    # ------------------------------------------------------------------

    def reconstruct(self, frames_dir: str) -> StageResult:
        """Run COLMAP SfM reconstruction.

        Returns: {colmap_dir, cameras, points}
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            return StageResult(
                success=False, stage="reconstruct",
                error=f"Frames directory not found: {frames_dir}",
            )

        colmap_dir = self.job_dir / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)

        # Try SplatReady first
        splatready_config = {
            "video_path": "",
            "base_output_folder": str(self.job_dir),
            "frame_rate": self.config.ingest.fps,
            "skip_extraction": True,
            "reconstruction_method": self.config.reconstruct.method,
            "colmap_exe_path": self.config.reconstruct.colmap_exe,
            "use_fisheye": self.config.reconstruct.use_fisheye,
            "max_image_size": self.config.ingest.max_image_size,
            "min_scale": self.config.reconstruct.min_scale,
            "skip_reconstruction": False,
        }

        config_path = self.job_dir / "splatready_config.json"
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
                        success=False, stage="reconstruct",
                        error=f"SplatReady failed: {proc.stderr[:500]}",
                    )
            except subprocess.TimeoutExpired:
                return StageResult(
                    success=False, stage="reconstruct",
                    error="COLMAP reconstruction timed out",
                )
        else:
            # Fallback: run COLMAP directly
            try:
                self._run_colmap_direct(colmap_dir, frames_path)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
                return StageResult(
                    success=False, stage="reconstruct",
                    error=f"COLMAP failed: {exc}",
                )

        dataset_dir = colmap_dir / "undistorted"
        if not dataset_dir.exists():
            for alt in [colmap_dir, self.job_dir / "colmap"]:
                if (alt / "images").exists() and (alt / "sparse").exists():
                    dataset_dir = alt
                    break
            else:
                return StageResult(
                    success=False, stage="reconstruct",
                    error="COLMAP output not found at expected paths",
                )

        # Count cameras / points if available
        sparse_dir = dataset_dir / "sparse" / "0"
        cameras = len(list(sparse_dir.glob("cameras.*"))) if sparse_dir.exists() else 0
        points_file = sparse_dir / "points3D.bin" if sparse_dir.exists() else None
        points_size = _get_file_size_mb(points_file) if points_file else 0.0

        return StageResult(
            success=True, stage="reconstruct",
            metrics={"colmap_dir": str(dataset_dir), "cameras_found": cameras, "points_size_mb": points_size},
            artifacts={"colmap_dir": str(dataset_dir)},
        )

    def _run_colmap_direct(self, output_dir: Path, frame_dir: Path) -> None:
        """Fallback: run COLMAP feature extraction + matching + mapper."""
        db_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        colmap = self.config.reconstruct.colmap_exe

        subprocess.run([
            colmap, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(frame_dir),
            "--ImageReader.single_camera", "1",
        ], check=True, capture_output=True, timeout=300)

        subprocess.run([
            colmap, self.config.reconstruct.matcher + "_matcher",
            "--database_path", str(db_path),
        ], check=True, capture_output=True, timeout=300)

        subprocess.run([
            colmap, "mapper",
            "--database_path", str(db_path),
            "--image_path", str(frame_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.multiple_models", "0",
        ], check=True, capture_output=True, timeout=1800)

        undist_dir = output_dir / "undistorted"
        subprocess.run([
            colmap, "image_undistorter",
            "--image_path", str(frame_dir),
            "--input_path", str(sparse_dir / "0"),
            "--output_path", str(undist_dir),
            "--output_type", "COLMAP",
        ], check=True, capture_output=True, timeout=300)

        undist_sparse = undist_dir / "sparse"
        if undist_sparse.exists() and not (undist_sparse / "0").exists():
            (undist_sparse / "0").mkdir(exist_ok=True)
            for f in undist_sparse.glob("*.bin"):
                shutil.move(str(f), str(undist_sparse / "0" / f.name))

    # ------------------------------------------------------------------
    # Stage 5: Train (Gaussian splatting)
    # ------------------------------------------------------------------

    def train(self, colmap_dir: str, iterations: int = 30000) -> StageResult:
        """Run LichtFeld headless training.

        Returns: {ply_path, loss}
        """
        dataset_dir = Path(colmap_dir)
        if not dataset_dir.exists():
            return StageResult(
                success=False, stage="train",
                error=f"COLMAP directory not found: {colmap_dir}",
            )

        model_dir = self.job_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        lfs_binary = self.config.training.lichtfeld_binary
        if not Path(lfs_binary).exists():
            for candidate in [
                "/opt/gaussian-toolkit/build/LichtFeld-Studio",
                "/usr/local/bin/lichtfeld-studio",
                str(Path.home() / "workspace/gaussians/LichtFeld-Studio/build/LichtFeld-Studio"),
            ]:
                if Path(candidate).exists():
                    lfs_binary = candidate
                    break

        if not Path(lfs_binary).exists():
            return StageResult(
                success=False, stage="train",
                error=f"LichtFeld binary not found at {lfs_binary}",
            )

        strategy = self.config.training.strategy
        sh_degree = self.config.training.sh_degree

        train_cmd = [
            lfs_binary, "--headless",
            "--data-path", str(dataset_dir),
            "--output-path", str(model_dir),
            "--iter", str(iterations),
            "--strategy", strategy,
            "--sh-degree", str(sh_degree),
            "--log-level", "info",
        ]
        logger.info("Training: %s", " ".join(train_cmd))

        try:
            proc = subprocess.run(
                train_cmd, capture_output=True, text=True,
                timeout=3600,
                env={**os.environ, "LD_LIBRARY_PATH": f"{Path(lfs_binary).parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"},
            )
            if proc.returncode != 0:
                logger.warning("Training stderr: %s", proc.stderr[-500:] if proc.stderr else "none")
        except subprocess.TimeoutExpired:
            return StageResult(success=False, stage="train", error="Training timed out (3600s)")

        # Find trained PLY
        ply_files = sorted(model_dir.rglob("*.ply"))
        if not ply_files:
            return StageResult(
                success=False, stage="train",
                error="No PLY output from training",
                metrics={"stderr_tail": (proc.stderr or "")[-300:]},
            )

        ply_path = ply_files[-1]
        ply_size_mb = _get_file_size_mb(ply_path)

        return StageResult(
            success=True, stage="train",
            metrics={
                "ply_path": str(ply_path),
                "ply_size_mb": round(ply_size_mb, 1),
                "iterations": iterations,
                "strategy": strategy,
            },
            artifacts={"ply_path": str(ply_path), "model_dir": str(model_dir)},
        )

    # ------------------------------------------------------------------
    # Stage 6: Segment (SAM2/SAM3)
    # ------------------------------------------------------------------

    def segment(self, ply_path: str, frames_dir: str) -> StageResult:
        """SAM2/SAM3 segmentation + mask projection.

        Returns: {objects: list[{label, object_id, mask_pixels}], masks_dir}
        """
        import cv2

        frames_path = Path(frames_dir)
        if not frames_path.exists():
            return StageResult(
                success=False, stage="segment",
                error=f"Frames directory not found: {frames_dir}",
            )

        frame_paths = sorted(
            p for p in frames_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not frame_paths:
            return StageResult(
                success=False, stage="segment",
                error=f"No frames found in {frames_dir}",
            )

        decompose_cfg = self.config.decompose
        concepts = decompose_cfg.sam3_concepts or decompose_cfg.descriptions or [
            "paintings", "frames", "sculptures", "furniture",
            "walls", "floor", "ceiling", "fixtures", "doorways",
        ]

        # Try SAM3 first
        if decompose_cfg.use_sam3:
            try:
                from pipeline.sam3_segmentor import SAM3Segmentor
                import numpy as np

                seg = SAM3Segmentor(
                    device="cuda",
                    confidence_threshold=decompose_cfg.sam3_confidence_threshold,
                )

                image = cv2.imread(str(frame_paths[0]))
                if image is None:
                    return StageResult(
                        success=False, stage="segment",
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

                # Save masks
                masks_dir = self.job_dir / "sam3_masks"
                masks_dir.mkdir(parents=True, exist_ok=True)
                for i, obj_id in enumerate(result.object_ids):
                    mask_path = masks_dir / f"mask_{int(obj_id):04d}.npy"
                    np.save(str(mask_path), result.masks[i])

                seg.unload()

                if not objects:
                    objects = [{"label": "full_scene", "count": -1}]

                return StageResult(
                    success=True, stage="segment",
                    metrics={
                        "object_count": len(objects),
                        "method": "sam3",
                        "concepts": concepts,
                    },
                    artifacts={
                        "objects": json.dumps(objects),
                        "masks_dir": str(masks_dir),
                    },
                )
            except Exception as exc:
                if not decompose_cfg.sam3_fallback_to_sam2:
                    return StageResult(
                        success=False, stage="segment",
                        error=f"SAM3 segmentation failed: {exc}",
                    )
                logger.warning("SAM3 failed (%s), falling back to full-scene", exc)

        # Fallback: treat entire scene as one object
        return StageResult(
            success=True, stage="segment",
            metrics={"object_count": 1, "method": "full_scene"},
            artifacts={"objects": json.dumps([{"label": "full_scene", "count": -1}])},
        )

    # ------------------------------------------------------------------
    # Stage 7: Extract objects
    # ------------------------------------------------------------------

    def extract_objects(self, ply_path: str, labels: dict | list | str | None = None) -> StageResult:
        """Extract per-object PLY files from the trained model.

        Args:
            ply_path: Path to the trained gaussian PLY.
            labels: Object definitions from segment(). Can be a JSON string,
                    list of dicts, or None (full_scene fallback).

        Returns: {object_plys: list[str]}
        """
        trained_ply = Path(ply_path)
        if not trained_ply.exists():
            return StageResult(
                success=False, stage="extract_objects",
                error=f"PLY not found: {ply_path}",
            )

        # Parse labels
        if labels is None:
            objects = [{"label": "full_scene", "count": -1}]
        elif isinstance(labels, str):
            objects = json.loads(labels)
        elif isinstance(labels, dict):
            objects = [labels]
        else:
            objects = list(labels)

        objects_dir = self.job_dir / "objects"
        objects_dir.mkdir(parents=True, exist_ok=True)

        masks_dir = self.job_dir / "sam3_masks"
        has_masks = masks_dir.exists() and any(masks_dir.glob("*.npy"))

        extracted: list[dict[str, Any]] = []

        for obj in objects:
            label = obj.get("label", "unknown")
            safe_name = label.replace(" ", "_").replace("/", "_")[:50]
            out_ply = objects_dir / f"{safe_name}.ply"

            if label == "full_scene" or not has_masks:
                shutil.copy2(str(trained_ply), str(out_ply))
                extracted.append({"label": label, "ply": str(out_ply)})
                logger.info("Copied trained PLY as '%s': %s", label, out_ply)
            else:
                # Try mask-based extraction
                try:
                    ok = self._extract_with_mask(obj, trained_ply, out_ply, masks_dir)
                    if ok:
                        extracted.append({"label": label, "ply": str(out_ply)})
                        continue
                except Exception as exc:
                    logger.warning("Mask extraction failed for '%s': %s", label, exc)

                # Fallback: copy full PLY
                shutil.copy2(str(trained_ply), str(out_ply))
                extracted.append({"label": label, "ply": str(out_ply)})

        if not extracted:
            return StageResult(
                success=False, stage="extract_objects",
                error="No objects could be extracted",
            )

        return StageResult(
            success=True, stage="extract_objects",
            metrics={"extracted_count": len(extracted)},
            artifacts={
                "object_plys": json.dumps([e["ply"] for e in extracted]),
                **{f"ply:{e['label']}": e["ply"] for e in extracted},
            },
        )

    def _extract_with_mask(
        self,
        obj: dict[str, Any],
        trained_ply: Path,
        output_path: Path,
        masks_dir: Path,
    ) -> bool:
        """Extract a subset of gaussians using a SAM3 mask. Returns True on success."""
        import numpy as np

        object_id = obj.get("object_id")
        if object_id is None:
            return False

        mask_path = masks_dir / f"mask_{int(object_id):04d}.npy"
        if not mask_path.exists():
            return False

        mask = np.load(str(mask_path))

        import trimesh
        pcd = trimesh.load(str(trained_ply))
        if hasattr(pcd, "vertices"):
            points = np.asarray(pcd.vertices)
        else:
            points = np.asarray(pcd.points) if hasattr(pcd, "points") else None
        if points is None or len(points) == 0:
            return False

        h, w = mask.shape[-2], mask.shape[-1]
        xy = points[:, :2]
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        xy_range = xy_max - xy_min
        xy_range[xy_range == 0] = 1.0

        px = ((xy[:, 0] - xy_min[0]) / xy_range[0] * (w - 1)).astype(int).clip(0, w - 1)
        py = ((xy[:, 1] - xy_min[1]) / xy_range[1] * (h - 1)).astype(int).clip(0, h - 1)

        mask_2d = mask[0] if mask.ndim == 3 else mask
        inside = mask_2d[py, px] > 0
        if inside.sum() == 0:
            return False

        filtered_points = points[inside]
        colors = None
        if hasattr(pcd, "visual") and hasattr(pcd.visual, "vertex_colors"):
            all_colors = np.asarray(pcd.visual.vertex_colors)
            colors = all_colors[inside]

        filtered_pcd = trimesh.PointCloud(filtered_points)
        if colors is not None:
            filtered_pcd.colors = colors

        filtered_pcd.export(str(output_path))
        logger.info("Extracted %d/%d points for '%s'", inside.sum(), len(points), obj.get("label", "unknown"))
        return True

    # ------------------------------------------------------------------
    # Stage 8: Mesh objects
    # ------------------------------------------------------------------

    def mesh_objects(self, object_plys: list[str] | str) -> StageResult:
        """Generate meshes per object PLY.

        Args:
            object_plys: List of PLY paths, or a JSON string of a list.

        Returns: {meshes: list[{label, mesh, ply, vertex_count, method}]}
        """
        if isinstance(object_plys, str):
            object_plys = json.loads(object_plys)

        import trimesh

        meshes_dir = self.job_dir / "objects" / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []

        for ply_path in object_plys:
            ply = Path(ply_path)
            if not ply.exists():
                logger.warning("PLY not found: %s", ply_path)
                continue

            label = ply.stem
            obj_dir = meshes_dir / label
            obj_dir.mkdir(parents=True, exist_ok=True)
            mesh_glb = obj_dir / f"{label}.glb"
            mesh_obj = obj_dir / f"{label}.obj"

            mesh_result = self._mesh_single(label, str(ply), obj_dir, mesh_obj, mesh_glb)
            if mesh_result is not None:
                results.append(mesh_result)

        if not results:
            return StageResult(
                success=False, stage="mesh_objects",
                error="No meshes could be generated",
            )

        return StageResult(
            success=True, stage="mesh_objects",
            metrics={"mesh_count": len(results)},
            artifacts={
                "meshes": json.dumps(results),
                **{f"mesh:{r['label']}": r["mesh"] for r in results},
            },
        )

    def _mesh_single(
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

        points, colors = _load_ply_points(ply_path)
        if points is None or len(points) == 0:
            logger.warning("No points loaded from PLY for '%s'", label)
            return None

        centroid = points.mean(axis=0)
        extent = points.max(axis=0) - points.min(axis=0)
        scale = extent.max()
        if scale == 0:
            scale = 1.0
        points_norm = (points - centroid) / scale

        logger.info("PLY '%s': %d points, extent=%.1f", label, len(points), scale)

        def _export_mesh(mesh: "trimesh.Trimesh", method: str) -> dict[str, Any] | None:
            mesh.vertices = mesh.vertices * scale + centroid
            mesh.export(str(mesh_glb_path))
            mesh.export(str(mesh_obj_path))
            vc = len(mesh.vertices)
            logger.info("%s mesh for '%s': %d verts", method, label, vc)
            return {
                "label": label, "mesh": str(mesh_glb_path), "mesh_obj": str(mesh_obj_path),
                "ply": ply_path, "vertex_count": vc, "method": method,
            }

        # Strategy 0: gsplat depth rendering -> TSDF (preferred, GPU-accelerated)
        try:
            import torch
            if torch.cuda.is_available():
                from pipeline.mesh_extractor import MeshExtractor, TSDFConfig

                extractor = MeshExtractor(config=TSDFConfig(
                    target_faces=self.config.mesh.max_vertices // 2,
                ))
                mesh, color_images, cameras = extractor.extract_from_gsplat(
                    ply_path,
                    num_views=64,
                    render_size=1024,
                    target_faces=self.config.mesh.max_vertices // 2,
                )
                # gsplat returns mesh in world coordinates already, no rescale needed
                mesh.export(str(mesh_glb_path))
                mesh.export(str(mesh_obj_path))
                vc = len(mesh.vertices)
                logger.info("gsplat mesh for '%s': %d verts", label, vc)

                # Bake texture from gsplat color renders
                texture_path = obj_dir / f"{label}_diffuse.png"
                try:
                    from pipeline.texture_baker import TextureBaker, BakeConfig
                    baker = TextureBaker(config=BakeConfig(texture_size=2048))
                    # Convert cameras to numpy poses for the baker
                    cam_poses = []
                    for viewmat, K in cameras[:len(color_images)]:
                        # Baker expects camera-to-world (inverse of viewmat)
                        vm_np = viewmat.cpu().numpy()
                        cam_to_world = np.linalg.inv(vm_np)
                        cam_poses.append(cam_to_world)
                    color_uint8 = [
                        np.clip(img * 255, 0, 255).astype(np.uint8) for img in color_images
                    ]
                    textured_mesh, tex_path = baker.bake(
                        mesh,
                        output_texture_path=str(texture_path),
                        color_images=color_uint8,
                        camera_poses=cam_poses,
                    )
                    textured_mesh.export(str(mesh_glb_path))
                    textured_mesh.export(str(mesh_obj_path))
                    logger.info("Baked texture for '%s': %s", label, tex_path)
                except Exception as tex_exc:
                    logger.warning("Texture baking failed for '%s': %s", label, tex_exc)

                return {
                    "label": label, "mesh": str(mesh_glb_path), "mesh_obj": str(mesh_obj_path),
                    "ply": ply_path, "vertex_count": vc, "method": "gsplat",
                }
        except Exception as exc:
            logger.warning("gsplat meshing failed for '%s': %s", label, exc)

        # Strategy 1: Hunyuan3D
        if self.config.hunyuan3d.enabled:
            try:
                from pipeline.hunyuan3d_client import Hunyuan3DClient
                import inspect

                h3d_kwargs = {
                    "comfyui_url": self.config.hunyuan3d.comfyui_url,
                    "api_url": self.config.hunyuan3d.api_url,
                    "quality": self.config.hunyuan3d.quality,
                    "turbo": self.config.hunyuan3d.turbo,
                    "timeout": self.config.hunyuan3d.timeout,
                    "seed": self.config.hunyuan3d.seed,
                }
                sig = inspect.signature(Hunyuan3DClient.__init__)
                if "multiview" in sig.parameters:
                    h3d_kwargs["multiview"] = self.config.hunyuan3d.multiview

                h3d = Hunyuan3DClient(**h3d_kwargs)
                result = h3d.reconstruct_from_gaussians(ply_path)
                if result.mesh is not None:
                    result.mesh.export(str(mesh_glb_path))
                    result.mesh.export(str(mesh_obj_path))
                    vc = len(result.mesh.vertices)
                    return {
                        "label": label, "mesh": str(mesh_glb_path), "mesh_obj": str(mesh_obj_path),
                        "ply": ply_path, "vertex_count": vc, "method": "hunyuan3d",
                    }
            except Exception as exc:
                logger.warning("Hunyuan3D failed for '%s': %s", label, exc)

        # Strategy 2: TSDF
        try:
            from pipeline.mesh_extractor import MeshExtractor, TSDFConfig

            tsdf_cfg = TSDFConfig(
                target_faces=self.config.mesh.max_vertices // 2,
                mcp_endpoint=self.config.mcp_endpoint,
            )
            extractor = MeshExtractor(config=tsdf_cfg)
            mesh = extractor.extract_from_pointcloud(
                points_norm, colors=colors,
                target_faces=self.config.mesh.max_vertices // 2,
            )
            result = _export_mesh(mesh, "tsdf")
            if result:
                return result
        except Exception as exc:
            logger.warning("TSDF meshing failed for '%s': %s", label, exc)

        # Strategy 3: Point cloud marching cubes
        try:
            from pipeline.mesh_extractor import MeshExtractor

            extractor = MeshExtractor()
            mesh = extractor.extract_from_pointcloud(
                points_norm, colors=colors,
                target_faces=self.config.mesh.max_vertices // 2,
            )
            result = _export_mesh(mesh, "pointcloud")
            if result:
                return result
        except Exception as exc:
            logger.warning("Point-cloud meshing failed for '%s': %s", label, exc)

        # Strategy 4: Open3D Poisson
        try:
            _mesh_with_open3d(ply_path, str(mesh_glb_path))
            return {
                "label": label, "mesh": str(mesh_glb_path), "ply": ply_path,
                "vertex_count": 0, "method": "open3d",
            }
        except Exception as exc:
            logger.warning("Open3D fallback failed for '%s': %s", label, exc)

        # Strategy 5: Convex hull
        try:
            pcd = trimesh.PointCloud(points)
            mesh = pcd.convex_hull
            if mesh is not None and len(mesh.vertices) > 0:
                mesh.export(str(mesh_glb_path))
                mesh.export(str(mesh_obj_path))
                vc = len(mesh.vertices)
                return {
                    "label": label, "mesh": str(mesh_glb_path), "mesh_obj": str(mesh_obj_path),
                    "ply": ply_path, "vertex_count": vc, "method": "convex_hull",
                }
        except Exception as exc:
            logger.warning("Convex hull fallback failed for '%s': %s", label, exc)

        return None

    # ------------------------------------------------------------------
    # Stage 9: Texture bake
    # ------------------------------------------------------------------

    def texture_bake(self, meshes: list[dict[str, Any]] | str) -> StageResult:
        """Bake diffuse textures onto each object mesh.

        Args:
            meshes: List of mesh info dicts (from mesh_objects) or JSON string.

        Returns: {baked_count, textured_meshes}
        """
        if isinstance(meshes, str):
            meshes = json.loads(meshes)

        if not meshes:
            return StageResult(
                success=True, stage="texture_bake",
                metrics={"skipped": True, "reason": "no meshes"},
            )

        from pipeline.texture_baker import TextureBaker, BakeConfig
        import trimesh

        bake_cfg = BakeConfig(mcp_endpoint=self.config.mcp_endpoint)
        baker = TextureBaker(config=bake_cfg)
        baked_count = 0

        for mesh_info in meshes:
            label = mesh_info.get("label", "unknown")
            mesh_path = mesh_info.get("mesh_obj") or mesh_info.get("mesh")
            if not mesh_path or not Path(mesh_path).exists():
                continue

            safe_name = label.replace(" ", "_").replace("/", "_")[:50]
            texture_dir = self.job_dir / "objects" / "meshes" / safe_name
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
            success=True, stage="texture_bake",
            metrics={"baked_count": baked_count, "total_meshes": len(meshes)},
            artifacts={
                "textured_meshes": json.dumps(meshes),
            },
        )

    # ------------------------------------------------------------------
    # Stage 10: Assemble USD
    # ------------------------------------------------------------------

    def assemble_usd(self, objects: dict | list | str, cameras: dict | str | None = None) -> StageResult:
        """Compose USD scene from meshes using the standalone Python 3.11 assembler.

        The standalone script (scripts/assemble_usd_scene.py) runs under a
        Python 3.11 venv with usd-core, since usd-core has no wheels for
        Python 3.12+. The script discovers COLMAP cameras, meshes, textures,
        and PLY files from the job directory and builds a proper USD stage.

        Falls back to a minimal USDA stub if the subprocess fails.

        Args:
            objects: Mesh results (from mesh_objects) as list/dict/JSON string.
            cameras: Optional camera data (unused in file-based mode).

        Returns: {usd_path}
        """
        if isinstance(objects, str):
            objects = json.loads(objects)
        if isinstance(objects, dict):
            objects = [objects]

        usd_dir = self.job_dir / "usd"
        usd_dir.mkdir(parents=True, exist_ok=True)
        usd_path = usd_dir / "scene.usda"

        # Copy trained PLY as scene.ply (needed by the assembler)
        final_ply = self.job_dir / "scene.ply"
        model_plys = sorted(self.job_dir.rglob("model/**/*.ply"))
        if model_plys and not final_ply.exists():
            shutil.copy2(str(model_plys[-1]), str(final_ply))
            logger.info("Copied trained PLY as scene.ply")

        # Locate the standalone assembler script
        script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "assemble_usd_scene.py"
        if not script_path.exists():
            # Docker container path
            script_path = Path("/opt/gaussian-toolkit/scripts/assemble_usd_scene.py")

        # Find Python 3.11 with usd-core
        usd_python = _find_usd_python()

        assembled = False
        if usd_python and script_path.exists():
            try:
                result = subprocess.run(
                    [
                        str(usd_python),
                        str(script_path),
                        "--job-dir", str(self.job_dir),
                        "--output", str(usd_path),
                    ],
                    capture_output=True, text=True, timeout=120,
                    env={**os.environ, "PYTHONPATH": ""},
                )
                if result.returncode == 0:
                    assembled = True
                    logger.info("USD scene assembled via standalone script:\n%s", result.stdout)
                else:
                    logger.warning(
                        "Standalone USD assembler failed (rc=%d):\nstdout: %s\nstderr: %s",
                        result.returncode, result.stdout, result.stderr,
                    )
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.warning("Standalone USD assembler error: %s", exc)

        if not assembled:
            logger.warning("Falling back to minimal USDA stub")
            meshes = [r for r in objects if Path(r.get("mesh", "")).exists()]
            _write_minimal_usda(usd_path, meshes)

        usd_size = usd_path.stat().st_size if usd_path.exists() else 0

        return StageResult(
            success=True, stage="assemble_usd",
            metrics={
                "mesh_count": len(objects),
                "usd_size_bytes": usd_size,
                "used_standalone_assembler": assembled,
            },
            artifacts={
                "usd_path": str(usd_path),
                "final_ply": str(final_ply) if final_ply.exists() else "",
            },
        )

    # ------------------------------------------------------------------
    # Stage 11: Validate
    # ------------------------------------------------------------------

    def validate(self) -> StageResult:
        """Final validation: check that key artifacts exist on disk.

        Returns: {usd_exists, ply_exists, mesh_count}
        """
        usd_path = self.job_dir / "usd" / "scene.usda"
        final_ply = self.job_dir / "scene.ply"
        meshes_dir = self.job_dir / "objects" / "meshes"

        mesh_count = 0
        if meshes_dir.exists():
            mesh_count = len(list(meshes_dir.rglob("*.glb"))) + len(list(meshes_dir.rglob("*.obj")))

        if not usd_path.exists() and mesh_count == 0:
            return StageResult(
                success=False, stage="validate",
                error="No USD scene or mesh files found",
            )

        return StageResult(
            success=True, stage="validate",
            metrics={
                "usd_exists": usd_path.exists(),
                "ply_exists": final_ply.exists(),
                "usd_size_mb": round(_get_file_size_mb(usd_path), 2),
                "ply_size_mb": round(_get_file_size_mb(final_ply), 2),
                "mesh_count": mesh_count,
            },
            artifacts={
                "usd_path": str(usd_path) if usd_path.exists() else "",
                "final_ply": str(final_ply) if final_ply.exists() else "",
            },
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Current job status for web UI.

        Reads the pipeline_status.json if it exists, otherwise inspects
        the directory tree to determine progress.
        """
        status_file = self.job_dir / self.config.status_file
        if status_file.exists():
            try:
                return json.loads(status_file.read_text())
            except Exception:
                pass

        # Infer status from directory structure
        stages_done = []
        if (self.job_dir / "frames").exists():
            stages_done.append("ingest")
        if (self.job_dir / "frames_selected").exists() or (self.job_dir / "frames_cleaned").exists():
            stages_done.append("select_frames")
        if (self.job_dir / "colmap").exists():
            stages_done.append("reconstruct")
        if (self.job_dir / "model").exists() and any(self.job_dir.rglob("model/**/*.ply")):
            stages_done.append("train")
        if (self.job_dir / "sam3_masks").exists():
            stages_done.append("segment")
        if (self.job_dir / "objects").exists():
            stages_done.append("extract_objects")
        if (self.job_dir / "objects" / "meshes").exists():
            stages_done.append("mesh_objects")
        if (self.job_dir / "usd" / "scene.usda").exists():
            stages_done.append("assemble_usd")

        return {
            "job_dir": str(self.job_dir),
            "stages_completed": stages_done,
            "progress": len(stages_done) / len(STAGE_NAMES),
        }
