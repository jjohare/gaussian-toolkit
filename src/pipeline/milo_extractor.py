# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""MILo mesh extraction -- differentiable mesh-in-the-loop gaussian splatting.

Runs MILo (github.com/Anttwo/MILo, SIGGRAPH Asia 2025) in an isolated conda
environment (Python 3.9, CUDA 11.8). Produces high-quality meshes via Delaunay
triangulation + learned SDF.

MILo cannot run in our main Python 3.12 environment. It requires:
  - Python 3.9, PyTorch 2.3.1, CUDA 11.8
  - conda env ``milo`` with CGAL, GMP, and 8 custom CUDA submodules
  - /opt/milo/train.py (or MILO_DIR env var)

This module wraps MILo as subprocess calls within the conda env and returns
paths to the resulting mesh PLY.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MILO_DIR = Path(os.environ.get("MILO_DIR", "/opt/milo"))
MILO_CONDA_ENV = os.environ.get("MILO_CONDA_ENV", "milo")


@dataclass
class MiloConfig:
    """Configuration for MILo mesh extraction.

    Attributes:
        imp_metric: Scene type -- ``indoor`` or ``outdoor``.
        rasterizer: Rasterizer backend -- ``radegs``, ``gof``, or ``ms``.
        mesh_config: Mesh density -- ``verylowres``, ``lowres``, ``default``,
            ``highres``, ``veryhighres``.
        dense_gaussians: Enable dense gaussian initialization.
        decoupled_appearance: Enable per-image appearance embeddings.
        data_device: Device for dataset storage (``cpu`` reduces VRAM).
        iterations: Training iterations (MILo default is 18000).
        mesh_extract_method: Extraction method -- ``sdf`` (best quality),
            ``integration``, or ``regular_tsdf``.
        train_timeout: Max seconds for training subprocess.
        extract_timeout: Max seconds for mesh extraction subprocess.
    """
    imp_metric: str = "indoor"
    rasterizer: str = "radegs"
    mesh_config: str = "default"
    dense_gaussians: bool = False
    decoupled_appearance: bool = False
    data_device: str = "cpu"
    iterations: int = 18000
    mesh_extract_method: str = "sdf"
    train_timeout: int = 3600
    extract_timeout: int = 600


def is_milo_available() -> bool:
    """Check if MILo is installed and the conda env exists.

    Returns True only if ``MILO_DIR/train.py`` exists AND the conda env
    can import torch successfully.
    """
    if not (MILO_DIR / "train.py").exists():
        logger.debug("MILo train.py not found at %s", MILO_DIR / "train.py")
        return False

    try:
        result = subprocess.run(
            ["conda", "run", "-n", MILO_CONDA_ENV,
             "python", "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.debug("MILo conda env check failed: %s", result.stderr[:200])
            return False
        logger.debug("MILo conda env OK, torch=%s", result.stdout.strip())
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.debug("MILo availability check failed: %s", exc)
        return False


def _find_sparse_dir(colmap_path: Path) -> Optional[Path]:
    """Locate the COLMAP sparse model directory within a dataset."""
    candidates = [
        colmap_path / "sparse" / "0",
        colmap_path / "sparse",
        colmap_path / "undistorted" / "sparse" / "0",
        colmap_path / "undistorted" / "sparse",
    ]
    for candidate in candidates:
        if (candidate / "cameras.bin").exists() or (candidate / "cameras.txt").exists():
            return candidate
    return None


def _find_dataset_root(sparse_dir: Path) -> Path:
    """Derive the COLMAP dataset root from the sparse model directory.

    MILo expects the root that contains both ``sparse/`` and ``images/``.
    """
    # sparse/0 -> parent is sparse/ -> parent is dataset root
    if sparse_dir.name == "0":
        return sparse_dir.parent.parent
    return sparse_dir.parent


def run_milo(
    colmap_dir: str,
    output_dir: str,
    config: Optional[MiloConfig] = None,
) -> dict[str, Any]:
    """Run MILo training + mesh extraction on a COLMAP dataset.

    Args:
        colmap_dir: Path to COLMAP dataset (must contain ``sparse/0/`` and
            ``images/``).
        output_dir: Where MILo writes its output (checkpoints + meshes).
        config: MILo configuration. Uses defaults if None.

    Returns:
        Dict with keys:

        - ``success`` (bool): Whether the full pipeline completed.
        - ``mesh_path`` (str | None): Path to the extracted mesh PLY.
        - ``ply_path`` (str | None): Path to the trained gaussian PLY.
        - ``duration`` (float): Total wall-clock seconds.
        - ``error`` (str | None): Error message on failure.
    """
    cfg = config or MiloConfig()
    result: dict[str, Any] = {
        "success": False,
        "mesh_path": None,
        "ply_path": None,
        "duration": 0.0,
        "error": None,
    }

    colmap_path = Path(colmap_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate COLMAP dataset structure
    sparse_dir = _find_sparse_dir(colmap_path)
    if sparse_dir is None:
        result["error"] = f"No COLMAP sparse model found in {colmap_dir}"
        return result

    dataset_root = _find_dataset_root(sparse_dir)

    t_start = time.time()

    # -- Step 1: MILo training -----------------------------------------------
    train_cmd = [
        "conda", "run", "--no-capture-output", "-n", MILO_CONDA_ENV,
        "python", str(MILO_DIR / "train.py"),
        "-s", str(dataset_root),
        "-m", str(output_path),
        "--imp_metric", cfg.imp_metric,
        "--rasterizer", cfg.rasterizer,
        "--mesh_config", cfg.mesh_config,
        "--data_device", cfg.data_device,
    ]
    if cfg.dense_gaussians:
        train_cmd.append("--dense_gaussians")
    if cfg.decoupled_appearance:
        train_cmd.append("--decoupled_appearance")

    logger.info("MILo training: %s", " ".join(train_cmd))

    try:
        proc = subprocess.run(
            train_cmd,
            capture_output=True, text=True,
            timeout=cfg.train_timeout,
            cwd=str(MILO_DIR),
        )
        if proc.returncode != 0:
            result["error"] = f"MILo training failed (rc={proc.returncode}): {proc.stderr[-1000:]}"
            logger.error("MILo training stderr: %s", proc.stderr[-2000:])
            return result
    except subprocess.TimeoutExpired:
        result["error"] = f"MILo training timed out ({cfg.train_timeout}s)"
        return result

    train_duration = time.time() - t_start
    logger.info("MILo training completed in %.0fs", train_duration)

    # -- Step 2: Mesh extraction ---------------------------------------------
    extract_script = {
        "sdf": "mesh_extract_sdf.py",
        "integration": "mesh_extract_integration.py",
        "regular_tsdf": "mesh_extract_regular_tsdf.py",
    }.get(cfg.mesh_extract_method, "mesh_extract_sdf.py")

    extract_cmd = [
        "conda", "run", "--no-capture-output", "-n", MILO_CONDA_ENV,
        "python", str(MILO_DIR / extract_script),
        "-s", str(dataset_root),
        "-m", str(output_path),
        "--rasterizer", cfg.rasterizer,
    ]

    logger.info("MILo mesh extraction (%s): %s", extract_script, " ".join(extract_cmd))

    try:
        proc = subprocess.run(
            extract_cmd,
            capture_output=True, text=True,
            timeout=cfg.extract_timeout,
            cwd=str(MILO_DIR),
        )
        if proc.returncode != 0:
            result["error"] = f"MILo mesh extraction failed (rc={proc.returncode}): {proc.stderr[-1000:]}"
            logger.error("MILo extraction stderr: %s", proc.stderr[-2000:])
            return result
    except subprocess.TimeoutExpired:
        result["error"] = f"MILo mesh extraction timed out ({cfg.extract_timeout}s)"
        return result

    total_duration = time.time() - t_start

    # -- Locate output artifacts ---------------------------------------------
    # MILo writes mesh_*sdf*.ply for SDF method, mesh_*.ply for others
    mesh_candidates = sorted(
        output_path.glob("mesh_*sdf*.ply"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not mesh_candidates:
        mesh_candidates = sorted(
            output_path.glob("mesh_*.ply"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    if not mesh_candidates:
        result["error"] = "No mesh PLY found in MILo output directory"
        return result

    mesh_path = mesh_candidates[0]

    # Find trained gaussian PLY (point_cloud/<iteration>/point_cloud.ply)
    ply_candidates = sorted(output_path.rglob("point_cloud/*/point_cloud.ply"))
    ply_path = ply_candidates[-1] if ply_candidates else None

    result["success"] = True
    result["mesh_path"] = str(mesh_path)
    result["ply_path"] = str(ply_path) if ply_path else None
    result["duration"] = total_duration

    logger.info(
        "MILo complete: mesh=%s (%d bytes), duration=%.0fs",
        mesh_path.name, mesh_path.stat().st_size, total_duration,
    )
    return result


def load_milo_mesh(mesh_path: str) -> Any:
    """Load a MILo mesh PLY and return as a trimesh.Trimesh.

    Args:
        mesh_path: Path to the PLY mesh file produced by MILo.

    Returns:
        A ``trimesh.Trimesh`` instance.

    Raises:
        ImportError: If trimesh is not installed.
        ValueError: If the file cannot be loaded as a mesh.
    """
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    logger.info(
        "Loaded MILo mesh: %d vertices, %d faces",
        len(mesh.vertices), len(mesh.faces),
    )
    return mesh
