# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Multi-view renderer for Gaussian Splatting objects.

Renders N views from optimal camera positions around an object's
Gaussian representation. Output RGBA images are suitable for feeding
into Hunyuan3D 2.0 multi-view conditioning.

The renderer operates on PLY files containing 3DGS data (positions,
opacities, SH coefficients, covariances) and produces clean RGBA
images with transparent backgrounds.

Usage::

    from pipeline.multiview_renderer import MultiViewRenderer, RenderConfig

    renderer = MultiViewRenderer(RenderConfig(image_size=512, num_views=6))
    views = renderer.render("object.ply")
    # views is a list of ViewResult with .image (H,W,4 uint8) and .camera
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from plyfile import PlyData
    _HAS_PLYFILE = True
except ImportError:
    _HAS_PLYFILE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraParams:
    """Extrinsic + intrinsic parameters for a single viewpoint."""
    view_matrix: np.ndarray        # 4x4 world-to-camera
    projection_matrix: np.ndarray  # 4x4 projection
    position: np.ndarray           # 3-vector, world-space camera origin
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    label: str = ""

    @property
    def name(self) -> str:
        return self.label or f"az{self.azimuth_deg:.0f}_el{self.elevation_deg:.0f}"


@dataclass
class RenderConfig:
    """Multi-view rendering parameters."""
    image_size: int = 512
    num_views: int = 6
    fov_deg: float = 49.13  # ~equivalent to 35mm focal length
    camera_distance: float = 2.5
    near: float = 0.01
    far: float = 100.0
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sh_degree: int = 3
    elevation_angles: list[float] = field(default_factory=lambda: [0.0, 30.0])
    azimuth_preset: str = "uniform"  # "uniform", "canonical_6", "canonical_4"
    center_object: bool = True
    scale_to_unit: bool = True
    antialias: bool = True


@dataclass
class ViewResult:
    """Result of rendering a single view."""
    image: np.ndarray       # H x W x 4, uint8 RGBA
    depth: np.ndarray       # H x W, float32
    camera: CameraParams
    alpha_coverage: float   # fraction of pixels with alpha > 0


# ---------------------------------------------------------------------------
# Camera orbit generation
# ---------------------------------------------------------------------------

def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float64)


def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute 4x4 view matrix (world-to-camera) for a look-at camera."""
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    true_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(true_up, eye)
    view[2, 3] = np.dot(forward, eye)
    return view


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Compute 4x4 perspective projection matrix."""
    fov_rad = math.radians(fov_deg)
    f = 1.0 / math.tan(fov_rad / 2.0)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


def generate_orbit_cameras(
    config: RenderConfig,
    center: np.ndarray | None = None,
) -> list[CameraParams]:
    """Generate camera positions on a spherical orbit around the object.

    For Hunyuan3D multi-view conditioning, the canonical setup uses
    4 views: front (0), left (90/270), back (180), right (270/90).
    For higher quality, 6 or more views add 3/4 angles and top-down.

    Returns
    -------
    list[CameraParams]
        Camera parameters for each viewpoint.
    """
    if center is None:
        center = np.zeros(3, dtype=np.float64)

    proj = _perspective(config.fov_deg, 1.0, config.near, config.far)

    # Determine azimuth angles
    if config.azimuth_preset == "canonical_4":
        azimuths = [0.0, 90.0, 180.0, 270.0]
        elevations = [0.0, 0.0, 0.0, 0.0]
        labels = ["front", "left", "back", "right"]
    elif config.azimuth_preset == "canonical_6":
        azimuths = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
        elevations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        labels = [f"az{a:.0f}" for a in azimuths]
    elif config.azimuth_preset == "hunyuan_mv":
        # Hunyuan3D multi-view: front, left, back, right at 0 elevation
        # Plus optional elevated views for better coverage
        azimuths = [0.0, 90.0, 180.0, 270.0]
        elevations = [0.0, 0.0, 0.0, 0.0]
        labels = ["front", "left", "back", "right"]
        if config.num_views > 4:
            # Add 3/4-view angles
            extra_az = [45.0, 135.0, 225.0, 315.0]
            extra_el = [30.0, 30.0, 30.0, 30.0]
            extra_labels = ["front_left_up", "back_left_up",
                            "back_right_up", "front_right_up"]
            n_extra = min(config.num_views - 4, len(extra_az))
            azimuths.extend(extra_az[:n_extra])
            elevations.extend(extra_el[:n_extra])
            labels.extend(extra_labels[:n_extra])
        if config.num_views > 8:
            # Add top-down views
            azimuths.extend([0.0, 180.0])
            elevations.extend([80.0, 80.0])
            labels.extend(["top_front", "top_back"])
        if config.num_views > 10:
            # Add bottom views
            azimuths.extend([0.0, 180.0])
            elevations.extend([-30.0, -30.0])
            labels.extend(["bottom_front", "bottom_back"])
    else:  # "uniform"
        n = config.num_views
        azimuths = [i * 360.0 / n for i in range(n)]
        elevations = [config.elevation_angles[0]] * n
        labels = [f"view_{i}" for i in range(n)]

    cameras = []
    for az, el, label in zip(azimuths, elevations, labels):
        az_rad = math.radians(az)
        el_rad = math.radians(el)

        # Spherical to Cartesian
        d = config.camera_distance
        x = d * math.cos(el_rad) * math.sin(az_rad)
        y = d * math.sin(el_rad)
        z = d * math.cos(el_rad) * math.cos(az_rad)
        eye = center + np.array([x, y, z], dtype=np.float64)

        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        # Avoid degenerate up vector when looking straight down/up
        if abs(el) > 75:
            up = np.array([0.0, 0.0, -1.0 if el > 0 else 1.0], dtype=np.float64)

        view = _look_at(eye, center, up)

        cameras.append(CameraParams(
            view_matrix=view,
            projection_matrix=proj,
            position=eye,
            azimuth_deg=az,
            elevation_deg=el,
            label=label,
        ))

    return cameras


# ---------------------------------------------------------------------------
# Gaussian PLY loading
# ---------------------------------------------------------------------------

@dataclass
class GaussianData:
    """Parsed Gaussian splatting data from PLY."""
    positions: np.ndarray      # (N, 3) float32
    opacities: np.ndarray      # (N,) float32, sigmoid-activated
    scales: np.ndarray         # (N, 3) float32, log-space
    rotations: np.ndarray      # (N, 4) float32, quaternion wxyz
    sh_dc: np.ndarray          # (N, 3) float32, SH band 0
    sh_rest: np.ndarray        # (N, C, 3) float32, higher SH bands
    sh_degree: int

    @property
    def count(self) -> int:
        return len(self.positions)

    @property
    def bounds_min(self) -> np.ndarray:
        return self.positions.min(axis=0)

    @property
    def bounds_max(self) -> np.ndarray:
        return self.positions.max(axis=0)

    @property
    def center(self) -> np.ndarray:
        return (self.bounds_min + self.bounds_max) / 2.0

    @property
    def extent(self) -> float:
        return float(np.linalg.norm(self.bounds_max - self.bounds_min))


def load_gaussian_ply(path: str | Path) -> GaussianData:
    """Load a Gaussian splatting PLY file.

    Supports the standard 3DGS PLY format with properties:
    x, y, z, opacity, scale_0..2, rot_0..3, f_dc_0..2, f_rest_0..N
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {path}")

    if _HAS_PLYFILE:
        return _load_with_plyfile(path)
    return _load_ply_manual(path)


def _load_with_plyfile(path: Path) -> GaussianData:
    """Load using the plyfile library."""
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    n = len(vertex)

    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    # Opacity (stored as logit, needs sigmoid)
    raw_opacity = vertex["opacity"].astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-raw_opacity))

    # Scales (stored as log)
    scales = np.stack([
        vertex["scale_0"], vertex["scale_1"], vertex["scale_2"],
    ], axis=-1).astype(np.float32)

    # Rotations (quaternion wxyz)
    rotations = np.stack([
        vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"],
    ], axis=-1).astype(np.float32)
    # Normalize quaternions
    norms = np.linalg.norm(rotations, axis=-1, keepdims=True)
    rotations = rotations / (norms + 1e-12)

    # SH coefficients
    sh_dc = np.stack([
        vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"],
    ], axis=-1).astype(np.float32)

    # Detect SH degree from available properties
    sh_rest_names = sorted([
        p.name for p in vertex.properties if p.name.startswith("f_rest_")
    ])
    n_sh_rest = len(sh_rest_names)

    if n_sh_rest > 0:
        sh_rest_flat = np.stack(
            [vertex[name].astype(np.float32) for name in sh_rest_names],
            axis=-1,
        )
        # n_sh_rest should be (degree+1)^2 - 1 coefficients, times 3 colors
        n_coeffs = n_sh_rest // 3
        sh_rest = sh_rest_flat.reshape(n, n_coeffs, 3)
        sh_degree = int(math.sqrt(n_coeffs + 1)) - 1
    else:
        sh_rest = np.zeros((n, 0, 3), dtype=np.float32)
        sh_degree = 0

    return GaussianData(
        positions=positions,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        sh_degree=sh_degree,
    )


def _load_ply_manual(path: Path) -> GaussianData:
    """Fallback PLY loader without plyfile dependency.

    Parses the binary PLY header to find property offsets, then reads
    the vertex data block directly with numpy.
    """
    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_vertices = 0
        properties: list[tuple[str, str]] = []
        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[-1], parts[1]))

        # Build dtype
        type_map = {
            "float": np.float32,
            "double": np.float64,
            "uchar": np.uint8,
            "int": np.int32,
            "uint": np.uint32,
            "short": np.int16,
            "ushort": np.uint16,
        }
        dtype_list = [(name, type_map.get(dtype, np.float32)) for name, dtype in properties]
        dt = np.dtype(dtype_list)

        data = np.frombuffer(f.read(n_vertices * dt.itemsize), dtype=dt)

    positions = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)
    raw_opacity = data["opacity"].astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-raw_opacity))

    scales = np.stack([
        data["scale_0"], data["scale_1"], data["scale_2"],
    ], axis=-1).astype(np.float32)

    rotations = np.stack([
        data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"],
    ], axis=-1).astype(np.float32)
    norms = np.linalg.norm(rotations, axis=-1, keepdims=True)
    rotations = rotations / (norms + 1e-12)

    sh_dc = np.stack([
        data["f_dc_0"], data["f_dc_1"], data["f_dc_2"],
    ], axis=-1).astype(np.float32)

    sh_names = sorted([name for name, _ in dtype_list if name.startswith("f_rest_")])
    if sh_names:
        sh_rest_flat = np.stack([data[name].astype(np.float32) for name in sh_names], axis=-1)
        n_coeffs = len(sh_names) // 3
        sh_rest = sh_rest_flat.reshape(n_vertices, n_coeffs, 3)
        sh_degree = int(math.sqrt(n_coeffs + 1)) - 1
    else:
        sh_rest = np.zeros((n_vertices, 0, 3), dtype=np.float32)
        sh_degree = 0

    return GaussianData(
        positions=positions,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        sh_degree=sh_degree,
    )


# ---------------------------------------------------------------------------
# Software Gaussian rasterizer (CPU, numpy-based)
# ---------------------------------------------------------------------------

def _sh_eval_band0(dc: np.ndarray) -> np.ndarray:
    """Evaluate SH band 0: constant term. dc shape: (N,3)."""
    C0 = 0.28209479177387814
    return dc * C0 + 0.5


def _sh_eval_direction(
    dc: np.ndarray,
    sh_rest: np.ndarray,
    direction: np.ndarray,
    sh_degree: int,
) -> np.ndarray:
    """Evaluate spherical harmonics for given view direction.

    Parameters
    ----------
    dc : (N, 3) SH band 0 coefficients
    sh_rest : (N, C, 3) higher-order SH coefficients
    direction : (N, 3) normalized view directions
    sh_degree : max SH degree to evaluate

    Returns
    -------
    (N, 3) RGB colors, clamped to [0, 1]
    """
    C0 = 0.28209479177387814
    result = dc * C0

    if sh_degree < 1 or sh_rest.shape[1] == 0:
        return np.clip(result + 0.5, 0.0, 1.0)

    x = direction[:, 0:1]
    y = direction[:, 1:2]
    z = direction[:, 2:3]

    C1 = 0.4886025119029199
    result = result - C1 * y * sh_rest[:, 0] + C1 * z * sh_rest[:, 1] - C1 * x * sh_rest[:, 2]

    if sh_degree >= 2 and sh_rest.shape[1] >= 8:
        C2_0 = 1.0925484305920792
        C2_1 = -1.0925484305920792
        C2_2 = 0.31539156525252005
        C2_3 = -1.0925484305920792
        C2_4 = 0.5462742152960396

        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (result
                  + C2_0 * xy * sh_rest[:, 3]
                  + C2_1 * yz * sh_rest[:, 4]
                  + C2_2 * (2.0 * zz - xx - yy) * sh_rest[:, 5]
                  + C2_3 * xz * sh_rest[:, 6]
                  + C2_4 * (xx - yy) * sh_rest[:, 7])

    if sh_degree >= 3 and sh_rest.shape[1] >= 15:
        C3_0 = -0.5900435899266435
        C3_1 = 2.890611442640554
        C3_2 = -0.4570457994644658
        C3_3 = 0.3731763325901154
        C3_4 = -0.4570457994644658
        C3_5 = 1.445305721320277
        C3_6 = -0.5900435899266435

        result = (result
                  + C3_0 * y * (3.0 * xx - yy) * sh_rest[:, 8]
                  + C3_1 * xy * z * sh_rest[:, 9]
                  + C3_2 * y * (4.0 * zz - xx - yy) * sh_rest[:, 10]
                  + C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_rest[:, 11]
                  + C3_4 * x * (4.0 * zz - xx - yy) * sh_rest[:, 12]
                  + C3_5 * z * (xx - yy) * sh_rest[:, 13]
                  + C3_6 * x * (xx - 3.0 * yy) * sh_rest[:, 14])

    return np.clip(result + 0.5, 0.0, 1.0)


def _quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix.

    q shape: (N, 4) -> output (N, 3, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((len(q), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _compute_2d_covariance(
    positions_cam: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    focal_x: float,
    focal_y: float,
) -> np.ndarray:
    """Compute 2D covariance matrices for splatting.

    Projects 3D Gaussian covariances to screen space via the
    Jacobian of the perspective projection.

    Returns
    -------
    (N, 2, 2) screen-space covariance matrices
    """
    # 3D covariance: R @ S @ S^T @ R^T
    R = _quaternion_to_rotation(rotations)
    S = np.exp(scales)  # (N, 3) from log-space

    # M = R @ diag(S)
    M = R * S[:, np.newaxis, :]  # (N, 3, 3)
    cov3d = np.einsum("nij,nkj->nik", M, M)  # (N, 3, 3)

    tx = positions_cam[:, 0]
    ty = positions_cam[:, 1]
    tz = positions_cam[:, 2]
    tz2 = tz * tz

    # Jacobian of perspective projection
    J = np.zeros((len(positions_cam), 2, 3), dtype=np.float32)
    J[:, 0, 0] = focal_x / tz
    J[:, 0, 2] = -focal_x * tx / tz2
    J[:, 1, 1] = focal_y / tz
    J[:, 1, 2] = -focal_y * ty / tz2

    # 2D covariance: J @ cov3d @ J^T
    JC = np.einsum("nij,njk->nik", J, cov3d)
    cov2d = np.einsum("nij,nkj->nik", JC, J)

    # Add small regularization for numerical stability
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    return cov2d


def render_gaussians(
    gaussians: GaussianData,
    camera: CameraParams,
    config: RenderConfig,
) -> ViewResult:
    """Render Gaussian splats from a single viewpoint.

    Uses a CPU-based tile rasterizer with front-to-back alpha compositing.
    For production use, replace with a CUDA rasterizer (e.g. diff-gaussian-rasterization).

    Parameters
    ----------
    gaussians : GaussianData
        Loaded Gaussian splatting data.
    camera : CameraParams
        Camera extrinsics and intrinsics.
    config : RenderConfig
        Rendering configuration.

    Returns
    -------
    ViewResult
        RGBA image, depth map, and camera parameters.
    """
    W = H = config.image_size
    fov_rad = math.radians(config.fov_deg)
    focal = (W / 2.0) / math.tan(fov_rad / 2.0)

    # Transform positions to camera space
    pos_h = np.hstack([
        gaussians.positions,
        np.ones((gaussians.count, 1), dtype=np.float32),
    ])
    view = camera.view_matrix.astype(np.float32)
    pos_cam = (view @ pos_h.T).T[:, :3]  # (N, 3)

    # Cull behind camera
    valid = pos_cam[:, 2] > config.near
    if valid.sum() == 0:
        empty_img = np.zeros((H, W, 4), dtype=np.uint8)
        empty_depth = np.full((H, W), config.far, dtype=np.float32)
        return ViewResult(
            image=empty_img, depth=empty_depth,
            camera=camera, alpha_coverage=0.0,
        )

    idx = np.where(valid)[0]
    pos_cam = pos_cam[idx]
    opacities = gaussians.opacities[idx]
    scales = gaussians.scales[idx]
    rotations = gaussians.rotations[idx]
    sh_dc = gaussians.sh_dc[idx]
    sh_rest = gaussians.sh_rest[idx]

    # Project to screen
    px = focal * pos_cam[:, 0] / pos_cam[:, 2] + W / 2.0
    py = focal * pos_cam[:, 1] / pos_cam[:, 2] + H / 2.0

    # Evaluate SH for view-dependent color
    view_dirs = -pos_cam.copy()
    view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=-1, keepdims=True) + 1e-12)
    colors = _sh_eval_direction(sh_dc, sh_rest, view_dirs, min(config.sh_degree, gaussians.sh_degree))

    # Compute 2D covariances
    cov2d = _compute_2d_covariance(pos_cam, scales, rotations, focal, focal)

    # Sort by depth (front to back)
    depth_order = np.argsort(pos_cam[:, 2])

    # Alpha-composite (front to back)
    image = np.zeros((H, W, 3), dtype=np.float64)
    alpha_acc = np.zeros((H, W), dtype=np.float64)
    depth_map = np.full((H, W), config.far, dtype=np.float32)

    # For efficiency, use a splat radius heuristic
    for i in depth_order:
        if alpha_acc.max() > 0.999:
            break

        cx_f, cy_f = float(px[i]), float(py[i])
        opacity = float(opacities[i])
        color = colors[i]
        z_val = float(pos_cam[i, 2])

        # Eigendecomposition of 2D covariance for splat extent
        cov = cov2d[i]
        det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
        if det <= 0:
            continue
        inv_det = 1.0 / det
        cov_inv = np.array([
            [cov[1, 1] * inv_det, -cov[0, 1] * inv_det],
            [-cov[1, 0] * inv_det, cov[0, 0] * inv_det],
        ])

        # Splat radius: 3-sigma extent
        trace = cov[0, 0] + cov[1, 1]
        disc = max(0.0, trace * trace / 4.0 - det)
        eig_max = trace / 2.0 + math.sqrt(disc)
        radius = int(math.ceil(3.0 * math.sqrt(max(eig_max, 0.01))))
        radius = min(radius, 128)  # cap for performance

        x_min = max(0, int(cx_f) - radius)
        x_max = min(W, int(cx_f) + radius + 1)
        y_min = max(0, int(cy_f) - radius)
        y_max = min(H, int(cy_f) + radius + 1)

        if x_min >= x_max or y_min >= y_max:
            continue

        # Create pixel grid
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        dx = xx.astype(np.float64) - cx_f
        dy = yy.astype(np.float64) - cy_f

        # Gaussian evaluation: exp(-0.5 * [dx,dy] @ cov_inv @ [dx,dy]^T)
        power = -0.5 * (
            cov_inv[0, 0] * dx * dx
            + (cov_inv[0, 1] + cov_inv[1, 0]) * dx * dy
            + cov_inv[1, 1] * dy * dy
        )
        gaussian = np.exp(np.clip(power, -20.0, 0.0))
        alpha = opacity * gaussian

        # Front-to-back compositing
        remaining = 1.0 - alpha_acc[y_min:y_max, x_min:x_max]
        contrib = alpha * remaining

        for c in range(3):
            image[y_min:y_max, x_min:x_max, c] += contrib * float(color[c])
        alpha_acc[y_min:y_max, x_min:x_max] += contrib

        # Update depth (first-hit)
        depth_mask = (contrib > 0.01) & (depth_map[y_min:y_max, x_min:x_max] >= config.far)
        depth_map[y_min:y_max, x_min:x_max][depth_mask] = z_val

    # Convert to RGBA uint8
    rgb = np.clip(image * 255, 0, 255).astype(np.uint8)
    alpha_u8 = np.clip(alpha_acc * 255, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha_u8])

    coverage = float(np.mean(alpha_acc > 0.01))

    return ViewResult(
        image=rgba,
        depth=depth_map,
        camera=camera,
        alpha_coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Multi-view renderer (main entry point)
# ---------------------------------------------------------------------------

class MultiViewRenderer:
    """Renders multiple views of a Gaussian splat object for Hunyuan3D input.

    Parameters
    ----------
    config : RenderConfig
        Rendering and camera configuration.
    """

    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()

    def render(
        self,
        ply_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> list[ViewResult]:
        """Render multi-view images from a Gaussian PLY file.

        Parameters
        ----------
        ply_path : str | Path
            Path to the 3DGS PLY file.
        output_dir : str | Path | None
            If provided, save rendered images to this directory.

        Returns
        -------
        list[ViewResult]
            Rendered views with RGBA images and camera parameters.
        """
        ply_path = Path(ply_path)
        logger.info("Loading Gaussian PLY: %s", ply_path)
        gaussians = load_gaussian_ply(ply_path)
        logger.info(
            "Loaded %d Gaussians, SH degree %d, bounds: %s to %s",
            gaussians.count, gaussians.sh_degree,
            gaussians.bounds_min, gaussians.bounds_max,
        )

        # Optionally center and scale
        if self.config.center_object:
            center = gaussians.center.copy()
            gaussians.positions = gaussians.positions - center
            logger.info("Centered object (shifted by %s)", center)

        if self.config.scale_to_unit:
            extent = gaussians.extent
            if extent > 0:
                scale_factor = 2.0 / extent
                gaussians.positions *= scale_factor
                gaussians.scales += math.log(scale_factor)
                logger.info("Scaled to unit sphere (factor %.4f)", scale_factor)

        # Generate camera orbit
        cameras = generate_orbit_cameras(self.config, center=np.zeros(3))
        logger.info("Generated %d camera positions", len(cameras))

        # Render each view
        views: list[ViewResult] = []
        for i, cam in enumerate(cameras):
            logger.info("Rendering view %d/%d: %s", i + 1, len(cameras), cam.name)
            view = render_gaussians(gaussians, cam, self.config)
            views.append(view)
            logger.info(
                "  View %s: alpha coverage %.1f%%",
                cam.name, view.alpha_coverage * 100,
            )

        # Save if output directory provided
        if output_dir is not None:
            self._save_views(views, Path(output_dir))

        return views

    def render_from_data(
        self,
        gaussians: GaussianData,
        output_dir: str | Path | None = None,
    ) -> list[ViewResult]:
        """Render multi-view images from pre-loaded Gaussian data.

        Parameters
        ----------
        gaussians : GaussianData
            Pre-loaded Gaussian splatting data.
        output_dir : str | Path | None
            If provided, save rendered images to this directory.

        Returns
        -------
        list[ViewResult]
        """
        if self.config.center_object:
            center = gaussians.center.copy()
            gaussians.positions = gaussians.positions - center

        if self.config.scale_to_unit:
            extent = gaussians.extent
            if extent > 0:
                scale_factor = 2.0 / extent
                gaussians.positions *= scale_factor
                gaussians.scales += math.log(scale_factor)

        cameras = generate_orbit_cameras(self.config, center=np.zeros(3))
        views = []
        for cam in cameras:
            views.append(render_gaussians(gaussians, cam, self.config))

        if output_dir is not None:
            self._save_views(views, Path(output_dir))

        return views

    def _save_views(self, views: list[ViewResult], output_dir: Path) -> list[Path]:
        """Save rendered views as PNG files."""
        if not _HAS_PIL:
            logger.warning("Pillow not available, cannot save images")
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for view in views:
            name = view.camera.name
            path = output_dir / f"{name}.png"
            img = PILImage.fromarray(view.image, mode="RGBA")
            img.save(path)
            saved.append(path)
            logger.info("Saved %s (coverage %.1f%%)", path, view.alpha_coverage * 100)

        return saved

    @staticmethod
    def views_to_pil(views: list[ViewResult]) -> list:
        """Convert ViewResult list to PIL Image list."""
        if not _HAS_PIL:
            raise ImportError("Pillow required for PIL conversion")
        return [PILImage.fromarray(v.image, mode="RGBA") for v in views]
