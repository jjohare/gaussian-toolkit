"""TSDF-based mesh extraction from Gaussian splat objects.

Renders depth maps from multiple viewpoints via LichtFeld MCP,
fuses them into a TSDF volume, and extracts a polygonal mesh
using marching cubes.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from skimage.measure import marching_cubes
import trimesh

from .mesh_cleaner import MeshCleaner

logger = logging.getLogger(__name__)

LFS_MCP = "/usr/local/bin/lfs-mcp"


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def default(cls, width: int = 512, height: int = 512, fov_deg: float = 60.0) -> CameraIntrinsics:
        fov_rad = np.radians(fov_deg)
        f = (width / 2.0) / np.tan(fov_rad / 2.0)
        return cls(width=width, height=height, fx=f, fy=f, cx=width / 2.0, cy=height / 2.0)


@dataclass
class TSDFConfig:
    """Configuration for TSDF volume and mesh extraction."""
    voxel_size: float = 0.005
    sdf_trunc: float = 0.04
    volume_bounds_min: np.ndarray = field(default_factory=lambda: np.array([-1.0, -1.0, -1.0]))
    volume_bounds_max: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    num_viewpoints: int = 36
    render_width: int = 512
    render_height: int = 512
    camera_distance: float = 2.5
    camera_fov_deg: float = 60.0
    target_faces: int = 50000
    smooth_iterations: int = 3
    min_component_ratio: float = 0.01
    mcp_endpoint: str = "http://127.0.0.1:45677/mcp"


class TSDFVolume:
    """Truncated Signed Distance Function volume for depth fusion.

    Implements volumetric TSDF integration and marching-cubes extraction
    without requiring Open3D, using raw numpy arrays.
    """

    def __init__(self, config: TSDFConfig):
        self.config = config
        self.voxel_size = config.voxel_size
        self.sdf_trunc = config.sdf_trunc

        bounds_min = config.volume_bounds_min
        bounds_max = config.volume_bounds_max

        self.origin = bounds_min.copy()
        dims = ((bounds_max - bounds_min) / self.voxel_size).astype(int)
        self.nx, self.ny, self.nz = int(dims[0]), int(dims[1]), int(dims[2])

        logger.info("TSDF volume: %d x %d x %d voxels (%.1fM)",
                     self.nx, self.ny, self.nz,
                     self.nx * self.ny * self.nz / 1e6)

        self.tsdf = np.ones((self.nx, self.ny, self.nz), dtype=np.float32)
        self.weight = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.color = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float32)

    def _voxel_centers(self) -> np.ndarray:
        """Compute world coordinates of all voxel centers."""
        x = np.arange(self.nx) * self.voxel_size + self.origin[0] + self.voxel_size / 2
        y = np.arange(self.ny) * self.voxel_size + self.origin[1] + self.voxel_size / 2
        z = np.arange(self.nz) * self.voxel_size + self.origin[2] + self.voxel_size / 2
        return x, y, z

    def integrate(
        self,
        depth_map: np.ndarray,
        color_image: np.ndarray,
        intrinsics: CameraIntrinsics,
        extrinsics: np.ndarray,
        weight: float = 1.0,
    ) -> None:
        """Integrate a single depth frame into the TSDF volume.

        Args:
            depth_map: HxW float array of depth values in meters.
            color_image: HxWx3 uint8 or float array of color values.
            intrinsics: Camera intrinsic parameters.
            extrinsics: 4x4 camera-to-world transform (camera pose).
            weight: Integration weight for this frame.
        """
        h, w = depth_map.shape[:2]
        world_to_cam = np.linalg.inv(extrinsics)

        x_coords, y_coords, z_coords = self._voxel_centers()

        color_f = color_image.astype(np.float32)
        if color_f.max() > 1.0:
            color_f /= 255.0

        # Process in slabs along x to manage memory
        slab_size = max(1, min(self.nx, 64))
        for x_start in range(0, self.nx, slab_size):
            x_end = min(x_start + slab_size, self.nx)
            nx_slab = x_end - x_start

            # Build world-space coordinates for this slab
            xx, yy, zz = np.meshgrid(
                x_coords[x_start:x_end], y_coords, z_coords, indexing="ij"
            )
            pts_world = np.stack([xx.ravel(), yy.ravel(), zz.ravel(), np.ones(nx_slab * self.ny * self.nz)], axis=0)

            # Transform to camera coordinates
            pts_cam = world_to_cam @ pts_world
            cam_x = pts_cam[0]
            cam_y = pts_cam[1]
            cam_z = pts_cam[2]

            # Skip voxels behind camera
            valid = cam_z > 0
            if not np.any(valid):
                continue

            # Project to image plane
            u = (intrinsics.fx * cam_x / cam_z + intrinsics.cx).astype(np.float64)
            v = (intrinsics.fy * cam_y / cam_z + intrinsics.cy).astype(np.float64)

            # Bounds check
            valid &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
            if not np.any(valid):
                continue

            u_int = np.clip(u.astype(int), 0, w - 1)
            v_int = np.clip(v.astype(int), 0, h - 1)

            # Look up depth at projected pixel
            depth_vals = depth_map[v_int, u_int]
            valid &= depth_vals > 0

            # Compute SDF
            sdf = depth_vals - cam_z
            valid &= sdf >= -self.sdf_trunc
            tsdf_val = np.clip(sdf / self.sdf_trunc, -1.0, 1.0)

            # Reshape back to slab dimensions
            tsdf_slab = tsdf_val.reshape(nx_slab, self.ny, self.nz)
            valid_slab = valid.reshape(nx_slab, self.ny, self.nz)

            # Look up colors
            col_r = color_f[v_int, u_int, 0].reshape(nx_slab, self.ny, self.nz)
            col_g = color_f[v_int, u_int, 1].reshape(nx_slab, self.ny, self.nz)
            col_b = color_f[v_int, u_int, 2].reshape(nx_slab, self.ny, self.nz)

            # Weighted running average update
            old_w = self.weight[x_start:x_end]
            new_w = old_w + weight * valid_slab.astype(np.float32)
            mask = valid_slab & (new_w > 0)

            old_tsdf = self.tsdf[x_start:x_end]
            self.tsdf[x_start:x_end] = np.where(
                mask,
                (old_w * old_tsdf + weight * tsdf_slab) / np.maximum(new_w, 1e-8),
                old_tsdf,
            )
            for c_idx, col_ch in enumerate([col_r, col_g, col_b]):
                old_c = self.color[x_start:x_end, :, :, c_idx]
                self.color[x_start:x_end, :, :, c_idx] = np.where(
                    mask,
                    (old_w * old_c + weight * col_ch) / np.maximum(new_w, 1e-8),
                    old_c,
                )
            self.weight[x_start:x_end] = new_w

    def extract_mesh(self) -> trimesh.Trimesh:
        """Extract a triangle mesh from the TSDF using marching cubes.

        Returns:
            A trimesh.Trimesh with vertex colors.
        """
        observed = self.weight > 0
        if not np.any(observed):
            raise ValueError("TSDF volume is empty -- no depth frames were integrated")

        tsdf_vol = self.tsdf.copy()
        tsdf_vol[~observed] = 1.0  # mark unobserved as outside

        try:
            verts, faces, normals, _ = marching_cubes(
                tsdf_vol,
                level=0.0,
                spacing=(self.voxel_size, self.voxel_size, self.voxel_size),
            )
        except ValueError as e:
            raise ValueError(f"Marching cubes failed (surface may not cross zero): {e}") from e

        # Shift vertices to world coordinates
        verts += self.origin

        # Interpolate vertex colors from the volume
        voxel_indices = ((verts - self.origin) / self.voxel_size).astype(int)
        voxel_indices = np.clip(
            voxel_indices,
            [0, 0, 0],
            [self.nx - 1, self.ny - 1, self.nz - 1],
        )
        vertex_colors = self.color[
            voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
        ]
        vertex_colors = np.clip(vertex_colors * 255, 0, 255).astype(np.uint8)
        alpha = np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
        vertex_colors = np.hstack([vertex_colors, alpha])

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
            vertex_colors=vertex_colors,
            process=False,
        )
        logger.info("Extracted mesh: %d vertices, %d faces", len(verts), len(faces))
        return mesh


def _generate_orbit_cameras(
    n_views: int,
    distance: float,
    fov_deg: float,
    width: int,
    height: int,
    elevation_angles: Optional[list[float]] = None,
) -> list[tuple[np.ndarray, CameraIntrinsics]]:
    """Generate camera poses orbiting around the origin.

    Returns a list of (extrinsics_4x4, intrinsics) pairs.
    Cameras look at the origin from a sphere of given radius.
    """
    if elevation_angles is None:
        elevation_angles = [-20.0, 0.0, 30.0]

    intrinsics = CameraIntrinsics.default(width, height, fov_deg)
    cameras = []

    views_per_ring = max(1, n_views // len(elevation_angles))
    for elev_deg in elevation_angles:
        elev = np.radians(elev_deg)
        for i in range(views_per_ring):
            azimuth = 2.0 * np.pi * i / views_per_ring

            # Camera position on sphere
            cx = distance * np.cos(elev) * np.cos(azimuth)
            cy = distance * np.cos(elev) * np.sin(azimuth)
            cz = distance * np.sin(elev)
            pos = np.array([cx, cy, cz])

            # Look-at matrix (camera looks at origin, up = +Z)
            forward = -pos / np.linalg.norm(pos)
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
            if np.linalg.norm(right) < 1e-6:
                world_up = np.array([0.0, 1.0, 0.0])
                right = np.cross(forward, world_up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            up /= np.linalg.norm(up)

            extrinsics = np.eye(4, dtype=np.float64)
            extrinsics[:3, 0] = right
            extrinsics[:3, 1] = up
            extrinsics[:3, 2] = forward  # +z = viewing direction
            extrinsics[:3, 3] = pos

            cameras.append((extrinsics, intrinsics))

    return cameras


def _call_mcp_render(
    width: int,
    height: int,
    camera_pose: np.ndarray,
    render_type: str = "color",
    mcp_endpoint: str = "http://127.0.0.1:45677/mcp",
) -> Optional[np.ndarray]:
    """Call LichtFeld MCP render.capture to get an image.

    Args:
        width: Image width.
        height: Image height.
        camera_pose: 4x4 camera-to-world matrix.
        render_type: "color" or "depth".
        mcp_endpoint: MCP endpoint URL.

    Returns:
        numpy array of the rendered image, or None on failure.
    """
    # Flatten camera pose for JSON transport
    pose_list = camera_pose.flatten().tolist()
    args = {
        "width": width,
        "height": height,
        "camera_pose": pose_list,
        "render_type": render_type,
        "output_format": "numpy",
    }
    args_json = json.dumps(args)

    try:
        result = subprocess.run(
            [LFS_MCP, "call", "render.capture", args_json],
            capture_output=True, text=True, timeout=30,
            env={"LICHTFELD_MCP_ENDPOINT": mcp_endpoint, "PATH": "/usr/bin:/usr/local/bin"},
        )
        if result.returncode != 0:
            logger.warning("MCP render.capture failed: %s", result.stderr)
            return None

        response = json.loads(result.stdout)
        content = response.get("result", {}).get("content", [])
        for item in content:
            if item.get("type") == "resource":
                uri = item.get("resource", {}).get("uri", "")
                if uri.startswith("file://"):
                    path = uri[7:]
                    return np.load(path) if path.endswith(".npy") else None
            if item.get("type") == "text":
                data = json.loads(item["text"])
                if "data" in data:
                    arr = np.array(data["data"], dtype=np.float32)
                    if render_type == "depth":
                        return arr.reshape(height, width)
                    return arr.reshape(height, width, -1)
        logger.warning("No parseable image data in MCP response")
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        logger.warning("MCP render call failed: %s", e)
        return None


def _render_depth_from_gaussians(
    camera_pose: np.ndarray,
    intrinsics: CameraIntrinsics,
    mcp_endpoint: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Render depth and color from LichtFeld MCP.

    Returns (depth_map, color_image) or (None, None) on failure.
    """
    depth = _call_mcp_render(
        intrinsics.width, intrinsics.height, camera_pose,
        render_type="depth", mcp_endpoint=mcp_endpoint,
    )
    color = _call_mcp_render(
        intrinsics.width, intrinsics.height, camera_pose,
        render_type="color", mcp_endpoint=mcp_endpoint,
    )
    return depth, color


class MeshExtractor:
    """Extract polygonal meshes from Gaussian splat objects.

    Pipeline:
    1. Render depth + color from N viewpoints via LichtFeld MCP
    2. Fuse into TSDF volume
    3. Extract mesh via marching cubes
    4. Clean and decimate

    Can also operate on pre-rendered depth/color numpy arrays
    when MCP is unavailable (for testing or offline use).
    """

    def __init__(self, config: Optional[TSDFConfig] = None):
        self.config = config or TSDFConfig()

    def extract_from_mcp(self) -> trimesh.Trimesh:
        """Full pipeline: render from MCP, fuse, extract, clean.

        Requires LichtFeld Studio MCP to be running with a loaded scene.
        """
        cfg = self.config
        cameras = _generate_orbit_cameras(
            n_views=cfg.num_viewpoints,
            distance=cfg.camera_distance,
            fov_deg=cfg.camera_fov_deg,
            width=cfg.render_width,
            height=cfg.render_height,
        )

        tsdf = TSDFVolume(cfg)
        integrated_count = 0

        for i, (extrinsics, intrinsics) in enumerate(cameras):
            logger.info("Rendering viewpoint %d/%d", i + 1, len(cameras))
            depth, color = _render_depth_from_gaussians(
                extrinsics, intrinsics, cfg.mcp_endpoint,
            )
            if depth is None or color is None:
                logger.warning("Skipping viewpoint %d (render failed)", i + 1)
                continue

            tsdf.integrate(depth, color, intrinsics, extrinsics)
            integrated_count += 1

        if integrated_count == 0:
            raise RuntimeError("No depth frames could be rendered from MCP")

        logger.info("Integrated %d/%d viewpoints", integrated_count, len(cameras))
        mesh = tsdf.extract_mesh()

        cleaner = MeshCleaner()
        mesh = cleaner.clean(
            mesh,
            target_faces=cfg.target_faces,
            smooth_iterations=cfg.smooth_iterations,
            min_component_ratio=cfg.min_component_ratio,
        )
        return mesh

    def extract_from_arrays(
        self,
        depth_maps: list[np.ndarray],
        color_images: list[np.ndarray],
        extrinsics_list: list[np.ndarray],
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> trimesh.Trimesh:
        """Extract mesh from pre-rendered depth/color arrays.

        Args:
            depth_maps: List of HxW float depth arrays.
            color_images: List of HxWx3 color arrays (uint8 or float).
            extrinsics_list: List of 4x4 camera-to-world matrices.
            intrinsics: Camera intrinsics (uses defaults if None).

        Returns:
            Cleaned trimesh.Trimesh.
        """
        if not (len(depth_maps) == len(color_images) == len(extrinsics_list)):
            raise ValueError("depth_maps, color_images, and extrinsics_list must have equal length")

        cfg = self.config
        if intrinsics is None:
            h, w = depth_maps[0].shape[:2]
            intrinsics = CameraIntrinsics.default(w, h, cfg.camera_fov_deg)

        tsdf = TSDFVolume(cfg)

        for i, (depth, color, ext) in enumerate(zip(depth_maps, color_images, extrinsics_list)):
            logger.info("Integrating frame %d/%d", i + 1, len(depth_maps))
            tsdf.integrate(depth, color, intrinsics, ext)

        mesh = tsdf.extract_mesh()

        cleaner = MeshCleaner()
        mesh = cleaner.clean(
            mesh,
            target_faces=cfg.target_faces,
            smooth_iterations=cfg.smooth_iterations,
            min_component_ratio=cfg.min_component_ratio,
        )
        return mesh

    def extract_from_pointcloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        target_faces: Optional[int] = None,
    ) -> trimesh.Trimesh:
        """Extract mesh from a point cloud using ball-pivoting or Poisson reconstruction.

        Falls back to convex hull if advanced reconstruction is unavailable.

        Args:
            points: Nx3 array of point positions.
            colors: Nx3 array of point colors (0-255 uint8 or 0-1 float).
            target_faces: Target face count for decimation.

        Returns:
            trimesh.Trimesh
        """
        target = target_faces or self.config.target_faces

        # Estimate normals via PCA on local neighborhoods
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        k = min(30, len(points))
        _, indices = tree.query(points, k=k)

        normals = np.zeros_like(points)
        for i in range(len(points)):
            neighbors = points[indices[i]]
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]  # smallest eigenvalue = normal direction

        # Orient normals outward (toward centroid-opposite direction)
        centroid = points.mean(axis=0)
        for i in range(len(normals)):
            if np.dot(normals[i], points[i] - centroid) < 0:
                normals[i] = -normals[i]

        # Use Poisson surface reconstruction via marching cubes on an implicit function
        # Build a voxel grid and compute the indicator function
        padding = 0.1
        bounds_min = points.min(axis=0) - padding
        bounds_max = points.max(axis=0) + padding

        cfg = TSDFConfig(
            voxel_size=self.config.voxel_size,
            volume_bounds_min=bounds_min,
            volume_bounds_max=bounds_max,
        )
        dims = ((bounds_max - bounds_min) / cfg.voxel_size).astype(int)
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])

        # Splat points into voxel grid as a density field
        grid = np.zeros((nx, ny, nz), dtype=np.float32)
        voxel_idx = ((points - bounds_min) / cfg.voxel_size).astype(int)
        voxel_idx = np.clip(voxel_idx, 0, [nx - 1, ny - 1, nz - 1])
        for idx in voxel_idx:
            grid[idx[0], idx[1], idx[2]] += 1.0

        # Smooth the density and extract isosurface
        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid, sigma=2.0)
        threshold = grid.max() * 0.1

        try:
            verts, faces, mesh_normals, _ = marching_cubes(
                grid, level=threshold,
                spacing=(cfg.voxel_size, cfg.voxel_size, cfg.voxel_size),
            )
            verts += bounds_min
        except ValueError:
            logger.warning("Marching cubes on point cloud failed, falling back to convex hull")
            cloud = trimesh.PointCloud(points)
            mesh = cloud.convex_hull
            cleaner = MeshCleaner()
            return cleaner.clean(mesh, target_faces=target)

        # Transfer colors via nearest-neighbor lookup
        vertex_colors = None
        if colors is not None:
            colors_f = colors.astype(np.float32)
            if colors_f.max() > 1.0:
                colors_f /= 255.0
            _, nn_idx = tree.query(verts, k=1)
            vc = colors_f[nn_idx]
            vc = np.clip(vc * 255, 0, 255).astype(np.uint8)
            alpha = np.full((len(vc), 1), 255, dtype=np.uint8)
            vertex_colors = np.hstack([vc, alpha])

        mesh = trimesh.Trimesh(
            vertices=verts, faces=faces, vertex_normals=mesh_normals,
            vertex_colors=vertex_colors, process=True,
        )

        cleaner = MeshCleaner()
        return cleaner.clean(mesh, target_faces=target)
