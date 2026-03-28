"""Texture baking from Gaussian splat renders onto UV-mapped meshes.

Projects rendered color images from multiple viewpoints onto a mesh's
UV atlas to produce a diffuse texture map.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import xatlas

logger = logging.getLogger(__name__)

LFS_MCP = "/usr/local/bin/lfs-mcp"


@dataclass
class BakeConfig:
    """Configuration for texture baking."""
    texture_size: int = 2048
    num_viewpoints: int = 36
    camera_distance: float = 2.5
    camera_fov_deg: float = 60.0
    render_width: int = 1024
    render_height: int = 1024
    padding_pixels: int = 4
    mcp_endpoint: str = "http://127.0.0.1:45677/mcp"
    background_color: tuple[int, int, int] = (128, 128, 128)


def _generate_uv_atlas_xatlas(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate UV atlas using xatlas.

    Args:
        mesh: Input mesh (must have vertices and faces).

    Returns:
        (new_vertices, new_faces, uv_coords) where uv_coords is Nx2 in [0,1].
    """
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)

    # xatlas may need normals
    normals = None
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
        normals = mesh.vertex_normals.astype(np.float32)

    logger.info("Running xatlas UV parameterization on %d verts, %d faces",
                 len(vertices), len(faces))

    vmapping, new_faces, uvs = xatlas.parametrize(
        vertices, faces, normals,
    )

    new_vertices = vertices[vmapping]
    new_normals = normals[vmapping] if normals is not None else None

    logger.info("xatlas: %d -> %d vertices, %d UV coords",
                 len(vertices), len(new_vertices), len(uvs))

    return new_vertices, new_faces, uvs, new_normals, vmapping


def _generate_uv_atlas_trimesh(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Fallback UV unwrapping using trimesh (angle-based flattening).

    Returns:
        (vertices, faces) with UV stored in mesh visual.
    """
    # trimesh can auto-unwrap when creating TextureVisuals
    # Use a simple planar projection as absolute fallback
    vertices = mesh.vertices
    bounds = mesh.bounds
    extent = bounds[1] - bounds[0]
    extent = np.maximum(extent, 1e-8)

    # Project UV from the two largest axes
    axis_order = np.argsort(-extent)
    u_axis, v_axis = axis_order[0], axis_order[1]

    uv = np.zeros((len(vertices), 2), dtype=np.float32)
    uv[:, 0] = (vertices[:, u_axis] - bounds[0, u_axis]) / extent[u_axis]
    uv[:, 1] = (vertices[:, v_axis] - bounds[0, v_axis]) / extent[v_axis]

    return uv


class TextureBaker:
    """Bake diffuse textures from Gaussian splat renders onto a UV-mapped mesh.

    Pipeline:
    1. Generate UV atlas via xatlas
    2. Render color images from multiple viewpoints
    3. For each texel, find visible surface point and project into renders
    4. Composite into a single diffuse texture atlas
    """

    def __init__(self, config: Optional[BakeConfig] = None):
        self.config = config or BakeConfig()

    def bake(
        self,
        mesh: trimesh.Trimesh,
        output_texture_path: Optional[str | Path] = None,
        color_images: Optional[list[np.ndarray]] = None,
        camera_poses: Optional[list[np.ndarray]] = None,
    ) -> tuple[trimesh.Trimesh, Path]:
        """Bake texture onto mesh.

        If color_images and camera_poses are provided, uses those directly.
        Otherwise renders from MCP.

        Args:
            mesh: Input mesh to texture.
            output_texture_path: Where to save the texture PNG.
            color_images: Pre-rendered color images (list of HxWx3 uint8).
            camera_poses: Camera-to-world 4x4 matrices for each image.

        Returns:
            (textured_mesh, texture_path)
        """
        cfg = self.config

        if output_texture_path is None:
            output_texture_path = Path("/tmp/lfs_diffuse_texture.png")
        else:
            output_texture_path = Path(output_texture_path)

        # Step 1: Generate UV atlas
        mesh_with_uv, uv_coords = self._unwrap_uvs(mesh)

        # Step 2: Get color images
        if color_images is None or camera_poses is None:
            color_images, camera_poses = self._render_views(cfg)

        if not color_images:
            logger.warning("No color images available, generating solid texture")
            texture = np.full(
                (cfg.texture_size, cfg.texture_size, 3),
                cfg.background_color, dtype=np.uint8,
            )
            self._save_texture(texture, output_texture_path)
            textured_mesh = self._apply_texture(mesh_with_uv, uv_coords, output_texture_path)
            return textured_mesh, output_texture_path

        # Step 3: Bake by projecting renders onto UV space
        texture = self._project_to_texture(
            mesh_with_uv, uv_coords, color_images, camera_poses,
        )

        # Step 4: Dilate to fill seam gaps
        texture = self._dilate_texture(texture, cfg.padding_pixels)

        # Step 5: Save and apply
        self._save_texture(texture, output_texture_path)
        textured_mesh = self._apply_texture(mesh_with_uv, uv_coords, output_texture_path)

        logger.info("Texture baked: %dx%d -> %s",
                     cfg.texture_size, cfg.texture_size, output_texture_path)
        return textured_mesh, output_texture_path

    def bake_from_vertex_colors(
        self,
        mesh: trimesh.Trimesh,
        output_texture_path: Optional[str | Path] = None,
    ) -> tuple[trimesh.Trimesh, Path]:
        """Bake vertex colors into a UV texture atlas.

        Useful when the mesh already has per-vertex color from TSDF extraction
        and we want a proper texture map instead.

        Args:
            mesh: Mesh with vertex colors.
            output_texture_path: Where to save the texture.

        Returns:
            (textured_mesh, texture_path)
        """
        cfg = self.config

        if output_texture_path is None:
            output_texture_path = Path("/tmp/lfs_diffuse_texture.png")
        else:
            output_texture_path = Path(output_texture_path)

        mesh_with_uv, uv_coords = self._unwrap_uvs(mesh)

        # Rasterize vertex colors into texture space
        texture = self._rasterize_vertex_colors(
            mesh_with_uv, uv_coords, cfg.texture_size,
        )
        texture = self._dilate_texture(texture, cfg.padding_pixels)

        self._save_texture(texture, output_texture_path)
        textured_mesh = self._apply_texture(mesh_with_uv, uv_coords, output_texture_path)

        return textured_mesh, output_texture_path

    def _unwrap_uvs(
        self, mesh: trimesh.Trimesh,
    ) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Generate UV coordinates for the mesh.

        Returns:
            (mesh_with_new_topology, uv_coords_Nx2)
        """
        try:
            new_verts, new_faces, uvs, new_normals, vmapping = _generate_uv_atlas_xatlas(mesh)

            # Transfer vertex colors if present
            vertex_colors = None
            if mesh.visual and hasattr(mesh.visual, 'vertex_colors'):
                old_colors = mesh.visual.vertex_colors
                if old_colors is not None and len(old_colors) == len(mesh.vertices):
                    vertex_colors = old_colors[vmapping]

            new_mesh = trimesh.Trimesh(
                vertices=new_verts,
                faces=new_faces,
                vertex_normals=new_normals,
                vertex_colors=vertex_colors,
                process=False,
            )
            return new_mesh, uvs
        except Exception as e:
            logger.warning("xatlas failed: %s, using fallback UV projection", e)
            uvs = _generate_uv_atlas_trimesh(mesh)
            return mesh, uvs

    def _render_views(
        self, cfg: BakeConfig,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Render color images from MCP for texture baking."""
        from .mesh_extractor import _generate_orbit_cameras, _call_mcp_render

        cameras = _generate_orbit_cameras(
            n_views=cfg.num_viewpoints,
            distance=cfg.camera_distance,
            fov_deg=cfg.camera_fov_deg,
            width=cfg.render_width,
            height=cfg.render_height,
        )

        images = []
        poses = []
        for i, (extrinsics, intrinsics) in enumerate(cameras):
            color = _call_mcp_render(
                intrinsics.width, intrinsics.height, extrinsics,
                render_type="color", mcp_endpoint=cfg.mcp_endpoint,
            )
            if color is not None:
                if color.dtype != np.uint8:
                    color = np.clip(color * 255, 0, 255).astype(np.uint8)
                images.append(color)
                poses.append(extrinsics)

        logger.info("Rendered %d/%d views for texture baking", len(images), len(cameras))
        return images, poses

    def _project_to_texture(
        self,
        mesh: trimesh.Trimesh,
        uv_coords: np.ndarray,
        color_images: list[np.ndarray],
        camera_poses: list[np.ndarray],
    ) -> np.ndarray:
        """Project rendered images onto the UV texture atlas.

        For each texel, find the corresponding 3D point on the mesh surface,
        project it into each camera view, and blend the visible colors.
        """
        cfg = self.config
        tex_size = cfg.texture_size
        texture = np.full((tex_size, tex_size, 3), cfg.background_color, dtype=np.float32)
        weight_map = np.zeros((tex_size, tex_size), dtype=np.float32)

        vertices = mesh.vertices
        faces = mesh.faces

        # Build per-face UV triangles for rasterization
        for fi in range(len(faces)):
            v_idx = faces[fi]
            tri_uv = uv_coords[v_idx]  # 3x2
            tri_3d = vertices[v_idx]    # 3x3
            face_normal = np.cross(
                tri_3d[1] - tri_3d[0], tri_3d[2] - tri_3d[0],
            )
            fn_len = np.linalg.norm(face_normal)
            if fn_len < 1e-12:
                continue
            face_normal /= fn_len

            # Rasterize this triangle in UV space
            uv_pixel = tri_uv * tex_size
            min_u = max(0, int(np.floor(uv_pixel[:, 0].min())) - 1)
            max_u = min(tex_size - 1, int(np.ceil(uv_pixel[:, 0].max())) + 1)
            min_v = max(0, int(np.floor(uv_pixel[:, 1].min())) - 1)
            max_v = min(tex_size - 1, int(np.ceil(uv_pixel[:, 1].max())) + 1)

            if max_u <= min_u or max_v <= min_v:
                continue

            # Precompute barycentric coordinate helpers
            uv0, uv1, uv2 = uv_pixel[0], uv_pixel[1], uv_pixel[2]
            d00 = np.dot(uv1 - uv0, uv1 - uv0)
            d01 = np.dot(uv1 - uv0, uv2 - uv0)
            d11 = np.dot(uv2 - uv0, uv2 - uv0)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-12:
                continue

            for py in range(min_v, max_v + 1):
                for px in range(min_u, max_u + 1):
                    p = np.array([px + 0.5, py + 0.5])
                    dp0 = p - uv0
                    d20 = np.dot(dp0, uv1 - uv0)
                    d21 = np.dot(dp0, uv2 - uv0)
                    bary_v = (d11 * d20 - d01 * d21) / denom
                    bary_w = (d00 * d21 - d01 * d20) / denom
                    bary_u = 1.0 - bary_v - bary_w

                    if bary_u < -0.01 or bary_v < -0.01 or bary_w < -0.01:
                        continue

                    # Interpolate 3D position
                    world_pt = (bary_u * tri_3d[0] + bary_v * tri_3d[1] + bary_w * tri_3d[2])

                    # Project into each camera and sample color
                    total_color = np.zeros(3, dtype=np.float32)
                    total_w = 0.0

                    for img, cam_pose in zip(color_images, camera_poses):
                        cam_inv = np.linalg.inv(cam_pose)
                        pt_cam = cam_inv[:3, :3] @ world_pt + cam_inv[:3, 3]

                        if pt_cam[2] <= 0:
                            continue

                        # Visibility: check if face normal faces camera
                        cam_dir = cam_pose[:3, 3] - world_pt
                        cam_dir /= np.linalg.norm(cam_dir) + 1e-12
                        ndotv = np.dot(face_normal, cam_dir)
                        if ndotv < 0.05:
                            continue

                        h, w = img.shape[:2]
                        fov_rad = np.radians(cfg.camera_fov_deg)
                        fx = (w / 2.0) / np.tan(fov_rad / 2.0)
                        fy = fx

                        px_img = fx * pt_cam[0] / pt_cam[2] + w / 2.0
                        py_img = fy * pt_cam[1] / pt_cam[2] + h / 2.0

                        ix = int(round(px_img))
                        iy = int(round(py_img))

                        if 0 <= ix < w and 0 <= iy < h:
                            sample = img[iy, ix].astype(np.float32)
                            view_weight = ndotv  # weight by facing angle
                            total_color += sample * view_weight
                            total_w += view_weight

                    if total_w > 0:
                        # Flip V for image coordinates (UV origin at bottom-left)
                        ty = tex_size - 1 - py
                        if 0 <= ty < tex_size:
                            texture[ty, px] = total_color / total_w
                            weight_map[ty, px] += total_w

        return np.clip(texture, 0, 255).astype(np.uint8)

    def _rasterize_vertex_colors(
        self,
        mesh: trimesh.Trimesh,
        uv_coords: np.ndarray,
        tex_size: int,
    ) -> np.ndarray:
        """Rasterize vertex colors into UV texture space.

        For each face, interpolate vertex colors across the UV triangle.
        """
        texture = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
        weight_map = np.zeros((tex_size, tex_size), dtype=np.float32)

        vertices = mesh.vertices
        faces = mesh.faces

        # Get vertex colors
        vc = None
        if mesh.visual and hasattr(mesh.visual, 'vertex_colors'):
            vc = np.array(mesh.visual.vertex_colors, dtype=np.float32)
            if vc.shape[1] >= 3:
                vc = vc[:, :3]
                if vc.max() <= 1.0:
                    vc *= 255.0
        if vc is None:
            vc = np.full((len(vertices), 3), 180.0, dtype=np.float32)

        for fi in range(len(faces)):
            v_idx = faces[fi]
            tri_uv = uv_coords[v_idx] * tex_size  # 3x2 in pixel coords
            tri_colors = vc[v_idx]  # 3x3

            min_u = max(0, int(np.floor(tri_uv[:, 0].min())))
            max_u = min(tex_size - 1, int(np.ceil(tri_uv[:, 0].max())))
            min_v = max(0, int(np.floor(tri_uv[:, 1].min())))
            max_v = min(tex_size - 1, int(np.ceil(tri_uv[:, 1].max())))

            uv0, uv1, uv2 = tri_uv[0], tri_uv[1], tri_uv[2]
            d00 = np.dot(uv1 - uv0, uv1 - uv0)
            d01 = np.dot(uv1 - uv0, uv2 - uv0)
            d11 = np.dot(uv2 - uv0, uv2 - uv0)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-12:
                continue

            for py in range(min_v, max_v + 1):
                for px in range(min_u, max_u + 1):
                    p = np.array([px + 0.5, py + 0.5])
                    dp0 = p - uv0
                    d20 = np.dot(dp0, uv1 - uv0)
                    d21 = np.dot(dp0, uv2 - uv0)
                    bary_v = (d11 * d20 - d01 * d21) / denom
                    bary_w = (d00 * d21 - d01 * d20) / denom
                    bary_u = 1.0 - bary_v - bary_w

                    if bary_u < -0.01 or bary_v < -0.01 or bary_w < -0.01:
                        continue

                    color = bary_u * tri_colors[0] + bary_v * tri_colors[1] + bary_w * tri_colors[2]
                    ty = tex_size - 1 - py
                    if 0 <= ty < tex_size:
                        texture[ty, px] += color
                        weight_map[ty, px] += 1.0

        # Average overlapping writes
        mask = weight_map > 0
        for c in range(3):
            texture[:, :, c][mask] /= weight_map[mask]

        return np.clip(texture, 0, 255).astype(np.uint8)

    def _dilate_texture(
        self,
        texture: np.ndarray,
        iterations: int = 4,
    ) -> np.ndarray:
        """Dilate texture to fill padding around UV seams.

        Expands colored pixels into neighboring empty pixels.
        """
        from scipy.ndimage import maximum_filter, minimum_filter

        result = texture.copy().astype(np.float32)
        bg = np.array(self.config.background_color, dtype=np.float32)

        # Create mask of filled pixels
        filled = np.any(np.abs(result - bg) > 5, axis=2)

        for _ in range(iterations):
            for c in range(3):
                channel = result[:, :, c]
                dilated = maximum_filter(channel, size=3)
                # Only fill unfilled pixels
                result[:, :, c] = np.where(filled, channel, dilated)
            # Update filled mask
            filled = np.any(np.abs(result - bg) > 5, axis=2)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _save_texture(self, texture: np.ndarray, path: Path) -> None:
        """Save texture array as PNG."""
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(texture)
        img.save(str(path))
        logger.info("Saved texture: %s (%dx%d)", path, texture.shape[1], texture.shape[0])

    def _apply_texture(
        self,
        mesh: trimesh.Trimesh,
        uv_coords: np.ndarray,
        texture_path: Path,
    ) -> trimesh.Trimesh:
        """Apply a texture image to the mesh using UV coordinates.

        Returns a new mesh with TextureVisuals.
        """
        from PIL import Image

        img = Image.open(str(texture_path))
        material = trimesh.visual.material.SimpleMaterial(image=img)
        visual = trimesh.visual.TextureVisuals(uv=uv_coords, material=material)

        textured = mesh.copy()
        textured.visual = visual
        return textured
