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
        intrinsics_list: Optional[list[np.ndarray]] = None,
    ) -> np.ndarray:
        """Project rendered images onto the UV texture atlas.

        For each texel, find the corresponding 3D point on the mesh surface,
        project it into each camera view, and blend the visible colors.

        Args:
            intrinsics_list: Optional list of 3x3 K matrices per camera.
                If provided, overrides camera_fov_deg for projection.
        """
        cfg = self.config
        tex_size = cfg.texture_size
        texture = np.full((tex_size, tex_size, 3), cfg.background_color, dtype=np.float32)
        weight_map = np.zeros((tex_size, tex_size), dtype=np.float32)

        vertices = mesh.vertices
        faces = mesh.faces

        # Precompute per-camera world-to-camera transforms
        cam_invs = [np.linalg.inv(p) for p in camera_poses]

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

                    for ci, (img, cam_pose) in enumerate(zip(color_images, camera_poses)):
                        cam_inv = cam_invs[ci]
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

                        # Use K matrix if provided, else fall back to FOV
                        if intrinsics_list is not None and ci < len(intrinsics_list):
                            K = intrinsics_list[ci]
                            fx, fy = K[0, 0], K[1, 1]
                            cx, cy = K[0, 2], K[1, 2]
                        else:
                            fov_rad = np.radians(cfg.camera_fov_deg)
                            fx = (w / 2.0) / np.tan(fov_rad / 2.0)
                            fy = fx
                            cx, cy = w / 2.0, h / 2.0

                        px_img = fx * pt_cam[0] / pt_cam[2] + cx
                        py_img = fy * pt_cam[1] / pt_cam[2] + cy

                        ix = int(round(px_img))
                        iy = int(round(py_img))

                        if 0 <= ix < w and 0 <= iy < h:
                            sample = img[iy, ix].astype(np.float32)
                            # Skip near-black pixels (background)
                            if sample.sum() < 15:
                                continue
                            view_weight = ndotv
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

        Attempts GPU-accelerated rasterization via PyTorch, falls back to
        vectorized numpy if CUDA is unavailable.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return self._rasterize_vertex_colors_gpu(mesh, uv_coords, tex_size)
        except ImportError:
            pass
        return self._rasterize_vertex_colors_cpu(mesh, uv_coords, tex_size)

    def _rasterize_vertex_colors_gpu(
        self,
        mesh: trimesh.Trimesh,
        uv_coords: np.ndarray,
        tex_size: int,
    ) -> np.ndarray:
        """GPU-accelerated vertex color rasterization using PyTorch.

        Processes faces in batches on GPU. For 100K faces at 2048 resolution,
        completes in ~2-5 seconds vs ~300s on CPU.
        """
        import torch

        device = torch.device('cuda')
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
            vc = np.full((len(mesh.vertices), 3), 180.0, dtype=np.float32)

        # Create full-resolution pixel grid on GPU
        px = torch.arange(tex_size, device=device, dtype=torch.float32) + 0.5
        py = torch.arange(tex_size, device=device, dtype=torch.float32) + 0.5
        grid_y, grid_x = torch.meshgrid(py, px, indexing='ij')  # (H, W)
        # Flatten to (H*W, 2)
        pixels = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

        # Output buffers on GPU
        texture_flat = torch.zeros(tex_size * tex_size, 3, device=device, dtype=torch.float32)
        weight_flat = torch.zeros(tex_size * tex_size, device=device, dtype=torch.float32)

        # Move face UVs and colors to GPU
        face_uvs_np = uv_coords[faces] * tex_size  # (F, 3, 2)
        face_colors_np = vc[faces]  # (F, 3, 3)
        face_uvs = torch.tensor(face_uvs_np, device=device, dtype=torch.float32)
        face_colors = torch.tensor(face_colors_np, device=device, dtype=torch.float32)

        # Process in batches of faces to control GPU memory
        batch_size = 4096
        num_faces = len(faces)
        logger.info("GPU texture rasterization: %d faces, %dx%d, batch=%d",
                     num_faces, tex_size, tex_size, batch_size)

        for batch_start in range(0, num_faces, batch_size):
            batch_end = min(batch_start + batch_size, num_faces)
            b_uvs = face_uvs[batch_start:batch_end]  # (B, 3, 2)
            b_colors = face_colors[batch_start:batch_end]  # (B, 3, 3)
            B = b_uvs.shape[0]

            # Bounding boxes per face
            bb_min = b_uvs.min(dim=1).values  # (B, 2)
            bb_max = b_uvs.max(dim=1).values  # (B, 2)

            # For each face, find pixels in its bounding box
            # Use a coarse test: for each pixel, check which face BBs contain it
            # This is done per-face to avoid O(F*pixels) memory
            for fi in range(B):
                uv0 = b_uvs[fi, 0]
                uv1 = b_uvs[fi, 1]
                uv2 = b_uvs[fi, 2]

                mn = bb_min[fi]
                mx = bb_max[fi]

                # Pixel range for this face
                pu_min = max(0, int(mn[0].item()) - 1)
                pu_max = min(tex_size - 1, int(mx[0].item()) + 1)
                pv_min = max(0, int(mn[1].item()) - 1)
                pv_max = min(tex_size - 1, int(mx[1].item()) + 1)

                if pu_max <= pu_min or pv_max <= pv_min:
                    continue

                # Get pixel subset
                h = pv_max - pv_min + 1
                w = pu_max - pu_min + 1
                local_x = torch.arange(pu_min, pu_max + 1, device=device, dtype=torch.float32) + 0.5
                local_y = torch.arange(pv_min, pv_max + 1, device=device, dtype=torch.float32) + 0.5
                gy, gx = torch.meshgrid(local_y, local_x, indexing='ij')
                pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (N, 2)

                # Barycentric coordinates
                e1 = uv1 - uv0
                e2 = uv2 - uv0
                d00 = e1 @ e1
                d01 = e1 @ e2
                d11 = e2 @ e2
                denom = d00 * d11 - d01 * d01
                if abs(denom.item()) < 1e-12:
                    continue

                dp0 = pts - uv0.unsqueeze(0)  # (N, 2)
                d20 = dp0 @ e1
                d21 = dp0 @ e2
                inv_denom = 1.0 / denom
                bv = (d11 * d20 - d01 * d21) * inv_denom
                bw = (d00 * d21 - d01 * d20) * inv_denom
                bu = 1.0 - bv - bw

                inside = (bu >= -0.01) & (bv >= -0.01) & (bw >= -0.01)
                if not inside.any():
                    continue

                # Interpolate colors
                c0 = b_colors[fi, 0]  # (3,)
                c1 = b_colors[fi, 1]
                c2 = b_colors[fi, 2]
                interp = (bu[inside].unsqueeze(1) * c0
                          + bv[inside].unsqueeze(1) * c1
                          + bw[inside].unsqueeze(1) * c2)

                # Compute linear indices (flip V)
                px_c = pts[inside, 0].long()
                py_c = pts[inside, 1].long()
                ty_c = tex_size - 1 - py_c

                valid = (px_c >= 0) & (px_c < tex_size) & (ty_c >= 0) & (ty_c < tex_size)
                if not valid.any():
                    continue

                linear_idx = ty_c[valid] * tex_size + px_c[valid]
                texture_flat.index_add_(0, linear_idx, interp[valid])
                weight_flat.index_add_(0, linear_idx, torch.ones(valid.sum(), device=device))

            if (batch_start + batch_size) % 20000 < batch_size:
                logger.info("  GPU rasterized %d/%d faces", batch_end, num_faces)

        # Average and return
        mask = weight_flat > 0
        texture_flat[mask] /= weight_flat[mask].unsqueeze(1)
        texture = texture_flat.reshape(tex_size, tex_size, 3).cpu().numpy()
        return np.clip(texture, 0, 255).astype(np.uint8)

    def _rasterize_vertex_colors_cpu(
        self,
        mesh: trimesh.Trimesh,
        uv_coords: np.ndarray,
        tex_size: int,
    ) -> np.ndarray:
        """CPU fallback: vectorized numpy vertex color rasterization."""
        texture = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
        weight_map = np.zeros((tex_size, tex_size), dtype=np.float32)

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
            vc = np.full((len(mesh.vertices), 3), 180.0, dtype=np.float32)

        face_uvs = uv_coords[faces] * tex_size
        face_colors = vc[faces]

        for fi in range(len(faces)):
            tri_uv = face_uvs[fi]
            tri_colors = face_colors[fi]

            min_u = max(0, int(np.floor(tri_uv[:, 0].min())))
            max_u = min(tex_size - 1, int(np.ceil(tri_uv[:, 0].max())))
            min_v = max(0, int(np.floor(tri_uv[:, 1].min())))
            max_v = min(tex_size - 1, int(np.ceil(tri_uv[:, 1].max())))

            if max_u <= min_u or max_v <= min_v:
                continue

            uv0 = tri_uv[0]
            e1 = tri_uv[1] - uv0
            e2 = tri_uv[2] - uv0
            d00 = e1 @ e1
            d01 = e1 @ e2
            d11 = e2 @ e2
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-12:
                continue
            inv_denom = 1.0 / denom

            px_range = np.arange(min_u, max_u + 1, dtype=np.float32) + 0.5
            py_range = np.arange(min_v, max_v + 1, dtype=np.float32) + 0.5
            grid_x, grid_y = np.meshgrid(px_range, py_range)
            points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

            dp0 = points - uv0[np.newaxis, :]
            d20 = dp0 @ e1
            d21 = dp0 @ e2
            bary_v = (d11 * d20 - d01 * d21) * inv_denom
            bary_w = (d00 * d21 - d01 * d20) * inv_denom
            bary_u = 1.0 - bary_v - bary_w

            inside = (bary_u >= -0.01) & (bary_v >= -0.01) & (bary_w >= -0.01)
            if not inside.any():
                continue

            colors = (bary_u[inside, np.newaxis] * tri_colors[0]
                      + bary_v[inside, np.newaxis] * tri_colors[1]
                      + bary_w[inside, np.newaxis] * tri_colors[2])

            px_coords = points[inside, 0].astype(np.int32)
            ty_coords = (tex_size - 1 - points[inside, 1]).astype(np.int32)

            valid = (px_coords >= 0) & (px_coords < tex_size) & (ty_coords >= 0) & (ty_coords < tex_size)
            if not valid.any():
                continue

            np.add.at(texture, (ty_coords[valid], px_coords[valid]), colors[valid])
            np.add.at(weight_map, (ty_coords[valid], px_coords[valid]), 1.0)

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
