#!/usr/bin/env python3
"""Validation tests for the mesh extraction pipeline.

Tests:
1. Import all pipeline modules
2. Create a synthetic sphere mesh
3. Run mesh cleaning (decimate, smooth)
4. UV unwrap via xatlas
5. Bake vertex colors to texture
6. Save as OBJ with texture
7. TSDF volume: integrate synthetic depth, extract mesh
"""

from __future__ import annotations

import sys
import tempfile
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_mesh_pipeline")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        logger.info("[PASS] %s %s", name, detail)
    else:
        FAIL += 1
        logger.error("[FAIL] %s %s", name, detail)


def test_imports():
    """Test that all pipeline modules import correctly."""
    import trimesh
    check("import trimesh", True, f"v{trimesh.__version__}")

    import xatlas
    check("import xatlas", True)

    import pymeshfix
    check("import pymeshfix", True)

    from skimage.measure import marching_cubes
    check("import marching_cubes", True)

    from src.pipeline.mesh_extractor import MeshExtractor, TSDFVolume, TSDFConfig, CameraIntrinsics
    check("import MeshExtractor", True)

    from src.pipeline.mesh_cleaner import MeshCleaner
    check("import MeshCleaner", True)

    from src.pipeline.texture_baker import TextureBaker, BakeConfig
    check("import TextureBaker", True)


def test_sphere_creation():
    """Create a UV sphere and verify its properties."""
    import trimesh
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    check("sphere creation", len(sphere.vertices) > 0,
          f"{len(sphere.vertices)} verts, {len(sphere.faces)} faces")
    check("sphere watertight", sphere.is_watertight)
    return sphere


def test_mesh_cleaner(sphere):
    """Test cleaning pipeline on the sphere."""
    from src.pipeline.mesh_cleaner import MeshCleaner

    cleaner = MeshCleaner()

    # Test smoothing
    smoothed = cleaner.laplacian_smooth(sphere, iterations=2, lamb=0.3)
    check("laplacian_smooth", len(smoothed.vertices) == len(sphere.vertices),
          "vertex count preserved")

    # Test decimation
    target = 100
    decimated = cleaner.decimate(sphere, target_faces=target)
    check("decimate", len(decimated.faces) <= target * 1.5,
          f"{len(decimated.faces)} faces (target {target})")

    # Test component removal (add a tiny floating triangle)
    import trimesh
    extra_verts = np.array([[5, 5, 5], [5.01, 5, 5], [5, 5.01, 5]], dtype=np.float64)
    extra_faces = np.array([[0, 1, 2]])
    extra = trimesh.Trimesh(vertices=extra_verts, faces=extra_faces)
    combined = trimesh.util.concatenate([sphere, extra])
    check("combined has 2 components",
          len(combined.split(only_watertight=False)) >= 2)

    cleaned = cleaner.remove_small_components(combined, min_ratio=0.01)
    components_after = len(cleaned.split(only_watertight=False))
    check("small component removed", components_after == 1,
          f"{components_after} components remain")

    # Test hole filling
    filled = cleaner.fill_holes(sphere.copy())
    check("fill_holes runs", len(filled.faces) > 0)

    # Test full pipeline
    full_cleaned = cleaner.clean(sphere.copy(), target_faces=200, smooth_iterations=1)
    check("full clean pipeline", len(full_cleaned.faces) > 0,
          f"{len(full_cleaned.faces)} faces")

    return full_cleaned


def test_uv_unwrap(mesh):
    """Test xatlas UV unwrapping."""
    import xatlas

    verts = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)

    vmapping, new_faces, uvs = xatlas.parametrize(verts, faces)

    check("xatlas parametrize", len(uvs) > 0,
          f"{len(uvs)} UV coords")
    check("UV range [0,1]",
          uvs.min() >= -0.01 and uvs.max() <= 1.01,
          f"min={uvs.min():.3f} max={uvs.max():.3f}")

    return vmapping, new_faces, uvs


def test_texture_bake(mesh):
    """Test baking vertex colors to texture."""
    import trimesh
    from src.pipeline.texture_baker import TextureBaker, BakeConfig

    # Add vertex colors to mesh
    colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    # Color by position: red=x, green=y, blue=z
    verts_norm = mesh.vertices - mesh.vertices.min(axis=0)
    extent = verts_norm.max(axis=0)
    extent = np.maximum(extent, 1e-8)
    verts_norm /= extent
    colors[:, 0] = (verts_norm[:, 0] * 255).astype(np.uint8)
    colors[:, 1] = (verts_norm[:, 1] * 255).astype(np.uint8)
    colors[:, 2] = (verts_norm[:, 2] * 255).astype(np.uint8)
    colors[:, 3] = 255
    mesh.visual.vertex_colors = colors

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "diffuse.png"
        config = BakeConfig(texture_size=256)
        baker = TextureBaker(config)
        textured_mesh, out_path = baker.bake_from_vertex_colors(mesh, tex_path)

        check("texture file exists", out_path.exists(), str(out_path))
        check("texture file size > 0", out_path.stat().st_size > 100)
        check("textured mesh has visual",
              textured_mesh.visual is not None)

        # Save as OBJ
        obj_path = Path(tmpdir) / "output.obj"
        textured_mesh.export(str(obj_path), file_type="obj")
        check("OBJ export", obj_path.exists(),
              f"{obj_path.stat().st_size} bytes")

    return True


def test_tsdf_integration():
    """Test TSDF volume with synthetic depth data."""
    from src.pipeline.mesh_extractor import TSDFVolume, TSDFConfig, CameraIntrinsics

    config = TSDFConfig(
        voxel_size=0.02,
        sdf_trunc=0.08,
        volume_bounds_min=np.array([-0.8, -0.8, -0.8]),
        volume_bounds_max=np.array([0.8, 0.8, 0.8]),
    )
    tsdf = TSDFVolume(config)
    check("TSDF volume created", tsdf.nx > 0,
          f"{tsdf.nx}x{tsdf.ny}x{tsdf.nz}")

    # Generate synthetic depth maps of a sphere
    radius = 0.3
    intrinsics = CameraIntrinsics.default(128, 128, 60.0)

    n_views = 12
    for i in range(n_views):
        azimuth = 2 * np.pi * i / n_views
        dist = 1.5
        cx = dist * np.cos(azimuth)
        cy = dist * np.sin(azimuth)
        cz = 0.0
        pos = np.array([cx, cy, cz])

        forward = -pos / np.linalg.norm(pos)
        right = np.cross(forward, np.array([0, 0, 1]))
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(forward, np.array([0, 1, 0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        extrinsics = np.eye(4)
        extrinsics[:3, 0] = right
        extrinsics[:3, 1] = up
        extrinsics[:3, 2] = forward  # +z = viewing direction
        extrinsics[:3, 3] = pos

        # Render synthetic depth: ray-sphere intersection
        depth = _render_sphere_depth(intrinsics, extrinsics, radius)
        color = np.full((intrinsics.height, intrinsics.width, 3), 200, dtype=np.uint8)

        tsdf.integrate(depth, color, intrinsics, extrinsics)

    # Extract mesh
    mesh = tsdf.extract_mesh()
    check("TSDF mesh extracted", len(mesh.vertices) > 0,
          f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    check("TSDF mesh has faces", len(mesh.faces) > 10)

    return mesh


def _render_sphere_depth(intrinsics, extrinsics, radius):
    """Ray-trace a sphere at origin to generate a synthetic depth map (vectorized)."""
    h, w = intrinsics.height, intrinsics.width

    cam_pos = extrinsics[:3, 3]
    cam_rot = extrinsics[:3, :3]  # columns: right, up, forward
    cam_rot_inv = np.linalg.inv(cam_rot)

    # Build pixel grid
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    dx = (uu - intrinsics.cx) / intrinsics.fx
    dy = (vv - intrinsics.cy) / intrinsics.fy
    ones = np.ones_like(dx)

    # Ray directions in camera space: [dx, dy, 1] (normalized)
    rays_cam = np.stack([dx, dy, ones], axis=-1)  # HxWx3
    norms = np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    rays_cam /= norms

    # Transform to world space: ray_world = cam_rot @ ray_cam
    rays_world = np.einsum("ij,hwj->hwi", cam_rot, rays_cam)
    rays_world /= np.linalg.norm(rays_world, axis=-1, keepdims=True)

    # Ray-sphere intersection: |cam_pos + t * ray|^2 = radius^2
    oc = cam_pos  # sphere at origin
    a = np.sum(rays_world * rays_world, axis=-1)  # should be ~1
    b = 2.0 * np.sum(oc[np.newaxis, np.newaxis, :] * rays_world, axis=-1)
    c_val = np.dot(oc, oc) - radius ** 2

    disc = b * b - 4 * a * c_val
    hit_mask = disc >= 0

    t = np.zeros((h, w), dtype=np.float64)
    t[hit_mask] = (-b[hit_mask] - np.sqrt(disc[hit_mask])) / (2 * a[hit_mask])

    valid = hit_mask & (t > 0)

    # Compute depth as z-component in camera frame
    hit_world = cam_pos[np.newaxis, np.newaxis, :] + t[:, :, np.newaxis] * rays_world
    offset = hit_world - cam_pos[np.newaxis, np.newaxis, :]
    hit_cam = np.einsum("ij,hwj->hwi", cam_rot_inv, offset)

    depth = np.zeros((h, w), dtype=np.float32)
    depth[valid] = hit_cam[:, :, 2][valid].astype(np.float32)

    return depth


def test_obj_roundtrip():
    """Test saving and loading an OBJ file."""
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = Path(tmpdir) / "sphere.obj"
        sphere.export(str(obj_path), file_type="obj")
        check("OBJ save", obj_path.exists())

        loaded = trimesh.load(str(obj_path))
        check("OBJ load", len(loaded.vertices) > 0,
              f"{len(loaded.vertices)} verts")

    return True


def main():
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    logger.info("=" * 60)
    logger.info("LichtFeld Studio - Mesh Pipeline Validation")
    logger.info("=" * 60)

    test_imports()
    sphere = test_sphere_creation()
    cleaned = test_mesh_cleaner(sphere)
    test_uv_unwrap(cleaned)
    test_texture_bake(sphere.copy())
    test_tsdf_integration()
    test_obj_roundtrip()

    logger.info("=" * 60)
    logger.info("Results: %d PASSED, %d FAILED", PASS, FAIL)
    logger.info("=" * 60)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
