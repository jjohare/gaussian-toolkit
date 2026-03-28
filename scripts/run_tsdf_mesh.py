#!/usr/bin/env python3
"""TSDF mesh extraction from Gaussian splat PLY via software depth rendering.

Loads a trained 3DGS PLY, renders depth maps from orbit cameras using
a numpy-based gaussian splatting depth renderer, fuses into a TSDF volume,
extracts mesh via marching cubes, cleans it, and saves OBJ + GLB.

No MCP server or Open3D required.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData

# Add src/pipeline to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from pipeline.mesh_extractor import (
    CameraIntrinsics,
    TSDFConfig,
    TSDFVolume,
    _generate_orbit_cameras,
)
from pipeline.mesh_cleaner import MeshCleaner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Gaussian PLY loader
# ---------------------------------------------------------------------------

@dataclass
class GaussianCloud:
    """Loaded gaussian splat data with activated parameters."""
    positions: np.ndarray    # (N, 3) world-space centers
    colors: np.ndarray       # (N, 3) RGB in [0, 1]
    opacities: np.ndarray    # (N,)   activated opacity in [0, 1]
    scales: np.ndarray       # (N, 3) activated scales (world units)
    rotations: np.ndarray    # (N, 4) quaternions (w, x, y, z)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def sh_c0_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    """Convert zeroth-order SH coefficient to RGB color."""
    C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
    return np.clip(f_dc * C0 + 0.5, 0.0, 1.0)


def load_gaussian_ply(ply_path: str | Path) -> GaussianCloud:
    """Load a 3DGS PLY file and activate parameters."""
    t0 = time.perf_counter()
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]

    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
    colors = sh_c0_to_rgb(f_dc)
    opacities = sigmoid(v["opacity"].astype(np.float32))
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32))
    rotations = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    # Normalize quaternions
    rot_norms = np.linalg.norm(rotations, axis=-1, keepdims=True)
    rotations = rotations / np.maximum(rot_norms, 1e-8)

    dt = time.perf_counter() - t0
    logger.info("Loaded %d gaussians from %s in %.1fs", len(positions), ply_path, dt)
    logger.info("  Position range: [%.1f, %.1f] x [%.1f, %.1f] x [%.1f, %.1f]",
                positions[:, 0].min(), positions[:, 0].max(),
                positions[:, 1].min(), positions[:, 1].max(),
                positions[:, 2].min(), positions[:, 2].max())
    logger.info("  Opacity: mean=%.3f, >0.5: %d (%.1f%%)",
                opacities.mean(),
                (opacities > 0.5).sum(),
                100.0 * (opacities > 0.5).mean())
    logger.info("  Scale mean: %.4f", scales.mean())

    return GaussianCloud(
        positions=positions,
        colors=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )


# ---------------------------------------------------------------------------
#  Outlier filtering
# ---------------------------------------------------------------------------

def filter_outliers(cloud: GaussianCloud, percentile: float = 95.0,
                    min_opacity: float = 0.05) -> GaussianCloud:
    """Remove outlier gaussians based on distance from centroid and opacity.

    Args:
        cloud: Input gaussian cloud.
        percentile: Keep positions within this distance percentile.
        min_opacity: Minimum activated opacity to keep.

    Returns:
        Filtered GaussianCloud.
    """
    # Opacity filter
    opacity_mask = cloud.opacities >= min_opacity
    logger.info("Opacity filter (>%.2f): %d -> %d",
                min_opacity, len(cloud.opacities), opacity_mask.sum())

    # Distance filter on opacity-passing points
    positions = cloud.positions[opacity_mask]
    centroid = np.median(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    dist_threshold = np.percentile(distances, percentile)

    dist_mask_inner = distances <= dist_threshold
    logger.info("Distance filter (%.0fth pct, threshold=%.1f): %d -> %d",
                percentile, dist_threshold, len(distances), dist_mask_inner.sum())

    # Combine masks
    full_indices = np.where(opacity_mask)[0]
    keep_indices = full_indices[dist_mask_inner]

    # Scale filter: remove extremely large gaussians
    kept_scales = cloud.scales[keep_indices]
    max_scale = np.max(kept_scales, axis=1)
    scale_threshold = np.percentile(max_scale, 99)
    scale_mask = max_scale <= scale_threshold
    keep_indices = keep_indices[scale_mask]
    logger.info("Scale filter (99th pct, threshold=%.4f): kept %d",
                scale_threshold, len(keep_indices))

    return GaussianCloud(
        positions=cloud.positions[keep_indices],
        colors=cloud.colors[keep_indices],
        opacities=cloud.opacities[keep_indices],
        scales=cloud.scales[keep_indices],
        rotations=cloud.rotations[keep_indices],
    )


# ---------------------------------------------------------------------------
#  Gaussian splatting depth renderer (numpy)
# ---------------------------------------------------------------------------

def render_depth_and_color(
    cloud: GaussianCloud,
    extrinsics: np.ndarray,
    intrinsics: CameraIntrinsics,
    near: float = 0.1,
    far: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render depth and color from a gaussian cloud using alpha compositing.

    Projects gaussian centers to screen, sorts by depth, and composites
    using opacity and a gaussian footprint weight.

    Args:
        cloud: Filtered gaussian cloud.
        extrinsics: 4x4 camera-to-world matrix.
        intrinsics: Camera intrinsics.
        near: Near clipping plane.
        far: Far clipping plane.

    Returns:
        (depth_map, color_image) as HxW float32 and HxWx3 float32.
    """
    W, H = intrinsics.width, intrinsics.height
    world_to_cam = np.linalg.inv(extrinsics)

    # Transform positions to camera space
    pos = cloud.positions  # (N, 3)
    N = len(pos)
    ones = np.ones((N, 1), dtype=np.float32)
    pos_h = np.hstack([pos, ones])  # (N, 4)
    cam_pts = (world_to_cam @ pos_h.T).T  # (N, 4)

    cam_x = cam_pts[:, 0]
    cam_y = cam_pts[:, 1]
    cam_z = cam_pts[:, 2]

    # Depth clipping
    valid = (cam_z > near) & (cam_z < far)
    if not np.any(valid):
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W, 3), dtype=np.float32)

    # Project to pixel coordinates
    u = (intrinsics.fx * cam_x / cam_z + intrinsics.cx)
    v = (intrinsics.fy * cam_y / cam_z + intrinsics.cy)

    # Screen bounds
    valid &= (u >= 0) & (u < W) & (v >= 0) & (v < H)

    indices = np.where(valid)[0]
    if len(indices) == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W, 3), dtype=np.float32)

    depths = cam_z[indices]
    us = u[indices]
    vs = v[indices]
    opacities = cloud.opacities[indices]
    colors = cloud.colors[indices]

    # Compute projected gaussian radius for weighting
    # Use the max scale projected by focal length / depth
    scales_max = np.max(cloud.scales[indices], axis=1)
    projected_radius = intrinsics.fx * scales_max / depths
    # Clamp radius to reasonable range for splatting
    projected_radius = np.clip(projected_radius, 0.5, 50.0)

    # Sort front-to-back for proper alpha compositing
    sort_idx = np.argsort(depths)
    depths = depths[sort_idx]
    us = us[sort_idx]
    vs = vs[sort_idx]
    opacities = opacities[sort_idx]
    colors = colors[sort_idx]
    projected_radius = projected_radius[sort_idx]

    # Alpha compositing into framebuffer
    depth_buf = np.zeros((H, W), dtype=np.float32)
    color_buf = np.zeros((H, W, 3), dtype=np.float32)
    accum_alpha = np.zeros((H, W), dtype=np.float32)

    ui = us.astype(np.int32)
    vi = vs.astype(np.int32)

    # For efficiency, splat each gaussian as a single pixel weighted by opacity
    # For larger gaussians, splat a small kernel
    KERNEL_THRESHOLD = 2.0  # only splat kernel if projected radius > this

    # Single-pixel fast path for small gaussians
    small_mask = projected_radius <= KERNEL_THRESHOLD
    if np.any(small_mask):
        s_ui = ui[small_mask]
        s_vi = vi[small_mask]
        s_depths = depths[small_mask]
        s_opacities = opacities[small_mask]
        s_colors = colors[small_mask]

        for j in range(len(s_ui)):
            px, py = s_ui[j], s_vi[j]
            alpha_here = accum_alpha[py, px]
            if alpha_here > 0.99:
                continue
            a = s_opacities[j] * (1.0 - alpha_here)
            depth_buf[py, px] += a * s_depths[j]
            color_buf[py, px] += a * s_colors[j]
            accum_alpha[py, px] += a

    # Kernel splatting for larger gaussians
    large_mask = ~small_mask
    large_indices = np.where(large_mask)[0]
    for j in large_indices:
        px, py = int(us[j]), int(vs[j])
        r = int(np.ceil(projected_radius[j]))
        r = min(r, 5)  # cap kernel size for performance

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = px + dx, py + dy
                if nx < 0 or nx >= W or ny < 0 or ny >= H:
                    continue
                # Gaussian weight
                dist_sq = dx * dx + dy * dy
                sigma = projected_radius[j] * 0.5
                w = np.exp(-0.5 * dist_sq / max(sigma * sigma, 0.01))

                alpha_here = accum_alpha[ny, nx]
                if alpha_here > 0.99:
                    continue
                a = opacities[j] * w * (1.0 - alpha_here)
                depth_buf[ny, nx] += a * depths[j]
                color_buf[ny, nx] += a * colors[j]
                accum_alpha[ny, nx] += a

    # Normalize by accumulated alpha
    valid_px = accum_alpha > 1e-6
    depth_buf[valid_px] /= accum_alpha[valid_px]
    for c in range(3):
        color_buf[:, :, c][valid_px] /= accum_alpha[valid_px]

    # Zero out depth where no coverage
    depth_buf[~valid_px] = 0.0

    return depth_buf, color_buf


def render_depth_and_color_vectorized(
    cloud: GaussianCloud,
    extrinsics: np.ndarray,
    intrinsics: CameraIntrinsics,
    near: float = 0.1,
    far: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated depth renderer using opacity-weighted front-surface depth.

    Strategy for TSDF-quality depth maps:
    1. Project all gaussians to screen coordinates
    2. For each pixel, find the median depth of high-opacity gaussians
       within a depth band (to get the front surface, not the cloud average)
    3. Use scatter_reduce with 'amin' for a fast min-depth approximation,
       then smooth lightly to fill gaps

    Falls back to numpy version if torch/CUDA unavailable.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            raise ImportError("No CUDA")
    except ImportError:
        return render_depth_and_color(cloud, extrinsics, intrinsics, near, far)

    W, H = intrinsics.width, intrinsics.height
    device = torch.device("cuda")

    world_to_cam = torch.tensor(np.linalg.inv(extrinsics), dtype=torch.float32, device=device)

    pos = torch.tensor(cloud.positions, dtype=torch.float32, device=device)
    N = pos.shape[0]
    ones = torch.ones((N, 1), dtype=torch.float32, device=device)
    pos_h = torch.cat([pos, ones], dim=1)
    cam_pts = (world_to_cam @ pos_h.T).T

    cam_x, cam_y, cam_z = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]

    valid = (cam_z > near) & (cam_z < far)

    u_f = intrinsics.fx * cam_x / cam_z + intrinsics.cx
    v_f = intrinsics.fy * cam_y / cam_z + intrinsics.cy

    # Allow margin for splat radius
    margin = 5.0
    valid &= (u_f >= -margin) & (u_f < W + margin) & (v_f >= -margin) & (v_f < H + margin)

    indices = torch.where(valid)[0]
    if len(indices) == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W, 3), dtype=np.float32)

    depths_v = cam_z[indices]
    us_v = u_f[indices]
    vs_v = v_f[indices]
    opacities_v = torch.tensor(cloud.opacities, dtype=torch.float32, device=device)[indices]
    colors_v = torch.tensor(cloud.colors, dtype=torch.float32, device=device)[indices]
    scales_v = torch.tensor(cloud.scales, dtype=torch.float32, device=device)[indices]

    # Compute projected radius for each gaussian
    scales_max = scales_v.max(dim=1).values
    projected_radius = (intrinsics.fx * scales_max / depths_v).clamp(min=1.0, max=10.0)
    splat_r = projected_radius.long().clamp(min=1, max=5)

    # Build depth map using min-depth per pixel (front surface)
    # Also accumulate weighted color from gaussians near the front surface
    INF = float(far * 2)
    depth_buf = torch.full((H, W), INF, dtype=torch.float32, device=device)
    color_buf = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    weight_buf = torch.zeros((H, W), dtype=torch.float32, device=device)

    # Phase 1: Scatter minimum depth with small splat kernels
    for r in range(0, 6):
        if r == 0:
            mask = splat_r >= 1  # all gaussians get center pixel
        else:
            mask = splat_r >= r

        if not mask.any():
            continue

        batch_u = us_v[mask]
        batch_v = vs_v[mask]
        batch_d = depths_v[mask]

        offsets = [(0, 0)] if r == 0 else []
        if r > 0:
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r and (dx != 0 or dy != 0):
                        offsets.append((dx, dy))

        for dx, dy in offsets:
            px = (batch_u + dx).long()
            py = (batch_v + dy).long()
            in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            if not in_bounds.any():
                continue
            px_valid = px[in_bounds]
            py_valid = py[in_bounds]
            d_valid = batch_d[in_bounds]
            # Scatter min
            depth_buf[py_valid, px_valid] = torch.minimum(
                depth_buf[py_valid, px_valid], d_valid
            )

    # Phase 2: Accumulate weighted color for gaussians near the front surface
    # Use a depth band around the min-depth at each pixel
    depth_band = depth_buf.max() * 0.05  # 5% of max depth as band width
    depth_flat = depth_buf.reshape(-1)

    ui = us_v.long().clamp(0, W - 1)
    vi = vs_v.long().clamp(0, H - 1)
    pixel_idx = vi * W + ui

    # Get the min-depth at each gaussian's pixel
    ref_depth = depth_flat[pixel_idx]
    # Keep gaussians within the depth band of the front surface
    near_surface = (depths_v - ref_depth).abs() < depth_band
    near_surface &= ref_depth < INF

    if near_surface.any():
        ns_idx = pixel_idx[near_surface]
        ns_w = opacities_v[near_surface]
        ns_c = colors_v[near_surface]

        weight_flat = weight_buf.reshape(-1)
        weight_flat.scatter_add_(0, ns_idx, ns_w)

        color_flat = color_buf.reshape(H * W, 3)
        for ci in range(3):
            color_flat[:, ci].scatter_add_(0, ns_idx, ns_w * ns_c[:, ci])

    # Normalize color
    valid_mask = weight_buf > 1e-6
    for ci in range(3):
        color_buf[:, :, ci][valid_mask] /= weight_buf[valid_mask]

    # Mark pixels with no depth as 0
    depth_buf[depth_buf >= INF] = 0.0

    # Phase 3: Light dilation to fill 1-pixel gaps
    depth_np = depth_buf.cpu().numpy()
    color_np = color_buf.cpu().numpy()

    from scipy.ndimage import maximum_filter, uniform_filter
    valid_px = depth_np > 0
    # Dilate depth slightly to fill small holes
    dilated = maximum_filter(depth_np, size=3)
    fill_mask = (~valid_px) & (dilated > 0)
    depth_np[fill_mask] = dilated[fill_mask]
    # Smooth color into holes
    for ci in range(3):
        smoothed = uniform_filter(color_np[:, :, ci], size=3)
        color_np[:, :, ci][fill_mask] = smoothed[fill_mask]

    return depth_np, color_np


# ---------------------------------------------------------------------------
#  Scene bounds estimation
# ---------------------------------------------------------------------------

def estimate_scene_bounds(
    cloud: GaussianCloud,
    padding_factor: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Estimate scene bounding box and camera orbit parameters.

    Returns:
        (bounds_min, bounds_max, scene_radius, camera_distance)
    """
    pos = cloud.positions
    centroid = np.median(pos, axis=0)
    distances = np.linalg.norm(pos - centroid, axis=1)
    scene_radius = np.percentile(distances, 98)

    bounds_min = centroid - scene_radius * padding_factor
    bounds_max = centroid + scene_radius * padding_factor

    camera_distance = scene_radius * 2.5

    logger.info("Scene centroid: [%.1f, %.1f, %.1f]", *centroid)
    logger.info("Scene radius: %.1f", scene_radius)
    logger.info("TSDF bounds: [%.1f..%.1f] x [%.1f..%.1f] x [%.1f..%.1f]",
                bounds_min[0], bounds_max[0],
                bounds_min[1], bounds_max[1],
                bounds_min[2], bounds_max[2])
    logger.info("Camera distance: %.1f", camera_distance)

    return bounds_min, bounds_max, scene_radius, camera_distance


# ---------------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------------

def run_tsdf_pipeline(
    ply_path: str,
    output_dir: str,
    num_views: int = 36,
    voxel_size: float | None = None,
    render_width: int = 512,
    render_height: int = 512,
    target_faces: int = 50000,
    use_gpu: bool = True,
    outlier_percentile: float = 95.0,
    min_opacity: float = 0.05,
) -> dict:
    """Run the full TSDF mesh extraction pipeline.

    Args:
        ply_path: Path to trained gaussian PLY.
        output_dir: Directory for output files.
        num_views: Number of orbit viewpoints.
        voxel_size: TSDF voxel size (auto-computed if None).
        render_width: Depth map width.
        render_height: Depth map height.
        target_faces: Target face count after decimation.
        use_gpu: Use CUDA-accelerated rendering if available.
        outlier_percentile: Distance percentile for outlier filtering.
        min_opacity: Minimum opacity threshold.

    Returns:
        Dict with timing stats and output paths.
    """
    import trimesh

    stats = {"timings": {}}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load gaussian PLY
    t0 = time.perf_counter()
    cloud = load_gaussian_ply(ply_path)
    stats["timings"]["load_ply"] = time.perf_counter() - t0
    stats["raw_gaussians"] = len(cloud.positions)

    # 2. Filter outliers
    t0 = time.perf_counter()
    cloud = filter_outliers(cloud, percentile=outlier_percentile, min_opacity=min_opacity)
    stats["timings"]["filter_outliers"] = time.perf_counter() - t0
    stats["filtered_gaussians"] = len(cloud.positions)

    # 3. Estimate scene bounds
    bounds_min, bounds_max, scene_radius, camera_distance = estimate_scene_bounds(cloud)

    # Auto-compute voxel size: target ~300^3 volume
    if voxel_size is None:
        extent = (bounds_max - bounds_min).max()
        voxel_size = extent / 300.0
        logger.info("Auto voxel size: %.4f (for ~300^3 volume)", voxel_size)

    sdf_trunc = voxel_size * 5.0

    # 4. Configure TSDF
    config = TSDFConfig(
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        volume_bounds_min=bounds_min,
        volume_bounds_max=bounds_max,
        num_viewpoints=num_views,
        render_width=render_width,
        render_height=render_height,
        camera_distance=camera_distance,
        camera_fov_deg=60.0,
        target_faces=target_faces,
    )

    # Shift cameras to orbit around scene centroid
    centroid = (bounds_min + bounds_max) / 2.0

    # 5. Generate orbit cameras
    cameras = _generate_orbit_cameras(
        n_views=num_views,
        distance=camera_distance,
        fov_deg=config.camera_fov_deg,
        width=render_width,
        height=render_height,
    )
    # Translate cameras to orbit centroid instead of origin
    translated_cameras = []
    for ext, intr in cameras:
        ext_shifted = ext.copy()
        ext_shifted[:3, 3] += centroid
        translated_cameras.append((ext_shifted, intr))
    cameras = translated_cameras

    logger.info("Generated %d orbit cameras", len(cameras))

    # 6. Render depth maps and fuse into TSDF
    tsdf = TSDFVolume(config)
    render_fn = render_depth_and_color_vectorized if use_gpu else render_depth_and_color

    t0 = time.perf_counter()
    integrated = 0
    for i, (ext, intr) in enumerate(cameras):
        t_view = time.perf_counter()
        depth, color = render_fn(cloud, ext, intr,
                                 near=camera_distance * 0.01,
                                 far=camera_distance * 5.0)

        coverage = (depth > 0).sum() / depth.size
        dt_view = time.perf_counter() - t_view

        if coverage < 0.001:
            logger.warning("View %d/%d: coverage=%.3f%% (skipping), %.2fs",
                          i + 1, len(cameras), coverage * 100, dt_view)
            continue

        tsdf.integrate(depth, color, intr, ext)
        integrated += 1
        logger.info("View %d/%d: coverage=%.1f%%, depth_range=[%.1f, %.1f], %.2fs",
                    i + 1, len(cameras), coverage * 100,
                    depth[depth > 0].min() if (depth > 0).any() else 0,
                    depth[depth > 0].max() if (depth > 0).any() else 0,
                    dt_view)

    stats["timings"]["render_and_fuse"] = time.perf_counter() - t0
    stats["views_rendered"] = len(cameras)
    stats["views_integrated"] = integrated

    if integrated == 0:
        raise RuntimeError("No depth frames had sufficient coverage. "
                          "Check outlier filtering and camera distance.")

    # 7. Extract mesh
    t0 = time.perf_counter()
    mesh = tsdf.extract_mesh()
    stats["timings"]["marching_cubes"] = time.perf_counter() - t0
    stats["raw_mesh_vertices"] = len(mesh.vertices)
    stats["raw_mesh_faces"] = len(mesh.faces)

    # 8. Clean mesh -- decimate first if very large, then clean
    t0 = time.perf_counter()
    cleaner = MeshCleaner()

    if len(mesh.faces) > target_faces * 4:
        # For very large meshes, decimate first to avoid slow component analysis
        logger.info("Pre-decimating from %d to %d faces before cleaning",
                    len(mesh.faces), target_faces * 2)
        mesh = cleaner.decimate(mesh, target_faces * 2)
        mesh = cleaner.remove_degenerate_faces(mesh)

    mesh = cleaner.clean(
        mesh,
        target_faces=target_faces,
        smooth_iterations=5,
        min_component_ratio=0.1,
    )
    stats["timings"]["mesh_clean"] = time.perf_counter() - t0
    stats["final_vertices"] = len(mesh.vertices)
    stats["final_faces"] = len(mesh.faces)
    stats["is_watertight"] = bool(mesh.is_watertight)

    # 9. Save outputs
    t0 = time.perf_counter()

    obj_path = out_dir / "tsdf_mesh.obj"
    glb_path = out_dir / "tsdf_mesh.glb"

    mesh.export(str(obj_path))
    logger.info("Saved OBJ: %s", obj_path)

    mesh.export(str(glb_path))
    logger.info("Saved GLB: %s", glb_path)

    stats["timings"]["save"] = time.perf_counter() - t0
    stats["output_obj"] = str(obj_path)
    stats["output_glb"] = str(glb_path)

    total_time = sum(stats["timings"].values())
    stats["timings"]["total"] = total_time
    stats["peak_memory_mb"] = _get_peak_memory_mb()

    # Save stats
    import json
    stats_path = out_dir / "tsdf_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Stats saved: %s", stats_path)

    _print_summary(stats)
    return stats


def _get_peak_memory_mb() -> float:
    """Get peak RSS memory usage in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def _print_summary(stats: dict) -> None:
    """Print a formatted summary of the pipeline results."""
    print("\n" + "=" * 60)
    print("TSDF Mesh Extraction Summary")
    print("=" * 60)
    print(f"  Raw gaussians:      {stats.get('raw_gaussians', 0):,}")
    print(f"  Filtered gaussians: {stats.get('filtered_gaussians', 0):,}")
    print(f"  Views integrated:   {stats.get('views_integrated', 0)}/{stats.get('views_rendered', 0)}")
    print(f"  Raw mesh:           {stats.get('raw_mesh_vertices', 0):,} verts, {stats.get('raw_mesh_faces', 0):,} faces")
    print(f"  Final mesh:         {stats.get('final_vertices', 0):,} verts, {stats.get('final_faces', 0):,} faces")
    print(f"  Watertight:         {stats.get('is_watertight', False)}")
    print(f"  Peak memory:        {stats.get('peak_memory_mb', 0):.0f} MB")
    print()
    print("Timings:")
    for key, val in stats.get("timings", {}).items():
        print(f"  {key:20s}: {val:.2f}s")
    print()
    print(f"Output OBJ: {stats.get('output_obj', '')}")
    print(f"Output GLB: {stats.get('output_glb', '')}")
    print("=" * 60)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TSDF mesh extraction from Gaussian splat PLY")
    parser.add_argument("--ply", required=True, help="Path to trained .ply file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-views", type=int, default=36, help="Number of orbit viewpoints")
    parser.add_argument("--voxel-size", type=float, default=None, help="TSDF voxel size (auto if omitted)")
    parser.add_argument("--render-size", type=int, default=512, help="Render resolution (square)")
    parser.add_argument("--target-faces", type=int, default=50000, help="Target face count")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU rendering")
    parser.add_argument("--outlier-pct", type=float, default=95.0, help="Outlier distance percentile")
    parser.add_argument("--min-opacity", type=float, default=0.05, help="Minimum opacity threshold")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_tsdf_pipeline(
        ply_path=args.ply,
        output_dir=args.output_dir,
        num_views=args.num_views,
        voxel_size=args.voxel_size,
        render_width=args.render_size,
        render_height=args.render_size,
        target_faces=args.target_faces,
        use_gpu=not args.no_gpu,
        outlier_percentile=args.outlier_pct,
        min_opacity=args.min_opacity,
    )


if __name__ == "__main__":
    main()
