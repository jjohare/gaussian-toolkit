# Gap Analysis: Pipeline Output -- Local vs Docker

**Date**: 2026-03-30
**Model**: gallery_tour_60s, 30K iterations MRNF, 3.76M gaussians
**Hardware**: NVIDIA RTX A6000 (48GB), Python 3.14, gsplat 1.5.3, torch 2.10+cu128

---

## Executive Summary

The Docker pipeline used a **7K-iteration MCMC model (1M gaussians)** while the
quality model is **30K-iteration MRNF (3.76M gaussians)**. This 3.76x gaussian
count difference is the root cause of the mesh quality gap. Additionally, several
pipeline stages have critical bugs: the texture baker has an O(n^2) Python loop
that hangs on real meshes, the TSDF voxel size is not adaptive to scene extent,
and OpenUSD (pxr) is missing from the runtime.

---

## Stage-by-Stage Results

### Stage 1: gsplat Depth Rendering

| Metric | LOCAL (30K model) | DOCKER (7K model) | Gap |
|--------|-------------------|--------------------|----|
| Gaussian count | 3,760,013 | 1,000,000 | 3.76x more detail |
| PLY load time | 10.9s | ~3s | Expected with size |
| Render time/frame | 16-675ms | ~50ms | First frame JIT warmup |
| Depth coverage | 99.2-100% | 72% (manifests says) | +27% coverage |
| Depth range | 0.14-25.3m | Not recorded | -- |
| Views rendered | 8 (COLMAP cams) | 72 (orbit cams) | Different camera source |

**Root cause of gap**: Docker used 1M-gaussian model with lower coverage. The quality
model fills the entire view frustum for virtually all COLMAP cameras.

**Required fix**: None for depth rendering itself. The fix is upstream: always use the
quality model (30K+ iterations) for mesh extraction.

### Stage 2: gsplat RGB Rendering

| Metric | LOCAL (30K model) | DOCKER (7K model) |
|--------|-------------------|-------------------|
| PSNR vs training images | 12.7-20.6 dB (avg ~16.5) | Not measured |
| Render resolution | Native COLMAP (1602x899) | 1024x1024 orbit |
| Color fidelity | Good, some view-dependent artifacts | Lower (fewer gaussians, less SH detail) |

**PSNR breakdown (local, 8 COLMAP views)**:
- frame_0002: 18.21 dB
- frame_0018: 15.58 dB
- frame_0035: 12.69 dB
- frame_0052: 20.55 dB
- frame_0069: 18.87 dB
- frame_0086: 13.81 dB
- frame_0103: 16.58 dB
- frame_0120: 15.62 dB

**Root cause of lower PSNR on some views**: These are views where the camera moves
through narrow corridors or views distant walls. The 30K model loss was 0.015 (final)
with overall PSNR 18.67 dB and SSIM 0.8746. Per-view variation is expected.

**Required fix**: No code change needed. For quality comparison, add PSNR/SSIM
computation to the render stage. File: `src/pipeline/stages.py`, add after line ~700
(mesh_objects stage) a metric recording step.

### Stage 3: TSDF Mesh Extraction

| Metric | LOCAL (30K, adaptive) | DOCKER quality_mesh (30K) | DOCKER tsdf (7K) |
|--------|----------------------|---------------------------|-------------------|
| Input gaussians | 3,760,013 | 3,760,013 (filtered to 3,048,152) | 1,000,000 (filtered to 389,315) |
| Views rendered | 64 | 36 | 72 |
| Render resolution | 1024x1024 | 1024x1024 | 1024x1024 |
| Voxel size | 0.1213m (adaptive) | 0.1229m | 0.1229m |
| Raw vertices | 268,048 | 572,288 | 1,368,389 |
| Raw faces | 528,442 | 1,131,026 | 2,710,502 |
| Final vertices (decimated) | 268,048 (no cleaning) | 23,202 (50K target) | 22,212 (50K target) |
| Final faces (decimated) | 528,442 (no cleaning) | 49,999 | 48,757 |
| Volume coverage | 95.5% | Not recorded | 72% |
| Render+integrate time | 16.8s | 203.6s | 409.5s |
| Mesh extraction time | 0.2s | 0.97s | 1.86s |
| Total time | 17.0s | 220.9s | 429.0s |
| Watertight | false | false | false |
| Peak memory | ~1.5GB | 3,180MB | 3,032MB |

**Root cause of Docker being 25x slower**: The Docker pipeline's `extract_from_gsplat`
script in `scripts/extract_mesh_gsplat.py` does NOT use gsplat's batch rasterization.
It processes each view sequentially with full CPU-side TSDF integration including a
Python slab loop. The local run benefits from torch JIT compilation reducing per-view
render time to ~16ms (vs ~5s in Docker after first-frame JIT).

**Root cause of mesh quality gap**: The 7K Docker model produces 389K gaussians after
outlier filtering (vs 3.05M for the quality model). Fewer gaussians = less surface
coverage = worse depth maps = sparser TSDF = lower quality mesh.

**Required fixes**:
1. `src/pipeline/mesh_extractor.py` line 267 (`TSDFConfig`): Make `voxel_size` adaptive
   to scene extent. Currently hardcoded to 0.005 which would create a 6000^3 grid for
   this scene. The `extract_from_gsplat` function in `stages.py` does compute adaptive
   voxel size, but the default TSDFConfig does not.
   ```python
   # Line 267: Change default to sentinel value
   voxel_size: float = 0.0  # 0 = auto-compute from scene bounds
   ```
   Then in `TSDFVolume.__init__`, compute adaptive size if 0:
   ```python
   if self.voxel_size == 0:
       extent = np.max(config.volume_bounds_max - config.volume_bounds_min)
       self.voxel_size = extent / 300.0  # Target ~300^3 grid
   ```

2. `src/pipeline/stages.py` line ~900 (mesh_objects): Add the quality model path
   as a parameter instead of always using the 7K model.

### Stage 4: Texture Baking

| Metric | LOCAL (50K decimated) | DOCKER gsplat_mesh_test | Gap |
|--------|----------------------|-------------------------|-----|
| Input faces | 50,000 | 99,999 | -- |
| Texture size | 1024x1024 | 491KB PNG | Similar |
| Bake time | 45.7s (50K faces) | ~60s estimated | -- |
| Coverage | 100% | Not measured | -- |
| Method | vertex color rasterization | vertex color rasterization | Same |

**CRITICAL BUG**: The texture baker `_rasterize_vertex_colors` in
`src/pipeline/texture_baker.py` lines 380-449 uses a **triple-nested Python loop**
(faces x UV pixels x vertices). For the full 528K-face mesh at 2048 resolution:
- 50K faces at 1024 = 45.7 seconds
- Extrapolated: 528K faces at 2048 = **~8 hours**
- The process was killed after 43 minutes with no output

The same O(n^2) issue exists in `_project_to_texture` (lines 268-378) which has a
**quadruple-nested loop** (faces x UV pixels x cameras x projection).

**Required fixes**:
1. `src/pipeline/texture_baker.py` lines 380-449: Replace Python loop with vectorized
   numpy rasterization or GPU-accelerated texture baking using `nvdiffrast` or
   `pytorch3d.renderer.TexturesUV`. Estimated speedup: 100-1000x.

2. `src/pipeline/texture_baker.py` lines 268-378: Same vectorization needed for
   multi-view projection baking.

3. Alternative: Decimate mesh to 50K faces before texture baking (already done in
   the quality mesh pipeline). Add this as a mandatory step in the pipeline.

### Stage 5: USD Assembly

| Metric | LOCAL | DOCKER | Gap |
|--------|-------|--------|-----|
| pxr available | NO (Python 3.14) | YES (Python 3.12 w/ usd-core) | Missing dep |
| Output format | Minimal USDA stub (395 bytes) | Full USDA with objects (18.7KB) | Massive |
| Camera prims | 0 | 0 (not in objects USDA) | Both missing |
| Object prims | 1 (reference only) | 27+ (with positions, gaussian counts) | Docker better |
| Materials | None | None | Both missing |
| Variant sets | None | None | Both missing |

**Root cause**: `usd-core` package does not support Python 3.14 (only 3.10-3.12).
The `usd_assembler.py` module with full `pxr` support (variant sets, camera prims,
materials) cannot run. The fallback `_write_minimal_usda` produces only a stub.

**Required fixes**:
1. `Dockerfile.consolidated`: Pin Python to 3.12 for USD compatibility, or install
   `usd-core` in a Python 3.12 venv and invoke the assembler as a subprocess.

2. `src/pipeline/stages.py` line 1434: The standalone assembler script
   `scripts/assemble_usd_scene.py` exists but also requires `pxr`. Should detect
   available Python version and use the right interpreter.

3. `src/pipeline/preflight.py`: Currently logs a warning when pxr is missing but
   does NOT fail. Since USD is the final deliverable, this should be at minimum a
   loud warning with degraded output notification.

---

## Docker-Specific Gaps

### COLMAP Registration: 13-36%

**Local ground truth**: 121 images all registered (100% from existing COLMAP data).

**Docker result**: 13-36% registration rate on fresh COLMAP runs.

**Root cause**: The Docker pipeline runs COLMAP from scratch on extracted frames.
Frame extraction FPS, image quality, and COLMAP matcher settings affect registration.
The `sequential` matcher (default) struggles with video that has rapid camera motion.

**Required fix**: `src/pipeline/config.py` line 33: Change default matcher from
`"sequential"` to `"exhaustive"` for better registration on challenging video.
Also add vocabulary tree matcher as fallback.

### Segmentation: Full-Scene Fallback

**Docker result**: 117 segments detected, but many are noise (tiny objects < 100
gaussians). The pipeline fell back to full-scene segmentation.

**Local ground truth**: With the quality model, SAM2/SAM3 segmentation would have
3.76M gaussians to work with, providing much better coverage.

**Required fix**: `src/pipeline/config.py` line 56: Increase `min_object_gaussians`
from 100 to 1000 to filter noise segments. The Docker model only had 1M total
gaussians, so the threshold was set low.

---

## Summary of Required Fixes (Priority Order)

| # | File | Line(s) | Fix | Impact |
|---|------|---------|-----|--------|
| 1 | `src/pipeline/texture_baker.py` | 380-449, 268-378 | Vectorize rasterization loops with numpy/GPU | CRITICAL: 1000x speedup |
| 2 | `src/pipeline/mesh_extractor.py` | 267 | Auto-compute voxel_size from scene bounds | HIGH: prevents OOM on large scenes |
| 3 | `src/pipeline/preflight.py` | 79-88 | Expand checks, add version requirements, add sam2/sam3/scipy/PIL | HIGH: fail-fast on missing deps |
| 4 | `Dockerfile.consolidated` | Python version | Pin to 3.12 for usd-core compatibility | HIGH: enables full USD output |
| 5 | `src/pipeline/config.py` | 33 | Default matcher to "exhaustive" | MEDIUM: better COLMAP registration |
| 6 | `src/pipeline/config.py` | 56 | min_object_gaussians: 100 -> 1000 | MEDIUM: filter noise segments |
| 7 | `src/pipeline/stages.py` | ~900 | Accept model path parameter for mesh_objects | MEDIUM: allow quality model selection |
| 8 | `src/pipeline/stages.py` | 1434 | USD assembler Python version detection | LOW: enables full USD on mixed envs |
