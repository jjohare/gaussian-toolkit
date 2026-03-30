# MILo Integration Research -- SIGGRAPH Asia 2025

**Repository**: https://github.com/Anttwo/MILo
**Paper**: "MILo: Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction"
**Published**: SIGGRAPH Asia 2025, Journal Track (TOG)
**Authors**: Antoine Guedon, Diego Gomez, Nissim Maruani, Bingchen Gong, George Drettakis, Maks Ovsjanikov (Ecole polytechnique / Inria)
**License**: Gaussian-Splatting License (non-commercial, research and evaluation only)

---

## 1. Dependencies

### Core Stack

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.9 | Conda environment |
| PyTorch | 2.3.1 | Pinned exactly |
| torchvision | 0.18.1 | Pinned exactly |
| torchaudio | 2.3.1 | Pinned exactly |
| CUDA | 11.8 (tested) / 12.1 (untested) | Must set CPATH, LD_LIBRARY_PATH, PATH |
| MKL | 2023.1.0 | Pinned via conda |

### Python Packages (requirements.txt)

| Package | Version |
|---------|---------|
| open3d | 0.19.0 |
| trimesh | 4.6.8 |
| scikit-image | 0.24.0 |
| opencv-python | 4.11.0.86 |
| plyfile | 1.1 |
| tqdm | 4.67.1 |

### Optional (Blender addon)

| Package | Notes |
|---------|-------|
| torch_geometric | For mesh editing/animation pipeline |
| torch_cluster | For mesh editing/animation pipeline |
| Blender | 4.0.2 recommended |

### Submodules (build-from-source, CUDA C++ extensions)

| Submodule | Purpose | Build Method |
|-----------|---------|--------------|
| `diff-gaussian-rasterization` | RaDe-GS rasterizer (default) | `pip install submodules/diff-gaussian-rasterization` |
| `diff-gaussian-rasterization_gof` | GOF rasterizer (alternative) | `pip install submodules/diff-gaussian-rasterization_gof` |
| `diff-gaussian-rasterization_ms` | Mini-Splatting2 rasterizer (fast densification) | `pip install submodules/diff-gaussian-rasterization_ms` |
| `simple-knn` | KNN utility | `pip install submodules/simple-knn` |
| `fused-ssim` | Fused SSIM loss | `pip install submodules/fused-ssim` |
| `tetra_triangulation` | Delaunay tetrahedralization (from Tetra-NeRF) | cmake + make + pip install -e . **Requires CGAL, GMP (conda-forge)** |
| `nvdiffrast` | NVIDIA differentiable rasterizer | `pip install -e .` (git submodule) |
| `Depth-Anything-V2` | Monocular depth (optional, for depth-order reg) | git submodule, needs `vitl` checkpoint |

### Critical Native Dependencies

- **CGAL** -- installed via `conda install conda-forge::cgal`. Required for Delaunay triangulation.
- **GMP** -- installed via `conda install conda-forge::gmp`. Required by CGAL.
- **cmake** -- installed via `conda install cmake`. Required for tetra_triangulation build.

---

## 2. Input Format: COLMAP Dataset Compatibility

**Yes, MILo accepts standard COLMAP datasets.** The scene loader (`scene/__init__.py`) checks for:

1. `sparse/` directory (COLMAP format) -- primary path
2. `transforms_train.json` (Blender/NeRF-synthetic format) -- secondary path

The COLMAP path uses the same `colmap_loader.py` from the original 3DGS codebase. It reads:
- `sparse/0/cameras.bin` (or .txt)
- `sparse/0/images.bin` (or .txt)
- `sparse/0/points3D.bin` (or .txt)
- Images directory

### Can it accept a pre-trained 3DGS PLY?

**No, not directly as a starting point for training.** MILo trains from scratch using COLMAP point clouds as initialization (same as original 3DGS). The `load_ply` path is only used when resuming from a MILo checkpoint (`--start_checkpoint`).

However, the **functional API** (`milo.functional`) can accept raw Gaussian parameters (means, scales, rotations, opacities) from any source, including a pre-trained 3DGS model. This enables:
- Post-hoc mesh extraction from existing trained Gaussians
- Integration of MILo's differentiable mesh pipeline into other codebases

### SuGaR-style workflow compatibility

If you have a COLMAP dataset that SuGaR was trained on, you can feed the exact same COLMAP dataset to MILo. The COLMAP format is identical. You cannot skip MILo's training by feeding it a SuGaR PLY -- MILo must train its own Gaussians because the mesh-in-the-loop regularization is integral to the training process.

---

## 3. Command-Line Interface / Python API

### CLI: Training

```bash
cd milo/

# Basic training (outdoor scene, RaDe-GS rasterizer)
python train.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    --imp_metric outdoor \
    --rasterizer radegs

# Dense Gaussians + high-res mesh + appearance decoupling
python train.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    --imp_metric indoor \
    --rasterizer radegs \
    --dense_gaussians \
    --mesh_config highres \
    --decoupled_appearance \
    --data_device cpu
```

#### Key Training Arguments

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `-s` | path | required | Source COLMAP dataset |
| `-m` | path | required | Output model directory |
| `--imp_metric` | `indoor` / `outdoor` | required | Scene type (changes importance sampling) |
| `--rasterizer` | `radegs` / `gof` | `radegs` | Rasterizer backend |
| `--dense_gaussians` | flag | off | More Gaussians, subset used for Delaunay |
| `--mesh_config` | `default`/`highres`/`veryhighres`/`lowres`/`verylowres` | `default` | Mesh resolution preset |
| `--decoupled_appearance` | flag | off | Handle exposure variations |
| `--depth_order` | flag | off | Enable depth-order regularization (needs DepthAnythingV2 checkpoint) |
| `--data_device` | `cpu` / `cuda` | `cuda` | Where to load images (cpu = less VRAM) |
| `--eval` | flag | off | Train/test split for evaluation |
| `--log_interval` | int | none | Log images every N iterations |
| `--sampling_factor` | float | 1.0 | Reduce Gaussians for low-res configs |

### CLI: Mesh Extraction (3 methods)

```bash
# Method 1: Learned SDF (RECOMMENDED -- best quality)
python mesh_extract_sdf.py \
    -s <COLMAP_DATASET> -m <MODEL_DIR> \
    --rasterizer radegs

# Method 2: Integrated Opacity Field
python mesh_extract_integration.py \
    -s <COLMAP_DATASET> -m <MODEL_DIR> \
    --rasterizer gof --sdf_mode integration

# Method 3: Regular TSDF (non-scalable, bounded scenes only)
python mesh_extract_regular_tsdf.py \
    -s <COLMAP_DATASET> -m <MODEL_DIR> \
    --rasterizer radegs --mesh_res 1024
```

### Python API: Functional Interface

The `milo.functional` module provides standalone functions for integration into external codebases:

```python
from functional import (
    sample_gaussians_on_surface,
    extract_gaussian_pivots,
    compute_initial_sdf_values,
    compute_delaunay_triangulation,
    extract_mesh,
    frustum_cull_mesh,
)

# Inputs: raw Gaussian parameters (means, scales, rotations, opacities)
# + training cameras with same structure as original 3DGS Camera class
# + a render function wrapper returning {"render": ..., "depth": ...}

# Pipeline:
# 1. sample_gaussians_on_surface() -> indices of surface Gaussians
# 2. compute_initial_sdf_values() -> SDF values for pivots
# 3. compute_delaunay_triangulation() -> tetrahedralization
# 4. extract_mesh() -> differentiable mesh (vertices + faces)
# 5. frustum_cull_mesh() -> view-culled mesh for rendering
```

All functions are differentiable. Gradients flow from mesh operations back to Gaussian parameters.

---

## 4. Output Format, Quality, and Timing

### Output Files

| File | Location | Description |
|------|----------|-------------|
| `point_cloud.ply` | `<model>/point_cloud/iteration_18000/` | Trained 3DGS checkpoint |
| `mesh_learnable_sdf.ply` | `<model>/` | Final mesh (SDF method) -- PLY with vertex colors |
| `mesh_integration_sdf.ply` | `<model>/` | Mesh (integration method) |
| `mesh_depth_fusion_sdf.ply` | `<model>/` | Mesh (depth fusion method) |
| `mesh_regular_tsdf_res<N>.ply` | `<model>/` | Mesh (regular TSDF) |
| `mesh_regular_tsdf_res<N>_post.ply` | `<model>/` | Mesh (TSDF, postprocessed) |

### Mesh Resolution Presets

| Config | Max Delaunay Vertices | Approx Output Size | Use Case |
|--------|----------------------|---------------------|----------|
| `verylowres` | 250K | < 20 MB | Mobile, web |
| `lowres` | 500K | < 50 MB | Lightweight apps |
| `default` | 2M (base) / 5M (dense) | Medium | General purpose |
| `highres` | 9M | Large | High quality (needs `--dense_gaussians`) |
| `veryhighres` | 14M | Very large | Maximum quality (needs `--dense_gaussians`) |

### Quality

MILo produces meshes via Delaunay triangulation of Gaussian pivots + marching tetrahedra with learned SDF. The mesh-in-the-loop training ensures bidirectional consistency between Gaussians and mesh. Produces higher quality meshes with significantly fewer vertices compared to SuGaR/Gaussian Frosting/GOF/RaDe-GS.

### Training Duration

- Default training runs for 18,000 iterations (defined in mesh config `stop_iter: 18000`)
- Mesh regularization starts at iteration 8,001 (`start_iter: 8001`)
- Delaunay triangulation recomputed every 500 iterations (`delaunay_reset_interval: 500`)
- SDF values reset every 500 iterations
- SDF refinement during mesh extraction: 1,000 additional iterations (frozen Gaussians)

Exact wall-clock time depends on GPU, scene size, and mesh config. Expect similar to original 3DGS training (20-60 min on A6000) plus overhead from Delaunay and mesh operations.

---

## 5. Installation Path

**Build from source only.** There is no pip package.

### Installation Steps

```bash
# 1. Clone with submodules
git clone https://github.com/Anttwo/MILo.git --recursive
cd MILo

# 2. Create conda environment
conda create -n milo python=3.9
conda activate milo

# 3. Set CUDA paths
export CPATH=/usr/local/cuda-11.8/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH

# 4. Run installer (installs PyTorch, requirements, all submodules)
python install.py  # --cuda_version 12.1 for CUDA 12.1

# OR manual install (see README section 1 for details)
```

### Docker

No Docker setup is provided in the repository.

### Conda Environment File

No `environment.yml` is provided. The `install.py` script handles everything.

---

## 6. Rasterizer: gsplat vs Original 3DGS

**MILo does NOT use gsplat.** Zero references to gsplat anywhere in the codebase.

MILo ships with three custom CUDA rasterizers, all forked from the original GraphDeco 3DGS rasterizer:

| Rasterizer | Package Name | Origin |
|------------|-------------|--------|
| RaDe-GS (default) | `diff_gaussian_rasterization` | Fork of 3DGS rasterizer with depth/normal support |
| GOF | `diff_gaussian_rasterization_gof` | Gaussian Opacity Fields rasterizer |
| Mini-Splatting2 | `diff_gaussian_rasterization_ms` | Fast densification rasterizer (also provides `SparseGaussianAdam`) |

All three are `torch.utils.cpp_extension.CUDAExtension` packages compiled from CUDA C++. They are incompatible with gsplat's API.

---

## 7. Integration Plan for Gaussian-Toolkit Pipeline

### Option A: Direct COLMAP Pipeline (Recommended)

Feed existing COLMAP datasets directly to MILo training. This gives the best quality since the mesh-in-the-loop regularization is active during training.

```
COLMAP dataset -> MILo train.py -> mesh_extract_sdf.py -> PLY mesh + 3DGS checkpoint
```

### Option B: Post-hoc Mesh Extraction via Functional API

Use `milo.functional` to extract meshes from pre-trained Gaussians (from any source):

```python
# Load pre-trained Gaussians from any PLY
gaussians = load_your_pretrained_gaussians("model.ply")
# Use milo.functional pipeline
mesh = extract_mesh(...)
```

Caveats:
- Quality will be lower than Option A (no training-time mesh consistency)
- Still requires COLMAP cameras for SDF initialization
- Requires MILo's Camera class interface

### Option C: Hybrid (SuGaR replacement)

Replace SuGaR in the pipeline with MILo for the mesh extraction stage, keeping the same COLMAP dataset:

```
Images -> COLMAP -> MILo (train + extract) -> Mesh PLY
                                            -> 3DGS PLY (iteration 18000)
```

### Key Considerations

1. **VRAM**: Default config is moderate. `highres` and `veryhighres` with `--dense_gaussians` need significant GPU memory. Use `--data_device cpu` to reduce VRAM.
2. **License**: Non-commercial only (Gaussian-Splatting License). Cannot be used in production.
3. **CUDA 11.8 lock-in**: Only CUDA 11.8 is tested. CUDA 12.x may work but is unverified.
4. **No gsplat compatibility**: If the pipeline relies on gsplat, MILo cannot be a drop-in replacement for the rasterizer. The functional API is the best integration point.
5. **Same author as SuGaR**: Antoine Guedon authored both SuGaR and MILo. The Blender addon is nearly identical to the SuGaR/Frosting addon.

### Recommended Next Steps

1. Create a conda environment with the exact pinned dependencies
2. Test with an existing COLMAP dataset from the pipeline (Option A)
3. Evaluate mesh quality against SuGaR on the same scene
4. If post-hoc extraction needed, prototype Option B with the functional API
5. Benchmark VRAM usage and training time across mesh configs
