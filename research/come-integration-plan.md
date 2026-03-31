# CoMe Integration Research -- Confidence-Based Mesh Extraction from 3D Gaussians

**Repository**: https://github.com/r4dl/CoMe
**Paper**: "CoMe: Confidence-Based Mesh Extraction from 3D Gaussians" (arXiv:2603.24725)
**Project Page**: https://r4dl.github.io/CoMe/
**Authors**: Lukas Radl*, Felix Windisch*, Andreas Kurz*, Thomas Kohler, Michael Steiner, Markus Steinberger (Graz University of Technology / Huawei Technologies)
**License**: Not yet specified (repo has no LICENSE file as of 2026-03-31)
**Code Status**: Repository is public but contains only README with "coming soon" -- NO CODE RELEASED YET

---

## 1. Critical Finding: Code Not Yet Available

As of 2026-03-31, the GitHub repository at `r4dl/CoMe` exists and is public (41 stars, created 2026-03-23) but contains only a placeholder README. The actual implementation has not been released. The project page also states "Code (coming soon)".

**Implication**: CoMe cannot be integrated into the pipeline today. This document captures all available technical details for future integration once code drops.

---

## 2. What CoMe Is (and Is Not)

### It IS a training method
CoMe trains its own 3D Gaussians from scratch with confidence-aware optimization. It is NOT a post-hoc mesh extraction tool that can be applied to an arbitrary pre-trained 3DGS PLY file.

### Input Requirements
- **Posed images + COLMAP camera parameters** (same as standard 3DGS)
- Standard COLMAP dataset format (images/ + sparse/)
- Cannot accept a standalone `.ply` 3DGS file without the original training data

### What it builds on
- Built directly on the **SOF (Sorted Opacity Fields)** codebase from the same lab (`r4dl/SOF`)
- Uses the original 3DGS rendering pipeline (not gsplat, not nerfstudio)
- Uses `diff-gaussian-rasterization` and `simple-knn` as CUDA submodules

---

## 3. Expected Dependencies (Inferred from SOF)

Since CoMe builds on SOF, the dependency profile will be very similar:

### Core Stack (from SOF environment.yml)

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10 | Conda environment |
| PyTorch | >=2.1 | Flexible; tested with 2.x |
| pytorch-cuda | 12.1 | CUDA toolkit via conda |
| torchvision | (matched) | Via pytorch channel |

### Python Packages (from SOF)

| Package | Notes |
|---------|-------|
| plyfile | PLY I/O |
| open3d | Mesh processing |
| trimesh | Mesh utilities |
| tqdm | Progress bars |
| dacite | Config dataclass parsing |
| ninja | JIT CUDA compilation |
| GPUtil | GPU monitoring |
| opencv-python | Image processing |
| einops | Tensor ops |
| scikit-image | Image metrics |

### CUDA Submodules (from SOF, build-from-source)

| Submodule | Purpose |
|-----------|---------|
| `diff-gaussian-rasterization` | 3DGS rasterizer |
| `simple-knn` | KNN utility |
| `tetra-triangulation` | Marching tetrahedra for mesh extraction (CMake + pip) |
| `fused-ssim` | Fast SSIM computation (from `rahul-goel/fused-ssim`) |

### CoMe-Specific Additions (inferred from paper)
- Custom fused CUDA kernel for appearance embedding (claimed 5x faster than naive PyTorch)
- Confidence-steered densification module
- Color and normal variance loss computation

---

## 4. CUDA/PyTorch Compatibility Assessment

### Can it run with PyTorch 2.3+ and CUDA 11.8?

**Likely not without modifications.** SOF pins `pytorch-cuda=12.1`. The CUDA submodules (`diff-gaussian-rasterization`, `tetra-triangulation`) contain CUDA C++ code compiled against a specific toolkit version. Running against CUDA 11.8 would require:
1. Recompiling all submodules with CUDA 11.8
2. Verifying no CUDA 12.x-specific API usage in the custom kernels
3. Potential ABI compatibility issues

### Local Environment Mismatch

| Component | Our Environment | SOF/CoMe Target |
|-----------|----------------|-----------------|
| Python | 3.14.3 | 3.10 |
| PyTorch | 2.10.0 | >=2.1 |
| CUDA | 13.1 | 12.1 |

Our environment is significantly newer. The submodule C++ code may need updates for Python 3.14 and CUDA 13.1 compatibility. A conda environment with Python 3.10 and CUDA 12.1 is the safest path.

---

## 5. Command-Line Interface (Expected, Based on SOF)

```bash
# Training (SOF pattern, CoMe will be similar)
python train.py --splatting_config configs/<config>.json -s <dataset_path>

# Mesh extraction -- unbounded scenes (Marching Tetrahedra)
python extract_mesh_tets.py -m <model_path>

# Mesh extraction -- bounded scenes (TSDF)
python extract_mesh_tsdf.py -m <model_path>

# Rendering
python render.py -m <model_dir> --skip_train

# Metrics
python metrics.py -m <model_dir>
```

---

## 6. Output Format

### Mesh Extraction
- **Algorithm**: Marching Tetrahedra with binary search (unbounded) or TSDF (bounded)
- **Format**: PLY meshes (based on SOF output pattern)
- **Geometry only**: No texture/color baked onto mesh vertices. The paper focuses entirely on geometric reconstruction quality (F1 scores, Chamfer distance)
- **No UV mapping or texture atlas**

### Implication for Pipeline
If textured meshes are needed, a separate texturing pass would be required (e.g., xatlas UV unwrapping + reprojection from training images).

---

## 7. Performance Benchmarks

All timings on RTX 4090:

| Method | Optimization | Mesh Extraction | Total | F1 Score (T&T) |
|--------|-------------|-----------------|-------|-----------------|
| **CoMe** | **18 min** | **6.6 min** | **~25 min** | **Best reported** |
| MILo | 60 min | 8.6 min | ~69 min | Competitive |
| GOF | 40 min | - | ~40 min | Lower |
| SOF | 17 min | - | ~17 min | Lower than CoMe |
| PGSR | 28 min | - | ~28 min | - |

### Key Claims
- **3x faster than MILo** in total pipeline time
- Optimization time comparable to SOF (18 vs 17 min) with better mesh quality
- Tested on Tanks & Temples, ScanNet++, Mip-NeRF 360, DTU

---

## 8. Technical Architecture

### Confidence Mechanism
- Learnable scalar confidence value per Gaussian primitive
- Rendered confidence map via alpha-blending (same as color rendering)
- Loss: `L_conf = L_rgb * C_hat - beta * log(C_hat)` where beta=0.075
- Effect: Low-confidence Gaussians in hard-to-reconstruct regions get down-weighted, preventing over-densification artifacts

### Confidence-Steered Densification
- Standard 3DGS clones/splits Gaussians with high gradient
- CoMe adds confidence weighting to densification decisions
- Prevents repeated cloning of tiny Gaussians in difficult regions (floaters)

### Additional Losses
- **Color variance loss**: Penalizes blending of Gaussians with different colors at same pixel
- **Normal variance loss**: Penalizes blending of Gaussians with different normals
- **SSIM-decoupled appearance module**: Separates camera-dependent exposure from geometry

---

## 9. Comparison with MILo (Already Researched)

| Aspect | CoMe | MILo |
|--------|------|------|
| Approach | Confidence-weighted 3DGS + marching tets | Mesh-in-the-loop with differentiable rendering |
| Speed | ~25 min total | ~69 min total |
| Input | COLMAP poses + images | COLMAP poses + images |
| Post-hoc on PLY? | No (trains from scratch) | No (trains from scratch) |
| Output | Geometry-only PLY mesh | Geometry-only mesh |
| Textured? | No | No |
| Code available? | **NO (coming soon)** | **YES** |
| Base codebase | SOF / original 3DGS | RaDe-GS / original 3DGS |
| License | Unknown | Gaussian-Splatting (non-commercial) |
| Python | 3.10 (expected) | 3.9 |
| CUDA | 12.1 (expected) | 11.8 |

---

## 10. Integration Recommendations

### Immediate (Now)
1. **Do not plan integration around CoMe** -- code is not available
2. **Use MILo** as the primary mesh extraction path (code available, tested)
3. **Watch the repo** (`gh api repos/r4dl/CoMe/subscription -X PUT -f subscribed=true`) for code release notifications

### When Code Drops
1. **Clone into a conda environment** with Python 3.10, CUDA 12.1 to match SOF's tested config
2. **Test on DTU and Tanks & Temples** to reproduce paper numbers before pipeline integration
3. **Evaluate mesh quality** vs MILo on our specific scenes
4. **Add texturing pass** if needed (CoMe produces geometry only)
5. **Check license** -- SOF uses a custom license, CoMe may follow suit

### Pipeline Position
```
COLMAP poses + images
    |
    v
CoMe training (~18 min on 4090)
    |
    v
Mesh extraction (~7 min)
    |
    v
Geometry-only PLY mesh
    |
    v
[Optional] Texturing pass (xatlas + reprojection)
    |
    v
Textured mesh (OBJ/GLB)
```

### Not Suitable For
- Post-hoc extraction from existing 3DGS models (requires retraining)
- Real-time or interactive mesh generation
- Textured output without additional processing
- pip install (build-from-source with CUDA submodules required)

---

## 11. Answers to Key Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | GitHub repo URL | https://github.com/r4dl/CoMe (public, no code yet) |
| 2 | Accept standard 3DGS PLY directly? | **No.** Trains from scratch; needs COLMAP data |
| 3 | Command-line interface? | Expected: `train.py` + `extract_mesh_tets.py` (based on SOF) |
| 4 | Python/CUDA versions? | Python 3.10, CUDA 12.1 (inferred from SOF). PyTorch >=2.1 |
| 5 | Compatible with PyTorch 2.3+ / CUDA 11.8? | Unlikely without recompilation. Target is CUDA 12.1 |
| 6 | Speed claim? | ~25 min total (18 min train + 7 min extract) on RTX 4090, 3x faster than MILo |
| 7 | Textured meshes? | **No.** Geometry-only PLY |
| 8 | Output formats? | PLY mesh (geometry only) |
| 9 | Post-hoc on any trained 3DGS? | **No.** Must train with confidence framework |
| 10 | Available on pip? | **No.** Build from source with CUDA submodules |

---

## Sources

- [CoMe Project Page](https://r4dl.github.io/CoMe/)
- [CoMe arXiv Paper](https://arxiv.org/abs/2603.24725)
- [CoMe GitHub Repository](https://github.com/r4dl/CoMe)
- [SOF GitHub Repository](https://github.com/r4dl/SOF) (parent codebase)
- [r4dl GitHub Profile](https://github.com/r4dl) (Lukas Radl)
