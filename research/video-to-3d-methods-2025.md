# Video to Structured 3D USD Scenes: Methods Survey (2024-2025)

Research compiled: 2026-03-30

Pipeline target: Walkthrough Video -> COLMAP SfM -> 3D Gaussian Splatting -> Per-Object Segmentation -> Mesh Extraction -> Hierarchical USD Scene

---

## 1. 3D Gaussian Splatting + Object Segmentation

Methods that attach semantic/instance features to individual Gaussians, enabling per-object extraction.

| Method | Venue | Code Available | Per-Object Gaussians | Approach | GitHub |
|--------|-------|---------------|---------------------|----------|--------|
| **SAGA** (Segment Any 3D Gaussians) | AAAI 2025 | Yes | Yes | Scale-gated affinity features distilled from SAM; 4ms segmentation with 2D prompts | [Jumpat/SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians) |
| **Gaussian Grouping** | ECCV 2024 | Yes | Yes | Identity Encoding per Gaussian; lifts 2D SAM masks to 3D; open-world instance/stuff grouping | [lkeab/gaussian-grouping](https://github.com/lkeab/gaussian-grouping) |
| **OpenGaussian** | NeurIPS 2024 | Yes | Yes | Point-level open-vocabulary understanding; SAM masks + CLIP embeddings; codebook discretization | [yanmin-wu/OpenGaussian](https://github.com/yanmin-wu/OpenGaussian) |
| **LEGaussians** | CVPR 2024 | Yes | Partial (language queries) | Language-embedded Gaussians with quantized CLIP features; open-vocabulary spatial queries | [buaavrcg/LEGaussians](https://github.com/buaavrcg/LEGaussians) |
| **LangSplat** | CVPR 2024 | Yes | Partial (language queries) | Compressed CLIP features + SAM hierarchical supervision; precise spatial language queries | [minghanqin/LangSplat](https://github.com/minghanqin/LangSplat) |
| **SAGS** (Segment Anything in 3D Gaussians) | 2024 | Yes | Yes | Alternative SAM-based 3DGS segmentation | [XuHu0529/SAGS](https://github.com/XuHu0529/SAGS) |
| **SuperGSeg** | 2025 | Yes | Yes | Neural Super-Gaussians with hierarchical segmentation; 2D language features lifted to 3D | [supergseg.github.io](https://supergseg.github.io/) |
| **SAM2Object** | CVPR 2025 | Yes | Yes (zero-shot) | SAM2-based zero-shot 3D instance segmentation; cross-view consistency consolidation | [jihuaizhaohd/SAM2Object](https://github.com/jihuaizhaohd/SAM2Object) |

### Recommendation for Pipeline

**SAGA** and **Gaussian Grouping** are the most mature for extracting per-object Gaussian subsets. SAGA provides interactive GUI-based segmentation and is efficient (4ms per query). Gaussian Grouping produces clean instance groupings suitable for downstream mesh extraction. **OpenGaussian** adds open-vocabulary capability at the point level, which is valuable for automated labeling in a USD scene graph.

---

## 2. Gaussian-to-Mesh Extraction Methods

Methods that convert 3D Gaussian representations into polygonal meshes, ideally with UV textures.

| Method | Venue | Code Available | UV Textures | Mesh Quality | Approach | GitHub |
|--------|-------|---------------|-------------|-------------|----------|--------|
| **SuGaR** | CVPR 2024 | Yes | Yes (via Gaussian rendering on mesh) | High; Poisson reconstruction preserves detail | Regularizes Gaussians to align with surface; Poisson mesh extraction; Gaussian-on-mesh rendering | [Anttwo/SuGaR](https://github.com/Anttwo/SuGaR) |
| **2DGS** (2D Gaussian Splatting) | SIGGRAPH 2024 | Yes | No (requires post-processing) | Medium; tends to over-smooth surfaces | Collapses 3D Gaussians to 2D disks; improved geometric accuracy but less fine detail | [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) |
| **GOF** (Gaussian Opacity Fields) | SIGGRAPH Asia 2024 | Yes | No (mesh only) | High; adaptive Marching Tetrahedra extraction | Level-set identification in opacity field; regularization for completeness | [autonomousvision/gaussian-opacity-fields](https://github.com/autonomousvision/gaussian-opacity-fields) |
| **PGSR** | 2024 | Yes | No | High for indoor; multi-view geometric consistency | Planar-based Gaussians; enforces cross-view appearance and geometry consistency | [zju3dv/PGSR](https://github.com/zju3dv/PGSR) |
| **GS2Mesh** | ECCV 2024 | Yes | No (OBJ mesh) | High; uses stereo prior for smooth surfaces | Pre-trained stereo model as geometric prior; works on noisy Gaussian clouds | [yanivw12/gs2mesh](https://github.com/yanivw12/gs2mesh) |
| **NeuSG** | 2024 | Yes | No | High; combines neural implicit + Gaussian guidance | SDF surface guided by Gaussian spatial distribution; volume rendering | Paper: arXiv |
| **3DGSR** | TOG 2024 | Yes | No | High; implicit surface with Gaussian constraints | Implicit surface reconstruction with 3DGS; unified optimization | Paper: arXiv |
| **Gaussian Surfels** | NeurIPS 2024 | Yes | No | Medium-High | Disk-like Gaussians for surface modeling; normal consistency | [turandai/gaussian_surfels](https://github.com/turandai/gaussian_surfels) |

### Benchmark Comparisons (DTU Dataset, Chamfer Distance -- lower is better)

| Method | CD (mm) | Notes |
|--------|---------|-------|
| Sparse2DGS (3 views) | 1.13 | Best sparse-view result |
| PGSR (3 views) | 2.08 | Good multi-view consistency |
| 2DGS (3 views) | 2.81 | Over-smoothing hurts detail |
| GOF (3 views) | 2.82 | Comparable to 2DGS |

### Recommendation for Pipeline

**SuGaR** is the strongest choice for the target pipeline because it is the only method that directly produces textured meshes with Gaussian-on-mesh rendering (enabling UV baking). For highest geometric accuracy, **GOF** with Marching Tetrahedra produces clean adaptive meshes. A practical workflow: use **SuGaR** for textured mesh extraction per segmented object, or use **GOF/GS2Mesh** for geometry and apply texture baking as a post-process.

---

## 3. End-to-End Pipelines: Video to Scene Graph with USD Output

No single open-source tool currently provides a complete video-to-USD pipeline with per-object textured meshes. The workflow must be assembled from components:

### Proposed Pipeline Architecture

```
Video Frames
    |
    v
COLMAP (SfM + MVS)  -->  Camera poses + sparse point cloud
    |
    v
gsplat / 3DGS Training  -->  Trained Gaussian scene
    |
    v
SAGA / Gaussian Grouping  -->  Per-object Gaussian subsets
    |
    v
SuGaR / GOF (per object)  -->  Individual textured meshes (.obj/.ply)
    |
    v
USD Assembly Script  -->  Hierarchical .usda/.usdc scene
```

### Existing Tools and Their USD Capabilities

| Tool | USD Export | Notes |
|------|-----------|-------|
| **NVIDIA Omniverse** | Native USD | Can import meshes and assemble USD scenes; Connector SDK available |
| **Blender (3.5+)** | USD export plugin | Import OBJ/PLY meshes, arrange scene hierarchy, export as .usda/.usdc |
| **Houdini** | Native USD (Solaris) | Full USD pipeline with LOP network; industry standard |
| **Maya** | USD plugin (mayaUsd) | Robust USD export with material assignment |
| **OpenUSD Python API** | Native | `pxr.Usd`, `pxr.UsdGeom` for programmatic scene assembly |
| **Nerfstudio** | No USD export | Exports .ply (splats) and .obj (meshes via TSDF/Poisson) |
| **LichtFeld Studio** | No USD export (yet) | Exports Gaussian .ply; mesh export requires external tools |

### Programmatic USD Assembly

The OpenUSD Python API (`pip install usd-core`) enables direct scene graph construction:

```python
from pxr import Usd, UsdGeom, UsdShade, Sdf

stage = Usd.Stage.CreateNew("scene.usda")
root = UsdGeom.Xform.Define(stage, "/World")

# Per-object mesh reference
for obj_name, mesh_path in objects.items():
    xform = UsdGeom.Xform.Define(stage, f"/World/{obj_name}")
    mesh_ref = stage.DefinePrim(f"/World/{obj_name}/mesh")
    mesh_ref.GetReferences().AddReference(mesh_path)

stage.Save()
```

---

## 4. SAM / SAM2 / SAM3 in 3D Scene Decomposition

### 2D-to-3D Mask Lifting Approaches

| Method | Approach | Consistency | GitHub |
|--------|----------|-------------|--------|
| **SAM2 + Multi-View Projection** | Track object in video with SAM2; project masks onto 3D Gaussians via known camera poses | Frame-to-frame via SAM2 memory mechanism | Meta SAM2 repo |
| **SAM2Object** (CVPR 2025) | Zero-shot 3D instance segmentation; consolidates SAM2 view consistency with 3D geometric priors | High cross-view consistency | [jihuaizhaohd/SAM2Object](https://github.com/jihuaizhaohd/SAM2Object) |
| **SAM2Point** | Adapts SAM2 for zero-shot 3D segmentation; supports 3D point/box/mask prompts | Generalizes across indoor/outdoor/LiDAR | [ZiyuGuo99/SAM2Point](https://github.com/ZiyuGuo99/SAM2Point) |
| **SAGA (internal)** | Distills SAM into per-Gaussian affinity features during training | Learned 3D consistency | [Jumpat/SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians) |
| **Gaussian Grouping** | Lifts 2D SAM masks to 3D via identity encoding optimization | Jointly optimized with reconstruction | [lkeab/gaussian-grouping](https://github.com/lkeab/gaussian-grouping) |

### SAM3 (2025)

SAM3 is a unified model for detection, segmentation, and tracking in images and video using both text and visual prompts. SAM3D extends this to single-image 3D reconstruction, producing meshes or point clouds from segmented regions. This is relevant for object-level reconstruction but less so for multi-view scene decomposition.

### SOTA for Consistent 3D Segmentation from Video

The current state of the art combines:

1. **SAM2** for temporally consistent 2D video segmentation (memory-based tracking across frames)
2. **Multi-view aggregation** to accumulate mask votes across camera views onto 3D Gaussians
3. **Feature distillation** (SAGA, OpenGaussian) to bake segmentation capability directly into the 3D representation

**SAM2Object** (CVPR 2025) represents the most recent advance, achieving zero-shot 3D instance segmentation with robust cross-view consistency.

---

## 5. Scene Export: LichtFeld Studio, Nerfstudio, gsplat

### LichtFeld Studio

- **Repository**: [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio)
- **Current exports**: Gaussian .ply files
- **Mesh export**: Not built-in; requires external tools (SuGaR, GOF, etc.)
- **Key features**: COLMAP integration, MCMC optimization, bilateral grid appearance, 3DGUT distorted camera support, pose optimization, Python plugin system, MCP automation
- **Pipeline role**: Training and scene editing; output Gaussians feed into segmentation and mesh extraction

### Nerfstudio

- **Repository**: [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- **Export commands**:
  - `ns-export gaussian-splat` -- exports .ply Gaussian splats
  - `ns-export pointcloud` -- exports .ply point clouds
  - `ns-export tsdf` -- TSDF fusion mesh extraction (.obj)
  - `ns-export poisson` -- Poisson surface reconstruction (.obj), highest quality built-in mesh
- **Mesh quality**: TSDF and Poisson are general-purpose; not as high quality as dedicated methods (SuGaR, GOF)
- **No USD export**: Outputs .ply and .obj only

### gsplat

- **Repository**: [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)
- **Role**: CUDA-accelerated Gaussian splatting rasterization library
- **Export**: Provides the rendering backend; does not directly export meshes
- **Integration**: Used as the rasterization engine by Nerfstudio's Splatfacto and other methods

### glTF 3DGS Standard (August 2025)

Khronos officially added 3D Gaussian Splatting to the glTF ecosystem via the `KHR_gaussian_splatting` extension, enabling interoperable exchange of Gaussian splat data across tools and viewers.

---

## 6. Consolidated Comparison Table

### Segmentation Methods (for per-object Gaussian extraction)

| Method | Code | Per-Object | Open-Vocab | Speed | Best For |
|--------|------|-----------|------------|-------|----------|
| SAGA | [GitHub](https://github.com/Jumpat/SegAnyGAussians) | Yes | Via prompts | 4ms/query | Interactive segmentation |
| Gaussian Grouping | [GitHub](https://github.com/lkeab/gaussian-grouping) | Yes | Yes (open-world) | Training-time | Automatic scene decomposition |
| OpenGaussian | [GitHub](https://github.com/yanmin-wu/OpenGaussian) | Yes | Yes (CLIP) | Moderate | Language-driven selection |
| LEGaussians | [GitHub](https://github.com/buaavrcg/LEGaussians) | Partial | Yes | Fast queries | Spatial language queries |
| SAM2Object | [GitHub](https://github.com/jihuaizhaohd/SAM2Object) | Yes | Zero-shot | Moderate | Zero-shot 3D instances |

### Mesh Extraction Methods (for converting Gaussians to polygonal meshes)

| Method | Code | UV Textures | Quality (DTU CD) | Speed | Best For |
|--------|------|-------------|-------------------|-------|----------|
| SuGaR | [GitHub](https://github.com/Anttwo/SuGaR) | Yes | Good | ~30min | Textured mesh output |
| GOF | [GitHub](https://github.com/autonomousvision/gaussian-opacity-fields) | No | Good | Moderate | Clean adaptive meshes |
| 2DGS | [GitHub](https://github.com/hbb1/2d-gaussian-splatting) | No | 2.81mm | Fast | Quick geometry extraction |
| PGSR | [GitHub](https://github.com/zju3dv/PGSR) | No | 2.08mm | Moderate | Indoor/architectural |
| GS2Mesh | [GitHub](https://github.com/yanivw12/gs2mesh) | No | Good | Moderate | Smooth surface recovery |
| Gaussian Surfels | [GitHub](https://github.com/turandai/gaussian_surfels) | No | Medium-High | Fast | Normal-consistent surfaces |

---

## 7. Recommended Pipeline for This Project

### Phase 1: Reconstruction
1. **COLMAP** for SfM from video frames
2. **gsplat** (via LichtFeld Studio or Nerfstudio) for 3DGS training

### Phase 2: Segmentation
3. **Gaussian Grouping** for automatic per-object Gaussian extraction (preferred for automation)
4. **SAGA** for interactive refinement if needed
5. **SAM2** as the underlying 2D segmentation backbone

### Phase 3: Mesh Extraction
6. **SuGaR** per object for textured mesh extraction (produces .obj with texture)
7. Alternative: **GOF** for geometry + separate texture baking

### Phase 4: USD Assembly
8. **OpenUSD Python API** to assemble per-object meshes into hierarchical .usda scene
9. **Blender** (USD plugin) or **NVIDIA Omniverse** for visual validation and material refinement

### Key Gaps to Address
- No tool produces USD directly from Gaussians; assembly script required
- Per-object SuGaR extraction needs automation (run SuGaR on each Gaussian subset)
- Material/PBR properties need manual or AI-assisted assignment post mesh extraction
- Scale and coordinate system alignment between tools needs careful handling

---

## Sources

- [Gaussian Grouping - GitHub (ECCV 2024)](https://github.com/lkeab/gaussian-grouping)
- [SAGA - SegAnyGAussians (AAAI 2025)](https://github.com/Jumpat/SegAnyGAussians)
- [OpenGaussian (NeurIPS 2024)](https://github.com/yanmin-wu/OpenGaussian)
- [LEGaussians (CVPR 2024)](https://github.com/buaavrcg/LEGaussians)
- [SuGaR (CVPR 2024)](https://github.com/Anttwo/SuGaR)
- [GOF - Gaussian Opacity Fields (SIGGRAPH Asia 2024)](https://github.com/autonomousvision/gaussian-opacity-fields)
- [GS2Mesh (ECCV 2024)](https://github.com/yanivw12/gs2mesh)
- [SAM2Object (CVPR 2025)](https://github.com/jihuaizhaohd/SAM2Object)
- [SAM2Point](https://github.com/ZiyuGuo99/SAM2Point)
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio)
- [Nerfstudio - gsplat](https://github.com/nerfstudio-project/gsplat)
- [Nerfstudio Export Geometry Docs](https://docs.nerf.studio/quickstart/export_geometry.html)
- [Awesome 3DGS Applications Survey](https://github.com/heshuting555/Awesome-3DGS-Applications)
- [Awesome 3D Gaussian Splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)
- [OpenUSD Documentation](https://openusd.org/dev/intro.html)
- [SAM3 Overview](https://www.edge-ai-vision.com/2025/11/sam3-a-new-era-for-open-vocabulary-segmentation-and-edge-ai/)
- [SAGS - Segment Anything in 3D Gaussians](https://github.com/XuHu0529/SAGS)
- [Gaussian Surfels](https://github.com/turandai/gaussian_surfels)
- [Segment Any 3D Gaussians - arXiv](https://arxiv.org/abs/2312.00860)
- [SuperGSeg](https://supergseg.github.io/)
- [Surface Reconstruction Survey (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453780/)
