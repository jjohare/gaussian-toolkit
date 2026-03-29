# Tool Catalogue: 28 Tools Assessed

## Summary Matrix

| # | Tool | Stage | Viability | License | Priority | Stars |
|---|------|-------|-----------|---------|----------|-------|
| 1 | [[SplatReady]] | Ingest | 10/10 | MIT | **INSTALLED** | — |
| 2 | [[InstantSplat]] | Ingest | 8/10 | CC BY-SA 4.0 | HIGH | 2,800 |
| 3 | [[MonoGS]] | Ingest | 7/10 | Academic | MEDIUM | 1,600 |
| 4 | [[Gaussian-SLAM]] | Ingest | 6/10 | Academic | LOW | 320 |
| 5 | [[Mip-Splatting]] | Quality | 9/10 | CC BY-SA 4.0 | HIGH | 1,200 |
| 6 | [[Deblurring-3DGS]] | Quality | 7/10 | Unspecified | CONDITIONAL | 260 |
| 7 | [[NVIDIA-PPISP]] | Quality | 8/10 | Apache 2.0 | HIGH | — |
| 8 | [[TRIPS]] | Quality | 4/10 | Unspecified | LOW | — |
| 9 | [[Gaussian-Grouping]] | Segmentation | 9/10 | Apache-2.0 | **CRITICAL** | 968 |
| 10 | [[SAGA]] | Segmentation | 8/10 | Apache-2.0 | **CRITICAL** | 934 |
| 11 | [[OpenGaussian]] | Segmentation | 7/10 | Unspecified | HIGH | 191 |
| 12 | [[LangSplat]] | Semantics | 8/10 | Unspecified | HIGH | 1,000 |
| 13 | [[LEGaussians]] | Semantics | 7/10 | Unspecified | MEDIUM | 300 |
| 14 | [[Feature-3DGS]] | Semantics | 8/10 | Unspecified | HIGH | 637 |
| 15 | [[SuGaR]] | Meshing | 9/10 | Unspecified | **CRITICAL** | 3,300 |
| 16 | [[CoMe]] | Meshing | 6/10 | Unspecified | WATCH | — |
| 17 | [[MeshSplatting]] | Meshing | 7/10 | Apache-2.0 | HIGH | 507 |
| 18 | [[mesh2splat-EA]] | Conversion | 7/10 | Open Source | MEDIUM | 624 |
| 19 | [[GaussianEditor]] | Editing | 5/10 | S-Lab (NC) | LOW | 1,400 |
| 20 | [[RL3DEdit]] | Editing | 7/10 | MIT | MEDIUM | 192 |
| 21 | [[DreamGaussian]] | Generation | 8/10 | MIT | MEDIUM | 4,300 |
| 22 | [[4D-Gaussians]] | Dynamic | 6/10 | Apache-2.0 | LOW | 3,500 |
| 23 | [[4DGS-Video-Gen]] | Dynamic | 4/10 | MIT/NC model | SKIP | 137 |
| 24 | [[gsgen]] | Generation | 5/10 | MIT | LOW | 843 |
| 25 | [[TriplaneGaussian]] | Generation | 5/10 | MIT | LOW | 600 |
| 26 | [[Flux2-Repair-LoRA]] | Repair | 4/10 | CC BY-NC 4.0 | SKIP | — |
| 27 | [[Qwen-Gauss-Splash]] | Repair | 5/10 | Apache 2.0 | LOW | — |
| 28 | [[GOF-SOF]] | Meshing | 9/10 | — | **CRITICAL** | — |
| 29 | [[SAM3]] | Segmentation | 10/10 | Apache-2.0 | **CRITICAL** | — |
| 30 | [[SAM3.1-Object-Multiplex]] | Segmentation | 9/10 | Apache-2.0 | HIGH | — |
| 31 | [[Hunyuan3D-2.0]] | Meshing | 9/10 | Tencent | **CRITICAL** | — |

## Critical Path Tools (Must-Have)

### 1. Gaussian Grouping — Joint Reconstruction + Segmentation
- **Repo**: https://github.com/lkeab/gaussian-grouping
- **Paper**: ECCV 2024 (ETH Zurich)
- **License**: Apache-2.0
- **What**: Augments each Gaussian with compact Identity Encoding during training. Supervised by SAM 2D masks with 3D spatial consistency regularisation.
- **Output**: Per-Gaussian identity labels grouping into object instances
- **Why Critical**: Only Apache-licensed method that produces per-object labels during training. Enables direct per-label extraction without post-hoc clustering.
- **Integration**: Replace standard 3DGS training step. SAM mask generation is automatic (~15 min preprocessing).

### 2. SAGA — Post-hoc Interactive Segmentation
- **Repo**: https://github.com/Jumpat/SegAnyGAussians
- **Paper**: AAAI 2025
- **License**: Apache-2.0
- **What**: Trains contrastive affinity features on pre-trained 3DGS. Interactive point-prompt segmentation in 4ms per query.
- **Why Critical**: Works on ANY pre-trained 3DGS model. Interactive refinement complements Gaussian Grouping's automatic labels.

### 3. SuGaR — Mesh Extraction with UV Textures
- **Repo**: https://github.com/Anttwo/SuGaR
- **Paper**: CVPR 2024
- **What**: Surface-aligned Gaussians + Poisson reconstruction. Produces OBJ meshes with UV-mapped diffuse textures.
- **Why Critical**: Only method producing UV-mapped textured meshes from Gaussians. Direct USD compatibility.

### 4. SOF/GOF — High-Quality Geometry Extraction
- **Repos**: https://github.com/r4dl/SOF, https://github.com/autonomousvision/gaussian-opacity-fields
- **Papers**: SIGGRAPH Asia 2025, SIGGRAPH Asia 2024
- **What**: Marching Tetrahedra on Gaussian opacity fields. SOF is 10x faster than GOF.
- **Why Critical**: Best geometric accuracy for mesh extraction. Portable to C++/CUDA.

### 5. SAM3 — Concept Segmentation with 4M Concepts
- **Paper**: Segment Anything Model 3 (Meta, 2025)
- **License**: Apache-2.0
- **Viability**: 10/10
- **What**: Segment anything with text+visual prompts across 4M concepts. No prompt engineering needed — describe objects in natural language.
- **Output**: Per-frame instance masks with concept labels
- **Why Critical**: Replaces SAM2 grid-point prompts with semantic concept prompts. Handles complex scenes without manual prompt placement. SAM3.1 adds Object Multiplex for multi-object tracking across video frames.
- **Integration**: Drop-in replacement for SAM2 in `sam2_segmentor.py`. Client built as `sam3d_client.py`. Running in consolidated Docker via ComfyUI SAM3D nodes.

### 6. Hunyuan3D 2.0 — Multi-View to Textured Mesh
- **Repo**: https://github.com/Tencent/Hunyuan3D-2
- **License**: Tencent
- **Viability**: 9/10
- **What**: Takes multi-view renders of an object and produces a textured polygonal mesh with UV maps. Works on isolated objects extracted from the Gaussian scene.
- **Output**: Textured OBJ/GLB mesh with UV-mapped diffuse textures
- **Why Critical**: Produces higher-quality meshes than TSDF or Marching Cubes for individual objects. Handles complex geometry that volumetric methods struggle with.
- **Integration**: Client built as `hunyuan3d_client.py`. Multi-view renders generated by `multiview_renderer.py`. Runs on GPU 0 in consolidated Docker.

## Repository Links

### Segmentation
- Gaussian Grouping: https://github.com/lkeab/gaussian-grouping
- SAGA: https://github.com/Jumpat/SegAnyGAussians
- OpenGaussian: https://github.com/yanmin-wu/OpenGaussian
- LangSplat: https://github.com/minghanqin/LangSplat
- LEGaussians: https://github.com/buaavrcg/LEGaussians
- Feature 3DGS: https://github.com/ShijieZhou-UCLA/feature-3dgs

### Mesh Extraction
- SuGaR: https://github.com/Anttwo/SuGaR
- GOF: https://github.com/autonomousvision/gaussian-opacity-fields
- SOF: https://github.com/r4dl/SOF
- MeshSplatting: https://github.com/meshsplatting/mesh-splatting
- GS2Mesh: https://github.com/yanivw12/gs2mesh
- MILo: https://github.com/Anttwo/MILo
- 2DGS: https://github.com/hbb1/2d-gaussian-splatting
- PGSR: https://github.com/zju3dv/PGSR

### Scene Editing
- RL3DEdit: https://github.com/AMAP-ML/RL3DEdit
- GaussianEditor: https://github.com/buaacyw/GaussianEditor
- DreamGaussian: https://github.com/dreamgaussian/dreamgaussian

### Reconstruction
- InstantSplat: https://github.com/NVlabs/InstantSplat
- MonoGS: https://github.com/muskie82/MonoGS
- Mip-Splatting: https://github.com/autonomousvision/mip-splatting
- Deblurring 3DGS: https://github.com/benhenryL/Deblurring-3D-Gaussian-Splatting

### Utility
- mesh2splat (EA): https://github.com/electronicarts/mesh2splat
- gsgen: https://github.com/gsgen3d/gsgen
- 4D Gaussians: https://github.com/hustvl/4DGaussians
- TriplaneGaussian: https://github.com/VAST-AI-Research/TriplaneGaussian
