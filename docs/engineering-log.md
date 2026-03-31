# Engineering Log

Development history and key decisions for the Gaussian Toolkit fork of LichtFeld Studio.

---

## Phase 1: Foundation

### LichtFeld Studio Fork

Forked [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio), a native C++23/CUDA workstation for 3D Gaussian Splatting. LichtFeld provides training, visualization, editing, and export with an MCP server exposing 70+ tools. We chose it over alternatives (gsplat standalone, nerfstudio) because:

- MCP integration allows agentic control from Claude Code
- Native C++ performance for training (not Python-bound)
- Built-in scene graph, selection system, and multi-format export (PLY, SOG, SPZ, HTML, USD)

Established [BOUNDARIES.md](../BOUNDARIES.md) to enforce clean separation: upstream code is never modified on our branch.

### Docker Consolidation

Built a single consolidated Dockerfile (`Dockerfile.consolidated`) on `nvidia/cuda:12.8.1-devel-ubuntu24.04` containing:

- COLMAP 4.1.0 (built from source with METIS/GKlib)
- LichtFeld Studio (host-compiled binary, bind-mounted)
- Python 3.12 pipeline modules
- ComfyUI with SAM3D and FLUX nodes
- Claude Code (Node.js 23)
- Blender (headless)
- ttyd web terminal, VNC, supervisord

Single-command deployment: `docker compose -f docker-compose.consolidated.yml up -d`.

### SplatReady Integration

Integrated the SplatReady plugin for automated video-to-COLMAP pipeline: PyAV frame extraction at configurable FPS, automatic COLMAP feature extraction, exhaustive matching, sparse reconstruction, and undistortion.

---

## Phase 2: TSDF Mesh Extraction

### Initial Approach

After 3DGS training produces a gaussian splat model, we need polygonal meshes for downstream use (game engines, USD scenes, web viewers). First approach: render depth maps from the trained gaussians using gsplat, then fuse them into a mesh via Open3D TSDF.

### Implementation

Built `mesh_extractor.py` using:

1. gsplat to render depth + RGB from training viewpoints
2. Open3D `ScalableTSDFVolume` to fuse depth frames
3. Marching cubes to extract a triangle mesh
4. Vertex colour transfer from the gaussian splat

Results: 22K vertices, 49K faces. Geometric accuracy was acceptable for large structures but poor for fine details.

### Vertex Colours vs Texture Baking

TSDF meshes come with vertex colours, not UV-mapped textures. For web delivery this is sufficient (model-viewer handles vertex colours). For production USD scenes, texture baking is needed. Built `texture_baker.py` skeleton using xatlas for UV unwrapping, but deferred full implementation after discovering the quality ceiling.

### Discovery: TSDF Quality Ceiling

TSDF fusion from expected (rendered) depth has a hard quality ceiling. The depth maps from gaussian splatting are noisy at object boundaries and in regions with sparse training views. No amount of TSDF parameter tuning (voxel size, truncation distance, depth scale) fixes this because the problem is in the input signal, not the fusion algorithm.

---

## Phase 3: Mesh Extraction Research

### Methods Evaluated

| Method | Source | Approach | Finding |
|--------|--------|----------|---------|
| SuGaR | Guédon & Lepetit 2024 | Regularise gaussians to lie on surfaces, then Poisson mesh | Good surface alignment but slow (hours). Requires modified training. |
| GOF (Gaussian Opacity Fields) | Yu et al. 2024 | Learn opacity fields, extract level set | Better than TSDF but still limited by training quality |
| MILo | Wewer et al. SIGGRAPH Asia 2025 | Differentiable mesh-in-the-loop: Delaunay triangulation + learned SDF, mesh participates in the gaussian loss | Best quality. Mesh quality is bounded by gaussian quality. |
| CoMe (Compact Mesh) | various | Mesh compression of gaussians | Targets compression, not reconstruction quality |

### Key Insight: Training Quality is the Bottleneck

MILo produces the best meshes among evaluated methods, but all methods share a common ceiling: **the mesh can only be as good as the trained gaussians**. If the gaussians are noisy (floaters, stretched ellipsoids, missing regions), no mesh extraction method recovers lost geometry.

Root causes of poor gaussian quality in our test scenes:

1. **YouTube-compressed video**: H.264 compression artifacts reduce feature matching quality in COLMAP, producing fewer and less accurate camera poses
2. **Featureless walls**: Large uniform surfaces have no visual features for COLMAP to match, creating holes in the sparse reconstruction
3. **Reflective surfaces**: Glass cases, polished floors, and metallic frames violate the Lambertian assumption in both COLMAP and 3DGS
4. **Insufficient view coverage**: Walk-through videos miss ceiling details and behind-object views

The correct fix is better input capture, not better mesh extraction.

---

## Phase 4: MILo Integration

### CUDA Version Conflict

MILo requires:
- CUDA 11.8 (its CUDA extensions fail to compile with 12.x)
- GCC <= 11 (CUDA 11.8 does not support GCC 12+)
- PyTorch 2.3.1 with cu118

Our main container runs CUDA 12.8 + GCC 14 + Python 3.12. These are fundamentally incompatible. Conda environments were attempted but the CUDA toolkit version is a system-level constraint, not a Python-level one.

### Sidecar Container Solution

Built `docker/Dockerfile.milo` on `nvidia/cuda:11.8.0-devel-ubuntu22.04` with:
- Python 3.10 (Ubuntu 22.04 default)
- PyTorch 2.3.1 + cu118
- All 4 MILo rasterizer variants compiled from source
- nvdiffrast, simple-knn, fused-ssim
- tetra-triangulation (Delaunay, CGAL + pybind11)

The sidecar runs on GPU 1, sleeps until called. The main container invokes it via:
```bash
docker exec milo python3 train.py --source_path /data/output/JOB/colmap ...
```

Shared `/data/output` volume allows both containers to read COLMAP data and write mesh results without network transfer.

### MILo Extractor Module

Built `src/pipeline/milo_extractor.py` to:
1. Check if the `milo` container is running
2. Convert pipeline paths to container-relative paths
3. Call MILo training via `docker exec` with appropriate arguments
4. Monitor progress via log file polling
5. Convert MILo's PLY output to GLB for the web viewer
6. Fall back to TSDF if the sidecar is unavailable

---

## Phase 5: Blender Scene Assembly

### Motivation

The pipeline needs a final assembly step that:
- Imports TSDF or MILo meshes
- Cleans debris (small disconnected components)
- Creates proper materials from vertex colours
- Sets up lighting
- Renders preview images
- Exports a USD scene with proper hierarchy

### Implementation

Built `src/pipeline/blender_assembler.py` to run headless:
```bash
blender --background --python blender_assembler.py -- --input mesh.glb --output-usd scene.usda
```

Uses Blender's Cycles renderer with GPU compute for texture baking. Creates a 3-point lighting setup, imports COLMAP camera poses for aligned preview renders.

---

## Phase 6: Web Interface

### Flask App

Built `src/web/` with:
- Video upload (drag-and-drop, file size validation)
- Job management (create, track, cancel, delete)
- SSE log streaming for real-time pipeline progress
- 3D model preview via Google's `<model-viewer>` web component
- Preview image carousel from Blender renders
- ZIP download of all job outputs
- Anthropic API key management (stored on persistent volume, not in container image)

### SAM3 Object Segmentation

SAM3 (Segment Anything Model 3) provides concept-based segmentation with 4M concepts using text + visual prompts. Requires `HF_TOKEN` environment variable for model downloads from HuggingFace. Falls back to SAM2 grid-point prompts if SAM3 is unavailable.

---

## Current State

The end-to-end pipeline works: video upload through web UI, frame extraction, COLMAP SfM, 3DGS training via LichtFeld MCP, object segmentation, TSDF or MILo mesh extraction, Blender assembly, USD export, and web preview/download.

Primary quality limiter remains the input video. High-quality results require:
- 4K or higher resolution source video
- Slow, deliberate camera motion with overlap
- Multiple passes from different heights/angles
- Avoiding reflective and transparent surfaces
- Good, even lighting

The mesh extraction backend (TSDF vs MILo) matters less than the quality of the trained gaussians, which in turn depends almost entirely on the quality of the input video and COLMAP reconstruction.
