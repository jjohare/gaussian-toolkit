# Research: Video to Structured 3D USD Scene

## Objective

Reconstruct 3D polygonal scenes in USD format from video, via Gaussian Splatting. Identify all objects in the scene, isolate them, reconstruct each as individual textured 3D meshes with metadata, and assemble into a hierarchical scene graph.

## Project Scale

- **21 pipeline modules** in `src/pipeline/`
- **5 web interface files** in `src/web/` (upload UI on port 7860)
- **15 research docs** across `research/`
- **25+ git commits** on the `gaussian-toolkit` branch
- **Consolidated Docker** running on remote (192.168.2.48:7860 web, :8188 ComfyUI)
- **33 per-object PLY files** extracted from gallery tour
- **USD scene with 59 prims** assembled
- **TSDF mesh: 22K vertices, 49K faces**
- **SAM3 upgrade in progress** (from SAM2)
- **Hunyuan3D 2.0** multi-view integration built
- **FLUX inpainting client** built
- **Web upload interface** running on :7860

## Pipeline Status

### Complete

| Component | Module | Status |
|-----------|--------|--------|
| Video frame extraction | `orchestrator.py` (PyAV) | Tested, working |
| COLMAP SfM (feature extract, match, sparse, undistort) | `orchestrator.py` + COLMAP 4.1.0 | Tested, working |
| 3DGS Training via LichtFeld MCP | `mcp_client.py` | Tested, 7k iter in 2m15s |
| SAM2 2D segmentation | `sam2_segmentor.py` | Tested, 13 frames in 46s |
| Mask projection (2D masks to 3D Gaussians) | `mask_projector.py` | Tested, 98.3% coverage |
| Mesh extraction (Marching Cubes + TSDF) | `mesh_extractor.py` | Tested, TSDF: 22K verts / 49K faces |
| Mesh cleaning (decimation, hole fill) | `mesh_cleaner.py` | Tested |
| USD scene assembly | `usd_assembler.py` | Tested, 59 prims, variant sets |
| Quality gates (per-stage pass/fail) | `quality_gates.py` | Tested |
| CLI entry point | `cli.py` + `__main__.py` | Working |
| Pipeline configuration | `config.py` | YAML/dict based |
| Coordinate transforms | `coordinate_transform.py` | COLMAP <-> 3DGS <-> USD |
| Frame quality scoring | `frame_quality.py` | Blur/exposure filtering |
| MCP bridge script | `scripts/lichtfeld_mcp_bridge.py` | Working |
| Hardware tracing | `scripts/hardware_trace.py` | GPU/RAM/CPU logging |
| Object separation | `scripts/run_object_separation.py` | 33 objects, 98.3% coverage |
| TSDF mesh extraction | `scripts/run_tsdf_mesh.py` | 22K verts, 49K faces |
| USD gallery assembly | `scripts/assemble_gallery_usd.py` | 59 prims |
| Multi-view renderer | `multiview_renderer.py` | Camera orbit renders |
| Hunyuan3D 2.0 client | `hunyuan3d_client.py` | Multi-view to textured mesh |
| FLUX inpainting client | `comfyui_inpainter.py` | Background recovery via ComfyUI |
| Web upload interface | `src/web/app.py` | Flask on :7860 |
| Consolidated Docker | `Dockerfile.consolidated` | Dual RTX 6000 Ada, all services |

### In Progress

| Component | Module | Status | Blocker |
|-----------|--------|--------|---------|
| SAM3 upgrade | `sam3d_client.py` | Client built, upgrading from SAM2 | SAM3 model integration in consolidated Docker |
| Texture baking | `texture_baker.py` | Skeleton written | Depends on clean mesh extraction |
| Material assignment | `material_assigner.py` | Skeleton written | Depends on texture baking |
| COLMAP output parsing | `colmap_parser.py` | Basic binary reader | Needs robust error handling for malformed models |

### Known Issues

1. **COLMAP sparse reconstruction is the bottleneck** -- ~20 minutes on 32 cores for 15 frames. No GPU acceleration available for the sparse BA solver. Workaround: use fewer frames or switch to incremental mapper.

2. **SAM2 prompt strategy** -- Currently using grid-point prompts. Quality depends heavily on prompt placement. SAM3 upgrade will replace this with text+visual concept prompts (4M concepts, no prompt engineering needed).

3. **Mask projection noise on thin geometry** -- The Gaussian-space voting from 2D masks produces noisy labels on thin structures (branches, wires). Depth-weighted voting and multi-view consistency checks are needed.

4. **Mesh extraction produces non-manifold geometry** -- Marching Cubes on the Gaussian density field can produce self-intersecting faces. TSDF fusion now available as alternative (22K verts, 49K faces).

5. **No texture UV unwrapping** -- Meshes are vertex-coloured only. xatlas is installed but not integrated for proper UV unwrapping and texture baking.

6. **USD variant sets are placeholder** -- The assembler creates Gaussian and Mesh variant sets but the Gaussian variant currently stores only a path reference, not embedded splat data.

7. **SAM3 integration pending** -- SAM3 client built (`sam3d_client.py`) but full integration with the consolidated Docker container still in progress.

## Target Pipeline

```
Video -> Frames -> COLMAP SfM -> 3DGS Training -> SAM3 Concept Segmentation
    -> Per-Object Gaussian Extraction -> Hunyuan3D 2.0 Mesh Creation
    -> Background Inpainting (FLUX) -> USD Scene Assembly (variant sets)
```

## Research Structure

```
research/
├── README.md                          # This file
├── landscape/
│   ├── tool-catalogue.md              # 31 tools assessed with viability scores
│   ├── segmentation-methods.md        # 3D Gaussian segmentation SOTA
│   ├── mesh-extraction-methods.md     # Gaussian-to-mesh conversion SOTA
│   └── field-overview.md              # Landscape synthesis and gap analysis
├── pipelines/
│   ├── proposed-pipeline.md           # Recommended end-to-end architecture
│   └── alternative-pipelines.md       # Alternative approaches considered
├── components/
│   ├── hunyuan3d-integration.md       # Hunyuan3D 2.0 multi-view mesh creation
│   ├── inpainting-recovery.md         # Background recovery via diffusion
│   └── quality-control.md            # Agent quality decision trees
├── references/
│   ├── existing-capabilities.md       # What LichtFeld/COLMAP already provide
│   └── (papers.md, repos.md)         # Academic references
└── decisions/
    ├── prd.md                         # Product Requirements Document
    ├── prd-consolidated-docker.md     # Consolidated Docker PRD
    ├── adr-001-pipeline-architecture.md
    └── ddd-domain-model.md
```

## Key Findings

### Critical Path

The current pipeline uses these core components:

1. **SplatReady** (installed) -- Video to COLMAP dataset
2. **SAM2/SAM3** (SAM2 validated, SAM3 upgrading) -- 2D segmentation with concept prompts
3. **Mask Projection** (validated) -- 2D masks to 3D Gaussian labels, 98.3% coverage
4. **TSDF Mesh Extraction** (validated) -- Open3D TSDF fusion, 22K verts / 49K faces
5. **Hunyuan3D 2.0** (client built) -- Per-object multi-view to textured mesh
6. **FLUX Inpainting** (client built) -- Background recovery via ComfyUI
7. **USD Assembly** (validated) -- 59-prim hierarchical scene with variant sets

### Reconstruct-Then-Segment Validated

Evidence strongly favours **reconstruct-then-segment** for our use case:
- SAM2 mask projection achieved 98.3% Gaussian coverage with 33 objects
- Post-hoc segmentation allows quality gating before decomposition
- SAM3 will improve this further with text+visual concept prompts (4M concepts)

### Hybrid Approach (Current)

1. Train full scene 3DGS (7k iter, 2m15s, 1M gaussians)
2. SAM2/SAM3 segmentation on training views
3. Project 2D masks onto 3D Gaussians (98.3% coverage)
4. Extract per-object PLY files (33 objects)
5. Per-object Hunyuan3D 2.0 mesh creation (multi-view to textured mesh)
6. Inpaint removed objects from training views via FLUX/ComfyUI
7. Assemble multi-object USD scene with variant sets (59 prims)

### Gap Analysis

| Capability | Status | Primary Tool |
|-----------|--------|--------------|
| Video to Frames | Complete | SplatReady / PyAV |
| COLMAP SfM | Complete | COLMAP 4.1.0 |
| 3DGS Training | Complete | LichtFeld Studio MCP |
| Object Segmentation | Complete (SAM2), Upgrading (SAM3) | SAM2 + mask projection / SAM3 |
| TSDF Mesh Extraction | Complete | Open3D TSDF fusion |
| Per-Object Mesh | **In Progress** | Hunyuan3D 2.0 |
| Background Inpainting | **Built** | ComfyUI + FLUX |
| Texture Baking | **In Progress** | xatlas + custom baker |
| USD Assembly | Complete | OpenUSD Python (59 prims) |
| Agentic Orchestration | Complete | LichtFeld MCP (70+ tools) |
| Web Interface | Complete | Flask on :7860 |
| Consolidated Docker | Complete | Dual RTX 6000 Ada |
