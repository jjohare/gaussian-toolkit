# PRD: Agentic Video-to-Scene Pipeline

## Product Vision

Transform video recordings into structured, editable 3D USD scenes containing individually addressable objects as textured polygonal meshes, orchestrated by an AI agent swarm.

## Problem Statement

Current tools can reconstruct video as monolithic 3D Gaussian Splat scenes but cannot decompose them into individual objects with standard mesh representations. Artists and automated systems need structured scene graphs with separate, manipulable objects — not opaque point clouds.

## Target Users

1. **3D artists**: Import real-world scenes into DCC tools (Blender, Maya, Houdini) with editable objects
2. **Game developers**: Capture real environments as game-ready USD scenes
3. **Digital twin operators**: Reconstruct facilities with individually addressable assets
4. **VFX studios**: Pre-vis and set extension from location captures

## Success Criteria

| Metric | Target |
|--------|--------|
| Input | Any video file (MP4, MOV) from phone, drone, or camera |
| Output | USD scene with individual textured mesh objects |
| Object detection | 80%+ of visible objects identified and extracted |
| Mesh quality | PSNR > 25 dB round-trip (mesh → gaussian → render vs original) |
| Automation | Fully agentic — no human intervention required for standard scenes |
| Runtime | < 2 hours for a 60-second video on single GPU |
| Formats | USD primary, with PLY/OBJ/glTF secondary export |

## Non-Goals (v1)

- Dynamic/animated scene reconstruction
- Real-time processing
- Mobile deployment
- Texture PBR (metallic/roughness/normal) — diffuse only for v1
- Semantic relationship inference ("chair is at table")
- Scene completion (filling unseen areas)

## Functional Requirements

### FR-1: Video Ingestion
- Accept MP4, MOV, AVI video input
- Extract frames at adaptive FPS based on camera motion
- Assess quality: blur detection, exposure analysis, coverage estimation
- Support DJI SRT GPS metadata

### FR-2: Scene Reconstruction
- Run COLMAP SfM for camera estimation
- Train 3D Gaussian Splatting with Gaussian Grouping (joint segmentation)
- Apply PPISP correction for photometric consistency
- Apply pose optimisation if initial loss is high
- Quality gate: PSNR > 25 dB against training views

### FR-3: Object Decomposition
- Extract per-object Gaussian groups from identity labels
- Support manual refinement via SAGA interactive segmentation
- Support text-based selection ("select all chairs") via LangSplat/LLM
- Handle objects of varying scale (small props to room-scale structures)

### FR-4: Mesh Extraction
- Extract textured polygonal mesh per object (SuGaR primary, SOF fallback)
- Generate UV-mapped diffuse textures
- Clean meshes: remove disconnected components, smooth, decimate
- Configurable polygon budget (per-object and total scene)

### FR-5: Background Recovery
- Inpaint removed objects from training views via diffusion model
- Retrain clean background Gaussian
- Extract background mesh (optional, environment layer)

### FR-6: USD Scene Assembly
- Compose multi-object hierarchical USD scene graph
- Per-object variant sets: Gaussian representation + Mesh representation
- Camera prims from COLMAP extrinsics
- UsdPreviewSurface materials with baked diffuse textures
- Correct coordinate transforms (COLMAP → USD)
- Scene metadata: source video, reconstruction parameters, quality metrics

### FR-7: Agentic Orchestration
- State machine driving all stages
- Quality decisions at each gate
- Automatic retries with parameter adjustment
- Partial result output on unrecoverable failure
- Progress reporting via MCP events
- All operations via LichtFeld MCP server + Python APIs

## Technical Architecture

### Deployment: Consolidated Docker

The entire pipeline runs in a single consolidated Docker container on a dedicated workstation:

- **Host**: 192.168.2.48 (HP-Desktop)
- **GPUs**: 2x RTX 6000 Ada (96 GB total VRAM)
- **CPU**: AMD Threadripper PRO 48-core
- **RAM**: 251 GB
- **Model staging**: 128 GB on `/home/john/comfyui-models-staging`
- **Services**: Web UI :7860, ComfyUI :8188, LichtFeld MCP :45677, VNC :5901

### Component Stack

```
┌─────────────────────────────────────────────────────────┐
│                ORCHESTRATOR AGENT                         │
│     State machine, quality decisions, retry logic         │
├──────────┬──────────┬──────────┬───────────────────────┤
│ Ingest   │ Recon    │ Decompose│ Assemble               │
│ Agent    │ Agent    │ Agent    │ Agent                  │
├──────────┼──────────┼──────────┼───────────────────────┤
│SplatReady│COLMAP    │SAM3      │OpenUSD Python          │
│PyAV      │LichtFeld │SAM2      │LichtFeld export        │
│OpenCV    │MCP 70+   │Hunyuan3D │Blender (materials)     │
│Web UI    │          │FLUX/CUI  │                         │
│ (:7860)  │          │TSDF/O3D  │                         │
└──────────┴──────────┴──────────┴───────────────────────┘
```

### Data Flow

```
video.mp4
  → frames/ (JPEG, quality-filtered)
    → colmap/ (cameras.txt, images.txt, points3D.txt)
      → scene.ply (trained 3DGS with per-Gaussian labels)
        → objects/
        │   ├── object_001.ply (Gaussian)
        │   ├── object_001.obj (mesh + UV)
        │   ├── object_001.png (diffuse texture)
        │   ├── object_002.ply
        │   ├── object_002.obj
        │   └── ...
        ├── background/
        │   ├── background.ply (clean Gaussian)
        │   └── background.obj (optional mesh)
        └── scene.usda (composed USD with all objects)
```

### Interface Contracts

**Ingest → Reconstruct**: Directory of JPEG frames + COLMAP dataset
**Reconstruct → Decompose**: Trained 3DGS scene with per-Gaussian labels (via LichtFeld MCP)
**Decompose → Assemble**: Per-object .ply (Gaussian) + .obj (mesh) + .png (texture) files
**Orchestrator ↔ All**: LichtFeld MCP JSON-RPC 2.0 on port 45677

### New Dependencies

| Dependency | Purpose | License | Integration |
|-----------|---------|---------|-------------|
| SAM3 | Concept segmentation (4M concepts) | Apache-2.0 | Python, upgrading from SAM2 |
| SAM2 | 2D instance segmentation (fallback) | Apache-2.0 | Python, validated |
| Hunyuan3D 2.0 | Multi-view to textured mesh | Tencent | Python client, per-object |
| Gaussian Grouping | Joint training + segmentation | Apache-2.0 | Python, replaces standard training |
| SAGA | Interactive refinement | Apache-2.0 | Python, post-hoc on any model |
| SuGaR | Mesh extraction + UV texturing | Unspecified* | Python, per-object |
| SOF/GOF | High-quality mesh extraction | Unspecified* | C++/CUDA port |
| Open3D | TSDF fusion (22K verts, 49K faces) | MIT | Python, validated |
| xatlas | UV atlas generation | MIT | C++ lib |
| usd-core | USD scene composition | Modified Apache 2.0 | Python |
| ComfyUI | Node workflows (SAM3D/TRELLIS) | GPL-3.0 | Docker :8188 |
| FLUX | Inpainting via ComfyUI | Apache-2.0 | ComfyUI workflow |
| Flask | Web upload interface | BSD | Python :7860 |

*SuGaR and SOF/GOF are academic code. Licensing requires direct author contact for commercial use.

## Development Phases

### Phase 1: Segmented Training (2-3 weeks)
- Integrate Gaussian Grouping with LichtFeld training pipeline
- SAM2 preprocessing for automatic mask generation
- Per-label Gaussian extraction via MCP selection tools
- Validation: verify object labels match visual objects

### Phase 2: Mesh Extraction (1-2 weeks)
- SuGaR integration for per-object textured mesh extraction
- TSDF fallback for background environment mesh
- Mesh cleaning pipeline (Open3D/trimesh)
- Quality validation: round-trip mesh2splat comparison

### Phase 3: Background Recovery (1-2 weeks)
- ComfyUI FLUX inpainting workflow
- Per-view mask generation from object extraction
- Background Gaussian retraining from inpainted views
- Quality validation: LPIPS coherence check

### Phase 4: USD Assembly (1 week)
- Master scene composition script (OpenUSD Python)
- Variant sets: Gaussian + Mesh per object
- Camera prims from COLMAP
- UsdPreviewSurface materials with baked textures
- Coordinate transform chain validation

### Phase 5: Orchestrator Agent (2 weeks)
- State machine implementation
- Quality gate decision logic
- Retry and fallback strategies
- Progress reporting via MCP events
- End-to-end integration testing

### Phase 6: Polish (2 weeks)
- Performance optimisation (parallel per-object processing)
- Edge case handling (transparent objects, reflections, thin structures)
- Documentation and example workflows
- CLI wrapper: `video2scene <video> <output>`

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Gaussian Grouping label quality | High | Low | SAGA refinement fallback |
| SuGaR mesh over-smoothing | Medium | Medium | SOF alternative, adjustable resolution |
| Inpainting artefacts | Medium | Medium | Multi-pass, agent quality check |
| VRAM limits (large scenes) | High | Medium | Per-object streaming, CPU offload |
| Coordinate transform errors | Medium | High | Unit tests with known-good datasets |
| License ambiguity (SuGaR, SOF) | High | Medium | Contact authors, TSDF fallback |
| Training time (>2 hours) | Low | Medium | Reduce iterations, subsample frames |
