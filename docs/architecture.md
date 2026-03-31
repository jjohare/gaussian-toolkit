# Architecture

## Two-Container Deployment

Gaussian Toolkit runs as two Docker containers sharing a volume for output data.

### gaussian-toolkit (main container)

| Property | Value |
|----------|-------|
| Base image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Python | 3.12 |
| GPU assignment | Device 0 (RTX 6000 Ada, 48 GB) |
| Ports | 7860 (web UI), 7681 (ttyd terminal), 8188 (ComfyUI), 45677 (LichtFeld MCP), 5901 (VNC) |
| Process manager | supervisord |
| Memory limit | 200 GB |
| Shared memory | 64 GB |

Runs: COLMAP SfM, LichtFeld Studio 3DGS training, SAM3 segmentation, Blender scene assembly, Flask web UI, ComfyUI workflows, Claude Code (agentic orchestrator).

### milo (sidecar container)

| Property | Value |
|----------|-------|
| Base image | `nvidia/cuda:11.8.0-devel-ubuntu22.04` |
| Python | 3.10 |
| GPU assignment | Device 1 (RTX 6000 Ada, 48 GB) |
| Entrypoint | `sleep infinity` (called via `docker exec`) |
| CUDA extensions | diff-gaussian-rasterization (3 variants), simple-knn, fused-ssim, nvdiffrast, tetra-triangulation |

MILo requires CUDA 11.8 + GCC <= 11 for its CUDA extension compilation. This is incompatible with the main container (CUDA 12.8 + GCC 14). The sidecar isolates these dependencies.

The main container calls MILo via:
```bash
docker exec milo python3 train.py --source_path /data/output/JOB/colmap ...
```

### Shared Resources

```
Volumes:
  ./output:/data/output        # Both containers read/write pipeline outputs
  hf-cache:/opt/hf-cache       # HuggingFace model cache (shared)
  models-data:/opt/models       # Persistent model storage
  claude-session:/home/ubuntu/.claude  # Claude Code OAuth (main only)
```

## System Diagram

```
┌─────────────────────────────────────────────────────────┐
│  gaussian-toolkit container (GPU 0)                      │
│                                                          │
│  ┌──────────┐ ┌───────────┐ ┌────────────┐             │
│  │ Flask UI │ │ LichtFeld │ │  COLMAP    │             │
│  │  :7860   │ │ MCP :45677│ │  SfM       │             │
│  └────┬─────┘ └─────┬─────┘ └────────────┘             │
│       │              │                                   │
│  ┌────▼──────────────▼─────────────────────┐            │
│  │        Pipeline (28 Python modules)      │            │
│  │  stages → colmap → train → segment →     │            │
│  │  extract mesh → blender assemble         │            │
│  └────┬─────────────────────────────────────┘            │
│       │                                                  │
│  ┌────▼─────┐ ┌──────────┐ ┌───────────┐               │
│  │ Blender  │ │ ComfyUI  │ │ SAM3      │               │
│  │ (Cycles) │ │  :8188   │ │ segment   │               │
│  └──────────┘ └──────────┘ └───────────┘               │
│                                                          │
│  Claude Code (ttyd :7681) — orchestrates entire pipeline │
└────────────────┬─────────────────────────────────────────┘
                 │ docker exec / shared /data/output volume
┌────────────────▼─────────────────────────────────────────┐
│  milo container (GPU 1)                                   │
│                                                          │
│  MILo (SIGGRAPH Asia 2025)                               │
│  Differentiable mesh-in-the-loop gaussian splatting      │
│  CUDA 11.8 + PyTorch 2.3.1 + nvdiffrast                 │
│  Delaunay triangulation + learned SDF                    │
└──────────────────────────────────────────────────────────┘
```

## Claude Code as Orchestrator

Claude Code runs inside the main container (accessible via ttyd on port 7681). It drives the pipeline by:

1. Receiving a job from the Flask web UI
2. Calling pipeline stages in sequence via Python imports
3. Invoking LichtFeld MCP tools for 3DGS training control (70+ tools on :45677)
4. Running Blender headless for scene assembly and Cycles GPU texture baking
5. Optionally calling the MILo sidecar for high-quality mesh extraction
6. Writing results back to `/data/output/JOB_ID/`

The pipeline modules (`src/pipeline/stages.py`) are designed as independent, stateless functions. Claude Code decides what to run next based on each stage's output. There is no hidden state machine.

## Data Flow

```
/data/output/JOB_ID/
├── input.mp4                    # Uploaded video
├── frames/                      # Extracted JPEG frames
├── colmap/                      # COLMAP sparse model + undistorted images
│   ├── images/
│   └── sparse/0/
├── training/                    # 3DGS training output
│   └── point_cloud.ply
├── segmentation/                # SAM3 per-object masks
├── objects/
│   ├── gaussians/               # Per-object PLY splats
│   └── meshes/                  # Per-object GLB meshes (TSDF or MILo)
├── blender/                     # Blender scene + baked textures
├── usd/                         # USD scene hierarchy
├── previews/                    # Blender-rendered preview images
└── download/                    # ZIP bundle for web download
```

## Key Dependencies

| Component | Version | Purpose |
|-----------|---------|---------|
| LichtFeld Studio | 0.4.2+ | 3DGS training, MCP server (70+ tools) |
| COLMAP | 4.1.0 | Structure-from-Motion |
| Open3D | 0.18+ | TSDF fusion, mesh processing |
| MILo | latest | High-quality mesh extraction (sidecar) |
| SAM3 | latest | Concept segmentation (4M concepts) |
| Blender | 4.x | Scene assembly, Cycles GPU texture bake |
| Flask | 3.x | Web interface |
| PyAV | latest | Video frame extraction |
| gsplat | latest | Depth rendering for TSDF |
| OpenUSD | 25.02+ | USD scene export |
