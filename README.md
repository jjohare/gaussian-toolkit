# Gaussian Toolkit

Video-to-3D-scene pipeline built on [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio). Upload a video, get a textured polygonal mesh and USD scene.

## Architecture

```
Video Upload → Frame Extraction → COLMAP SfM → 3DGS Training →
  → Object Segmentation (SAM3) → Mesh Extraction (TSDF or MILo) →
  → Blender Assembly + Texture Bake → USD Scene + Web Viewer
```

Two-container Docker deployment:

| Container | Base | GPU | Purpose |
|-----------|------|-----|---------|
| `gaussian-toolkit` | Ubuntu 24.04, CUDA 12.8, Python 3.12 | GPU 0 | COLMAP, LichtFeld 3DGS, web UI, Blender, SAM3 |
| `milo` | Ubuntu 22.04, CUDA 11.8, Python 3.10 | GPU 1 | MILo mesh extraction (SIGGRAPH Asia 2025) |

## Quick Start

```bash
# Set HuggingFace token (needed for SAM3 segmentation models)
export HF_TOKEN=hf_your_token_here

# Start both containers
docker compose -f docker-compose.consolidated.yml up -d

# Open the web interface
# http://localhost:7860
```

Upload a video at the web UI. The pipeline runs autonomously via Claude Code inside the container.

## Pipeline Stages

| Stage | Tool | Output |
|-------|------|--------|
| Frame extraction | PyAV | JPEG frames |
| Structure-from-Motion | COLMAP 4.1.0 | Camera poses + sparse point cloud |
| 3DGS training | LichtFeld Studio (MCP) | Trained gaussian PLY (~1M splats) |
| Object segmentation | SAM3 (4M concepts, text+visual) | Per-object 2D masks |
| Mask projection | Custom (ray casting) | Per-object 3D gaussian labels |
| Mesh extraction | TSDF (fast) or MILo (high quality) | GLB meshes with vertex colours |
| Scene assembly | Blender (Cycles GPU) | Texture-baked USD scene |
| Web delivery | Flask + model-viewer | Preview carousel, download ZIP |

### Mesh Extraction Backends

- **TSDF** (default): Open3D volumetric fusion from rendered depth maps. Fast (~2 min), lower geometric quality. Good for previews.
- **MILo** (SIGGRAPH Asia 2025): Differentiable mesh-in-the-loop gaussian splatting. Delaunay triangulation + learned SDF. High quality, runs in the dedicated sidecar container on GPU 1.

## Web Interface

The Flask app on port 7860 provides:

- Video upload with drag-and-drop
- Real-time pipeline progress (SSE log streaming)
- 3D preview via `<model-viewer>` (Google)
- Preview image carousel from Blender renders
- Download ZIP of all outputs (mesh, USD, previews)
- Anthropic API key management for Claude Code orchestration

## Services

| Port | Service |
|------|---------|
| 7860 | Web UI (Flask) |
| 7681 | Web terminal (ttyd / Claude Code) |
| 8188 | ComfyUI |
| 45677 | LichtFeld MCP server (70+ tools) |
| 5901 | VNC (Blender remote desktop) |

## Pipeline Modules

28 Python modules in `src/pipeline/`:

| Category | Modules |
|----------|---------|
| Core | `stages.py`, `orchestrator.py`, `cli.py`, `config.py`, `preflight.py` |
| Reconstruction | `colmap_parser.py`, `coordinate_transform.py`, `frame_selector.py`, `frame_quality.py` |
| Segmentation | `sam2_segmentor.py`, `sam3_segmentor.py`, `sam3d_client.py`, `mask_projector.py` |
| Mesh | `mesh_extractor.py` (TSDF), `milo_extractor.py` (MILo), `mesh_cleaner.py` |
| Texturing | `texture_baker.py`, `material_assigner.py` |
| Scene | `blender_assembler.py`, `usd_assembler.py` |
| Rendering | `multiview_renderer.py`, `hunyuan3d_client.py`, `comfyui_inpainter.py` |
| Utilities | `mcp_client.py`, `quality_gates.py`, `person_remover.py` |

Web interface in `src/web/`: `app.py`, `job_manager.py`, `pipeline_runner.py`, templates, static assets.

## Hardware

Tested on:

| Component | Spec |
|-----------|------|
| GPU | 2x NVIDIA RTX 6000 Ada (48 GB VRAM each, 96 GB total) |
| CPU | AMD Threadripper PRO 48-core |
| RAM | 251 GB |
| Storage | NVMe SSD |

Minimum: single GPU with 12 GB VRAM (TSDF only, no MILo sidecar).

## Project Boundaries

This is a fork of LichtFeld Studio. Upstream code (`src/core/`, `src/app/`, `src/mcp/`, `src/rendering/`, `src/training/`) is not modified. All pipeline additions live in `src/pipeline/`, `src/web/`, `docker/`, and `scripts/`. See [BOUNDARIES.md](BOUNDARIES.md) for the full separation policy.

## License

Upstream LichtFeld Studio: GPL-3.0. Pipeline additions: GPL-3.0 (derivative work).
