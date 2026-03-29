# PRD: Consolidated Video-to-Scene Docker Image

## Overview

Single Docker image containing the complete video-to-structured-3D-USD-scene pipeline, deployed on a dual RTX 6000 Ada system (2x48GB VRAM, 251GB RAM, 48-core Threadripper PRO).

## Target System

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen Threadripper PRO 7965WX 24-Core (48 threads) |
| RAM | 251GB |
| GPU 0 | NVIDIA RTX 6000 Ada Generation 48GB |
| GPU 1 | NVIDIA RTX 6000 Ada Generation 48GB |
| Disk | 570GB free on /home |
| OS | CachyOS (Arch-based) |
| Docker | 29.1.4 with nvidia-container-toolkit |
| CUDA Driver | 590.48.01 |

## Base Image Decision

**Ubuntu 24.04 with CUDA 12.8** — not CachyOS.

Rationale:
- ComfyUI ecosystem tested primarily on Ubuntu
- Hunyuan3D 2.0 official Docker uses Ubuntu
- CUDA devel images have stable apt packages
- vcpkg and LichtFeld build fine on Ubuntu
- Broadest compatibility for all Python packages
- We need Python 3.11 (usd-core) + Python 3.14 system — Ubuntu handles multi-python cleanly

## Container Components

### Core Pipeline
| Component | Purpose | VRAM Budget |
|-----------|---------|-------------|
| LichtFeld Studio | 3DGS training + MCP (70+ tools) | 8-15GB (training) |
| COLMAP 4.1.0 | Structure-from-Motion | 2GB (GPU features) |
| SAM2 (hiera-large) | Video object segmentation | 9.5GB |
| ComfyUI 0.3.x | FLUX inpainting + Hunyuan3D 2.0 | 20-40GB (model dependent) |
| Blender 5.0.1 | Scene inspection + mesh rendering | 2GB |

### ComfyUI Nodes Required
| Node Package | Purpose |
|--------------|---------|
| ComfyUI-Manager | Node management |
| comfyui-sam3dobjects | TRELLIS/SAM3D 3D reconstruction |
| Hunyuan3D v2 nodes | Multi-view image to 3D |
| ControlNet inpainting | AliMama inpainting |
| SAM2 nodes | Segmentation |
| GroundingDINO | Open-vocabulary detection |

### Models (~120GB total)
| Model | Size | Stage |
|-------|------|-------|
| flux2_dev_fp8mixed | 33GB | Inpainting |
| mistral_3_small_flux2_fp8 | 17GB | FLUX2 text encoder |
| Trellis2 checkpoints (6) | 10GB | 3D asset creation |
| SAM3D checkpoints | 8GB | 3D scene generation |
| UltraShape v1 | 6.9GB | Shape generation |
| SAM3 | 3.2GB | Segment Anything 3D |
| FLUX2 Turbo LoRA | 2.6GB | Fast inference |
| Hunyuan3D v2 weights | ~15GB | Multi-view to 3D (download) |
| SAM2 hiera-large | ~2.5GB | Video segmentation (download) |
| GroundingDINO | 662MB | Object detection |
| FLUX2 VAE + clip_l | ~500MB | Decoding |
| Misc (SAM vit_b, tiny) | ~500MB | Fallbacks |

### VRAM Management Strategy

The dual A6000 Ada setup gives 96GB total but models must be staged:

```
STAGE 1 — INGEST (CPU only, no VRAM)
  Frame extraction, quality assessment

STAGE 2 — COLMAP (GPU0, 2GB)
  Feature extraction + matching

STAGE 3 — TRAINING (GPU0, 15GB)
  LichtFeld 3DGS with MCMC
  → torch.cuda.empty_cache() after

STAGE 4 — SEGMENTATION (GPU0, 10GB)
  SAM2 on all frames
  → Unload SAM2, free VRAM

STAGE 5 — SCENE INSPECTION (GPU0, 2GB)
  Blender headless render for agent verification
  LichtFeld MCP for render.capture

STAGE 6 — INPAINTING (GPU0, 40GB)
  Load FLUX2 + text encoder
  Inpaint all views with removed objects
  → POST /free {"unload_models": true, "free_memory": true}

STAGE 7 — 3D ASSET CREATION (GPU0, 40GB)
  Per-object: render multi-view from gaussians
  Hunyuan3D 2.0 multi-view → textured mesh
  → Unload between objects if needed

STAGE 8 — ASSEMBLY (CPU only)
  USD scene composition
  Blender final validation
```

### Web Interface

Simple Flask/FastAPI app:
- `GET /` — Upload form (video file)
- `POST /upload` — Accept video, start pipeline
- `GET /status/<job_id>` — Pipeline progress (JSON + SSE)
- `GET /download/<job_id>` — Download result archive (USD + meshes + textures)
- `GET /preview/<job_id>` — View Blender renders of intermediate stages
- WebSocket for real-time log streaming

Port: 7860 (standard for ML demos)

### Dockerfile Structure

```dockerfile
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Phase 1: System deps (apt)
# Phase 2: Python 3.11 (for usd-core) + Python 3.12 (system)
# Phase 3: Node.js 23 (for Claude tooling)
# Phase 4: COLMAP from source
# Phase 5: METIS + GKlib from source
# Phase 6: vcpkg + LichtFeld Studio from source
# Phase 7: Blender 5.0.1
# Phase 8: ComfyUI + custom nodes
# Phase 9: Pipeline code (from gaussian-toolkit repo)
# Phase 10: Web interface
# Phase 11: Model download helper
# Phase 12: Entrypoint with supervisord
```

### Model Provisioning

Models copied from host staging directory via volume mount at build time:
```yaml
volumes:
  - /home/john/comfyui-models-staging:/models-staging:ro
  - gaussian-models:/opt/models:rw
```

First-run script copies from staging to container model dirs.

### Ports

| Port | Service |
|------|---------|
| 7860 | Web interface |
| 8188 | ComfyUI direct |
| 45677 | LichtFeld MCP |
| 5901 | VNC (Blender inspection) |

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      memory: 200G
      cpus: '48'
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu, compute, utility, graphics]
shm_size: 64gb
```

## Quality Targets

| Metric | Target |
|--------|--------|
| Object detection recall | >80% of visible objects |
| Mesh quality (vertices per object) | 10K-100K |
| Texture resolution | 2048x2048 per object |
| Scene graph completeness | All objects with transforms |
| USD validity | Passes usdchecker |
| End-to-end time (60s video) | <2 hours |
| Agentic iterations per scene | Up to 5 refinement passes |

## Build Command

```bash
cd /home/john/githubs/gaussian
docker build -t gaussian-toolkit:latest -f Dockerfile .
docker compose up -d
```

## Test Procedure

1. Open http://192.168.2.48:7860 in browser
2. Upload gallery_tour_60s.mp4
3. Monitor pipeline via /status endpoint
4. Download USD + assets when complete
5. Import into Blender/Omniverse to verify
