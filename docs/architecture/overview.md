# Gaussian Toolkit Architecture

## System Overview

Gaussian Toolkit integrates multiple components into a unified 3D Gaussian Splatting pipeline running in a consolidated Docker container on dual RTX 6000 Ada GPUs (96GB total VRAM).

```
                     ┌──────────────────────────────────────────────────────┐
                     │               Gaussian Toolkit                       │
                     │          Consolidated Docker Container               │
                     ├──────────────────────────────────────────────────────┤
  Video Input ──────►│  SplatReady     COLMAP      LichtFeld               │──────► Trained Model
  Image Folder ─────►│  (frames)  ──► (SfM)  ──► (training)               │──────► PLY/SPZ/USD/HTML
  COLMAP Dataset ───►│                                                      │──────► Per-Object PLY
  Web Upload ──────►│  SAM3        Hunyuan3D    ComfyUI                    │──────► Textured Meshes
   (:7860)          │  (segment) ──► (mesh)  ──► (inpaint)                │──────► USD Scene
                     │                                                      │
                     │  Web UI :7860 | ComfyUI :8188 | MCP :45677          │
                     │  VNC :5901   | Threadripper PRO 48-core             │
                     │  2x RTX 6000 Ada (96GB VRAM) | 251GB RAM           │
                     └──────────────────────────────────────────────────────┘
                                    ▲
                                    │ lfs-mcp CLI / MCP bridge / Web UI
                                    ▼
                            Claude Code / Agents
```

## Component Stack

| Component | Version | License | Purpose |
|-----------|---------|---------|---------|
| LichtFeld Studio | 0.4.2+ | GPL-3.0 | 3DGS training, visualisation, editing, export |
| COLMAP | 4.1.0 | BSD | Structure-from-Motion reconstruction |
| SplatReady | 1.0.0 | Plugin | Video-to-COLMAP pipeline automation |
| SAM3 | latest | Apache-2.0 | Concept segmentation (4M concepts, text+visual prompts) |
| SAM2 | hiera-large | Apache-2.0 | Video segmentation (fallback, validated) |
| Hunyuan3D 2.0 | latest | Tencent | Multi-view to textured mesh creation |
| ComfyUI | latest | GPL-3.0 | Node-based workflow engine (SAM3D/TRELLIS nodes) |
| FLUX | dev/schnell | Apache-2.0 | Inpainting via ComfyUI workflows |
| Open3D | 0.18+ | MIT | TSDF fusion, mesh processing |
| OpenUSD | 25.02+ | Modified Apache 2.0 | USD scene composition |
| METIS | 5.2.1 | Apache-2.0 | Graph partitioning (COLMAP dependency) |
| Flask | 3.x | BSD | Web upload interface on :7860 |
| vcpkg | latest | MIT | C++ dependency management (91 packages) |

## Data Flow

### Full Pipeline (video2splat -> scene assembly)

```
Video File (.mp4/.mov) or Web Upload (:7860)
    │
    ▼ [Stage 1: SplatReady - PyAV frame extraction]
JPEG Frames + GPS EXIF
    │
    ▼ [Stage 2: COLMAP - 6-step SfM pipeline]
    │   feature_extractor → exhaustive_matcher → mapper
    │   → model_aligner → image_undistorter → model_converter
    │
    ▼
COLMAP Undistorted Dataset
    │   images/ + sparse/0/{cameras,images,points3D}.txt
    │
    ▼ [Stage 3: LichtFeld - CUDA-accelerated training]
Trained 3D Gaussian Splat Model (1M gaussians)
    │
    ▼ [Stage 4: SAM3 Concept Segmentation]
    │   text+visual prompts, 4M concepts
    │   (fallback: SAM2 grid-point prompts)
    │
    ▼ [Stage 5: Mask Projection to 3D]
    │   2D masks → 3D Gaussian labels (98.3% coverage)
    │   33 per-object PLY files extracted
    │
    ▼ [Stage 6: Per-Object Mesh Creation]
    │   Hunyuan3D 2.0: multi-view renders → textured mesh
    │   TSDF fallback: Open3D fusion (22K verts, 49K faces)
    │
    ▼ [Stage 7: Background Recovery]
    │   FLUX inpainting via ComfyUI (:8188)
    │
    ▼ [Stage 8: USD Scene Assembly]
USD Scene (59 prims, variant sets: Gaussian + Mesh per object)
    + PLY / SOG / SPZ / HTML exports
```

### MCP-Controlled Pipeline

```
Claude Agent
    │
    ├─► lfs-mcp call scene.load_dataset {"path": "..."}
    ├─► lfs-mcp call training.start
    ├─► lfs-mcp call training.get_state  (poll loop)
    ├─► lfs-mcp call render.capture {"width": 1920}
    ├─► lfs-mcp call selection.by_description {"description": "floaters"}
    ├─► lfs-mcp call gaussians.write {"delete_selected": true}
    └─► lfs-mcp call scene.export_spz {"path": "output.spz"}
```

## Consolidated Docker Architecture

The entire pipeline runs in a single consolidated Docker container:

```
┌─────────────────────────────────────────────────────────┐
│           Consolidated Docker Container                   │
│    Host: 192.168.2.48 (HP-Desktop)                       │
│    2x RTX 6000 Ada (96GB VRAM) | 251GB RAM               │
│    Threadripper PRO 48-core                              │
├─────────────────────────────────────────────────────────┤
│  Services (supervisord):                                  │
│    Web UI (:7860)        - Flask upload + job manager     │
│    ComfyUI (:8188)       - SAM3D/TRELLIS nodes           │
│    LichtFeld MCP (:45677) - 70+ tools                    │
│    VNC (:5901)           - Remote desktop                │
├─────────────────────────────────────────────────────────┤
│  Model Staging:                                           │
│    /home/john/comfyui-models-staging (128GB)             │
│    Stage-based VRAM unloading for model rotation         │
├─────────────────────────────────────────────────────────┤
│  Pipeline (21 modules):                                   │
│    orchestrator, sam2_segmentor, sam3d_client,            │
│    hunyuan3d_client, comfyui_inpainter, mask_projector,  │
│    mesh_extractor, mesh_cleaner, usd_assembler,          │
│    texture_baker, material_assigner, mcp_client,         │
│    multiview_renderer, quality_gates, config, cli,       │
│    coordinate_transform, frame_quality, colmap_parser,   │
│    __init__, __main__                                    │
└─────────────────────────────────────────────────────────┘
```

## GPU Architecture

- **CUDA Toolkit**: 13.1
- **Target architectures**: sm_89 (RTX 6000 Ada), sm_86 (RTX A6000/3090), sm_75 (RTX 6000/2080)
- **C++ Standard**: C++23
- **CUDA Standard**: C++20
- **Build system**: CMake + Ninja + vcpkg

## MCP Server Architecture

```
scripts/lichtfeld_mcp_bridge.py    <-- stdio MCP client (Claude Desktop/Codex)
        │
        │ HTTP POST to http://127.0.0.1:45677/mcp
        ▼
src/mcp/mcp_http_server.cpp        <-- cpp-httplib HTTP listener
        │
        ▼
src/mcp/mcp_server.cpp             <-- JSON-RPC 2.0 dispatcher
        │
        ├──► ToolRegistry (singleton)       70+ tools
        └──► ResourceRegistry (singleton)   8+ resources
```

### Tool Runtime Model

Tools are registered in two backends depending on the application mode:

- **Headless**: `TrainingContext` singleton manages scene/trainer directly
- **GUI**: `Visualizer` provides the backend with live viewport interaction

Each tool carries metadata:
- `category` (training, scene, render, selection, etc.)
- `kind` (command vs query)
- `runtime` (shared, headless, gui)
- `thread_affinity` (any, training_context, main_thread)
- `destructive`, `long_running`, `user_visible` flags

## Directory Layout

```
gaussian-toolkit/
├── docs/                          # This documentation
│   ├── architecture/              # System design
│   ├── build/                     # Build instructions
│   ├── integration/               # MCP, skills, Docker
│   ├── workflows/                 # Usage workflows
│   └── troubleshooting/           # Common issues
├── docker/                        # Docker configuration
│   ├── Dockerfile                 # Base container
│   ├── docker-compose.yml         # Service composition
│   ├── entrypoint.sh              # Container entry
│   ├── run_docker.sh              # Launch helper
│   └── supervisord.conf           # Process manager
├── Dockerfile.consolidated        # Consolidated Docker (all services)
├── docker-compose.consolidated.yml # Consolidated compose
├── scripts/
│   ├── tools/                     # CLI wrappers (lfs-mcp, video2splat)
│   ├── run_gallery_pipeline.py    # Full gallery pipeline
│   ├── run_object_separation.py   # Object extraction
│   ├── run_tsdf_mesh.py           # TSDF mesh extraction
│   └── assemble_gallery_usd.py   # USD scene assembly
├── src/
│   ├── pipeline/                  # 21 Python pipeline modules
│   ├── web/                       # 5 web interface files (Flask :7860)
│   ├── mcp/                       # Built-in MCP server
│   ├── app/                       # Application entry + GUI tools
│   └── ...
├── research/                      # 15 research documents
├── external/                      # Git submodules
└── build/                         # Build output (not committed)
```
