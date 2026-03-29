# Existing Capabilities: What We Already Have

## LichtFeld Studio (v0.4.2)

### Reconstruction
- COLMAP 4.1.0 (CUDA-accelerated, headless)
- SplatReady plugin (video -> COLMAP pipeline)
- Training strategies: MCMC, MRNF, IGS+
- Pose optimisation: direct + MLP modes
- PPISP correction (exposure, vignetting, colour, CRF)
- SH degrees 0-3

### MCP Server (70+ tools)
- Training: load_dataset, start, get_state, loss_history, ask_advisor
- Camera: get, set_view, reset, list, go_to_dataset_camera
- Render: capture (base64 PNG), settings.get/set
- Selection: rect, polygon, lasso, ring, brush, click, by_description (LLM)
- Export: PLY, SOG, SPZ, USD, HTML (async with status/cancel)
- Scene graph: list_nodes, select, visibility, rename, reparent, add_group, duplicate, merge
- History: undo/redo with transactions
- Gaussians: read/write GPU tensor data
- Editor: Python code execution
- Events: pub/sub system
- Plugins: invoke/list

### File Format Support
- Import: PLY, SOG, SPZ, USD/USDA/USDC/USDZ, OBJ, FBX, glTF, GLB, STL, DAE
- Export: PLY, SOG, SPZ, USD, HTML
- Checkpoint: .resume format

### Scene Graph
- Node types: SPLAT, MESH, GROUP, POINTCLOUD
- Hierarchical tree with visibility, locking, renaming
- Duplication, merging, reparenting

### mesh2splat (EA, BSD-3)
- Mesh -> Gaussian conversion
- PBR texture support (diffuse, metallic-roughness, normal)
- Integrated in rendering pipeline

## COLMAP (4.1.0)

- Feature extraction (SIFT, GPU-accelerated)
- Exhaustive/sequential/spatial matching
- Sparse mapper
- Model alignment
- Image undistortion
- Dense reconstruction (CUDA)
- Format export (TXT, BIN)

## Consolidated Docker (192.168.2.48)

### Hardware
- 2x NVIDIA RTX 6000 Ada (96 GB total VRAM)
- AMD Threadripper PRO 48-core
- 251 GB RAM
- 128 GB staged models on fast SSD

### Services
- Web UI on :7860 (Flask upload + job manager)
- ComfyUI on :8188 (SAM3D/TRELLIS nodes, FLUX inpainting)
- LichtFeld MCP on :45677 (70+ tools)
- VNC on :5901 (remote desktop)

### VRAM Management
- Stage-based model unloading across dual GPUs
- GPU 0: Training, segmentation, Hunyuan3D
- GPU 1: ComfyUI, FLUX inpainting

## SAM3 (Upgrading from SAM2)

### SAM3 Concept Segmentation
- 4M concepts with text+visual prompts
- No prompt engineering needed -- describe objects in natural language
- SAM3.1 Object Multiplex for multi-object tracking across video frames
- Client built as `sam3d_client.py`
- Running via ComfyUI SAM3D nodes on consolidated Docker

### SAM2 Automatic Segmentation (Validated, Fallback)
- Model: `facebook/sam2-hiera-large` (HuggingFace, ~2min checkpoint download)
- Mode: `segment_video_auto` -- per-frame automatic mask generation + greedy IoU tracking
- Parameters: `points_per_side=32`, `pred_iou_thresh=0.80`, `stability_score_thresh=0.92`
- Performance on gallery tour (121 frames, 1600x899): ~5 minutes total on CUDA
- Cross-frame IoU tracking produced 405+ unique object IDs (high fragmentation due to
  per-frame auto-generation with 0.30 IoU threshold -- many small objects split across frames)

### Mask Projection onto 3D Gaussians (Validated)
- Pipeline: `MaskProjector.assign_labels_batched()` with 200K batch size
- Projection: world-to-camera transform + pinhole intrinsics, nearest-pixel label lookup
- Voting: majority vote across all views where Gaussian is visible, `min_votes=2`
- Results on 1M Gaussian scene (950K after 95th-percentile outlier filter):
  - 933,837 / 950,000 Gaussians labeled (98.3%)
  - 16,163 Gaussians unlabeled (background / insufficient votes)
  - 33 objects with >= 10 Gaussians extracted as separate PLY files
  - Dominant object (label 35): 661,548 Gaussians (69.6%) -- likely room/walls
  - Second largest (label 173): 107,438 Gaussians (11.3%)
  - Third (label 169): 57,291 Gaussians (6.0%)
  - Total pipeline time: 438.8 seconds (7.3 minutes)

### Key Observations
- IoU-based frame-to-frame tracking produces too many fragmented IDs for a walkthrough
  video where viewpoint changes dramatically between frames. SAM3 concept prompts will
  solve this by using semantic labels instead of geometric IoU tracking.
- The `min_votes=2` threshold is important -- single-view labels are unreliable due to
  perspective distortion and self-occlusion at grazing angles.
- Memory usage is manageable: 950K Gaussians x 121 views processed in 200K batches
  without exceeding GPU/RAM limits.
- Per-object PLY files retain all original Gaussian attributes (SH coefficients, opacity,
  scale, rotation) so they can be rendered directly in any 3DGS viewer.
- Background PLY (66K Gaussians including 50K outliers) captures unmatched regions.

### Output Artifacts
- 33 per-object PLY files: `/test-data/gallery_output/objects/object_NNN.ply`
- Background PLY: `/test-data/gallery_output/objects/background.ply`
- Summary JSON: `/test-data/gallery_output/objects/object_separation_summary.json`

## Hunyuan3D 2.0 Client (Built)

- Multi-view to textured mesh creation
- Takes multi-view renders of isolated objects
- Produces textured OBJ/GLB with UV-mapped diffuse textures
- Client: `src/pipeline/hunyuan3d_client.py`
- Multi-view renders: `src/pipeline/multiview_renderer.py`

## FLUX Inpainting Client (Built)

- Background recovery via ComfyUI workflows
- Removes objects from training views, inpaints clean background
- Client: `src/pipeline/comfyui_inpainter.py`
- ComfyUI endpoint: :8188 on consolidated Docker

## Web Upload Interface (Running)

- Flask application on :7860
- Drag-and-drop video upload
- Job management and progress monitoring
- Files: `src/web/app.py`, `src/web/job_manager.py`, `src/web/pipeline_runner.py`

## TSDF Mesh Extraction (Validated)

- Open3D TSDF fusion
- 22K vertices, 49K faces on gallery tour
- 12 minutes extraction time, ~3 GB RAM
- Script: `scripts/run_tsdf_mesh.py`

## USD Scene Assembly (Validated)

- 59 prims assembled from gallery tour
- Variant sets: Gaussian + Mesh per object
- Camera prims from COLMAP extrinsics
- Script: `scripts/assemble_gallery_usd.py`

## ComfyUI Docker

- FLUX, SDXL, SD 1.5 models
- SAM3D and TRELLIS custom nodes
- Inpainting workflows
- LoRA loading
- ControlNet
- API access via HTTP on :8188
- Custom node ecosystem
- 128 GB staged models from `/home/john/comfyui-models-staging`

## Blender (5.0.1)

- MCP socket server
- Python scripting
- USD import/export
- Mesh operations
- Material/shader assignment
- Rendering (Cycles, EEVEE)
- Modifier stack

## CLI Tools

- `lichtfeld-studio` -- Full application
- `lfs-mcp` -- MCP HTTP wrapper
- `video2splat` -- Full video-to-training pipeline
- `colmap` -- SfM toolkit

## Python Environment

- PyAV 17.0.0 (video frame extraction)
- Pillow (image processing)
- piexif (EXIF GPS embedding)
- Open3D (TSDF fusion, mesh processing)
- OpenCV available via pacman
- Flask (web interface)

## Pipeline Modules (21)

```
src/pipeline/
├── __init__.py              # Package init
├── __main__.py              # Entry point
├── cli.py                   # CLI interface
├── config.py                # YAML/dict configuration
├── orchestrator.py          # Pipeline orchestration
├── mcp_client.py            # LichtFeld MCP client
├── sam2_segmentor.py        # SAM2 segmentation (validated)
├── sam3d_client.py          # SAM3 client (upgrading)
├── hunyuan3d_client.py      # Hunyuan3D 2.0 mesh client
├── comfyui_inpainter.py     # FLUX inpainting via ComfyUI
├── mask_projector.py        # 2D-to-3D mask projection
├── mesh_extractor.py        # TSDF + Marching Cubes
├── mesh_cleaner.py          # Decimation, hole fill
├── usd_assembler.py         # USD scene assembly
├── texture_baker.py         # Texture baking (skeleton)
├── material_assigner.py     # Material assignment (skeleton)
├── multiview_renderer.py    # Camera orbit renders
├── quality_gates.py         # Per-stage quality checks
├── coordinate_transform.py  # COLMAP <-> 3DGS <-> USD
├── frame_quality.py         # Blur/exposure filtering
└── colmap_parser.py         # COLMAP binary reader
```

## Web Interface (5 files)

```
src/web/
├── __init__.py
├── app.py                   # Flask application
├── job_manager.py           # Background job tracking
├── pipeline_runner.py       # Pipeline execution wrapper
├── static/                  # CSS/JS assets
└── templates/               # HTML templates
```

## Docker Phase 2.7

All tools pre-built in container image with CUDA support. Reproducible builds.
Consolidated Docker on dual RTX 6000 Ada is the primary deployment target.

## RuVector Memory

3DGS stack knowledge stored:
- 13 memory entries (tools, APIs, workflows)
- 10 patterns (workflow, build, troubleshooting)
- 13-node, 17-edge knowledge graph
