# Hunyuan3D 2.0 Integration for Gaussian-to-Mesh Pipeline

## Overview

Hunyuan3D 2.0 is Tencent's open-source 3D asset generation system comprising
two foundation models: a shape generator (DiT) and a texture synthesizer (Paint).
This document covers integration with our Gaussian splatting decomposition pipeline,
where per-object Gaussians are rendered to multi-view images and fed into Hunyuan3D
for high-quality textured mesh reconstruction.

## Architecture

### Model Components

| Model | Params | Size | Purpose |
|-------|--------|------|---------|
| Hunyuan3D-DiT-v2-0 | 1.1B | ~4.9 GB | Single-view shape generation |
| Hunyuan3D-DiT-v2-0-Turbo | 1.1B | ~4.9 GB | Step-distilled (faster) |
| Hunyuan3D-DiT-v2-0-Fast | 1.1B | ~4.9 GB | Guidance-distilled (faster) |
| Hunyuan3D-DiT-v2-mv | 1.1B | 4.93 GB | Multi-view shape generation |
| Hunyuan3D-DiT-v2-mv-Turbo | 1.1B | 4.93 GB | Multi-view turbo |
| Hunyuan3D-Paint-v2-0 | 1.3B | ~5.2 GB | Texture synthesis |
| Hunyuan3D-Paint-v2-0-Turbo | 1.3B | ~5.2 GB | Texture turbo |
| Hunyuan3D-VAE-v2-0 | - | ~2.5 GB | 3D VAE decoder (octree) |
| Hunyuan3D-Delight-v2-0 | - | - | De-lighting for materials |
| Hunyuan3D-2mini | 0.6B | - | Lightweight variant |

### HuggingFace Repositories

**Main repository** (74.9 GB total): `tencent/Hunyuan3D-2`
- Subdirectories: hunyuan3d-dit-v2-0/, hunyuan3d-dit-v2-0-fast/,
  hunyuan3d-dit-v2-0-turbo/, hunyuan3d-paint-v2-0/, hunyuan3d-paint-v2-0-turbo/,
  hunyuan3d-vae-v2-0/, hunyuan3d-vae-v2-0-turbo/, hunyuan3d-vae-v2-0-withencoder/,
  hunyuan3d-delight-v2-0/
- Each model subdirectory contains: config.yaml + model.safetensors (or model.fp16.safetensors)
- License: tencent-hunyuan-community (not gated, free to download)

**Multi-view repository** (29.6 GB total): `tencent/Hunyuan3D-2mv`
- Subdirectories: hunyuan3d-dit-v2-mv/, hunyuan3d-dit-v2-mv-fast/, hunyuan3d-dit-v2-mv-turbo/
- Key file: `hunyuan3d-dit-v2-mv/model.fp16.safetensors` (4.93 GB)

### VRAM Requirements

- Shape generation only: **6 GB VRAM minimum**
- Shape + texture generation: **16 GB VRAM minimum**
- Recommended for production: 24 GB (RTX 4090 / A5000)

## ComfyUI Integration

### Native ComfyUI Nodes (Built-in)

Our remote ComfyUI (192.168.2.48:8189) has these native nodes:

**EmptyLatentHunyuan3Dv2**
- Inputs: `resolution` (int, 1-8192, default 3072), `batch_size` (int, 1-4096, default 1)
- Output: LATENT (zero-initialized 3D latent space)

**Hunyuan3Dv2Conditioning** (single-view)
- Input: `clip_vision_output` (CLIP_VISION_OUTPUT from CLIPVisionEncode)
- Outputs: `positive` (CONDITIONING), `negative` (CONDITIONING)

**Hunyuan3Dv2ConditioningMultiView** (multi-view)
- Inputs: `front`, `left`, `back`, `right` (all CLIP_VISION_OUTPUT)
- Outputs: `positive` (CONDITIONING), `negative` (CONDITIONING)
- All 4 views should be provided for best results; front is most critical

**VAEDecodeHunyuan3D**
- Inputs: `samples` (LATENT), `vae` (VAE), `num_chunks` (1000-500000, default 8000),
  `octree_resolution` (16-512, default 256)
- Output: VOXEL (3D voxel grid, convertible to mesh)

### Checkpoint Loading

The `ImageOnlyCheckpointLoader` node loads Hunyuan3D checkpoints. Files go into
`ComfyUI/models/checkpoints/`:
- `hunyuan3d-dit-v2-mv.safetensors` for multi-view
- `hunyuan3d-dit-v2-mv-turbo.safetensors` for multi-view turbo
- `hunyuan3d-dit-v2.safetensors` for single-view

The checkpoint provides 3 outputs: MODEL, VAE, CLIP_VISION.

### Third-Party Wrappers

**ComfyUI-Hunyuan3DWrapper** (by kijai): Most complete integration, supports
texture generation via Paint model. Requires custom_rasterizer wheel compilation.

**ComfyUI-3D-Pack**: Alternative integration but has dependency conflicts.

### Workflow Pipeline (Native)

```
LoadImage(s) -> CLIPVisionEncode -> Hunyuan3Dv2Conditioning[MultiView]
                                           |
ImageOnlyCheckpointLoader -------> KSampler <-- EmptyLatentHunyuan3Dv2
         |                            |
         +-----> VAEDecodeHunyuan3D <--+
                      |
                   SaveGLB
```

### Limitation: No Native Texture Support

ComfyUI's native Hunyuan3D nodes produce **geometry only** (no textures/materials).
For textured output, the ComfyUI-Hunyuan3DWrapper is needed, which runs the
Hunyuan3D-Paint model as a second pass. The native workflow outputs GLB files
to `ComfyUI/output/mesh/`.

## Multi-View Rendering Strategy

### Camera Configuration for Hunyuan3D

Hunyuan3D's multi-view conditioning expects **4 canonical views**:
- **Front** (0 degrees azimuth)
- **Left** (90 degrees azimuth)
- **Back** (180 degrees azimuth)
- **Right** (270 degrees azimuth)

All views at 0 degrees elevation, looking at the object center.

### Rendering from Gaussians

The `MultiViewRenderer` class handles:
1. Loading the 3DGS PLY file (positions, SH coefficients, opacities, scales, rotations)
2. Centering and normalizing the object to a unit sphere
3. Computing orbiting camera positions
4. Software-based Gaussian splatting with:
   - Spherical harmonic evaluation for view-dependent color
   - 2D covariance projection via perspective Jacobian
   - Front-to-back alpha compositing
5. Outputting RGBA images with transparent backgrounds

### Image Preprocessing for Hunyuan3D

- Images should have transparent or solid-color backgrounds
- Recommended resolution: 512x512 (preprocessed by ComfyUI internally)
- Alpha channel preserves the object silhouette for cleaner conditioning
- Optional background removal with ComfyUI_essentials nodes

## Integration with Pipeline

### Files Created

- `src/pipeline/multiview_renderer.py` - Multi-view rendering from Gaussian PLY
- `src/pipeline/hunyuan3d_client.py` - ComfyUI client for Hunyuan3D workflows
- `src/pipeline/workflows/hunyuan3d_multiview.json` - Multi-view workflow template
- `src/pipeline/workflows/hunyuan3d_singleview.json` - Single-view workflow template

### Usage in the Decomposition Pipeline

After the EXTRACT_OBJECTS stage produces per-object Gaussian PLY files, the
Hunyuan3D client can replace or augment the existing mesh extraction:

```python
from pipeline.hunyuan3d_client import Hunyuan3DClient

client = Hunyuan3DClient(
    comfyui_url="http://192.168.2.48:8189",
    quality="standard",
)

# Multi-view: renders 4 views from the Gaussian PLY, then runs Hunyuan3D
result = client.reconstruct_from_gaussians("objects/chair.ply")

# Save the textured GLB
client.save_result(result, output_dir="output/chair")
```

### Quality Presets

| Preset | Steps | CFG | Latent Res | Octree Res | Chunks | Use Case |
|--------|-------|-----|------------|------------|--------|----------|
| draft | 20 | 5.0 | 2048 | 128 | 4000 | Quick preview |
| standard | 50 | 5.5 | 3072 | 256 | 8000 | Production |
| high | 75 | 6.0 | 4096 | 384 | 16000 | High detail |
| ultra | 100 | 6.5 | 4096 | 512 | 32000 | Maximum quality |

### Fallback Strategy

1. Try multi-view reconstruction (4 views from Gaussians)
2. If multi-view model unavailable, fall back to single-view (front render)
3. If Hunyuan3D unavailable, fall back to SAM3D or Tripo (existing clients)

## Models to Download

For our ComfyUI server, the minimum required models are:

### Multi-view (recommended)
```bash
# Primary multi-view model (4.93 GB)
huggingface-cli download tencent/Hunyuan3D-2mv \
  hunyuan3d-dit-v2-mv/model.fp16.safetensors \
  --local-dir ComfyUI/models/checkpoints/

# Rename to ComfyUI convention
mv ComfyUI/models/checkpoints/hunyuan3d-dit-v2-mv/model.fp16.safetensors \
   ComfyUI/models/checkpoints/hunyuan3d-dit-v2-mv.safetensors
```

### Single-view (fallback)
```bash
# Single-view model (4.93 GB)
huggingface-cli download tencent/Hunyuan3D-2 \
  hunyuan3d-dit-v2-0/model.safetensors \
  --local-dir ComfyUI/models/checkpoints/

mv ComfyUI/models/checkpoints/hunyuan3d-dit-v2-0/model.safetensors \
   ComfyUI/models/checkpoints/hunyuan3d-dit-v2.safetensors
```

### Optional turbo variants
```bash
# Multi-view turbo (4.93 GB)
huggingface-cli download tencent/Hunyuan3D-2mv \
  hunyuan3d-dit-v2-mv-turbo/model.fp16.safetensors

# Single-view fast (4.93 GB)
huggingface-cli download tencent/Hunyuan3D-2 \
  hunyuan3d-dit-v2-0-fast/model.safetensors
```

### For texture generation (optional, requires ComfyUI-Hunyuan3DWrapper)
```bash
# Paint model (5.2 GB)
huggingface-cli download tencent/Hunyuan3D-2 \
  hunyuan3d-paint-v2-0/model.safetensors

# Delight model (for PBR material extraction)
huggingface-cli download tencent/Hunyuan3D-2 \
  hunyuan3d-delight-v2-0/
```

Total minimum download: ~5 GB (multi-view only)
Total recommended: ~15 GB (multi-view + single-view + turbo)
Total with textures: ~25 GB (all models)

## Comparison with Existing Backends

| Feature | SAM3D | Tripo | Hunyuan3D MV |
|---------|-------|-------|--------------|
| Input | Single image | Single image | 4 views |
| Texture | Yes (baked) | Yes (cloud) | Geometry only (native) |
| VRAM | ~12 GB | Cloud API | ~6-16 GB |
| Speed | 30-120s | 20-60s | 30-90s |
| Quality | High | High | Very high geometry |
| Multi-view | No | No | Yes |
| Offline | Yes | No | Yes |
| License | MIT | Proprietary | Community |

### Key Advantage of Multi-View

When reconstructing objects from Gaussian splats, we already have the full 3D
representation. By rendering multiple consistent views, Hunyuan3D receives far
more geometric information than a single-image approach, resulting in more
accurate shape reconstruction - especially for complex or occluded geometry.

## Known Issues

1. Native ComfyUI nodes do not support texture generation (geometry only)
2. The VAE decoder uses octree voxelization; high resolutions (>384) require
   significant VRAM and time
3. The SaveGLB node output path format varies between ComfyUI versions
4. Background removal is critical - artifacts in rendered views degrade quality
5. Hunyuan3D 2.1 exists (with PBR material support) but ComfyUI integration
   is still maturing

## References

- [Hunyuan3D-2 GitHub](https://github.com/Tencent/Hunyuan3D-2)
- [Hunyuan3D-2 HuggingFace](https://huggingface.co/tencent/Hunyuan3D-2)
- [Hunyuan3D-2mv HuggingFace](https://huggingface.co/tencent/Hunyuan3D-2mv)
- [ComfyUI Native 3D Tutorial](https://docs.comfy.org/tutorials/3d/hunyuan3D-2)
- [ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)
- [Hunyuan3Dv2ConditioningMultiView Node Docs](https://www.runcomfy.com/comfyui-nodes/ComfyUI/hunyuan3-dv2-conditioning-multi-view)
- [VAEDecodeHunyuan3D Node Docs](https://www.runcomfy.com/comfyui-nodes/ComfyUI/vae-decode-hunyuan3-d)
- [EmptyLatentHunyuan3Dv2 Node Docs](https://www.runcomfy.com/comfyui-nodes/ComfyUI/empty-latent-hunyuan3-dv2)
