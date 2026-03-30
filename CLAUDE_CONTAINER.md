# Video-to-Scene Pipeline -- Claude Code Orchestration

You are running inside the gaussian-toolkit Docker container on a dual RTX 6000 Ada system.
**You are the orchestrator.** There is no state machine. You run each pipeline stage manually,
inspect results between steps, and decide what to do next.

**CRITICAL: You MUST complete ALL stages through to USD assembly and validation.
Do NOT stop after training. The full pipeline is:
ingest -> select_frames -> reconstruct -> train -> segment -> extract_objects -> mesh_objects -> assemble_usd -> validate**

## Available Tools

- LichtFeld Studio: `/opt/gaussian-toolkit/build/LichtFeld-Studio`
- COLMAP: `/usr/local/bin/colmap`
- Blender: `/usr/local/bin/blender` (DISPLAY=:1, VNC on :5901)
- ComfyUI API: `http://localhost:8188`
- Python pipeline stages: `from pipeline.stages import PipelineStages`
- Web API: `http://localhost:7860`

## When a job arrives

Check for new jobs:

```bash
curl -s http://localhost:7860/jobs | python3 -m json.tool
```

For each queued job, run the pipeline stage by stage.

---

## Step 1: Ingest -- extract frames from video

Extract frames at 4fps to oversample, then select the best subset.

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.ingest('/data/output/JOB_ID/input.mp4', fps=4.0)
print(result)
"
```

Update the web UI:
```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "ingest", "progress": 0.05, "message": "Extracting frames at 4fps"}'
```

**Inspect**: `ls /data/output/JOB_ID/frames/ | wc -l`

---

## Step 2: Remove people (if needed)

Look at the frames. If people are visible:

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.remove_people('/data/output/JOB_ID/frames/')
print(result)
"
```

If no people, skip to step 3 using the frames directory directly.

---

## Step 3: Select best frames (IMPORTANT for COLMAP registration)

Select 60-80 diverse, high-quality frames from the oversampled set.
This is critical -- sending all frames to COLMAP causes low registration rates.

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.select_frames('/data/output/JOB_ID/frames/', target=80)
print(result)
"
```

**Check**: The selected frame count should be 60-80. If less than 40, re-run with lower blur_threshold.

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "select_frames", "progress": 0.1, "message": "Selected N frames from M extracted"}'
```

---

## Step 4: COLMAP reconstruction

Use the **sequential** matcher for video input (NOT exhaustive -- sequential is faster
and produces better registration rates for video where frames are temporally ordered).

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.reconstruct('/data/output/JOB_ID/frames_selected/', matcher='sequential')
print(result)
"
```

**Check**: Look for `sparse/0/cameras.bin` and `images/` in the colmap dir.
**CRITICAL**: Check the registration rate. At least 70% of input frames should register.
If registration is below 50%, re-run with `matcher='exhaustive'` or reduce frame count.

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "reconstruct", "progress": 0.25, "message": "COLMAP: N/M frames registered"}'
```

---

## Step 5: Train gaussian splatting

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.train('/data/output/JOB_ID/colmap/undistorted/', iterations=30000)
print(result)
"
```

Update progress during training:
```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "train", "progress": 0.5, "message": "30k iter, loss 0.02"}'
```

**Quality check**: The PLY should be > 10 MB for a good scene.
If training fails or quality is poor, adjust `iterations` or try `strategy="mcmc"`.

**DO NOT STOP HERE. Continue to segmentation.**

---

## Step 6: Segment (SAM3 object detection)

SAM3 requires the BPE vocab file. It is located at:
`/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz`

The environment variable `SAM3_BPE_PATH` points to it. If SAM3 fails, the pipeline
will fall back to SAM2 automatic mask generation.

```bash
python3 -c "
import os
os.environ.setdefault('SAM3_BPE_PATH', '/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz')
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.segment(
    '/data/output/JOB_ID/model/point_cloud.ply',
    '/data/output/JOB_ID/frames/'
)
print(result)
"
```

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "segment", "progress": 0.65, "message": "Segmentation complete: N objects"}'
```

---

## Step 7: Extract per-object PLY files

```bash
python3 -c "
from pipeline.stages import PipelineStages
import json
p = PipelineStages('/data/output/JOB_ID')
# Use the objects JSON from segment() result
objects = [{'label': 'full_scene', 'count': -1}]
result = p.extract_objects('/data/output/JOB_ID/model/point_cloud.ply', objects)
print(result)
"
```

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "extract_objects", "progress": 0.7, "message": "Extracted N object PLYs"}'
```

---

## Step 8: Generate meshes (TSDF fusion)

Use TSDF fusion for mesh extraction. This produces watertight meshes with good topology.

```bash
python3 -c "
from pipeline.stages import PipelineStages
import json
p = PipelineStages('/data/output/JOB_ID')
plys = ['/data/output/JOB_ID/objects/full_scene.ply']
result = p.mesh_objects(plys)
print(result)
"
```

**Inspect in Blender**:
```bash
DISPLAY=:1 blender /data/output/JOB_ID/objects/meshes/full_scene/full_scene.glb
```

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "mesh_objects", "progress": 0.8, "message": "Mesh: Nk verts, Mk faces"}'
```

---

## Step 9: Texture bake (optional)

```bash
python3 -c "
from pipeline.stages import PipelineStages
import json
p = PipelineStages('/data/output/JOB_ID')
meshes = json.load(open('/data/output/JOB_ID/objects/meshes/mesh_manifest.json'))
result = p.texture_bake(meshes)
print(result)
"
```

---

## Step 10: Assemble USD scene

This is the final assembly step. Creates the hierarchical scene graph with variant sets.

```bash
python3 -c "
from pipeline.stages import PipelineStages
import json
p = PipelineStages('/data/output/JOB_ID')
meshes = json.load(open('/data/output/JOB_ID/objects/meshes/mesh_manifest.json'))
result = p.assemble_usd(meshes)
print(result)
"
```

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "assemble_usd", "progress": 0.95, "message": "USD scene assembled: N prims"}'
```

---

## Step 11: Validate

Run final validation across all outputs.

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.validate()
print(result)
"
```

---

## Mark job complete

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/complete \
  -H 'Content-Type: application/json' \
  -d '{"success": true}'
```

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "validate", "progress": 1.0, "message": "Pipeline complete"}'
```

---

## At each step: CHECK QUALITY

- Render a preview in Blender: `DISPLAY=:1 blender --background --python-expr "..."`
- If quality is poor, adjust parameters and re-run the stage
- The pipeline is not a script -- YOU decide what to do next
- You can skip stages, re-run stages, or change parameters between stages

## Quality Targets

- Frame selection: 60-80 diverse frames from 4fps extraction
- COLMAP: 70%+ registration rate, use sequential matcher for video
- Training: 30k+ iterations, MRNF strategy, loss < 0.02
- Segmentation: SAM3 text prompts for semantic labels (fallback to SAM2 if BPE missing)
- Mesh: per-object textured meshes via TSDF fusion
- USD: hierarchical scene graph with variant sets

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SAM3_BPE_PATH` | `/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz` | SAM3 text tokenizer vocab |
| `HF_TOKEN` | (from compose) | HuggingFace model downloads |
| `ANTHROPIC_API_KEY` | (from compose) | Claude Code API key |

## REST API for progress reporting

| Method | Endpoint | Body | Purpose |
|--------|----------|------|---------|
| POST | `/api/job/<id>/stage` | `{"stage": "...", "progress": 0.5, "message": "..."}` | Report stage progress |
| POST | `/api/job/<id>/stage/complete` | `{"stage": "...", "success": true}` | Mark stage done |
| POST | `/api/job/<id>/complete` | `{"success": true}` | Mark job done |
| GET | `/jobs` | -- | List all jobs |
| GET | `/status/<id>` | -- | Job detail |
| GET | `/api/job/<id>/previews` | -- | List preview images |

## IMPORTANT: Save preview images at each stage

The web UI has an image carousel that displays previews. **You MUST save preview images
during the pipeline so the user can monitor progress visually.** Save as PNG/JPG to the
job output directory. The carousel auto-refreshes every 30 seconds.

### MANDATORY previews (save these or the user cannot monitor progress):

After EACH stage below, save the specified preview images. The web carousel at
http://localhost:7860 auto-detects any PNG/JPG in the job output directory.

```bash
JOB_DIR=/data/output/JOB_ID

# 1. After frame extraction: sample frames
cp $JOB_DIR/frames/frame_00001.jpg $JOB_DIR/preview_01_frame_sample.jpg

# 2. After training: render RGB from COLMAP camera viewpoints using gsplat
python3 -c "
import sys, os; sys.path.insert(0, 'src')
from pipeline.mesh_extractor import load_3dgs_ply, render_gsplat
from pipeline.colmap_parser import parse_cameras_bin, parse_images_bin
import torch, numpy as np
from PIL import Image
from pathlib import Path

JOB = '$JOB_DIR'
ply = sorted(Path(f'{JOB}/model').glob('*.ply'))[-1]
gs = load_3dgs_ply(str(ply))

# Load COLMAP cameras for viewpoint-matched renders
sparse = Path(f'{JOB}/colmap/undistorted/sparse/0')
if sparse.exists():
    import struct
    # Parse binary cameras for intrinsics
    with open(sparse / 'cameras.bin', 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cid, model, w, h = struct.unpack('<iiii', f.read(16))
            nparams = {0:3,1:4,2:4,3:5,4:4,5:5,6:12,7:8,8:12,9:8}.get(model,4)
            params = struct.unpack(f'<{nparams}d', f.read(8*nparams))
            focal = params[0]
            break  # Use first camera
    K = torch.tensor([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=torch.float32, device='cuda')

    # Parse binary images for extrinsics (first 8)
    with open(sparse / 'images.bin', 'rb') as f:
        nimgs = struct.unpack('<Q', f.read(8))[0]
        preview_dir = Path(f'{JOB}/previews')
        preview_dir.mkdir(exist_ok=True)
        for idx in range(min(nimgs, 8)):
            img_id = struct.unpack('<i', f.read(4))[0]
            qw,qx,qy,qz = struct.unpack('<4d', f.read(32))
            tx,ty,tz = struct.unpack('<3d', f.read(24))
            cam_id = struct.unpack('<i', f.read(4))[0]
            name = b''
            while True:
                c = f.read(1)
                if c == b'\\x00': break
                name += c
            npts = struct.unpack('<Q', f.read(8))[0]
            f.read(npts * 24)  # skip point2D data

            # Build viewmat
            R = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
            t = np.array([tx,ty,tz])
            vm = np.eye(4)
            vm[:3,:3] = R; vm[:3,3] = t
            viewmat = torch.tensor(vm, dtype=torch.float32, device='cuda')

            depth, rgb, alpha = render_gsplat(gs, viewmat, K, w, h)

            # Save RGB render
            rgb_img = Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8))
            rgb_img.save(str(preview_dir / f'preview_02_render_view{idx:02d}.jpg'), quality=90)

            # Save depth colormap
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(w/100, h/100), dpi=100)
            valid = depth[alpha > 0.5]
            vmin, vmax = (valid.min(), valid.max()) if len(valid) > 0 else (0, 1)
            ax.imshow(depth, cmap='turbo', vmin=vmin, vmax=vmax)
            ax.axis('off')
            fig.savefig(str(preview_dir / f'preview_03_depth_view{idx:02d}.jpg'), bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(fig)

        print(f'Saved {min(nimgs,8)} viewpoint-matched RGB renders + depth maps')
else:
    print('No COLMAP sparse data found, skipping viewpoint renders')
"

# 3. After mesh: render in Blender
blender --background --factory-startup --python-expr "
import bpy, mathutils
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
bpy.ops.wm.obj_import(filepath='$JOB_DIR/objects/meshes/full_scene/full_scene.obj')
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        mat = bpy.data.materials.new(name='Viz'); mat.diffuse_color = (0.4, 0.6, 0.8, 1.0); obj.data.materials.append(mat)
bpy.ops.object.camera_add(location=(10,-10,8))
cam=bpy.context.active_object
cam.rotation_euler=(mathutils.Vector((0,0,2))-cam.location).to_track_quat('-Z','Y').to_euler()
bpy.context.scene.camera=cam
bpy.ops.object.light_add(type='SUN',location=(5,-5,10))
bpy.context.active_object.data.energy=2.5
bpy.context.scene.render.engine='CYCLES'
bpy.context.scene.cycles.device='CPU'
bpy.context.scene.cycles.samples=32
bpy.context.scene.render.resolution_x=1280
bpy.context.scene.render.resolution_y=720
bpy.context.scene.render.filepath='$JOB_DIR/previews/preview_04_mesh.jpg'
bpy.context.scene.render.image_settings.file_format='JPEG'
bpy.ops.render.render(write_still=True)
"
```

**The most important previews are the viewpoint-matched RGB renders and depth maps.**
These show the user exactly what the trained model looks like from the original camera positions.
