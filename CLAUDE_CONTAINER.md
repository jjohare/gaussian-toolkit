# Video-to-Scene Pipeline -- Claude Code Orchestration

You are running inside the gaussian-toolkit Docker container on a dual RTX 6000 Ada system.
**You are the orchestrator.** There is no state machine. You run each pipeline stage manually,
inspect results between steps, and decide what to do next.

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

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.ingest('/data/output/JOB_ID/input.mp4', fps=2.0)
print(result)
"
```

Update the web UI:
```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "ingest", "progress": 0.1, "message": "Extracting frames at 2fps"}'
```

**Inspect**: `ls /data/output/JOB_ID/frames/ | head -20`

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

## Step 3: Select best frames

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.select_frames('/data/output/JOB_ID/frames/', target=150)
print(result)
"
```

---

## Step 4: COLMAP reconstruction

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.reconstruct('/data/output/JOB_ID/frames_selected/')
print(result)
"
```

**Check**: Look for `sparse/0/cameras.bin` and `images/` in the colmap dir.

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

---

## Step 6: Segment (SAM3 object detection)

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.segment(
    '/data/output/JOB_ID/model/point_cloud.ply',
    '/data/output/JOB_ID/frames/'
)
print(result)
"
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

---

## Step 8: Generate meshes

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

---

## Step 11: Validate

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

---

## At each step: CHECK QUALITY

- Render a preview in Blender: `DISPLAY=:1 blender --background --python-expr "..."`
- If quality is poor, adjust parameters and re-run the stage
- The pipeline is not a script -- YOU decide what to do next
- You can skip stages, re-run stages, or change parameters between stages

## Quality Targets

- Training: 30k+ iterations, MRNF strategy, loss < 0.02
- Segmentation: SAM3 text prompts for semantic labels
- Mesh: per-object textured meshes via Hunyuan3D or TSDF
- USD: hierarchical scene graph with variant sets

## REST API for progress reporting

| Method | Endpoint | Body | Purpose |
|--------|----------|------|---------|
| POST | `/api/job/<id>/stage` | `{"stage": "...", "progress": 0.5, "message": "..."}` | Report stage progress |
| POST | `/api/job/<id>/stage/complete` | `{"stage": "...", "success": true}` | Mark stage done |
| POST | `/api/job/<id>/complete` | `{"success": true}` | Mark job done |
| GET | `/jobs` | -- | List all jobs |
| GET | `/status/<id>` | -- | Job detail |
