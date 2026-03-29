# Gaussian Toolkit — Claude Code Agent Instructions

You are running inside the gaussian-toolkit Docker container on a dual RTX 6000 Ada system.

## Your Role
You orchestrate the video-to-scene pipeline. When a job appears in /data/output/, you:
1. Check job status via: curl http://localhost:7860/jobs
2. Pick up queued jobs
3. Run the pipeline stages manually for maximum quality control
4. Inspect results in Blender (DISPLAY=:1, VNC on :5901)
5. Adjust parameters and retry as needed
6. Report progress

## Available Tools
- LichtFeld Studio: /opt/gaussian-toolkit/build/LichtFeld-Studio
- COLMAP: /usr/local/bin/colmap
- Blender: /usr/local/bin/blender
- ComfyUI API: http://localhost:8188
- Python pipeline: /opt/gaussian-toolkit/src/pipeline/
- Web API: http://localhost:7860

## Key Commands
```bash
# Check jobs
curl -s http://localhost:7860/jobs | python3 -c "import sys,json; [print(f'{j[\"job_id\"]}: {j[\"state\"]}') for j in json.load(sys.stdin)]"

# Run pipeline manually for a job
cd /opt/gaussian-toolkit
python3 -m pipeline.cli video2scene /data/output/<job_id>/input.mp4 /data/output/<job_id>/

# Train with quality settings
LD_LIBRARY_PATH=/opt/gaussian-toolkit/build:$LD_LIBRARY_PATH \
  /opt/gaussian-toolkit/build/LichtFeld-Studio --headless \
  --data-path /data/output/<job_id>/colmap/undistorted \
  --output-path /data/output/<job_id>/model \
  --iter 30000 --strategy mrnf --eval --sh-degree 3

# Inspect in Blender
DISPLAY=:1 blender /data/output/<job_id>/scene/mesh.glb

# SAM3 segmentation
python3 -c "
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# ... see src/pipeline/sam3_segmentor.py
"
```

## Quality Targets
- Training: 30k+ iterations, MRNF strategy, loss < 0.02
- Segmentation: SAM3 text prompts for semantic labels
- Mesh: per-object textured meshes via Hunyuan3D or TSDF
- USD: hierarchical scene graph with variant sets
