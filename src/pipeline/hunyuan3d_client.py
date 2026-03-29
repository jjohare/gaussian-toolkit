# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hunyuan3D 2.0 client for image-to-3D mesh reconstruction.

Integrates multi-view rendering from Gaussian splats with the
Hunyuan3D 2.0 ComfyUI nodes to produce textured GLB meshes.

Supports two modes:
  1. Multi-view: Renders 4 canonical views (front/left/back/right) from
     a Gaussian PLY, feeds them through Hunyuan3Dv2ConditioningMultiView
     for high-fidelity shape generation.
  2. Single-view: Uses a single rendered view with Hunyuan3Dv2Conditioning
     as a faster fallback.

Usage::

    from pipeline.hunyuan3d_client import Hunyuan3DClient

    client = Hunyuan3DClient(comfyui_url="http://192.168.2.48:8189")
    result = client.reconstruct_from_gaussians("object.ply")
    result.mesh.export("output.glb")
"""

from __future__ import annotations

import copy
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
import trimesh

from .multiview_renderer import (
    MultiViewRenderer,
    RenderConfig,
    ViewResult,
    GaussianData,
    load_gaussian_ply,
)

logger = logging.getLogger(__name__)

WORKFLOW_DIR = Path(__file__).parent / "workflows"
HUNYUAN3D_MV_WORKFLOW = WORKFLOW_DIR / "hunyuan3d_multiview.json"
HUNYUAN3D_SV_WORKFLOW = WORKFLOW_DIR / "hunyuan3d_singleview.json"


# ---------------------------------------------------------------------------
# Model specifications for downloading
# ---------------------------------------------------------------------------

HUNYUAN3D_MODELS = {
    "dit-v2-mv": {
        "hf_repo": "tencent/Hunyuan3D-2mv",
        "hf_path": "hunyuan3d-dit-v2-mv/model.fp16.safetensors",
        "comfyui_name": "hunyuan3d-dit-v2-mv.safetensors",
        "model_type": "checkpoints",
        "size_gb": 4.93,
        "description": "Multi-view shape generation (1.1B params)",
    },
    "dit-v2-mv-turbo": {
        "hf_repo": "tencent/Hunyuan3D-2mv",
        "hf_path": "hunyuan3d-dit-v2-mv-turbo/model.fp16.safetensors",
        "comfyui_name": "hunyuan3d-dit-v2-mv-turbo.safetensors",
        "model_type": "checkpoints",
        "size_gb": 4.93,
        "description": "Multi-view turbo (step distilled, 1.1B params)",
    },
    "dit-v2": {
        "hf_repo": "tencent/Hunyuan3D-2",
        "hf_path": "hunyuan3d-dit-v2-0/model.safetensors",
        "comfyui_name": "hunyuan3d-dit-v2.safetensors",
        "model_type": "checkpoints",
        "size_gb": 4.93,
        "description": "Single-view shape generation (1.1B params)",
    },
    "dit-v2-fast": {
        "hf_repo": "tencent/Hunyuan3D-2",
        "hf_path": "hunyuan3d-dit-v2-0-fast/model.safetensors",
        "comfyui_name": "hunyuan3d-dit-v2-fast.safetensors",
        "model_type": "checkpoints",
        "size_gb": 4.93,
        "description": "Single-view fast (guidance distilled, 1.1B params)",
    },
    "vae-v2": {
        "hf_repo": "tencent/Hunyuan3D-2",
        "hf_path": "hunyuan3d-vae-v2-0/model.safetensors",
        "comfyui_name": "hunyuan3d-vae-v2.safetensors",
        "model_type": "vae",
        "size_gb": 2.5,
        "description": "3D VAE decoder (octree voxelization)",
    },
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Hunyuan3DResult:
    """Result from a Hunyuan3D reconstruction."""
    mesh: Optional[trimesh.Trimesh] = None
    glb_data: Optional[bytes] = None
    views_rendered: int = 0
    duration_seconds: float = 0.0
    backend: str = "hunyuan3d-mv"
    prompt_id: str = ""
    error: Optional[str] = None
    output_paths: dict[str, str] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return len(self.mesh.vertices) if self.mesh is not None else 0

    @property
    def face_count(self) -> int:
        return len(self.mesh.faces) if self.mesh is not None else 0

    @property
    def has_texture(self) -> bool:
        if self.mesh is None:
            return False
        return self.mesh.visual is not None and hasattr(self.mesh.visual, "material")


@dataclass
class QualityPreset:
    """Named quality configuration for Hunyuan3D generation."""
    name: str
    steps: int
    cfg: float
    resolution: int
    octree_resolution: int
    num_chunks: int
    sampler: str = "euler"
    scheduler: str = "sgm_uniform"


QUALITY_PRESETS = {
    "draft": QualityPreset(
        name="draft", steps=20, cfg=5.0, resolution=2048,
        octree_resolution=128, num_chunks=4000,
    ),
    "standard": QualityPreset(
        name="standard", steps=50, cfg=5.5, resolution=3072,
        octree_resolution=256, num_chunks=8000,
    ),
    "high": QualityPreset(
        name="high", steps=75, cfg=6.0, resolution=4096,
        octree_resolution=384, num_chunks=16000,
    ),
    "ultra": QualityPreset(
        name="ultra", steps=100, cfg=6.5, resolution=4096,
        octree_resolution=512, num_chunks=32000,
    ),
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class Hunyuan3DClient:
    """Client for Hunyuan3D 2.0 3D generation via ComfyUI.

    Parameters
    ----------
    comfyui_url : str
        Native ComfyUI URL (port 8189).
    api_url : str | None
        SaladTechnologies API wrapper URL (port 3001). If None, uses
        native ComfyUI API only.
    timeout : int
        Maximum seconds to wait for generation.
    poll_interval : float
        Seconds between status polls.
    quality : str
        Quality preset name: "draft", "standard", "high", "ultra".
    """

    def __init__(
        self,
        comfyui_url: str = "http://192.168.2.48:8189",
        api_url: str | None = "http://192.168.2.48:3001",
        timeout: int = 600,
        poll_interval: float = 2.0,
        quality: str = "standard",
    ):
        self.comfyui_url = comfyui_url.rstrip("/")
        self.api_url = api_url.rstrip("/") if api_url else None
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.quality = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])
        self.session = requests.Session()

        self._renderer: MultiViewRenderer | None = None

    # ---------------------------------------------------------------
    # ComfyUI interaction (native API)
    # ---------------------------------------------------------------

    def health_check(self) -> bool:
        """Check if ComfyUI is reachable."""
        try:
            resp = self.session.get(
                f"{self.comfyui_url}/system_stats", timeout=10,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _upload_image(self, image_path: Path) -> str:
        """Upload image to ComfyUI input directory."""
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            resp = self.session.post(
                f"{self.comfyui_url}/upload/image",
                files=files, timeout=30,
            )
        resp.raise_for_status()
        return resp.json().get("name", image_path.name)

    def _submit_prompt(self, prompt: dict) -> str:
        """Submit workflow prompt, return prompt_id."""
        resp = self.session.post(
            f"{self.comfyui_url}/prompt",
            json={"prompt": prompt},
            timeout=30,
        )
        data = resp.json()
        if "error" in data:
            node_errors = data.get("node_errors", {})
            details = "; ".join(
                f"node {nid}: {e.get('errors', e)}"
                for nid, e in node_errors.items()
            ) if node_errors else str(data.get("error"))
            raise RuntimeError(f"ComfyUI validation error: {details}")
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"No prompt_id in response: {data}")
        return prompt_id

    def _poll_completion(self, prompt_id: str) -> dict:
        """Poll ComfyUI history until prompt completes."""
        deadline = time.monotonic() + self.timeout
        last_log = 0.0
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            resp = self.session.get(
                f"{self.comfyui_url}/history/{prompt_id}",
                timeout=10,
            )
            hist = resp.json()
            if prompt_id not in hist:
                if time.monotonic() - last_log > 30:
                    logger.info("Waiting for Hunyuan3D prompt %s...", prompt_id[:8])
                    last_log = time.monotonic()
                continue

            entry = hist[prompt_id]
            status = entry.get("status", {}).get("status_str", "unknown")

            if status == "success":
                return entry
            if status == "error":
                messages = entry.get("status", {}).get("messages", [])
                raise RuntimeError(f"Hunyuan3D execution error: {messages}")

            if time.monotonic() - last_log > 30:
                logger.info("Hunyuan3D %s status: %s", prompt_id[:8], status)
                last_log = time.monotonic()

        raise TimeoutError(
            f"Hunyuan3D prompt {prompt_id} timed out after {self.timeout}s"
        )

    def _download_file(self, filepath: str) -> bytes:
        """Download output file from ComfyUI."""
        filename = Path(filepath).name
        subfolder = str(Path(filepath).parent)

        for file_type in ("output", "temp", "input"):
            resp = self.session.get(
                f"{self.comfyui_url}/view",
                params={
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": file_type,
                },
                timeout=60,
            )
            if resp.status_code == 200 and len(resp.content) > 0:
                return resp.content

        # Try mesh-specific output subdirectories
        for sub in ("", "mesh", "hunyuan3d"):
            resp = self.session.get(
                f"{self.comfyui_url}/view",
                params={
                    "filename": filename,
                    "subfolder": sub,
                    "type": "output",
                },
                timeout=60,
            )
            if resp.status_code == 200 and len(resp.content) > 0:
                return resp.content

        raise FileNotFoundError(f"Cannot download {filepath} from ComfyUI")

    def _extract_output_paths(self, history: dict) -> dict[str, str]:
        """Extract file paths from ComfyUI history output nodes."""
        outputs = history.get("outputs", {})
        paths: dict[str, str] = {}

        for node_id, node_output in outputs.items():
            # GLB/mesh file outputs
            if "text" in node_output:
                text_items = node_output["text"]
                if isinstance(text_items, list):
                    for item in text_items:
                        if isinstance(item, str) and any(
                            item.endswith(ext) for ext in (".glb", ".obj", ".ply")
                        ):
                            ext = Path(item).suffix.lstrip(".")
                            paths[f"node_{node_id}_{ext}"] = item
                elif isinstance(text_items, str) and any(
                    text_items.endswith(ext) for ext in (".glb", ".obj", ".ply")
                ):
                    ext = Path(text_items).suffix.lstrip(".")
                    paths[f"node_{node_id}_{ext}"] = text_items

            # SaveGLB / save node outputs
            if "meshes" in node_output:
                for mesh_info in node_output["meshes"]:
                    if isinstance(mesh_info, dict):
                        fname = mesh_info.get("filename", "")
                        sub = mesh_info.get("subfolder", "")
                        full = f"{sub}/{fname}" if sub else fname
                        paths[f"node_{node_id}_glb"] = full

            # Image outputs (preview renders)
            if "images" in node_output:
                for img_info in node_output["images"]:
                    if isinstance(img_info, dict):
                        fname = img_info.get("filename", "")
                        sub = img_info.get("subfolder", "")
                        full = f"{sub}/{fname}" if sub else fname
                        paths[f"node_{node_id}_image"] = full

        return paths

    # ---------------------------------------------------------------
    # GLB loading
    # ---------------------------------------------------------------

    def _load_glb(self, data: bytes) -> Optional[trimesh.Trimesh]:
        """Load GLB binary into trimesh."""
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            scene = trimesh.load(tmp.name, file_type="glb", force="scene")

        if isinstance(scene, trimesh.Scene):
            meshes = [
                g for g in scene.geometry.values()
                if isinstance(g, trimesh.Trimesh)
            ]
            if not meshes:
                return None
            if len(meshes) == 1:
                return meshes[0]
            return trimesh.util.concatenate(meshes)
        if isinstance(scene, trimesh.Trimesh):
            return scene
        return None

    # ---------------------------------------------------------------
    # Workflow construction
    # ---------------------------------------------------------------

    def _load_workflow(self, path: Path) -> dict:
        """Load workflow JSON template."""
        with open(path) as f:
            return json.load(f)

    def _build_multiview_prompt(
        self,
        front_filename: str,
        left_filename: str,
        back_filename: str,
        right_filename: str,
        seed: int = 42,
        model_name: str | None = None,
    ) -> dict:
        """Build the multi-view Hunyuan3D workflow prompt.

        Parameters
        ----------
        front_filename, left_filename, back_filename, right_filename : str
            ComfyUI-uploaded filenames for each view.
        seed : int
            Generation seed.
        model_name : str | None
            Override the checkpoint name (e.g. for turbo variant).
        """
        prompt = self._load_workflow(HUNYUAN3D_MV_WORKFLOW)

        # Set view images
        prompt["2"]["inputs"]["image"] = front_filename
        prompt["3"]["inputs"]["image"] = left_filename
        prompt["4"]["inputs"]["image"] = back_filename
        prompt["5"]["inputs"]["image"] = right_filename

        # Override checkpoint if specified
        if model_name:
            prompt["1"]["inputs"]["ckpt_name"] = model_name

        # Apply quality preset
        q = self.quality
        prompt["11"]["inputs"]["resolution"] = q.resolution
        prompt["12"]["inputs"]["seed"] = seed
        prompt["12"]["inputs"]["steps"] = q.steps
        prompt["12"]["inputs"]["cfg"] = q.cfg
        prompt["12"]["inputs"]["sampler_name"] = q.sampler
        prompt["12"]["inputs"]["scheduler"] = q.scheduler
        prompt["13"]["inputs"]["num_chunks"] = q.num_chunks
        prompt["13"]["inputs"]["octree_resolution"] = q.octree_resolution

        return prompt

    def _build_singleview_prompt(
        self,
        image_filename: str,
        seed: int = 42,
        model_name: str | None = None,
    ) -> dict:
        """Build the single-view Hunyuan3D workflow prompt."""
        prompt = self._load_workflow(HUNYUAN3D_SV_WORKFLOW)

        prompt["2"]["inputs"]["image"] = image_filename

        if model_name:
            prompt["1"]["inputs"]["ckpt_name"] = model_name

        q = self.quality
        prompt["5"]["inputs"]["resolution"] = q.resolution
        prompt["6"]["inputs"]["seed"] = seed
        prompt["6"]["inputs"]["steps"] = q.steps
        prompt["6"]["inputs"]["cfg"] = q.cfg
        prompt["6"]["inputs"]["sampler_name"] = q.sampler
        prompt["6"]["inputs"]["scheduler"] = q.scheduler
        prompt["7"]["inputs"]["num_chunks"] = q.num_chunks
        prompt["7"]["inputs"]["octree_resolution"] = q.octree_resolution

        return prompt

    # ---------------------------------------------------------------
    # Multi-view rendering
    # ---------------------------------------------------------------

    def _get_renderer(self) -> MultiViewRenderer:
        """Get or create the multi-view renderer."""
        if self._renderer is None:
            self._renderer = MultiViewRenderer(RenderConfig(
                image_size=512,
                num_views=4,
                azimuth_preset="hunyuan_mv",
                fov_deg=49.13,
                camera_distance=2.5,
                sh_degree=3,
                center_object=True,
                scale_to_unit=True,
            ))
        return self._renderer

    def _render_and_save_views(
        self,
        ply_path: Path,
        work_dir: Path,
    ) -> dict[str, Path]:
        """Render multi-view images and save to work directory.

        Returns
        -------
        dict with keys "front", "left", "back", "right" -> Path
        """
        renderer = self._get_renderer()
        views = renderer.render(ply_path, output_dir=work_dir)

        # Map view labels to file paths
        view_map: dict[str, Path] = {}
        for v in views:
            label = v.camera.label
            path = work_dir / f"{label}.png"
            if label in ("front", "left", "back", "right"):
                view_map[label] = path

        # Ensure all 4 required views exist
        required = {"front", "left", "back", "right"}
        missing = required - set(view_map.keys())
        if missing:
            logger.warning(
                "Missing views %s, using front view as fallback", missing,
            )
            front_path = view_map.get("front")
            if front_path is None and views:
                front_path = work_dir / f"{views[0].camera.name}.png"
            if front_path and front_path.exists():
                for m in missing:
                    view_map[m] = front_path

        return view_map

    # ---------------------------------------------------------------
    # Public reconstruction methods
    # ---------------------------------------------------------------

    def reconstruct_multiview(
        self,
        ply_path: str | Path,
        seed: int = 42,
        turbo: bool = False,
        work_dir: str | Path | None = None,
    ) -> Hunyuan3DResult:
        """Reconstruct a 3D mesh from Gaussian splat PLY using multi-view.

        Parameters
        ----------
        ply_path : str | Path
            Path to the 3DGS PLY file.
        seed : int
            Generation seed.
        turbo : bool
            Use turbo (step-distilled) model for faster generation.
        work_dir : str | Path | None
            Directory for intermediate files. Auto-created if None.

        Returns
        -------
        Hunyuan3DResult
            Contains the reconstructed mesh and metadata.
        """
        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY not found: {ply_path}")

        logger.info("Hunyuan3D multi-view reconstruction: %s", ply_path.name)
        t0 = time.monotonic()

        # Prepare working directory
        if work_dir is None:
            work_dir_obj = tempfile.mkdtemp(prefix="hunyuan3d_")
            work_dir = Path(work_dir_obj)
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        # Render multi-view images
        logger.info("Rendering multi-view images...")
        view_paths = self._render_and_save_views(ply_path, work_dir)
        logger.info("Rendered views: %s", list(view_paths.keys()))

        # Upload all views to ComfyUI
        uploaded: dict[str, str] = {}
        for view_name, view_path in view_paths.items():
            uploaded[view_name] = self._upload_image(view_path)
            logger.info("Uploaded %s as %s", view_name, uploaded[view_name])

        # Build and submit workflow
        model_name = None
        if turbo:
            model_name = "hunyuan3d-dit-v2-mv-turbo.safetensors"

        prompt = self._build_multiview_prompt(
            front_filename=uploaded["front"],
            left_filename=uploaded["left"],
            back_filename=uploaded["back"],
            right_filename=uploaded["right"],
            seed=seed,
            model_name=model_name,
        )

        prompt_id = self._submit_prompt(prompt)
        logger.info("Submitted Hunyuan3D MV prompt %s", prompt_id)

        # Poll for completion
        history = self._poll_completion(prompt_id)
        elapsed = time.monotonic() - t0
        logger.info("Hunyuan3D MV completed in %.1fs", elapsed)

        # Extract and download outputs
        output_paths = self._extract_output_paths(history)
        logger.info("Output paths: %s", output_paths)

        result = Hunyuan3DResult(
            backend="hunyuan3d-mv-turbo" if turbo else "hunyuan3d-mv",
            views_rendered=len(view_paths),
            duration_seconds=elapsed,
            prompt_id=prompt_id,
            output_paths=output_paths,
        )

        # Download GLB files
        for key, filepath in output_paths.items():
            if filepath.lower().endswith(".glb"):
                try:
                    data = self._download_file(filepath)
                    result.glb_data = data
                    result.mesh = self._load_glb(data)
                    if result.mesh is not None:
                        logger.info(
                            "Loaded mesh: %d verts, %d faces",
                            result.vertex_count, result.face_count,
                        )
                except (FileNotFoundError, requests.RequestException) as e:
                    logger.warning("Could not download %s: %s", filepath, e)

        if result.mesh is None:
            result.error = "No mesh data in outputs"
            logger.warning("Hunyuan3D MV: %s. Paths: %s", result.error, output_paths)

        return result

    def reconstruct_singleview(
        self,
        image_path: str | Path,
        seed: int = 42,
        fast: bool = False,
    ) -> Hunyuan3DResult:
        """Reconstruct from a single image (fallback mode).

        Parameters
        ----------
        image_path : str | Path
            Path to input image (PNG/JPG).
        seed : int
            Generation seed.
        fast : bool
            Use guidance-distilled fast model.

        Returns
        -------
        Hunyuan3DResult
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info("Hunyuan3D single-view reconstruction: %s", image_path.name)
        t0 = time.monotonic()

        image_filename = self._upload_image(image_path)
        logger.info("Uploaded image as: %s", image_filename)

        model_name = None
        if fast:
            model_name = "hunyuan3d-dit-v2-fast.safetensors"

        prompt = self._build_singleview_prompt(
            image_filename=image_filename,
            seed=seed,
            model_name=model_name,
        )

        prompt_id = self._submit_prompt(prompt)
        logger.info("Submitted Hunyuan3D SV prompt %s", prompt_id)

        history = self._poll_completion(prompt_id)
        elapsed = time.monotonic() - t0
        logger.info("Hunyuan3D SV completed in %.1fs", elapsed)

        output_paths = self._extract_output_paths(history)

        result = Hunyuan3DResult(
            backend="hunyuan3d-sv-fast" if fast else "hunyuan3d-sv",
            views_rendered=1,
            duration_seconds=elapsed,
            prompt_id=prompt_id,
            output_paths=output_paths,
        )

        for key, filepath in output_paths.items():
            if filepath.lower().endswith(".glb"):
                try:
                    data = self._download_file(filepath)
                    result.glb_data = data
                    result.mesh = self._load_glb(data)
                except (FileNotFoundError, requests.RequestException) as e:
                    logger.warning("Could not download %s: %s", filepath, e)

        if result.mesh is None:
            result.error = "No mesh data in outputs"

        return result

    def reconstruct_from_gaussians(
        self,
        ply_path: str | Path,
        seed: int = 42,
        turbo: bool = False,
        fallback_singleview: bool = True,
        work_dir: str | Path | None = None,
    ) -> Hunyuan3DResult:
        """Main entry point: reconstruct mesh from Gaussian PLY.

        Tries multi-view first (4 canonical views rendered from the
        Gaussian representation), falling back to single-view if needed.

        Parameters
        ----------
        ply_path : str | Path
            Path to the 3DGS PLY file.
        seed : int
            Generation seed.
        turbo : bool
            Use turbo model variant.
        fallback_singleview : bool
            If multi-view fails, try single-view with rendered front view.
        work_dir : str | Path | None
            Working directory for intermediate files.

        Returns
        -------
        Hunyuan3DResult
        """
        try:
            result = self.reconstruct_multiview(
                ply_path, seed=seed, turbo=turbo, work_dir=work_dir,
            )
            if result.mesh is not None:
                return result
        except Exception as e:
            logger.error("Multi-view reconstruction failed: %s", e)
            if not fallback_singleview:
                return Hunyuan3DResult(
                    backend="hunyuan3d-mv",
                    error=str(e),
                )

        # Fallback: render a front view and use single-view
        logger.info("Falling back to single-view reconstruction")
        try:
            ply_path = Path(ply_path)
            if work_dir is None:
                work_dir = Path(tempfile.mkdtemp(prefix="hunyuan3d_sv_"))
            else:
                work_dir = Path(work_dir)

            renderer = self._get_renderer()
            views = renderer.render(ply_path, output_dir=work_dir)

            # Use the front view (or first available)
            front_path = work_dir / "front.png"
            if not front_path.exists() and views:
                front_path = work_dir / f"{views[0].camera.name}.png"

            return self.reconstruct_singleview(
                front_path, seed=seed, fast=turbo,
            )
        except Exception as e2:
            logger.error("Single-view fallback also failed: %s", e2)
            return Hunyuan3DResult(
                backend="hunyuan3d-sv",
                error=str(e2),
            )

    # ---------------------------------------------------------------
    # Model management
    # ---------------------------------------------------------------

    def probe_available_models(self) -> dict[str, list[str]]:
        """Query ComfyUI for available Hunyuan3D models.

        Returns
        -------
        dict with keys "checkpoints", "vae" -> list of model names.
        """
        result: dict[str, list[str]] = {"checkpoints": [], "vae": []}

        for node_name, key_name, result_key in [
            ("ImageOnlyCheckpointLoader", "ckpt_name", "checkpoints"),
            ("VAELoader", "vae_name", "vae"),
        ]:
            try:
                resp = self.session.get(
                    f"{self.comfyui_url}/object_info/{node_name}",
                    timeout=10,
                )
                data = resp.json()
                names = (
                    data.get(node_name, {})
                    .get("input", {})
                    .get("required", {})
                    .get(key_name, [[]])[0]
                )
                if isinstance(names, list):
                    hunyuan = [n for n in names if "hunyuan3d" in n.lower()]
                    result[result_key] = hunyuan
            except Exception as e:
                logger.debug("Could not probe %s: %s", node_name, e)

        return result

    def check_model_availability(self) -> dict[str, bool]:
        """Check which Hunyuan3D models are available on the server.

        Returns
        -------
        dict mapping model key to availability boolean.
        """
        available = self.probe_available_models()
        all_names = available.get("checkpoints", []) + available.get("vae", [])

        status = {}
        for key, spec in HUNYUAN3D_MODELS.items():
            status[key] = spec["comfyui_name"] in all_names

        return status

    def download_model(
        self,
        model_key: str,
        hf_token: str | None = None,
    ) -> None:
        """Download a Hunyuan3D model to the ComfyUI server.

        Requires the SaladTechnologies API wrapper (port 3001) for
        the /download endpoint.

        Parameters
        ----------
        model_key : str
            Key from HUNYUAN3D_MODELS (e.g. "dit-v2-mv").
        hf_token : str | None
            HuggingFace token for gated models.
        """
        if self.api_url is None:
            raise RuntimeError(
                "api_url required for model downloads (SaladTechnologies wrapper)"
            )

        spec = HUNYUAN3D_MODELS.get(model_key)
        if spec is None:
            raise ValueError(
                f"Unknown model key: {model_key}. "
                f"Available: {list(HUNYUAN3D_MODELS.keys())}"
            )

        url = (
            f"https://huggingface.co/{spec['hf_repo']}/resolve/main/{spec['hf_path']}"
        )

        payload: dict[str, Any] = {
            "url": url,
            "model_type": spec["model_type"],
            "filename": spec["comfyui_name"],
            "wait": True,
        }
        if hf_token:
            payload["auth"] = {"type": "bearer", "token": hf_token}

        logger.info(
            "Downloading %s (%.1f GB) -> %s/%s",
            model_key, spec["size_gb"],
            spec["model_type"], spec["comfyui_name"],
        )
        resp = self.session.post(
            f"{self.api_url}/download",
            json=payload,
            timeout=1800,  # 30 min for large models
        )
        resp.raise_for_status()
        logger.info("Downloaded %s successfully", model_key)

    def ensure_models(
        self,
        multiview: bool = True,
        turbo: bool = False,
        hf_token: str | None = None,
    ) -> bool:
        """Ensure required models are available, downloading if needed.

        Parameters
        ----------
        multiview : bool
            Require multi-view model.
        turbo : bool
            Require turbo variant.
        hf_token : str | None
            HuggingFace token for downloads.

        Returns
        -------
        bool
            True if all required models are available.
        """
        status = self.check_model_availability()

        needed = []
        if multiview:
            key = "dit-v2-mv-turbo" if turbo else "dit-v2-mv"
            if not status.get(key, False):
                needed.append(key)
        else:
            key = "dit-v2-fast" if turbo else "dit-v2"
            if not status.get(key, False):
                needed.append(key)

        if not needed:
            logger.info("All required Hunyuan3D models available")
            return True

        if self.api_url is None:
            logger.warning(
                "Models needed but no api_url for download: %s", needed,
            )
            return False

        for key in needed:
            self.download_model(key, hf_token=hf_token)

        return True

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    def save_result(
        self,
        result: Hunyuan3DResult,
        output_dir: str | Path,
        prefix: str = "hunyuan3d",
    ) -> dict[str, Path]:
        """Save reconstruction result to disk.

        Parameters
        ----------
        result : Hunyuan3DResult
            Reconstruction result to save.
        output_dir : str | Path
            Output directory.
        prefix : str
            Filename prefix.

        Returns
        -------
        dict mapping output type to file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        if result.glb_data:
            path = output_dir / f"{prefix}.glb"
            path.write_bytes(result.glb_data)
            saved["glb"] = path
            logger.info("Saved GLB: %s (%.1f MB)", path, len(result.glb_data) / 1e6)

        if result.mesh is not None and "glb" not in saved:
            path = output_dir / f"{prefix}.glb"
            result.mesh.export(str(path), file_type="glb")
            saved["glb"] = path

        # Save metadata
        meta_path = output_dir / f"{prefix}_meta.json"
        meta = {
            "backend": result.backend,
            "views_rendered": result.views_rendered,
            "duration_seconds": result.duration_seconds,
            "vertex_count": result.vertex_count,
            "face_count": result.face_count,
            "has_texture": result.has_texture,
            "prompt_id": result.prompt_id,
            "quality_preset": self.quality.name,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        saved["meta"] = meta_path

        return saved
