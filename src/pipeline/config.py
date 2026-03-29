# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pipeline configuration with typed defaults and JSON persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class IngestConfig:
    """Frame extraction parameters."""
    fps: float = 1.0
    max_image_size: int = 2000
    min_frames: int = 20
    max_frames: int = 500
    blur_threshold: float = 100.0
    exposure_range: tuple[float, float] = (0.1, 0.9)


@dataclass
class ReconstructConfig:
    """COLMAP / SplatReady reconstruction parameters."""
    method: str = "colmap"
    colmap_exe: str = "/usr/local/bin/colmap"
    use_fisheye: bool = False
    min_scale: float = 0.5
    matcher: str = "exhaustive"


@dataclass
class TrainingConfig:
    """Gaussian splatting training parameters."""
    max_iterations: int = 30000
    strategy: str = "mcmc"
    target_psnr: float = 25.0
    target_ssim: float = 0.85
    convergence_window: int = 500
    convergence_threshold: float = 0.001
    checkpoint_interval: int = 5000


@dataclass
class DecomposeConfig:
    """Scene decomposition parameters."""
    selection_method: str = "by_description"
    min_object_gaussians: int = 100
    descriptions: list[str] = field(default_factory=list)
    use_sam3: bool = True
    sam3_confidence_threshold: float = 0.5
    sam3_concepts: list[str] = field(default_factory=lambda: [
        "paintings", "frames", "sculptures", "furniture",
        "walls", "floor", "ceiling", "fixtures", "doorways",
    ])
    sam3_fallback_to_sam2: bool = True


@dataclass
class MeshConfig:
    """Mesh extraction parameters."""
    min_vertices: int = 100
    max_vertices: int = 500_000
    watertight_check: bool = True
    normal_consistency_threshold: float = 0.8


@dataclass
class InpaintConfig:
    """Background inpainting parameters."""
    method: str = "comfyui"
    comfyui_api_url: str = "http://192.168.2.48:3001"
    comfyui_direct_url: str = "http://192.168.2.48:8189"
    local_ip: str = "192.168.2.1"
    hf_token: str = ""
    model: str = "flux-fill"
    denoise: float = 0.75
    steps: int = 28
    guidance: float = 30.0
    auto_download_models: bool = True
    blend_radius: float = 2.0
    iterations: int = 10000


@dataclass
class Hunyuan3DConfig:
    """Hunyuan3D 2.0 mesh reconstruction parameters."""
    enabled: bool = True
    comfyui_url: str = "http://192.168.2.48:8189"
    api_url: str = "http://192.168.2.48:3001"
    quality: str = "standard"
    multiview: bool = True
    turbo: bool = False
    fallback_singleview: bool = True
    fallback_sam3d: bool = True
    timeout: int = 600
    seed: int = 42
    num_views: int = 4
    render_size: int = 512
    camera_distance: float = 2.5


@dataclass
class ExportConfig:
    """USD / final export parameters."""
    format: str = "usd"
    include_materials: bool = True
    coordinate_system: str = "right_handed_y_up"


@dataclass
class QualityConfig:
    """Quality gate thresholds."""
    gate1_min_psnr: float = 20.0
    gate1_min_ssim: float = 0.75
    gate2_min_mesh_vertices: int = 500
    gate2_normal_consistency: float = 0.7
    gate2_roundtrip_psnr: float = 18.0
    final_min_psnr: float = 22.0


@dataclass
class RetryConfig:
    """Retry behaviour per stage."""
    max_retries: int = 3
    parameter_adjustments: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "RECONSTRUCT": {"min_scale": 0.25, "matcher": "sequential"},
        "RECONSTRUCT:2": {"max_image_size": 1500},
        "QUALITY_GATE_1": {},
        "MESH_OBJECTS": {"max_vertices": 1_000_000},
        "RETRAIN_BG": {"max_iterations": 50000},
    })


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    mcp_endpoint: str = "http://127.0.0.1:45677/mcp"
    mcp_timeout: float = 30.0
    mcp_training_timeout: float = 600.0
    status_file: str = "pipeline_status.json"
    output_dir: str = "./output"

    ingest: IngestConfig = field(default_factory=IngestConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decompose: DecomposeConfig = field(default_factory=DecomposeConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    hunyuan3d: Hunyuan3DConfig = field(default_factory=Hunyuan3DConfig)
    inpaint: InpaintConfig = field(default_factory=InpaintConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        cfg = cls()
        direct_fields = {
            "mcp_endpoint", "mcp_timeout", "mcp_training_timeout",
            "status_file", "output_dir",
        }
        for k in direct_fields:
            if k in data:
                setattr(cfg, k, data[k])

        sub_map: dict[str, type] = {
            "ingest": IngestConfig,
            "reconstruct": ReconstructConfig,
            "training": TrainingConfig,
            "decompose": DecomposeConfig,
            "mesh": MeshConfig,
            "hunyuan3d": Hunyuan3DConfig,
            "inpaint": InpaintConfig,
            "export": ExportConfig,
            "quality": QualityConfig,
            "retry": RetryConfig,
        }
        for key, klass in sub_map.items():
            if key in data and isinstance(data[key], dict):
                sub = klass()
                for field_name, value in data[key].items():
                    if hasattr(sub, field_name):
                        current = getattr(sub, field_name)
                        if isinstance(current, tuple) and isinstance(value, list):
                            value = tuple(value)
                        setattr(sub, field_name, value)
                setattr(cfg, key, sub)
        return cfg

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.ingest.fps <= 0:
            errors.append("ingest.fps must be positive")
        if self.ingest.min_frames < 3:
            errors.append("ingest.min_frames must be >= 3")
        if self.ingest.min_frames > self.ingest.max_frames:
            errors.append("ingest.min_frames must be <= max_frames")
        if self.training.max_iterations < 1000:
            errors.append("training.max_iterations must be >= 1000")
        if self.training.target_psnr < 10:
            errors.append("training.target_psnr must be >= 10")
        if self.quality.gate1_min_psnr < 5:
            errors.append("quality.gate1_min_psnr must be >= 5")
        if self.retry.max_retries < 0:
            errors.append("retry.max_retries must be >= 0")
        if not self.mcp_endpoint.startswith("http"):
            errors.append("mcp_endpoint must be an HTTP URL")
        return errors
