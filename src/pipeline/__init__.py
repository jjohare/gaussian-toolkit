# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""USD scene assembly pipeline for Gaussian Toolkit.

Modules that depend on pxr (usd-core) are imported lazily so that the
COLMAP parser and coordinate transform utilities remain usable without
the USD runtime installed.
"""

from .colmap_parser import (
    ColmapCamera,
    ColmapImage,
    ColmapPoint3D,
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
)
from .coordinate_transform import (
    CoordinateTransformer,
    colmap_to_usd_position,
    colmap_to_usd_rotation,
)

__all__ = [
    "ColmapCamera",
    "ColmapImage",
    "ColmapPoint3D",
    "parse_cameras_txt",
    "parse_images_txt",
    "parse_points3d_txt",
    "CoordinateTransformer",
    "colmap_to_usd_position",
    "colmap_to_usd_rotation",
    "MaterialAssigner",
    "UsdSceneAssembler",
    "MeshExtractor",
    "MeshCleaner",
    "TextureBaker",
    "PipelineConfig",
    "PipelineOrchestrator",
    "PipelineState",
    "McpClient",
    # SAM2 video segmentation pipeline
    "SAM2Segmentor",
    "SegmentationResult",
    "PromptSpec",
    "MaskProjector",
    "ViewProjection",
    "FrameQualityAssessor",
    "FrameQuality",
    "Recommendation",
]


def __getattr__(name: str):
    if name == "MaterialAssigner":
        from .material_assigner import MaterialAssigner
        return MaterialAssigner
    if name == "UsdSceneAssembler":
        from .usd_assembler import UsdSceneAssembler
        return UsdSceneAssembler
    if name == "MeshExtractor":
        from .mesh_extractor import MeshExtractor
        return MeshExtractor
    if name == "MeshCleaner":
        from .mesh_cleaner import MeshCleaner
        return MeshCleaner
    if name == "TextureBaker":
        from .texture_baker import TextureBaker
        return TextureBaker
    if name == "PipelineConfig":
        from .config import PipelineConfig
        return PipelineConfig
    if name == "PipelineOrchestrator":
        from .orchestrator import PipelineOrchestrator
        return PipelineOrchestrator
    if name == "PipelineState":
        from .orchestrator import PipelineState
        return PipelineState
    if name == "McpClient":
        from .mcp_client import McpClient
        return McpClient
    # SAM2 video segmentation pipeline (lazy to avoid heavy torch import)
    if name == "SAM2Segmentor":
        from .sam2_segmentor import SAM2Segmentor
        return SAM2Segmentor
    if name == "SegmentationResult":
        from .sam2_segmentor import SegmentationResult
        return SegmentationResult
    if name == "PromptSpec":
        from .sam2_segmentor import PromptSpec
        return PromptSpec
    if name == "MaskProjector":
        from .mask_projector import MaskProjector
        return MaskProjector
    if name == "ViewProjection":
        from .mask_projector import ViewProjection
        return ViewProjection
    if name == "FrameQualityAssessor":
        from .frame_quality import FrameQualityAssessor
        return FrameQualityAssessor
    if name == "FrameQuality":
        from .frame_quality import FrameQuality
        return FrameQuality
    if name == "Recommendation":
        from .frame_quality import Recommendation
        return Recommendation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
