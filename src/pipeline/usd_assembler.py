# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Master USD scene composition for the Gaussian Toolkit.

Composes individual per-object USD files (meshes and/or Gaussians) into a
hierarchical USD stage using composition arcs (references), variant sets,
camera prims from COLMAP extrinsics, and scene-level metadata.

Scene hierarchy::

    /World
        /Environment
            /Background
        /Objects
            /obj_001  (variant set: representation = {gaussian, mesh})
            /obj_002
            ...
        /Cameras
            /cam_001
            /cam_002
            ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdLux

from .colmap_parser import ColmapCamera, ColmapImage
from .coordinate_transform import (
    CoordinateTransformer,
    build_usd_transform_from_colmap,
    colmap_camera_world_position,
    colmap_to_usd_position,
)
from .material_assigner import MaterialAssigner


# ---------------------------------------------------------------------------
#  Object descriptor
# ---------------------------------------------------------------------------

@dataclass
class ObjectDescriptor:
    """Describes a single object to be placed in the scene."""

    name: str
    gaussian_usd_path: Optional[str] = None
    mesh_usd_path: Optional[str] = None
    centroid: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    diffuse_texture: Optional[str] = None
    diffuse_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    opacity: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Assembler
# ---------------------------------------------------------------------------

class UsdSceneAssembler:
    """Assembles a hierarchical USD scene from per-object USD exports.

    Typical usage::

        assembler = UsdSceneAssembler()
        assembler.add_object(ObjectDescriptor(
            name="chair",
            gaussian_usd_path="objects/chair_gs.usda",
            mesh_usd_path="objects/chair_mesh.usda",
            centroid=(1.0, 0.0, -2.0),
        ))
        assembler.set_colmap_cameras(transformer)
        assembler.write("/output/scene.usda")
    """

    def __init__(
        self,
        up_axis: str = "Y",
        meters_per_unit: float = 1.0,
    ) -> None:
        self._up_axis = up_axis
        self._meters_per_unit = meters_per_unit
        self._objects: List[ObjectDescriptor] = []
        self._transformer: Optional[CoordinateTransformer] = None
        self._scene_metadata: Dict[str, str] = {}

    # ------------------------------------------------------------------
    #  Builder API
    # ------------------------------------------------------------------

    def add_object(self, obj: ObjectDescriptor) -> None:
        """Register an object for scene assembly."""
        self._objects.append(obj)

    def set_colmap_cameras(self, transformer: CoordinateTransformer) -> None:
        """Provide COLMAP camera data for camera prim creation."""
        self._transformer = transformer

    def set_metadata(self, key: str, value: str) -> None:
        """Store arbitrary scene-level metadata."""
        self._scene_metadata[key] = value

    # ------------------------------------------------------------------
    #  Write
    # ------------------------------------------------------------------

    def write(self, output_path: str | Path) -> Usd.Stage:
        """Build and export the assembled USD scene.

        Returns the stage so callers can do additional inspection.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stage = Usd.Stage.CreateNew(str(output_path))
        self._configure_stage(stage)
        self._build_hierarchy(stage)
        self._place_objects(stage)
        self._place_cameras(stage)
        self._write_scene_metadata(stage)
        stage.GetRootLayer().Save()
        return stage

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _configure_stage(self, stage: Usd.Stage) -> None:
        """Set global stage metrics."""
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if self._up_axis == "Y" else UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, self._meters_per_unit)
        stage.SetDefaultPrim(stage.DefinePrim("/World"))

    def _build_hierarchy(self, stage: Usd.Stage) -> None:
        """Create the fixed scene hierarchy."""
        world = stage.DefinePrim("/World", "Xform")
        Usd.ModelAPI(world).SetKind(Kind.Tokens.assembly)

        stage.DefinePrim("/World/Environment", "Xform")
        stage.DefinePrim("/World/Environment/Background", "Xform")
        stage.DefinePrim("/World/Objects", "Xform")
        stage.DefinePrim("/World/Cameras", "Xform")
        stage.DefinePrim("/World/Materials", "Scope")

        # Default dome light in background
        dome = UsdLux.DomeLight.Define(stage, "/World/Environment/Background/DomeLight")
        dome.CreateIntensityAttr(1.0)

    def _place_objects(self, stage: Usd.Stage) -> None:
        """Place each registered object under /World/Objects with variant sets."""
        mat_assigner = MaterialAssigner(stage)

        for idx, obj in enumerate(self._objects):
            safe_name = _sanitize_prim_name(obj.name)
            obj_path = f"/World/Objects/{safe_name}"

            xform = UsdGeom.Xform.Define(stage, obj_path)
            prim = xform.GetPrim()
            Usd.ModelAPI(prim).SetKind(Kind.Tokens.component)

            # World-space transform from centroid
            ux, uy, uz = colmap_to_usd_position(*obj.centroid)
            _set_xform_from_components(
                xform, translate=(ux, uy, uz),
                rotate_quat=obj.rotation_quat,
                scale=obj.scale,
            )

            # Variant set: gaussian vs mesh
            has_gaussian = obj.gaussian_usd_path is not None
            has_mesh = obj.mesh_usd_path is not None

            if has_gaussian or has_mesh:
                vset = prim.GetVariantSets().AddVariantSet("representation")

                if has_gaussian:
                    vset.AddVariant("gaussian")
                    vset.SetVariantSelection("gaussian")
                    with vset.GetVariantEditContext():
                        ref_prim = stage.DefinePrim(f"{obj_path}/GaussianData", "Xform")
                        ref_prim.GetReferences().AddReference(
                            assetPath=obj.gaussian_usd_path,
                        )

                if has_mesh:
                    vset.AddVariant("mesh")
                    vset.SetVariantSelection("mesh")
                    with vset.GetVariantEditContext():
                        ref_prim = stage.DefinePrim(f"{obj_path}/MeshData", "Xform")
                        ref_prim.GetReferences().AddReference(
                            assetPath=obj.mesh_usd_path,
                        )

                # Default to gaussian if available, else mesh
                vset.SetVariantSelection("gaussian" if has_gaussian else "mesh")

            # Material
            mat_path = f"/World/Materials/{safe_name}_mat"
            mat_assigner.create_textured_material(
                mat_path,
                diffuse_texture=obj.diffuse_texture,
                diffuse_color=obj.diffuse_color,
                opacity=obj.opacity,
            )
            if has_mesh:
                # Bind material to the mesh variant's data prim
                mesh_data_path = f"{obj_path}/MeshData"
                mesh_prim = stage.GetPrimAtPath(mesh_data_path)
                if mesh_prim and mesh_prim.IsValid():
                    mat_assigner.bind_material(mesh_data_path, mat_path)

            # Per-object custom metadata
            for mk, mv in obj.metadata.items():
                prim.SetCustomDataByKey(mk, mv)

    def _place_cameras(self, stage: Usd.Stage) -> None:
        """Create USD camera prims from COLMAP extrinsics."""
        if self._transformer is None:
            return

        for image in self._transformer.images:
            cam_intrinsic = self._transformer.cameras.get(image.camera_id)
            if cam_intrinsic is None:
                continue

            safe_name = _sanitize_prim_name(
                Path(image.name).stem if image.name else f"cam_{image.image_id:04d}"
            )
            cam_path = f"/World/Cameras/{safe_name}"

            cam = UsdGeom.Camera.Define(stage, cam_path)

            # Transform
            xf_matrix = build_usd_transform_from_colmap(image)
            xform_op = cam.AddTransformOp()
            xform_op.Set(Gf.Matrix4d(
                xf_matrix[0][0], xf_matrix[0][1], xf_matrix[0][2], xf_matrix[0][3],
                xf_matrix[1][0], xf_matrix[1][1], xf_matrix[1][2], xf_matrix[1][3],
                xf_matrix[2][0], xf_matrix[2][1], xf_matrix[2][2], xf_matrix[2][3],
                xf_matrix[3][0], xf_matrix[3][1], xf_matrix[3][2], xf_matrix[3][3],
            ))

            # Intrinsics
            # USD Camera uses focal length in mm with a 36mm horizontal aperture (full-frame)
            horizontal_aperture_mm = 36.0
            focal_mm = (cam_intrinsic.focal_x / cam_intrinsic.width) * horizontal_aperture_mm
            vertical_aperture_mm = horizontal_aperture_mm * (cam_intrinsic.height / cam_intrinsic.width)

            cam.CreateFocalLengthAttr(focal_mm)
            cam.CreateHorizontalApertureAttr(horizontal_aperture_mm)
            cam.CreateVerticalApertureAttr(vertical_aperture_mm)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))

            # Store original COLMAP data as custom attributes
            prim = cam.GetPrim()
            prim.SetCustomDataByKey("colmap:camera_id", image.camera_id)
            prim.SetCustomDataByKey("colmap:image_id", image.image_id)
            prim.SetCustomDataByKey("colmap:image_name", image.name)
            prim.SetCustomDataByKey("colmap:model", cam_intrinsic.model)
            prim.SetCustomDataByKey("colmap:width", cam_intrinsic.width)
            prim.SetCustomDataByKey("colmap:height", cam_intrinsic.height)

    def _write_scene_metadata(self, stage: Usd.Stage) -> None:
        """Write scene-level metadata as custom data on /World."""
        world = stage.GetPrimAtPath("/World")
        if not world.IsValid():
            return
        world.SetCustomDataByKey("lichtfeld:pipeline_version", "1.0")
        world.SetCustomDataByKey("lichtfeld:up_axis", self._up_axis)
        world.SetCustomDataByKey("lichtfeld:meters_per_unit", self._meters_per_unit)
        world.SetCustomDataByKey("lichtfeld:object_count", len(self._objects))
        for key, value in self._scene_metadata.items():
            world.SetCustomDataByKey(f"lichtfeld:{key}", value)


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def _sanitize_prim_name(name: str) -> str:
    """Make a string safe for use as a USD prim name.

    USD prim names must start with a letter or underscore and contain only
    alphanumeric characters and underscores.
    """
    sanitized = ""
    for ch in name:
        if ch.isalnum() or ch == "_":
            sanitized += ch
        else:
            sanitized += "_"
    if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == "_"):
        sanitized = f"_{sanitized}"
    return sanitized


def _set_xform_from_components(
    xform: UsdGeom.Xform,
    *,
    translate: Tuple[float, float, float] = (0, 0, 0),
    rotate_quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
    scale: Tuple[float, float, float] = (1, 1, 1),
) -> None:
    """Set translate / rotate / scale ops on an Xformable."""
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    # USD quaternion orient expects (real, i, j, k)
    xform.AddOrientOp().Set(Gf.Quatf(
        rotate_quat[0], rotate_quat[1], rotate_quat[2], rotate_quat[3],
    ))
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))
