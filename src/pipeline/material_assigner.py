# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""USD material creation and assignment for the Gaussian Toolkit pipeline.

Creates UsdPreviewSurface materials with diffuse texture mapping,
opacity support, and proper UV reader shader connections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


class MaterialAssigner:
    """Creates and assigns UsdPreviewSurface materials on a USD stage.

    Usage::

        assigner = MaterialAssigner(stage)
        mat_path = assigner.create_textured_material(
            "/World/Materials/obj_01",
            diffuse_texture="/textures/obj_01_diffuse.png",
        )
        assigner.bind_material("/World/Objects/obj_01/mesh", mat_path)
    """

    def __init__(self, stage: Usd.Stage) -> None:
        self._stage = stage

    @property
    def stage(self) -> Usd.Stage:
        return self._stage

    # ------------------------------------------------------------------
    #  Material creation
    # ------------------------------------------------------------------

    def create_textured_material(
        self,
        material_path: str,
        *,
        diffuse_texture: Optional[str] = None,
        diffuse_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
        opacity: float = 1.0,
        roughness: float = 0.5,
        metallic: float = 0.0,
    ) -> str:
        """Create a UsdPreviewSurface material with optional diffuse texture.

        Args:
            material_path: USD path for the material, e.g. "/World/Materials/foo".
            diffuse_texture: Relative or absolute path to the diffuse texture file.
                If None, a solid diffuse_color is used instead.
            diffuse_color: Fallback RGB diffuse colour (each channel 0..1).
            opacity: Surface opacity (0 = fully transparent, 1 = fully opaque).
            roughness: Surface roughness for specular response.
            metallic: Metallic factor.

        Returns:
            The Sdf path string of the created material.
        """
        material = UsdShade.Material.Define(self._stage, material_path)

        # Surface shader
        shader_path = f"{material_path}/PreviewSurface"
        shader = UsdShade.Shader.Define(self._stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

        # Opacity
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        if opacity < 1.0:
            shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.01)

        # Connect surface output
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

        if diffuse_texture is not None:
            self._connect_diffuse_texture(
                material_path, shader, diffuse_texture, diffuse_color,
            )
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*diffuse_color)
            )

        return material_path

    def create_solid_material(
        self,
        material_path: str,
        *,
        diffuse_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
        opacity: float = 1.0,
        roughness: float = 0.5,
        metallic: float = 0.0,
    ) -> str:
        """Convenience wrapper: create a material with no texture."""
        return self.create_textured_material(
            material_path,
            diffuse_texture=None,
            diffuse_color=diffuse_color,
            opacity=opacity,
            roughness=roughness,
            metallic=metallic,
        )

    # ------------------------------------------------------------------
    #  Binding
    # ------------------------------------------------------------------

    def bind_material(
        self,
        prim_path: str,
        material_path: str,
    ) -> None:
        """Bind a material to a gprim at *prim_path*.

        The prim must already exist on the stage.
        """
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")
        material = UsdShade.Material.Get(self._stage, material_path)
        if not material.GetPrim().IsValid():
            raise ValueError(f"Material not found: {material_path}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _connect_diffuse_texture(
        self,
        material_path: str,
        surface_shader: UsdShade.Shader,
        texture_path: str,
        fallback_color: Tuple[float, float, float],
    ) -> None:
        """Wire up a diffuse texture through a UV reader."""
        # Texture reader
        tex_path = f"{material_path}/DiffuseTexture"
        tex_reader = UsdShade.Shader.Define(self._stage, tex_path)
        tex_reader.CreateIdAttr("UsdUVTexture")
        tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
        tex_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_reader.CreateInput(
            "fallback", Sdf.ValueTypeNames.Float4,
        ).Set(Gf.Vec4f(fallback_color[0], fallback_color[1], fallback_color[2], 1.0))

        # RGB output
        tex_output = tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        # Connect to surface diffuseColor
        surface_shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f,
        ).ConnectToSource(tex_output)

        # UV coordinate reader (primvar st)
        uv_path = f"{material_path}/UVReader"
        uv_reader = UsdShade.Shader.Define(self._stage, uv_path)
        uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
        uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv_output = uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        # Connect UV reader to texture sampler
        tex_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv_output
        )
