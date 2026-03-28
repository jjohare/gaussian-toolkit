"""Blender render script for TSDF mesh preview.

Usage:
    blender --background --factory-startup --python render_tsdf_preview.py -- \
        --input /path/to/tsdf_mesh.glb \
        --output /path/to/preview.png \
        --width 1920 --height 1080
"""

import argparse
import math
import sys

import bpy


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)


def setup_renderer(width: int, height: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True


def import_mesh(filepath: str):
    ext = filepath.lower().rsplit(".", 1)[-1]
    if ext in ("glb", "gltf"):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == "obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    imported = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not imported:
        raise RuntimeError("No mesh objects imported")
    return imported


def center_and_frame(objects):
    """Center imported objects and compute bounding sphere."""
    # Join all mesh objects
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()

    obj = bpy.context.active_object

    # Center at origin
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)
    bpy.context.view_layer.update()

    # Compute bounding sphere radius
    verts = obj.data.vertices
    max_dist = max(v.co.length for v in verts)
    return obj, max_dist


def setup_material(obj):
    """Apply a clay-like material for mesh visualization."""
    mat = bpy.data.materials.new(name="TSDFClay")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.75, 0.72, 0.68, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Metallic"].default_value = 0.0
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Check if mesh has vertex colors
    if obj.data.color_attributes:
        attr_node = nodes.new("ShaderNodeVertexColor")
        attr_node.layer_name = obj.data.color_attributes[0].name
        links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def setup_camera(radius: float):
    """Position camera to frame the mesh."""
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.type = "PERSP"
    cam_data.lens = 50
    cam_data.clip_start = radius * 0.01
    cam_data.clip_end = radius * 20

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Position at 45-degree elevation, looking at origin
    dist = radius * 2.5
    elev = math.radians(25)
    azim = math.radians(35)

    cam_obj.location = (
        dist * math.cos(elev) * math.cos(azim),
        dist * math.cos(elev) * math.sin(azim),
        dist * math.sin(elev),
    )

    # Point camera at origin
    direction = cam_obj.location.copy()
    direction.negate()
    rot = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot.to_euler()

    return cam_obj


def setup_lighting(radius: float):
    """Three-point lighting setup."""
    lights = []

    # Key light (warm)
    key = bpy.data.lights.new(name="Key", type="AREA")
    key.energy = radius * radius * 2.0
    key.color = (1.0, 0.95, 0.9)
    key.size = radius
    key_obj = bpy.data.objects.new("Key", key)
    key_obj.location = (radius * 1.5, -radius * 1.0, radius * 2.0)
    bpy.context.scene.collection.objects.link(key_obj)
    lights.append(key_obj)

    # Fill light (cool)
    fill = bpy.data.lights.new(name="Fill", type="AREA")
    fill.energy = radius * radius * 0.8
    fill.color = (0.9, 0.93, 1.0)
    fill.size = radius * 1.5
    fill_obj = bpy.data.objects.new("Fill", fill)
    fill_obj.location = (-radius * 1.5, radius * 0.5, radius * 1.0)
    bpy.context.scene.collection.objects.link(fill_obj)
    lights.append(fill_obj)

    # Rim light
    rim = bpy.data.lights.new(name="Rim", type="AREA")
    rim.energy = radius * radius * 1.5
    rim.color = (1.0, 1.0, 1.0)
    rim.size = radius * 0.5
    rim_obj = bpy.data.objects.new("Rim", rim)
    rim_obj.location = (-radius * 0.5, radius * 2.0, radius * 1.5)
    bpy.context.scene.collection.objects.link(rim_obj)
    lights.append(rim_obj)

    # Point all lights at origin
    for l_obj in lights:
        direction = l_obj.location.copy()
        direction.negate()
        rot = direction.to_track_quat("-Z", "Y")
        l_obj.rotation_euler = rot.to_euler()

    # Environment light
    world = bpy.data.worlds.new(name="World")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    bg = nodes.get("Background")
    if bg:
        bg.inputs["Strength"].default_value = 0.3
        bg.inputs["Color"].default_value = (0.15, 0.17, 0.2, 1.0)
    bpy.context.scene.world = world


def setup_ground_plane(radius: float):
    """Add a subtle ground shadow catcher."""
    bpy.ops.mesh.primitive_plane_add(size=radius * 6, location=(0, 0, -radius * 0.02))
    plane = bpy.context.active_object
    plane.is_shadow_catcher = True


def render(output_path: str):
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered to: {output_path}")


def main():
    # Parse args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input mesh file (GLB/OBJ)")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args(argv)

    clear_scene()
    setup_renderer(args.width, args.height)

    objects = import_mesh(args.input)
    obj, radius = center_and_frame(objects)
    setup_material(obj)
    setup_camera(radius)
    setup_lighting(radius)
    setup_ground_plane(radius)
    render(args.output)


if __name__ == "__main__":
    main()
