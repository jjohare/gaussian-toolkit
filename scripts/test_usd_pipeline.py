#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test for the USD scene assembly pipeline.

Creates a test scene with 2 objects and a camera, writes to /tmp/test_scene.usda,
re-reads it, and prints the hierarchy.

Usage:
    python scripts/test_usd_pipeline.py
"""

from __future__ import annotations

import os
import sys
import tempfile

# Ensure project root is on the path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "src"))

from pathlib import Path


def _write_test_colmap_files(colmap_dir: Path) -> None:
    """Write minimal COLMAP cameras.txt, images.txt, and points3D.txt."""
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # cameras.txt: two PINHOLE cameras
    (colmap_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "1 PINHOLE 1920 1080 1200.0 1200.0 960.0 540.0\n"
        "2 SIMPLE_RADIAL 1280 720 900.0 640.0 360.0 0.01\n"
    )

    # images.txt: two images (each image = 2 lines, 2nd line is points2D)
    (colmap_dir / "images.txt").write_text(
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        "1 0.9239 0.3827 0.0 0.0 1.0 0.5 3.0 1 frame_001.jpg\n"
        "100.0 200.0 1 300.0 400.0 2\n"
        "2 1.0 0.0 0.0 0.0 -0.5 0.2 5.0 2 frame_002.jpg\n"
        "150.0 250.0 3\n"
    )

    # points3D.txt
    (colmap_dir / "points3D.txt").write_text(
        "# 3D point list\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "1 0.5 -0.3 2.1 200 150 100 0.5 1 0 2 1\n"
        "2 -1.0 0.8 1.5 50 200 50 0.3 1 1\n"
        "3 0.0 0.0 4.0 255 255 255 0.1 2 0\n"
    )


def test_colmap_parser(colmap_dir: Path) -> None:
    """Validate the COLMAP parser on the test files."""
    from pipeline.colmap_parser import (
        parse_cameras_txt,
        parse_images_txt,
        parse_points3d_txt,
    )

    cameras = parse_cameras_txt(colmap_dir / "cameras.txt")
    assert len(cameras) == 2, f"Expected 2 cameras, got {len(cameras)}"
    assert cameras[1].model == "PINHOLE"
    assert cameras[2].model == "SIMPLE_RADIAL"
    assert cameras[1].width == 1920
    assert cameras[1].focal_x == 1200.0
    assert cameras[1].focal_y == 1200.0
    assert cameras[2].focal_x == 900.0
    assert cameras[2].focal_y == 900.0  # SIMPLE_RADIAL: fx == fy
    print("[PASS] colmap_parser: cameras.txt")

    images = parse_images_txt(colmap_dir / "images.txt")
    assert len(images) == 2, f"Expected 2 images, got {len(images)}"
    assert images[0].image_id == 1
    assert images[0].name == "frame_001.jpg"
    assert abs(images[0].qw - 0.9239) < 1e-4
    assert images[1].image_id == 2
    print("[PASS] colmap_parser: images.txt")

    points = parse_points3d_txt(colmap_dir / "points3D.txt")
    assert len(points) == 3, f"Expected 3 points, got {len(points)}"
    assert points[0].point3d_id == 1
    assert points[0].r == 200
    assert len(points[0].image_ids) == 2
    print("[PASS] colmap_parser: points3D.txt")

    return cameras, images


def test_coordinate_transform() -> None:
    """Validate coordinate transforms with known test cases."""
    from pipeline.coordinate_transform import (
        colmap_camera_world_position,
        colmap_to_usd_position,
        colmap_to_usd_rotation,
        validate_round_trip,
    )
    from pipeline.colmap_parser import ColmapImage

    # Identity camera at origin looking down +Z in COLMAP
    identity_img = ColmapImage(
        image_id=0, qw=1.0, qx=0.0, qy=0.0, qz=0.0,
        tx=0.0, ty=0.0, tz=0.0, camera_id=1, name="test",
    )
    cx, cy, cz = colmap_camera_world_position(identity_img)
    assert abs(cx) < 1e-6 and abs(cy) < 1e-6 and abs(cz) < 1e-6
    print("[PASS] coordinate_transform: identity camera world pos = origin")

    # Position transform: Y and Z flip, scale by 0.5
    ux, uy, uz = colmap_to_usd_position(2.0, 4.0, 6.0)
    assert abs(ux - 1.0) < 1e-6   # 2.0 * 0.5
    assert abs(uy - (-2.0)) < 1e-6  # -4.0 * 0.5
    assert abs(uz - (-3.0)) < 1e-6  # -6.0 * 0.5
    print("[PASS] coordinate_transform: position RDF->Y-up + scale")

    # Round-trip
    assert validate_round_trip((1.0, 2.0, 3.0))
    assert validate_round_trip((-5.5, 0.0, 100.0))
    print("[PASS] coordinate_transform: round-trip validation")


def test_usd_scene_assembly(colmap_dir: Path) -> str:
    """Build a test scene with 2 objects and a camera, write to disk."""
    from pxr import Usd, UsdGeom

    from pipeline.coordinate_transform import CoordinateTransformer
    from pipeline.usd_assembler import ObjectDescriptor, UsdSceneAssembler

    output_path = "/tmp/test_scene.usda"

    # Load COLMAP data
    transformer = CoordinateTransformer.from_colmap_dir(colmap_dir)

    # Build scene
    assembler = UsdSceneAssembler(up_axis="Y", meters_per_unit=1.0)
    assembler.set_colmap_cameras(transformer)
    assembler.set_metadata("project", "test_pipeline")
    assembler.set_metadata("source", "synthetic")

    assembler.add_object(ObjectDescriptor(
        name="table",
        centroid=(0.0, 0.0, 2.0),
        diffuse_color=(0.6, 0.4, 0.2),
    ))

    assembler.add_object(ObjectDescriptor(
        name="lamp",
        centroid=(1.5, 1.0, 2.5),
        diffuse_color=(1.0, 0.95, 0.8),
        opacity=0.9,
        metadata={"category": "lighting"},
    ))

    stage = assembler.write(output_path)
    print(f"[PASS] usd_assembler: wrote {output_path}")

    # Validate re-read
    stage2 = Usd.Stage.Open(output_path)
    assert stage2 is not None, "Failed to re-open stage"

    # Check up axis
    up = UsdGeom.GetStageUpAxis(stage2)
    assert up == UsdGeom.Tokens.y, f"Expected Y-up, got {up}"
    print("[PASS] usd_assembler: stage re-read, Y-up confirmed")

    # Check meters per unit
    mpu = UsdGeom.GetStageMetersPerUnit(stage2)
    assert abs(mpu - 1.0) < 1e-6, f"Expected meters_per_unit=1.0, got {mpu}"
    print("[PASS] usd_assembler: meters_per_unit = 1.0")

    return output_path


def print_scene_hierarchy(usda_path: str) -> None:
    """Re-open the USDA file and print its full prim hierarchy."""
    from pxr import Usd

    stage = Usd.Stage.Open(usda_path)
    print("\n--- Scene Hierarchy ---")
    for prim in stage.Traverse():
        depth = len(prim.GetPath().GetString().split("/")) - 2
        indent = "  " * depth
        type_name = prim.GetTypeName() or "(untyped)"
        vsets = prim.GetVariantSets()
        vset_names = vsets.GetNames() if vsets else []
        vset_info = ""
        if vset_names:
            selections = [f"{n}={vsets.GetVariantSet(n).GetVariantSelection()}" for n in vset_names]
            vset_info = f"  [{', '.join(selections)}]"
        print(f"{indent}{prim.GetName()} ({type_name}){vset_info}")
    print("--- End Hierarchy ---\n")


def main() -> int:
    print("=" * 60)
    print("LichtFeld Studio - USD Pipeline Integration Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="lfs_colmap_test_") as tmpdir:
        colmap_dir = Path(tmpdir)
        _write_test_colmap_files(colmap_dir)

        # 1. COLMAP parser
        test_colmap_parser(colmap_dir)

        # 2. Coordinate transforms
        test_coordinate_transform()

        # 3. USD scene assembly
        try:
            from pxr import Usd  # noqa: F401
        except ImportError:
            print(
                "\n[SKIP] pxr (usd-core) not available on this Python. "
                "Install with: pip install usd-core  (requires Python 3.10-3.12)"
            )
            print("[PASS] All non-USD tests passed.")
            return 0

        usda_path = test_usd_scene_assembly(colmap_dir)

        # 4. Print hierarchy
        print_scene_hierarchy(usda_path)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
