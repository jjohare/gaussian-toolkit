# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Coordinate system alignment between COLMAP and USD.

COLMAP uses a Right-Down-Forward (RDF) convention:
  +X = right, +Y = down, +Z = forward (into scene)

USD uses Y-up:
  +X = right, +Y = up, +Z = toward viewer (out of screen)

The conversion applies a 180-degree rotation around the X axis:
  y_usd = -y_colmap
  z_usd = -z_colmap

LichtFeld Studio additionally applies SCENE_SCALE = 0.5 to positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .colmap_parser import (
    ColmapCamera,
    ColmapImage,
    parse_cameras_txt,
    parse_images_txt,
)

# LichtFeld's canonical scene scale from src/io/formats/usd.cpp
SCENE_SCALE: float = 0.5


# ---------------------------------------------------------------------------
#  Quaternion utilities (w, x, y, z convention throughout)
# ---------------------------------------------------------------------------

def _quat_multiply(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Hamilton product of two quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_conjugate(
    q: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    w, x, y, z = q
    return (w, -x, -y, -z)


def _quat_to_rotation_matrix(
    q: Tuple[float, float, float, float],
) -> List[List[float]]:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix (row-major)."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _rotation_matrix_to_quat(
    m: List[List[float]],
) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability.
    """
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2][1] - m[1][2]) * s
        y = (m[0][2] - m[2][0]) * s
        z = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


# ---------------------------------------------------------------------------
#  The RDF -> Y-up rotation: 180 degrees around X
#  As a quaternion: (cos(90), sin(90), 0, 0) = (0, 1, 0, 0)
#  But we want the matrix form:  diag(1, -1, -1)
# ---------------------------------------------------------------------------
_RDF_TO_YUP_QUAT: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)


def colmap_to_usd_position(
    tx: float,
    ty: float,
    tz: float,
    *,
    apply_scene_scale: bool = True,
) -> Tuple[float, float, float]:
    """Convert a COLMAP world-space position to USD Y-up coordinates.

    COLMAP stores camera-to-world as R, t where the world position of
    the camera is  C = -R^T @ t.  This function expects the *world*
    position (already extracted), not the raw t vector.

    The RDF->Y-up transform flips Y and Z.
    """
    scale = SCENE_SCALE if apply_scene_scale else 1.0
    return (tx * scale, -ty * scale, -tz * scale)


def colmap_to_usd_rotation(
    qw: float, qx: float, qy: float, qz: float,
) -> Tuple[float, float, float, float]:
    """Convert COLMAP extrinsic quaternion to USD Y-up quaternion.

    COLMAP quaternion encodes camera-to-world rotation in RDF.
    We pre-multiply by the RDF->Y-up rotation to get the USD rotation.

    Returns (w, x, y, z).
    """
    colmap_q = (qw, qx, qy, qz)
    return _quat_multiply(_RDF_TO_YUP_QUAT, colmap_q)


def colmap_camera_world_position(image: ColmapImage) -> Tuple[float, float, float]:
    """Compute the world-space camera centre from COLMAP extrinsics.

    C = -R^T @ t
    """
    R = _quat_to_rotation_matrix(image.quaternion)
    tx, ty, tz = image.translation
    # R^T @ t  (R is row-major, R^T columns = R rows)
    cx = -(R[0][0] * tx + R[1][0] * ty + R[2][0] * tz)
    cy = -(R[0][1] * tx + R[1][1] * ty + R[2][1] * tz)
    cz = -(R[0][2] * tx + R[1][2] * ty + R[2][2] * tz)
    return (cx, cy, cz)


# ---------------------------------------------------------------------------
#  4x4 Transform matrix builder
# ---------------------------------------------------------------------------

def build_usd_transform_from_colmap(
    image: ColmapImage,
    *,
    apply_scene_scale: bool = True,
) -> List[List[float]]:
    """Build a 4x4 row-major transform matrix for a USD camera prim.

    Converts from COLMAP RDF to USD Y-up, applies scene scale.
    Returns the matrix as list-of-lists suitable for Gf.Matrix4d.
    """
    # World position
    cx, cy, cz = colmap_camera_world_position(image)
    ux, uy, uz = colmap_to_usd_position(cx, cy, cz, apply_scene_scale=apply_scene_scale)

    # Rotation
    usd_q = colmap_to_usd_rotation(*image.quaternion)
    R = _quat_to_rotation_matrix(usd_q)

    return [
        [R[0][0], R[0][1], R[0][2], 0.0],
        [R[1][0], R[1][1], R[1][2], 0.0],
        [R[2][0], R[2][1], R[2][2], 0.0],
        [ux, uy, uz, 1.0],
    ]


# ---------------------------------------------------------------------------
#  High-level transformer class
# ---------------------------------------------------------------------------

@dataclass
class CoordinateTransformer:
    """Loads COLMAP data and provides transforms for all pipeline components.

    Attributes:
        cameras: Dict mapping camera_id to ColmapCamera (intrinsics).
        images: List of ColmapImage (extrinsics).
        scene_scale: The uniform scale factor applied to positions.
    """

    cameras: Dict[int, ColmapCamera]
    images: List[ColmapImage]
    scene_scale: float = SCENE_SCALE

    @classmethod
    def from_colmap_dir(
        cls,
        sparse_dir: Path | str,
        *,
        scene_scale: Optional[float] = None,
    ) -> "CoordinateTransformer":
        """Load from a COLMAP sparse reconstruction directory.

        Expects cameras.txt and images.txt inside *sparse_dir*.
        """
        sparse_dir = Path(sparse_dir)
        cameras = parse_cameras_txt(sparse_dir / "cameras.txt")
        images = parse_images_txt(sparse_dir / "images.txt")
        return cls(
            cameras=cameras,
            images=images,
            scene_scale=scene_scale if scene_scale is not None else SCENE_SCALE,
        )

    def camera_usd_position(
        self, image: ColmapImage,
    ) -> Tuple[float, float, float]:
        """USD world-space position for a camera."""
        cx, cy, cz = colmap_camera_world_position(image)
        return colmap_to_usd_position(
            cx, cy, cz, apply_scene_scale=(self.scene_scale != 1.0),
        )

    def camera_usd_transform(
        self, image: ColmapImage,
    ) -> List[List[float]]:
        """Full 4x4 USD transform for a camera."""
        return build_usd_transform_from_colmap(
            image, apply_scene_scale=(self.scene_scale != 1.0),
        )

    def object_usd_position(
        self, x: float, y: float, z: float,
    ) -> Tuple[float, float, float]:
        """Convert an arbitrary COLMAP world position to USD."""
        return colmap_to_usd_position(
            x, y, z, apply_scene_scale=(self.scene_scale != 1.0),
        )


# ---------------------------------------------------------------------------
#  Validation helpers
# ---------------------------------------------------------------------------

def validate_round_trip(
    pos: Tuple[float, float, float],
    *,
    tol: float = 1e-6,
) -> bool:
    """Verify that COLMAP->USD->back preserves the original position.

    The inverse of (x, -y, -z) * scale is (x/s, -y/s, -z/s).
    """
    ux, uy, uz = colmap_to_usd_position(*pos)
    inv_scale = 1.0 / SCENE_SCALE
    rx, ry, rz = ux * inv_scale, -uy * inv_scale, -uz * inv_scale
    return (
        abs(rx - pos[0]) < tol
        and abs(ry - pos[1]) < tol
        and abs(rz - pos[2]) < tol
    )
