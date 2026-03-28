# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Project 2D instance masks onto 3D Gaussians via multi-view majority voting.

Given per-frame instance masks (from SAM2) and camera parameters (from COLMAP),
each 3D Gaussian is projected into every view.  The pixel it lands on determines
which object label that view "votes" for.  The final label for each Gaussian is
the majority vote across all views.

Typical usage::

    projector = MaskProjector(cameras, images, image_hw=(1080, 1920))
    labels = projector.assign_labels(gaussian_xyz, frame_masks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.pipeline.colmap_parser import ColmapCamera, ColmapImage

logger = logging.getLogger(__name__)


def _quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a unit quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),      1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    return R


@dataclass(frozen=True, slots=True)
class ViewProjection:
    """Precomputed projection data for a single camera view.

    Attributes:
        image_name: Filename of the source image (for mask lookup).
        R: 3x3 world-to-camera rotation.
        t: 3-element translation vector.
        K: 3x3 intrinsic matrix.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    image_name: str
    R: np.ndarray      # (3, 3)
    t: np.ndarray      # (3,)
    K: np.ndarray      # (3, 3)
    width: int
    height: int

    def project(self, xyz_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project world-space points to pixel coordinates.

        Args:
            xyz_world: ``(N, 3)`` world coordinates.

        Returns:
            pixels: ``(N, 2)`` integer pixel coordinates (col, row).
            visible: ``(N,)`` boolean mask for points within the image bounds
                     and in front of the camera.
        """
        # Transform to camera space: p_cam = R @ p_world + t
        p_cam = (self.R @ xyz_world.T).T + self.t  # (N, 3)

        depth = p_cam[:, 2]
        in_front = depth > 1e-6

        # Project to normalised image coords, then to pixels
        p_img = (self.K @ p_cam.T).T  # (N, 3)
        u = p_img[:, 0] / np.where(in_front, p_img[:, 2], 1.0)
        v = p_img[:, 1] / np.where(in_front, p_img[:, 2], 1.0)

        col = np.round(u).astype(np.int64)
        row = np.round(v).astype(np.int64)

        in_bounds = (
            in_front
            & (col >= 0) & (col < self.width)
            & (row >= 0) & (row < self.height)
        )

        pixels = np.stack([col, row], axis=1)
        return pixels, in_bounds


class MaskProjector:
    """Projects 2D segmentation masks onto 3D Gaussians.

    Parameters:
        cameras: Mapping from camera_id to ``ColmapCamera``.
        images: Mapping from image_id to ``ColmapImage``.
        image_hw: Default ``(H, W)`` override when camera params lack resolution.
    """

    def __init__(
        self,
        cameras: Dict[int, ColmapCamera],
        images: Dict[int, ColmapImage],
        image_hw: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.views: List[ViewProjection] = []
        self._image_name_to_view_idx: Dict[str, int] = {}

        for img in images.values():
            cam = cameras[img.camera_id]
            R = _quaternion_to_rotation(img.qw, img.qx, img.qy, img.qz)
            t = np.array([img.tx, img.ty, img.tz], dtype=np.float64)

            h = image_hw[0] if image_hw else cam.height
            w = image_hw[1] if image_hw else cam.width

            K = np.array([
                [cam.focal_x, 0.0, cam.center_x],
                [0.0, cam.focal_y, cam.center_y],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

            view = ViewProjection(
                image_name=img.name, R=R, t=t, K=K, width=w, height=h,
            )
            self._image_name_to_view_idx[img.name] = len(self.views)
            self.views.append(view)

        logger.info("MaskProjector initialised with %d views", len(self.views))

    @classmethod
    def from_colmap_dir(
        cls,
        colmap_dir: str,
        image_hw: Optional[Tuple[int, int]] = None,
    ) -> "MaskProjector":
        """Create from a COLMAP sparse reconstruction directory.

        Expects ``cameras.txt``, ``images.txt`` (and optionally ``points3D.txt``)
        in the given directory.
        """
        from pathlib import Path
        from src.pipeline.colmap_parser import parse_cameras_txt, parse_images_txt

        d = Path(colmap_dir)
        cameras = {c.camera_id: c for c in parse_cameras_txt(d / "cameras.txt")}
        images = {i.image_id: i for i in parse_images_txt(d / "images.txt")}
        return cls(cameras=cameras, images=images, image_hw=image_hw)

    def assign_labels(
        self,
        gaussian_xyz: np.ndarray,
        frame_masks: Dict[str, np.ndarray],
        *,
        background_label: int = 0,
        min_votes: int = 1,
    ) -> np.ndarray:
        """Assign an integer object label to each 3D Gaussian.

        Args:
            gaussian_xyz: ``(G, 3)`` world-space Gaussian centres.
            frame_masks: Dict mapping image filename to a 2D integer label
                         map of shape ``(H, W)``, where 0 = background and
                         positive integers are object IDs.
            background_label: Label assigned to Gaussians with no valid votes.
            min_votes: Minimum number of non-background votes required.

        Returns:
            ``(G,)`` int32 array of per-Gaussian labels.
        """
        n_gaussians = gaussian_xyz.shape[0]
        # Collect votes: for each Gaussian, count how many times each label is seen
        vote_counts: Dict[int, Dict[int, int]] = {}  # gaussian_idx -> {label: count}

        for view in self.views:
            if view.image_name not in frame_masks:
                continue

            mask = frame_masks[view.image_name]
            pixels, visible = view.project(gaussian_xyz)

            vis_idx = np.where(visible)[0]
            if len(vis_idx) == 0:
                continue

            cols = pixels[vis_idx, 0]
            rows = pixels[vis_idx, 1]
            labels_at_pixels = mask[rows, cols]

            for gi, label in zip(vis_idx, labels_at_pixels):
                gi = int(gi)
                label = int(label)
                if label == background_label:
                    continue
                if gi not in vote_counts:
                    vote_counts[gi] = {}
                vote_counts[gi][label] = vote_counts[gi].get(label, 0) + 1

        # Majority vote
        result = np.full(n_gaussians, background_label, dtype=np.int32)
        for gi, counts in vote_counts.items():
            total = sum(counts.values())
            if total < min_votes:
                continue
            best_label = max(counts, key=counts.get)
            result[gi] = best_label

        labeled = int(np.sum(result != background_label))
        logger.info("Labeled %d / %d Gaussians (%d background)",
                     labeled, n_gaussians, n_gaussians - labeled)
        return result

    def assign_labels_batched(
        self,
        gaussian_xyz: np.ndarray,
        frame_masks: Dict[str, np.ndarray],
        *,
        background_label: int = 0,
        min_votes: int = 1,
        batch_size: int = 100_000,
    ) -> np.ndarray:
        """Memory-efficient version of ``assign_labels`` for large point clouds.

        Processes Gaussians in batches to limit peak memory usage.
        """
        n = gaussian_xyz.shape[0]
        result = np.full(n, background_label, dtype=np.int32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_xyz = gaussian_xyz[start:end]
            batch_labels = self.assign_labels(
                batch_xyz, frame_masks,
                background_label=background_label,
                min_votes=min_votes,
            )
            result[start:end] = batch_labels
            logger.debug("Batch %d-%d complete", start, end)

        return result

    @staticmethod
    def segmentation_results_to_label_maps(
        results: Sequence,
        image_names: Sequence[str],
    ) -> Dict[str, np.ndarray]:
        """Convert a list of ``SegmentationResult`` into integer label maps.

        Resolves overlapping masks by preferring the mask with the lowest
        (non-zero) object ID, which typically corresponds to the largest object.

        Args:
            results: List of ``SegmentationResult`` from ``SAM2Segmentor``.
            image_names: Corresponding image filenames, same length as results.

        Returns:
            Dict mapping filename to ``(H, W)`` int32 label maps.
        """
        label_maps: Dict[str, np.ndarray] = {}

        for seg_result, name in zip(results, image_names):
            if seg_result.masks.shape[0] == 0:
                continue
            h, w = seg_result.masks.shape[1], seg_result.masks.shape[2]
            label_map = np.zeros((h, w), dtype=np.int32)

            # Apply masks in reverse order so that lower IDs overwrite higher
            sorted_indices = np.argsort(seg_result.object_ids)[::-1]
            for idx in sorted_indices:
                oid = seg_result.object_ids[idx]
                label_map[seg_result.masks[idx]] = oid

            label_maps[name] = label_map

        return label_maps
