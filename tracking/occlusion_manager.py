"""Projected-overlap and RGB-D front/rear visibility analysis."""

from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from camera.base import FrameData
from config import TrackingConfig


@dataclass(frozen=True)
class OcclusionGeometry:
    """Estimator-neutral bounds needed to project one tracked object."""

    bbox: np.ndarray
    to_origin: np.ndarray


@dataclass(frozen=True)
class ProjectedBounds:
    """Clipped image rectangle and predicted camera-space depth range."""

    rectangle: Tuple[int, int, int, int]
    center_depth_m: float
    minimum_depth_m: float
    maximum_depth_m: float

    @property
    def area(self) -> int:
        left, top, right, bottom = self.rectangle
        return max(0, right - left) * max(0, bottom - top)


@dataclass(frozen=True)
class OcclusionDecision:
    """Visibility decision for one predicted object."""

    object_id: str
    occluded: bool = False
    occluding_object_id: Optional[str] = None
    occludes_object_id: Optional[str] = None
    overlap_ratio: float = 0.0
    observed_depth_m: Optional[float] = None
    predicted_depth_m: Optional[float] = None
    reason: str = "no_confirmed_occlusion"


def project_object_bounds(
    pose: np.ndarray,
    geometry: OcclusionGeometry,
    K: np.ndarray,
    image_shape: Tuple[int, int],
) -> Optional[ProjectedBounds]:
    """Project the eight oriented-bounds corners into a clipped rectangle."""

    pose_array = np.asarray(pose, dtype=np.float64)
    matrix = np.asarray(K, dtype=np.float64)
    bbox = np.asarray(geometry.bbox, dtype=np.float64)
    to_origin = np.asarray(geometry.to_origin, dtype=np.float64)
    if pose_array.shape != (4, 4) or matrix.shape != (3, 3):
        raise ValueError("pose and K must have shapes (4, 4) and (3, 3).")
    if bbox.shape != (2, 3) or to_origin.shape != (4, 4):
        raise ValueError("bbox and to_origin must have shapes (2, 3) and (4, 4).")
    if not all(
        np.all(np.isfinite(values))
        for values in (pose_array, matrix, bbox, to_origin)
    ):
        raise ValueError("Projection inputs must contain only finite values.")

    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive.")

    low = np.min(bbox, axis=0)
    high = np.max(bbox, axis=0)
    corners = np.asarray(list(product(*zip(low, high))), dtype=np.float64)
    centered_to_camera = pose_array @ np.linalg.inv(to_origin)
    camera_corners = (
        centered_to_camera[:3, :3] @ corners.T
    ).T + centered_to_camera[:3, 3]
    valid = camera_corners[:, 2] > 1.0e-6
    if not np.any(valid):
        return None

    projected = (matrix @ camera_corners[valid].T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    left = max(0, int(np.floor(np.min(pixels[:, 0]))))
    top = max(0, int(np.floor(np.min(pixels[:, 1]))))
    right = min(width, int(np.ceil(np.max(pixels[:, 0]))) + 1)
    bottom = min(height, int(np.ceil(np.max(pixels[:, 1]))) + 1)
    if right <= left or bottom <= top:
        return None

    valid_depths = camera_corners[valid, 2]
    return ProjectedBounds(
        rectangle=(left, top, right, bottom),
        center_depth_m=float(centered_to_camera[2, 3]),
        minimum_depth_m=float(np.min(valid_depths)),
        maximum_depth_m=float(np.max(valid_depths)),
    )


def _intersection(
    first: ProjectedBounds,
    second: ProjectedBounds,
) -> Optional[Tuple[int, int, int, int]]:
    left = max(first.rectangle[0], second.rectangle[0])
    top = max(first.rectangle[1], second.rectangle[1])
    right = min(first.rectangle[2], second.rectangle[2])
    bottom = min(first.rectangle[3], second.rectangle[3])
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def analyze_depth_occlusion(
    frame: FrameData,
    predicted_poses: Mapping[str, np.ndarray],
    geometries: Mapping[str, OcclusionGeometry],
    config: TrackingConfig,
) -> Dict[str, OcclusionDecision]:
    """Conservatively identify rear objects hidden by known front objects."""

    missing_geometry = set(predicted_poses) - set(geometries)
    if missing_geometry:
        raise KeyError(f"Missing occlusion geometry: {sorted(missing_geometry)}")

    projections = {
        object_id: project_object_bounds(
            pose,
            geometries[object_id],
            frame.K,
            (frame.height, frame.width),
        )
        for object_id, pose in predicted_poses.items()
    }
    decisions = {
        object_id: OcclusionDecision(
            object_id=object_id,
            predicted_depth_m=(
                None if projection is None else projection.center_depth_m
            ),
        )
        for object_id, projection in projections.items()
    }

    for first_id, second_id in combinations(predicted_poses, 2):
        first = projections[first_id]
        second = projections[second_id]
        if first is None or second is None:
            continue
        intersection = _intersection(first, second)
        if intersection is None:
            continue
        left, top, right, bottom = intersection
        intersection_area = (right - left) * (bottom - top)
        overlap_ratio = intersection_area / float(min(first.area, second.area))
        if overlap_ratio < config.occlusion_overlap_ratio:
            continue

        if first.center_depth_m <= second.center_depth_m:
            front_id, front = first_id, first
            rear_id, rear = second_id, second
        else:
            front_id, front = second_id, second
            rear_id, rear = first_id, first
        depth_separation = rear.center_depth_m - front.center_depth_m
        if depth_separation < config.occlusion_min_depth_separation_m:
            continue

        overlap_depth = frame.depth_m[top:bottom, left:right]
        valid_depth = overlap_depth[
            np.isfinite(overlap_depth) & (overlap_depth >= 0.001)
        ]
        if valid_depth.size < config.occlusion_min_valid_depth_pixels:
            continue
        observed_depth = float(np.median(valid_depth))
        front_error = abs(observed_depth - front.center_depth_m)
        rear_error = abs(observed_depth - rear.center_depth_m)
        front_depth_range_match = (
            front.minimum_depth_m - config.occlusion_depth_margin_m
            <= observed_depth
            <= front.maximum_depth_m + config.occlusion_depth_margin_m
        )
        if not front_depth_range_match:
            continue
        if front_error + config.occlusion_depth_margin_m >= rear_error:
            continue

        previous_rear = decisions[rear_id]
        if not previous_rear.occluded or overlap_ratio > previous_rear.overlap_ratio:
            decisions[rear_id] = OcclusionDecision(
                object_id=rear_id,
                occluded=True,
                occluding_object_id=front_id,
                overlap_ratio=overlap_ratio,
                observed_depth_m=observed_depth,
                predicted_depth_m=rear.center_depth_m,
                reason="overlap_depth_matches_front",
            )
        decisions[front_id] = OcclusionDecision(
            object_id=front_id,
            occluded=False,
            occludes_object_id=rear_id,
            overlap_ratio=max(overlap_ratio, decisions[front_id].overlap_ratio),
            observed_depth_m=observed_depth,
            predicted_depth_m=front.center_depth_m,
            reason="visible_front_object",
        )

    return decisions
