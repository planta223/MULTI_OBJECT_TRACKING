"""Minimal OpenCV pose overlay for live smoke testing."""

from itertools import product
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


_BOX_EDGES = tuple(
    (index, index ^ (1 << axis))
    for index in range(8)
    for axis in range(3)
    if index < (index ^ (1 << axis))
)


def _project_points(
    points: np.ndarray,
    object_to_camera: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    camera_points = (
        object_to_camera[:3, :3] @ points.T
    ).T + object_to_camera[:3, 3]
    valid = camera_points[:, 2] > 1e-6
    pixels = np.zeros((len(points), 2), dtype=np.int32)
    if np.any(valid):
        projected = (K @ camera_points[valid].T).T
        uv = projected[:, :2] / projected[:, 2:3]
        uv = np.clip(np.rint(uv), -1_000_000, 1_000_000)
        pixels[valid] = uv.astype(np.int32)
    return pixels, valid


def draw_pose_overlay(
    rgb: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
    bbox: np.ndarray,
    to_origin: np.ndarray,
    fps: Optional[float] = None,
    label: str = "Pipe1",
    axis_length_m: float = 0.05,
) -> np.ndarray:
    """Draw the oriented CAD bbox and XYZ axes over an RGB image."""

    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    centered_to_camera = pose @ np.linalg.inv(to_origin)

    low = np.min(bbox, axis=0)
    high = np.max(bbox, axis=0)
    corners = np.asarray(list(product(*zip(low, high))), dtype=np.float64)
    pixels, valid = _project_points(corners, centered_to_camera, K)
    for start, end in _BOX_EDGES:
        if valid[start] and valid[end]:
            cv2.line(
                image,
                tuple(pixels[start]),
                tuple(pixels[end]),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    axes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length_m, 0.0, 0.0],
            [0.0, axis_length_m, 0.0],
            [0.0, 0.0, axis_length_m],
        ],
        dtype=np.float64,
    )
    axis_pixels, axis_valid = _project_points(axes, centered_to_camera, K)
    axis_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    if axis_valid[0]:
        origin = tuple(axis_pixels[0])
        for index, color in enumerate(axis_colors, start=1):
            if axis_valid[index]:
                cv2.arrowedLine(
                    image,
                    origin,
                    tuple(axis_pixels[index]),
                    color,
                    3,
                    cv2.LINE_AA,
                    tipLength=0.08,
                )

    translation = pose[:3, 3]
    text = (
        f"{label} XYZ=({translation[0]:+.3f}, "
        f"{translation[1]:+.3f}, {translation[2]:+.3f}) m"
    )
    if fps is not None:
        text += f"  track={fps:.1f} FPS"
    cv2.putText(
        image,
        text,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "X:red  Y:green  Z:blue  Q/Esc: quit",
        (15, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image
