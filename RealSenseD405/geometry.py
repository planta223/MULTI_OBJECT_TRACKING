"""Basic camera geometry independent of FoundationPose."""

from typing import Any

import numpy as np


def make_camera_matrix(intrinsics: Any) -> np.ndarray:
    """Build a 3x3 K matrix from a RealSense intrinsics object."""

    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def deproject_pixel(u: int, v: int, depth_m: float, K: np.ndarray) -> np.ndarray:
    """Deproject one pixel with the pinhole model into camera XYZ meters."""

    matrix = np.asarray(K, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("K must have shape (3, 3).")
    if not np.isfinite(depth_m) or depth_m < 0:
        raise ValueError("depth_m must be finite and non-negative.")
    x = (float(u) - matrix[0, 2]) * depth_m / matrix[0, 0]
    y = (float(v) - matrix[1, 2]) * depth_m / matrix[1, 1]
    return np.array([x, y, depth_m], dtype=np.float64)
