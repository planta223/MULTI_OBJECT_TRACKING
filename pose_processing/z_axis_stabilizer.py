"""Temporal pose stabilization for objects with axial Z symmetry.

Only the direction of the raw object's Z axis is retained.  Rotation about
that axis is discarded, while a per-instance stable X/Y basis is propagated
between frames.  Translation and the remaining homogeneous-pose entries are
copied unchanged from the input pose.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


_VECTOR_NORM_EPSILON = 1.0e-12
_PROJECTION_NORM_EPSILON = 1.0e-8
_CAMERA_BASIS = np.eye(3, dtype=np.float64)


@dataclass(frozen=True)
class RotationDiagnostics:
    """Numerical quality metrics for a candidate 3x3 rotation matrix."""

    determinant: float
    orthogonality_error: float
    axis_norms: Tuple[float, float, float]
    maximum_axis_dot: float


def rotation_diagnostics(rotation: np.ndarray) -> RotationDiagnostics:
    """Return determinant and orthonormality metrics for a 3x3 matrix."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation contains non-finite values.")

    gram = matrix.T @ matrix
    off_diagonal = gram - np.diag(np.diag(gram))
    return RotationDiagnostics(
        determinant=float(np.linalg.det(matrix)),
        orthogonality_error=float(np.linalg.norm(gram - np.eye(3), ord="fro")),
        axis_norms=tuple(float(value) for value in np.linalg.norm(matrix, axis=0)),
        maximum_axis_dot=float(np.max(np.abs(off_diagonal))),
    )


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    """Normalize a finite 3-vector without overflowing on large values."""

    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")

    scale = float(np.max(np.abs(values)))
    if scale <= _VECTOR_NORM_EPSILON:
        raise ValueError(f"{name} norm is too small to define a direction.")
    scaled = values / scale
    return scaled / np.linalg.norm(scaled)


def _project_reference(reference: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """Project a reference direction onto the plane perpendicular to Z."""

    return reference - np.dot(reference, z_axis) * z_axis


def _most_stable_camera_axis(z_axis: np.ndarray) -> np.ndarray:
    """Select the camera basis axis least aligned with the current Z axis."""

    alignment = np.abs(_CAMERA_BASIS @ z_axis)
    return _CAMERA_BASIS[int(np.argmin(alignment))].copy()


class ZAxisPoseStabilizer:
    """Remove raw axial roll while preserving temporal Z-axis continuity.

    State belongs exclusively to this instance, so callers should create one
    stabilizer per independently tracked object.
    """

    def __init__(self) -> None:
        self._previous_x: Optional[np.ndarray] = None
        self._previous_z: Optional[np.ndarray] = None

    @property
    def initialized(self) -> bool:
        """Whether at least one pose has been stabilized since the last reset."""

        return self._previous_x is not None and self._previous_z is not None

    @property
    def previous_x(self) -> Optional[np.ndarray]:
        """Return a copy of the previous stable X axis, if initialized."""

        if self._previous_x is None:
            return None
        return self._previous_x.copy()

    @property
    def previous_z(self) -> Optional[np.ndarray]:
        """Return a copy of the previous stable Z axis, if initialized."""

        if self._previous_z is None:
            return None
        return self._previous_z.copy()

    def reset(self) -> None:
        """Forget the stable basis, for example before a new registration."""

        self._previous_x = None
        self._previous_z = None

    @staticmethod
    def _validated_pose(raw_pose: np.ndarray) -> np.ndarray:
        try:
            pose = np.asarray(raw_pose, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("raw_pose must contain numeric values.") from exc
        if pose.shape != (4, 4):
            raise ValueError(f"raw_pose must have shape (4, 4), got {pose.shape}.")
        if not np.all(np.isfinite(pose)):
            raise ValueError("raw_pose contains non-finite values.")
        return np.array(pose, copy=True, order="C")

    def stabilize(self, raw_pose: np.ndarray) -> np.ndarray:
        """Return a pose whose X/Y basis ignores raw rotation about object Z."""

        stable_pose = self._validated_pose(raw_pose)
        z_axis = _normalize(stable_pose[:3, 2], name="raw rotation Z axis")

        if self._previous_z is not None and np.dot(z_axis, self._previous_z) < 0.0:
            z_axis = -z_axis

        reference = (
            _CAMERA_BASIS[0].copy()
            if self._previous_x is None
            else self._previous_x.copy()
        )
        x_projected = _project_reference(reference, z_axis)

        if np.linalg.norm(x_projected) <= _PROJECTION_NORM_EPSILON:
            reference = _most_stable_camera_axis(z_axis)
            x_projected = _project_reference(reference, z_axis)

        x_axis = _normalize(x_projected, name="projected stable X axis")
        y_axis = _normalize(np.cross(z_axis, x_axis), name="stable Y axis")
        # Recompute X to remove the last floating-point component along Z and
        # guarantee a right-handed basis with columns (X, Y, Z).
        x_axis = _normalize(np.cross(y_axis, z_axis), name="stable X axis")

        stable_pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        self._previous_x = x_axis.copy()
        self._previous_z = z_axis.copy()
        return stable_pose
