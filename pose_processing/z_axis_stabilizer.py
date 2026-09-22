"""축 방향 Z symmetry를 가진 객체의 시간 기반 pose 안정화.

원본 객체 Z축의 방향만 유지한다. 해당 축을 중심으로 한 회전은 버리고,
인스턴스별로 안정적인 X/Y basis를 frame 사이에 전파한다. translation과 나머지
homogeneous pose 항목은 입력 pose에서 변경 없이 복사한다.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


_VECTOR_NORM_EPSILON = 1.0e-12
_PROJECTION_NORM_EPSILON = 1.0e-8
_CAMERA_BASIS = np.eye(3, dtype=np.float64)


@dataclass(frozen=True)
class RotationDiagnostics:
    """후보 3x3 회전 행렬의 수치 품질 지표."""

    determinant: float
    orthogonality_error: float
    axis_norms: Tuple[float, float, float]
    maximum_axis_dot: float


def rotation_diagnostics(rotation: np.ndarray) -> RotationDiagnostics:
    """3x3 행렬의 determinant와 orthonormality 지표를 반환한다."""

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
    """큰 값에서 overflow 없이 유한한 3-vector를 정규화한다."""

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
    """기준 방향을 Z에 수직인 평면으로 투영한다."""

    return reference - np.dot(reference, z_axis) * z_axis


def _most_stable_camera_axis(z_axis: np.ndarray) -> np.ndarray:
    """현재 Z축과 가장 덜 나란한 카메라 basis 축을 선택한다."""

    alignment = np.abs(_CAMERA_BASIS @ z_axis)
    return _CAMERA_BASIS[int(np.argmin(alignment))].copy()


class ZAxisPoseStabilizer:
    """시간에 따른 Z축 연속성을 유지하면서 원본 축 방향 roll을 제거한다.

    상태는 이 인스턴스에만 속하므로 독립적으로 추적하는 객체마다 stabilizer를
    하나씩 생성해야 한다.
    """

    def __init__(self) -> None:
        self._previous_x: Optional[np.ndarray] = None
        self._previous_z: Optional[np.ndarray] = None

    @property
    def initialized(self) -> bool:
        """마지막 reset 이후 pose를 하나 이상 안정화했는지 나타낸다."""

        return self._previous_x is not None and self._previous_z is not None

    @property
    def previous_x(self) -> Optional[np.ndarray]:
        """초기화되었다면 이전의 안정적인 X축 복사본을 반환한다."""

        if self._previous_x is None:
            return None
        return self._previous_x.copy()

    @property
    def previous_z(self) -> Optional[np.ndarray]:
        """초기화되었다면 이전의 안정적인 Z축 복사본을 반환한다."""

        if self._previous_z is None:
            return None
        return self._previous_z.copy()

    def reset(self) -> None:
        """새 registration 전과 같이 필요할 때 안정적인 basis 이력을 지운다."""

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
        """객체 Z축 중심의 원본 회전을 무시한 X/Y basis pose를 반환한다."""

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
        # Z 방향에 남은 마지막 부동소수점 성분을 제거하고 열이 (X, Y, Z)인
        # 오른손 basis를 보장하도록 X를 다시 계산한다.
        x_axis = _normalize(np.cross(y_axis, z_axis), name="stable X axis")

        stable_pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        self._previous_x = x_axis.copy()
        self._previous_z = z_axis.copy()
        return stable_pose
