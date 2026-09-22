"""객체별 애플리케이션 수준 pose 후처리.

이 모듈은 FoundationPose 원본 출력을 사용하지만 FoundationPose estimator를
소유하거나 변경하지 않는다. 시간 기반 후처리 상태가 객체 사이에 섞이지 않도록
추적 객체마다 :class:`ObjectPoseProcessor` 인스턴스를 하나씩 생성한다.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .task_symmetry import PipeTaskPoseCanonicalizer, TaskPoseResult
from .z_axis_stabilizer import ZAxisPoseStabilizer


def _immutable_pose(pose: np.ndarray, *, name: str) -> np.ndarray:
    try:
        values = np.asarray(pose, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if values.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")

    copied = np.array(values, copy=True, order="C")
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class ProcessedPose:
    """estimator 원본 출력과 별도로 생성한 애플리케이션 출력 pose."""

    raw_pose: np.ndarray
    output_pose: np.ndarray
    z_axis_stabilized: bool
    task_symmetry_index: Optional[int] = None
    task_symmetry_used_for_output: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_pose",
            _immutable_pose(self.raw_pose, name="raw_pose"),
        )
        object.__setattr__(
            self,
            "output_pose",
            _immutable_pose(self.output_pose, name="output_pose"),
        )
        if self.z_axis_stabilized and self.task_symmetry_used_for_output:
            raise ValueError(
                "Z-axis stabilization and task symmetry cannot both define "
                "the output pose."
            )


class ObjectPoseProcessor:
    """추적 객체 하나의 독립적인 출력 pose 상태를 소유한다.

    Z축 안정화와 이산 task symmetry는 서로 분리한다. 안정화를 활성화하면 항상
    원본 pose를 직접 사용한다. task symmetry를 진단 목적으로 실행할 수 있지만
    그 결과를 stabilizer에 전달하지 않는다.
    """

    def __init__(
        self,
        *,
        enable_z_axis_stabilization: bool = False,
        enable_task_symmetry_output: bool = False,
        enable_task_symmetry_diagnostic: bool = False,
    ) -> None:
        if enable_z_axis_stabilization and enable_task_symmetry_output:
            raise ValueError(
                "Choose either Z-axis stabilization or task symmetry for the "
                "output pose, not both."
            )

        self.enable_z_axis_stabilization = bool(enable_z_axis_stabilization)
        self.enable_task_symmetry_output = bool(enable_task_symmetry_output)
        self.enable_task_symmetry_diagnostic = bool(
            enable_task_symmetry_diagnostic
        )
        self._z_axis_stabilizer = ZAxisPoseStabilizer()
        self._task_canonicalizer = PipeTaskPoseCanonicalizer(
            debug=self.enable_task_symmetry_diagnostic
        )

    @property
    def z_axis_initialized(self) -> bool:
        """이 객체의 Z축 stabilizer가 현재 이력을 갖고 있는지 나타낸다."""

        return self._z_axis_stabilizer.initialized

    def reset(self) -> None:
        """새 registration을 위해 모든 시간 기반 후처리 상태를 지운다."""

        self._z_axis_stabilizer.reset()
        self._task_canonicalizer.reset()

    def process(self, raw_pose: np.ndarray, frame_id: int = 0) -> ProcessedPose:
        """입력을 변경하지 않고 원본 데이터에서 출력 pose를 만든다."""

        raw = _immutable_pose(raw_pose, name="raw_pose")
        task_result: Optional[TaskPoseResult] = None
        if (
            self.enable_task_symmetry_output
            or self.enable_task_symmetry_diagnostic
        ):
            task_result = self._task_canonicalizer.canonicalize(raw, frame_id)

        if self.enable_z_axis_stabilization:
            output = self._z_axis_stabilizer.stabilize(raw)
        elif self.enable_task_symmetry_output:
            if task_result is None:
                raise RuntimeError("Task-symmetry output was not computed.")
            output = task_result.canonical_pose
        else:
            output = raw.copy()

        return ProcessedPose(
            raw_pose=raw,
            output_pose=output,
            z_axis_stabilized=self.enable_z_axis_stabilization,
            task_symmetry_index=(
                None if task_result is None else task_result.symmetry_index
            ),
            task_symmetry_used_for_output=self.enable_task_symmetry_output,
        )
