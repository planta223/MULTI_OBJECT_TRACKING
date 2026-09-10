"""Per-object application-level pose post-processing.

This module consumes raw FoundationPose outputs but never owns or mutates a
FoundationPose estimator.  Create one :class:`ObjectPoseProcessor` per tracked
object so temporal post-processing state cannot leak between objects.
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
    """Raw estimator output and the distinct application output pose."""

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
    """Own independent output-pose state for one tracked object.

    Z-axis stabilization and discrete task symmetry remain separate.  When
    stabilization is enabled it always consumes the raw pose directly.  Task
    symmetry may still run as a diagnostic, but its result is not fed into the
    stabilizer.
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
        """Whether this object's Z-axis stabilizer currently has history."""

        return self._z_axis_stabilizer.initialized

    def reset(self) -> None:
        """Clear all temporal post-processing state for a new registration."""

        self._z_axis_stabilizer.reset()
        self._task_canonicalizer.reset()

    def process(self, raw_pose: np.ndarray, frame_id: int = 0) -> ProcessedPose:
        """Build an output pose from raw data without modifying the input."""

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
