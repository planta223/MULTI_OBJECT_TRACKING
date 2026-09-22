"""애플리케이션 pose 처리의 CPU 전용 단일/다중 객체 검사."""

import math
from pathlib import Path
import sys
from typing import Dict, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import TrackingMode, TrackingState
from pose_estimation.pose_result import PoseResult
from pose_processing.object_pose_processor import ObjectPoseProcessor
from pose_processing.z_axis_stabilizer import rotation_diagnostics


def _normalize(vector: Sequence[float]) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    return values / np.linalg.norm(values)


def _raw_pose(
    z_axis: Sequence[float],
    axial_phase_deg: float,
    translation: Sequence[float],
) -> np.ndarray:
    z_axis = _normalize(z_axis)
    camera_basis = np.eye(3, dtype=np.float64)
    reference = camera_basis[int(np.argmin(np.abs(camera_basis @ z_axis)))]
    x_axis = _normalize(reference - np.dot(reference, z_axis) * z_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    base = np.column_stack((x_axis, y_axis, z_axis))

    phase = math.radians(axial_phase_deg)
    cosine = math.cos(phase)
    sine = math.sin(phase)
    axial_rotation = np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = base @ axial_rotation
    pose[:3, 3] = translation
    return pose


def _pose_result(object_id: str, frame_id: int, pose: np.ndarray) -> PoseResult:
    return PoseResult(
        cycle_id=frame_id,
        object_id=object_id,
        source_frame_id=frame_id,
        host_wall_time_s=1000.0 + frame_id,
        host_monotonic_time_s=2000.0 + frame_id,
        mode=TrackingMode.TRACK,
        state=TrackingState.TRACKING,
        valid=True,
        pose=pose,
        processing_time_s=0.01,
    )


def _assert_output_quality(output_pose: np.ndarray) -> None:
    diagnostics = rotation_diagnostics(output_pose[:3, :3])
    assert diagnostics.orthogonality_error < 1.0e-12
    assert abs(diagnostics.determinant - 1.0) < 1.0e-12


def _processor_map(object_ids: Sequence[str]) -> Dict[str, ObjectPoseProcessor]:
    return {
        object_id: ObjectPoseProcessor(enable_z_axis_stabilization=True)
        for object_id in object_ids
    }


def test_single_object_count_and_raw_contract() -> None:
    print("\nSingle-object processor")
    processors = _processor_map(("pipe1",))
    assert len(processors) == 1
    processor = processors["pipe1"]
    previous_output_rotation = None

    for frame_id, phase in enumerate((0.0, 80.0, 170.0, 300.0)):
        raw = _raw_pose(
            (0.0, 0.0, 1.0),
            phase,
            (frame_id + 0.1, -frame_id - 0.2, 1.5),
        )
        result = _pose_result("pipe1", frame_id, raw)
        result_pose_before = result.pose.copy()
        processed = processor.process(result.pose, result.source_frame_id)

        assert processed.z_axis_stabilized
        assert not processed.task_symmetry_used_for_output
        assert np.array_equal(result.pose, result_pose_before)
        assert np.array_equal(processed.raw_pose, result.pose)
        assert np.array_equal(processed.output_pose[:3, 3], result.pose[:3, 3])
        assert not processed.raw_pose.flags.writeable
        assert not processed.output_pose.flags.writeable
        _assert_output_quality(processed.output_pose)
        if previous_output_rotation is not None:
            np.testing.assert_allclose(
                processed.output_pose[:3, :3],
                previous_output_rotation,
                atol=1.0e-12,
                rtol=0.0,
            )
        previous_output_rotation = processed.output_pose[:3, :3]

    print("OBJECT_COUNT=1: raw contract and axial stabilization - PASS")


def test_two_object_state_and_reset_isolation() -> None:
    print("\nTwo independent object processors")
    object_ids = ("pipe1", "pipe2")
    processors = _processor_map(object_ids)
    assert len(processors) == 2
    assert processors["pipe1"] is not processors["pipe2"]

    z_axes = {
        "pipe1": _normalize((0.0, 0.0, 1.0)),
        "pipe2": _normalize((0.0, 1.0, 1.0)),
    }
    phases = {
        "pipe1": (0.0, 95.0, 205.0, 345.0),
        "pipe2": (35.0, 145.0, 260.0, 330.0),
    }
    first_output_rotation = {}

    for frame_id in range(4):
        for object_index, object_id in enumerate(object_ids):
            translation = (
                10.0 * object_index + frame_id,
                -20.0 * object_index - frame_id,
                0.5 + object_index,
            )
            raw = _raw_pose(
                z_axes[object_id],
                phases[object_id][frame_id],
                translation,
            )
            result = _pose_result(object_id, frame_id, raw)
            raw_before = result.pose.copy()
            processed = processors[object_id].process(
                result.pose, result.source_frame_id
            )

            assert np.array_equal(result.pose, raw_before)
            assert np.array_equal(processed.output_pose[:3, 3], raw[:3, 3])
            np.testing.assert_allclose(
                processed.output_pose[:3, 2],
                z_axes[object_id],
                atol=1.0e-12,
                rtol=0.0,
            )
            _assert_output_quality(processed.output_pose)

            if frame_id == 0:
                first_output_rotation[object_id] = processed.output_pose[
                    :3, :3
                ].copy()
            else:
                np.testing.assert_allclose(
                    processed.output_pose[:3, :3],
                    first_output_rotation[object_id],
                    atol=1.0e-12,
                    rtol=0.0,
                )

    assert processors["pipe1"].z_axis_initialized
    assert processors["pipe2"].z_axis_initialized
    pipe2_rotation_before_reset = first_output_rotation["pipe2"].copy()

    processors["pipe1"].reset()
    assert not processors["pipe1"].z_axis_initialized
    assert processors["pipe2"].z_axis_initialized

    pipe2_after_other_reset = processors["pipe2"].process(
        _raw_pose(z_axes["pipe2"], 77.0, (9.0, 8.0, 7.0)),
        frame_id=4,
    )
    np.testing.assert_allclose(
        pipe2_after_other_reset.output_pose[:3, :3],
        pipe2_rotation_before_reset,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert np.array_equal(
        pipe2_after_other_reset.output_pose[:3, 3],
        (9.0, 8.0, 7.0),
    )
    print("OBJECT_COUNT=2: roll removal and different Z directions - PASS")
    print("OBJECT_COUNT=2: state isolation and reset isolation - PASS")


def test_processing_modes_are_explicit() -> None:
    print("\nExplicit output modes")
    passthrough = ObjectPoseProcessor()
    raw = _raw_pose((0.0, 0.0, 1.0), 47.0, (1.0, 2.0, 3.0))
    processed = passthrough.process(raw)
    assert not processed.z_axis_stabilized
    assert not processed.task_symmetry_used_for_output
    assert processed.task_symmetry_index is None
    assert np.array_equal(processed.output_pose, raw)

    task_output = ObjectPoseProcessor(enable_task_symmetry_output=True)
    first = task_output.process(raw, frame_id=0)
    second = task_output.process(raw, frame_id=1)
    assert not second.z_axis_stabilized
    assert second.task_symmetry_used_for_output
    assert first.task_symmetry_index == 0
    assert second.task_symmetry_index is not None

    diagnostic = ObjectPoseProcessor(
        enable_z_axis_stabilization=True,
        enable_task_symmetry_diagnostic=True,
    )
    diagnosed = diagnostic.process(raw, frame_id=0)
    assert diagnosed.z_axis_stabilized
    assert not diagnosed.task_symmetry_used_for_output
    assert diagnosed.task_symmetry_index == 0

    try:
        ObjectPoseProcessor(
            enable_z_axis_stabilization=True,
            enable_task_symmetry_output=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting output modes did not raise ValueError.")
    print("passthrough/task diagnostic/output mode contracts - PASS")


def main() -> int:
    test_single_object_count_and_raw_contract()
    test_two_object_state_and_reset_isolation()
    test_processing_modes_are_explicit()
    print("\nObjectPoseProcessor CPU synthetic tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
