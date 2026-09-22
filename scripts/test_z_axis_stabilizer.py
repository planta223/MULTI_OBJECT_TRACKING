"""연속적인 Z축 pose 안정화를 위한 CPU 전용 합성 검사."""

import math
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pose_processing.z_axis_stabilizer import (
    ZAxisPoseStabilizer,
    rotation_diagnostics,
)


ANGLE_TOLERANCE_DEG = 1.0e-5
ORTHOGONALITY_TOLERANCE = 1.0e-12
DETERMINANT_TOLERANCE = 1.0e-12


def _normalize(vector: Sequence[float]) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    return values / np.linalg.norm(values)


def _rotation_with_z(z_axis: Sequence[float], axial_phase_deg: float) -> np.ndarray:
    """지정한 Z축과 축 방향 phase를 가진 합성 proper rotation을 만든다."""

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
    return base @ axial_rotation


def _pose(
    z_axis: Sequence[float],
    axial_phase_deg: float = 0.0,
    translation: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _rotation_with_z(z_axis, axial_phase_deg)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return pose


def _rotation_delta_deg(previous: Optional[np.ndarray], current: np.ndarray) -> float:
    if previous is None:
        return float("nan")
    relative = previous.T @ current
    cosine = (np.trace(relative) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _assert_rotation_quality(rotation: np.ndarray) -> None:
    diagnostics = rotation_diagnostics(rotation)
    np.testing.assert_allclose(
        diagnostics.axis_norms,
        (1.0, 1.0, 1.0),
        atol=ORTHOGONALITY_TOLERANCE,
        rtol=0.0,
    )
    assert diagnostics.orthogonality_error <= ORTHOGONALITY_TOLERANCE
    assert diagnostics.maximum_axis_dot <= ORTHOGONALITY_TOLERANCE
    assert abs(diagnostics.determinant - 1.0) <= DETERMINANT_TOLERANCE


def _print_frame(
    frame_id: int,
    axial_phase_deg: float,
    raw_rotation: np.ndarray,
    stable_rotation: np.ndarray,
    previous_raw: Optional[np.ndarray],
    previous_stable: Optional[np.ndarray],
) -> tuple[float, float]:
    raw_delta = _rotation_delta_deg(previous_raw, raw_rotation)
    stable_delta = _rotation_delta_deg(previous_stable, stable_rotation)
    diagnostics = rotation_diagnostics(stable_rotation)
    print(
        f"frame={frame_id:02d} phase={axial_phase_deg:7.1f}deg "
        f"raw_delta={raw_delta:10.6f}deg "
        f"stable_delta={stable_delta:10.6f}deg "
        f"z={np.array2string(stable_rotation[:, 2], precision=6)} "
        f"det={diagnostics.determinant:.12f} "
        f"orth_err={diagnostics.orthogonality_error:.3e}"
    )
    return raw_delta, stable_delta


def _run_sequence(
    name: str,
    z_axes: Iterable[Sequence[float]],
    phases_deg: Sequence[float],
) -> tuple[list[np.ndarray], list[float], list[float]]:
    print(f"\n{name}")
    stabilizer = ZAxisPoseStabilizer()
    stable_rotations = []
    raw_deltas = []
    stable_deltas = []
    previous_raw = None
    previous_stable = None

    for frame_id, (z_axis, phase_deg) in enumerate(zip(z_axes, phases_deg)):
        raw = _pose(z_axis, phase_deg)
        stable = stabilizer.stabilize(raw)
        _assert_rotation_quality(stable[:3, :3])
        raw_delta, stable_delta = _print_frame(
            frame_id,
            phase_deg,
            raw[:3, :3],
            stable[:3, :3],
            previous_raw,
            previous_stable,
        )
        np.testing.assert_allclose(
            stable[:3, 2],
            stabilizer.previous_z,
            atol=0.0,
            rtol=0.0,
        )
        stable_rotations.append(stable[:3, :3].copy())
        raw_deltas.append(raw_delta)
        stable_deltas.append(stable_delta)
        previous_raw = raw[:3, :3]
        previous_stable = stable[:3, :3]

    return stable_rotations, raw_deltas, stable_deltas


def test_case_a_static_z_changing_raw_roll() -> None:
    phases = (0.0, 30.0, 90.0, 150.0, 240.0, 350.0)
    stable, raw_deltas, stable_deltas = _run_sequence(
        "Case A - static Z with changing raw roll",
        ([0.0, 0.0, 1.0] for _ in phases),
        phases,
    )

    for rotation in stable:
        np.testing.assert_allclose(rotation, np.eye(3), atol=1.0e-12, rtol=0.0)
    assert max(raw_deltas[1:]) >= 100.0
    assert max(stable_deltas[1:]) <= ANGLE_TOLERANCE_DEG
    print(
        f"Case A summary: max raw delta={max(raw_deltas[1:]):.6f}deg, "
        f"max stable delta={max(stable_deltas[1:]):.6f}deg - PASS"
    )


def test_case_b_slowly_moving_z() -> None:
    z_axes = (
        (0.00, 0.00, 1.00),
        (0.03, 0.00, 1.00),
        (0.06, 0.02, 1.00),
        (0.09, 0.04, 0.995),
        (0.12, 0.06, 0.990),
    )
    phases = (0.0, 75.0, 150.0, 240.0, 330.0)
    stable, _, stable_deltas = _run_sequence(
        "Case B - slowly moving Z",
        z_axes,
        phases,
    )

    for rotation, expected_z in zip(stable, z_axes):
        np.testing.assert_allclose(
            rotation[:, 2], _normalize(expected_z), atol=1.0e-12, rtol=0.0
        )
    assert max(stable_deltas[1:]) < 5.0
    print(f"Case B summary: max stable delta={max(stable_deltas[1:]):.6f}deg - PASS")


def test_case_c_z_sign_flip() -> None:
    z_axes = (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    phases = (0.0, 45.0, 120.0, 300.0)
    stable, _, stable_deltas = _run_sequence(
        "Case C - raw Z sign flips",
        z_axes,
        phases,
    )

    for rotation in stable:
        np.testing.assert_allclose(
            rotation[:, 2], (0.0, 0.0, 1.0), atol=1.0e-12, rtol=0.0
        )
    assert max(stable_deltas[1:]) <= ANGLE_TOLERANCE_DEG
    print("Case C summary: stable Z never flipped - PASS")


def test_case_d_initial_reference_degeneracy() -> None:
    nearly_camera_x = _normalize((1.0, 1.0e-14, 0.0))
    raw = _pose(nearly_camera_x, 63.0)
    stable = ZAxisPoseStabilizer().stabilize(raw)
    _assert_rotation_quality(stable[:3, :3])
    np.testing.assert_allclose(
        stable[:3, 2], nearly_camera_x, atol=1.0e-12, rtol=0.0
    )
    assert np.all(np.isfinite(stable))
    _print_frame(0, 63.0, raw[:3, :3], stable[:3, :3], None, None)
    print("Case D summary: camera-basis fallback produced a finite rotation - PASS")


def test_case_e_translation_preservation() -> None:
    print("\nCase E - translation preservation")
    stabilizer = ZAxisPoseStabilizer()
    translations = (
        (0.125, -0.250, 1.500),
        (-3.0, 2.0, 0.001),
        (17.25, -8.5, 4.125),
    )
    for frame_id, translation in enumerate(translations):
        raw = _pose(
            (0.02 * frame_id, 0.01 * frame_id, 1.0),
            91.0 * frame_id,
            translation,
        )
        stable = stabilizer.stabilize(raw)
        assert np.array_equal(stable[:3, 3], raw[:3, 3])
        assert np.array_equal(stable[3, :], raw[3, :])
        _assert_rotation_quality(stable[:3, :3])
        print(
            f"frame={frame_id:02d} translation="
            f"{np.array2string(stable[:3, 3], precision=6)} exact_match=True"
        )
    print("Case E summary: translation preserved exactly - PASS")


def test_invalid_inputs_state_isolation_and_reset() -> None:
    print("\nInput validation, instance isolation, and reset")
    stabilizer = ZAxisPoseStabilizer()
    invalid_poses = []
    invalid_poses.append(np.eye(3))
    non_finite = np.eye(4)
    non_finite[0, 0] = np.nan
    invalid_poses.append(non_finite)
    zero_z = np.eye(4)
    zero_z[:3, 2] = 0.0
    invalid_poses.append(zero_z)

    for raw in invalid_poses:
        try:
            stabilizer.stabilize(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid pose did not raise ValueError.")
    assert not stabilizer.initialized

    pipe1_stabilizer = ZAxisPoseStabilizer()
    pipe2_stabilizer = ZAxisPoseStabilizer()
    pipe1_stabilizer.stabilize(_pose((0.0, 0.0, 1.0), 120.0))
    assert pipe1_stabilizer.initialized
    assert not pipe2_stabilizer.initialized
    pipe1_stabilizer.reset()
    assert not pipe1_stabilizer.initialized
    assert pipe1_stabilizer.previous_x is None
    assert pipe1_stabilizer.previous_z is None
    print("Validation/state summary: clear exceptions, isolated state, reset - PASS")


def main() -> int:
    test_case_a_static_z_changing_raw_roll()
    test_case_b_slowly_moving_z()
    test_case_c_z_sign_flip()
    test_case_d_initial_reference_degeneracy()
    test_case_e_translation_preservation()
    test_invalid_inputs_state_isolation_and_reset()
    print("\nZ-axis stabilizer CPU synthetic tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
