"""D-cut pipe의 조립 task pose 동치 관계.

이 transform들은 의도적으로 기하학적 mesh symmetry가 *아니다*. D-cut은
FoundationPose가 보는 CAD의 일부로 유지된다. downstream flange 조립 task에서
서로 바꿔 쓸 수 있는 방향만 표현하며, FoundationPose에 ``symmetry_tfs``로
전달해서는 안 된다.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np


# 임시 CAD frame 가정을 이곳에 모아 둔다. 이후 CAD 좌표계를 확정해도
# canonicalization 알고리즘을 변경하지 않고 갱신할 수 있다.
PIPE_LENGTH_AXIS = "z"
FLANGE_PHASE_DEGREES = (0.0, 120.0, 240.0)
FRONT_BACK_FLIP_AXIS = "x"
FRONT_BACK_DEGREES = (0.0, 180.0)


def _axis_rotation(axis: str, angle_degrees: float) -> np.ndarray:
    """CAD 축 하나를 중심으로 하는 4x4 오른손 회전을 생성한다."""

    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.eye(4, dtype=np.float64)

    if axis == "x":
        rotation[:3, :3] = (
            (1.0, 0.0, 0.0),
            (0.0, cosine, -sine),
            (0.0, sine, cosine),
        )
    elif axis == "y":
        rotation[:3, :3] = (
            (cosine, 0.0, sine),
            (0.0, 1.0, 0.0),
            (-sine, 0.0, cosine),
        )
    elif axis == "z":
        rotation[:3, :3] = (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
    else:
        raise ValueError(f"Unsupported CAD axis: {axis!r}")
    return rotation


def build_pipe_task_symmetry_transforms() -> Tuple[np.ndarray, ...]:
    """task 관점에서 동치인 CAD frame 회전 여섯 개를 반환한다.

    index 순서는 identity/Z120/Z240 이후, 각 phase 회전에 X180 앞뒤 task
    동치를 결합한 세 항목이다. 각 후보는 외부에서
    ``raw_pose @ transform``으로 생성한다.
    """

    transforms = []
    for front_back_degrees in FRONT_BACK_DEGREES:
        front_back = _axis_rotation(
            FRONT_BACK_FLIP_AXIS,
            front_back_degrees,
        )
        for phase_degrees in FLANGE_PHASE_DEGREES:
            phase = _axis_rotation(PIPE_LENGTH_AXIS, phase_degrees)
            transforms.append(phase @ front_back)
    return tuple(transforms)


PIPE_TASK_SYMMETRY_TRANSFORMS = build_pipe_task_symmetry_transforms()


def rotation_distance_rad(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """두 pose 사이의 SO(3) geodesic 회전 거리를 반환한다."""

    rotation_a = np.asarray(pose_a, dtype=np.float64)[:3, :3]
    rotation_b = np.asarray(pose_b, dtype=np.float64)[:3, :3]
    cosine = (np.trace(rotation_a.T @ rotation_b) - 1.0) / 2.0
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


@dataclass(frozen=True)
class TaskPoseResult:
    """FoundationPose 원본 결과 하나와 task 기준 canonical 결과."""

    frame_id: int
    raw_pose: np.ndarray
    canonical_pose: np.ndarray
    symmetry_index: int
    raw_delta_deg: Optional[float]
    canonical_delta_deg: Optional[float]


class PipeTaskPoseCanonicalizer:
    """FoundationPose 상태와 독립적으로 task pose 연속성을 유지한다."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = bool(debug)
        self._previous_canonical_pose: Optional[np.ndarray] = None

    @property
    def previous_canonical_pose(self) -> Optional[np.ndarray]:
        if self._previous_canonical_pose is None:
            return None
        return self._previous_canonical_pose.copy()

    def reset(self) -> None:
        """새 registration 전과 같이 필요할 때 task 연속성 이력을 지운다."""

        self._previous_canonical_pose = None

    @staticmethod
    def _validated_pose(raw_pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(raw_pose, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"raw_pose must have shape (4, 4), got {pose.shape}.")
        if not np.all(np.isfinite(pose)):
            raise ValueError("raw_pose contains non-finite values.")
        return np.array(pose, copy=True, order="C")

    def canonicalize(self, raw_pose: np.ndarray, frame_id: int) -> TaskPoseResult:
        """이전 출력에 가장 가까운 task 동치 pose를 선택한다."""

        raw = self._validated_pose(raw_pose)
        previous = self._previous_canonical_pose

        if previous is None:
            # registration은 FoundationPose가 반환한 전체 D-cut CAD pose에서
            # 그대로 시작하며 첫 frame에는 task 재지정을 하지 않는다.
            symmetry_index = 0
            canonical = raw.copy()
            raw_delta_deg = None
            canonical_delta_deg = None
        else:
            candidates = tuple(
                raw @ transform for transform in PIPE_TASK_SYMMETRY_TRANSFORMS
            )
            candidate_distances = tuple(
                rotation_distance_rad(previous, candidate)
                for candidate in candidates
            )
            symmetry_index = int(np.argmin(candidate_distances))
            canonical = np.array(
                candidates[symmetry_index],
                copy=True,
                order="C",
            )
            raw_delta_deg = math.degrees(rotation_distance_rad(previous, raw))
            canonical_delta_deg = math.degrees(
                candidate_distances[symmetry_index]
            )

        self._previous_canonical_pose = canonical.copy()
        result = TaskPoseResult(
            frame_id=int(frame_id),
            raw_pose=raw,
            canonical_pose=canonical,
            symmetry_index=symmetry_index,
            raw_delta_deg=raw_delta_deg,
            canonical_delta_deg=canonical_delta_deg,
        )
        if self.debug:
            self._print_debug(result)
        return result

    @staticmethod
    def _print_debug(result: TaskPoseResult) -> None:
        raw_delta = (
            "n/a"
            if result.raw_delta_deg is None
            else f"{result.raw_delta_deg:.1f} deg"
        )
        canonical_delta = (
            "n/a"
            if result.canonical_delta_deg is None
            else f"{result.canonical_delta_deg:.1f} deg"
        )
        print(
            f"frame={result.frame_id} raw_delta={raw_delta} "
            f"symmetry={result.symmetry_index} "
            f"canonical_delta={canonical_delta}"
        )
