"""Assembly-task pose equivalence for the D-cut pipe.

These transforms are deliberately *not* geometric mesh symmetries.  The
D-cut remains part of the CAD seen by FoundationPose.  They only express
orientations that are interchangeable for the downstream flange-assembly
task, and must never be passed to FoundationPose as ``symmetry_tfs``.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np


# Temporary CAD-frame assumptions are centralized here so they can be updated
# after a definitive CAD coordinate-frame check without changing the
# canonicalization algorithm.
PIPE_LENGTH_AXIS = "z"
FLANGE_PHASE_DEGREES = (0.0, 120.0, 240.0)
FRONT_BACK_FLIP_AXIS = "x"
FRONT_BACK_DEGREES = (0.0, 180.0)


def _axis_rotation(axis: str, angle_degrees: float) -> np.ndarray:
    """Create a 4x4 right-handed rotation about one CAD axis."""

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
    """Return the six task-equivalent CAD-frame rotations.

    Index order is identity/Z120/Z240 followed by the corresponding three
    phase rotations combined with the X180 front/back task equivalence.
    Each candidate is formed externally as ``raw_pose @ transform``.
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
    """Return the geodesic SO(3) rotation distance between two poses."""

    rotation_a = np.asarray(pose_a, dtype=np.float64)[:3, :3]
    rotation_b = np.asarray(pose_b, dtype=np.float64)[:3, :3]
    cosine = (np.trace(rotation_a.T @ rotation_b) - 1.0) / 2.0
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


@dataclass(frozen=True)
class TaskPoseResult:
    """One raw FoundationPose result and its task-canonical counterpart."""

    frame_id: int
    raw_pose: np.ndarray
    canonical_pose: np.ndarray
    symmetry_index: int
    raw_delta_deg: Optional[float]
    canonical_delta_deg: Optional[float]


class PipeTaskPoseCanonicalizer:
    """Maintain task-pose continuity independently of FoundationPose state."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = bool(debug)
        self._previous_canonical_pose: Optional[np.ndarray] = None

    @property
    def previous_canonical_pose(self) -> Optional[np.ndarray]:
        if self._previous_canonical_pose is None:
            return None
        return self._previous_canonical_pose.copy()

    def reset(self) -> None:
        """Forget task continuity, for example before a fresh registration."""

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
        """Choose the task-equivalent pose closest to the previous output."""

        raw = self._validated_pose(raw_pose)
        previous = self._previous_canonical_pose

        if previous is None:
            # Registration starts from the full D-cut CAD pose exactly as
            # returned by FoundationPose; no task relabeling on frame one.
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
