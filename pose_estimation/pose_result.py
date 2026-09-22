"""pose 핵심 계층과 출력 adapter가 공유하는 결과 계약."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from constants import TrackingMode, TrackingState


@dataclass(frozen=True)
class PoseResult:
    """처리 주기 한 번에서 객체 하나에 대한 결과.

    ``pose``가 있으면 원본 CAD 좌표계에서 OpenCV 카메라 좌표계로 변환하는
    기준 4x4 transform이다.
    """

    cycle_id: int
    object_id: str
    source_frame_id: int
    host_wall_time_s: float
    host_monotonic_time_s: float
    mode: TrackingMode
    state: TrackingState
    valid: bool
    pose: Optional[np.ndarray]
    processing_time_s: float
    message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.cycle_id < 0:
            raise ValueError("cycle_id must be non-negative.")
        if not self.object_id:
            raise ValueError("object_id must not be empty.")
        if self.source_frame_id < 0:
            raise ValueError("source_frame_id must be non-negative.")
        if not isinstance(self.mode, TrackingMode):
            raise TypeError("mode must be a TrackingMode value.")
        if not isinstance(self.state, TrackingState):
            raise TypeError("state must be a TrackingState value.")
        if self.processing_time_s < 0 or not np.isfinite(self.processing_time_s):
            raise ValueError("processing_time_s must be finite and non-negative.")
        if not np.isfinite(self.host_wall_time_s):
            raise ValueError("host_wall_time_s must be finite.")
        if not np.isfinite(self.host_monotonic_time_s):
            raise ValueError("host_monotonic_time_s must be finite.")

        if self.valid:
            if self.pose is None:
                raise ValueError("pose is required when valid=True.")
            if not isinstance(self.pose, np.ndarray):
                raise TypeError("pose must be a NumPy array.")
            if self.pose.shape != (4, 4):
                raise ValueError(
                    f"pose shape must be (4, 4); received {self.pose.shape}."
                )
            if not np.issubdtype(self.pose.dtype, np.number):
                raise TypeError(f"pose must be numeric; received {self.pose.dtype}.")
            if not np.all(np.isfinite(self.pose)):
                raise ValueError("pose must contain only finite values.")

            pose = np.array(self.pose, copy=True, order="C")
            pose.setflags(write=False)
            object.__setattr__(self, "pose", pose)
        elif self.pose is not None:
            raise ValueError("pose must be None when valid=False.")

    @property
    def translation_m(self) -> Optional[np.ndarray]:
        if self.pose is None:
            return None
        return self.pose[:3, 3].copy()

    @property
    def rotation_matrix(self) -> Optional[np.ndarray]:
        if self.pose is None:
            return None
        return self.pose[:3, :3].copy()
