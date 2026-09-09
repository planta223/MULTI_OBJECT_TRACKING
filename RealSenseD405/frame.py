"""Immutable D405 frame data and the verified latest-frame buffer policy."""

from dataclasses import dataclass
import threading
from typing import Optional

import numpy as np


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    copied = np.array(array, copy=True, order="C")
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class D405Frame:
    """One depth-to-color-aligned D405 frame.

    `depth_raw` preserves Z16 sensor units for recording. `depth_m` contains
    the same aligned depth converted to meters for pose estimation.
    """

    source_frame_id: int
    rgb: np.ndarray
    depth_raw: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    device_timestamp_ms: Optional[float]
    timestamp_domain: Optional[str]
    host_wall_time_s: float
    host_monotonic_time_s: float

    def __post_init__(self) -> None:
        if self.source_frame_id < 0:
            raise ValueError("source_frame_id must be non-negative.")
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise TypeError("rgb must have uint8 shape (H, W, 3).")
        if self.depth_raw.dtype != np.uint16 or self.depth_raw.ndim != 2:
            raise TypeError("depth_raw must have uint16 shape (H, W).")
        if self.depth_m.dtype != np.float32 or self.depth_m.ndim != 2:
            raise TypeError("depth_m must have float32 shape (H, W).")
        if self.rgb.shape[:2] != self.depth_raw.shape or self.depth_raw.shape != self.depth_m.shape:
            raise ValueError("RGB, raw depth, and metric depth resolutions must match.")
        if self.K.shape != (3, 3) or not np.all(np.isfinite(self.K)):
            raise ValueError("K must be a finite 3x3 matrix.")
        object.__setattr__(self, "rgb", _readonly_copy(self.rgb))
        object.__setattr__(self, "depth_raw", _readonly_copy(self.depth_raw))
        object.__setattr__(self, "depth_m", _readonly_copy(self.depth_m))
        object.__setattr__(self, "K", _readonly_copy(self.K))


class LatestFrameBuffer:
    """Thread-safe single slot that retains only the newest D405 frame."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: Optional[D405Frame] = None

    def publish(self, frame: D405Frame) -> None:
        if not isinstance(frame, D405Frame):
            raise TypeError("frame must be a D405Frame instance.")
        with self._condition:
            if self._latest is not None and frame.source_frame_id <= self._latest.source_frame_id:
                raise ValueError(
                    "Published source_frame_id must increase strictly: "
                    f"latest={self._latest.source_frame_id}, received={frame.source_frame_id}."
                )
            self._latest = frame
            self._condition.notify_all()

    def clear(self) -> None:
        with self._condition:
            self._latest = None
            self._condition.notify_all()

    def wait_for_newer(self, frame_id: int, timeout_s: Optional[float] = None) -> Optional[D405Frame]:
        with self._condition:
            available = self._condition.wait_for(
                lambda: self._latest is not None and self._latest.source_frame_id > frame_id,
                timeout=timeout_s,
            )
            return self._latest if available else None
