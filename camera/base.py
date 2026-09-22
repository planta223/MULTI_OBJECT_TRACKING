"""Camera-neutral frame contract and minimal source interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    copied = np.array(array, copy=True, order="C")
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class FrameData:
    """One synchronized RGB-D observation in application units."""

    source_frame_id: int
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    device_timestamp_ms: Optional[float]
    timestamp_domain: Optional[str]
    host_wall_time_s: float
    host_monotonic_time_s: float
    # Exact ROS metadata is optional because direct SDK and recorded-sequence
    # sources do not necessarily have a ROS Header.  ROS adapters should carry
    # these fields unchanged so downstream outputs can preserve acquisition
    # time and the real optical frame instead of inventing either value.
    source_timestamp_ns: Optional[int] = None
    camera_frame_id: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.source_frame_id, bool) or not isinstance(
            self.source_frame_id, (int, np.integer)
        ):
            raise TypeError("source_frame_id must be an integer.")
        if self.source_frame_id < 0:
            raise ValueError("source_frame_id must be non-negative.")
        if not isinstance(self.rgb, np.ndarray) or self.rgb.dtype != np.uint8:
            raise TypeError("rgb must be a uint8 NumPy array.")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"rgb shape must be (H, W, 3); got {self.rgb.shape}.")
        if not isinstance(self.depth_m, np.ndarray) or self.depth_m.dtype != np.float32:
            raise TypeError("depth_m must be a float32 NumPy array.")
        if self.depth_m.ndim != 2 or self.rgb.shape[:2] != self.depth_m.shape:
            raise ValueError("RGB and depth resolutions must match.")
        if not isinstance(self.K, np.ndarray) or self.K.shape != (3, 3):
            raise ValueError("K must be a 3x3 NumPy array.")
        if not np.issubdtype(self.K.dtype, np.floating) or not np.all(np.isfinite(self.K)):
            raise TypeError("K must contain finite floating-point values.")
        if self.device_timestamp_ms is not None and not np.isfinite(self.device_timestamp_ms):
            raise ValueError("device_timestamp_ms must be finite when provided.")
        if self.timestamp_domain is not None and not isinstance(self.timestamp_domain, str):
            raise TypeError("timestamp_domain must be a string or None.")
        if not np.isfinite(self.host_wall_time_s) or not np.isfinite(self.host_monotonic_time_s):
            raise ValueError("Host timestamps must be finite.")
        if self.source_timestamp_ns is not None:
            if isinstance(self.source_timestamp_ns, bool) or not isinstance(
                self.source_timestamp_ns, (int, np.integer)
            ):
                raise TypeError("source_timestamp_ns must be an integer or None.")
            if self.source_timestamp_ns < 0:
                raise ValueError("source_timestamp_ns must be non-negative.")
            object.__setattr__(
                self, "source_timestamp_ns", int(self.source_timestamp_ns)
            )
        if self.camera_frame_id is not None:
            if not isinstance(self.camera_frame_id, str):
                raise TypeError("camera_frame_id must be a string or None.")
            if not self.camera_frame_id.strip():
                raise ValueError("camera_frame_id must not be empty.")
        object.__setattr__(self, "rgb", _readonly_copy(self.rgb))
        object.__setattr__(self, "depth_m", _readonly_copy(self.depth_m))
        object.__setattr__(self, "K", _readonly_copy(self.K))

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


class CameraSource(ABC):
    """Minimal camera interface consumed by the tracking application."""

    @property
    def device_name(self) -> Optional[str]:
        return None

    @property
    def device_serial(self) -> Optional[str]:
        return None

    @property
    def depth_scale(self) -> float:
        """Meters represented by one depth value; metric float sources use 1."""

        return 1.0

    @property
    def color_stream_info(self):
        return None

    @property
    def depth_stream_info(self):
        return None

    def raise_if_failed(self) -> None:
        """Propagate asynchronous acquisition failures, if the source has any."""

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def get_next_frame(self) -> FrameData:
        pass
