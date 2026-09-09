"""Threaded Intel RealSense RGB-D input adapter.

The acquisition thread exclusively owns every librealsense pipeline operation.
Consumers receive only immutable :class:`D405Frame` snapshots through a
single-slot :class:`LatestFrameBuffer`; no SDK frame or profile object crosses
the input boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from types import ModuleType
from typing import Optional

import numpy as np

from .frame import D405Frame, LatestFrameBuffer


_FRAME_WAIT_TIMEOUT_MS = 1000
_START_TIMEOUT_S = 15.0
_STOP_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class D405Config:
    """Stream selection for the standalone D405 acquisition layer."""

    width: int = 848
    height: int = 480
    fps: int = 30
    serial: Optional[str] = None


@dataclass(frozen=True)
class D405StreamInfo:
    """Actual stream parameters selected by librealsense."""

    width: int
    height: int
    fps: int
    format: str


class D405Camera:
    """Publish aligned RealSense RGB-D frames from one producer thread."""

    def __init__(
        self,
        config: "D405Config",
        frame_buffer: Optional[LatestFrameBuffer] = None,
    ) -> None:
        if not isinstance(config, D405Config):
            raise TypeError("config must be a D405Config instance.")
        if config.width <= 0 or config.height <= 0 or config.fps <= 0:
            raise ValueError("Camera width, height, and FPS must be positive.")
        if config.serial is not None:
            if not isinstance(config.serial, str):
                raise TypeError("Camera serial must be a string or None.")
            if not config.serial.strip():
                raise ValueError("Camera serial must be non-empty when provided.")

        self.config = config
        self.frame_buffer = frame_buffer or LatestFrameBuffer()

        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._producer_error: Optional[BaseException] = None
        self._last_returned_frame_id = -1

        self._device_name: Optional[str] = None
        self._device_serial: Optional[str] = None
        self._depth_scale: Optional[float] = None
        self._color_stream_info: Optional[D405StreamInfo] = None
        self._depth_stream_info: Optional[D405StreamInfo] = None

    @staticmethod
    def _load_sdk() -> ModuleType:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise ImportError(
                "pyrealsense2 is required for live RealSense input."
            ) from error
        return rs

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def producer_error(self) -> Optional[BaseException]:
        """The exception that terminated acquisition, if one occurred."""

        with self._state_lock:
            return self._producer_error

    @property
    def device_name(self) -> Optional[str]:
        with self._state_lock:
            return self._device_name

    @property
    def device_serial(self) -> Optional[str]:
        with self._state_lock:
            return self._device_serial

    @property
    def depth_scale(self) -> Optional[float]:
        with self._state_lock:
            return self._depth_scale

    @property
    def color_stream_info(self) -> Optional[D405StreamInfo]:
        with self._state_lock:
            return self._color_stream_info

    @property
    def depth_stream_info(self) -> Optional[D405StreamInfo]:
        with self._state_lock:
            return self._depth_stream_info

    def raise_if_failed(self) -> None:
        """Raise the producer exception in the calling/consumer thread."""

        error = self.producer_error
        if error is not None:
            raise RuntimeError("RealSense acquisition thread failed.") from error

    def start(self) -> None:
        """Start the pipeline in its owner thread and wait until it is ready."""

        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("D405Camera is already running.")

            self._producer_error = None
            self._device_name = None
            self._device_serial = None
            self._depth_scale = None
            self._color_stream_info = None
            self._depth_stream_info = None
            self._last_returned_frame_id = -1
            self._stop_event.clear()
            self._started_event.clear()
            self.frame_buffer.clear()
            self._thread = threading.Thread(
                target=self._acquisition_loop,
                name="realsense-acquisition",
                daemon=True,
            )
            thread = self._thread

        thread.start()
        if not self._started_event.wait(timeout=_START_TIMEOUT_S):
            self._stop_event.set()
            raise TimeoutError(
                f"RealSense pipeline did not start within {_START_TIMEOUT_S:.0f} s."
            )

        self.raise_if_failed()

    def stop(self) -> None:
        """Request producer shutdown and wait for pipeline.stop() to finish."""

        with self._state_lock:
            thread = self._thread

        if thread is None:
            return

        self._stop_event.set()
        thread.join(timeout=_STOP_TIMEOUT_S)
        if thread.is_alive():
            raise RuntimeError(
                f"RealSense acquisition thread did not stop within "
                f"{_STOP_TIMEOUT_S:.0f} s."
            )

        with self._state_lock:
            if self._thread is thread:
                self._thread = None

    def get_next_frame(self) -> D405Frame:
        """Wait for the next frame newer than the last one returned here."""

        if not self.is_running:
            self.raise_if_failed()
            raise RuntimeError("D405Camera.start() must be called first.")

        frame = self.frame_buffer.wait_for_newer(
            self._last_returned_frame_id,
            timeout_s=(_FRAME_WAIT_TIMEOUT_MS / 1000.0) + 0.5,
        )
        if frame is None:
            self.raise_if_failed()
            raise TimeoutError("Timed out waiting for a RealSense frame.")

        self._last_returned_frame_id = frame.source_frame_id
        return frame

    @staticmethod
    def _stream_info(stream_profile: object) -> D405StreamInfo:
        video_profile = stream_profile.as_video_stream_profile()
        return D405StreamInfo(
            width=int(video_profile.width()),
            height=int(video_profile.height()),
            fps=int(video_profile.fps()),
            format=str(video_profile.format()),
        )

    @staticmethod
    def _intrinsic_matrix(video_profile: object) -> np.ndarray:
        intrinsics = video_profile.get_intrinsics()
        return np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _timestamp_domain(frame: object) -> Optional[str]:
        try:
            return str(frame.get_frame_timestamp_domain())
        except (AttributeError, RuntimeError):
            return None

    def _store_pipeline_metadata(
        self,
        rs: ModuleType,
        pipeline_profile: object,
        depth_scale: float,
    ) -> None:
        device = pipeline_profile.get_device()
        color_profile = pipeline_profile.get_stream(rs.stream.color)
        depth_profile = pipeline_profile.get_stream(rs.stream.depth)

        def device_info(field: object) -> Optional[str]:
            try:
                if device.supports(field):
                    return str(device.get_info(field))
            except (AttributeError, RuntimeError):
                pass
            return None

        with self._state_lock:
            self._device_name = device_info(rs.camera_info.name)
            self._device_serial = device_info(rs.camera_info.serial_number)
            self._depth_scale = depth_scale
            self._color_stream_info = self._stream_info(color_profile)
            self._depth_stream_info = self._stream_info(depth_profile)

    def _make_frame(
        self,
        aligned_frames: object,
        depth_scale: float,
        host_wall_time_s: float,
        host_monotonic_time_s: float,
    ) -> Optional[D405Frame]:
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            return None

        rgb = np.asanyarray(color_frame.get_data())
        raw_depth = np.asanyarray(aligned_depth_frame.get_data())
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"Unexpected RealSense RGB frame: shape={rgb.shape}, "
                f"dtype={rgb.dtype}."
            )
        if raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
            raise ValueError(
                f"Unexpected RealSense depth frame: shape={raw_depth.shape}, "
                f"dtype={raw_depth.dtype}."
            )

        depth_m = raw_depth.astype(np.float32)
        depth_m *= np.float32(depth_scale)

        # Librealsense defines zero in Z16 as "no depth", and multiplication
        # already maps it to 0.0 m.  There is no general SDK guarantee that
        # raw 65535 is invalid/saturated, so it is deliberately preserved.
        depth_m[raw_depth == 0] = np.float32(0.0)

        aligned_depth_profile = (
            aligned_depth_frame.profile.as_video_stream_profile()
        )
        K = self._intrinsic_matrix(aligned_depth_profile)

        return D405Frame(
            source_frame_id=int(color_frame.get_frame_number()),
            rgb=rgb,
            depth_raw=raw_depth,
            depth_m=depth_m,
            K=K,
            device_timestamp_ms=float(color_frame.get_timestamp()),
            timestamp_domain=self._timestamp_domain(color_frame),
            host_wall_time_s=host_wall_time_s,
            host_monotonic_time_s=host_monotonic_time_s,
        )

    def _acquisition_loop(self) -> None:
        pipeline = None
        pipeline_started = False
        try:
            rs = self._load_sdk()
            context = rs.context()
            if len(context.query_devices()) == 0:
                raise RuntimeError(
                    "No accessible RealSense device was found. Check the USB "
                    "connection and Linux udev permissions."
                )
            pipeline = rs.pipeline()
            sdk_config = rs.config()
            if self.config.serial is not None:
                sdk_config.enable_device(self.config.serial)
            sdk_config.enable_stream(
                rs.stream.color,
                self.config.width,
                self.config.height,
                rs.format.rgb8,
                self.config.fps,
            )
            sdk_config.enable_stream(
                rs.stream.depth,
                self.config.width,
                self.config.height,
                rs.format.z16,
                self.config.fps,
            )
            align = rs.align(rs.stream.color)

            pipeline_profile = pipeline.start(sdk_config)
            pipeline_started = True
            depth_scale = float(
                pipeline_profile.get_device()
                .first_depth_sensor()
                .get_depth_scale()
            )
            if not np.isfinite(depth_scale) or depth_scale <= 0:
                raise ValueError(
                    f"RealSense returned invalid depth scale: {depth_scale}."
                )

            self._store_pipeline_metadata(rs, pipeline_profile, depth_scale)
            self._started_event.set()

            while not self._stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(_FRAME_WAIT_TIMEOUT_MS)
                except RuntimeError:
                    if self._stop_event.is_set():
                        break
                    raise

                host_wall_time_s = time.time()
                host_monotonic_time_s = time.monotonic()
                aligned_frames = align.process(frames)
                frame = self._make_frame(
                    aligned_frames=aligned_frames,
                    depth_scale=depth_scale,
                    host_wall_time_s=host_wall_time_s,
                    host_monotonic_time_s=host_monotonic_time_s,
                )
                if frame is not None:
                    self.frame_buffer.publish(frame)
        except BaseException as error:
            with self._state_lock:
                self._producer_error = error
            self._started_event.set()
        finally:
            if pipeline is not None and pipeline_started:
                try:
                    pipeline.stop()
                except BaseException as error:
                    with self._state_lock:
                        if self._producer_error is None:
                            self._producer_error = error
            self._started_event.set()
