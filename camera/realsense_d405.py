"""Application adapter for the standalone RealSense D405 layer."""

from config import CameraConfig
from RealSenseD405.camera import D405Camera, D405Config

from .base import CameraSource, FrameData


class RealSenseD405Source(CameraSource):
    """Convert D405 sensor frames into camera-neutral application frames."""

    def __init__(self, config: CameraConfig) -> None:
        if config.camera_type != "realsense_d405":
            raise ValueError(f"Expected realsense_d405, got {config.camera_type!r}.")
        self._camera = D405Camera(
            D405Config(
                width=config.width,
                height=config.height,
                fps=config.fps,
                serial=config.serial,
            )
        )

    @property
    def is_running(self) -> bool:
        return self._camera.is_running

    @property
    def device_name(self):
        return self._camera.device_name

    @property
    def device_serial(self):
        return self._camera.device_serial

    @property
    def depth_scale(self):
        return self._camera.depth_scale

    @property
    def color_stream_info(self):
        return self._camera.color_stream_info

    @property
    def depth_stream_info(self):
        return self._camera.depth_stream_info

    def start(self) -> None:
        self._camera.start()

    def stop(self) -> None:
        self._camera.stop()

    def raise_if_failed(self) -> None:
        self._camera.raise_if_failed()

    def get_next_frame(self) -> FrameData:
        frame = self._camera.get_next_frame()
        return FrameData(
            source_frame_id=frame.source_frame_id,
            rgb=frame.rgb,
            depth_m=frame.depth_m,
            K=frame.K,
            device_timestamp_ms=frame.device_timestamp_ms,
            timestamp_domain=frame.timestamp_domain,
            host_wall_time_s=frame.host_wall_time_s,
            host_monotonic_time_s=frame.host_monotonic_time_s,
        )
