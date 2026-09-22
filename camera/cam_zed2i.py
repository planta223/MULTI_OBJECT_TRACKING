"""Map the sibling ZED2iCamera package to the application frame contract."""

from config import CameraConfig

try:
    from zed2i_camera import ZED2iCamera, ZED2iConfig
except ImportError as error:
    raise ImportError(
        "The sibling ZED2iCamera package is required. Install the ZED SDK "
        "Python API, then run 'python3 -m pip install -e ../ZED2iCamera'."
    ) from error

from .base import CameraSource, FrameData


class ZED2iSource(CameraSource):
    """Convert ZED 2i sensor frames into camera-neutral application frames."""

    def __init__(self, config: CameraConfig) -> None:
        if config.camera_type != "cam_zed2i":
            raise ValueError(f"Expected cam_zed2i, got {config.camera_type!r}.")
        self._camera = ZED2iCamera(
            ZED2iConfig(
                resolution=config.zed_resolution,
                fps=config.fps,
                serial=config.serial,
                depth_mode=config.zed_depth_mode,
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


def create_source(config: CameraConfig) -> CameraSource:
    """Factory hook used by :mod:`camera.factory`."""

    return ZED2iSource(config)
