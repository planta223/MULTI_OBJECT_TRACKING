"""ZED 2i 직접 카메라 adapter의 CPU 전용 계약 검사."""

from unittest.mock import patch

import numpy as np

from camera.base import FrameData
from camera.factory import create_camera_source
from camera.cam_zed2i import ZED2iSource
from config import CameraConfig
from zed2i_camera import ZED2iFrame


class FakeZED2iCamera:
    def __init__(self, config) -> None:
        self.config = config
        self.is_running = False
        self.device_name = "mock ZED 2i"
        self.device_serial = "123456"
        self.depth_scale = 1.0
        self.color_stream_info = "mock RGB8"
        self.depth_stream_info = "mock F32_M"

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    def raise_if_failed(self) -> None:
        return None

    def get_next_frame(self) -> ZED2iFrame:
        return ZED2iFrame(
            source_frame_id=9,
            rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            depth_m=np.full((2, 3), 1.5, dtype=np.float32),
            K=np.array(
                [[700.0, 0.0, 1.5], [0.0, 710.0, 1.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            device_timestamp_ms=12.25,
            timestamp_domain="zed_image_clock",
            host_wall_time_s=20.0,
            host_monotonic_time_s=10.0,
        )


def test_adapter_maps_zed_frame_to_application_contract() -> None:
    config = CameraConfig(
        camera_type="cam_zed2i",
        fps=30,
        serial="123456",
        zed_resolution="HD720",
        zed_depth_mode="NEURAL",
    )
    with patch("camera.cam_zed2i.ZED2iCamera", FakeZED2iCamera):
        source = ZED2iSource(config)
        source.start()
        frame = source.get_next_frame()

        assert isinstance(frame, FrameData)
        assert frame.source_frame_id == 9
        assert frame.rgb.shape == (2, 3, 3)
        assert frame.rgb.dtype == np.uint8
        assert frame.depth_m.shape == (2, 3)
        assert frame.depth_m.dtype == np.float32
        assert np.all(frame.depth_m == np.float32(1.5))
        assert frame.K.shape == (3, 3)
        assert frame.device_timestamp_ms == 12.25
        assert frame.timestamp_domain == "zed_image_clock"
        assert source.depth_scale == 1.0

        source.stop()
        assert source.is_running is False


def test_factory_loads_zed_adapter_lazily() -> None:
    config = CameraConfig(camera_type="cam_zed2i")
    with patch("camera.cam_zed2i.ZED2iCamera", FakeZED2iCamera):
        source = create_camera_source(config)

    assert isinstance(source, ZED2iSource)
