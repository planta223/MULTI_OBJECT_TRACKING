"""D405 직접 카메라 adapter의 CPU 전용 계약 검사."""

from unittest.mock import patch

import numpy as np

from camera.base import FrameData
from camera.factory import create_camera_source
from camera.cam_d405 import RealSenseD405Source
from config import CameraConfig
from realsense_d405 import D405Frame


class FakeD405Camera:
    def __init__(self, config) -> None:
        self.config = config
        self.is_running = False
        self.device_name = "mock D405"
        self.device_serial = "mock-serial"
        self.depth_scale = 0.001
        self.color_stream_info = None
        self.depth_stream_info = None

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    def raise_if_failed(self) -> None:
        return None

    def get_next_frame(self) -> D405Frame:
        raw_depth = np.array([[0, 1000], [2000, 3000]], dtype=np.uint16)
        return D405Frame(
            source_frame_id=7,
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_raw=raw_depth,
            depth_m=raw_depth.astype(np.float32) * np.float32(self.depth_scale),
            K=np.eye(3, dtype=np.float64),
            device_timestamp_ms=10.5,
            timestamp_domain="mock_clock",
            host_wall_time_s=20.0,
            host_monotonic_time_s=15.0,
        )


def test_adapter_maps_sensor_frame_to_application_contract() -> None:
    with patch("camera.cam_d405.D405Camera", FakeD405Camera):
        source = RealSenseD405Source(CameraConfig(camera_type="cam_d405"))
        source.start()
        frame = source.get_next_frame()

        assert isinstance(frame, FrameData)
        assert frame.source_frame_id == 7
        assert frame.rgb.shape == (2, 2, 3)
        assert frame.depth_m.dtype == np.float32
        assert np.isclose(frame.depth_m[1, 1], np.float32(3.0))
        assert frame.K.shape == (3, 3)
        assert frame.timestamp_domain == "mock_clock"
        assert source.depth_scale == 0.001

        source.stop()
        assert source.is_running is False


def test_factory_loads_selected_adapter_lazily() -> None:
    with patch("camera.cam_d405.D405Camera", FakeD405Camera):
        source = create_camera_source(CameraConfig(camera_type="cam_d405"))

    assert isinstance(source, RealSenseD405Source)
