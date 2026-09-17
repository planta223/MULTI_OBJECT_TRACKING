"""CPU-only tests for the ROS ZED decoder and exact-stamp matcher."""

from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from camera.base import FrameData
from camera.factory import create_camera_source
from camera.ros_zed import (
    RosZedFrameSynchronizer,
    RosZedSource,
    _load_ros_dependencies,
    camera_matrix_from_message,
    decode_color_message,
    decode_depth_message,
)
from config import CameraConfig


def _header(stamp_ns: int):
    return SimpleNamespace(
        stamp=SimpleNamespace(
            sec=stamp_ns // 1_000_000_000,
            nanosec=stamp_ns % 1_000_000_000,
        )
    )


def _compressed(array: np.ndarray, extension: str, stamp_ns: int):
    success, encoded = cv2.imencode(extension, array)
    assert success
    return SimpleNamespace(header=_header(stamp_ns), data=encoded.tobytes())


def _info(stamp_ns: int, width: int = 4, height: int = 2):
    return SimpleNamespace(
        header=_header(stamp_ns),
        width=width,
        height=height,
        k=[700.0, 0.0, width / 2, 0.0, 710.0, height / 2, 0.0, 0.0, 1.0],
    )


def test_jpeg_bgr_is_decoded_as_rgb_uint8() -> None:
    bgr = np.full((8, 8, 3), (10, 80, 220), dtype=np.uint8)
    rgb = decode_color_message(_compressed(bgr, ".jpg", 1))

    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
    assert int(rgb[0, 0, 0]) > int(rgb[0, 0, 2])


def test_uint16_png_millimeters_are_decoded_as_float32_meters() -> None:
    depth_mm = np.array([[0, 1000], [2500, 4000]], dtype=np.uint16)
    depth_m = decode_depth_message(_compressed(depth_mm, ".png", 2))

    assert depth_m.dtype == np.float32
    np.testing.assert_array_equal(
        depth_m,
        np.array([[0.0, 1.0], [2.5, 4.0]], dtype=np.float32),
    )


def test_camera_info_k_becomes_float64_3x3() -> None:
    K = camera_matrix_from_message(_info(3))

    assert K.shape == (3, 3)
    assert K.dtype == np.float64
    assert K[0, 0] == 700.0
    assert K[1, 1] == 710.0


def test_equal_stamps_make_one_frame_and_upscale_aligned_depth_nearest() -> None:
    synchronizer = RosZedFrameSynchronizer()
    stamp = 4_000_000_005
    bgr = np.full((2, 4, 3), (15, 80, 210), dtype=np.uint8)
    depth_mm = np.array([[0, 1000]], dtype=np.uint16)

    assert synchronizer.add_depth(_compressed(depth_mm, ".png", stamp)) is None
    assert synchronizer.add_color(_compressed(bgr, ".jpg", stamp)) is None
    frame = synchronizer.add_camera_info(_info(stamp))

    assert isinstance(frame, FrameData)
    assert frame.source_frame_id == 0
    assert frame.rgb.shape == (2, 4, 3)
    assert frame.depth_m.shape == (2, 4)
    np.testing.assert_array_equal(
        frame.depth_m,
        np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
    )
    assert frame.K.shape == (3, 3)
    assert synchronizer.add_camera_info(_info(stamp)) is None


def test_different_stamps_are_not_combined() -> None:
    synchronizer = RosZedFrameSynchronizer()
    bgr = np.zeros((2, 4, 3), dtype=np.uint8)
    depth_mm = np.zeros((1, 2), dtype=np.uint16)

    assert synchronizer.add_color(_compressed(bgr, ".jpg", 10)) is None
    assert synchronizer.add_depth(_compressed(depth_mm, ".png", 11)) is None
    assert synchronizer.add_camera_info(_info(12)) is None
    assert synchronizer.wait_for_newer(-1, timeout_s=0.001) is None


def test_slow_consumer_receives_only_latest_completed_frame() -> None:
    synchronizer = RosZedFrameSynchronizer()
    bgr = np.zeros((2, 4, 3), dtype=np.uint8)
    depth_mm = np.zeros((1, 2), dtype=np.uint16)

    for stamp in (40, 41):
        synchronizer.add_color(_compressed(bgr, ".jpg", stamp))
        synchronizer.add_depth(_compressed(depth_mm, ".png", stamp))
        synchronizer.add_camera_info(_info(stamp))

    latest = synchronizer.wait_for_newer(-1, timeout_s=0.001)
    assert latest is not None
    assert latest.source_frame_id == 1
    assert latest.device_timestamp_ms == 41 / 1_000_000.0


def test_non_proportional_depth_resolution_fails_clearly() -> None:
    synchronizer = RosZedFrameSynchronizer()
    stamp = 20
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    depth_mm = np.zeros((2, 3), dtype=np.uint16)

    synchronizer.add_color(_compressed(bgr, ".jpg", stamp))
    synchronizer.add_depth(_compressed(depth_mm, ".png", stamp))
    with pytest.raises(ValueError, match="aspect ratios differ"):
        synchronizer.add_camera_info(_info(stamp, width=4, height=4))


def test_camera_info_resolution_mismatch_fails_clearly() -> None:
    synchronizer = RosZedFrameSynchronizer()
    stamp = 30
    bgr = np.zeros((2, 4, 3), dtype=np.uint8)
    depth_mm = np.zeros((1, 2), dtype=np.uint16)

    synchronizer.add_color(_compressed(bgr, ".jpg", stamp))
    synchronizer.add_depth(_compressed(depth_mm, ".png", stamp))
    with pytest.raises(ValueError, match="CameraInfo resolution"):
        synchronizer.add_camera_info(_info(stamp, width=8, height=4))


def test_factory_adds_ros_zed_without_loading_ros_at_construction() -> None:
    source = create_camera_source(CameraConfig(camera_type="ros_zed"))

    assert isinstance(source, RosZedSource)


def test_missing_ros_runtime_has_actionable_error() -> None:
    with patch(
        "camera.ros_zed.import_module",
        side_effect=ModuleNotFoundError("mock missing ROS"),
    ):
        with pytest.raises(ImportError, match="requires ROS2/rclpy"):
            _load_ros_dependencies()
