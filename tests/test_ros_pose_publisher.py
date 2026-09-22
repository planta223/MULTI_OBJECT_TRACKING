"""최종 pose 검증과 ROS message 채우기의 CPU 전용 검사."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from camera.base import FrameData
from pose_logging.ros_pose_publisher import RosPipePosePublisher, pose_components


def _frame() -> FrameData:
    return FrameData(
        source_frame_id=7,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3), dtype=np.float32),
        K=np.eye(3, dtype=np.float64),
        device_timestamp_ms=12_345.000000006,
        timestamp_domain="ros_publish_clock",
        host_wall_time_s=100.0,
        host_monotonic_time_s=200.0,
        source_timestamp_ns=12_345_000_000_006,
        camera_frame_id="camera_left_optical_frame",
    )


class _PoseStamped:
    def __init__(self):
        self.header = SimpleNamespace(
            stamp=SimpleNamespace(sec=0, nanosec=0), frame_id=""
        )
        self.pose = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        )


class _String:
    def __init__(self):
        self.data = ""


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Logger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)


class _Node:
    def __init__(self):
        self.publishers = []
        self.logger = _Logger()

    def create_publisher(self, message_type, topic, qos):
        del message_type, topic, qos
        publisher = _Publisher()
        self.publishers.append(publisher)
        return publisher

    def destroy_publisher(self, publisher):
        del publisher

    def get_logger(self):
        return self.logger


class _QoSProfile:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Policy:
    KEEP_LAST = "keep_last"
    BEST_EFFORT = "best_effort"
    RELIABLE = "reliable"
    VOLATILE = "volatile"
    TRANSIENT_LOCAL = "transient_local"


_FAKE_QOS = SimpleNamespace(
    QoSProfile=_QoSProfile,
    HistoryPolicy=_Policy,
    ReliabilityPolicy=_Policy,
    DurabilityPolicy=_Policy,
)
_FAKE_GEOMETRY = SimpleNamespace(PoseStamped=_PoseStamped)
_FAKE_STANDARD = SimpleNamespace(String=_String)


def _publisher():
    node = _Node()
    dependency_tuple = (_FAKE_GEOMETRY, _FAKE_STANDARD, _FAKE_QOS)
    patcher = patch(
        "pose_logging.ros_pose_publisher._load_ros_output_dependencies",
        return_value=dependency_tuple,
    )
    patcher.start()
    try:
        publisher = RosPipePosePublisher(node)
    finally:
        patcher.stop()
    return node, publisher


def test_pose_components_normalize_small_rotation_drift() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array(
        [[0.0, -1.0001, 0.0], [0.9999, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    pose[:3, 3] = (0.1, -0.2, 0.8)

    translation, quaternion = pose_components(pose)

    np.testing.assert_allclose(translation, (0.1, -0.2, 0.8))
    np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(
        quaternion,
        (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)),
        atol=1.0e-5,
    )


def test_pose_uses_exact_image_stamp_and_optical_frame() -> None:
    node, publisher = _publisher()
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (0.12, -0.34, 0.56)

    assert publisher.publish_pose(pose, _frame())

    message = node.publishers[0].messages[-1]
    assert message.header.stamp.sec == 12_345
    assert message.header.stamp.nanosec == 6
    assert message.header.frame_id == "camera_left_optical_frame"
    assert message.pose.position.x == 0.12
    assert message.pose.position.y == -0.34
    assert message.pose.position.z == 0.56
    assert message.pose.orientation.w == 1.0
    assert node.publishers[1].messages[-1].data == "TRACKING"


def test_nonfinite_pose_is_not_published_and_status_is_invalid() -> None:
    node, publisher = _publisher()
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = np.nan

    assert not publisher.publish_pose(pose, _frame())

    assert node.publishers[0].messages == []
    assert node.publishers[1].messages[-1].data == "INVALID"
    assert node.logger.errors


def test_reflection_is_not_published() -> None:
    node, publisher = _publisher()
    pose = np.eye(4, dtype=np.float64)
    pose[0, 0] = -1.0

    assert not publisher.publish_pose(pose, _frame())

    assert node.publishers[0].messages == []
    assert node.publishers[1].messages[-1].data == "INVALID"


def test_missing_source_header_never_falls_back_to_processing_time() -> None:
    node, publisher = _publisher()
    frame = _frame()
    frame_without_ros_header = FrameData(
        source_frame_id=frame.source_frame_id,
        rgb=frame.rgb,
        depth_m=frame.depth_m,
        K=frame.K,
        device_timestamp_ms=frame.device_timestamp_ms,
        timestamp_domain=frame.timestamp_domain,
        host_wall_time_s=frame.host_wall_time_s,
        host_monotonic_time_s=frame.host_monotonic_time_s,
    )

    assert not publisher.publish_pose(np.eye(4), frame_without_ros_header)
    assert node.publishers[0].messages == []
    assert node.publishers[1].messages[-1].data == "INVALID"
