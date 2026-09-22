"""최종 후처리 Pipe pose를 위한 non-blocking ROS2 출력 adapter.

ROS import는 의도적으로 지연한다. tracking 및 pose 처리 계층은 ROS가 없는
환경에서도 사용할 수 있으며, 이 adapter는 애플리케이션이 이미 선택한 최종
``C_T_P``만 직렬화한다.
"""

from importlib import import_module
from typing import Any, Tuple

import numpy as np

from camera.base import FrameData
from pose_processing.z_axis_stabilizer import rotation_diagnostics


LEFT_PIPE_POSE_TOPIC = "/vision/left_pipe/pose"
LEFT_PIPE_STATUS_TOPIC = "/vision/left_pipe/tracking_status"

# FoundationPose는 일반적으로 부동소수점 정밀도 범위에서 유효한 SO(3) 행렬을
# 반환한다. 아래 제한은 실제로 손상되었거나 rigid하지 않은 transform을 거부하되,
# 최종 SVD projection 전의 작은 추론/수치 오차는 허용한다.
_MAX_ORTHOGONALITY_ERROR = 5.0e-2
_MAX_DETERMINANT_ERROR = 5.0e-2
_HOMOGENEOUS_ROW_ATOL = 1.0e-6
_QUATERNION_NORM_EPSILON = 1.0e-12


def _load_ros_output_dependencies():
    try:
        geometry_messages = import_module("geometry_msgs.msg")
        standard_messages = import_module("std_msgs.msg")
        qos = import_module("rclpy.qos")
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            "ROS pose output requires rclpy, geometry_msgs, and std_msgs in "
            "the active ROS2 Python environment."
        ) from error
    return geometry_messages, standard_messages, qos


def _qos_enum(qos: Any, modern_name: str, legacy_name: str):
    """Foxy와 최신 rclpy의 QoS enum 표기를 모두 지원한다."""

    value = getattr(qos, modern_name, None)
    if value is None:
        value = getattr(qos, legacy_name)
    return value


def _pose_qos(qos: Any):
    """최신 pose만 유지하고 reliable 전송을 기다리지 않는다."""

    history = _qos_enum(qos, "HistoryPolicy", "QoSHistoryPolicy")
    reliability = _qos_enum(qos, "ReliabilityPolicy", "QoSReliabilityPolicy")
    durability = _qos_enum(qos, "DurabilityPolicy", "QoSDurabilityPolicy")
    return qos.QoSProfile(
        history=history.KEEP_LAST,
        depth=1,
        reliability=reliability.BEST_EFFORT,
        durability=durability.VOLATILE,
    )


def _status_qos(qos: Any):
    """늦게 참여한 Control process도 최신 상태를 받을 수 있게 유지한다."""

    history = _qos_enum(qos, "HistoryPolicy", "QoSHistoryPolicy")
    reliability = _qos_enum(qos, "ReliabilityPolicy", "QoSReliabilityPolicy")
    durability = _qos_enum(qos, "DurabilityPolicy", "QoSDurabilityPolicy")
    return qos.QoSProfile(
        history=history.KEEP_LAST,
        depth=1,
        reliability=reliability.RELIABLE,
        durability=durability.TRANSIENT_LOCAL,
    )


def _project_to_so3(rotation: np.ndarray) -> np.ndarray:
    """회전에 가까운 행렬을 검증하고 작은 수치 오차를 SO(3)에 투영한다."""

    diagnostics = rotation_diagnostics(rotation)
    if diagnostics.determinant <= 0.0:
        raise ValueError(
            "Pose rotation is reflected or singular: "
            f"det={diagnostics.determinant:.6g}."
        )
    if abs(diagnostics.determinant - 1.0) > _MAX_DETERMINANT_ERROR:
        raise ValueError(
            "Pose rotation determinant is outside tolerance: "
            f"det={diagnostics.determinant:.6g}."
        )
    if diagnostics.orthogonality_error > _MAX_ORTHOGONALITY_ERROR:
        raise ValueError(
            "Pose rotation is not sufficiently orthogonal: "
            f"error={diagnostics.orthogonality_error:.6g}."
        )

    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    normalized = u @ vt
    if np.linalg.det(normalized) <= 0.0:
        # 방어적 처리다. 양수인 입력 determinant만으로도 가장 가까운 직교 행렬은
        # 이미 오른손 좌표계여야 한다.
        u[:, -1] *= -1.0
        normalized = u @ vt
    normalized_diagnostics = rotation_diagnostics(normalized)
    if (
        abs(normalized_diagnostics.determinant - 1.0) > 1.0e-9
        or normalized_diagnostics.orthogonality_error > 1.0e-9
    ):
        raise ValueError("Failed to normalize pose rotation onto SO(3).")
    return normalized


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """SO(3) 행렬을 정규화된 quaternion ``[x, y, z, w]``로 변환한다."""

    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = 2.0 * np.sqrt(
                max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            )
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ],
                dtype=np.float64,
            )
        elif diagonal_index == 1:
            scale = 2.0 * np.sqrt(
                max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            )
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = 2.0 * np.sqrt(
                max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            )
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=np.float64,
            )

    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= _QUATERNION_NORM_EPSILON:
        raise ValueError("Pose rotation produced an invalid quaternion.")
    quaternion /= norm
    # q와 -q는 같은 회전을 나타낸다. 부호를 고정하면 log와 단순 downstream
    # 연속성 검사에서 불필요한 부호 반전을 피할 수 있다.
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def pose_components(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """검증된 meter 단위 translation과 정규화된 xyzw quaternion을 반환한다."""

    try:
        matrix = np.asarray(pose, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Pose matrix must contain numeric values.") from error
    if matrix.shape != (4, 4):
        raise ValueError(f"Pose matrix must have shape (4, 4), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Pose matrix contains NaN or Inf.")
    if not np.allclose(
        matrix[3, :],
        np.array([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=_HOMOGENEOUS_ROW_ATOL,
    ):
        raise ValueError("Pose matrix has an invalid homogeneous bottom row.")

    rotation = _project_to_so3(matrix[:3, :3])
    return matrix[:3, 3].copy(), _rotation_to_quaternion(rotation)


def _source_header(frame: FrameData) -> Tuple[int, str]:
    """원본 ROS 획득 metadata를 요구하며 wall time으로 대체하지 않는다."""

    if frame.source_timestamp_ns is None:
        raise ValueError("Frame has no exact source ROS timestamp.")
    if frame.camera_frame_id is None:
        raise ValueError("Frame has no source camera optical frame_id.")
    return frame.source_timestamp_ns, frame.camera_frame_id


class RosPipePosePublisher:
    """최종 left-Pipe ``C_T_P``를 카메라 source node에서 발행한다.

    ``publish_pose`` performs only bounded matrix checks, message construction,
    and a best-effort depth-one publish.  It contains no waits and never feeds
    data back into the inference/tracking path.
    """

    def __init__(
        self,
        node: Any,
        pose_topic: str = LEFT_PIPE_POSE_TOPIC,
        status_topic: str = LEFT_PIPE_STATUS_TOPIC,
    ) -> None:
        if not pose_topic or not status_topic:
            raise ValueError("ROS output topic names must not be empty.")
        geometry_messages, standard_messages, qos = _load_ros_output_dependencies()
        self._node = node
        self._pose_message_type = geometry_messages.PoseStamped
        self._status_message_type = standard_messages.String
        self._pose_publisher = node.create_publisher(
            self._pose_message_type, pose_topic, _pose_qos(qos)
        )
        self._status_publisher = node.create_publisher(
            self._status_message_type, status_topic, _status_qos(qos)
        )
        self._last_status = None

    def publish_status(self, status: str) -> bool:
        """REGISTERING/TRACKING/LOST/INVALID 중 하나를 기다림 없이 발행한다."""

        if status not in {"REGISTERING", "TRACKING", "LOST", "INVALID"}:
            raise ValueError(f"Unsupported tracking status: {status!r}.")
        if status == self._last_status:
            return True
        message = self._status_message_type()
        message.data = status
        try:
            self._status_publisher.publish(message)
        except Exception as error:  # DDS 출력 오류가 visual tracking을 중단시키면 안 된다.
            self._node.get_logger().error(
                f"Failed to publish left Pipe tracking status: {error}"
            )
            return False
        self._last_status = status
        return True

    def publish_pose(self, pose: np.ndarray, frame: FrameData) -> bool:
        """source image header의 stamp를 사용해 유효한 최종 pose를 발행한다."""

        try:
            translation, quaternion = pose_components(pose)
            stamp_ns, frame_id = _source_header(frame)
            message = self._pose_message_type()
            message.header.stamp.sec = stamp_ns // 1_000_000_000
            message.header.stamp.nanosec = stamp_ns % 1_000_000_000
            message.header.frame_id = frame_id
            message.pose.position.x = float(translation[0])
            message.pose.position.y = float(translation[1])
            message.pose.position.z = float(translation[2])
            message.pose.orientation.x = float(quaternion[0])
            message.pose.orientation.y = float(quaternion[1])
            message.pose.orientation.z = float(quaternion[2])
            message.pose.orientation.w = float(quaternion[3])
            self._pose_publisher.publish(message)
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            self._node.get_logger().error(
                f"Not publishing invalid left Pipe pose: {error}"
            )
            self.publish_status("INVALID")
            return False
        except Exception as error:  # DDS 오류가 tracking loop에 영향을 주지 않게 한다.
            self._node.get_logger().error(
                f"Failed to publish left Pipe pose: {error}"
            )
            self.publish_status("INVALID")
            return False

        self.publish_status("TRACKING")
        return True

    def destroy(self) -> None:
        """소유한 카메라 node가 제거되기 전에 publisher를 해제한다."""

        for publisher in (self._pose_publisher, self._status_publisher):
            try:
                self._node.destroy_publisher(publisher)
            except Exception:
                # 애플리케이션 finally 경로의 node 종료 자체가 best-effort이며,
                # 여기에는 보호해야 할 tracking 결과가 없다.
                pass
