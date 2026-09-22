"""Non-blocking ROS2 output adapter for the final processed Pipe pose.

ROS imports are deliberately lazy.  The tracking and pose-processing layers
remain usable in non-ROS environments, and this adapter only serializes the
already-final ``C_T_P`` selected by the application.
"""

from importlib import import_module
from typing import Any, Tuple

import numpy as np

from camera.base import FrameData
from pose_processing.z_axis_stabilizer import rotation_diagnostics


LEFT_PIPE_POSE_TOPIC = "/vision/left_pipe/pose"
LEFT_PIPE_STATUS_TOPIC = "/vision/left_pipe/tracking_status"

# FoundationPose normally returns a valid SO(3) matrix to floating-point
# precision.  These limits reject a genuinely corrupt/non-rigid transform but
# allow tiny inference/numerical drift before the final SVD projection.
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
    """Support both Foxy and newer rclpy QoS enum spellings."""

    value = getattr(qos, modern_name, None)
    if value is None:
        value = getattr(qos, legacy_name)
    return value


def _pose_qos(qos: Any):
    """Keep only the newest pose and never wait for reliable delivery."""

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
    """Keep the latest state available to late-joining Control processes."""

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
    """Validate a near-rotation and project small numerical drift onto SO(3)."""

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
        # This is defensive; the positive input determinant should already
        # make the nearest orthogonal matrix right-handed.
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
    """Convert an SO(3) matrix to normalized quaternion ``[x, y, z, w]``."""

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
    # q and -q encode the same rotation.  Fixing the sign avoids gratuitous
    # sign flips in logs and simple downstream continuity checks.
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def pose_components(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return validated metric translation and normalized xyzw quaternion."""

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
    """Require original ROS acquisition metadata; never substitute wall time."""

    if frame.source_timestamp_ns is None:
        raise ValueError("Frame has no exact source ROS timestamp.")
    if frame.camera_frame_id is None:
        raise ValueError("Frame has no source camera optical frame_id.")
    return frame.source_timestamp_ns, frame.camera_frame_id


class RosPipePosePublisher:
    """Publish the final left-Pipe ``C_T_P`` on the camera source node.

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
        """Publish one of REGISTERING/TRACKING/LOST/INVALID without waiting."""

        if status not in {"REGISTERING", "TRACKING", "LOST", "INVALID"}:
            raise ValueError(f"Unsupported tracking status: {status!r}.")
        if status == self._last_status:
            return True
        message = self._status_message_type()
        message.data = status
        try:
            self._status_publisher.publish(message)
        except Exception as error:  # DDS output must not stop visual tracking.
            self._node.get_logger().error(
                f"Failed to publish left Pipe tracking status: {error}"
            )
            return False
        self._last_status = status
        return True

    def publish_pose(self, pose: np.ndarray, frame: FrameData) -> bool:
        """Publish a valid final pose stamped with its source image header."""

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
        except Exception as error:  # Keep DDS failures out of the tracking loop.
            self._node.get_logger().error(
                f"Failed to publish left Pipe pose: {error}"
            )
            self.publish_status("INVALID")
            return False

        self.publish_status("TRACKING")
        return True

    def destroy(self) -> None:
        """Release publishers before their owning camera node is destroyed."""

        for publisher in (self._pose_publisher, self._status_publisher):
            try:
                self._node.destroy_publisher(publisher)
            except Exception:
                # Node shutdown is already best-effort in the application
                # finally path; there is no tracking result to protect here.
                pass
