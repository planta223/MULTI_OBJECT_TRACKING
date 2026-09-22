"""기존 ``cam_zed.py`` 토픽을 위한 ROS2 subscriber adapter.

이 모듈은 import 시점에 ROS나 카메라 SDK를 요구하지 않는다. 따라서 동기화와
디코딩 핵심 로직은 CPU 환경에서 테스트할 수 있으며, ROS import는
:meth:`RosZedSource.start` 호출 시점까지 지연된다.
"""

from dataclasses import dataclass
from importlib import import_module
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from config import CameraConfig

from .base import CameraSource, FrameData


_MAX_PENDING_STAMPS = 16
_EXECUTOR_STOP_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class RosZedStreamInfo:
    width: int
    height: int
    format: str


def _stamp_ns(message: Any) -> int:
    """ROS 메시지 header stamp를 정수 nanosecond key 하나로 반환한다."""

    try:
        sec = int(message.header.stamp.sec)
        nanosec = int(message.header.stamp.nanosec)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("ROS message is missing a valid header.stamp.") from error
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError(
            f"Invalid ROS timestamp: sec={sec}, nanosec={nanosec}."
        )
    return sec * 1_000_000_000 + nanosec


def _camera_frame_id(message: Any) -> str:
    """ROS Header의 optical frame을 검증해 반환한다."""

    try:
        frame_id = message.header.frame_id
    except AttributeError as error:
        raise ValueError("ROS message is missing header.frame_id.") from error
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError("ROS message header.frame_id must be a non-empty string.")
    return frame_id


def decode_color_message(message: Any) -> np.ndarray:
    """``CompressedImage`` JPEG byte를 연속 RGB uint8 배열로 디코딩한다."""

    encoded = np.frombuffer(message.data, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode ROS ZED color JPEG.")
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(
            "Decoded ROS ZED color must be BGR uint8 HxWx3; "
            f"got shape={bgr.shape}, dtype={bgr.dtype}."
        )
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def decode_depth_message(message: Any) -> np.ndarray:
    """Millimeter 단위 ``16UC1; png``를 meter 단위 float32로 디코딩한다."""

    encoded = np.frombuffer(message.data, dtype=np.uint8)
    depth_mm = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise ValueError("Failed to decode ROS ZED depth PNG.")
    if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
        raise ValueError(
            "Decoded ROS ZED depth must be uint16 HxW millimeters; "
            f"got shape={depth_mm.shape}, dtype={depth_mm.dtype}."
        )
    return np.ascontiguousarray(
        depth_mm.astype(np.float32) * np.float32(0.001)
    )


def camera_matrix_from_message(message: Any) -> np.ndarray:
    """CameraInfo에서 rectified-left 3x3 intrinsic 행렬을 읽는다."""

    K = np.asarray(message.k, dtype=np.float64)
    if K.size != 9:
        raise ValueError(f"CameraInfo.k must contain 9 values; got {K.size}.")
    K = np.ascontiguousarray(K.reshape(3, 3))
    if not np.all(np.isfinite(K)):
        raise ValueError("CameraInfo.k contains non-finite values.")
    return K


def _normalize_depth_to_color(
    depth_m: np.ndarray,
    color_shape: Tuple[int, int],
) -> np.ndarray:
    """저해상도 aligned-left depth를 color pixel grid에 대응시킨다.

    ``cam_zed.py``는 같은 ZED grab에서 두 출력을 얻고 SDK에 저해상도
    aligned-left depth를 요청한다. 따라서 비율을 유지한 resize는 좌표 대응을
    보존한다. 불연속 경계에 새로운 depth 값이 보간되지 않도록 nearest-neighbor를
    사용한다.
    """

    color_height, color_width = color_shape
    depth_height, depth_width = depth_m.shape
    if (depth_height, depth_width) == color_shape:
        return depth_m
    if depth_width * color_height != color_width * depth_height:
        raise ValueError(
            "ROS ZED color/depth aspect ratios differ, so aligned depth cannot "
            "be mapped by proportional nearest-neighbor scaling: "
            f"color={color_width}x{color_height}, "
            f"depth={depth_width}x{depth_height}."
        )
    return np.ascontiguousarray(
        cv2.resize(
            depth_m,
            (color_width, color_height),
            interpolation=cv2.INTER_NEAREST,
        ),
        dtype=np.float32,
    )


class RosZedFrameSynchronizer:
    """제한된 저장 공간을 쓰는 exact-stamp 3개 토픽 matcher."""

    def __init__(self, max_pending_stamps: int = _MAX_PENDING_STAMPS) -> None:
        if max_pending_stamps <= 0:
            raise ValueError("max_pending_stamps must be positive.")
        self._max_pending_stamps = max_pending_stamps
        self._pending_lock = threading.Lock()
        self._color: Dict[int, Any] = {}
        self._depth: Dict[int, Any] = {}
        self._info: Dict[int, Any] = {}
        self._completed = set()
        self._completed_order = []
        self._condition = threading.Condition()
        self._latest: Optional[FrameData] = None
        self._failure: Optional[BaseException] = None
        self._next_frame_id = 0

    def clear(self) -> None:
        with self._pending_lock:
            self._color.clear()
            self._depth.clear()
            self._info.clear()
            self._completed.clear()
            self._completed_order.clear()
            self._next_frame_id = 0
        with self._condition:
            self._latest = None
            self._failure = None
            self._condition.notify_all()

    def fail(self, error: BaseException) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = error
            self._condition.notify_all()

    def raise_if_failed(self) -> None:
        with self._condition:
            error = self._failure
        if error is not None:
            raise RuntimeError("ROS ZED subscriber callback failed.") from error

    def add_color(self, message: Any) -> Optional[FrameData]:
        return self._add("color", message)

    def add_depth(self, message: Any) -> Optional[FrameData]:
        return self._add("depth", message)

    def add_camera_info(self, message: Any) -> Optional[FrameData]:
        return self._add("info", message)

    def _add(self, kind: str, message: Any) -> Optional[FrameData]:
        stamp = _stamp_ns(message)
        triple = None
        frame_id = None
        with self._pending_lock:
            if stamp in self._completed:
                return None
            target = {"color": self._color, "depth": self._depth, "info": self._info}[
                kind
            ]
            target[stamp] = message
            self._prune_locked()
            if stamp in self._color and stamp in self._depth and stamp in self._info:
                triple = (
                    self._color.pop(stamp),
                    self._depth.pop(stamp),
                    self._info.pop(stamp),
                )
                frame_id = self._next_frame_id
                self._next_frame_id += 1
                self._completed.add(stamp)
                self._completed_order.append(stamp)
                while len(self._completed_order) > self._max_pending_stamps:
                    expired = self._completed_order.pop(0)
                    self._completed.discard(expired)

        if triple is None or frame_id is None:
            return None
        frame = self._make_frame(stamp, frame_id, *triple)
        with self._condition:
            self._latest = frame
            self._condition.notify_all()
        return frame

    def _prune_locked(self) -> None:
        stamps = set(self._color) | set(self._depth) | set(self._info)
        overflow = len(stamps) - self._max_pending_stamps
        if overflow <= 0:
            return
        for stamp in sorted(stamps)[:overflow]:
            self._color.pop(stamp, None)
            self._depth.pop(stamp, None)
            self._info.pop(stamp, None)

    @staticmethod
    def _make_frame(
        stamp_ns: int,
        source_frame_id: int,
        color_message: Any,
        depth_message: Any,
        info_message: Any,
    ) -> FrameData:
        frame_ids = {
            _camera_frame_id(color_message),
            _camera_frame_id(depth_message),
            _camera_frame_id(info_message),
        }
        if len(frame_ids) != 1:
            raise ValueError(
                "Synchronized ROS ZED messages use different frame_id values: "
                f"{sorted(frame_ids)}."
            )
        camera_frame_id = frame_ids.pop()
        rgb = decode_color_message(color_message)
        original_depth_m = decode_depth_message(depth_message)
        K = camera_matrix_from_message(info_message)
        info_width = int(info_message.width)
        info_height = int(info_message.height)
        rgb_height, rgb_width = rgb.shape[:2]
        if (info_height, info_width) != (rgb_height, rgb_width):
            raise ValueError(
                "ROS ZED CameraInfo resolution does not match color: "
                f"info={info_width}x{info_height}, "
                f"color={rgb_width}x{rgb_height}."
            )
        depth_m = _normalize_depth_to_color(original_depth_m, rgb.shape[:2])
        host_wall_time_s = time.time()
        host_monotonic_time_s = time.monotonic()
        return FrameData(
            source_frame_id=source_frame_id,
            rgb=rgb,
            depth_m=depth_m,
            K=K,
            device_timestamp_ms=float(stamp_ns) / 1_000_000.0,
            timestamp_domain="ros_publish_clock",
            host_wall_time_s=host_wall_time_s,
            host_monotonic_time_s=host_monotonic_time_s,
            source_timestamp_ns=stamp_ns,
            camera_frame_id=camera_frame_id,
        )

    def wait_for_newer(self, frame_id: int, timeout_s: float) -> Optional[FrameData]:
        with self._condition:
            available = self._condition.wait_for(
                lambda: self._failure is not None
                or (
                    self._latest is not None
                    and self._latest.source_frame_id > frame_id
                ),
                timeout=timeout_s,
            )
            if self._failure is not None:
                error = self._failure
                raise RuntimeError("ROS ZED subscriber callback failed.") from error
            return self._latest if available else None


def _load_ros_dependencies():
    try:
        rclpy = import_module("rclpy")
        executors = import_module("rclpy.executors")
        qos = import_module("rclpy.qos")
        sensor_messages = import_module("sensor_msgs.msg")
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            "cam_ros_zed2i camera backend requires ROS2/rclpy and sensor_msgs in "
            "the active Python environment."
        ) from error
    return rclpy, executors, qos, sensor_messages


class RosZedSource(CameraSource):
    """제조사 SDK를 import하지 않고 ``cam_zed.py``를 구독한다."""

    def __init__(self, config: CameraConfig) -> None:
        if config.camera_type != "cam_ros_zed2i":
            raise ValueError(
                f"Expected cam_ros_zed2i, got {config.camera_type!r}."
            )
        for field_name in (
            "ros_color_topic",
            "ros_depth_topic",
            "ros_camera_info_topic",
            "ros_node_name",
        ):
            if not getattr(config, field_name):
                raise ValueError(f"{field_name} must not be empty.")
        if (
            not np.isfinite(config.ros_frame_timeout_sec)
            or config.ros_frame_timeout_sec <= 0
        ):
            raise ValueError("ros_frame_timeout_sec must be finite and positive.")
        self.config = config
        self._synchronizer = RosZedFrameSynchronizer()
        self._last_returned_frame_id = -1
        self._node = None
        self._executor = None
        self._thread = None
        self._rclpy = None
        self._owns_rclpy = False
        self._stopping = threading.Event()
        self._stream_lock = threading.Lock()
        self._color_stream_info = None
        self._depth_stream_info = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def device_name(self) -> str:
        return "ZED 2i via ROS2"

    @property
    def depth_scale(self) -> float:
        return 1.0

    @property
    def color_stream_info(self):
        with self._stream_lock:
            return self._color_stream_info

    @property
    def depth_stream_info(self):
        with self._stream_lock:
            return self._depth_stream_info

    @property
    def ros_node(self):
        """가벼운 in-process 출력 adapter가 사용할 내부 node를 제공한다.

        Node의 소유와 삭제 책임은 이 카메라 입력에 있다. 이를 재사용하면
        rqt_graph에서 tracking 애플리케이션이 단일
        ``foundationpose_cam_ros_zed2i`` node로 표시되고 두 번째 executor
        thread도 만들지 않는다.
        """

        if self._node is None:
            raise RuntimeError("RosZedSource.start() must be called first.")
        return self._node

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError("RosZedSource is already running.")
        rclpy, executors, qos, sensor_messages = _load_ros_dependencies()
        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)
        self._synchronizer.clear()
        self._last_returned_frame_id = -1
        self._stopping.clear()
        try:
            self._node = rclpy.create_node(self.config.ros_node_name)
            profile = qos.qos_profile_sensor_data
            self._node.create_subscription(
                sensor_messages.CompressedImage,
                self.config.ros_color_topic,
                self._color_callback,
                profile,
            )
            self._node.create_subscription(
                sensor_messages.CompressedImage,
                self.config.ros_depth_topic,
                self._depth_callback,
                profile,
            )
            self._node.create_subscription(
                sensor_messages.CameraInfo,
                self.config.ros_camera_info_topic,
                self._info_callback,
                profile,
            )
            self._executor = executors.SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._thread = threading.Thread(
                target=self._spin,
                name="ros-zed-executor",
                daemon=False,
            )
            self._thread.start()
        except BaseException:
            self.stop()
            raise

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except BaseException as error:
            if not self._stopping.is_set():
                self._synchronizer.fail(error)

    def _handle_message(self, callback, message: Any) -> None:
        try:
            frame = callback(message)
            if frame is not None:
                with self._stream_lock:
                    self._color_stream_info = RosZedStreamInfo(
                        frame.width, frame.height, "RGB8"
                    )
                    self._depth_stream_info = RosZedStreamInfo(
                        frame.width, frame.height, "F32_M"
                    )
        except BaseException as error:
            self._synchronizer.fail(error)

    def _color_callback(self, message: Any) -> None:
        self._handle_message(self._synchronizer.add_color, message)

    def _depth_callback(self, message: Any) -> None:
        self._handle_message(self._synchronizer.add_depth, message)

    def _info_callback(self, message: Any) -> None:
        self._handle_message(self._synchronizer.add_camera_info, message)

    def raise_if_failed(self) -> None:
        self._synchronizer.raise_if_failed()

    def get_next_frame(self) -> FrameData:
        if not self.is_running:
            self.raise_if_failed()
            raise RuntimeError("RosZedSource.start() must be called first.")
        frame = self._synchronizer.wait_for_newer(
            self._last_returned_frame_id,
            self.config.ros_frame_timeout_sec,
        )
        if frame is None:
            self.raise_if_failed()
            raise TimeoutError(
                "Timed out waiting for synchronized ROS ZED color, depth, and "
                f"CameraInfo after {self.config.ros_frame_timeout_sec:.3g} s."
            )
        self._last_returned_frame_id = frame.source_frame_id
        return frame

    def stop(self) -> None:
        self._stopping.set()
        executor = self._executor
        thread = self._thread
        node = self._node
        rclpy = self._rclpy
        if executor is not None:
            executor.shutdown(timeout_sec=_EXECUTOR_STOP_TIMEOUT_S)
        if thread is not None:
            thread.join(timeout=_EXECUTOR_STOP_TIMEOUT_S)
        if thread is not None and thread.is_alive():
            raise RuntimeError("ROS ZED executor thread did not stop.")
        if executor is not None and node is not None:
            executor.remove_node(node)
        if node is not None:
            node.destroy_node()
        if self._owns_rclpy and rclpy is not None and rclpy.ok():
            rclpy.shutdown()
        self._executor = None
        self._thread = None
        self._node = None
        self._rclpy = None
        self._owns_rclpy = False


def create_source(config: CameraConfig) -> CameraSource:
    return RosZedSource(config)
