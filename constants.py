"""실험마다 바뀌지 않는 애플리케이션 공통 상수."""

from enum import Enum


PIPE1_ID = "pipe1"
PIPE2_ID = "pipe2"
SUPPORTED_OBJECT_COUNTS = (1, 2)
SUPPORTED_SEGMENTATION_MODES = ("manual", "yolo")

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2
PIPE_LENGTH_AXIS_INDEX = AXIS_Z


class TrackingState(str, Enum):
    """Tracking 객체 하나의 수명주기 상태."""

    UNINITIALIZED = "uninitialized"
    TRACKING = "tracking"
    LOST = "lost"


class TrackingMode(str, Enum):
    """Pose 결과를 생성한 FoundationPose 연산 종류."""

    REGISTER = "register"
    TRACK = "track"
