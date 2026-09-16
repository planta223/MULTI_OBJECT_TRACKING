"""Application-wide constants that do not vary between experiments."""

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
    """Lifecycle state of one tracked object."""

    UNINITIALIZED = "uninitialized"
    TRACKING = "tracking"
    LOST = "lost"


class TrackingMode(str, Enum):
    """FoundationPose operation that produced a pose result."""

    REGISTER = "register"
    TRACK = "track"
