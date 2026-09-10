"""Object lifecycle and multi-object orchestration."""

from .identity_tracker import IdentityDecision, MultiObjectIdentityTracker
from .kalman_position_tracker import ConstantVelocityPositionKalmanFilter
from .object_tracker import ObjectTracker
from .tracking_manager import TrackingManager

__all__ = [
    "ConstantVelocityPositionKalmanFilter",
    "IdentityDecision",
    "MultiObjectIdentityTracker",
    "ObjectTracker",
    "TrackingManager",
]
