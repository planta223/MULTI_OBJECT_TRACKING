"""객체별 pose estimator에 사용하는 안정적인 tracking 계층 이름."""

from pose_estimation.foundationpose_tracker import (
    CadModel,
    FoundationPoseTracker,
    load_cad_model,
)

ObjectTracker = FoundationPoseTracker

__all__ = ["CadModel", "ObjectTracker", "load_cad_model"]
