"""Stable tracking-layer name for the object-specific pose estimator."""

from pose_estimation.foundationpose_tracker import (
    CadModel,
    FoundationPoseTracker,
    load_cad_model,
)

ObjectTracker = FoundationPoseTracker

__all__ = ["CadModel", "ObjectTracker", "load_cad_model"]
