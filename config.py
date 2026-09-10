"""Runtime configuration for the multi-object tracking application."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Optional, Tuple

from constants import (
    SUPPORTED_IDENTITY_MODES,
    SUPPORTED_OBJECT_COUNTS,
    SUPPORTED_SEGMENTATION_MODES,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
FOUNDATIONPOSE_ROOT = WORKSPACE_ROOT / "FoundationPose"

# Explicit application defaults. No object is configured until a real CAD is
# supplied; in particular Pipe1 is not duplicated as a placeholder for Pipe2.
CAMERA_TYPE = "realsense_d405"
SEGMENTATION_MODE = "manual"
ENABLE_VISUALIZATION = True
ENABLE_POSE_LOGGING = False
OBJECTS: Tuple["ObjectConfig", ...] = ()
OBJECT_COUNT = len(OBJECTS)


@dataclass(frozen=True)
class FeatureFlags:
    enable_visualization: bool = ENABLE_VISUALIZATION
    enable_pose_logging: bool = ENABLE_POSE_LOGGING
    enable_video_save: bool = False
    enable_manual_reregistration: bool = False
    enable_relative_pose: bool = False


@dataclass(frozen=True)
class CameraConfig:
    camera_type: str = CAMERA_TYPE
    width: int = 848
    height: int = 480
    fps: int = 30
    serial: Optional[str] = None


@dataclass(frozen=True)
class FoundationPoseConfig:
    root: Path = FOUNDATIONPOSE_ROOT
    register_refine_iter: int = 5
    track_refine_iter: int = 2
    debug: int = 0


@dataclass(frozen=True)
class TrackingConfig:
    """Estimator-independent temporal identity protection settings."""

    identity_mode: str = "depth_motion"
    process_acceleration_std_mps2: float = 0.75
    measurement_position_std_m: float = 0.02
    initial_position_std_m: float = 0.02
    initial_velocity_std_mps: float = 0.25
    mahalanobis_gate_threshold_sq: float = 11.345
    association_margin_sq: float = 1.0
    collapse_distance_m: float = 0.05
    occlusion_overlap_ratio: float = 0.25
    occlusion_min_depth_separation_m: float = 0.04
    occlusion_depth_margin_m: float = 0.01
    occlusion_min_valid_depth_pixels: int = 20
    max_predict_only_seconds: float = 1.0
    max_prediction_std_m: float = 0.25
    identity_debug: bool = False


@dataclass(frozen=True)
class ObjectConfig:
    object_id: str
    model_path: Path
    mesh_scale_to_meter: float


@dataclass(frozen=True)
class SegmentationConfig:
    mode: str = SEGMENTATION_MODE


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path = PROJECT_ROOT / "results"
    video_fps: float = 10.0


@dataclass(frozen=True)
class AppConfig:
    features: FeatureFlags = field(default_factory=FeatureFlags)
    camera: CameraConfig = field(default_factory=CameraConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    foundationpose: FoundationPoseConfig = field(default_factory=FoundationPoseConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    objects: Tuple[ObjectConfig, ...] = field(default_factory=lambda: OBJECTS)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def object_count(self) -> int:
        return len(self.objects)

    def validate(self, check_model_paths: bool = False) -> None:
        if not isinstance(self.camera.camera_type, str) or re.fullmatch(
            r"[a-z][a-z0-9_]*",
            self.camera.camera_type,
        ) is None:
            raise ValueError(
                "camera_type must name a lowercase module in camera/: "
                f"{self.camera.camera_type!r}."
            )
        if self.segmentation.mode not in SUPPORTED_SEGMENTATION_MODES:
            raise NotImplementedError(
                f"Segmentation mode is not implemented: {self.segmentation.mode}"
            )
        if self.camera.width <= 0 or self.camera.height <= 0 or self.camera.fps <= 0:
            raise ValueError("Camera width, height, and FPS must be positive.")
        if self.object_count not in SUPPORTED_OBJECT_COUNTS:
            raise ValueError(
                "Configure one or two real objects; "
                f"received {self.object_count}."
            )
        if self.foundationpose.register_refine_iter <= 0:
            raise ValueError("register_refine_iter must be positive.")
        if self.foundationpose.track_refine_iter <= 0:
            raise ValueError("track_refine_iter must be positive.")
        if self.foundationpose.debug < 0:
            raise ValueError("FoundationPose debug level cannot be negative.")
        if not isinstance(self.foundationpose.root, Path):
            raise TypeError("FoundationPose root must be a pathlib.Path.")
        if self.tracking.identity_mode not in SUPPORTED_IDENTITY_MODES:
            raise ValueError(
                f"Unsupported identity mode: {self.tracking.identity_mode!r}."
            )
        positive_tracking_values = {
            "process_acceleration_std_mps2": self.tracking.process_acceleration_std_mps2,
            "measurement_position_std_m": self.tracking.measurement_position_std_m,
            "initial_position_std_m": self.tracking.initial_position_std_m,
            "initial_velocity_std_mps": self.tracking.initial_velocity_std_mps,
            "mahalanobis_gate_threshold_sq": self.tracking.mahalanobis_gate_threshold_sq,
            "collapse_distance_m": self.tracking.collapse_distance_m,
            "occlusion_min_depth_separation_m": self.tracking.occlusion_min_depth_separation_m,
            "occlusion_depth_margin_m": self.tracking.occlusion_depth_margin_m,
            "max_predict_only_seconds": self.tracking.max_predict_only_seconds,
            "max_prediction_std_m": self.tracking.max_prediction_std_m,
        }
        for name, value in positive_tracking_values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if (
            not math.isfinite(self.tracking.association_margin_sq)
            or self.tracking.association_margin_sq < 0
        ):
            raise ValueError("association_margin_sq must be finite and non-negative.")
        if not 0 < self.tracking.occlusion_overlap_ratio <= 1:
            raise ValueError("occlusion_overlap_ratio must be in (0, 1].")
        if self.tracking.occlusion_min_valid_depth_pixels < 1:
            raise ValueError("occlusion_min_valid_depth_pixels must be positive.")

        object_ids = [item.object_id for item in self.objects]
        if any(not object_id for object_id in object_ids):
            raise ValueError("object_id must not be empty.")
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("Object IDs must be unique.")
        for item in self.objects:
            if not math.isfinite(item.mesh_scale_to_meter) or item.mesh_scale_to_meter <= 0:
                raise ValueError(
                    f"mesh_scale_to_meter must be positive for {item.object_id}."
                )
            if check_model_paths and not Path(item.model_path).is_file():
                raise FileNotFoundError(
                    f"CAD file not found for {item.object_id}: {item.model_path}"
                )
        if not math.isfinite(self.output.video_fps) or self.output.video_fps <= 0:
            raise ValueError("Output video_fps must be positive.")
