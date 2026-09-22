"""Runtime configuration for the multi-object tracking application."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Optional, Tuple

from constants import (
    SUPPORTED_OBJECT_COUNTS,
    SUPPORTED_SEGMENTATION_MODES,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
FOUNDATIONPOSE_ROOT = WORKSPACE_ROOT / "FoundationPose"

# Explicit application defaults. No object is configured until a real CAD is
# supplied; in particular Pipe1 is not duplicated as a placeholder for Pipe2.
CAMERA_TYPE = "cam_ros_zed2i"
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
    zed_resolution: str = "HD720"
    zed_depth_mode: str = "NEURAL"
    ros_color_topic: str = "/cam/color/compressed"
    ros_depth_topic: str = "/cam/depth/compressed"
    ros_camera_info_topic: str = "/cam/color/camera_info"
    ros_frame_timeout_sec: float = 2.0
    ros_node_name: str = "foundationpose_cam_ros_zed2i"


@dataclass(frozen=True)
class FoundationPoseConfig:
    root: Path = FOUNDATIONPOSE_ROOT
    register_refine_iter: int = 5
    track_refine_iter: int = 2
    debug: int = 0


@dataclass(frozen=True)
class ObjectConfig:
    object_id: str
    model_path: Path
    mesh_scale_to_meter: float


@dataclass(frozen=True)
class SegmentationConfig:
    mode: str = SEGMENTATION_MODE
    model_path: Optional[Path] = None
    confidence: float = 0.5
    device: Optional[str] = None
    class_id: Optional[int] = None


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
        if (
            not math.isfinite(self.segmentation.confidence)
            or not 0.0 <= self.segmentation.confidence <= 1.0
        ):
            raise ValueError("Segmentation confidence must be in [0, 1].")
        if self.segmentation.class_id is not None and self.segmentation.class_id < 0:
            raise ValueError("Segmentation class_id must be non-negative.")
        if self.segmentation.mode == "yolo":
            if self.segmentation.model_path is None:
                raise ValueError("YOLO segmentation requires a model_path.")
            if check_model_paths and not Path(self.segmentation.model_path).is_file():
                raise FileNotFoundError(
                    "YOLO segmentation model not found: "
                    f"{self.segmentation.model_path}"
                )
        if self.camera.width <= 0 or self.camera.height <= 0 or self.camera.fps <= 0:
            raise ValueError("Camera width, height, and FPS must be positive.")
        if not self.camera.zed_resolution:
            raise ValueError("zed_resolution must not be empty.")
        if not self.camera.zed_depth_mode:
            raise ValueError("zed_depth_mode must not be empty.")
        if self.camera.camera_type == "cam_ros_zed2i":
            for field_name in (
                "ros_color_topic",
                "ros_depth_topic",
                "ros_camera_info_topic",
                "ros_node_name",
            ):
                if not getattr(self.camera, field_name):
                    raise ValueError(f"{field_name} must not be empty.")
            if (
                not math.isfinite(self.camera.ros_frame_timeout_sec)
                or self.camera.ros_frame_timeout_sec <= 0
            ):
                raise ValueError(
                    "ros_frame_timeout_sec must be finite and positive."
                )
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
