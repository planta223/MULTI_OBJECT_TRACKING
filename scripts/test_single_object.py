"""Live single-Pipe camera/FoundationPose regression."""

import argparse
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Optional, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    AppConfig,
    CAMERA_TYPE,
    CameraConfig,
    FeatureFlags,
    FoundationPoseConfig,
    ObjectConfig,
    OutputConfig,
    SegmentationConfig,
)
from constants import PIPE1_ID, SUPPORTED_SEGMENTATION_MODES
from pose_estimation.foundationpose_runtime import FoundationPoseRuntime
from pose_processing.object_pose_processor import ObjectPoseProcessor
from pose_estimation.pose_result import PoseResult
from pose_logging.ros_pose_publisher import (
    LEFT_PIPE_POSE_TOPIC,
    LEFT_PIPE_STATUS_TOPIC,
    RosPipePosePublisher,
)
from tracking.tracking_manager import TrackingManager
from camera import create_camera_source
from visualization.visualization import draw_pose_overlay
from segmentation import SegmentationCancelled, create_segmenter
from task_activation import wait_for_supervisor_activation


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    workspace_root = PROJECT_ROOT.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=workspace_root / "FoundationPose",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mesh-scale-to-meter",
        type=float,
        default=0.001,
        help=(
            "STL unit conversion. The default Pipe1 CAD measures about "
            "99.974 x 94.983 x 316.000 units and is treated as millimeters."
        ),
    )
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial")
    parser.add_argument("--camera-type", default=CAMERA_TYPE)
    parser.add_argument("--zed-resolution", default="HD720")
    parser.add_argument("--zed-depth-mode", default="NEURAL")
    parser.add_argument("--ros-color-topic", default="/cam/color/compressed")
    parser.add_argument("--ros-depth-topic", default="/cam/depth/compressed")
    parser.add_argument(
        "--ros-camera-info-topic", default="/cam/color/camera_info"
    )
    parser.add_argument("--ros-frame-timeout-sec", type=float, default=2.0)
    parser.add_argument("--left-pipe-pose-topic", default=LEFT_PIPE_POSE_TOPIC)
    parser.add_argument("--left-pipe-status-topic", default=LEFT_PIPE_STATUS_TOPIC)
    parser.add_argument(
        "--segmentation-mode",
        choices=SUPPORTED_SEGMENTATION_MODES,
        default="manual",
    )
    parser.add_argument("--yolo-model-path", type=Path)
    parser.add_argument("--yolo-confidence", type=float, default=0.5)
    parser.add_argument("--yolo-device")
    parser.add_argument("--yolo-class-id", type=int)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--register-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--axis-length-m", type=float, default=0.05)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument(
        "--task-symmetry-debug",
        action="store_true",
        help=(
            "Print raw/canonical rotation deltas and the selected assembly "
            "task-symmetry index."
        ),
    )
    parser.add_argument(
        "--z-axis-stabilization",
        action="store_true",
        help=(
            "Use raw FoundationPose Z-axis direction with temporally stable "
            "X/Y axes for visualization. FoundationPose tracking state is "
            "not modified."
        ),
    )
    args = parser.parse_args(argv)

    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("Camera width, height, and FPS must be positive.")
    if args.warmup_seconds < 0:
        parser.error("--warmup-seconds must be non-negative.")
    if args.ros_frame_timeout_sec <= 0:
        parser.error("--ros-frame-timeout-sec must be positive.")
    if args.register_refine_iter <= 0 or args.track_refine_iter <= 0:
        parser.error("FoundationPose iterations must be positive.")
    if args.mesh_scale_to_meter <= 0:
        parser.error("--mesh-scale-to-meter must be positive.")
    if args.axis_length_m <= 0:
        parser.error("--axis-length-m must be positive.")
    if not np.isfinite(args.yolo_confidence) or not 0 <= args.yolo_confidence <= 1:
        parser.error("--yolo-confidence must be in [0, 1].")
    if args.yolo_class_id is not None and args.yolo_class_id < 0:
        parser.error("--yolo-class-id must be non-negative.")
    if args.segmentation_mode == "yolo" and args.yolo_model_path is None:
        parser.error("--yolo-model-path is required with --segmentation-mode yolo.")
    if not args.left_pipe_pose_topic or not args.left_pipe_status_topic:
        parser.error("ROS output topic names must not be empty.")
    return args


def print_result(result: PoseResult) -> None:
    pose_summary = "None"
    if result.pose is not None:
        pose_summary = "t={} z={}".format(
            np.array2string(result.pose[:3, 3], precision=5),
            np.array2string(result.pose[:3, 2], precision=5),
        )
    fps = (
        1.0 / result.processing_time_s
        if result.processing_time_s > 0.0
        else float("inf")
    )
    print(
        f"frame={result.source_frame_id} mode={result.mode.value} "
        f"valid={result.valid} processing={result.processing_time_s * 1000:.2f} "
        f"ms fps={fps:.2f} raw_pose={pose_summary}"
    )
    if result.message:
        print(f"  message={result.message}")


def require_valid(result: PoseResult) -> None:
    if not result.valid or result.pose is None:
        raise RuntimeError(
            f"FoundationPose {result.mode.value} failed: {result.message}"
        )


def _rotation_delta_degrees(
    rotation_a: np.ndarray,
    rotation_b: np.ndarray,
) -> float:
    relative = rotation_a.T @ rotation_b
    cosine = (np.trace(relative) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _print_stabilization_debug(
    frame_id: int,
    raw_pose: np.ndarray,
    stable_pose: np.ndarray,
    previous_stable_pose: Optional[np.ndarray],
) -> None:
    raw_to_stable = _rotation_delta_degrees(
        raw_pose[:3, :3], stable_pose[:3, :3]
    )
    stable_delta = (
        None
        if previous_stable_pose is None
        else _rotation_delta_degrees(
            previous_stable_pose[:3, :3], stable_pose[:3, :3]
        )
    )
    stable_delta_text = "n/a" if stable_delta is None else f"{stable_delta:.3f}deg"
    print(
        f"z_stable frame={frame_id} "
        f"raw_z={np.array2string(raw_pose[:3, 2], precision=5)} "
        f"stable_z={np.array2string(stable_pose[:3, 2], precision=5)} "
        f"raw_to_stable={raw_to_stable:.3f}deg "
        f"stable_delta={stable_delta_text} "
        f"det={np.linalg.det(stable_pose[:3, :3]):.9f}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    camera_config = CameraConfig(
        camera_type=args.camera_type,
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
        zed_resolution=args.zed_resolution,
        zed_depth_mode=args.zed_depth_mode,
        ros_color_topic=args.ros_color_topic,
        ros_depth_topic=args.ros_depth_topic,
        ros_camera_info_topic=args.ros_camera_info_topic,
        ros_frame_timeout_sec=args.ros_frame_timeout_sec,
    )
    object_config = ObjectConfig(
        object_id=PIPE1_ID,
        model_path=args.model_path.expanduser().resolve(),
        mesh_scale_to_meter=args.mesh_scale_to_meter,
    )
    foundationpose_config = FoundationPoseConfig(
        root=args.foundationpose_root.expanduser().resolve(),
        register_refine_iter=args.register_refine_iter,
        track_refine_iter=args.track_refine_iter,
        debug=args.debug,
    )
    segmentation_config = SegmentationConfig(
        mode=args.segmentation_mode,
        model_path=(
            None
            if args.yolo_model_path is None
            else args.yolo_model_path.expanduser().resolve()
        ),
        confidence=args.yolo_confidence,
        device=args.yolo_device,
        class_id=args.yolo_class_id,
    )
    camera = create_camera_source(camera_config)
    window_name = "Pipe1 live FoundationPose"

    with tempfile.TemporaryDirectory(
        prefix="pipe1_live_foundationpose_"
    ) as debug_dir:
        app_config = AppConfig(
            features=FeatureFlags(
                enable_visualization=True,
                enable_pose_logging=False,
                enable_video_save=False,
                enable_manual_reregistration=False,
                enable_relative_pose=False,
            ),
            camera=camera_config,
            segmentation=segmentation_config,
            foundationpose=foundationpose_config,
            objects=(object_config,),
            output=OutputConfig(output_dir=Path(debug_dir)),
        )
        app_config.validate(check_model_paths=True)
        segmenter = create_segmenter(app_config.segmentation, (PIPE1_ID,))

        runtime = FoundationPoseRuntime(args.foundationpose_root)
        manager = TrackingManager(app_config, runtime=runtime)
        tracker = manager.trackers[PIPE1_ID]
        pose_processor = ObjectPoseProcessor(
            enable_z_axis_stabilization=args.z_axis_stabilization,
            enable_task_symmetry_output=not args.z_axis_stabilization,
            enable_task_symmetry_diagnostic=args.task_symmetry_debug,
        )
        print(f"Pipe1 CAD: {tracker.cad.model_path}")
        print(f"Mesh scale: {args.mesh_scale_to_meter} m/unit")
        print(
            "CAD oriented extents: "
            f"{np.array2string(tracker.extents, precision=6)} m"
        )

        pose_publisher = None
        try:
            camera.start()
            if args.camera_type == "cam_ros_zed2i":
                # This is the single-object LEFT-Pipe entrypoint, so Pipe1 has
                # an explicit role here.  Reuse the camera subscriber node;
                # rclpy publish() enqueues a best-effort depth-one sample and
                # does not add another executor or wait to the tracking loop.
                pose_publisher = RosPipePosePublisher(
                    camera.ros_node,
                    pose_topic=args.left_pipe_pose_topic,
                    status_topic=args.left_pipe_status_topic,
                )
            print(f"Camera: {camera.device_name} ({camera.device_serial})")
            print(f"Color stream: {camera.color_stream_info}")
            print(f"Depth stream: {camera.depth_stream_info}")
            print(f"Depth scale: {camera.depth_scale} m/unit")

            if args.warmup_seconds:
                time.sleep(args.warmup_seconds)
            if wait_for_supervisor_activation():
                print("Task activation received; starting YOLO and registration.")
            if pose_publisher is not None:
                pose_publisher.publish_status("REGISTERING")
            camera.raise_if_failed()
            frozen_frame = camera.get_next_frame()
            print(f"Frozen frame for mask: {frozen_frame.source_frame_id}")

            masks = segmenter.segment(frozen_frame)
            # A new registration starts a new application-level stable basis.
            # FoundationPose retains its own independent raw pose chain.
            pose_processor.reset()
            register_result = manager.register_all(frozen_frame, masks)[0]
            print_result(register_result)
            if not register_result.valid and pose_publisher is not None:
                pose_publisher.publish_status("INVALID")
            require_valid(register_result)
            processed_pose = pose_processor.process(
                register_result.pose,
                register_result.source_frame_id,
            )
            if pose_publisher is not None:
                # output_pose is the application-level final C_T_P.  Do not
                # publish the centered visualization transform.
                pose_publisher.publish_pose(
                    processed_pose.output_pose,
                    frozen_frame,
                )
            visualization_label = (
                "Pipe1 Z-axis stable"
                if processed_pose.z_axis_stabilized
                else "Pipe1 task-canonical"
            )
            previous_stable_pose = None
            if args.z_axis_stabilization:
                _print_stabilization_debug(
                    register_result.source_frame_id,
                    processed_pose.raw_pose,
                    processed_pose.output_pose,
                    previous_stable_pose,
                )
                previous_stable_pose = processed_pose.output_pose.copy()

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            while True:
                camera.raise_if_failed()
                frame = camera.get_next_frame()

                track_result = manager.track_all(frame)[0]
                print_result(track_result)
                if not track_result.valid and pose_publisher is not None:
                    pose_publisher.publish_status("LOST")
                require_valid(track_result)
                processed_pose = pose_processor.process(
                    track_result.pose,
                    track_result.source_frame_id,
                )
                if pose_publisher is not None:
                    pose_publisher.publish_pose(
                        processed_pose.output_pose,
                        frame,
                    )
                if args.z_axis_stabilization:
                    if track_result.source_frame_id % 30 == 0:
                        _print_stabilization_debug(
                            track_result.source_frame_id,
                            processed_pose.raw_pose,
                            processed_pose.output_pose,
                            previous_stable_pose,
                        )
                    previous_stable_pose = processed_pose.output_pose.copy()
                track_fps = (
                    1.0 / track_result.processing_time_s
                    if track_result.processing_time_s > 0.0
                    else None
                )
                image = draw_pose_overlay(
                    rgb=frame.rgb,
                    K=frame.K,
                    pose=processed_pose.output_pose,
                    bbox=tracker.bbox,
                    to_origin=tracker.to_origin,
                    fps=track_fps,
                    label=visualization_label,
                    axis_length_m=args.axis_length_m,
                )
                cv2.imshow(window_name, image)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
        except SegmentationCancelled as error:
            if pose_publisher is not None:
                pose_publisher.publish_status("INVALID")
            print(error)
        except Exception:
            if pose_publisher is not None:
                pose_publisher.publish_status("INVALID")
            raise
        finally:
            if pose_publisher is not None:
                pose_publisher.destroy()
            camera.stop()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
