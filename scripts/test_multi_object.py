"""Live dual-object camera regression using one frame for Pipe1 and Pipe2."""

import argparse
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict, Optional, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera import create_camera_source
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
from constants import PIPE1_ID, PIPE2_ID, SUPPORTED_SEGMENTATION_MODES
from pose_estimation.foundationpose_runtime import FoundationPoseRuntime
from pose_estimation.pose_result import PoseResult
from pose_logging.ros_pose_publisher import (
    LEFT_PIPE_POSE_TOPIC,
    LEFT_PIPE_STATUS_TOPIC,
    RosPipePosePublisher,
)
from pose_processing.object_pose_processor import ObjectPoseProcessor
from segmentation import SegmentationCancelled, create_segmenter
from task_activation import wait_for_supervisor_activation
from tracking.tracking_manager import TrackingManager
from visualization.visualization import draw_pose_overlay_on_bgr


_DISPLAY_NAMES = {PIPE1_ID: "Pipe1", PIPE2_ID: "Pipe2"}
_DISPLAY_COLORS = {
    PIPE1_ID: (0, 255, 255),
    PIPE2_ID: (255, 0, 255),
}
_TEXT_ROWS = {PIPE1_ID: 28, PIPE2_ID: 56}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=PROJECT_ROOT.parent / "FoundationPose",
    )
    parser.add_argument(
        "--pipe1-model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "pipe1.obj",
    )
    parser.add_argument(
        "--pipe2-model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "pipe2.obj",
    )
    parser.add_argument(
        "--mesh-scale-to-meter",
        type=float,
        default=0.001,
        help="CAD unit conversion; both Pipe OBJ files are in millimeters.",
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
    parser.add_argument(
        "--left-pipe-object-id",
        choices=(PIPE1_ID, PIPE2_ID),
        help=(
            "Explicitly map one registered tracker identity to the left-hand "
            "Pipe ROS output. Required to publish a pose in dual-object mode."
        ),
    )
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
    parser.add_argument(
        "--show-auto-mask",
        action="store_true",
        help="Show and save the YOLO masks before FoundationPose registration.",
    )
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--register-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--axis-length-m", type=float, default=0.05)
    parser.add_argument("--debug", type=int, default=0)
    output_pose_group = parser.add_mutually_exclusive_group()
    output_pose_group.add_argument(
        "--task-symmetry-output",
        action="store_true",
        help="Apply task-symmetry canonicalization independently to each object.",
    )
    output_pose_group.add_argument(
        "--z-axis-stabilization",
        action="store_true",
        help="Apply Z-axis stabilization independently to each object.",
    )
    parser.add_argument(
        "--task-symmetry-debug",
        action="store_true",
        help="Print per-object task-symmetry diagnostics without enabling output.",
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
    if not np.isfinite(args.mesh_scale_to_meter) or args.mesh_scale_to_meter <= 0:
        parser.error("--mesh-scale-to-meter must be finite and positive.")
    if not np.isfinite(args.axis_length_m) or args.axis_length_m <= 0:
        parser.error("--axis-length-m must be finite and positive.")
    if args.debug < 0:
        parser.error("--debug must be non-negative.")
    if not np.isfinite(args.yolo_confidence) or not 0 <= args.yolo_confidence <= 1:
        parser.error("--yolo-confidence must be in [0, 1].")
    if args.yolo_class_id is not None and args.yolo_class_id < 0:
        parser.error("--yolo-class-id must be non-negative.")
    if args.segmentation_mode == "yolo" and args.yolo_model_path is None:
        parser.error("--yolo-model-path is required with --segmentation-mode yolo.")
    if args.show_auto_mask and args.segmentation_mode != "yolo":
        parser.error("--show-auto-mask requires --segmentation-mode yolo.")
    if (
        args.left_pipe_object_id is not None
        and args.camera_type != "cam_ros_zed2i"
    ):
        parser.error(
            "--left-pipe-object-id requires --camera-type cam_ros_zed2i."
        )
    if not args.left_pipe_pose_topic or not args.left_pipe_status_topic:
        parser.error("ROS output topic names must not be empty.")
    return args


def _print_result(result: PoseResult) -> None:
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
        f"object={result.object_id} frame={result.source_frame_id} "
        f"mode={result.mode.value} state={result.state.value} "
        f"valid={result.valid} "
        f"processing={result.processing_time_s * 1000:.2f} ms "
        f"fps={fps:.2f} raw_pose={pose_summary}"
    )
    if result.message:
        print(f"  message={result.message}")


def _results_by_object(results: Sequence[PoseResult]) -> Dict[str, PoseResult]:
    by_object = {result.object_id: result for result in results}
    expected = {PIPE1_ID, PIPE2_ID}
    if set(by_object) != expected or len(results) != len(expected):
        raise RuntimeError(
            "TrackingManager returned unexpected object IDs: "
            f"{sorted(by_object)}; expected {sorted(expected)}."
        )
    return by_object


def _draw_results(
    frame_rgb: np.ndarray,
    K: np.ndarray,
    results: Dict[str, PoseResult],
    processed_poses: Dict[str, np.ndarray],
    manager: TrackingManager,
    axis_length_m: float,
) -> np.ndarray:
    image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for object_id in (PIPE1_ID, PIPE2_ID):
        result = results[object_id]
        color = _DISPLAY_COLORS[object_id]
        text_y = _TEXT_ROWS[object_id]
        if result.valid:
            tracker = manager.trackers[object_id]
            track_fps = (
                1.0 / result.processing_time_s
                if result.processing_time_s > 0.0
                else None
            )
            image = draw_pose_overlay_on_bgr(
                image_bgr=image,
                K=K,
                pose=processed_poses[object_id],
                bbox=tracker.bbox,
                to_origin=tracker.to_origin,
                fps=track_fps,
                label=_DISPLAY_NAMES[object_id],
                axis_length_m=axis_length_m,
                overlay_color=color,
                text_origin=(15, text_y),
            )
        else:
            message = result.message or "invalid pose"
            cv2.putText(
                image,
                f"{_DISPLAY_NAMES[object_id]} LOST: {message[:70]}",
                (15, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        image,
        "Pipe1:yellow  Pipe2:magenta  X:red Y:green Z:blue  Q/Esc:quit",
        (15, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


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
    object_configs = (
        ObjectConfig(
            object_id=PIPE1_ID,
            model_path=args.pipe1_model_path.expanduser().resolve(),
            mesh_scale_to_meter=args.mesh_scale_to_meter,
        ),
        ObjectConfig(
            object_id=PIPE2_ID,
            model_path=args.pipe2_model_path.expanduser().resolve(),
            mesh_scale_to_meter=args.mesh_scale_to_meter,
        ),
    )
    camera = create_camera_source(camera_config)
    window_name = "Pipe1 + Pipe2 live FoundationPose"

    with tempfile.TemporaryDirectory(
        prefix="pipe1_pipe2_live_foundationpose_"
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
            objects=object_configs,
            output=OutputConfig(output_dir=Path(debug_dir)),
        )
        app_config.validate(check_model_paths=True)
        segmenter = create_segmenter(
            app_config.segmentation,
            (PIPE1_ID, PIPE2_ID),
        )

        runtime = FoundationPoseRuntime(foundationpose_config.root)
        manager = TrackingManager(app_config, runtime=runtime)
        processors = {
            object_id: ObjectPoseProcessor(
                enable_z_axis_stabilization=args.z_axis_stabilization,
                enable_task_symmetry_output=args.task_symmetry_output,
                enable_task_symmetry_diagnostic=args.task_symmetry_debug,
            )
            for object_id in (PIPE1_ID, PIPE2_ID)
        }

        for object_id in (PIPE1_ID, PIPE2_ID):
            tracker = manager.trackers[object_id]
            print(f"{_DISPLAY_NAMES[object_id]} CAD: {tracker.cad.model_path}")
            print(
                f"{_DISPLAY_NAMES[object_id]} CAD oriented extents: "
                f"{np.array2string(tracker.extents, precision=6)} m"
            )
        print(f"Mesh scale: {args.mesh_scale_to_meter} m/unit")

        pose_publisher = None
        try:
            camera.start()
            if args.camera_type == "cam_ros_zed2i":
                pose_publisher = RosPipePosePublisher(
                    camera.ros_node,
                    pose_topic=args.left_pipe_pose_topic,
                    status_topic=args.left_pipe_status_topic,
                )
                if args.left_pipe_object_id is None:
                    # pipe1/pipe2 are registration identities, not robot-hand
                    # identities.  Advertise INVALID but never guess a mapping.
                    pose_publisher.publish_status("INVALID")
                    print(
                        "Left Pipe pose publication is disabled: set "
                        "--left-pipe-object-id after verifying which initial "
                        "mask belongs to the left-hand grasped Pipe."
                    )
            print(f"Camera: {camera.device_name} ({camera.device_serial})")
            print(f"Color stream: {camera.color_stream_info}")
            print(f"Depth stream: {camera.depth_stream_info}")
            print(f"Depth scale: {camera.depth_scale} m/unit")

            if args.warmup_seconds:
                time.sleep(args.warmup_seconds)
            if wait_for_supervisor_activation():
                print("Task activation received; starting YOLO and registration.")
            if pose_publisher is not None and args.left_pipe_object_id is not None:
                pose_publisher.publish_status("REGISTERING")
            camera.raise_if_failed()
            frozen_frame = camera.get_next_frame()
            print(f"Frozen frame for both masks: {frozen_frame.source_frame_id}")
            if args.segmentation_mode == "manual":
                print("Select the Pipe1 mask, then select the Pipe2 mask.")
            else:
                print("Running YOLO initial instance segmentation.")

            masks = segmenter.segment(frozen_frame)

            if args.show_auto_mask:
                mask_vis = cv2.cvtColor(frozen_frame.rgb, cv2.COLOR_RGB2BGR)
                overlay = mask_vis.copy()

                # BGR: pipe1=yellow, pipe2=magenta
                overlay[masks[PIPE1_ID]] = (0, 255, 255)
                overlay[masks[PIPE2_ID]] = (255, 0, 255)

                mask_vis = cv2.addWeighted(mask_vis, 0.6, overlay, 0.4, 0)

                mask_window_name = "YOLO initial masks - press any key"
                cv2.imshow(mask_window_name, mask_vis)
                cv2.imwrite("yolo_initial_masks.png", mask_vis)
                cv2.waitKey(0)
                cv2.destroyWindow(mask_window_name)

            for processor in processors.values():
                processor.reset()

            registration_results = _results_by_object(
                manager.register_all(frozen_frame, masks)
            )
            for object_id in (PIPE1_ID, PIPE2_ID):
                _print_result(registration_results[object_id])
            failed_registration = [
                object_id
                for object_id, result in registration_results.items()
                if not result.valid
            ]
            if failed_registration:
                raise RuntimeError(
                    "Dual-object registration failed for: "
                    + ", ".join(failed_registration)
                )
            registration_processed_poses: Dict[str, np.ndarray] = {}
            for object_id in (PIPE1_ID, PIPE2_ID):
                registration_result = registration_results[object_id]
                registration_processed_poses[object_id] = processors[
                    object_id
                ].process(
                    registration_result.pose,
                    registration_result.source_frame_id,
                ).output_pose

            if pose_publisher is not None and args.left_pipe_object_id is not None:
                # output_pose is final C_T_P after the configured application
                # post-processing.  The OBB-centered overlay transform is not
                # exposed to control.
                pose_publisher.publish_pose(
                    registration_processed_poses[args.left_pipe_object_id],
                    frozen_frame,
                )

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            while True:
                camera.raise_if_failed()
                frame = camera.get_next_frame()

                # Exactly one camera acquisition is shared by both trackers.
                results = _results_by_object(manager.track_all(frame))
                processed_poses: Dict[str, np.ndarray] = {}
                for object_id in (PIPE1_ID, PIPE2_ID):
                    result = results[object_id]
                    _print_result(result)
                    if result.valid and result.pose is not None:
                        processed_poses[object_id] = processors[object_id].process(
                            result.pose,
                            result.source_frame_id,
                        ).output_pose

                if pose_publisher is not None and args.left_pipe_object_id is not None:
                    left_result = results[args.left_pipe_object_id]
                    if left_result.valid:
                        pose_publisher.publish_pose(
                            processed_poses[args.left_pipe_object_id],
                            frame,
                        )
                    else:
                        # No old pose is resent with this frame's timestamp.
                        pose_publisher.publish_status("LOST")

                image = _draw_results(
                    frame_rgb=frame.rgb,
                    K=frame.K,
                    results=results,
                    processed_poses=processed_poses,
                    manager=manager,
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
