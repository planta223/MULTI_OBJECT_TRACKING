"""Live dual-object regression for Pipe1 and Pipe2 with one camera frame."""

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
    CameraConfig,
    FeatureFlags,
    FoundationPoseConfig,
    ObjectConfig,
    OutputConfig,
    TrackingConfig,
)
from constants import PIPE1_ID, PIPE2_ID, SUPPORTED_IDENTITY_MODES
from pose_estimation.foundationpose_runtime import FoundationPoseRuntime
from pose_estimation.pose_result import PoseResult
from pose_processing.object_pose_processor import ObjectPoseProcessor
from segmentation.manual.polygon_segmenter import (
    ManualPolygonSegmenter,
    SegmentationCancelled,
)
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
    parser.add_argument("--camera-type", default="realsense_d405")
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--register-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--axis-length-m", type=float, default=0.05)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument(
        "--identity-mode",
        choices=SUPPORTED_IDENTITY_MODES,
        default="depth_motion",
        help="Identity protection: baseline, motion gating, or depth-aware gating.",
    )
    parser.add_argument(
        "--identity-debug",
        action="store_true",
        help="Print and overlay KF, occlusion, and measurement-gating decisions.",
    )
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
    if args.register_refine_iter <= 0 or args.track_refine_iter <= 0:
        parser.error("FoundationPose iterations must be positive.")
    if not np.isfinite(args.mesh_scale_to_meter) or args.mesh_scale_to_meter <= 0:
        parser.error("--mesh-scale-to-meter must be finite and positive.")
    if not np.isfinite(args.axis_length_m) or args.axis_length_m <= 0:
        parser.error("--axis-length-m must be finite and positive.")
    if args.debug < 0:
        parser.error("--debug must be non-negative.")
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
    identity_debug: bool,
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
    if identity_debug:
        for index, object_id in enumerate((PIPE1_ID, PIPE2_ID)):
            decision = manager.last_identity_decisions.get(object_id)
            if decision is None:
                continue
            distance = (
                "n/a"
                if decision.mahalanobis_distance_sq is None
                else f"{decision.mahalanobis_distance_sq:.2f}"
            )
            action = "ACCEPT" if decision.measurement_accepted else "PREDICT"
            cv2.putText(
                image,
                (
                    f"{object_id} {decision.state.value} "
                    f"{decision.visibility_label} {action} "
                    f"d2={distance} std={decision.prediction_std_m:.3f} "
                    f"{decision.reason}"
                ),
                (15, 108 + 22 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                _DISPLAY_COLORS[object_id],
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
    )
    foundationpose_config = FoundationPoseConfig(
        root=args.foundationpose_root.expanduser().resolve(),
        register_refine_iter=args.register_refine_iter,
        track_refine_iter=args.track_refine_iter,
        debug=args.debug,
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
            foundationpose=foundationpose_config,
            tracking=TrackingConfig(
                identity_mode=args.identity_mode,
                identity_debug=args.identity_debug,
            ),
            objects=object_configs,
            output=OutputConfig(output_dir=Path(debug_dir)),
        )
        app_config.validate(check_model_paths=True)

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

        try:
            camera.start()
            print(f"Camera: {camera.device_name} ({camera.device_serial})")
            print(f"Color stream: {camera.color_stream_info}")
            print(f"Depth stream: {camera.depth_stream_info}")
            print(f"Depth scale: {camera.depth_scale} m/unit")

            if args.warmup_seconds:
                time.sleep(args.warmup_seconds)
            camera.raise_if_failed()
            frozen_frame = camera.get_next_frame()
            print(f"Frozen frame for both masks: {frozen_frame.source_frame_id}")
            print("Select the Pipe1 mask, then select the Pipe2 mask.")

            segmenter = ManualPolygonSegmenter((PIPE1_ID, PIPE2_ID))
            masks = segmenter.segment(frozen_frame)
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
            for object_id in (PIPE1_ID, PIPE2_ID):
                registration_result = registration_results[object_id]
                processors[object_id].process(
                    registration_result.pose,
                    registration_result.source_frame_id,
                )

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            while True:
                camera.raise_if_failed()
                frame = camera.get_next_frame()

                # Exactly one camera acquisition is shared by both trackers.
                results = _results_by_object(manager.track_all(frame))
                if args.identity_debug:
                    for object_id in (PIPE1_ID, PIPE2_ID):
                        decision = manager.last_identity_decisions.get(object_id)
                        if decision is not None:
                            print(f"identity {decision.debug_text()}")
                processed_poses: Dict[str, np.ndarray] = {}
                for object_id in (PIPE1_ID, PIPE2_ID):
                    result = results[object_id]
                    _print_result(result)
                    if result.valid and result.pose is not None:
                        processed_poses[object_id] = processors[object_id].process(
                            result.pose,
                            result.source_frame_id,
                        ).output_pose

                image = _draw_results(
                    frame_rgb=frame.rgb,
                    K=frame.K,
                    results=results,
                    processed_poses=processed_poses,
                    manager=manager,
                    axis_length_m=args.axis_length_m,
                    identity_debug=args.identity_debug,
                )
                cv2.imshow(window_name, image)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
        except SegmentationCancelled as error:
            print(error)
        finally:
            camera.stop()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
