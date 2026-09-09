"""Single-object FoundationPose smoke test using the recorded T-LESS #6 data.

T-LESS #6 is used only to verify the real integration path:

    SequenceFrameSource -> ObjectTracker -> FoundationPose register/track

It is not a substitute for either Pipe CAD and this script performs no
dual-object or accuracy evaluation.
"""

import argparse
from pathlib import Path
import sys
import tempfile
from typing import Optional, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FoundationPoseConfig, ObjectConfig
from pose_estimation.foundationpose_runtime import FoundationPoseRuntime
from tracking.object_tracker import ObjectTracker
from pose_estimation.pose_result import PoseResult
from camera.sequence import SequenceFrameSource


TLESS_SMOKE_OBJECT_ID = "tless06_smoke"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    foundationpose_root = PROJECT_ROOT / "FoundationPose"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=foundationpose_root,
    )
    parser.add_argument(
        "--tless-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.0001,
        help="Recorded D405 raw-depth scale in meters per unit.",
    )
    parser.add_argument(
        "--track-frames",
        type=int,
        default=3,
        help="Number of frames to track after frame 0 registration.",
    )
    parser.add_argument("--register-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args(argv)

    if args.track_frames < 1:
        parser.error("--track-frames must be at least 1.")
    if args.register_refine_iter < 1 or args.track_refine_iter < 1:
        parser.error("FoundationPose refinement iterations must be positive.")
    if args.debug < 0:
        parser.error("--debug cannot be negative.")
    return args


def load_initial_mask(path: Path) -> np.ndarray:
    mask_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise FileNotFoundError(f"Failed to load initial mask: {path}")
    return mask_image > 0


def print_result(result: PoseResult) -> None:
    pose_shape = None if result.pose is None else result.pose.shape
    translation = result.translation_m
    translation_text = (
        "None"
        if translation is None
        else np.array2string(translation, precision=6, suppress_small=False)
    )
    print(
        "frame={frame} mode={mode} valid={valid} pose_shape={shape} "
        "translation_m={translation} processing_time_s={elapsed:.6f}".format(
            frame=result.source_frame_id,
            mode=result.mode.value,
            valid=result.valid,
            shape=pose_shape,
            translation=translation_text,
            elapsed=result.processing_time_s,
        )
    )
    if result.message:
        print(f"  message={result.message}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    sequence_root = args.tless_root / "sequence01"
    model_path = args.tless_root / "obj_06.ply"
    mask_path = sequence_root / "mask_000000.png"

    print("T-LESS #6 integration smoke test")
    print(f"FoundationPose root : {args.foundationpose_root}")
    print(f"Sequence root       : {sequence_root}")
    print(f"CAD                 : {model_path}")
    print(f"Initial mask        : {mask_path}")
    print(f"Depth scale         : {args.depth_scale} m/unit")
    print(f"Tracking frames     : {args.track_frames}")

    source = SequenceFrameSource(
        sequence_root=sequence_root,
        depth_scale=args.depth_scale,
    )
    source.start()
    try:
        initial_frame = source.get_next_frame()
        initial_mask = load_initial_mask(mask_path)

        runtime = FoundationPoseRuntime(args.foundationpose_root)
        foundationpose_config = FoundationPoseConfig(
            root=args.foundationpose_root.expanduser().resolve(),
            register_refine_iter=args.register_refine_iter,
            track_refine_iter=args.track_refine_iter,
            debug=args.debug,
        )
        object_config = ObjectConfig(
            object_id=TLESS_SMOKE_OBJECT_ID,
            model_path=model_path,
            mesh_scale_to_meter=0.001,
        )

        with tempfile.TemporaryDirectory(
            prefix="pipe_tracking_tless_smoke_"
        ) as debug_dir:
            tracker = ObjectTracker(
                object_config=object_config,
                foundationpose_config=foundationpose_config,
                runtime=runtime,
                debug_dir=Path(debug_dir),
            )

            register_result = tracker.register(
                frame=initial_frame,
                mask=initial_mask,
                cycle_id=0,
            )
            print_result(register_result)
            if not register_result.valid:
                print("Smoke test failed during register().", file=sys.stderr)
                return 1

            for cycle_id in range(1, args.track_frames + 1):
                frame = source.get_next_frame()
                track_result = tracker.track(frame=frame, cycle_id=cycle_id)
                print_result(track_result)
                if not track_result.valid:
                    print(
                        f"Smoke test failed during track() at cycle {cycle_id}.",
                        file=sys.stderr,
                    )
                    return 1
    finally:
        source.stop()

    print("T-LESS #6 integration smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
