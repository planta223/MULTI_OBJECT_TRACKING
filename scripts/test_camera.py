"""Camera-only RGB-D smoke test without pose inference."""

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera import create_camera_source
from config import CameraConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-type", default="realsense_d405")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial")
    parser.add_argument("--zed-resolution", default="HD720")
    parser.add_argument("--zed-depth-mode", default="NEURAL")
    parser.add_argument("--ros-color-topic", default="/cam/color/compressed")
    parser.add_argument("--ros-depth-topic", default="/cam/depth/compressed")
    parser.add_argument(
        "--ros-camera-info-topic", default="/cam/color/camera_info"
    )
    parser.add_argument("--ros-frame-timeout-sec", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.ros_frame_timeout_sec <= 0:
        parser.error("--ros-frame-timeout-sec must be positive")
    return args


def main() -> int:
    args = parse_args()
    source = create_camera_source(
        CameraConfig(
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
    )
    observed_ids = []
    started_at = time.monotonic()
    try:
        source.start()
        print(f"Device: {source.device_name} ({source.device_serial})")
        print(f"Color: {source.color_stream_info}")
        print(f"Depth: {source.depth_stream_info}")
        print(f"Depth scale: {source.depth_scale} m/unit")
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            source.raise_if_failed()
            frame = source.get_next_frame()
            valid = frame.depth_m[np.isfinite(frame.depth_m) & (frame.depth_m > 0)]
            valid_range = (
                "none"
                if valid.size == 0
                else f"{float(valid.min()):.4f}..{float(valid.max()):.4f} m"
            )
            print(
                f"frame={frame.source_frame_id} rgb={frame.rgb.shape} "
                f"rgb_dtype={frame.rgb.dtype} depth={frame.depth_m.shape} "
                f"depth_dtype={frame.depth_m.dtype} valid_depth={valid.size} "
                f"range={valid_range} timestamp_ms={frame.device_timestamp_ms} "
                f"domain={frame.timestamp_domain}"
            )
            if not observed_ids:
                print(f"K:\n{frame.K}")
            observed_ids.append(frame.source_frame_id)
            if args.preview:
                color_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                depth_vis = np.zeros(frame.depth_m.shape, dtype=np.uint8)
                if valid.size:
                    maximum = float(np.percentile(valid, 95))
                    if maximum > 0:
                        depth_vis = np.clip(
                            frame.depth_m * (255.0 / maximum), 0, 255
                        ).astype(np.uint8)
                cv2.imshow("Camera RGB", color_bgr)
                cv2.imshow("Camera depth", depth_vis)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        source.stop()
        cv2.destroyAllWindows()
    if len(observed_ids) < 2:
        raise RuntimeError("Too few frames were received.")
    elapsed = max(time.monotonic() - started_at, 1e-9)
    print(f"Observed FPS: {len(observed_ids) / elapsed:.2f}")
    print(f"{args.camera_type} acquisition regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
