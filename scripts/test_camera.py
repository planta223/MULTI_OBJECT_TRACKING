"""Short D405 acquisition regression test without pose inference or GUI."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.realsense_d405 import RealSenseD405Source
from config import CameraConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial")
    parser.add_argument("--duration", type=float, default=4.0)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def main() -> int:
    args = parse_args()
    source = RealSenseD405Source(
        CameraConfig(width=args.width, height=args.height, fps=args.fps, serial=args.serial)
    )
    observed_ids = []
    try:
        source.start()
        print(f"Device: {source.device_name} ({source.device_serial})")
        print(f"Color: {source.color_stream_info}")
        print(f"Depth: {source.depth_stream_info}")
        print(f"Depth scale: {source.depth_scale} m/unit")
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            frame = source.get_next_frame()
            valid = frame.depth_m[np.isfinite(frame.depth_m) & (frame.depth_m > 0)]
            print(
                f"frame={frame.source_frame_id} rgb={frame.rgb.shape} "
                f"depth={frame.depth_m.shape} valid_depth={valid.size}"
            )
            observed_ids.append(frame.source_frame_id)
    finally:
        source.stop()
    if len(observed_ids) < 2:
        raise RuntimeError("Too few frames were received.")
    print("D405 acquisition regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
