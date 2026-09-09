"""Record an aligned D405 RGB-D sequence without pose-estimation code."""

import argparse
import csv
from pathlib import Path
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealSenseD405.camera import D405Camera, D405Config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "recordings" / "sequence")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def main() -> int:
    args = parse_args()
    rgb_dir = args.output_dir / "rgb"
    depth_dir = args.output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    camera = D405Camera(D405Config(args.width, args.height, args.fps, args.serial))
    try:
        camera.start()
        deadline = time.monotonic() + args.duration
        first_frame = None
        while time.monotonic() < deadline:
            frame = camera.get_next_frame()
            if first_frame is None:
                first_frame = frame
            name = f"{frame.source_frame_id:06d}.png"
            cv2.imwrite(str(rgb_dir / name), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(depth_dir / name), frame.depth_raw)
            rows.append((frame.source_frame_id, name, frame.device_timestamp_ms, frame.host_wall_time_s))
    finally:
        camera.stop()
    if first_frame is None:
        raise RuntimeError("No D405 frames were recorded.")
    np.savetxt(args.output_dir / "K.txt", first_frame.K, fmt="%.8f")
    (args.output_dir / "depth_scale.txt").write_text(f"{camera.depth_scale:.12g}\n")
    with (args.output_dir / "frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame_id", "rgb_file", "device_timestamp_ms", "host_wall_time_s"))
        writer.writerows(rows)
    print(f"Recorded {len(rows)} frame(s) under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
