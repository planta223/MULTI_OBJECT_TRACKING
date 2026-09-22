"""정렬된 D405 RGB-D를 미리 보고 S 키로 snapshot을 저장한다."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealSenseD405.camera import D405Camera, D405Config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "recordings" / "captures")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial")
    return parser.parse_args()


def save_frame(root: Path, frame, depth_scale: float) -> None:
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "rgb" / f"{frame.source_frame_id:06d}.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(root / "depth" / f"{frame.source_frame_id:06d}.png"), frame.depth_raw)
    np.savetxt(root / "K.txt", frame.K, fmt="%.8f")
    (root / "depth_scale.txt").write_text(f"{depth_scale:.12g}\n")
    print(f"Saved frame {frame.source_frame_id} under {root}")


def main() -> int:
    args = parse_args()
    camera = D405Camera(D405Config(args.width, args.height, args.fps, args.serial))
    try:
        camera.start()
        while True:
            frame = camera.get_next_frame()
            color = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(frame.depth_m, alpha=100.0), cv2.COLORMAP_JET)
            cv2.imshow("D405 RGB", color)
            cv2.imshow("D405 aligned depth", depth_vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_frame(args.output_dir, frame, camera.depth_scale)
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
