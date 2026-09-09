"""Offline RGB-D sequence source for deterministic regression tests."""

import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .base import CameraSource, FrameData


PathLike = Union[str, Path]


class SequenceFrameSource(CameraSource):
    """Read aligned RGB-D pairs from configurable sequence paths.

    The defaults support FoundationPose's ``sequence01/rgb``, ``depth``, and
    ``K.txt`` layout. D405 Study recordings are supported by passing its data
    directory as ``sequence_root``, a sequence-specific ``rgb_glob`` such as
    ``"refine01_*.png"``, and ``K_path="intrinsic/refine01_K.txt"``.

    Integer depth images are raw sensor values and require ``depth_scale`` in
    meters per unit. Floating-point depth images are assumed to already be in
    meters. This source has no dependency on Pipe CAD or FoundationPose.
    """

    def __init__(
        self,
        sequence_root: PathLike,
        depth_scale: Optional[float],
        rgb_dir: PathLike = "rgb",
        depth_dir: PathLike = "depth",
        K_path: PathLike = "K.txt",
        rgb_glob: str = "*.png",
        metadata_csv_path: Optional[PathLike] = None,
        timestamp_domain: Optional[str] = None,
    ) -> None:
        self.sequence_root = Path(sequence_root).expanduser()
        self.rgb_dir = self._resolve(rgb_dir)
        self.depth_dir = self._resolve(depth_dir)
        self.K_path = self._resolve(K_path)
        self.rgb_glob = rgb_glob
        self.metadata_csv_path = (
            self._resolve(metadata_csv_path)
            if metadata_csv_path is not None
            else None
        )
        self.depth_scale = depth_scale
        self.timestamp_domain = timestamp_domain

        if depth_scale is not None and (
            not np.isfinite(depth_scale) or depth_scale <= 0
        ):
            raise ValueError("depth_scale must be finite and positive when provided.")
        if not rgb_glob:
            raise ValueError("rgb_glob must not be empty.")

        self._started = False
        self._index = 0
        self._K: Optional[np.ndarray] = None
        self._frames: List[Tuple[int, Path, Path]] = []
        self._metadata_by_rgb_file: Dict[str, Dict[str, str]] = {}
        self._metadata_by_frame_id: Dict[int, Dict[str, str]] = {}

    def _resolve(self, path: PathLike) -> Path:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else self.sequence_root / candidate

    @staticmethod
    def _parse_frame_id(rgb_path: Path) -> int:
        match = re.search(r"(\d+)$", rgb_path.stem)
        if match is None:
            raise ValueError(
                "Cannot extract a source frame ID from RGB filename: "
                f"{rgb_path.name}. A trailing integer is required."
            )
        return int(match.group(1))

    def _load_metadata(self) -> None:
        self._metadata_by_rgb_file.clear()
        self._metadata_by_frame_id.clear()

        if self.metadata_csv_path is None:
            return
        if not self.metadata_csv_path.is_file():
            raise FileNotFoundError(
                f"Sequence metadata CSV not found: {self.metadata_csv_path}"
            )

        with self.metadata_csv_path.open("r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(
                    f"Metadata CSV has no header: {self.metadata_csv_path}"
                )

            for row in reader:
                rgb_file = (row.get("rgb_file") or "").strip()
                if rgb_file:
                    self._metadata_by_rgb_file[rgb_file] = row

                frame_id_text = (row.get("frame_id") or "").strip()
                if frame_id_text:
                    try:
                        self._metadata_by_frame_id[int(frame_id_text)] = row
                    except ValueError as error:
                        raise ValueError(
                            "Invalid frame_id in metadata CSV "
                            f"{self.metadata_csv_path}: {frame_id_text!r}"
                        ) from error

    def start(self) -> None:
        self._started = False
        if not self.sequence_root.is_dir():
            raise FileNotFoundError(
                f"Sequence root directory not found: {self.sequence_root}"
            )
        if not self.rgb_dir.is_dir():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        if not self.depth_dir.is_dir():
            raise FileNotFoundError(f"Depth directory not found: {self.depth_dir}")
        if not self.K_path.is_file():
            raise FileNotFoundError(f"Camera intrinsic file not found: {self.K_path}")

        rgb_files = sorted(self.rgb_dir.glob(self.rgb_glob))
        if not rgb_files:
            raise FileNotFoundError(
                f"No RGB frames match {self.rgb_glob!r} in {self.rgb_dir}"
            )

        frames: List[Tuple[int, Path, Path]] = []
        seen_ids = set()
        missing_depth: List[Path] = []
        for rgb_path in rgb_files:
            frame_id = self._parse_frame_id(rgb_path)
            if frame_id in seen_ids:
                raise ValueError(
                    f"Duplicate source frame ID {frame_id} in {self.rgb_dir}."
                )
            seen_ids.add(frame_id)

            depth_path = self.depth_dir / rgb_path.name
            if not depth_path.is_file():
                missing_depth.append(depth_path)
            frames.append((frame_id, rgb_path, depth_path))

        frames.sort(key=lambda frame: frame[0])

        if missing_depth:
            examples = ", ".join(str(path) for path in missing_depth[:3])
            raise FileNotFoundError(
                f"Missing {len(missing_depth)} depth frame pair(s). "
                f"Examples: {examples}"
            )

        K = np.loadtxt(str(self.K_path), dtype=np.float64)
        if K.size != 9:
            raise ValueError(
                f"Camera intrinsic file must contain 9 values: {self.K_path}"
            )
        K = K.reshape(3, 3)
        if not np.all(np.isfinite(K)):
            raise ValueError(
                f"Camera intrinsic contains non-finite values: {self.K_path}"
            )

        self._load_metadata()
        self._frames = frames
        self._K = K
        self._index = 0
        self._started = True

    def stop(self) -> None:
        self._started = False

    def __len__(self) -> int:
        return len(self._frames)

    def _read_rgb(self, path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to load RGB frame: {path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _read_depth_m(self, path: Path) -> np.ndarray:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Failed to load depth frame: {path}")
        if depth.ndim != 2:
            raise ValueError(
                f"Depth frame must be single-channel; received {depth.shape} at {path}"
            )

        if np.issubdtype(depth.dtype, np.integer):
            if self.depth_scale is None:
                raise ValueError(
                    "Raw integer depth requires depth_scale in meters per unit: "
                    f"{path}"
                )
            depth_m = depth.astype(np.float32) * np.float32(self.depth_scale)
        elif np.issubdtype(depth.dtype, np.floating):
            depth_m = depth.astype(np.float32, copy=False)
        else:
            raise TypeError(f"Unsupported depth dtype {depth.dtype} at {path}")

        # Preserve the threshold used by the verified D405/FoundationPose
        # scripts: sub-millimeter values are treated as invalid depth.
        depth_m = np.array(depth_m, dtype=np.float32, copy=True, order="C")
        depth_m[depth_m < 0.001] = 0.0
        return depth_m

    def _device_timestamp(self, rgb_path: Path, frame_id: int) -> Optional[float]:
        metadata = self._metadata_by_rgb_file.get(rgb_path.name)
        if metadata is None:
            metadata = self._metadata_by_frame_id.get(frame_id)
        if metadata is None:
            return None

        value = (metadata.get("device_timestamp_ms") or "").strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError as error:
            raise ValueError(
                f"Invalid device_timestamp_ms for {rgb_path.name}: {value!r}"
            ) from error

    def get_next_frame(self) -> FrameData:
        if not self._started:
            raise RuntimeError("SequenceFrameSource.start() must be called first.")
        if self._index >= len(self._frames):
            raise StopIteration
        if self._K is None:
            raise RuntimeError("SequenceFrameSource is missing initialized K data.")

        frame_id, rgb_path, depth_path = self._frames[self._index]
        rgb = self._read_rgb(rgb_path)
        depth_m = self._read_depth_m(depth_path)
        host_wall_time_s = time.time()
        host_monotonic_time_s = time.monotonic()

        frame = FrameData(
            source_frame_id=frame_id,
            rgb=rgb,
            depth_m=depth_m,
            K=self._K,
            device_timestamp_ms=self._device_timestamp(rgb_path, frame_id),
            timestamp_domain=self.timestamp_domain,
            host_wall_time_s=host_wall_time_s,
            host_monotonic_time_s=host_monotonic_time_s,
        )
        self._index += 1
        return frame
