"""Interactive polygon-mask selection on a frozen RGB frame."""

from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from camera.base import FrameData
from segmentation.base import Segmenter


Point = Tuple[int, int]


class SegmentationCancelled(RuntimeError):
    """Raised when the user cancels manual mask selection."""


class ManualPolygonSegmenter(Segmenter):
    """Collect one polygon mask for each configured object ID."""

    def __init__(self, object_ids: Sequence[str]) -> None:
        if not object_ids:
            raise ValueError("At least one object ID is required.")
        if any(not object_id for object_id in object_ids):
            raise ValueError("Object IDs must be non-empty.")
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("Object IDs must be unique.")
        self.object_ids = tuple(object_ids)

    @staticmethod
    def _preview(
        image_bgr: np.ndarray,
        points: Sequence[Point],
        object_id: str,
    ) -> np.ndarray:
        preview = image_bgr.copy()
        if len(points) >= 3:
            polygon = np.asarray(points, dtype=np.int32)
            fill = preview.copy()
            cv2.fillPoly(fill, [polygon], (0, 160, 0))
            preview = cv2.addWeighted(preview, 0.7, fill, 0.3, 0.0)

        for point in points:
            cv2.circle(preview, point, 4, (0, 0, 255), -1)
        if len(points) >= 2:
            polygon = np.asarray(points, dtype=np.int32)
            cv2.polylines(
                preview,
                [polygon],
                len(points) >= 3,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            preview,
            f"{object_id}: click polygon around object",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            "Enter: confirm  Z: undo  R: reset  Q/Esc: cancel",
            (15, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return preview

    @staticmethod
    def _mask_from_points(
        height: int,
        width: int,
        points: Sequence[Point],
    ) -> np.ndarray:
        if len(points) < 3:
            raise ValueError("A polygon requires at least three points.")
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.asarray(points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 1)
        return np.ascontiguousarray(mask.astype(np.bool_))

    def _select_one(self, frame: FrameData, object_id: str) -> np.ndarray:
        image_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        points: List[Point] = []
        window_name = f"Manual mask - {object_id}"

        def mouse_callback(
            event: int,
            x: int,
            y: int,
            flags: int,
            parameter: object,
        ) -> None:
            del flags, parameter
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        try:
            while True:
                cv2.imshow(
                    window_name,
                    self._preview(image_bgr, points, object_id),
                )
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    raise SegmentationCancelled(
                        f"Mask selection cancelled for {object_id}."
                    )
                if key == ord("z") and points:
                    points.pop()
                elif key == ord("r"):
                    points.clear()
                elif key in (10, 13):
                    if len(points) >= 3:
                        break
                    print("A polygon requires at least three points.")
        finally:
            cv2.destroyWindow(window_name)

        return self._mask_from_points(frame.height, frame.width, points)

    def segment(self, frame: FrameData) -> Mapping[str, np.ndarray]:
        masks: Dict[str, np.ndarray] = {}
        for object_id in self.object_ids:
            masks[object_id] = self._select_one(frame, object_id)
        return masks
