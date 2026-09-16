"""Generate initial FoundationPose masks with Ultralytics YOLO segmentation."""

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import cv2
import numpy as np

from camera.base import FrameData
from segmentation.base import Segmenter


PathLike = Union[str, Path]


def _as_numpy(value: Any) -> np.ndarray:
    """Convert an Ultralytics NumPy/Torch result field to a NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class YoloSegmenter(Segmenter):
    """Assign initial instance masks to object IDs from left to right."""

    def __init__(
        self,
        object_ids: Sequence[str],
        model_path: Optional[PathLike],
        confidence: float = 0.5,
        device: Optional[str] = None,
        class_id: Optional[int] = None,
        *,
        model: Optional[Any] = None,
    ) -> None:
        if not object_ids:
            raise ValueError("At least one object ID is required.")
        if any(not object_id for object_id in object_ids):
            raise ValueError("Object IDs must be non-empty.")
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("Object IDs must be unique.")
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("YOLO confidence must be in [0, 1].")
        if class_id is not None and class_id < 0:
            raise ValueError("YOLO class_id must be non-negative.")

        self.object_ids = tuple(object_ids)
        self.confidence = float(confidence)
        self.device = device
        self.class_id = class_id
        self.model_path = None if model_path is None else Path(model_path).expanduser()
        self._model = model

        if self._model is None:
            if self.model_path is None:
                raise ValueError("YOLO segmentation requires a model path.")
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"YOLO segmentation model not found: {self.model_path}"
                )
            self._model = self._load_model(self.model_path)

    @staticmethod
    def _load_model(model_path: Path) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "Ultralytics is required for YOLO segmentation. Install it "
                "with 'python3 -m pip install -r requirements-yolo.txt'."
            ) from error
        try:
            return YOLO(str(model_path))
        except Exception as error:
            raise RuntimeError(
                f"Failed to load YOLO segmentation model: {model_path}"
            ) from error

    @staticmethod
    def _normalize_mask(mask: Any, frame: FrameData) -> np.ndarray:
        values = _as_numpy(mask).squeeze()
        if values.ndim != 2 or values.size == 0:
            raise ValueError(
                "YOLO returned an invalid instance mask shape: "
                f"{values.shape}."
            )
        if values.shape != (frame.height, frame.width):
            values = cv2.resize(
                values.astype(np.float32),
                (frame.width, frame.height),
                interpolation=cv2.INTER_NEAREST,
            )
        normalized = np.ascontiguousarray(values > 0.5, dtype=np.bool_)
        if normalized.shape != (frame.height, frame.width):
            raise ValueError(
                "YOLO mask resolution does not match the camera frame: "
                f"expected {(frame.height, frame.width)}, got {normalized.shape}."
            )
        if not np.any(normalized):
            raise ValueError("YOLO returned an empty instance mask.")
        return normalized

    def segment(self, frame: FrameData) -> Mapping[str, np.ndarray]:
        # Ultralytics documents HWC uint8 NumPy sources as BGR. FrameData is RGB.
        image_bgr = np.ascontiguousarray(frame.rgb[..., ::-1])
        predict_args = {
            "source": image_bgr,
            "conf": self.confidence,
            "retina_masks": True,
            "verbose": False,
        }
        if self.device is not None:
            predict_args["device"] = self.device

        try:
            results = self._model.predict(**predict_args)
        except Exception as error:
            raise RuntimeError(f"YOLO segmentation inference failed: {error}") from error
        if len(results) != 1:
            raise RuntimeError(
                "YOLO must return exactly one result for one registration frame; "
                f"received {len(results)}."
            )

        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None:
            raise RuntimeError("YOLO segmentation result contains no boxes.")

        confidences = _as_numpy(boxes.conf).reshape(-1)
        detection_count = len(confidences)
        if detection_count == 0:
            raise RuntimeError("YOLO found no detections in the registration frame.")
        if masks is None:
            raise RuntimeError(
                "YOLO model returned no instance masks. Use an instance "
                "segmentation model (.pt), not a detection-only model."
            )
        class_ids = _as_numpy(boxes.cls).reshape(-1)
        xyxy = _as_numpy(boxes.xyxy)
        mask_data = _as_numpy(masks.data)
        if (
            class_ids.shape != (detection_count,)
            or xyxy.shape != (detection_count, 4)
            or mask_data.ndim != 3
            or mask_data.shape[0] != detection_count
        ):
            raise RuntimeError(
                "YOLO boxes, classes, confidences, and masks have inconsistent "
                "detection counts."
            )

        candidates = []
        for index in range(detection_count):
            confidence = float(confidences[index])
            detected_class = int(class_ids[index])
            if confidence < self.confidence:
                continue
            if self.class_id is not None and detected_class != self.class_id:
                continue
            box = np.asarray(xyxy[index], dtype=np.float64)
            if not np.all(np.isfinite(box)):
                raise ValueError("YOLO returned non-finite bounding-box coordinates.")
            candidates.append(
                {
                    "confidence": confidence,
                    "center_x": float((box[0] + box[2]) / 2.0),
                    "mask": mask_data[index],
                }
            )

        needed = len(self.object_ids)
        if len(candidates) < needed:
            class_text = (
                " matching the configured class"
                if self.class_id is not None
                else ""
            )
            raise RuntimeError(
                f"YOLO found {len(candidates)} valid pipe instance(s){class_text}, "
                f"but {needed} object(s) are configured."
            )

        selected = sorted(
            candidates,
            key=lambda candidate: candidate["confidence"],
            reverse=True,
        )[:needed]
        selected.sort(key=lambda candidate: candidate["center_x"])

        return {
            object_id: self._normalize_mask(candidate["mask"], frame)
            for object_id, candidate in zip(self.object_ids, selected)
        }
