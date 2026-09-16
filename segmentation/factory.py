"""Create the configured initial-registration segmenter."""

from typing import Sequence

from config import SegmentationConfig
from .base import Segmenter


def create_segmenter(
    config: SegmentationConfig,
    object_ids: Sequence[str],
) -> Segmenter:
    """Build a segmenter without importing optional YOLO dependencies early."""

    if config.mode == "manual":
        from .manual import ManualPolygonSegmenter

        return ManualPolygonSegmenter(object_ids)
    if config.mode == "yolo":
        from .yolo import YoloSegmenter

        return YoloSegmenter(
            object_ids=object_ids,
            model_path=config.model_path,
            confidence=config.confidence,
            device=config.device,
            class_id=config.class_id,
        )
    raise ValueError(f"Unsupported segmentation mode: {config.mode!r}.")
