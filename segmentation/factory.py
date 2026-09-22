"""설정된 초기 registration segmenter를 생성한다."""

from typing import Sequence

from config import SegmentationConfig
from .base import Segmenter


def create_segmenter(
    config: SegmentationConfig,
    object_ids: Sequence[str],
) -> Segmenter:
    """선택 사항인 YOLO 의존성을 미리 import하지 않고 segmenter를 만든다."""

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
