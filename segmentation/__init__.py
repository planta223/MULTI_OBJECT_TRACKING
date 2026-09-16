"""Manual and future automatic segmentation interfaces."""

from .base import Segmenter
from .factory import create_segmenter
from .manual import ManualPolygonSegmenter, SegmentationCancelled

__all__ = [
    "ManualPolygonSegmenter",
    "SegmentationCancelled",
    "Segmenter",
    "create_segmenter",
]
