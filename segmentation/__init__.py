"""Manual and future automatic segmentation interfaces."""

from .base import Segmenter
from .manual import ManualPolygonSegmenter, SegmentationCancelled

__all__ = ["ManualPolygonSegmenter", "SegmentationCancelled", "Segmenter"]
