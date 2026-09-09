"""Replaceable segmentation interface."""

from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np

from camera.base import FrameData


class Segmenter(ABC):
    @abstractmethod
    def segment(self, frame: FrameData) -> Mapping[str, np.ndarray]:
        """Return object_id -> bool mask mappings of shape (H, W)."""
