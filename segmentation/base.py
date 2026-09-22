"""교체 가능한 segmentation 인터페이스."""

from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np

from camera.base import FrameData


class Segmenter(ABC):
    @abstractmethod
    def segment(self, frame: FrameData) -> Mapping[str, np.ndarray]:
        """(H, W) 형태의 object_id -> bool mask mapping을 반환한다."""
