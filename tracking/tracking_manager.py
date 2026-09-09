"""Sequential orchestration of one or two independent object trackers."""

from pathlib import Path
import time
from typing import Dict, List, Mapping, Optional, Union
import warnings

import numpy as np

from config import AppConfig
from constants import PIPE1_ID, PIPE2_ID, TrackingMode
from pose_estimation.foundationpose_runtime import FoundationPoseRuntime
from camera.base import FrameData
from .object_tracker import ObjectTracker
from pose_estimation.pose_result import PoseResult


PathLike = Union[str, Path]
_LARGE_MASK_OVERLAP_RATIO = 0.5


class TrackingManager:
    """Run configured trackers against one identical FrameData snapshot."""

    def __init__(
        self,
        config: AppConfig,
        foundationpose_root: Optional[PathLike] = None,
        runtime: Optional[FoundationPoseRuntime] = None,
    ) -> None:
        config.validate(check_model_paths=True)
        self.config = config
        configured_root = (
            foundationpose_root
            if foundationpose_root is not None
            else config.foundationpose.root
        )
        self.runtime = runtime or FoundationPoseRuntime(configured_root)
        self._next_cycle_id = 0

        object_configs = list(config.objects)
        by_id = {
            object_config.object_id: object_config
            for object_config in object_configs
        }
        if PIPE1_ID in by_id and PIPE2_ID in by_id:
            object_configs = [by_id[PIPE1_ID], by_id[PIPE2_ID]]

        debug_root = Path(config.output.output_dir) / "foundationpose_debug"
        self.trackers: Dict[str, ObjectTracker] = {}
        for object_config in object_configs:
            self.trackers[object_config.object_id] = ObjectTracker(
                object_config=object_config,
                foundationpose_config=config.foundationpose,
                runtime=self.runtime,
                debug_dir=debug_root / object_config.object_id,
            )

    def _take_cycle_id(self) -> int:
        cycle_id = self._next_cycle_id
        self._next_cycle_id += 1
        return cycle_id

    def _ordered_trackers(self) -> List[ObjectTracker]:
        return list(self.trackers.values())

    def _validate_masks(
        self,
        frame: FrameData,
        masks: Mapping[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if not isinstance(masks, Mapping):
            raise TypeError("masks must map object IDs to NumPy arrays.")

        missing = [object_id for object_id in self.trackers if object_id not in masks]
        if missing:
            raise KeyError(f"Missing registration mask(s): {', '.join(missing)}")

        normalized = {
            tracker.object_id: tracker.validate_registration_mask(
                frame, masks[tracker.object_id]
            )
            for tracker in self._ordered_trackers()
        }

        ordered_trackers = self._ordered_trackers()
        if len(ordered_trackers) == 2:
            first, second = ordered_trackers
            first_mask = normalized[first.object_id]
            second_mask = normalized[second.object_id]
            intersection = int(np.count_nonzero(first_mask & second_mask))
            smaller_area = min(
                int(np.count_nonzero(first_mask)),
                int(np.count_nonzero(second_mask)),
            )
            overlap_ratio = intersection / float(smaller_area)
            if overlap_ratio >= _LARGE_MASK_OVERLAP_RATIO:
                warnings.warn(
                    "Registration masks overlap over {:.1%} of the smaller "
                    "mask; verify object mask assignment.".format(
                        overlap_ratio
                    ),
                    RuntimeWarning,
                )
        return normalized

    @staticmethod
    def _precondition_failure(
        tracker: ObjectTracker,
        frame: FrameData,
        cycle_id: int,
        mode: TrackingMode,
        error: Exception,
        processing_time_s: float,
    ) -> PoseResult:
        result = PoseResult(
            cycle_id=cycle_id,
            object_id=tracker.object_id,
            source_frame_id=frame.source_frame_id,
            host_wall_time_s=frame.host_wall_time_s,
            host_monotonic_time_s=frame.host_monotonic_time_s,
            mode=mode,
            state=tracker.state,
            valid=False,
            pose=None,
            processing_time_s=processing_time_s,
            message=f"{type(error).__name__}: {error}",
        )
        tracker.last_result = result
        return result

    def register_all(
        self,
        frame: FrameData,
        masks: Mapping[str, np.ndarray],
    ) -> List[PoseResult]:
        """Register configured objects on the same frozen frame."""

        normalized_masks = self._validate_masks(frame, masks)
        cycle_id = self._take_cycle_id()
        results: List[PoseResult] = []
        for tracker in self._ordered_trackers():
            start_time = time.perf_counter()
            try:
                result = tracker.register(
                    frame=frame,
                    mask=normalized_masks[tracker.object_id],
                    cycle_id=cycle_id,
                )
            except Exception as error:
                result = self._precondition_failure(
                    tracker,
                    frame,
                    cycle_id,
                    TrackingMode.REGISTER,
                    error,
                    time.perf_counter() - start_time,
                )
            results.append(result)
        return results

    def track_all(self, frame: FrameData) -> List[PoseResult]:
        """Track configured objects on one identical frame snapshot."""

        cycle_id = self._take_cycle_id()
        results: List[PoseResult] = []
        for tracker in self._ordered_trackers():
            start_time = time.perf_counter()
            try:
                result = tracker.track(frame=frame, cycle_id=cycle_id)
            except Exception as error:
                result = self._precondition_failure(
                    tracker,
                    frame,
                    cycle_id,
                    TrackingMode.TRACK,
                    error,
                    time.perf_counter() - start_time,
                )
            results.append(result)
        return results
