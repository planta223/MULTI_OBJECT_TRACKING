"""Sequential orchestration of one or two independent object trackers."""

from pathlib import Path
import time
from typing import Dict, List, Mapping, Optional, Union
import warnings

import numpy as np

from config import AppConfig
from constants import PIPE1_ID, PIPE2_ID, TrackingMode, TrackingState
from pose_estimation.foundationpose_runtime import FoundationPoseRuntime
from camera.base import FrameData
from .object_tracker import ObjectTracker
from pose_estimation.pose_result import PoseResult
from .identity_tracker import IdentityDecision, MultiObjectIdentityTracker
from .occlusion_manager import OcclusionGeometry


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
        self.identity_tracker: Optional[MultiObjectIdentityTracker] = None
        self.last_identity_decisions: Dict[str, IdentityDecision] = {}
        if config.object_count > 1 and config.tracking.identity_mode != "none":
            geometries = {
                object_id: OcclusionGeometry(
                    bbox=tracker.bbox,
                    to_origin=tracker.to_origin,
                )
                for object_id, tracker in self.trackers.items()
            }
            self.identity_tracker = MultiObjectIdentityTracker(
                object_ids=self.trackers,
                geometries=geometries,
                config=config.tracking,
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
        if self.identity_tracker is not None:
            accepted_poses = {
                result.object_id: result.pose
                for result in results
                if result.valid and result.pose is not None
            }
            if len(accepted_poses) == len(self.trackers):
                self.identity_tracker.initialize(
                    accepted_poses,
                    frame.host_monotonic_time_s,
                )
        self.last_identity_decisions = {}
        return results

    def track_all(self, frame: FrameData) -> List[PoseResult]:
        """Track configured objects on one identical frame snapshot."""

        if self.identity_tracker is not None:
            return self._track_all_with_identity(frame)

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
        self.last_identity_decisions = {}
        return results

    def _track_all_with_identity(self, frame: FrameData) -> List[PoseResult]:
        """Run predict/occlusion/gating around estimator measurements."""

        if self.identity_tracker is None:
            raise RuntimeError("Identity tracker is not configured.")
        cycle_id = self._take_cycle_id()
        identity_cycle = self.identity_tracker.begin_cycle(frame)
        snapshots = {}
        candidate_results: Dict[str, PoseResult] = {}
        measured_poses = {}
        measurement_errors = {}

        for tracker in self._ordered_trackers():
            object_id = tracker.object_id
            occlusion = identity_cycle.occlusions[object_id]
            if occlusion.occluded:
                measured_poses[object_id] = None
                measurement_errors[object_id] = (
                    f"occluded_by_{occlusion.occluding_object_id}"
                )
                continue

            start_time = time.perf_counter()
            try:
                snapshots[object_id] = tracker.backup_tracking_state()
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
            candidate_results[object_id] = result
            measured_poses[object_id] = result.pose if result.valid else None
            if not result.valid:
                measurement_errors[object_id] = result.message or "estimator_failed"

        try:
            decisions = self.identity_tracker.evaluate(
                identity_cycle,
                measured_poses,
                measurement_errors,
            )
        except Exception:
            for object_id, snapshot in snapshots.items():
                self.trackers[object_id].restore_tracking_state(snapshot)
            raise

        results = []
        for tracker in self._ordered_trackers():
            object_id = tracker.object_id
            decision = decisions[object_id]
            candidate = candidate_results.get(object_id)
            if decision.measurement_accepted:
                if candidate is None:
                    raise RuntimeError(
                        f"Accepted identity decision has no candidate for {object_id}."
                    )
                tracker.last_result = candidate
                results.append(candidate)
                continue

            snapshot = snapshots.get(object_id)
            if snapshot is not None:
                tracker.restore_tracking_state(snapshot)

            output_pose = decision.output_pose
            state = decision.state
            message = f"identity: {decision.reason}"
            if output_pose is not None:
                try:
                    tracker.set_tracking_pose(output_pose)
                except Exception as error:
                    output_pose = None
                    state = TrackingState.LOST
                    message += f"; prediction state update failed: {error}"

            result = PoseResult(
                cycle_id=cycle_id,
                object_id=object_id,
                source_frame_id=frame.source_frame_id,
                host_wall_time_s=frame.host_wall_time_s,
                host_monotonic_time_s=frame.host_monotonic_time_s,
                mode=TrackingMode.TRACK,
                state=state,
                valid=output_pose is not None,
                pose=output_pose,
                processing_time_s=(
                    0.0 if candidate is None else candidate.processing_time_s
                ),
                message=message,
            )
            tracker.last_result = result
            results.append(result)

        self.last_identity_decisions = decisions
        return results
