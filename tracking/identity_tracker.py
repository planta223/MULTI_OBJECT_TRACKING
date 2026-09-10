"""Estimator-independent temporal identity validation for known objects."""

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Optional

import numpy as np

from camera.base import FrameData
from config import TrackingConfig
from constants import TrackingState
from .kalman_position_tracker import ConstantVelocityPositionKalmanFilter
from .occlusion_manager import (
    OcclusionDecision,
    OcclusionGeometry,
    analyze_depth_occlusion,
)


def _immutable_optional(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if array is None:
        return None
    copied = np.array(array, dtype=np.float64, copy=True, order="C")
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class IdentityCycle:
    """Predictions and visibility decisions fixed before estimator calls."""

    timestamp_s: float
    predicted_poses: Mapping[str, np.ndarray]
    occlusions: Mapping[str, OcclusionDecision]


@dataclass(frozen=True)
class IdentityDecision:
    """Identity-layer decision for one estimator candidate."""

    object_id: str
    state: TrackingState
    output_pose: Optional[np.ndarray]
    predicted_pose: np.ndarray
    measured_pose: Optional[np.ndarray]
    velocity_mps: np.ndarray
    measurement_accepted: bool
    predicted_only: bool
    reason: str
    innovation_m: Optional[np.ndarray] = None
    mahalanobis_distance_sq: Optional[float] = None
    prediction_std_m: float = 0.0
    occlusion: Optional[OcclusionDecision] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_pose", _immutable_optional(self.output_pose))
        object.__setattr__(
            self,
            "predicted_pose",
            _immutable_optional(self.predicted_pose),
        )
        object.__setattr__(
            self,
            "measured_pose",
            _immutable_optional(self.measured_pose),
        )
        object.__setattr__(
            self,
            "velocity_mps",
            _immutable_optional(self.velocity_mps),
        )
        object.__setattr__(
            self,
            "innovation_m",
            _immutable_optional(self.innovation_m),
        )

    @property
    def measured_position(self) -> Optional[np.ndarray]:
        if self.measured_pose is None:
            return None
        return self.measured_pose[:3, 3].copy()

    @property
    def predicted_position(self) -> np.ndarray:
        return self.predicted_pose[:3, 3].copy()

    @property
    def visibility_label(self) -> str:
        if self.occlusion is None:
            return "VISIBLE"
        if self.occlusion.occluded:
            return f"REAR_BEHIND_{self.occlusion.occluding_object_id}"
        if self.occlusion.occludes_object_id is not None:
            return f"FRONT_OF_{self.occlusion.occludes_object_id}"
        return "VISIBLE"

    def debug_text(self) -> str:
        measured = (
            "none"
            if self.measured_pose is None
            else np.array2string(self.measured_pose[:3, 3], precision=3)
        )
        predicted = np.array2string(self.predicted_pose[:3, 3], precision=3)
        velocity = np.array2string(self.velocity_mps, precision=3)
        distance = (
            "n/a"
            if self.mahalanobis_distance_sq is None
            else f"{self.mahalanobis_distance_sq:.2f}"
        )
        action = "ACCEPT" if self.measurement_accepted else "PREDICT"
        depth = (
            "n/a"
            if self.occlusion is None or self.occlusion.observed_depth_m is None
            else f"{self.occlusion.observed_depth_m:.3f}"
        )
        return (
            f"{self.object_id} {self.state.value} {self.visibility_label} {action} "
            f"meas={measured} pred={predicted} vel={velocity} "
            f"d2={distance} depth={depth} reason={self.reason}"
        )


@dataclass
class _IdentityTrack:
    kalman_filter: ConstantVelocityPositionKalmanFilter
    last_accepted_pose: np.ndarray
    last_accepted_timestamp_s: float
    state: TrackingState = TrackingState.TRACKING


class MultiObjectIdentityTracker:
    """Validate object-specific pose candidates without estimator internals."""

    def __init__(
        self,
        object_ids,
        geometries: Mapping[str, OcclusionGeometry],
        config: TrackingConfig,
    ) -> None:
        self.object_ids = tuple(object_ids)
        if len(self.object_ids) < 2:
            raise ValueError("Multi-object identity tracking requires two or more objects.")
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("Identity object IDs must be unique.")
        if set(self.object_ids) != set(geometries):
            raise ValueError("Identity geometry must match configured object IDs.")
        if config.identity_mode not in ("motion", "depth_motion"):
            raise ValueError("Identity tracker requires motion or depth_motion mode.")
        self.config = config
        self.geometries = dict(geometries)
        self._tracks: Dict[str, _IdentityTrack] = {}

    def _new_filter(self) -> ConstantVelocityPositionKalmanFilter:
        return ConstantVelocityPositionKalmanFilter(
            process_acceleration_std_mps2=self.config.process_acceleration_std_mps2,
            measurement_position_std_m=self.config.measurement_position_std_m,
            initial_position_std_m=self.config.initial_position_std_m,
            initial_velocity_std_mps=self.config.initial_velocity_std_mps,
        )

    def initialize(
        self,
        accepted_poses: Mapping[str, np.ndarray],
        timestamp_s: float,
    ) -> None:
        """Initialize identity state from successful registration poses."""

        if set(accepted_poses) != set(self.object_ids):
            raise ValueError("All configured objects require accepted registration poses.")
        self._tracks.clear()
        for object_id in self.object_ids:
            pose = self._pose(accepted_poses[object_id])
            kalman_filter = self._new_filter()
            kalman_filter.initialize(pose[:3, 3], timestamp_s)
            self._tracks[object_id] = _IdentityTrack(
                kalman_filter=kalman_filter,
                last_accepted_pose=pose,
                last_accepted_timestamp_s=float(timestamp_s),
            )

    @staticmethod
    def _pose(pose: np.ndarray) -> np.ndarray:
        values = np.asarray(pose, dtype=np.float64)
        if values.shape != (4, 4) or not np.all(np.isfinite(values)):
            raise ValueError("Pose must be a finite array with shape (4, 4).")
        return np.array(values, copy=True, order="C")

    def _require_initialized(self) -> None:
        if set(self._tracks) != set(self.object_ids):
            raise RuntimeError("Identity tracker requires successful registration first.")

    def begin_cycle(self, frame: FrameData) -> IdentityCycle:
        """Predict every track before deciding whether estimators may run."""

        self._require_initialized()
        timestamp = float(frame.host_monotonic_time_s)
        predicted_poses = {}
        for object_id in self.object_ids:
            track = self._tracks[object_id]
            position = track.kalman_filter.predict(timestamp)
            pose = track.last_accepted_pose.copy()
            pose[:3, 3] = position
            predicted_poses[object_id] = pose

        if self.config.identity_mode == "depth_motion":
            occlusions = analyze_depth_occlusion(
                frame,
                predicted_poses,
                self.geometries,
                self.config,
            )
        else:
            occlusions = {
                object_id: OcclusionDecision(object_id=object_id)
                for object_id in self.object_ids
            }
        return IdentityCycle(
            timestamp_s=timestamp,
            predicted_poses=predicted_poses,
            occlusions=occlusions,
        )

    def evaluate(
        self,
        cycle: IdentityCycle,
        measured_poses: Mapping[str, Optional[np.ndarray]],
        measurement_errors: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, IdentityDecision]:
        """Gate candidates jointly, update accepted tracks, and predict others."""

        self._require_initialized()
        if set(measured_poses) != set(self.object_ids):
            raise ValueError("Measurements must contain every configured object ID.")
        errors = dict(measurement_errors or {})
        normalized_measurements: Dict[str, Optional[np.ndarray]] = {}
        innovations = {}
        reasons = {}
        accepted = {}

        for object_id in self.object_ids:
            occlusion = cycle.occlusions[object_id]
            measurement = measured_poses[object_id]
            if measurement is not None:
                measurement = self._pose(measurement)
            normalized_measurements[object_id] = measurement

            if occlusion.occluded:
                accepted[object_id] = False
                reasons[object_id] = (
                    f"occluded_by_{occlusion.occluding_object_id}"
                )
                continue
            if measurement is None:
                accepted[object_id] = False
                reasons[object_id] = errors.get(object_id, "estimator_measurement_missing")
                continue

            innovation = self._tracks[object_id].kalman_filter.innovation(
                measurement[:3, 3]
            )
            innovations[object_id] = innovation
            accepted[object_id] = (
                innovation.mahalanobis_distance_sq
                <= self.config.mahalanobis_gate_threshold_sq
            )
            reasons[object_id] = (
                "innovation_gate_passed"
                if accepted[object_id]
                else "innovation_gate_rejected"
            )

            if accepted[object_id]:
                foreign_costs = []
                for other_id in self.object_ids:
                    if other_id == object_id:
                        continue
                    foreign_innovation = self._tracks[
                        other_id
                    ].kalman_filter.innovation(measurement[:3, 3])
                    foreign_costs.append(foreign_innovation.mahalanobis_distance_sq)
                if foreign_costs and (
                    min(foreign_costs) + self.config.association_margin_sq
                    < innovation.mahalanobis_distance_sq
                ):
                    accepted[object_id] = False
                    reasons[object_id] = "candidate_closer_to_other_track"

        initially_accepted = [
            object_id for object_id in self.object_ids if accepted[object_id]
        ]
        for first_id, second_id in combinations(initially_accepted, 2):
            first_measurement = normalized_measurements[first_id]
            second_measurement = normalized_measurements[second_id]
            if first_measurement is None or second_measurement is None:
                continue
            measured_separation = np.linalg.norm(
                first_measurement[:3, 3] - second_measurement[:3, 3]
            )
            predicted_separation = np.linalg.norm(
                cycle.predicted_poses[first_id][:3, 3]
                - cycle.predicted_poses[second_id][:3, 3]
            )
            if (
                measured_separation <= self.config.collapse_distance_m
                and predicted_separation > self.config.collapse_distance_m
            ):
                first_cost = innovations[first_id].mahalanobis_distance_sq
                second_cost = innovations[second_id].mahalanobis_distance_sq
                rejected_id = second_id if first_cost <= second_cost else first_id
                accepted[rejected_id] = False
                reasons[rejected_id] = "duplicate_candidate_collapse"

        decisions = {}
        for object_id in self.object_ids:
            track = self._tracks[object_id]
            measurement = normalized_measurements[object_id]
            innovation = innovations.get(object_id)
            occlusion = cycle.occlusions[object_id]
            if accepted[object_id] and measurement is not None:
                track.kalman_filter.update(
                    measurement[:3, 3],
                    cycle.timestamp_s,
                )
                track.last_accepted_pose = measurement.copy()
                track.last_accepted_timestamp_s = cycle.timestamp_s
                track.state = TrackingState.TRACKING
                output_pose = measurement
                predicted_only = False
            else:
                elapsed = cycle.timestamp_s - track.last_accepted_timestamp_s
                prediction_covariance = track.kalman_filter.prediction_covariance
                prediction_std = float(
                    np.sqrt(max(0.0, np.max(np.linalg.eigvalsh(prediction_covariance))))
                )
                if (
                    elapsed > self.config.max_predict_only_seconds
                    or prediction_std > self.config.max_prediction_std_m
                ):
                    track.state = TrackingState.LOST
                    output_pose = None
                else:
                    if occlusion.occluded or track.state is TrackingState.OCCLUDED:
                        track.state = TrackingState.OCCLUDED
                    else:
                        track.state = TrackingState.TRACKING
                    output_pose = cycle.predicted_poses[object_id]
                predicted_only = True

            prediction_covariance = track.kalman_filter.prediction_covariance
            prediction_std = float(
                np.sqrt(max(0.0, np.max(np.linalg.eigvalsh(prediction_covariance))))
            )
            decisions[object_id] = IdentityDecision(
                object_id=object_id,
                state=track.state,
                output_pose=output_pose,
                predicted_pose=cycle.predicted_poses[object_id],
                measured_pose=measurement,
                velocity_mps=track.kalman_filter.velocity,
                measurement_accepted=accepted[object_id],
                predicted_only=predicted_only,
                reason=reasons[object_id],
                innovation_m=(None if innovation is None else innovation.residual),
                mahalanobis_distance_sq=(
                    None if innovation is None else innovation.mahalanobis_distance_sq
                ),
                prediction_std_m=prediction_std,
                occlusion=occlusion,
            )
        return decisions
