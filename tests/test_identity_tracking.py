"""CPU-only Kalman, occlusion, gating, and rollback regression tests."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from camera.base import FrameData
from config import (
    AppConfig,
    FoundationPoseConfig,
    ObjectConfig,
    OutputConfig,
    TrackingConfig,
)
from constants import PIPE1_ID, PIPE2_ID, TrackingMode, TrackingState
from pose_estimation.foundationpose_tracker import FoundationPoseTracker
from pose_estimation.pose_result import PoseResult
from tracking.identity_tracker import MultiObjectIdentityTracker
from tracking.kalman_position_tracker import ConstantVelocityPositionKalmanFilter
from tracking.occlusion_manager import (
    OcclusionGeometry,
    analyze_depth_occlusion,
)
from tracking.tracking_manager import TrackingManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECT_IDS = (PIPE1_ID, PIPE2_ID)


def make_filter() -> ConstantVelocityPositionKalmanFilter:
    return ConstantVelocityPositionKalmanFilter(
        process_acceleration_std_mps2=0.1,
        measurement_position_std_m=0.005,
        initial_position_std_m=0.01,
        initial_velocity_std_mps=1.0,
    )


def pose_at(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (x, y, z)
    return pose


def frame_at(
    timestamp_s: float,
    depth_m: float = 0.0,
    frame_id: int = 1,
) -> FrameData:
    return FrameData(
        source_frame_id=frame_id,
        rgb=np.zeros((200, 200, 3), dtype=np.uint8),
        depth_m=np.full((200, 200), depth_m, dtype=np.float32),
        K=np.array(
            [[200.0, 0.0, 100.0], [0.0, 200.0, 100.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        device_timestamp_ms=timestamp_s * 1000.0,
        timestamp_domain="mock",
        host_wall_time_s=1000.0 + timestamp_s,
        host_monotonic_time_s=timestamp_s,
    )


def geometry_map() -> dict:
    bbox = np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]], dtype=np.float64)
    return {
        object_id: OcclusionGeometry(bbox=bbox, to_origin=np.eye(4))
        for object_id in OBJECT_IDS
    }


def identity_config(mode: str = "motion") -> TrackingConfig:
    return TrackingConfig(
        identity_mode=mode,
        process_acceleration_std_mps2=0.1,
        measurement_position_std_m=0.005,
        initial_position_std_m=0.01,
        initial_velocity_std_mps=1.0,
        occlusion_overlap_ratio=0.25,
        occlusion_min_depth_separation_m=0.04,
        occlusion_depth_margin_m=0.01,
        occlusion_min_valid_depth_pixels=20,
        max_predict_only_seconds=1.0,
        max_prediction_std_m=2.0,
    )


def test_cv_kalman_stationary_object() -> None:
    kalman_filter = make_filter()
    position = np.array([0.2, -0.1, 0.8])
    kalman_filter.initialize(position, 0.0)

    for timestamp in (0.05, 0.13, 0.28, 0.51, 0.9):
        kalman_filter.update(position, timestamp)

    np.testing.assert_allclose(kalman_filter.position, position, atol=1.0e-3)
    np.testing.assert_allclose(kalman_filter.velocity, 0.0, atol=3.0e-3)


def test_cv_kalman_constant_velocity() -> None:
    kalman_filter = make_filter()
    initial = np.array([-0.3, 0.1, 1.0])
    velocity = np.array([0.4, -0.2, 0.1])
    kalman_filter.initialize(initial, 0.0)

    for timestamp in np.linspace(0.1, 2.0, 20):
        kalman_filter.update(initial + velocity * timestamp, float(timestamp))

    np.testing.assert_allclose(
        kalman_filter.position,
        initial + velocity * 2.0,
        atol=3.0e-3,
    )
    np.testing.assert_allclose(kalman_filter.velocity, velocity, atol=5.0e-3)


def test_cv_kalman_uses_irregular_timestamp_delta() -> None:
    kalman_filter = make_filter()
    kalman_filter.initialize(np.zeros(3), 2.0)
    kalman_filter.update(np.array([0.2, 0.0, 0.0]), 2.2)
    before_position = kalman_filter.position
    before_velocity = kalman_filter.velocity

    predicted = kalman_filter.predict(3.05)

    np.testing.assert_allclose(
        predicted,
        before_position + before_velocity * 0.85,
        atol=1.0e-12,
    )
    assert kalman_filter.timestamp_s == 3.05


def test_normal_measurement_accepted_and_large_innovation_rejected() -> None:
    identity = MultiObjectIdentityTracker(OBJECT_IDS, geometry_map(), identity_config())
    identity.initialize(
        {PIPE1_ID: pose_at(-0.3, 0.0, 1.0), PIPE2_ID: pose_at(0.3, 0.0, 1.0)},
        0.0,
    )
    cycle = identity.begin_cycle(frame_at(0.1))
    decisions = identity.evaluate(
        cycle,
        {
            PIPE1_ID: pose_at(-0.29, 0.0, 1.0),
            PIPE2_ID: pose_at(2.0, 0.0, 1.0),
        },
    )

    assert decisions[PIPE1_ID].measurement_accepted
    assert not decisions[PIPE1_ID].predicted_only
    assert not decisions[PIPE2_ID].measurement_accepted
    assert decisions[PIPE2_ID].predicted_only
    assert decisions[PIPE2_ID].reason == "innovation_gate_rejected"


def test_non_overlapping_projection_is_not_occlusion() -> None:
    decisions = analyze_depth_occlusion(
        frame_at(0.1, depth_m=1.0),
        {
            PIPE1_ID: pose_at(-0.6, 0.0, 1.0),
            PIPE2_ID: pose_at(0.6, 0.0, 1.2),
        },
        geometry_map(),
        identity_config("depth_motion"),
    )

    assert not decisions[PIPE1_ID].occluded
    assert not decisions[PIPE2_ID].occluded


def test_overlap_and_depth_marks_only_rear_object_occluded() -> None:
    decisions = analyze_depth_occlusion(
        frame_at(0.1, depth_m=1.0),
        {
            PIPE1_ID: pose_at(0.0, 0.0, 1.0),
            PIPE2_ID: pose_at(0.0, 0.0, 1.2),
        },
        geometry_map(),
        identity_config("depth_motion"),
    )

    assert not decisions[PIPE1_ID].occluded
    assert decisions[PIPE1_ID].occludes_object_id == PIPE2_ID
    assert decisions[PIPE2_ID].occluded
    assert decisions[PIPE2_ID].occluding_object_id == PIPE1_ID
    assert decisions[PIPE2_ID].observed_depth_m == 1.0


def test_occlusion_sequence_rejects_collapse_and_accepts_reappearance() -> None:
    identity = MultiObjectIdentityTracker(
        OBJECT_IDS,
        geometry_map(),
        identity_config("depth_motion"),
    )
    identity.initialize(
        {
            PIPE1_ID: pose_at(-0.3, 0.0, 1.0),
            PIPE2_ID: pose_at(0.3, 0.0, 1.2),
        },
        0.0,
    )

    for index, timestamp in enumerate((0.1, 0.2), start=1):
        offset = 0.3 - timestamp
        cycle = identity.begin_cycle(frame_at(timestamp, depth_m=0.0, frame_id=index))
        decisions = identity.evaluate(
            cycle,
            {
                PIPE1_ID: pose_at(-offset, 0.0, 1.0),
                PIPE2_ID: pose_at(offset, 0.0, 1.2),
            },
        )
        assert all(decision.measurement_accepted for decision in decisions.values())

    overlap_cycle = identity.begin_cycle(frame_at(0.3, depth_m=1.0, frame_id=3))
    front_measurement = pose_at(0.0, 0.0, 1.0)
    overlap_decisions = identity.evaluate(
        overlap_cycle,
        {
            PIPE1_ID: front_measurement,
            # Simulate Pipe2 FoundationPose collapsing onto visible Pipe1.
            PIPE2_ID: front_measurement,
        },
    )

    assert overlap_decisions[PIPE1_ID].measurement_accepted
    assert overlap_decisions[PIPE2_ID].state is TrackingState.OCCLUDED
    assert not overlap_decisions[PIPE2_ID].measurement_accepted
    assert overlap_decisions[PIPE2_ID].predicted_only
    assert overlap_decisions[PIPE2_ID].output_pose[2, 3] > 1.1

    reappearance_cycle = identity.begin_cycle(
        frame_at(0.6, depth_m=0.0, frame_id=4)
    )
    pipe2_reappeared = reappearance_cycle.predicted_poses[PIPE2_ID].copy()
    reappearance_decisions = identity.evaluate(
        reappearance_cycle,
        {
            PIPE1_ID: reappearance_cycle.predicted_poses[PIPE1_ID],
            PIPE2_ID: pipe2_reappeared,
        },
    )

    assert reappearance_decisions[PIPE2_ID].measurement_accepted
    assert reappearance_decisions[PIPE2_ID].state is TrackingState.TRACKING
    np.testing.assert_allclose(
        reappearance_decisions[PIPE2_ID].output_pose,
        pipe2_reappeared,
    )


class FakeStatefulObjectTracker:
    instances = {}
    initial_poses = {
        PIPE1_ID: pose_at(-0.3, 0.0, 1.0),
        PIPE2_ID: pose_at(0.3, 0.0, 1.0),
    }
    candidate_poses = {
        PIPE1_ID: pose_at(-0.29, 0.0, 1.0),
        PIPE2_ID: pose_at(-0.29, 0.0, 1.0),
    }

    def __init__(self, object_config, foundationpose_config, runtime, debug_dir):
        del foundationpose_config, runtime, debug_dir
        self.object_config = object_config
        self.state = TrackingState.UNINITIALIZED
        self.last_result = None
        self.estimator = SimpleNamespace(pose_last=None)
        self.bbox = geometry_map()[self.object_id].bbox
        self.to_origin = np.eye(4)
        self.track_calls = 0
        self.instances[self.object_id] = self

    @property
    def object_id(self):
        return self.object_config.object_id

    def validate_registration_mask(self, frame, mask):
        del frame
        return np.ascontiguousarray(mask, dtype=np.bool_)

    def register(self, frame, mask, cycle_id):
        del mask
        pose = self.initial_poses[self.object_id].copy()
        self.estimator.pose_last = pose.copy()
        self.state = TrackingState.TRACKING
        return self._result(frame, cycle_id, pose, TrackingMode.REGISTER)

    def track(self, frame, cycle_id):
        self.track_calls += 1
        candidate = self.candidate_poses[self.object_id].copy()
        self.estimator.pose_last = candidate.copy()
        return self._result(frame, cycle_id, candidate, TrackingMode.TRACK)

    def backup_tracking_state(self):
        return (
            self.estimator.pose_last.copy(),
            self.state,
            self.last_result,
        )

    def restore_tracking_state(self, snapshot):
        pose_last, self.state, self.last_result = snapshot
        self.estimator.pose_last = pose_last.copy()

    def set_tracking_pose(self, pose):
        self.estimator.pose_last = pose.copy()
        self.state = TrackingState.TRACKING

    def _result(self, frame, cycle_id, pose, mode):
        result = PoseResult(
            cycle_id=cycle_id,
            object_id=self.object_id,
            source_frame_id=frame.source_frame_id,
            host_wall_time_s=frame.host_wall_time_s,
            host_monotonic_time_s=frame.host_monotonic_time_s,
            mode=mode,
            state=self.state,
            valid=True,
            pose=pose,
            processing_time_s=0.001,
        )
        self.last_result = result
        return result


def manager_config(tmp_path, mode="motion", object_ids=OBJECT_IDS):
    model_paths = {
        PIPE1_ID: PROJECT_ROOT / "models" / "pipe1.obj",
        PIPE2_ID: PROJECT_ROOT / "models" / "pipe2.obj",
    }
    return AppConfig(
        foundationpose=FoundationPoseConfig(root=PROJECT_ROOT.parent / "FoundationPose"),
        tracking=identity_config(mode),
        objects=tuple(
            ObjectConfig(object_id, model_paths[object_id], 0.001)
            for object_id in object_ids
        ),
        output=OutputConfig(output_dir=tmp_path),
    )


def registration_masks():
    first = np.zeros((200, 200), dtype=np.bool_)
    second = np.zeros((200, 200), dtype=np.bool_)
    first[20:60, 20:60] = True
    second[120:160, 120:160] = True
    return {PIPE1_ID: first, PIPE2_ID: second}


def test_rejected_candidate_does_not_contaminate_estimator_state(tmp_path) -> None:
    FakeStatefulObjectTracker.instances = {}
    with patch("tracking.tracking_manager.ObjectTracker", FakeStatefulObjectTracker):
        manager = TrackingManager(manager_config(tmp_path), runtime=object())
        manager.register_all(frame_at(0.0, frame_id=0), registration_masks())
        results = manager.track_all(frame_at(0.1, frame_id=1))

    by_id = {result.object_id: result for result in results}
    pipe2 = FakeStatefulObjectTracker.instances[PIPE2_ID]
    assert manager.last_identity_decisions[PIPE1_ID].measurement_accepted
    assert not manager.last_identity_decisions[PIPE2_ID].measurement_accepted
    assert by_id[PIPE2_ID].valid
    assert "identity:" in by_id[PIPE2_ID].message
    np.testing.assert_allclose(pipe2.estimator.pose_last, by_id[PIPE2_ID].pose)
    assert not np.allclose(
        pipe2.estimator.pose_last,
        FakeStatefulObjectTracker.candidate_poses[PIPE2_ID],
    )


def test_depth_occlusion_updates_front_and_skips_rear_estimator(tmp_path) -> None:
    initial_poses = {
        PIPE1_ID: pose_at(0.0, 0.0, 1.0),
        PIPE2_ID: pose_at(0.0, 0.0, 1.2),
    }
    candidate_poses = {
        PIPE1_ID: pose_at(0.0, 0.0, 1.0),
        PIPE2_ID: pose_at(0.0, 0.0, 1.0),
    }
    FakeStatefulObjectTracker.instances = {}
    with (
        patch.object(FakeStatefulObjectTracker, "initial_poses", initial_poses),
        patch.object(FakeStatefulObjectTracker, "candidate_poses", candidate_poses),
        patch("tracking.tracking_manager.ObjectTracker", FakeStatefulObjectTracker),
    ):
        manager = TrackingManager(
            manager_config(tmp_path, mode="depth_motion"),
            runtime=object(),
        )
        manager.register_all(
            frame_at(0.0, depth_m=1.0, frame_id=0),
            registration_masks(),
        )
        results = manager.track_all(frame_at(0.1, depth_m=1.0, frame_id=1))

    by_id = {result.object_id: result for result in results}
    pipe1 = FakeStatefulObjectTracker.instances[PIPE1_ID]
    pipe2 = FakeStatefulObjectTracker.instances[PIPE2_ID]
    assert pipe1.track_calls == 1
    assert pipe2.track_calls == 0
    assert manager.last_identity_decisions[PIPE1_ID].measurement_accepted
    assert manager.last_identity_decisions[PIPE2_ID].predicted_only
    assert by_id[PIPE2_ID].state is TrackingState.OCCLUDED
    np.testing.assert_allclose(by_id[PIPE2_ID].pose, initial_poses[PIPE2_ID])


def test_identity_none_preserves_raw_manager_behavior(tmp_path) -> None:
    FakeStatefulObjectTracker.instances = {}
    with patch("tracking.tracking_manager.ObjectTracker", FakeStatefulObjectTracker):
        manager = TrackingManager(manager_config(tmp_path, mode="none"), runtime=object())
        manager.register_all(frame_at(0.0, frame_id=0), registration_masks())
        results = manager.track_all(frame_at(0.1, frame_id=1))

    assert manager.identity_tracker is None
    assert manager.last_identity_decisions == {}
    np.testing.assert_allclose(results[1].pose, pose_at(-0.29, 0.0, 1.0))


def test_single_object_automatically_bypasses_identity_layer(tmp_path) -> None:
    FakeStatefulObjectTracker.instances = {}
    with patch("tracking.tracking_manager.ObjectTracker", FakeStatefulObjectTracker):
        manager = TrackingManager(
            manager_config(tmp_path, mode="depth_motion", object_ids=(PIPE1_ID,)),
            runtime=object(),
        )

    assert manager.identity_tracker is None


def test_foundationpose_wrapper_backup_restore_and_prediction_state() -> None:
    tracker = FoundationPoseTracker.__new__(FoundationPoseTracker)
    tracker.object_config = SimpleNamespace(object_id=PIPE1_ID)
    tracker.state = TrackingState.TRACKING
    tracker.last_result = "last accepted"
    centered_transform = np.eye(4, dtype=np.float64)
    centered_transform[0, 3] = -0.1
    tracker.estimator = SimpleNamespace(
        pose_last=np.eye(4, dtype=np.float32)[None],
        get_tf_to_centered_mesh=lambda: centered_transform,
    )

    snapshot = tracker.backup_tracking_state()
    tracker.estimator.pose_last[..., 0, 3] = 99.0
    tracker.state = TrackingState.LOST
    tracker.restore_tracking_state(snapshot)

    assert tracker.estimator.pose_last.shape == (1, 4, 4)
    assert tracker.estimator.pose_last[0, 0, 3] == 0.0
    assert tracker.state is TrackingState.TRACKING
    assert tracker.last_result == "last accepted"

    predicted_pose = pose_at(0.5, 0.0, 1.0)
    tracker.set_tracking_pose(predicted_pose)
    assert tracker.estimator.pose_last.shape == (1, 4, 4)
    np.testing.assert_allclose(
        tracker.estimator.pose_last[0, :3, 3],
        (0.6, 0.0, 1.0),
    )
