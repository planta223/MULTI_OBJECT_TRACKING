"""CPU-only dual-object orchestration and visualization regression tests."""

from pathlib import Path
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
from pose_estimation.pose_result import PoseResult
from scripts.test_multi_object import parse_args
from segmentation.manual import ManualPolygonSegmenter
from tracking.tracking_manager import TrackingManager
from visualization import draw_pose_overlay, draw_pose_overlay_on_bgr


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_frame(frame_id: int = 10) -> FrameData:
    return FrameData(
        source_frame_id=frame_id,
        rgb=np.zeros((8, 10, 3), dtype=np.uint8),
        depth_m=np.ones((8, 10), dtype=np.float32),
        K=np.array(
            [[100.0, 0.0, 5.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        device_timestamp_ms=1.0,
        timestamp_domain="mock",
        host_wall_time_s=2.0,
        host_monotonic_time_s=3.0,
    )


class FakeObjectTracker:
    instances = {}

    def __init__(
        self,
        object_config,
        foundationpose_config,
        runtime,
        debug_dir,
    ) -> None:
        del foundationpose_config, runtime, debug_dir
        self.object_config = object_config
        self.state = TrackingState.UNINITIALIZED
        self.last_result = None
        self.estimator = object()
        self.register_frame = None
        self.track_frame = None
        self.registration_mask = None
        self.instances[self.object_id] = self

    @property
    def object_id(self) -> str:
        return self.object_config.object_id

    def validate_registration_mask(self, frame, mask):
        del frame
        return np.ascontiguousarray(mask, dtype=np.bool_)

    def register(self, frame, mask, cycle_id):
        self.register_frame = frame
        self.registration_mask = mask.copy()
        self.state = TrackingState.TRACKING
        result = self._success(frame, cycle_id, TrackingMode.REGISTER)
        self.last_result = result
        return result

    def track(self, frame, cycle_id):
        self.track_frame = frame
        if self.object_id == PIPE1_ID:
            raise RuntimeError("synthetic Pipe1 failure")
        result = self._success(frame, cycle_id, TrackingMode.TRACK)
        self.last_result = result
        return result

    def _success(self, frame, cycle_id, mode):
        return PoseResult(
            cycle_id=cycle_id,
            object_id=self.object_id,
            source_frame_id=frame.source_frame_id,
            host_wall_time_s=frame.host_wall_time_s,
            host_monotonic_time_s=frame.host_monotonic_time_s,
            mode=mode,
            state=self.state,
            valid=True,
            pose=np.eye(4, dtype=np.float64),
            processing_time_s=0.001,
        )


def test_dual_cli_defaults_to_both_real_models() -> None:
    args = parse_args([])

    assert args.foundationpose_root == PROJECT_ROOT.parent / "FoundationPose"
    assert args.pipe1_model_path == PROJECT_ROOT / "models" / "pipe1.obj"
    assert args.pipe2_model_path == PROJECT_ROOT / "models" / "pipe2.obj"
    assert args.mesh_scale_to_meter == 0.001
    assert args.task_symmetry_output is False
    assert args.z_axis_stabilization is False
    assert args.identity_mode == "depth_motion"
    assert TrackingConfig().identity_mode == "depth_motion"


def test_manual_segmenter_uses_same_frame_and_separate_object_ids() -> None:
    frame = make_frame()
    calls = []
    segmenter = ManualPolygonSegmenter((PIPE1_ID, PIPE2_ID))

    def fake_select(selected_frame, object_id):
        calls.append((selected_frame, object_id))
        fill = 1 if object_id == PIPE1_ID else 2
        return np.full((frame.height, frame.width), fill, dtype=np.uint8)

    with patch.object(segmenter, "_select_one", side_effect=fake_select):
        masks = segmenter.segment(frame)

    assert calls == [(frame, PIPE1_ID), (frame, PIPE2_ID)]
    assert masks[PIPE1_ID] is not masks[PIPE2_ID]
    assert np.all(masks[PIPE1_ID] == 1)
    assert np.all(masks[PIPE2_ID] == 2)


def test_manager_shares_frame_but_isolates_trackers_and_failures(tmp_path) -> None:
    FakeObjectTracker.instances = {}
    config = AppConfig(
        foundationpose=FoundationPoseConfig(root=PROJECT_ROOT.parent / "FoundationPose"),
        tracking=TrackingConfig(identity_mode="none"),
        objects=(
            ObjectConfig(PIPE1_ID, PROJECT_ROOT / "models" / "pipe1.obj", 0.001),
            ObjectConfig(PIPE2_ID, PROJECT_ROOT / "models" / "pipe2.obj", 0.001),
        ),
        output=OutputConfig(output_dir=tmp_path),
    )
    frame = make_frame()
    masks = {
        PIPE1_ID: np.pad(np.ones((4, 4), dtype=np.bool_), ((0, 4), (0, 6))),
        PIPE2_ID: np.pad(np.ones((4, 4), dtype=np.bool_), ((4, 0), (6, 0))),
    }

    with patch("tracking.tracking_manager.ObjectTracker", FakeObjectTracker):
        manager = TrackingManager(config, runtime=object())
        registration = manager.register_all(frame, masks)
        tracking = manager.track_all(frame)

    pipe1 = FakeObjectTracker.instances[PIPE1_ID]
    pipe2 = FakeObjectTracker.instances[PIPE2_ID]
    assert pipe1 is not pipe2
    assert pipe1.estimator is not pipe2.estimator
    assert pipe1.register_frame is frame and pipe2.register_frame is frame
    assert pipe1.track_frame is frame and pipe2.track_frame is frame
    assert not np.array_equal(pipe1.registration_mask, pipe2.registration_mask)
    assert {result.cycle_id for result in registration} == {0}
    assert {result.cycle_id for result in tracking} == {1}
    assert [result.valid for result in tracking] == [False, True]
    assert "synthetic Pipe1 failure" in tracking[0].message
    assert tracking[1].object_id == PIPE2_ID


def test_bgr_overlay_composes_two_distinct_object_colors() -> None:
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    K = np.array(
        [[120.0, 0.0, 90.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    bbox = np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]], dtype=np.float64)
    pipe1_pose = np.eye(4, dtype=np.float64)
    pipe1_pose[2, 3] = 1.0
    pipe2_pose = pipe1_pose.copy()
    pipe2_pose[0, 3] = 0.35
    pipe1_color = (0, 255, 255)
    pipe2_color = (255, 0, 255)

    image = draw_pose_overlay_on_bgr(
        image,
        K,
        pipe1_pose,
        bbox,
        np.eye(4),
        label="Pipe1",
        overlay_color=pipe1_color,
        text_origin=(5, 15),
    )
    image = draw_pose_overlay_on_bgr(
        image,
        K,
        pipe2_pose,
        bbox,
        np.eye(4),
        label="Pipe2",
        overlay_color=pipe2_color,
        text_origin=(5, 35),
    )

    assert np.any(np.all(image == pipe1_color, axis=2))
    assert np.any(np.all(image == pipe2_color, axis=2))


def test_existing_single_object_overlay_wrapper_still_runs() -> None:
    rgb = np.zeros((120, 180, 3), dtype=np.uint8)
    K = np.array(
        [[120.0, 0.0, 90.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    bbox = np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[2, 3] = 1.0

    image = draw_pose_overlay(
        rgb=rgb,
        K=K,
        pose=pose,
        bbox=bbox,
        to_origin=np.eye(4),
        label="Pipe1",
    )

    assert image.shape == rgb.shape
    assert image.dtype == np.uint8
    assert not np.shares_memory(image, rgb)
