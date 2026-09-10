# MULTI_OBJECT_TRACKING

A responsibility-oriented migration of the existing D405, FoundationPose, and
pipe_tracking work. Single- and dual-pipe live regression entrypoints use the
same camera, segmentation, and tracking abstractions. Hardware accuracy still
depends on the selected CAD, masks, scene, and FoundationPose environment.

## Responsibilities

    ../RealSenseD405/ sibling D405 acquisition dependency (SDK and sensor tools)
    camera/           Camera-neutral application contract and thin adapters
    segmentation/     Minimal interface and manual polygon implementation
    pose_estimation/  FoundationPose dependency boundary and pose result
    pose_processing/  Unchanged discrete assembly-task canonicalization
    tracking/         Object lifecycle and one/two-object orchestration
    visualization/    OpenCV pose overlays
    pose_logging/     Reserved persistent pose-output layer
    scripts/          Hardware and tracking regression entrypoints
    ros2/             Future ROS2 integration notes

The output package is named pose_logging, not logging, because a top-level
logging package would shadow Python's standard library and can break
FoundationPose and third-party imports.

## Sibling dependencies

FoundationPose is not vendored. Its configured default path is the sibling
checkout `/home/kkb/Workspace/FoundationPose`. Follow NVIDIA's instructions to
build its native extensions and obtain model weights. A different checkout can
be selected with `--foundationpose-root`.

The D405 implementation is also a sibling dependency. Install it once in the
same environment used to run this application:

    cd /home/kkb/Workspace/MULTI_OBJECT_TRACKING
    python3 -m pip install -e ../RealSenseD405

Equivalently, `python3 -m pip install -r requirements-camera.txt` installs that
editable local dependency. This avoids runtime `sys.path` modification and
keeps the retained legacy `MULTI_OBJECT_TRACKING/RealSenseD405/` from
shadowing the new package: application code imports the unambiguous lowercase
module name `realsense_d405`.

No local experimental scripts or modifications from the existing FoundationPose
directory are copied into this dependency location.

## Current single-object regression

Run from the FoundationPose GPU environment with a D405 attached:

    cd /home/kkb/Workspace/MULTI_OBJECT_TRACKING
    python3 scripts/test_single_object.py \
      --foundationpose-root /home/kkb/Workspace/FoundationPose \
      --model-path /home/kkb/Workspace/FoundationPose/custom_data/pipe/pipe.stl \
      --mesh-scale-to-meter 0.001 \
      --task-symmetry-debug

run_tracking.py currently exposes the same verified single-object path. A real
model path is required so a nonexistent Pipe2 CAD is never fabricated or
silently replaced with the Pipe1 CAD.

The migrated task canonicalizer is unchanged: it compares six discrete
candidates formed from Z rotations of 0/120/240 degrees and X rotations of
0/180 degrees. The single-object post-processing options remain available.

## D405 tools

    python3 -m realsense_d405.tools.inspect_camera
    python3 -m realsense_d405.tools.capture_rgbd
    python3 -m realsense_d405.tools.record_rgbd --duration 10
    python3 scripts/test_camera.py --duration 4

Capture and recording output defaults to `recordings/` below the current
working directory. The sibling sensor package contains no T-LESS,
segmentation, pose-estimation, tracking, or application visualization code.

## Adding another camera

Put the vendor-specific implementation in another sibling package, for
example `../ZEDCamera/`. Add `camera/zed.py` with a `CameraSource` adapter and
a `create_source(config)` hook, then select it with `--camera-type zed`. The
factory imports only the selected adapter. Segmentation, pose estimation, and
tracking continue to receive the same `FrameData` contract and do not import a
vendor SDK.

The previous offline T-LESS integration smoke path is retained outside the
D405 sensor package:

    python3 scripts/test_tless_smoke.py \
      --foundationpose-root /home/kkb/Workspace/FoundationPose \
      --tless-root /home/kkb/Workspace/FoundationPose/custom_data/tless06

## Multi-object and ROS2 status

TrackingManager still creates a separate object tracker—and therefore a
separate FoundationPose estimator and pose_last—for each configured object.
Heavy scorer/refiner/raster resources are shared and inference calls remain
serialized. `scripts/test_multi_object.py` uses one D405 frame per cycle for
both trackers. Initial masks are selected in Pipe1 then Pipe2 order on the same
frozen frame. The live overlay uses yellow for Pipe1 and magenta for Pipe2.
Task symmetry and Z-axis stabilization are both off by default.

    python3 scripts/test_multi_object.py \
      --foundationpose-root /home/kkb/Workspace/FoundationPose \
      --pipe1-model-path models/pipe1.obj \
      --pipe2-model-path models/pipe2.obj \
      --mesh-scale-to-meter 0.001 \
      --identity-mode depth_motion \
      --identity-debug

Dual-object identity protection defaults to `depth_motion`. Each object has an
independent NumPy constant-velocity translation filter. Before calling an
estimator, predicted CAD bounds are projected into the shared RGB-D frame. If
their image regions overlap and observed depth agrees with the predicted front
object, the rear estimator call is skipped and its translation is predicted;
the last accepted raw rotation is retained. Visible estimator candidates pass
a configurable 3D Mahalanobis innovation gate before being committed.

The estimator state is snapshotted before `track_one()`. A rejected candidate
restores the previous `pose_last`, then seeds the next estimator call from the
identity-safe predicted pose. Task symmetry and Z-axis stabilization remain
strictly downstream output processing and never feed the identity filter.

Use `--identity-mode motion` to test prediction and gating without depth
occlusion handling, or `--identity-mode none` for the original unprotected
dual-estimator baseline. `--identity-debug` prints measured/predicted XYZ,
velocity, Mahalanobis distance, front/rear visibility, depth evidence, and the
ACCEPT/PREDICT decision while adding a compact state line to the live overlay.

`run_tracking.py` continues to expose the verified single-object path. Actual
dual-object D405 accuracy and performance must be checked with both physical
objects present.

ROS2 is not imported by the core. A future node can replace the orchestration
layer without changing estimator or tracking contracts; see ros2/README.md.

## Migration map

| Existing source | Migrated source |
|---|---|
| pipe_tracking/input/realsense_camera.py | ../RealSenseD405/src/realsense_d405/camera.py plus camera/realsense_d405.py |
| pipe_tracking/core/frame_data.py | camera/base.py |
| pipe_tracking/input/sequence_source.py | camera/sequence.py |
| d405_study/src/geometry.py | ../RealSenseD405/src/realsense_d405/geometry.py |
| d405_study/capture_rgbd.py | ../RealSenseD405/src/realsense_d405/tools/capture_rgbd.py |
| d405_study/record_rgbd.py | ../RealSenseD405/src/realsense_d405/tools/record_rgbd.py |
| pipe_tracking/segmentation/manual_segmentation.py | segmentation/manual/polygon_segmenter.py |
| pipe_tracking/core/foundationpose_runtime.py | pose_estimation/foundationpose_runtime.py |
| pipe_tracking/core/object_tracker.py | pose_estimation/foundationpose_tracker.py, exported as tracking.ObjectTracker |
| pipe_tracking/core/pose_result.py | pose_estimation/pose_result.py |
| pipe_tracking/core/pipe_task_symmetry.py | pose_processing/task_symmetry.py |
| pipe_tracking/core/tracking_manager.py | tracking/tracking_manager.py |
| pipe_tracking/output/visualization.py | visualization/visualization.py |
| pipe_tracking/scripts/live_test_pipe1.py | scripts/test_single_object.py |
| pipe_tracking/scripts/smoke_test_realsense.py | scripts/test_camera.py |
| pipe_tracking/scripts/smoke_test_tless.py | scripts/test_tless_smoke.py |

The existing `MULTI_OBJECT_TRACKING/RealSenseD405`, `d405_study`, and
`pipe_tracking` directories remain untouched until hardware regression is
completed. They are migration sources, not runtime dependencies of the new
adapter.
