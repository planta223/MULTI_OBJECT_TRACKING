# MULTI_OBJECT_TRACKING

A responsibility-oriented migration of the existing D405, FoundationPose, and
pipe_tracking work. Single- and dual-pipe live regression entrypoints use the
same camera, segmentation, and tracking abstractions. Hardware accuracy still
depends on the selected CAD, masks, scene, and FoundationPose environment.

## Responsibilities

    ../RealSenseD405/ sibling D405 acquisition dependency (SDK and sensor tools)
    ../ZED2iCamera/   sibling ZED 2i acquisition dependency (ZED SDK adapter)
    camera/           Camera-neutral application contract and thin adapters
    segmentation/     Manual or YOLO initial-registration masks
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

### Optional ZED 2i input

Install the Stereolabs ZED SDK and its matching `pyzed` Python API in the same
environment used by FoundationPose. `pyzed` is supplied by the SDK rather than
as a normal portable PyPI dependency. On Linux the SDK installer places its
Python installer under `/usr/local/zed`:

    cd /usr/local/zed
    python3 get_python_api.py
    python3 -c "import pyzed.sl as sl; print('pyzed OK')"

Then install the sibling package:

    cd /home/kkb/Workspace/MULTI_OBJECT_TRACKING
    python3 -m pip install -e ../ZED2iCamera

The application factory imports only the selected camera adapter. A D405 run
does not import `pyzed`, and a ZED run does not import `pyrealsense2`.

## Initial segmentation: manual or YOLO

Manual polygon selection remains the default and needs no YOLO dependency. To
generate the initial FoundationPose registration mask automatically, install
Ultralytics in the same Python/GPU environment used for FoundationPose:

    python3 -m pip install -r requirements-yolo.txt

The FoundationPose image may contain `ultralytics==8.0.120`, which cannot load
YOLO11 checkpoints containing modules such as `C3k2` and `C2PSA`. Install the
optional requirements file to use the verified `ultralytics==8.3.0` runtime.
Run the install in the existing FoundationPose environment so pip can retain
its already-compatible PyTorch/CUDA packages.

YOLO mode requires a custom Ultralytics **instance-segmentation** `.pt` model.
A detection-only model is rejected because bounding boxes are not sufficient
for FoundationPose registration. Generic pretrained segmentation weights do
not automatically recognize this project's Pipe class; train a segmentation
model on the actual Pipe data.

Place or copy the weight to a local path such as:

    models/yolo/pipe_seg.pt

Weight files below `models/yolo/` are intentionally ignored by Git, so copy
them separately when moving the project to another PC.

Single-object YOLO registration:

    python3 scripts/test_single_object.py \
      --model-path models/pipe1.obj \
      --mesh-scale-to-meter 0.001 \
      --segmentation-mode yolo \
      --yolo-model-path models/yolo/pipe_seg.pt \
      --yolo-confidence 0.5 \
      --yolo-class-id 0 \
      --yolo-device cuda:0

Dual-object YOLO registration:

    python3 scripts/test_multi_object.py \
      --pipe1-model-path models/pipe1.obj \
      --pipe2-model-path models/pipe2.obj \
      --mesh-scale-to-meter 0.001 \
      --segmentation-mode yolo \
      --yolo-model-path models/yolo/pipe_seg.pt \
      --yolo-confidence 0.5 \
      --yolo-class-id 0 \
      --yolo-device cuda:0

`--yolo-class-id` and `--yolo-device` are optional. For two same-class
instances, valid detections are filtered by class and confidence, the highest
confidence two are selected, and their bounding-box centers are sorted so the
left instance becomes `pipe1` and the right instance becomes `pipe2`. This is
only an initial-registration rule; it is not identity tracking.

YOLO runs once on the frozen registration frame. After registration,
FoundationPose handles every frame through its existing `track_one()` path.
There is no automatic YOLO re-registration after tracking loss.

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

## Camera selection and smoke tests

The default remains `realsense_d405`. Select `zed2i` for direct SDK input or
`ros_zed` for the existing host-side `cam_zed.py` ROS2 topics. Direct ZED
defaults to its native
HD720 left image (1280 x 720), 30 FPS, and NEURAL depth; `--width` and
`--height` remain D405 stream settings.

    python3 scripts/test_camera.py \
      --camera-type realsense_d405 \
      --duration 4 \
      --preview

    python3 scripts/test_camera.py \
      --camera-type zed2i \
      --zed-resolution HD720 \
      --zed-depth-mode NEURAL \
      --fps 30 \
      --duration 4 \
      --preview

    python3 scripts/test_camera.py \
      --camera-type ros_zed \
      --ros-color-topic /cam/color/compressed \
      --ros-depth-topic /cam/depth/compressed \
      --ros-camera-info-topic /cam/color/camera_info \
      --duration 4 \
      --preview

For the current Humble-host/Foxy-container deployment, source Foxy and select
the checked-in UDP-only Fast DDS participant profile before starting a Docker
subscriber process:

    source /opt/ros/foxy/setup.bash
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DOMAIN_ID=8
    export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/ros2/fastdds_udp.xml"

The profile was added only after direct bidirectional `std_msgs/String` tests
proved that the default cross-user transport path did not carry user data,
while UDP-only carried CameraInfo and both compressed image streams. A
container probe using the host publisher's UID 1000 also worked with default
transports, isolating the issue to the SHM path between the UID-1000 publisher
and root subscriber rather than Foxy/Humble compatibility alone. The host
publisher remains unchanged and can continue using its default transport.

The ZED adapter retrieves the rectified BGRA left view, converts it to RGB,
retrieves the depth measure aligned to that left view, requests meter units,
normalizes invalid depth to `0.0`, and builds `K` once from the rectified-left
calibration. Downstream segmentation and FoundationPose receive the same RGB,
metric-depth, and intrinsic contract for either camera.

Example dual-object ZED execution:

    python3 scripts/test_multi_object.py \
      --camera-type zed2i \
      --zed-resolution HD720 \
      --zed-depth-mode NEURAL \
      --fps 30 \
      --pipe1-model-path models/pipe1.obj \
      --pipe2-model-path models/pipe2.obj \
      --mesh-scale-to-meter 0.001

The existing YOLO flags can be added unchanged. YOLO consumes `FrameData.rgb`
and performs its own RGB-to-BGR conversion; camera adapters always output RGB.

## Unit-task-triggered FoundationPose worker

`scripts/run_task_supervisor.py` is a lightweight ROS2 process that does not
import YOLO, FoundationPose, or CUDA. It subscribes to
`/recog/unit_task/result` (`std_msgs/msg/Int8`), starts the dual-object worker
once when task 4 begins, keeps it running through task 5, and terminates the
worker process group when task 6 begins. Terminating the worker releases its
CUDA context and GPU memory while the ZED publisher and supervisor remain
alive. If task messages disappear while the worker is active, the default
two-second watchdog also stops it.

Start this supervisor inside the FoundationPose container before starting the
unit-task publisher. Arguments after `--` are the unchanged worker command:

    source /opt/ros/foxy/setup.bash
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DOMAIN_ID=8
    export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/ros2/fastdds_udp.xml"

    /usr/bin/python3 scripts/run_task_supervisor.py \
      --task-topic /recog/unit_task/result \
      --start-task-id 4 \
      --stop-task-id 6 \
      -- \
      /opt/conda/envs/my/bin/python scripts/test_multi_object.py \
        --foundationpose-root /home/rico/Pipe_Align_kkb/FoundationPose \
        --camera-type ros_zed \
        --ros-color-topic /cam/color/compressed \
        --ros-depth-topic /cam/depth/compressed \
        --ros-camera-info-topic /cam/color/camera_info \
        --pipe1-model-path models/pipe1.obj \
        --pipe2-model-path models/pipe2.obj \
        --mesh-scale-to-meter 0.001 \
        --segmentation-mode yolo \
        --yolo-model-path models/yolo/pipe_seg.pt \
        --yolo-confidence 0.5 \
        --yolo-class-id 0 \
        --yolo-device cuda:0 \
        --z-axis-stabilization

This dual-worker command intentionally does not guess a robot-hand identity.
After verifying which registration mask belongs to the left-hand grasped Pipe,
append either `--left-pipe-object-id pipe1` or
`--left-pipe-object-id pipe2` to enable `/vision/left_pipe/pose`.

Keep the host-side ZED publisher running with `zed_cam`. After the supervisor
reports that it is waiting on the task topic, run the dummy publisher on the
host in the same ROS domain:

    cd /home/rico/GT_unit_task_260731/unit_task_dummy
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DOMAIN_ID=8
    /usr/bin/python3 dummy_unit_task_publisher.py

Do not add `--show-auto-mask` to an unattended worker command because its
blocking confirmation window can consume the task-4 interval. Repeated task-4
or task-6 messages do not repeatedly start or stop the worker. A supervisor
that first connects during task 5 waits for the next task-4 transition rather
than starting tracking without the intended registration trigger.

## Adding another camera

Put the vendor-specific implementation in another sibling package. Add a
matching module under `camera/` with a `CameraSource` adapter and a
`create_source(config)` hook, then select it with `--camera-type`. The
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
      --mesh-scale-to-meter 0.001

`run_tracking.py` continues to expose the verified single-object path. Actual
dual-object D405 accuracy and performance must be checked with both physical
objects present.

ROS2 is not imported by the core or by non-ROS camera runs. The `ros_zed`
adapter imports it lazily only when started and normalizes the existing
publisher interface into the same `FrameData` contract. ROS ZED runs now
publish the final processed LEFT Pipe `C_T_P` as
`/vision/left_pipe/pose` (`geometry_msgs/msg/PoseStamped`) and tracking state as
`/vision/left_pipe/tracking_status` (`std_msgs/msg/String`). Dual-object runs
must explicitly map `pipe1` or `pipe2` with `--left-pipe-object-id`; those IDs
do not intrinsically mean left/right robot hand. See `ros2/README.md` for pose
semantics, QoS, validation, and Control-PC DDS checks, and
`../ZED2i_ROS/README.md` for the host/container camera boundary.

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
