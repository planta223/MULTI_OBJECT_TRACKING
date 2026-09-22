# ROS2 Pipe pose output

The ROS ZED tracking path publishes the final processed LEFT Pipe pose for the
Control PC without changing FoundationPose inference state.

## Transform contract

`/vision/left_pipe/pose` uses `geometry_msgs/msg/PoseStamped` and means

    C_T_P = Pipe CAD object frame P expressed in camera optical frame C

Position is in meters and orientation is quaternion `x,y,z,w`. No `B_T_C`,
target pose `C_T_P*`, or robot command is applied on the Vision PC.

The authoritative control variable is
`ObjectPoseProcessor.process(...).output_pose`. The flow is:

    synchronized ROS color/depth/CameraInfo
      -> FrameData(rgb, depth_m, K, exact stamp, optical frame_id)
      -> YOLO/manual registration mask
      -> FoundationPose register() / track_one()
      -> PoseResult.pose (raw CAD-object-to-camera pose)
      -> task-symmetry output OR Z-axis stabilization OR raw pass-through
      -> ProcessedPose.output_pose (published C_T_P)

Visualization deliberately computes a different local variable:

    centered_to_camera = output_pose @ inverse(to_origin)

That transform places the oriented bounding-box center and axes for drawing.
It is not published as the Pipe object pose. The text overlay still prints the
translation from `output_pose`.

Before publication, the 4x4 matrix and homogeneous row are checked for finite
values. Rotation determinant and orthogonality are sanity-checked, small
floating-point drift is projected to SO(3) with SVD, and the quaternion is
normalized. A failed check produces `INVALID` status and no pose sample.

## Topics and QoS

| Topic | Type | Meaning |
|---|---|---|
| `/vision/left_pipe/pose` | `geometry_msgs/msg/PoseStamped` | Valid final `C_T_P` only |
| `/vision/left_pipe/tracking_status` | `std_msgs/msg/String` | `REGISTERING`, `TRACKING`, `LOST`, or `INVALID` |

The pose uses `BEST_EFFORT`, `VOLATILE`, `KEEP_LAST(1)`. This favors fresh
feedback and prevents a slow Control subscriber from backing up the GPU
tracking loop. Status uses `RELIABLE`, `TRANSIENT_LOCAL`, `KEEP_LAST(1)` and is
published only on state transitions, allowing a late-joining Control process
to see the current state without adding per-frame reliable traffic.
`rclpy.publish()` only enqueues the small messages; no wait or service call is
added to the loop.

The pose header is copied from the synchronized input messages. The current
`ZED2i_ROS/cam_zed.py` publisher sets all three input headers to
`camera_left_optical_frame`, and the adapter rejects a synchronized triple if
their frame IDs disagree. The integer nanosecond image stamp is preserved
exactly; tracking completion wall time is never substituted. Invalid tracking
does not republish the previous pose with a new timestamp.

The upstream ZED publisher currently stamps the messages with its ROS clock in
`publish_loop()`. Thus this output preserves the input Image stamp exactly, but
that upstream stamp is publish time rather than a ZED SDK hardware-exposure
timestamp. Hardware timestamping, if required later, must be fixed at the
camera publisher so every downstream consumer receives the same basis.

## LEFT/RIGHT identity

The single-object entrypoint is explicitly the LEFT-Pipe path, so its sole
`pipe1` tracker is published automatically when `--camera-type ros_zed` is
used. This contract still depends on its registration mask actually selecting
the left-hand grasped Pipe. If multiple indistinguishable Pipe instances are
visible, single-object YOLO chooses the highest-confidence instance and cannot
infer the grasping hand; use an unambiguous scene/mask rather than assuming the
detection is LEFT.

The dual-object path does **not** infer robot-hand identity. Manual mode assigns
`pipe1` to the first selected mask and `pipe2` to the second. YOLO mode assigns
the two highest-confidence instances from image-left to image-right on the
registration frame. This is initial tracker identity, not left-hand/right-hand
semantics. Therefore dual mode publishes no pose until the operator supplies
one explicit mapping:

    --left-pipe-object-id pipe1

or

    --left-pipe-object-id pipe2

Without it, the status topic reports `INVALID`. A future RIGHT interface can
reuse the same publisher class with `/vision/right_pipe/...` topic names after
an equally explicit assignment is introduced.

## Dependencies and build

This repository is a direct-run Python application, not an ament package; it
contains no `package.xml`, `CMakeLists.txt`, or application `setup.py`.
Consequently this change requires no colcon metadata modification. ROS message
dependencies are standard ROS distribution packages and are imported lazily:

    sudo apt install ros-${ROS_DISTRO}-rclpy \
      ros-${ROS_DISTRO}-geometry-msgs \
      ros-${ROS_DISTRO}-sensor-msgs \
      ros-${ROS_DISTRO}-std-msgs

Build FoundationPose native extensions as before:

    cd /home/rico/Pipe_Align_kkb/FoundationPose
    bash build_all.sh

The Python test command used for the ROS output path is:

    cd /home/rico/Pipe_Align_kkb/MULTI_OBJECT_TRACKING
    /usr/bin/python3 -m pytest -q \
      tests/test_ros_zed_adapter.py \
      tests/test_ros_pose_publisher.py \
      tests/test_multi_object_flow.py \
      tests/test_task_supervisor.py \
      tests/test_yolo_segmenter.py

Use the FoundationPose GPU Python environment for the live process, provided
that environment can import the ROS Python packages.

## Run

For the current single LEFT Pipe scope:

    cd /home/rico/Pipe_Align_kkb/MULTI_OBJECT_TRACKING
    source /opt/ros/foxy/setup.bash
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DOMAIN_ID=8
    export ROS_LOCALHOST_ONLY=0
    export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/ros2/fastdds_udp.xml"

    /opt/conda/envs/my/bin/python run_tracking.py \
      --foundationpose-root /home/rico/Pipe_Align_kkb/FoundationPose \
      --camera-type ros_zed \
      --model-path models/pipe1.obj \
      --mesh-scale-to-meter 0.001 \
      --segmentation-mode yolo \
      --yolo-model-path models/yolo/pipe_seg.pt \
      --yolo-confidence 0.5 \
      --yolo-class-id 0 \
      --yolo-device cuda:0 \
      --z-axis-stabilization

For the existing dual worker, append the verified mapping, for example:

    /opt/conda/envs/my/bin/python scripts/test_multi_object.py \
      --foundationpose-root /home/rico/Pipe_Align_kkb/FoundationPose \
      --camera-type ros_zed \
      --pipe1-model-path models/pipe1.obj \
      --pipe2-model-path models/pipe2.obj \
      --segmentation-mode yolo \
      --yolo-model-path models/yolo/pipe_seg.pt \
      --yolo-class-id 0 \
      --yolo-device cuda:0 \
      --z-axis-stabilization \
      --left-pipe-object-id pipe1

Do not copy the example `pipe1` mapping blindly: verify the registration mask
belongs to the left-hand grasped physical Pipe.

## Local verification

In another shell with the same ROS environment:

    ros2 topic info -v /vision/left_pipe/pose
    ros2 topic echo /vision/left_pipe/tracking_status
    ros2 topic echo /vision/left_pipe/pose --qos-reliability best_effort
    ros2 topic hz /vision/left_pipe/pose

Also compare the output header directly with the image header:

    ros2 topic echo /cam/color/compressed --once --field header
    ros2 topic echo /vision/left_pipe/pose --once --field header \
      --qos-reliability best_effort

The stamps should correspond to input frames (with inference latency between
arrival and publication), and both frame IDs should be
`camera_left_optical_frame` for the current ZED publisher.

Representative output shape (numbers are illustrative only):

    header:
      stamp:
        sec: 1789701234
        nanosec: 123456789
      frame_id: camera_left_optical_frame
    pose:
      position:
        x: 0.118
        y: -0.064
        z: 0.742
      orientation:
        x: 0.012
        y: -0.701
        z: 0.019
        w: 0.713

## Control-PC DDS/network verification

1. Synchronize Vision, camera-host, and Control clocks with chrony/NTP. Stale
   detection based on the preserved source stamp is meaningful only when the
   machines share a clock basis.
2. On both Vision and Control shells set the same `ROS_DOMAIN_ID` (currently
   `8`), `ROS_LOCALHOST_ONLY=0`, and preferably
   `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`.
3. On the Vision container use `ros2/fastdds_udp.xml` as shown above. If the
   Control process is also containerized or has SHM/user-ID issues, copy that
   profile to the Control machine and set its absolute
   `FASTRTPS_DEFAULT_PROFILES_FILE` too.
4. Confirm multicast/unicast reachability and allow DDS UDP traffic through
   host/container firewalls. `ros2 multicast receive` on one PC and
   `ros2 multicast send` on the other is a useful first check.
5. Run `ros2 node list` and `ros2 topic list` on Control. It should discover
   `/foundationpose_ros_zed` and `/vision/left_pipe/pose`.
6. Run `ros2 topic info -v /vision/left_pipe/pose` on Control and verify one
   publisher with compatible BEST_EFFORT QoS, then run `echo` and `hz` there.
7. If discovery is stale after environment changes, run `ros2 daemon stop`
   followed by `ros2 daemon start`, then repeat the checks.

On Control, reject a pose when `now - msg.header.stamp` exceeds the controller's
chosen freshness limit, or when status is not `TRACKING`. Do not infer freshness
from subscriber arrival time alone.
