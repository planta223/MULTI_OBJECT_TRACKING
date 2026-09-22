# Tests

Run commands from the application root:

```bash
cd /home/rico/Pipe_Align_kkb/MULTI_OBJECT_TRACKING
```

## ROS ZED application tests

These tests use mocks and do not require a camera or GPU.

```bash
/usr/bin/python3 -m pytest -q \
  tests/test_ros_zed_adapter.py \
  tests/test_ros_pose_publisher.py \
  tests/test_multi_object_flow.py \
  tests/test_task_supervisor.py \
  tests/test_yolo_segmenter.py
```

## Full CPU test suite

Install the sibling camera packages first, or expose their source directories
temporarily. The latter does not modify the Python environment:

```bash
PYTHONPATH="$PWD/../RealSenseD405/src:$PWD/../ZED2iCamera/src" \
  /usr/bin/python3 -m pytest -q
```

## Run one file or one test

```bash
/usr/bin/python3 -m pytest -q tests/test_task_supervisor.py
/usr/bin/python3 -m pytest -q \
  tests/test_task_supervisor.py::test_worker_preloads_at_three_activates_at_four_and_stops_at_six
```

Use `-s` when printed output is needed and `-vv` for detailed test names:

```bash
/usr/bin/python3 -m pytest -s -vv tests/test_yolo_segmenter.py
```

## Test coverage by file

| File | Scope |
|---|---|
| `test_ros_zed_adapter.py` | ROS image decoding, exact-stamp synchronization, frame validation |
| `test_ros_pose_publisher.py` | Pose validation and ROS pose/status message generation |
| `test_multi_object_flow.py` | Dual-object configuration, tracker isolation, masks and overlays |
| `test_task_supervisor.py` | Task 3 preload, task 4 activation, task 6 stop and timeout handling |
| `test_yolo_segmenter.py` | YOLO filtering, left/right assignment and mask normalization |
| `test_realsense_d405_adapter.py` | Optional D405 adapter contract |
| `test_zed2i_adapter.py` | Optional direct ZED SDK adapter contract |

These are CPU-level regression tests. They do not measure live ROS transport,
camera accuracy, FoundationPose GPU inference, or task latency.
