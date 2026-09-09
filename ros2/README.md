# ROS2 integration

ROS2 is intentionally not a dependency of the tracking core. A future
`rclpy` node may replace the orchestration performed by `run_tracking.py`,
while reusing the camera adapter, segmenter, tracking manager, pose processing,
and result contracts unchanged. Do not import `rclpy` from those core layers.
