# 테스트

모든 명령은 애플리케이션 루트에서 실행한다.

```bash
cd /home/rico/Pipe_Align_kkb/MULTI_OBJECT_TRACKING
```

## ROS ZED 애플리케이션 테스트

이 테스트들은 mock을 사용하므로 카메라와 GPU가 필요하지 않다.

```bash
/usr/bin/python3 -m pytest -q \
  tests/test_cam_ros_zed2i.py \
  tests/test_ros_pose_publisher.py \
  tests/test_multi_object_flow.py \
  tests/test_task_supervisor.py \
  tests/test_yolo_segmenter.py
```

## 전체 CPU 테스트

먼저 sibling 카메라 패키지를 설치하거나 아래와 같이 소스 디렉터리를
일시적으로 노출한다. 아래 방식은 Python 환경을 변경하지 않는다.

```bash
PYTHONPATH="$PWD/../RealSenseD405/src:$PWD/../ZED2iCamera/src" \
  /usr/bin/python3 -m pytest -q
```

## 파일 하나 또는 테스트 하나 실행

```bash
/usr/bin/python3 -m pytest -q tests/test_task_supervisor.py
/usr/bin/python3 -m pytest -q \
  tests/test_task_supervisor.py::test_worker_preloads_at_three_activates_at_four_and_stops_at_six
```

출력 내용을 보려면 `-s`, 자세한 테스트 이름을 보려면 `-vv`를 사용한다.

```bash
/usr/bin/python3 -m pytest -s -vv tests/test_yolo_segmenter.py
```

## 파일별 검증 범위

| 파일 | 검증 범위 |
|---|---|
| `test_cam_ros_zed2i.py` | ROS 이미지 디코딩, 정확한 timestamp 동기화, 프레임 검증 |
| `test_ros_pose_publisher.py` | Pose 검증과 ROS pose/status 메시지 생성 |
| `test_multi_object_flow.py` | 이중 객체 설정, tracker 격리, mask 및 overlay |
| `test_task_supervisor.py` | Task 3 preload, task 4 활성화, task 6 종료 및 timeout 처리 |
| `test_yolo_segmenter.py` | YOLO 필터링, 좌우 객체 배정, mask 정규화 |
| `test_cam_d405.py` | 선택 사항인 D405 adapter 계약 |
| `test_cam_zed2i.py` | 선택 사항인 ZED SDK 직접 연결 adapter 계약 |

이 테스트들은 CPU 수준의 회귀 테스트다. 실제 ROS 전송, 카메라 정확도,
FoundationPose GPU 추론 및 task 지연시간은 측정하지 않는다.
