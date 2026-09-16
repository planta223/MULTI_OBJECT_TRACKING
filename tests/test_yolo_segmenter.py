"""CPU-only tests for YOLO initial-mask selection and normalization."""

from types import SimpleNamespace

import numpy as np
import pytest

from camera.base import FrameData
from constants import PIPE1_ID, PIPE2_ID
from segmentation.yolo import YoloSegmenter


def make_frame(height: int = 8, width: int = 10) -> FrameData:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[0, 0] = [10, 20, 30]
    return FrameData(
        source_frame_id=1,
        rgb=rgb,
        depth_m=np.ones((height, width), dtype=np.float32),
        K=np.eye(3, dtype=np.float64),
        device_timestamp_ms=None,
        timestamp_domain=None,
        host_wall_time_s=1.0,
        host_monotonic_time_s=2.0,
    )


class FakeModel:
    def __init__(self, detections):
        self.detections = detections
        self.last_predict_args = None

    def predict(self, **kwargs):
        self.last_predict_args = kwargs
        boxes = SimpleNamespace(
            xyxy=np.asarray([item[0] for item in self.detections], dtype=np.float32),
            conf=np.asarray([item[1] for item in self.detections], dtype=np.float32),
            cls=np.asarray([item[2] for item in self.detections], dtype=np.float32),
        )
        masks = SimpleNamespace(
            data=np.asarray([item[3] for item in self.detections], dtype=np.float32)
        )
        return [SimpleNamespace(boxes=boxes, masks=masks)]


def detection(center_x, confidence, fill_column, class_id=0, mask_shape=(8, 10)):
    mask = np.zeros(mask_shape, dtype=np.float32)
    mask[:, fill_column] = 1.0
    return (
        [center_x - 1, 0, center_x + 1, mask_shape[0] - 1],
        confidence,
        class_id,
        mask,
    )


def build_segmenter(detections, object_ids=(PIPE1_ID, PIPE2_ID), **kwargs):
    model = FakeModel(detections)
    segmenter = YoloSegmenter(
        object_ids=object_ids,
        model_path=None,
        confidence=kwargs.pop("confidence", 0.5),
        class_id=kwargs.pop("class_id", 0),
        model=model,
        **kwargs,
    )
    return segmenter, model


@pytest.mark.parametrize(
    "detections",
    [
        [detection(6, 0.9, 6), detection(2, 0.8, 2)],
        [detection(2, 0.8, 2), detection(6, 0.9, 6)],
    ],
)
def test_dual_instances_are_assigned_left_to_right_regardless_of_input_order(
    detections,
):
    frame = make_frame()
    segmenter, _ = build_segmenter(detections)

    masks = segmenter.segment(frame)

    assert masks[PIPE1_ID][:, 2].all()
    assert masks[PIPE2_ID][:, 6].all()


def test_confidence_threshold_excludes_detection_and_reports_shortage():
    frame = make_frame()
    segmenter, _ = build_segmenter(
        [detection(2, 0.9, 2), detection(6, 0.49, 6)],
        confidence=0.5,
    )

    with pytest.raises(RuntimeError, match=r"found 1 valid pipe instance.*2 object"):
        segmenter.segment(frame)


def test_top_two_confidences_are_selected_before_left_right_assignment():
    frame = make_frame()
    segmenter, _ = build_segmenter(
        [
            detection(1, 0.55, 1),
            detection(7, 0.95, 7),
            detection(4, 0.85, 4),
        ]
    )

    masks = segmenter.segment(frame)

    assert masks[PIPE1_ID][:, 4].all()
    assert masks[PIPE2_ID][:, 7].all()
    assert not masks[PIPE1_ID][:, 1].any()


def test_single_object_uses_highest_confidence_detection():
    frame = make_frame()
    segmenter, _ = build_segmenter(
        [detection(2, 0.7, 2), detection(6, 0.9, 6)],
        object_ids=(PIPE1_ID,),
    )

    masks = segmenter.segment(frame)

    assert masks[PIPE1_ID][:, 6].all()


def test_mask_is_resized_to_frame_as_bool_and_rgb_is_converted_once_to_bgr():
    frame = make_frame()
    segmenter, model = build_segmenter(
        [detection(1, 0.9, 1, mask_shape=(4, 5))],
        object_ids=(PIPE1_ID,),
    )

    masks = segmenter.segment(frame)

    assert masks[PIPE1_ID].shape == (frame.height, frame.width)
    assert masks[PIPE1_ID].dtype == np.bool_
    assert model.last_predict_args["source"][0, 0].tolist() == [30, 20, 10]
    assert model.last_predict_args["retina_masks"] is True


def test_class_filter_is_applied_before_assignment():
    frame = make_frame()
    segmenter, _ = build_segmenter(
        [detection(2, 0.99, 2, class_id=1), detection(6, 0.8, 6, class_id=0)],
        object_ids=(PIPE1_ID,),
        class_id=0,
    )

    masks = segmenter.segment(frame)

    assert masks[PIPE1_ID][:, 6].all()


def test_detection_only_result_is_rejected():
    frame = make_frame()

    class DetectionOnlyModel:
        def predict(self, **kwargs):
            del kwargs
            boxes = SimpleNamespace(
                xyxy=np.asarray([[1, 1, 2, 2]], dtype=np.float32),
                conf=np.asarray([0.9], dtype=np.float32),
                cls=np.asarray([0], dtype=np.float32),
            )
            return [SimpleNamespace(boxes=boxes, masks=None)]

    segmenter = YoloSegmenter(
        (PIPE1_ID,),
        model_path=None,
        model=DetectionOnlyModel(),
    )

    with pytest.raises(RuntimeError, match="instance segmentation model"):
        segmenter.segment(frame)


def test_zero_detections_are_reported_clearly():
    frame = make_frame()

    class EmptyModel:
        def predict(self, **kwargs):
            del kwargs
            boxes = SimpleNamespace(
                xyxy=np.empty((0, 4), dtype=np.float32),
                conf=np.empty((0,), dtype=np.float32),
                cls=np.empty((0,), dtype=np.float32),
            )
            return [SimpleNamespace(boxes=boxes, masks=None)]

    segmenter = YoloSegmenter(
        (PIPE1_ID,),
        model_path=None,
        model=EmptyModel(),
    )

    with pytest.raises(RuntimeError, match="found no detections"):
        segmenter.segment(frame)


def test_empty_mask_is_rejected():
    frame = make_frame()
    empty = np.zeros((8, 10), dtype=np.float32)
    segmenter, _ = build_segmenter(
        [([1, 1, 3, 3], 0.9, 0, empty)],
        object_ids=(PIPE1_ID,),
    )

    with pytest.raises(ValueError, match="empty instance mask"):
        segmenter.segment(frame)
