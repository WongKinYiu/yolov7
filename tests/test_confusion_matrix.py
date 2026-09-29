import torch

from utils.general import box_iou
from utils.metrics import ConfusionMatrix


def test_a_confused_class_keeps_predicted_and_true():
    labels = torch.tensor([[0, 0, 0, 10, 10]])
    detections = torch.tensor([[0, 0, 10, 10, 0.9, 1]])
    assert float(box_iou(labels[:, 1:], detections[:, :4])) == 1

    matrix = ConfusionMatrix(nc=2, conf=0.25, iou_thres=0.45)
    matrix.process_batch(detections, labels)

    assert matrix.matrix[1, 0] == 1
    assert matrix.matrix[0, 1] == 0


def test_a_matching_class_stays_on_the_diagonal():
    matrix = ConfusionMatrix(nc=2, conf=0.25, iou_thres=0.45)
    matrix.process_batch(
        torch.tensor([[0, 0, 10, 10, 0.9, 0]]),
        torch.tensor([[0, 0, 0, 10, 10]]),
    )
    assert matrix.matrix[0, 0] == 1
