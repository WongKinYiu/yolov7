import argparse
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def yolov7_inference(weights, img, img_size=640, conf_thres=0.25, iou_thres=0.45, device='', classes=None, agnostic_nms=False, augment=False, no_trace=False):
    save_txt = False  # Adjust as needed
    save_conf = False  # Adjust as needed

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA


    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size

    if half:
        model.half()

    # print(f"Input Image Shape: {img.shape}")

    im0 = img
    # Image preprocessing
    img = letterbox(im0, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # print(f"Processed Image Shape: {img.shape}")

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Warmup
    if device.type != 'cpu':
        old_img_b, old_img_h, old_img_w = img.shape[0], img.shape[2], img.shape[3]
        for i in range(3):
            model(img, augment=augment)[0]

    # Inference
    t1 = time_synchronized()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t3 = time_synchronized()

    results = []

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        if len(det):
            # Rescale boxes from img_size to original image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            det_boxes = []
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                # print('xywh ',xywh)
                line = (cls, *xywh)  # label format
                # Convert each element to a PyTorch tensor and move it to CPU
                line_tensor = tuple(torch.tensor(elem).cpu().numpy() if isinstance(elem, torch.Tensor) else elem for elem in line)
                det_boxes.append(np.array(line_tensor))

    
    return det_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--img-array', type=np.ndarray, help='image numpy array')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    # print(opt)