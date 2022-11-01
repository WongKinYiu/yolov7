import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages, letterbox
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class DetectObjects():
    def __init__(self,weights, imgsz, conf_thres, iou_thres=0.45, device_name='cpu') -> None:
        self.weights     = weights
        self.imgsz       = imgsz
        self.conf_thres  = conf_thres
        self.iou_thres   = iou_thres
        self.device_name = device_name

        # Initialize
        set_logging()
        self.device = select_device(device_name)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check img_size

        self.model = TracedModel(self.model, self.device, self.imgsz)


        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

    
    # Make Prediction and get coordinates of objects
    def predict(self, im0s):
        t0 = time.time()
        result = []

        # Padded resize
        img = letterbox(im0s, self.imgsz, stride=32)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)


        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=0)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=0)[0]
        t2 = time_synchronized()

        # Apply NMS
        # pred = non_max_suppression(pred, conf_thres, iou_thres, classes=opt.classes, agnostic=0)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=0)

        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                
                # Update result variable
                result.append(det.numpy())

            # Print time (inference + NMS)
            print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')


        for row in result[0]:
            plot_one_box((row[0],row[1],row[2],row[3]), im0s, label=self.names[int(row[5])], color=self.colors[int(row[5])], line_thickness=1)
        
        cv2.imshow("result",im0s);cv2.waitKey(0)
        return result






# with torch.no_grad():
#     yolo = DetectObjects('yolov7.pt',640,0.25)

#     img = cv2.imread('./inference/images/horses.jpg')
#     yolo.predict(img)
    
