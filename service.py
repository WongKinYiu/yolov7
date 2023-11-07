from pathlib import Path

import bentoml
import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
from bentoml.io import Image, PandasDataFrame
from models.experimental import attempt_load
from numpy import random
from utils.datasets import letterbox
from utils.general import (check_img_size, non_max_suppression, scale_coords,
                           xyxy2xywh)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


class Yolov7Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):

        self.conf_thres = 0.25
        self.iou_thres = 0.45
        img_size = 640

        current_dir = os.path.dirname(os.path.abspath(__file__))

        best_pt_files = glob.glob(os.path.join(current_dir, '**', 'best.pt'), recursive=True)

        if best_pt_files:
            best_pt_path = best_pt_files[0]
        else:
            raise Exception("model best.pt file not found")
        weights = os.path.join(best_pt_path)

        self.device = select_device("0")
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(img_size, s=self.stride)

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != "cpu":
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        print("Initialize Done.")

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, input_imgs):
        # Return predictions only
        result = {"xyxy": [], "render": None}
        numpy_image = np.array(input_imgs[0])
        img0 = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        path = ""
        dataset = [(path, img, img0, None)]
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf)
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        record = (("%g " * len(line)).rstrip() % line).split(" ")
                        record.append(self.names[int(cls)])
                        result["xyxy"].append(record)
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                # Change processed image from BGR to RBG
                result["render"] = im0[:, :, ::-1].transpose(0, 1, 2)
                # Print time (inference + NMS)
                print(f"{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS")

        return [result]


yolo_v7_runner = bentoml.Runner(Yolov7Runnable, max_batch_size=30)

svc = bentoml.Service("yolo_demo", runners=[yolo_v7_runner])


@svc.api(input=Image(), output=PandasDataFrame())
async def invocation(input_img):
    columns = ["class", "xmin", "ymin", "xmax", "ymax", "confidence", "name"]
    batch_ret = await yolo_v7_runner.inference.async_run([input_img])
    xyxy = batch_ret[0].get("xyxy", [])

    df = pd.DataFrame(data=xyxy, columns=columns)
    df[["xmin", "ymin", "xmax", "ymax", "confidence"]] = df[["xmin", "ymin", "xmax", "ymax", "confidence"]].astype(float)

    return df


@svc.api(input=Image(), output=Image())
async def render(input_img):
    batch_ret = await yolo_v7_runner.inference.async_run([input_img])
    img = batch_ret[0].get("render", None)
    return img
