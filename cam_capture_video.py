#!/usr/bin/python3

"""
    This sample demonstrate the continuously image capturing of the build-in Basler and Appropho camera.
    The basler camera python API used is called pypylon and the official package github url is https://github.com/basler/pypylon
"""

import logging
import os
import sys
from pypylon import pylon
import cv2
import torch
import torch.nn as nn
import tensorrt as trt
from collections import namedtuple, OrderedDict
import numpy as np
from utils.datasets import letterbox, create_dataloader
from utils.general import non_max_suppression, coco80_to_coco91_class, clip_coords, xywh2xyxy, box_iou, scale_coords
from utils.torch_utils import time_synchronized
from utils.metrics import ap_per_class
from tqdm import tqdm
import random
import argparse
import yaml

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def postprocess(boxes,r,dwdh):
    dwdh = torch.tensor(dwdh*2).to(boxes.device)
    boxes -= dwdh
    boxes /= r
    return boxes


class TrtModel(nn.Module):
    def __init__(self, weights='yolor_csp.engine', device=torch.device(0)):
        super().__init__()
        self.device = device
        print(f"Loading {weights} for TensorRT inference...")
        Binding = namedtuple('Binding',('name','dtype','shape','data','ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        self.fp16 = False
        self.dynamic = False
        for index in range(self.model.num_bindings):
            name = self.model.get_binding_name(index)
            dtype = trt.nptype(self.model.get_binding_dtype(index))
            if self.model.binding_is_input(index):
                if -1 in tuple(self.model.get_binding_shape(index)):
                    self.dynamic = True
                    self.context.set_binding_shape(index, tuple(model.get_profile_shape(0, index)[2]))
                if dtype == np.float16:
                    self.fp16 = True
            shape = tuple(self.context.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.batch_size = self.bindings['images'].shape[0]
    def forward(self, im):
        b, ch, h, w = im.shape
        if self.dynamic and im.shape != self.bindings['images'].shape:
            i_in, i_out = (self.model.get_binding_index(x) for x in ('images', 'output'))
            self.context.set_binding_shape(i_in, im.shape)  # reshape if dynamic
            self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
            self.bindings['output'].data.resize_(tuple(self.context.get_binding_shape(i_out)))
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings['output'].data

        if(isinstance(y, np.ndarray)):
            y = torch.tensor(y, device=self.device)
        return y


names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush']

colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

#---------------------------Parameters------------------------------------
#Width=800
#Height=600
#Gain=10
#ExposureTime=3500
#FrameRate=30
#-------------------------------------------------------------------

def main():
    log.info("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    
    # --------Basler camera--------------------       
    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    log.info("Starting initialize Basler camera...")
    camera.Open()

    #Setting image size
    #camera.Width.Value=Width
    #camera.Height.Value=Height
    
    #Setting Gain
    #camera.GainAuto.SetValue("Off") 
    #camera.Gain=Gain
    
    #Setting Exposure Time
    #camera.ExposureAuto.SetValue("Off")
    #camera.ExposureTime.SetValue(ExposureTime)    

    #Setting Frame Rate
    #camera.AcquisitionFrameRate.SetValue(FrameRate)  

    # Grabing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    log.info("Starting cature image...")        
    
    img_counter = 0
    model = TrtModel('yolov7.engine')

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                image = image.GetArray()
                t = time_synchronized()
                img = image.copy()
                img, ratio, dwdh = letterbox(image, stride=64, auto=False)
                img = img.transpose((2,0,1))
                img = np.expand_dims(img, 0)
                img = np.ascontiguousarray(img)
                im = torch.tensor(img.astype(np.float32), device=torch.device(0))
                im /= 255

                
                output = model(im)
                output = non_max_suppression(output, conf_thres=0.25, iou_thres=0.45)[0]
                t = time_synchronized() - t
                cv2.putText(image, "FPS: " + str(1/t), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), thickness=2)
                for i, bbox in enumerate(output):
                    cls = bbox[5].int()
                    conf = bbox[4]
                    box = bbox[:4]
                    name = names[cls]
                    color = colors[name]
                    box = postprocess(box, ratio, dwdh).round().int()
                    cv2.rectangle(image,box[:2].tolist(),box[2:].tolist(),color,2)
                    cv2.putText(image,name,(int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX,0.75,color,thickness=2)
                    
                
                cv2.imshow("Preview of Basler Camera --- Exit by press 'ESC' key", image)
		
                k = cv2.waitKey(1)
                if k == 27:
                        break
                elif k == 32:
                        img_name = "opencv_frame_{}.png". format(img_counter)
                        cv2.imwrite(img_name, image)
                        print("{} written!".format(img_name))
                        img_counter += 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)
