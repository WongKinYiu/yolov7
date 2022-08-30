import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import os
from tqdm import tqdm
import time

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

WEIGHTS = "/home/zoomi2022/yolov7/yolov7-w6-pose.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 640  # Detection size
print('Device', DEVICE)
model = attempt_load(WEIGHTS, DEVICE)

def predict_keypoints(image, image_size=640, conf_thresh=0.25, iou_thresh=0.65):
    image = np.asarray(image)
    
    # Resize image to the inference size
    ori_h, ori_w = image.shape[:2]
    image = cv2.resize(image, (image_size, image_size))
    
    # Transform image from numpy to torch format
    image_pt = torch.from_numpy(image).permute(2, 0, 1).to(DEVICE)
    image_pt = image_pt.float() / 255.0
    
    # Infer
    with torch.no_grad():
        pred = model(image_pt[None], augment=False)[0]
    
    # NMS
    pred = non_max_suppression_kpt(
        pred, conf_thresh, iou_thresh, 
        nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True
    )
    pred = output_to_keypoint(pred)
    
    # Drop
    pred = pred[:, 7:]
    
    # Resize boxes to the original image size
    pred[:, 0::3] *= ori_w / image_size
    pred[:, 1::3] *= ori_h / image_size
    
    return pred
