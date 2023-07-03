import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch    
import cv2
import numpy as np
import os
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, xywh2xyxy
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from models.yolo import Model
from utils.datasets import LoadStreams, LoadImages
import random
import pandas as pd
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from cam_utils import eigencam


gpu                     = True
gpu_number              = '4'

# Declare GPUS
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number


# ciou=True

cfg="cfg/training/yolov7-tiny-drone.yaml"

# label_info=pd.read_csv("rw_test2.csv")
# label_info=label_info.sort_values(
# label_info=label_info.reset_index(drop=True)
# data_dir="../../../data/Koutsoubn8/ijcnn_v7data/Real_world_test2"
ciou_save_path="cam_exp/grad_cam/ciou_cam_out"
nwd_save_path="cam_exp/grad_cam/nwd_cam_out"
# paths=label_info["path"]
data_name="grad_test"
imgsz=480

gen_image=True

#load ciou model
ciou_weights="weights/dyvir_weights/ciou_dyvir_ft.pt"
ciou_model = attempt_load(ciou_weights)
#load nwd model
nwd_weights="weights/dyvir_weights/nwd_dyvir_ft.pt"
nwd_model = attempt_load(nwd_weights)

ciou_avg_attrs=[]
nwd_avg_attrs=[]
source = "../../../data/Koutsoubn8/ijcnn_v7data/Real_world_test2/images/V_DRONE_003_220.jpg"
gtruths = "../../../data/Koutsoubn8/ijcnn_v7data/Real_world_test2/labels/V_DRONE_003_220.txt"
# gtruths = source[:-3] + "txt"
dataset = LoadImages(source, img_size=imgsz)

print("===================================================================")
ciou=True
ciou_avg_attr=eigencam(ciou_model,dataset, source, gtruths,ciou,imgsz,gen_image,ciou_save_path,img_name=data_name)
ciou_avg_attrs.append(ciou_avg_attr)
    # get nwd cam
print("===================================================================")
ciou=False
nwd_avg_attr=eigencam(nwd_model,dataset, source, gtruths,ciou,imgsz,gen_image,nwd_save_path,img_name=data_name)
nwd_avg_attrs.append(nwd_avg_attr)
# exit()