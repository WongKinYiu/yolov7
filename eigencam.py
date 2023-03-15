import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch    
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from models.yolo import Model
from utils.datasets import LoadStreams, LoadImages
import random
import pandas
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget




weights="weights/dyvir_weights/ciou_dyvir_ft.pt"
cfg="cfg/training/yolov7-tiny-drone.yaml"
source = "tchparkdronepic.png"
imgsz=480

rgb_img=np.array(Image.open(source).convert('RGB'))

# print(rgb_img.shape)
# exit()

model = attempt_load(weights)
dataset = LoadImages(source, img_size=imgsz)





  # load FP32 model
names = model.module.names if hasattr(model, 'module') else model.names
COLORS = np.random.uniform(0, 255, size=(80, 3))
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
# print(dataset)
# print(names)
# print(colors)

for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).float()# uint8 to fp16/32
    print(type(img))
    # exit()
    # img=cv2.resize(img,(imgsz,imgsz))
    img=torch.divide(img,255.0) 
  
    # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
# print(img.shape)
# exit()
rgb_img=cv2.resize(rgb_img,(img.shape[3],img.shape[2]))
rgb_img = np.float32(rgb_img) / 255
# exit()

gs = max(int(model.stride.max()), 32)  # grid size (max stride)
imgsz = check_img_size(imgsz, s=gs)  # check img_size
model.eval()

# print(model)
print("layer being visualized:", model.model)
print("layer being visualized:", model.model[-77])
# print(model.model.IDetect[-1])


# for name,params in model.named_parameters():
#     print(name)


# exit()
with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
    pred = model(img)

# pred = non_max_suppression(pred[0], 0.5, 0.7,classes=[0,1])
print(img.shape)
# exit()

target_layers = [model.model[1]]
targets = [ClassifierOutputTarget(1)]
# img=img.squeeze(0)
cam = EigenCAM(model, target_layers, use_cuda=False)
print(type(img))
print(img)
exit()
grayscale_cam = cam(input_tensor=img,targets=targets,eigen_smooth=True,aug_smooth=False)
grayscale_cam = grayscale_cam[0, :]
# transform = transforms.ToTensor()
im0=Image.open(source).convert("L")
# tensor = transform(im0).unsqueeze(0)
# print("img",rgb_img.shape)
# print("cam",grayscale_cam.shape)


cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
Image.fromarray(cam_image).save("eigenout.png")

 

