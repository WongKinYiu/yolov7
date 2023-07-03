import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch    
import cv2
import numpy as np
import os
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM, GradCAM, AblationCAM
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
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget, FasterRCNNBoxScoreTarget

import glob
from torch.autograd import Variable






def eigencam(model,dataset,source,gtruths,ciou=True,imgsz=480,gen_image=False,img_save_path=None,img_name="default"):
    rgb_img=np.array(Image.open(source))
    print(gtruths)
    # exit()
    #load gtruth
    gtruth=open(gtruths,"r")
    gtruth_info=gtruth.read()
    gtruth=gtruth_info.replace('\n',"").split(" ")
    coords_list=gtruth[1:]
    coords_list=np.float32(coords_list)*480
    tcoords=torch.tensor(coords_list)
    tcoords=xywh2xyxy(tcoords.unsqueeze(0))
    
    coords=[]
    for coord in tcoords.squeeze(0):
        coords.append(torch.tensor(coord))

    # print("gtruth",coords,type(coords))
    names = model.module.names if hasattr(model, 'module') else model.names
    COLORS = np.random.uniform(0, 255, size=(80, 3))
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    for path, img, im0s, vid_cap in dataset:
        cam_img=img
        img = torch.from_numpy(img).float()# uint8 to fp16/32
        img=torch.divide(img,255.0) 

        # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

    rgb_img=cv2.resize(rgb_img,(img.shape[3],img.shape[2]))
    rgb_img = np.float32(rgb_img) / 255

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size

    model=model.cuda()
    img=img.cuda()
    model.eval()

      # Calculating gradients would cause a GPU memory leak
    pred = model(img)
    # print(type(pred))
    # print(pred)
    pred = non_max_suppression(pred[0], 0.5, 0.7,classes=[0,1])
    # print(pred)
    
    # print(type(pred))
    # print(pred)
    for _, det in enumerate(pred):
        for *xyxy, conf, cls in reversed(det):
            print("class",cls)
        # print(type(xyxy))

    target_layers = [model.model[1]]
    if ciou:
        print("USING CIOU")
        
    else:
        print("USING NWD")
        # target_layers = [model.model[1]]
   

    targets = [ClassifierOutputTarget(1)]
    # print(xyxy)
    # print(pred[0][0:3])
    # exit()
    # targets = [FasterRCNNBoxScoreTarget(labels=cls, bounding_boxes=coords)]
    # targets = FasterRCNNBoxScoreTarget(labels=cls, bounding_boxes=coords)
    # print(targets(xyxy))
    # exit()
    
    # classes=['0','1']
    # targets=[0]
    # img=img.sque+eze(0)
    
    # exit()
    cam = EigenCAM(model, target_layers, use_cuda=True)
    # cam = GradCAM(model, target_layers, use_cuda=True)
    # cam = AblationCAM(model, target_layers, use_cuda=True)
    # print(type(img))
    # print(img)
    # exit()
    # img.requires_grad=True
    # print(img,img.shape)

    # print(target_layers)
    # exit()
    # img= Variable(img, requires_grad=True)
    # print(img)
    # exit()
    # print(img)
    # preprocess_image(img)

    # renormalize images between -1, 1
    img = cam_img.transpose((1,2,0))
    # print(img.shape)
    img = np.float32(img) / 255
    # print(img.shape)
    mean = tuple(np.mean(img, axis=(0,1)))
    std = tuple(np.std(img, axis=(0,1)))
    # print(mean,std)
    img=preprocess_image(img,mean=mean,std=std)
    # print(test)
    # exit()
    grayscale_cam = cam(input_tensor=img, targets=targets,)
    grayscale_cam = grayscale_cam[0, :]
    inverse_catch=grayscale_cam.mean()
    # print("CAM", grayscale_cam)
    df=pd.DataFrame(grayscale_cam)
    # df.to_csv("nwdattr.csv")
    #get gtruth attribute from the cam
    x1,y1,x2,y2 = coords
    # print("xys",x1,y1,x2,y2)
    x1 ,y1 ,x2 ,y2 =int(x1), int(y1), int(x2), int(y2)

    # exit()
    # print("full cam attribution: sum: ",grayscale_cam.sum(),"average: ",grayscale_cam.mean())
    gtruth_attributes = grayscale_cam[y1:y2, x1:x2]
    # print("gtruth cam attribution : sum: ", gtruth_attributes.sum(),"average: ",gtruth_attributes.mean())
    if gen_image:
        im0=Image.open(source).convert('L')
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        try:
            plot_one_box(xyxy, cam_image, color = (255,0,0) ,line_thickness=1)
            plot_one_box(coords, cam_image, color = (0,255,0) ,line_thickness=1)
        except:
            print("no prediction to print")
        Image.fromarray(cam_image).save(img_save_path +"/{}_cam.jpg".format(img_name))
        if inverse_catch >= 0.2:
            Image.fromarray(cam_image).save("cam_exp/eigen_cam_upq/inverse_catcher/{}_cam.jpg".format(img_name))
            
        # Image.fromarray(cam_image).save(img_save_path +"/camout.jpg")
    quantile_range= np.quantile(gtruth_attributes,[0.90,1.00])
    upper_quantile=gtruth_attributes[(gtruth_attributes >= quantile_range[0]) & (gtruth_attributes <= quantile_range[1])]
    # print("upper",upper_quantile)
    upper_quantile_avg=upper_quantile.mean()
    print("upper quartile average: ",upper_quantile_avg)    

    return upper_quantile_avg



