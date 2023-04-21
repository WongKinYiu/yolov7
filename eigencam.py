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
import glob

colors = np.random.uniform(0, 255, size=(80, 3))

def renormalize_cam_in_bounding_boxes(boxes, colors,image_float_np, grayscale_cam,labels=None):
    print("boxes",boxes)
    # exit()
    boxes_tensor=boxes
    x1,y1,x2,y2 = boxes
    x1 ,y1 ,x2 ,y2 =int(x1), int(y1), int(x2), int(y2)

    boxes=[]
    boxes.append((x1,y1,x2,y2))
    # exit()
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        # print(grayscale_cam[y1:y2, x1:x2])
        print(grayscale_cam[y1:y2, x1:x2].shape)
        
        print(x1,y1,x2,y2)
        exit()
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())     
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    Image.fromarray(eigencam_image_renormalized).save("eignenorm__preout.png")
    print(boxes_tensor)
    plot_one_box(boxes_tensor, eigencam_image_renormalized, color = (255,0,0) ,line_thickness=1)
    # Image.fromarray(image_with_bounding_boxes).save("eignenorm__preout.png")
    return eigencam_image_renormalized

ciou=False
# ciou=True

cfg="cfg/training/yolov7-tiny-drone.yaml"
# source="../../../data/Koutsoubn8/ijcnn_v7data/small_data/rw_16x16/images/V_DRONE_010_264.jpg"
# gtruth = source[0:-4] +".txt"
# gtruth="../../../data/Koutsoubn8/ijcnn_v7data/small_data/rw_16x16/labels/V_DRONE_010_264.txt"

if ciou:
    # print("USING CIOU")
    weights="weights/dyvir_weights/ciou_dyvir_ft.pt"
else:
    # print("USING NWD")
    weights="weights/dyvir_weights/nwd_dyvir_ft.pt"

label_info=pd.read_csv("eigenset.csv")
# label_info=label_info.sort_values(
# label_info=label_info.reset_index(drop=True)
data_dir="../../../data/Koutsoubn8/ijcnn_v7data/Real_world_test2"
paths=label_info["path"]
img_name=label_info["name"]

# print(source)
# print(gtruths)
# exit()

# source=glob.glob("../../../data/Koutsoubn8/eigen_out/images/*.jpg")
# gtruths=glob.glob("../../../data/Koutsoubn8/eigen_out/labels/*.txt")
# source.sort()
# gtruths.sort()
# print(gtruth)
# print(gtruths[1])
# exit()


attr_avgs=[]
attr_sums=[]
names=[]
imgsz=480
for i, _ in enumerate(paths):
# for i in range(5):
    # source ="../../../data/Koutsoubn8/" +paths[i][:-4] + ".jpg"
    # gtruths = "../../../data/Koutsoubn8/" + paths[i]
    source = data_dir + "/images/" + img_name[i] +".jpg"
    gtruths = data_dir + "/labels/" + img_name[i] +".txt"
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
    # print(coords)
    # exit()
    coords=[]
    for coord in tcoords.squeeze(0):
        coords.append(torch.tensor(coord))

    # print("gtruth",coords,type(coords))

    # coords=xywh2xyxy(coords)
    print("gtruth",coords,type(coords))
    # exit()
    # print(rgb_img.shape)
    # exit()

    model = attempt_load(weights)
    print(source)
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
        # print(type(img))
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

    model=model.cuda()
    img=img.cuda()

    model.eval()

    # print(model)
    # print("layer being visualized:", model.model)
    # print("layer being visualized:", model.model[-77])
    # print(model.model.IDetect[-1])


    # for name,params in model.named_parameters():
    #     print(name)


    # exit()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img)
    print(type(pred))
    # print(pred)
    pred = non_max_suppression(pred[0], 0.5, 0.7,classes=[0,1])
    # print(type(pred))
    # print(pred)
    for _, det in enumerate(pred):
        for *xyxy, conf, cls in reversed(det):
            print("xyxy",xyxy)
            # print(type(xyxy))



    # pred = non_max_suppression(pred[0], 0.5, 0.7,classes=[0,1])
    # print(img.shape)
    # exit()


    if ciou:
        print("USING CIOU")
        target_layers = [model.model[1]]
    else:
        print("USING NWD")
        target_layers = [model.model[2]]

    targets = [ClassifierOutputTarget(1)]
    print(ClassifierOutputTarget(1))
    # img=img.squeeze(0)
    cam = EigenCAM(model, target_layers, use_cuda=True)

    # print(type(img))
    # print(img)
    # exit()[rint()
    print(targets)
    exit()
    grayscale_cam = cam(input_tensor=img,targets=targets,eigen_smooth=True,aug_smooth=False)
    grayscale_cam = grayscale_cam[0, :]

    #get gtruth attribute from the cam
    x1,y1,x2,y2 = coords
    # print("xys",x1,y1,x2,y2)
    x1 ,y1 ,x2 ,y2 =int(x1), int(y1), int(x2), int(y2)

    # exit()
    print("full cam attribution: sum: ",grayscale_cam.sum(),"average: ",grayscale_cam.mean())
    gtruth_attributes = grayscale_cam[y1:y2, x1:x2]
    print("gtruth cam attribution : sum: ", gtruth_attributes.sum(),"average: ",gtruth_attributes.mean())
    # exit()
    attr_avgs.append(gtruth_attributes.mean())
    attr_sums.append(gtruth_attributes.sum())
    print(attr_avgs)
    im0=Image.open(source).convert('L')

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    plot_one_box(xyxy, cam_image, color = (255,0,0) ,line_thickness=1)
    plot_one_box(coords, cam_image, color = (0,255,0) ,line_thickness=1)
    # print(type(xyxy))
    # src=source[i]
    # print(src[42:-5])# print(i)
    names.append(source[42:-5])
    Image.fromarray(cam_image).save("nwd_eigen_attribution/{}_cam.jpg".format(img_name[i]))    
        # except:
        #     print("error missing label or image") # iff possible make the ex the image name
        # label_info=pd.read_csv("eigenset.csv")
        # label_info=label_info.sort_values(by=['name'])
        # label_info=label_info.reset_index(drop=True)
    



df1=pd.DataFrame(attr_avgs,columns=['nwd_avg_attribution'])
df1.to_csv("attribute_avgs.csv")

testdf=pd.DataFrame(names)
testdf=pd.concat([testdf,df1],axis=1)
testdf.to_csv("alignnames.csv")


label_info=pd.concat([label_info,df1],axis=1)
# print(label_info)
label_info.to_csv("eigenattr.csv")


exit()
renormalized_cam_image = renormalize_cam_in_bounding_boxes(xyxy, colors, rgb_img, grayscale_cam)

Image.fromarray(renormalized_cam_image).save("eignenorm_out.png")
print(renormalized_cam_image.shape)
 

