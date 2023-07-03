from utils.plots import plot_one_box, plot_one_box_PIL
from PIL import Image
import numpy as np
import cv2
from utils.general import xywh2xyxy, xywhn2xyxy, xyxy2xywh
import matplotlib.pyplot as plt
path="../../../data/Koutsoubn8/hex_data/cave/train/"
data_name= "_10"
img_path= path +"images/" + data_name + ".jpg"
label = path +"labels/" + data_name + ".txt"

img=np.array(Image.open(img_path))
img=cv2.resize(img,(480,480))
# img=Image.fromarray(img
# print(type(img))
# exit()
# print(img)
# print(img)
# exit()
gtruth=open(label,"r")
print(gtruth)
gtruth_info=gtruth.read()
print(gtruth_info)

def center_xy_coords(xywh):
    x,y,w,h = xywh
    center_x, center_y = (x+x+w)/2,(y+y+h)/2
    return np.array([center_x, center_y,w,h])

# exit()
gtruth_info=gtruth_info.replace('\n',"").split(" ")
coords_list=gtruth_info[1:]
coords=np.float32(coords_list)
print("coords",coords)
# coords=coords*0.25
# print("coords",coords)

# coords[1]=coords[1] * .25
# coords[3]=coords[3] * .25
# exit()

# coords=coords[[1,0,3,2]]
# coords[0]=coords[0]*1280
# coords[1]=coords[1]*720
# coords[2]=coords[2] *1280
# coords[3]=coords[3]*720
coords=coords*480


print("xywh",coords)
# exit()
# coords[0]=400   #y
# coords[1]=242   #x
# coords[2]=463   #w
# coords[3]=253   #h
# print("manual xywh", coords)




# [638,373,383,235]
# print(coords.shape)
# print(coords)

# coords[0]=coords[0]/1920
# coords[1]=coords[1]/1080
# coords[2]=coords[2]/1920
# coords[3]=coords[3]/1080


# coords=center_xy_coords(coords)
print(coords)
coords=coords.reshape((1,4))

# exit()
coords=xywh2xyxy((coords))
#
print("xyxy",coords)
coords=coords.reshape((4))
# exit()


plot_one_box(coords, img, color = (0,255,0) ,line_thickness=3)
img=Image.fromarray(img)
plt.imshow(img)
plt.savefig(("hex_coords.jpg"))
# img.save("hex_coords.jpg")