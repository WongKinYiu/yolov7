# %%
import os
import cv2, glob
import numpy as np
from pathlib import Path
import shutil

# %% create cfg
def create_cfg():
    yolov7yaml = """
# parameters
nc: 3  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple

# anchors
anchors:
    - [12,16, 19,36, 40,28]  # P3/8
    - [36,75, 76,55, 72,146]  # P4/16
    - [142,110, 192,243, 459,401]  # P5/32

# yolov7 backbone
backbone:
    # [from, number, module, args]
    [[-1, 1, Conv, [32, 3, 1]],  # 0

    [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
    [-1, 1, Conv, [64, 3, 1]],

    [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4
    [-1, 1, Conv, [64, 1, 1]],
    [-2, 1, Conv, [64, 1, 1]],
    [-1, 1, Conv, [64, 3, 1]],
    [-1, 1, Conv, [64, 3, 1]],
    [-1, 1, Conv, [64, 3, 1]],
    [-1, 1, Conv, [64, 3, 1]],
    [[-1, -3, -5, -6], 1, Concat, [1]],
    [-1, 1, Conv, [256, 1, 1]],  # 11

    [-1, 1, MP, []],
    [-1, 1, Conv, [128, 1, 1]],
    [-3, 1, Conv, [128, 1, 1]],
    [-1, 1, Conv, [128, 3, 2]],
    [[-1, -3], 1, Concat, [1]],  # 16-P3/8
    [-1, 1, Conv, [128, 1, 1]],
    [-2, 1, Conv, [128, 1, 1]],
    [-1, 1, Conv, [128, 3, 1]],
    [-1, 1, Conv, [128, 3, 1]],
    [-1, 1, Conv, [128, 3, 1]],
    [-1, 1, Conv, [128, 3, 1]],
    [[-1, -3, -5, -6], 1, Concat, [1]],
    [-1, 1, Conv, [512, 1, 1]],  # 24

    [-1, 1, MP, []],
    [-1, 1, Conv, [256, 1, 1]],
    [-3, 1, Conv, [256, 1, 1]],
    [-1, 1, Conv, [256, 3, 2]],
    [[-1, -3], 1, Concat, [1]],  # 29-P4/16
    [-1, 1, Conv, [256, 1, 1]],
    [-2, 1, Conv, [256, 1, 1]],
    [-1, 1, Conv, [256, 3, 1]],
    [-1, 1, Conv, [256, 3, 1]],
    [-1, 1, Conv, [256, 3, 1]],
    [-1, 1, Conv, [256, 3, 1]],
    [[-1, -3, -5, -6], 1, Concat, [1]],
    [-1, 1, Conv, [1024, 1, 1]],  # 37

    [-1, 1, MP, []],
    [-1, 1, Conv, [512, 1, 1]],
    [-3, 1, Conv, [512, 1, 1]],
    [-1, 1, Conv, [512, 3, 2]],
    [[-1, -3], 1, Concat, [1]],  # 42-P5/32
    [-1, 1, Conv, [256, 1, 1]],
    [-2, 1, Conv, [256, 1, 1]],
    [-1, 1, Conv, [256, 3, 1]],
    [-1, 1, Conv, [256, 3, 1]],
    [-1, 1, Conv, [256, 3, 1]],
    [-1, 1, Conv, [256, 3, 1]],
    [[-1, -3, -5, -6], 1, Concat, [1]],
    [-1, 1, Conv, [1024, 1, 1]],  # 50
    ]

# yolov7 head
head:
    [[-1, 1, SPPCSPC, [512]], # 51

    [-1, 1, Conv, [256, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    [37, 1, Conv, [256, 1, 1]], # route backbone P4
    [[-1, -2], 1, Concat, [1]],

    [-1, 1, Conv, [256, 1, 1]],
    [-2, 1, Conv, [256, 1, 1]],
    [-1, 1, Conv, [128, 3, 1]],
    [-1, 1, Conv, [128, 3, 1]],
    [-1, 1, Conv, [128, 3, 1]],
    [-1, 1, Conv, [128, 3, 1]],
    [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
    [-1, 1, Conv, [256, 1, 1]], # 63

    [-1, 1, Conv, [128, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    [24, 1, Conv, [128, 1, 1]], # route backbone P3
    [[-1, -2], 1, Concat, [1]],

    [-1, 1, Conv, [128, 1, 1]],
    [-2, 1, Conv, [128, 1, 1]],
    [-1, 1, Conv, [64, 3, 1]],
    [-1, 1, Conv, [64, 3, 1]],
    [-1, 1, Conv, [64, 3, 1]],
    [-1, 1, Conv, [64, 3, 1]],
    [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
    [-1, 1, Conv, [128, 1, 1]], # 75

    [-1, 1, MP, []],
    [-1, 1, Conv, [128, 1, 1]],
    [-3, 1, Conv, [128, 1, 1]],
    [-1, 1, Conv, [128, 3, 2]],
    [[-1, -3, 63], 1, Concat, [1]],

    [-1, 1, Conv, [256, 1, 1]],
    [-2, 1, Conv, [256, 1, 1]],
    [-1, 1, Conv, [128, 3, 1]],
    [-1, 1, Conv, [128, 3, 1]],
    [-1, 1, Conv, [128, 3, 1]],
    [-1, 1, Conv, [128, 3, 1]],
    [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
    [-1, 1, Conv, [256, 1, 1]], # 88

    [-1, 1, MP, []],
    [-1, 1, Conv, [256, 1, 1]],
    [-3, 1, Conv, [256, 1, 1]],
    [-1, 1, Conv, [256, 3, 2]],
    [[-1, -3, 51], 1, Concat, [1]],

    [-1, 1, Conv, [512, 1, 1]],
    [-2, 1, Conv, [512, 1, 1]],
    [-1, 1, Conv, [256, 3, 1]],
    [-1, 1, Conv, [256, 3, 1]],
    [-1, 1, Conv, [256, 3, 1]],
    [-1, 1, Conv, [256, 3, 1]],
    [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
    [-1, 1, Conv, [512, 1, 1]], # 101

    [75, 1, RepConv, [256, 3, 1]],
    [88, 1, RepConv, [512, 3, 1]],
    [101, 1, RepConv, [1024, 3, 1]],

    [[102,103,104], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
    ]
    """
    f = open("yolov7-3ch.yaml", "w")
    f.write(yolov7yaml)
    f.close()

def create_data_yaml():
    yaml = """train: test_dataset/train_origin/images
val: test_dataset/valid_origin/images
test: test_dataset/test_origin/images

nc: 3
names: ['Ball', 'Player', 'Ref']
    """
    f = open('test_dataset/data.yaml', 'w')
    f.write(yaml)
    f.close()
    shutil.copytree("test_dataset/train/labels", "test_dataset/train_origin/labels", dirs_exist_ok=True)
    shutil.copytree("test_dataset/test/labels", "test_dataset/test_origin/labels", dirs_exist_ok=True)
    shutil.copytree("test_dataset/valid/labels", "test_dataset/valid_origin/labels", dirs_exist_ok=True)

if __name__ == "__main__":
    create_cfg()
    create_data_yaml()