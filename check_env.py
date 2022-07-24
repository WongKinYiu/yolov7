try:
    import matplotlib
    import numpy
    import cv2
    import Pillow
    import PyYAML
    import requests
    import scipy
    import torch
    import torchvision
    import tqdm
    import protobuf
    import tensorboard
    import pandas
    import seaborn
    import ipython  # interactive notebook
    import psutil  # system utilization
    import thop  # FLOPs computation
    print('The package is installed normally')
except Exception as exp:
    print(f'error:{exp}')