import argparse

import torch

from models.yolo import Model

"""
Command Line to export the model to ONNX from .pt
python export.py --weights yolov7-4d-640-nms.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
"""

def create_yolo4d_model(cfg_file='cfg/deploy/yolov7.yaml', save_file='yolov7-4d-640-nms.pt'):
    print(f'Loading YOLOv7 config from : {cfg_file}')
    model = Model(cfg_file, ch=3, nc=80)
    ckpt = {'model':model}
    torch.save(ckpt, save_file)
    print(f'Saved YOLOv7 Model to : {save_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a new YOLOv7 model with only 4D tensors")
    parser.add_argument('-c','--cfg_file', help='Path to YOLOv7 config file', default='cfg/deploy/yolov7.yaml', required=False)
    parser.add_argument('-s','--save_file', help='Path to save the new yolo pt model', default='yolov7-4d-640-nms.pt' ,required=False)
    args = parser.parse_args()

    create_yolo4d_model(args.cfg_file, args.save_file)