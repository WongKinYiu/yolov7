#!/bin/bash

rm data/{val,train}/labels.cache

python train.py --workers 1 --device 0 --batch-size 8 \
                --epochs 100 --img 640 640 \
                --data data/custom_data.yaml \
                --hyp data/hyp.scratch.custom.yaml \
                --cfg cfg/training/yolov7-tiny.yaml \
                --name yolov7-tiny-custom \
                --weights yolov7-tiny.pt

cp  yolov7-tiny-custom/runs/train/yolov7-tiny-custom/weights/best.pt \
    ./yolov7-tiny_custom.pt


for img in $(ls data/val/images); do
  python detect.py --weights yolov7-tiny_custom.pt --conf 0.5 \
         --img-size 640 --source "data/val/images/$img" --view-img --no-trace;
done

cp runs/detect/exp/* ./
