#!/bin/bash

# Test Image
python3 detect.py \
--device 0,1 \
--weights 'weights/best.pt' \
--conf 0.25 \
--img-size 480 \
--source practice_data/small_set/yolo/images/22-11-7_Static_Camera_Blank_20.jpg

# # Test Video
# python3 detect.py \
# --weights 'runs/train/yolov7-custom21/weights/epoch_149.pt' \
# --conf 0.25 \
# --img-size 480 \
# --device 0 \
# --source ../mover/SimData_2022-11-04__11-49-13.mp4