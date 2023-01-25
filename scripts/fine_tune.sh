#!/bin/bash

# Train

python3 -m torch.distributed.launch --nproc_per_node 6 --master_port 9527 train.py \
--epochs 100 \
--workers 8 \
--device 0,1,2,3,4,5 \
--batch-size 768 \
--data data/hds.yaml \
--img 480  480 \
--cfg cfg/training/yolov7-tiny-drone.yaml \
--entity "drone_detection" \
--weights 'weights/yolov7-tiny.pt' \
--name NWD_hds_100k \
--hyp data/hyp.drone.tiny.yaml \
--multi-scale \
--save_period 2 \
--freeze 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 \
| tee NWD_hds_100k.out



