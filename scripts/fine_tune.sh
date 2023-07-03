#!/bin/bash

# Train

python3 -m torch.distributed.launch --nproc_per_node 3 --master_port 9527 train.py \
--epochs 100 \
--workers 8 \
--device 0,1,2  \
--batch-size 64 \
--data data/full_vr.yaml \
--img 480  480 \
--cfg cfg/training/yolov7-tiny-drone.yaml \
--entity "drone_detection" \
--weights 'weights/dyvir_weights/ciou_dyvir.pt' \
--name test1 \
--hyp data/hyp.drone.tiny.yaml \
--multi-scale \
--save_period 2 \
| tee test.out



