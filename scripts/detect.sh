#!/bin/bash

# Test
python3 detect.py \
--device 0,1 \
--weights 'weights/best.pt' \
--conf 0.25 \
--img-size 480 \
--source practice_data/small_set/22-12-15/Raw/SimData_2022-12-15__15-06-08.mp4
