#!/usr/bin/env bash

TAG=v0.0.3
docker build -t footprintai/yolov7-objectdetector:${TAG} -f Dockerfile .
docker push footprintai/yolov7-objectdetector:${TAG}
