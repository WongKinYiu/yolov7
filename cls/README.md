# yolov7

Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

This implimentation is based on [yolov5](https://github.com/ultralytics/yolov5).

All of installation, data preparation, and usage are as same as yolov5.

## Training

``` shell
python classifier.py --pretrained False --data imagenet --epochs 90 --img 224 --batch 256 --model yolov7 --name yolov7-cls --lr0 0.1 --optimizer SGD
```
