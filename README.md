# yolov7

Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

This implimentation is based on [yolov5](https://github.com/ultralytics/yolov5) and [yolov6](https://github.com/meituan/YOLOv6).

All of installation, data preparation, and usage are as same as yolov5.

## Training

``` shell
python train.py --data coco.yaml --batch 16 --weights '' --cfg cfg/yolov7.yaml --epochs 300 --name yolov7 --img 640 --hyp hyp.scratch.yaml --min-items 0
```

## Results

[`yolov7-u6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-u6.pt)

```
```
