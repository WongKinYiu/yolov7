# yolov7

Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

This implimentation is based on [yolov5](https://github.com/ultralytics/yolov5).

All of installation, data preparation, and usage are as same as yolov5.

## Training

``` shell
python classifier.py --pretrained False --data imagenet --epochs 90 --img 224 --batch 256 --model yolov7 --name yolov7-cls --lr0 0.1 --optimizer SGD
```

## Results

[`yolov7-cls.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-cls.pt)

```
Top-1: 78.3%
Top-5: 94.1%
```

[`yolov7-clsn.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-clsn.pt)

```
Top-1: 78.2%
Top-5: 94.3%
```
