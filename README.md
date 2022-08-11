# yolov7

Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

This implimentation is for [darknet](https://github.com/AlexeyAB/darknet).

## Usage

[`yolov7-tiny.weights`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.weights) [`yolov7.weights`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.weights) [`yolov7x.weights`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.weights) 

``` shell
./darknet detector demo cfg/coco.data cfg/yolov7-tiny_darknet.cfg yolov7-tiny.weights test.mp4
./darknet detector demo cfg/coco.data cfg/yolov7_darknet.cfg yolov7.weights test.mp4
./darknet detector demo cfg/coco.data cfg/yolov7x_darknet.cfg yolov7x.weights test.mp4
```
