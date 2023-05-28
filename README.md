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

## ELAN

| Model | Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | \#Param. | FLOPs | Setting |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ELAN-tiny (nyc) | 640 | 36.2% | 53.6% | 39.5% | 3.9M | 16.1G | {1,1} |
| ELAN-tiny | 640 | 41.1% | 58.2% | 44.7% | 4.9M | 19.4G | {2,1} |
| ELAN-tiny | 640 | 42.1% | 58.2% | 45.7% | 5.2M | 20.4G | {1,2} |
| ELAN-tiny | 640 | 44.5% | 60.9% | 48.2% | 6.2M | 23.6G | {2,2} |
| Dark-ELAN-tiny | 640 | 44.4% | 60.8% | 48.0% | 5.6M | 21.8G | {1,3} |
| Dark-ELAN-tiny | 640 | 45.0% | 61.3% | 48.6% | 6.2M | 23.6G | {3,3} |
| Res-ELAN-tiny | 640 | 44.3% | 60.7% | 48.0% | 5.4M | 21.1G | . |
| CSP-ELAN-tiny | 640 | % | % | % | 5.6M | 21.6G | {1,3,1} |
| CSP-ELAN-tiny | 640 | % | % | % | 5.9M | 22.7G | {1,3,2} |
| CSP-ELAN-tiny | 640 | 45.1% | 61.6% | 49.1% | 5.9M | 22.5G | {3,3,1} |
| CSP-ELAN-tiny | 640 | % | % | % | 6.5M | 24.5G | {3,3,2} |
|  |  |  |  |  |  |  |  |
| DarkN-ELAN-tiny | 640 | 42.1% | 58.6% | 45.8% | 4.4M | 17.9G | {1,3} |
| DarkN-ELAN-tiny | 640 | 43.5% | 59.8% | 47.3% | 5.0M | 19.7G | {3,3} |
| ResN-ELAN-tiny | 640 | 41.9% | 58.4% | 45.8% | 4.2M | 17.1G | . |
| CSPN-ELAN-tiny | 640 | 43.7% | 60.2% | 47.5% | 4.6M | 18.6G | {3,3,1} |
| CSPN-ELAN-tiny | 640 | 44.7% | 61.1% | 48.5% | 5.2M | 20.5G | {3,3,2} |
|  |  |  |  |  |  |  |  |
| ELAN-nano | 640 | % | % | % | 1.7M | 7.7G | {2,2} |
| ELAN-tina | 640 | % | % | % | 3.9M | 16.1G | {2,2} |
|  |  |  |  |  |  |  |  |
