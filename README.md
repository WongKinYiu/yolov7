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
 # Parameters 43.6M
 # FLOPs 130.5G
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.51983
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.69369
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.56392
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.35184
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.57494
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.67986
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.38951
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.64027
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.68785
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.52965
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.74464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.83022
```
