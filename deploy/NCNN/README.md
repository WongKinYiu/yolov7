# YOLOv7-CPP-ncnn

Cpp file compile of YOLOv7 object detection base on [ncnn](https://github.com/Tencent/ncnn).  

## Tutorial

### Step1: 
pytorch --> torchscript
```bash
python models/export.py --weights yolov7
```

### Step2: 
torchscript --> ncnn
```shell
pnnx yolov7.torchscript.pt inputshape=[1,3,640,640] inputshape=[1,3,320,320]
```
Then, a yolov7.param and yolov7.bin file is generated.

### Step3
Copy or Move yolov7.cpp file into workdir, modify the CMakeList.txt, then build yolov7

### Step4
Inference image with executable file yolox, enjoy the detect result:
```shell
./yolov7 demo.jpg
```

## Acknowledgement

* [ncnn](https://github.com/Tencent/ncnn)
* [ncnn-models](https://github.com/Baiyuetribe/ncnn-models)