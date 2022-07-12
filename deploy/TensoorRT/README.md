# YOLOv7-TensorRT 

## Dependencies
- TensorRT >= 7.2.3.4
- OpenCV >= 4.1.0
- PyCuda == 2021.1
- onnx >= 1.8.0
- onnx-simplifier == 0.4.0 

# PyTorch ->  ONNX -> TensorRT 
These scripts were last tested using the
[NGC TensorRT Container Version 20.06-py3](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt).
You can see the corresponding framework versions for this container [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-container-release-notes/rel_20.06.html#rel_20.06).

## Quickstart

### 1. Convert PyTorch model to ONNX

```
python -m models.export --weights yolov7.pt --grid
```
The above generated onnx model need to be simplified as some of the op's need to be normalized berfore TRT serialization.
```bash
onnxsim yolov7.onnx yolov7-sim.onnx
```

### 2. Convert ONNX model to TensorRT engine

See `./onnx_to_tensorrt.py -h` for full list of command line arguments.
Use below command if using TensorRT <= 8.0, before converting onnx to TRT engine for TRT < 8.0 custom plugin needs to be build,

```bash
cd plugin
mkdir build && cd build/
cmake ..
make
```
This will build TRT pulgin as shared lib for ScatterND op. now it can be preloaded with `onnx_to_tensorrt.py` script as below,

```bash
LD_PRELOAD=plugin/build/libscatterplugin.so python3 onnx_to_tensorrt.py --explicit-batch \
                      --onnx yolov7-sim.onnx \
                      --fp16 \
                      --int8 \
                      --calibration-cache="caches/yolov7.cache" \
                      -o yolov7.int8.engine
```

In case of using TensorRT >= 8.0 the TRT plugin for `ScatterND` op is provided as built in plugin hence no need to built it from scratch.
```bash
python3 onnx_to_tensorrt.py --explicit-batch \
                      --onnx yolov7-sim.onnx \
                      --fp16 \
                      --int8 \
                      --calibration-cache="caches/yolov7.cache" \
                      -o yolov7.int8.engine
```

### 3. Run Yolov7 TRT Demo

To run final demo with TensorRT < 8.0 run below script,
```bash
LD_PRELOAD=plugin/build/libscatterplugin.so python3 yolov7_trt.py ~/Videos/test.mp4
```
In case of using TensorRT >= 8.0 the TRT plugin for `ScatterND` op is provided as built in plugin hence no need to built it from scratch.
```bash
python3 yolov7_trt.py ~/Videos/test.mp4
```