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
See the [INT8 Calibration](#int8-calibration) section below for details on calibration
using your own model or different data, where you don't have an existing calibration cache
or want to create a new one.

### INT8 Calibration

This class can be tweaked to work for other kinds of models, inputs, etc.

However, to calibrate using different data or a different model, you can do so with the `--calibration-data` argument.

* This requires that you've mounted a dataset, such as Imagenet, to use for calibration.
    * Add something like `-v /imagenet:/imagenet` to your Docker command in Step (1)
      to mount a dataset found locally at `/imagenet`.
* You can specify your own `preprocess_func` by defining it inside of `calibrator.py`

```bash
# Path to dataset to use for calibration.
#   **Not necessary if you already have a calibration cache from a previous run.
CALIBRATION_DATA="/imagenet"

# Truncate calibration images to a random sample of this amount if more are found.
#   **Not necessary if you already have a calibration cache from a previous run.
MAX_CALIBRATION_SIZE=512

# Calibration cache to be used instead of calibration data if it already exists,
# or the cache will be created from the calibration data if it doesn't exist.
CACHE_FILENAME="caches/yolov6.cache"

# Path to ONNX model
ONNX_MODEL="model/yolov6.onnx"

# Path to write TensorRT engine to
OUTPUT="yolov6.int8.engine"

# Creates an int8 engine from your ONNX model, creating ${CACHE_FILENAME} based
# on your ${CALIBRATION_DATA}, unless ${CACHE_FILENAME} already exists, then
# it will use simply use that instead.
python3 onnx_to_tensorrt.py --fp16 --int8 -v \
        --max_calibration_size=${MAX_CALIBRATION_SIZE} \
        --calibration-data=${CALIBRATION_DATA} \
        --calibration-cache=${CACHE_FILENAME} \
        --preprocess_func=${PREPROCESS_FUNC} \
        --explicit-batch \
        --onnx ${ONNX_MODEL} -o ${OUTPUT}

```
*NOTE:* For some of the optional command line arguments, See `./onnx_to_tensorrt.py -h` for full list of command line options.

### 3. Run Yolov7 TRT Demo

To run final demo with TensorRT < 8.0 run below script,
```bash
LD_PRELOAD=plugin/build/libscatterplugin.so python3 yolov7_trt.py ~/Videos/test.mp4
```
In case of using TensorRT >= 8.0 the TRT plugin for `ScatterND` op is provided as built in plugin hence no need to built it from scratch.
```bash
python3 yolov7_trt.py ~/Videos/test.mp4
```