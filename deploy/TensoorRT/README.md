# ONNX -> TensorRT 
These scripts were last tested using the
[NGC TensorRT Container Version 20.06-py3](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt).
You can see the corresponding framework versions for this container [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-container-release-notes/rel_20.06.html#rel_20.06).

## Quickstart

### 1. Convert PyTorch model to ONNX

```
python -m models.export --weights yolov7.pt --device 0
```

### 2. Convert ONNX model to TensorRT engine

See `./onnx_to_tensorrt.py -h` for full list of command line arguments.

```bash
./onnx_to_tensorrt.py --explicit-batch \
                      --onnx yolov7.onnx \
                      --fp16 \
                      --int8 \
                      --calibration-cache="caches/yolov7.cache" \
                      -o yolov7.int8.engine
```


