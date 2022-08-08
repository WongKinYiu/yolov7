import os
import pycuda.driver as cuda
import tensorrt as trt
import sys
import numpy as np

TRT_LOGGER = trt.Logger()


def build_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()
            return engine, context
    else:
        print('engine file not found')
        sys.exit(0)


def infer(engine, context, input, stream):
    # get sizes of input and output and allocate memory required for input data and for output data
    host_outputs = []
    host_inputs = []
    cuda_inputs = []
    cuda_outputs = []
    bindings = []

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(
            binding)) * engine.max_batch_size
        batch_size = engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(cuda_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            input_shape = engine.get_binding_shape(binding)
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)

    # print(host_input.shape)
    for i in range(len(host_inputs)):
        np.copyto(host_inputs[i], input[i])
        cuda.memcpy_htod_async(cuda_inputs[i], host_inputs[i], stream)

    # run inference
    context.execute_async(bindings=bindings, stream_handle=stream.handle)

    for i in range(len(cuda_outputs)):
        cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)

    stream.synchronize()

    return host_outputs
