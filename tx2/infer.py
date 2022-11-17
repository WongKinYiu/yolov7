#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import time
import ctypes
import argparse
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

from image_batcher import ImageBatcher
from visualize import visualize_detections


class TensorRTInfer:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            shape = self.context.get_binding_shape(i)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                'index': i,
                'name': name,
                'dtype': dtype,
                'shape': list(shape),
                'allocation': allocation,
                'host_allocation': host_allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch):
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        cuda.memcpy_htod(self.inputs[0]['allocation'], batch)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])
        return [o['host_allocation'] for o in self.outputs]

    def process(self, batch, scales=None, nms_threshold=None):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Run inference
        outputs = self.infer(batch)

        # Process the results
        nums = outputs[0]
        boxes = outputs[1]
        scores = outputs[2]
        classes = outputs[3]
        detections = []
        normalized = (np.max(boxes) < 2.0)
        for i in range(self.batch_size):
            detections.append([])
            for n in range(int(nums[i])):
                scale = self.inputs[0]['shape'][2] if normalized else 1.0
                if scales and i < len(scales):
                    scale /= scales[i]
                if nms_threshold and scores[i][n] < nms_threshold:
                    continue
                detections[i].append({
                    'ymin': boxes[i][n][0] * scale,
                    'xmin': boxes[i][n][1] * scale,
                    'ymax': boxes[i][n][2] * scale,
                    'xmax': boxes[i][n][3] * scale,
                    'score': scores[i][n],
                    'class': int(classes[i][n]),
                })
        return detections


def main(args):
    if args.output:
        output_dir = os.path.realpath(args.output)
        os.makedirs(output_dir, exist_ok=True)

    labels = []
    if args.labels:
        with open(args.labels) as f:
            for i, label in enumerate(f):
                labels.append(label.strip())

    trt_infer = TensorRTInfer(args.engine)
    if args.input:
        print("Inferring data in {}".format(args.input))
        batcher = ImageBatcher(args.input, *trt_infer.input_spec())
        for batch, images, scales in batcher.get_batch():
            print("Processing Image {} / {}".format(batcher.image_index, batcher.num_images), end="\r")
            detections = trt_infer.process(batch, scales, args.nms_threshold)
            if args.output:
                for i in range(len(images)):
                    basename = os.path.splitext(os.path.basename(images[i]))[0]
                    # Image Visualizations
                    output_path = os.path.join(output_dir, "{}.png".format(basename))
                    visualize_detections(images[i], output_path, detections[i], labels)
                    # Text Results
                    output_results = ""
                    for d in detections[i]:
                        line = [d['xmin'], d['ymin'], d['xmax'], d['ymax'], d['score'], d['class']]
                        output_results += "\t".join([str(f) for f in line]) + "\n"
                    with open(os.path.join(output_dir, "{}.txt".format(basename)), "w") as f:
                        f.write(output_results)
    else:
        print("No input provided, running in benchmark mode")
        spec = trt_infer.input_spec()
        batch = 255 * np.random.rand(*spec[0]).astype(spec[1])
        iterations = 200
        times = []
        for i in range(20):  # GPU warmup iterations
            trt_infer.infer(batch)
        for i in range(iterations):
            start = time.time()
            trt_infer.infer(batch)
            times.append(time.time() - start)
            print("Iteration {} / {}".format(i + 1, iterations), end="\r")
        print("Benchmark results include time for H2D and D2H memory copies")
        print("Average Latency: {:.3f} ms".format(
            1000 * np.average(times)))
        print("Average Throughput: {:.1f} ips".format(
            trt_infer.batch_size / np.average(times)))

    print()
    print("Finished Processing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default=None, required=True,
                        help="The serialized TensorRT engine")
    parser.add_argument("-i", "--input", default=None,
                        help="Path to the image or directory to process")
    parser.add_argument("-o", "--output", default=None,
                        help="Directory where to save the visualization results")
    parser.add_argument("-l", "--labels", default="./labels_coco.txt",
                        help="File to use for reading the class labels from, default: ./labels_coco.txt")
    parser.add_argument("-t", "--nms_threshold", type=float,
                        help="Override the score threshold for the NMS operation, if higher than the built-in threshold")
    args = parser.parse_args()
    main(args)