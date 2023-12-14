from argparse import ArgumentParser

import cv2
import torch
import random
import time
import numpy as np
import tensorrt as trt
import av
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
import os


def load_tensorrt_model(weights_path: str, device):
    """

    Args:
        weights_path ():

    Returns:

    """
    # Infer TensorRT Engine
    binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(weights_path, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    print(type(model), model)
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        print('SHAPE: ', shape, dtype)
        if shape[0] <= 0:
            print('SHAPE IS UNUSUAL')
            shape = (1, shape[1])
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        print(data.shape)
        bindings[name] = binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()
    print(type(context), context)
    return context, binding_addrs, bindings


def letterbox(im, new_shape=(1920, 1920), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def postprocess(boxes,r,dwdh):
    dwdh = torch.tensor(dwdh*2).to(boxes.device)
    boxes -= dwdh
    boxes /= r
    return boxes



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-w", "--weights", help="Path to weights file")
    parser.add_argument("-i", "--input", help="Path to input video")
    parser.add_argument("-o", "--output", help="Path to output video")
    parser.add_argument("-n", "--names", help="Class names", nargs="+")
    parser.add_argument("--img_size", help="Image size width height", nargs="+")

    args = parser.parse_args()

    device = torch.device('cuda:0')
    colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(args.names)}

    context, binding_addrs, bindings = load_tensorrt_model(args.weights, device)
    # # INIT THE WARMUP
    # img = cv2.imread('./hockey_2.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = img.copy()
    # image, ratio, dwdh = letterbox(image, auto=False)
    # image = image.transpose((2, 0, 1))
    # image = np.expand_dims(image, 0)
    # image = np.ascontiguousarray(image)
    #
    # im = image.astype(np.float32)
    #
    # im = torch.from_numpy(im).to(device)
    # im /= 255
    #
    # # warmup for 10 times
    # for _ in range(10):
    #     tmp = torch.randn(1, 3, 1920, 1920).to(device)
    #     binding_addrs['images'] = int(tmp.data_ptr())
    #     context.execute_v2(list(binding_addrs.values()))
    
    # end of warmup
    container = av.open(args.input)

    # capture = cv2.VideoCapture(os.path.join(args.input))
    # get width and height
    print("Getting data from: ", args.input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25
    width, height = args.img_size
    print("writing to: ", args.output)
    print("", fps, width, height)
    # out_writer = cv2.VideoWriter(args.output, fourcc, fps, (int(width), int(height)))
    output = av.open(args.output, 'w')
    stream = output.add_stream("h264", 25)
    options = dict(
        threads='0',
        preset='fast',
        profile='high'
    )
    bitrate = 10_000_000
    stream.bit_rate = bitrate
    stream.pix_fmt = 'yuvj420p'
    stream.options = options
    stream.height = int(height)
    stream.width = int(width)
    count = 0

    
    total_time_list = []
    detection_time_list = []
    total_detections = 0
    for frame in container.decode(video=0):
        captured_img = frame.to_ndarray(format='bgr24')
        # here we get the image in a np.ndarray format
        # captured_img = capture.read()[1]
        print("next img; ", captured_img.shape)
        # if all_frames % sampling_rate != 0:
        #     continue
        if captured_img is None:
            break
        ###### MAGIC GOES HERE #########
        t0 = time.time()
        img = captured_img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
    
        im = image.astype(np.float32)
        print("prepped img: ", im.shape)
        t1 = time.time()
        im = torch.from_numpy(im).to(device)
        t2 = time.time()
        im /= 255
    
        start = time.perf_counter()
        binding_addrs['images'] = int(im.data_ptr())
        context.execute_v2(list(binding_addrs.values()))
        t3 = time.time()
    
        nums = bindings['num_dets'].data
        boxes = bindings['det_boxes'].data
        scores = bindings['det_scores'].data
        classes = bindings['det_classes'].data

        boxes = boxes[0, :nums[0][0]]
        scores = scores[0, :nums[0][0]]
        classes = classes[0, :nums[0][0]]
    
        d6 = 0
    
        total_detections += len(boxes)
    
        for box, score, cl in zip(boxes, scores, classes):
            t4 = time.time()
            box = postprocess(box, ratio, dwdh).round().int()
            t5 = time.time()
            d6 += t5 - t4
            name = args.names[cl]
            color = colors[name]
            name += ' ' + str(round(float(score), 3))
            cv2.rectangle(img, box[:2].tolist(), box[2:].tolist(), color, 2)
            cv2.putText(img, name, (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)
    
        ################################
        t4 = time.time()
        total_time_list.append(t4-t0)
        detection_time_list.append(t3 - t2)
        print(f'Preprocess: {t1 - t0}; To GPU: {t2 - t1}; Detection {t3 - t2}; Cost of detection {time.perf_counter() - start} s; Posprocess: {d6}; Total time: {t4 - t0}')
    
        # out_writer.write(img)
        # Write the dewarped frame to the output video
        frame_d = av.VideoFrame.from_ndarray(img, format='rgb24')
        packet = stream.encode(frame_d)
        output.mux(packet)

    # try:
    #     capture.release()
    # except NameError:
    #     print("No images found in the range")
    # cv2.destroyAllWindows()
    
    print('mean total time', np.mean(total_time_list))
    print('mean detection time', np.mean(detection_time_list))
    print('Total detection count', total_detections)
    print('total time', np.sum(total_time_list))
    packets = stream.encode()
    output.mux(packets)
    output.close()