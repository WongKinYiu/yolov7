#!/usr/bin/env python3
#
# Authored by xiang-wuu
# 2022.7.11
#
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
import cv2
from trt_infer import build_engine, infer

INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.3
IOU_THRESHOLD = 0.4


class YoLov7TRT(object):
    """
    description: A YOLOv7 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        stream = cuda.Stream()
        self.mean = None
        self.std = None

        # Deserialize the engine from file
        engine, context = build_engine(engine_file_path)

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        self.batch_size = 0

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            self.batch_size = engine.max_batch_size
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

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, input_image):
        # Make self the active context, pushing it on top of the context stack.

        host_outputs = infer(self.engine, self.context, [
                             input_image.ravel()], self.stream)

        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]

        return output

    def preprocess_image(self, image_raw):
        """
        description: Read an image from image path, convert it to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, h, w

    def non_max_suppression_fast(self, boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        # initialize the list of picked indexes
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int"), pick

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = torch.zeros_like(x) if isinstance(
            x, torch.Tensor) else np.zeros_like(x)
        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        out = np.array_split(output, self.batch_size)

        for output in out:
            # Get the num of boxes detected
            num = int(output[0])
            # Reshape to a two dimentional ndarray
            pred = np.reshape(output, (1, -1, int(5+80)))[0]
            # print(pred.shape)
            # to a torch Tensor
            pred = torch.Tensor(pred).cpu()
            # Get the boxes
            boxes = pred[:, :4]
            # Get the scores
            scores = pred[:, 4]
            # Get the classid
            classid = pred[:, 5]
            # Choose those boxes that score > CONF_THRESH
            si = scores > CONF_THRESH
            boxes = boxes[si, :]
            scores = scores[si]
            classid = classid[si]
            # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
            boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
            # Do nms
            indices = torchvision.ops.nms(
                boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
            # result_boxes, indices = self.non_max_suppression_fast(
            #    boxes, IOU_THRESHOLD)
            result_boxes = boxes[indices, :]
            result_scores = scores[indices]  # .cpu()
            result_classid = classid[indices]  # .cpu()
       
        return result_boxes, result_scores, result_classid
  
    def get_rect(self, bbox, image_h, image_w, input_h, input_w):
        """
        description: postprocess the bbox
        param:
            bbox:     [x1,y1,x2,y2]
            image_h:   height of original image
            image_w:   width of original image
            input_h:   height of network input
            input_w:   width of network input
        return:
            result_bbox: finally box
        """

        result_bbox = [0, 0, 0, 0]
        r_w = input_w / (image_w * 1.0)
        r_h = input_h / (image_h * 1.0)
        if r_h > r_w:
            l = bbox[0] / r_w
            r = bbox[2] / r_w
            t = (bbox[1] - (input_h - r_w * image_h) / 2) / r_w
            b = (bbox[3] - (input_h - r_w * image_h) / 2) / r_w
        else:
            l = (bbox[0] - (input_w - r_h * image_w) / 2) / r_h
            r = (bbox[2] - (input_w - r_h * image_w) / 2) / r_h
            t = bbox[1] / r_h
            b = bbox[3] / r_h
        result_bbox[0] = l
        result_bbox[1] = t
        result_bbox[2] = r
        result_bbox[3] = b
        return result_bbox
