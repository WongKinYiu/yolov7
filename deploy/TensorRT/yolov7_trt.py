
#!/usr/bin/env python3
#
# Authored by xiang-wuu
# 2022.7.12
#
"""
An example that uses TensorRT's Python api to make inferences.
"""

import time
import cv2
import sys
import numpy as np
import pycuda.autoinit
from yolov7 import YoLov7TRT
import pycuda.driver as cuda
from utils import plot_one_box, plot_label, time_synchronized, update_fps

if __name__ == "__main__":
    engine_file_path = "model.engine"

    # load coco labels
    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                  "hair drier", "toothbrush"]
    ms_count = 0
    mean_count = 0
    avg_ms = 0

    prev_time = time.time()
    j_count = 0
    fps = 0
    actual_fps = 0
    cfx = cuda.Device(0).make_context()

    # a  YoLov5TRT instance
    yolov5_wrapper = YoLov7TRT(engine_file_path)

    cam = cv2.VideoCapture(sys.argv[1])
    while True:
        cfx.push()
        prev_time, fps, actual_fps = update_fps(prev_time, fps, actual_fps)
        _, img = cam.read()
        if img is None:
            break
        img2 = img.copy()
        ms_count += 1

        if img is None:
            break
        t1 = time_synchronized()

        # Do image preprocess
        input_image, origin_h, origin_w = yolov5_wrapper.preprocess_image(
            img
        )

        t2 = time_synchronized()
        pre_time = t2 - t1

        t1 = time_synchronized()

        # prepare inputs for the batch
        input_image = np.vstack(
            [input_image])

        output = yolov5_wrapper.infer(
            input_image)
        # print(output.shape)
        t2 = time_synchronized()

        infer_time = t2 - t1

        t1 = time_synchronized()

        # Do postprocess
        result_boxes, result_scores, result_classid = yolov5_wrapper.post_process(
            output, origin_h, origin_w
        )
        t2 = time_synchronized()

        # Draw rectangles and labels on the original image
        for i in range(len(result_boxes)):
            h, w, c = img2.shape

            box = result_boxes[i]
            plot_one_box(
                box,
                img,
                color=(0, 0, 255),
                label="{}:{:.2f}".format(
                    categories[int(result_classid[i])], result_scores[i]
                ),
            )

        post_time = t2 - t1

        image_raw = cv2.resize(img, (640, 480))
        color = (0, 0, 255)
        avg_ms1 = pre_time + mean_count / ms_count
        plot_label(
            (15, 20),
            image_raw,
            color=(0, 0, 255),
            label="FPS: " + str(actual_fps),
            line_thickness=1.1
        )
        plot_label(
            (15, 30),
            image_raw,
            color=(0, 0, 255),
            label='pre latency: ' + str(avg_ms1),
            line_thickness=1
        )

        avg_ms2 = infer_time + mean_count / ms_count
        plot_label(
            (15, 42),
            image_raw,
            color=(0, 0, 255),
            label='infer latency: ' + str(avg_ms2),
            line_thickness=1
        )

        avg_ms3 = post_time + mean_count / ms_count
        plot_label(
            (15, 52),
            image_raw,
            color=(0, 0, 255),
            label='post latency: ' + str(avg_ms3),
            line_thickness=1
        )
        plot_label(
            (15, 63),
            image_raw,
            color=(0, 0, 255),
            label='total latency: ' + str(avg_ms1+avg_ms2+avg_ms3),
            line_thickness=1
        )

        cv2.imshow("output",
                   image_raw)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
    cfx.push()
