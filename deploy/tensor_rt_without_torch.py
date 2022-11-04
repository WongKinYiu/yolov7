import tensorrt as trt

import pycuda.autoinit
import pycuda.driver as cuda

from typing import List
import numpy as np

import cv2
import numpy as np

from tqdm import tqdm

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# EfficientNMS and other torch plugins
trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")


class Yolov7TensorRTDetector:
    def __init__(self, engine_path: str) -> None:
        engine_fname = engine_path.split("/")[-1]
        model_fname_prefix = engine_fname.split(".")[0]
        self.engine_path = engine_path

        self.large_letterbox = False
        for large_letterbox_variant in [
            "e6",
            "w6",
            "d6",
            "e6e",
        ]:
            if large_letterbox_variant in model_fname_prefix:
                self.large_letterbox = True
                break

        self.cls_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

        self.binding_addrs = None
        self.context = None
        self.engine = None

        self.stream = None
        self.batch_size = None

        self.bindings = None

        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Create a Context on this device,
        self.stream = cuda.Stream()
        self.batch_size = self.engine.max_batch_size

        self.bindings = {}
        for binding in self.engine:
            size = (
                trt.volume(self.engine.get_binding_shape(binding))
                * self.engine.max_batch_size
            )
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            is_input = self.engine.binding_is_input(binding)

            self.bindings[binding] = {
                "dtype": dtype,
                "size": size,
                "is_input": is_input,
                "host_mem": host_mem,
                "cuda_ptr": cuda_mem,
            }

        self.binding_addrs = [binding["cuda_ptr"] for binding in self.bindings.values()]

    def detect(self, image: np.ndarray) -> List:
        image, ratio, dwdh = Yolov7TensorRTDetector.letterbox(
            image,
            new_shape=(1280, 1280) if self.large_letterbox else (640, 640),
            auto=False,
        )
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image).astype(np.float32) / 255

        np.copyto(self.bindings["images"]["host_mem"], image.ravel())

        cuda.memcpy_htod_async(
            self.bindings["images"]["cuda_ptr"],
            self.bindings["images"]["host_mem"],
            self.stream,
        )

        self.context.execute_async_v2(
            bindings=self.binding_addrs,
            stream_handle=self.stream.handle,
        )

        for output_binding in ["num_dets", "det_boxes", "det_scores", "det_classes"]:
            cuda.memcpy_dtoh_async(
                self.bindings[output_binding]["host_mem"],
                self.bindings[output_binding]["cuda_ptr"],
                self.stream,
            )

        self.stream.synchronize()

        nums = self.bindings["num_dets"]["host_mem"][0]
        boxes = self.bindings["det_boxes"]["host_mem"]
        scores = self.bindings["det_scores"]["host_mem"]
        classes = self.bindings["det_classes"]["host_mem"]

        boxes = boxes.reshape(-1, 4)[:nums]
        scores = scores[:nums]
        classes = classes[:nums]

        bboxes = []
        for box, score, cl in zip(boxes, scores, classes):
            cls = self.cls_names[cl]

            box = Yolov7TensorRTDetector.postprocess(box, ratio, dwdh).round()

            bb = {
                "cls": cls,
                "x0": int(box[0]),
                "y0": int(box[1]),
                "x1": int(box[2]),
                "y1": int(box[3]),
                "confidence": float(score),
            }

            bboxes.append(bb)

        return bboxes

    @staticmethod
    def letterbox(
        im,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleup=True,
        stride=32,
    ):
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
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return im, r, (dw, dh)

    @staticmethod
    def postprocess(boxes, r, dwdh):
        dwdh = dwdh * 2
        boxes -= dwdh
        boxes /= r
        return boxes


if __name__ == "__main__":
    detector = Yolov7TensorRTDetector(engine_path="yolov7-tiny-nms.trt")

    image_np = np.random.randint(low=0, high=255, size=(1080, 1920, 3), dtype=np.uint8)
    bboxes = detector.detect(image_np)

    # quick approximate benchmark
    for _ in tqdm(range(100)):
        detector.detect(image_np)
