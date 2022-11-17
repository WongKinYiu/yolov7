from armellcpp import Detector
import numpy as np
from time import monotonic

# self: armellcpp.Detector, name: str, categories: List[str], image_width: int, image_height: int, batch_size: int, path: str, thresh: float, width: int, height: int, anchors: List[float], stride1: int, stride2: int, mask1: List[int], mask2: List[int], bboxes: int, yolo_layer_1: str, yolo_layer_2: str, nms_thresh: float)

# find(self: armellcpp.Detector, arg0: numpy.ndarray[numpy.float32], cuda_mem: bool = True) -> List[List[armellcpp.Detection]]

batch_size = 8

# init

det = Detector(
    "test",
    ["car", "van", "truck", "bus", "bicycle", "motorbike", "person", "trailer", "plate"],
    1920,
    1080,
    batch_size,
    "detectors/extended",
    0.1,
    512,
    288,
    [25,24, 71,26, 66,79,     148,117, 262,167, 400,220],
    32,
    16,
    [3,4,5],
    [0,1,2],
    3,
    "yolo_19",
    "yolo_28",
    0.05,
)

batch = np.random.rand(batch_size,3,288,512).astype(np.float32)


result = det.find(batch, False)

# test speed

for _ in range(10):
    t0 = monotonic()
    for _ in range(10):
        det.find(batch, False)
    took = (monotonic() - t0) / 10
    print("FPS:", 1 / took * batch_size)
