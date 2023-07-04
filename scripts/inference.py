from importlib_resources import files
from pathlib import Path
from time import perf_counter

import cv2
import torch

from yolov7.yolov7 import YOLOv7


imgpath = Path('test.jpg')
if not imgpath.is_file():
    raise AssertionError(f'{str(imgpath)} not found')

output_folder = 'inference'
Path(output_folder).mkdir(parents=True, exist_ok=True)

yolov7 = YOLOv7(
    weights=files('yolov7').joinpath('weights/yolov7_state.pt'),
    cfg=files('yolov7').joinpath('cfg/deploy/yolov7.yaml'),
    bgr=True,
    device='cuda',
    model_image_size=640,
    max_batch_size=64,
    half=True,
    same_size=True,
    conf_thresh=0.25,
    trace=False,
    cudnn_benchmark=False,
)

img = cv2.imread(str(imgpath))
bs = 512
imgs = [img for _ in range(bs)]

n = 3
dur = 0
for i in range(n):
    torch.cuda.synchronize()
    tic = perf_counter()
    dets = yolov7.detect_get_box_in(imgs, box_format='ltrb', classes=None, buffer_ratio=0.0)[0]
    # dets = yolov7.detect_get_box_in(imgs, box_format='ltrb', classes=['person'], buffer_ratio=0.0)[0]
    # print('detections: {}'.format(dets))
    torch.cuda.synchronize()
    toc = perf_counter()
    if i > 1:
        dur += toc - tic
print(f'Average time taken: {(dur/n*1000):0.2f}ms')

draw_frame = img.copy()
for det in dets:
    # print(det)
    bb, score, class_ = det
    l, t, r, b = bb
    cv2.rectangle(draw_frame, (l, t), (r, b), (255, 255, 0), 1)
    cv2.putText(draw_frame, class_, (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

output_path = Path(output_folder) / 'test_out.jpg'
cv2.imwrite(str(output_path), draw_frame)
