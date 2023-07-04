from importlib_resources import files
from pathlib import Path

import cv2

from yolov7.yolov7 import YOLOv7


src_folder = 'inference/images'
output_folder = 'inference/results'
Path(output_folder).mkdir(parents=True, exist_ok=True)

yolov7 = YOLOv7(
    weights=files('yolov7').joinpath('weights/yolov7_state.pt'),
    cfg=files('yolov7').joinpath('cfg/deploy/yolov7.yaml'),
    bgr=True,
    device='cuda',
    model_image_size=640,
    max_batch_size=16,
    half=True,
    same_size=False,
    conf_thresh=0.25,
    trace=False,
    cudnn_benchmark=False,
)

all_imgpaths = [imgpath for imgpath in Path(src_folder).rglob("*.jpg")]
all_imgs = [cv2.imread(str(imgpath)) for imgpath in all_imgpaths]

all_dets = yolov7.detect_get_box_in(all_imgs, box_format='ltrb', classes=None, buffer_ratio=0.0)
# print('detections: {}'.format(dets))

for idx, dets in enumerate(all_dets):
    draw_frame = all_imgs[idx].copy()
    print(f'img {all_imgpaths[idx].name}: {len(dets)} detections')
    for det in dets:
        # print(det)
        bb, score, class_ = det
        l,t,r,b = bb
        cv2.rectangle(draw_frame, (l,t), (r,b), (255,255,0), 1)
        cv2.putText(draw_frame, class_, (l, t+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0))

    output_path = Path(output_folder) / f'{all_imgpaths[idx].stem}_det.jpg'
    cv2.imwrite(str(output_path), draw_frame)
