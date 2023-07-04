import math
from importlib_resources import files
from pathlib import Path

import cv2

from yolov7.yolov7 import YOLOv7


vid_path = Path('testvideo.mp4')
if not vid_path.is_file():
    raise AssertionError(f'{str(vid_path)} not found')

output_dir = Path('inference')
output_dir.mkdir(parents=True, exist_ok=True)
out_fp = output_dir / f'{vid_path.stem}_inference.avi'
display_video = False

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

vidcap = cv2.VideoCapture(str(vid_path))
if not vidcap.isOpened():
    raise AssertionError(f'Cannot open video file {str(vid_path)}')

fps = vidcap.get(cv2.CAP_PROP_FPS)
fps = 25 if math.isinf(fps) else fps
vid_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_track = cv2.VideoWriter(str(out_fp), cv2.VideoWriter_fourcc(*'MJPG'), fps, (vid_width, vid_height))

if display_video:
    cv2.namedWindow('YOLOv7', cv2.WINDOW_NORMAL)

while True:
    ret, frame = vidcap.read()
    if not ret:
        break

    dets = yolov7.detect_get_box_in([frame], box_format='ltrb', classes=None)[0]

    show_frame = frame.copy()
    for det in dets:
        ltrb, conf, clsname = det
        l, t, r, b = ltrb
        cv2.rectangle(show_frame, (l, t), (r, b), (255, 255, 0), 1)
        cv2.putText(show_frame, f'{clsname}:{conf:0.2f}', (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

    out_track.write(show_frame)

    if display_video:
        cv2.imshow('YOLOv7', show_frame)
        if cv2.waitKey(1) == ord('q'):
            break

if display_video:
    cv2.destroyAllWindows()
