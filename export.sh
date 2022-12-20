MODEL_PATH=runs/train/yolov7_gosling_fixed_res8/weights/best.pt
python export.py  --weights $MODEL_PATH --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640