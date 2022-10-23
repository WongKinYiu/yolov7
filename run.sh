#/bin/bash
python3 train.py --data data/coco_kpts.yaml --cfg cfg/yolov7-w6-pose_leaky.yaml --weights weights/yolov7-w6-person.pt \
--batch-size 16 --img 384 --kpt-label --sync-bn --device 0 --name yolov7-w6-pose --hyp data/hyp.pose_custom.yaml \
--epochs 150