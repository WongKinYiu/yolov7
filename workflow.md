bash get_coco128.sh

python train.py --workers 2 --device 3 --batch-size 4 --epochs 3 --data data/coco128.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml