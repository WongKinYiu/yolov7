### Download pre-trained model
```
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
```

### Download coco128 datasets
```
bash scripts/get_coco128.sh
```

### Training with first gpu
```
python train.py --workers 2 --device 0 --batch-size 4 --epochs 3 --data data/coco128.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml
```

### Bento serve
```
bentoml serve yolo_demo:latest
```