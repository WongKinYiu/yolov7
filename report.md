got these results:
```
    Epoch   gpu_mem       box       obj       cls     total    labels  img_size
    35/299     5.35G   0.05399  0.004034  0.006746   0.06477        27       608: 100%|█████████████████████████████████| 241/241 [01:46<00:00,  2.27it/s]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|███████████████████████| 45/45 [00:14<00:00,  3.17it/s]
                 all        2829        1745       0.271       0.155     0.00747     0.00191

```
with this yaml file:
```
download: bash ./scripts/get_rdd2022.sh

train: ./data/RDD2022/India/train
test: ./data/RDD2022/Norway/test/images
val: ./data/RDD2022/Czech/train

nc: 4
names: ['D00', 'D10', 'D20', 'D40']
```

## Step guide for how to train the model
1. First, copy the dataset and convert annotations to yolo labels by running
```
python ./scripts/get_rdd2022.py
```
2. Download pre-trained yolov7 model
```
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
```
3. Run the training bash script
```
bash ./train_rdd2022.sh
```

## Description of model
The model can be found in cfg/training/yolov7-rdd2022.yaml. It is the same as the original cfg/training/yolov7.yaml except number of classes is set to 4.