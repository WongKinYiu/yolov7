# YoloV7 PTG

Table of Contents

[[_TOC_]]

## Data
### Convert KWCOCO to Input Data
```bash
$ python scripts/translate_coffee_to_coco.py --dset /data/PTG/cooking/object_anns/coffee+tea/berkeley/coffee_v2.3_and_tea_v2.2_obj_annotations_plus_bkgd.mscoco.json --output_dir datasets/coffee+tea/ --split train
```

You should have a file structure like:
```
├── labels
│   ├── train
|       ├── {image_filename}.txt
├── train.txt
```

## Weights
Download the pre-trained weights
```bash
$ mkdir weights
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt weights
```

## Configs
## Changes Needed
- Change `nc` in cfg/training/PTG/cooking/{filename}.yaml to the number of classes (including background)

- Change `nc` in data/PTG/cooking/{filename}.yaml to the number of classes (including background)

- The objects here must have an index of 0 (background) through `nc`. There cannot be any skipped indexes. If there are, add a `null-{index}` to the list

## Train
```bash
$ python train.py --workers 8 --device 0 --batch-size 4 --data data/PTG/cooking/coffee+tea_task_objects.yaml --img 1280 720 --cfg cfg/training/PTG/cooking/
yolov7_coffee+tea.yaml  --weights weights/yolov7.pt --name coffee+tea_yolov7 --hyp data/hyp.scratch.custom.yaml
```

## Inference
### Output to text files
```bash
$ python detect.py --weights runs/train/coffee+tea_yolov7/weights/best.pt --conf 0.2 --img-size 1280 --source /data/PTG/cooking/ros_bags/coffee/coffee_extracted/all_activities_20_extracted/images/ --project runs/detect/coffee+tea_yolov7 --save-txt 
```

### Output to kwcoco
```bash
$ python detect_ptg.py --recipes coffee tea --split val --weights runs/train/coffee+tea_yolov7/weights/best.pt --project runs/detect --name coffee+tea_yolov7 --save-img
```
