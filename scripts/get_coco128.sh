#!/bin/bash
# Download COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017)
# Example usage: bash scripts/get_coco128.sh


# Download/unzip images and labels
d='./' # unzip directory
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
f='coco128.zip' # or 'coco128-segments.zip', 68 MB
echo 'Downloading' $url$f '...'
curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f & # download, unzip, remove in background
wait # finish background tasks
