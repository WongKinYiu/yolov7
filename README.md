# License Plate Detection based on Yolov7 and OCR systems: PyTesseract, EasyOCR (soon), Attention OCR (soon)

## Installation

``` shell
# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx tesseract pipenv

# create virtualenv:  
pipenv install

# activate virtualenv:  
pipenv shell

# download data:  
gdown -O data/ 'https://drive.google.com/uc?id=1nps9Cv4-kqPZAA92sDP7rnAX76ABgdhy'

# download weights:  
gdown -O data/ 'https://drive.google.com/uc?id=1HH7ei39rPZNxYqH4TTQom01UFlYvxS04'

```

## Inference
On video:
``` shell
pipenv run python detect.py --weights data/yolov7_plate_number.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

On image:
``` shell
pipenv run python detect.py --weights data/yolov7_plate_number.pt --conf 0.25 --img-size 640 --source inference/images/test_range.jpg
```