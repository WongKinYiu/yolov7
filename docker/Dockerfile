FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN apt-get update
RUN apt-get install -y zip htop screen libgl1-mesa-glx

COPY . /yolov7
WORKDIR /yolov7

RUN pip install -r requirements.txt
