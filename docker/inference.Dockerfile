FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN apt-get update
RUN apt-get install -y zip htop screen libgl1-mesa-glx xcb
# xcb is for displaying from container to host for yolov7

COPY . /yolov7
WORKDIR /yolov7

RUN pip install -r requirements.txt

# docker build -f docker/inference.Dockerfile -t yolov7:inference .
