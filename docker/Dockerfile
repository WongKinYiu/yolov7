FROM nvcr.io/nvidia/pytorch:21.08-py3

MAINTAINER bin_ze binze.zero@gmail.com

COPY . /source_code/

# apt install required packages
RUN apt update
RUN apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
RUN pip install -r /source_code/requirements.txt

RUN mkdir -p /yolov7 && cp -r /app/source_code/* /yolov7
# go to code folder
WORKDIR /yolov7