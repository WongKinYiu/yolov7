# docker build -t yolov7 .

FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV cwd="/home/"
WORKDIR $cwd

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ENV TORCH_CUDA_ARCH_LIST="7.5 8.6"

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt -y update

RUN apt-get install --no-install-recommends -y \
    software-properties-common \
    build-essential \
    libgl1-mesa-glx

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###

RUN python3 -m pip install --upgrade pip setuptools

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
