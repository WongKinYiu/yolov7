FROM nvidia/cuda:11.7.0-base-ubuntu22.04

WORKDIR /app

COPY requirements.txt /requirements.txt

RUN apt-get update
RUN apt-get install python3-pip ffmpeg libsm6 libxext6 libpng-dev libjpeg-dev -y

RUN pip install -r /requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116

COPY . /app/

# Disabling MLFlow Git Integration Warnings because we're not using Git in our Training Env
ENV GIT_PYTHON_REFRESH=silence

ENTRYPOINT [ "bash" ]

