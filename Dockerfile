FROM python:3.9

WORKDIR /app

COPY requirements.txt /requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libpng-dev libjpeg-dev -y

RUN pip install -r /requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116

COPY . /app/

ENTRYPOINT [ "python3" ]
