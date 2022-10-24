FROM python:3.8

WORKDIR /workspace

ADD . /workspace

RUN pip install -r /workspace/yolov7/requirements.txt
RUN pip install Flask==2.1.0

CMD [ "python" , "/workspace/app.py" ]

RUN chown -R 42420:42420 /workspace

ENV HOME=/workspace
