FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx
RUN pip install seaborn thop

#https://github.com/tensorflow/tensorboard/issues/5648
#tensorboard not working on this nvidia docker img fix, you should uncomment the line below this one if your tensorboard doesnt work on this base image
#RUN sed -i "s/\"--bind_all\", default=True,/\"--bind_all\",/g" /opt/conda/lib/python3.8/site-packages/tensorboard/plugins/core/core_plugin.py
