FROM nvcr.io/nvidia/pytorch:21.08-py3
RUN apt update -y && apt install -y zip htop screen libgl1-mesa-glx
COPY requirements.txt /workspace
RUN pip install seaborn && pip install -r requirements.txt