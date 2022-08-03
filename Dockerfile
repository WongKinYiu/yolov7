ARG BASE_IMG=oliversssf1/python-cuda:poetry-conda-python-cuda11.4.0

FROM ${BASE_IMG} as base


ARG WORKSPACE_PATH
WORKDIR ${WORKSPACE_PATH}

RUN echo $WORKSPACE_PATH

RUN apt update -y && apt install -y htop tmux vim zip screen libgl1-mesa-glx

# Create conda environment according to environment.yml
# COPY environment.yml environment.yml
RUN eval "$($HOME/miniconda/bin/conda shell.bash hook)" && \
  conda create --name=yolov7 python=3.7.13
#   conda env create -f environment.yml && \
  
COPY requirements.txt requirements.txt

RUN eval "$($HOME/miniconda/bin/conda shell.bash hook)" && \
  conda activate yolov7 && \
  pip install -r requirements.txt


# Install development packages with poetry
# ENV PATH "$PATH:/root/.local/bin"
# COPY pyproject.toml pyproject.toml
# COPY poetry.lock poetry.lock
# RUN  eval "$($HOME/miniconda/bin/conda shell.bash hook)" && \
#   conda activate comprehelp && \
#   poetry install
