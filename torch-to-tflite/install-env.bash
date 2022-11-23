# create conda env
conda create --name torch-to-tflite python=3.8
conda activate torch-to-tflite
# install tensorflow
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
conda install pip
python3 -m pip install tensorflow==2.10.1
# install torch
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# install other packages
pip install pandas==1.5.1
pip install opencv-python==4.6.0.66
pip install tqdm==4.64.1
pip install pyyaml==6.0
pip install matplotlib==3.6.2
pip install seaborn==0.12.1
pip install scipy==1.9.3
pip install onnx==1.12.0
pip install onnx-tf==1.10.0
tensorflow-probability==0.18.0