#!/usr/bin/env python

from setuptools import find_packages, setup

# Reflects the "base" area of the requirements.txt file.
install_requires = [
    "matplotlib>=3.2.2",
    # See https://github.com/WongKinYiu/yolov7/issues/1537
    "numpy>=1.18.5,<1.24.0",
    "opencv-python>=4.1.1",
    "Pillow>=7.1.2",
    "PyYAML>=5.3.1",
    "requests>=2.23.0",
    "scipy>=1.4.1",
    "torch>=1.7.0,!=1.12.0",
    "torchvision>=0.8.1,!=0.13.0",
    "tqdm>=4.41.0",
    "protobuf<4.21.3",
]

setup(
    name="yolov7",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/cameron-a-johnson/yolov7",
    install_requires=install_requires,
    packages=find_packages(),
)
