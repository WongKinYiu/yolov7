import io
import os
import re

import setuptools


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setuptools.setup(
    name="yolov7",
    version="1.0.0",
    author="Two Six Technologies",
    license="MIT",
    description="Packaged version of the WongKinYiu yolov7 repository",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/twosixlabs/yolov7",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=get_requirements(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning, deep-learning, pytorch, vision, image-classification, object-detection, yolov7, detector, yolov5",
)
