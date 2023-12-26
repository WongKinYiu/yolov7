from setuptools import setup, find_namespace_packages

setup(
    name='yolov7',
    version='0.1.0',
    packages=find_namespace_packages(include=['yolov7.*']),
    url='https://github.com/ValV/yolov7',
    license='GPLv3',
    author='ValV',
    author_email='0x05a4@gmail.com',
    description='YOLOv7 core package',
    install_requires=[
    ]
)