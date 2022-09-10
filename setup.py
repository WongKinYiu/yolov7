from setuptools import setup, find_packages

setup(name='yolov7',
      version='1.0',
      packages=find_packages(),
      include_package_data = True,
      package_data={
        'yolov7': ['cfg/*/*.yaml', 'data/*.yaml', 'weights/*.pt'],
      })
