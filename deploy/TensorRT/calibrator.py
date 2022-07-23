#!/usr/bin/env python3

#
# Modified by xiang-wuu
# 2022.7.13
#
# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import glob
import random
import logging
import cv2

import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def preprocess_yolov7(image, channels=3, INPUT_H=640, INPUT_W=640):
    """
    description: Read an image from image path, convert it to RGB,
                 resize and pad it to target size, normalize to [0,1],
                 transform to NCHW format.
    param:
        image: PIL.Image
            The image resulting from PIL.Image.open(filename) to preprocess
        channels: int
            The number of channels the image has (Usually 1 or 3)
        INPUT_H: int
            The desired height of the image (usually 640)
        INPUT_W: int
            The desired width of the image  (usually 640)
    return:
        image:  the processed np image
    """
    # Convert PIL to numpy array
    image_raw = np.asarray(image).astype(np.float32)

    h, w, _ = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = INPUT_W / w
    r_h = INPUT_H / h
    if r_h > r_w:
        tw = INPUT_W
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((INPUT_H - th) / 2)
        ty2 = INPUT_H - th - ty1
    else:
        tw = int(r_h * w)
        th = INPUT_H
        tx1 = int((INPUT_W - tw) / 2)
        tx2 = INPUT_W - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image


def get_int8_calibrator(calib_cache, calib_data, max_calib_size, calib_batch_size):
    # Use calibration cache if it exists
    if os.path.exists(calib_cache):
        logger.info(
            "Skipping calibration files, using calibration cache: {:}".format(
                calib_cache
            )
        )
        calib_files = []
    # Use calibration files from validation dataset if no cache exists
    else:
        if not calib_data:
            raise ValueError(
                "ERROR: Int8 mode requested, but no calibration data provided. Please provide --calibration-data /path/to/calibration/files"
            )

        calib_files = get_calibration_files(calib_data, max_calib_size)

    # Choose pre-processing function for INT8 calibration
    preprocess_func = preprocess_yolov7

    int8_calibrator = ImageCalibrator(
        calibration_files=calib_files,
        batch_size=calib_batch_size,
        cache_file=calib_cache,
    )
    return int8_calibrator


def get_calibration_files(
    calibration_data,
    max_calibration_size=None,
    allowed_extensions=(".jpeg", ".jpg", ".png"),
):
    """Returns a list of all filenames ending with `allowed_extensions` found in the `calibration_data` directory.

    Parameters
    ----------
    calibration_data: str
        Path to directory containing desired files.
    max_calibration_size: int
        Max number of files to use for calibration. If calibration_data contains more than this number,
        a random sample of size max_calibration_size will be returned instead. If None, all samples will be used.

    Returns
    -------
    calibration_files: List[str]
         List of filenames contained in the `calibration_data` directory ending with `allowed_extensions`.
    """

    logger.info("Collecting calibration files from: {:}".format(calibration_data))
    calibration_files = [
        path
        for path in glob.iglob(os.path.join(calibration_data, "**"), recursive=True)
        if os.path.isfile(path) and path.lower().endswith(allowed_extensions)
    ]
    logger.info("Number of Calibration Files found: {:}".format(len(calibration_files)))

    if len(calibration_files) == 0:
        raise Exception(
            "ERROR: Calibration data path [{:}] contains no files!".format(
                calibration_data
            )
        )

    if max_calibration_size:
        if len(calibration_files) > max_calibration_size:
            logger.warning(
                "Capping number of calibration images to max_calibration_size: {:}".format(
                    max_calibration_size
                )
            )
            random.seed(42)  # Set seed for reproducibility
            calibration_files = random.sample(calibration_files, max_calibration_size)

    return calibration_files


# https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Int8/EntropyCalibrator2.html
class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Calibrator Class for Imagenet-based Image Classification Models.

    Parameters
    ----------
    calibration_files: List[str]
        List of image filenames to use for INT8 Calibration
    batch_size: int
        Number of images to pass through in one batch during calibration
    input_shape: Tuple[int]
        Tuple of integers defining the shape of input to the model (Default: (3, 224, 224))
    cache_file: str
        Name of file to read/write calibration cache from/to.
    preprocess_func: function -> numpy.ndarray
        Pre-processing function to run on calibration data. This should match the pre-processing
        done at inference time. In general, this function should return a numpy array of
        shape `input_shape`.
    """

    def __init__(
        self,
        calibration_files=[],
        batch_size=32,
        input_shape=(3, 640, 640),
        cache_file="calibration.cache",
        use_cv2=False,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.batch = np.zeros((self.batch_size, *self.input_shape), dtype=np.float32)
        self.device_input = cuda.mem_alloc(self.batch.nbytes)

        self.files = calibration_files
        self.use_cv2 = use_cv2
        # Pad the list so it is a multiple of batch_size
        if len(self.files) % self.batch_size != 0:
            logger.info(
                "Padding # calibration files to be a multiple of batch_size {:}".format(
                    self.batch_size
                )
            )
            self.files += calibration_files[
                (len(calibration_files) % self.batch_size) : self.batch_size
            ]

        self.batches = self.load_batches()
        self.preprocess_func = preprocess_yolov7

    def load_batches(self):
        # Populates a persistent self.batch buffer with images.
        for index in range(0, len(self.files), self.batch_size):
            for offset in range(self.batch_size):
                if self.use_cv2:
                    image = cv2.imread(self.files[index + offset])
                else:
                    image = Image.open(self.files[index + offset])
                self.batch[offset] = self.preprocess_func(image, *self.input_shape)
            logger.info(
                "Calibration images pre-processed: {:}/{:}".format(
                    index + self.batch_size, len(self.files)
                )
            )
            yield self.batch

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            # Assume self.batches is a generator that provides batch data.
            batch = next(self.batches)
            # Assume that self.device_input is a device buffer allocated by the constructor.
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info(
                    "Using calibration cache to save time: {:}".format(self.cache_file)
                )
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info(
                "Caching calibration data for future use: {:}".format(self.cache_file)
            )
            f.write(cache)
