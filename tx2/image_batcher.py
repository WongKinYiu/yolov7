#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#

import os
import sys
import random

import numpy as np
from PIL import Image


class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(self, input, shape, dtype, max_num_images=None, exact_batches=False, preprocessor="EfficientDet", shuffle_files=False):
        """
        :param input: The input directory to read images from.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param max_num_images: The maximum number of images to read from the directory.
        :param exact_batches: This defines how to handle a number of images that is not an exact multiple of the batch
        size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the
        last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        :param preprocessor: Set the preprocessor to use, depending on which network is being used.
        :param shuffle_files: Shuffle the list of files before batching.
        """
        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

        if os.path.isdir(input):
            self.images = [os.path.join(input, f) for f in os.listdir(input) if is_image(os.path.join(input, f))]
            self.images.sort()
            if shuffle_files:
                random.seed(47)
                random.shuffle(self.images)
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0:self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

        self.preprocessor = preprocessor

    def preprocess_image(self, image_path):
        """
        The image preprocessor loads an image from disk and prepares it as needed for batching. This includes padding,
        resizing, normalization, data type casting, and transposing.
        This Image Batcher implements one algorithm for now:
        * EfficientDet: Resizes and pads the image to fit the input size.
        :param image_path: The path to the image on disk to load.
        :return: Two values: A numpy array holding the image sample, ready to be contacatenated into the rest of the
        batch, and the resize scale used, if any.
        """

        def resize_pad(image, pad_color=(0, 0, 0)):
            """
            A subroutine to implement padding and resizing. This will resize the image to fit fully within the input
            size, and pads the remaining bottom-right portions with the value provided.
            :param image: The PIL image object
            :pad_color: The RGB values to use for the padded area. Default: Black/Zeros.
            :return: Two values: The PIL image object already padded and cropped, and the resize scale used.
            """
            width, height = image.size
            width_scale = width / self.width
            height_scale = height / self.height
            scale = 1.0 / max(width_scale, height_scale)
            image = image.resize((round(width * scale), round(height * scale)), resample=Image.BILINEAR)
            pad = Image.new("RGB", (self.width, self.height))
            pad.paste(pad_color, [0, 0, self.width, self.height])
            pad.paste(image)
            return pad, scale

        scale = None
        image = Image.open(image_path)
        image = image.convert(mode='RGB')
        if self.preprocessor == "EfficientDet":
            # For EfficientNet V2: Resize & Pad with ImageNet mean values and keep as [0,255] Normalization
            image, scale = resize_pad(image, (124, 116, 104))
            image = np.asarray(image, dtype=self.dtype)
            # [0-1] Normalization, Mean subtraction and Std Dev scaling are part of the EfficientDet graph, so
            # no need to do it during preprocessing here
        else:
            print("Preprocessing method {} not supported".format(self.preprocessor))
            sys.exit(1)
        if self.format == "NCHW":
            image = np.transpose(image, (2, 0, 1))
        return image, scale

    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        :return: A generator yielding three items per iteration: a numpy array holding a batch of images, the list of
        paths to the images loaded within this batch, and the list of resize scales for each image in the batch.
        """
        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            batch_scales = [None] * len(batch_images)
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i], batch_scales[i] = self.preprocess_image(image)
            self.batch_index += 1
            yield batch_data, batch_images, batch_scales
