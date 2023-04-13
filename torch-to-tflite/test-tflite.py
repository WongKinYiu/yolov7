import argparse
from pathlib import Path
import sys
from typing import List
from utils.general import non_max_suppression
from utils.plots import output_to_target, plot_images
import numpy as np
import cv2
import tensorflow as tf
import torch

RESIZE_SCALE = 0.58 # Use the same resize scale as the one calculated during training.
PAD_COLOR = (114, 114, 114)
NAMES = {0: "empty", 1: "yes", 2: "no", 3: "both", 4: "invalid"}


def test_tflite(
    tflite_model_path: Path,
    input_data_path: Path,
    output_dir: Path,
    conf_thres: float,
    iou_thres: float,
):
    print("-" * 100 + "\nLoading TFLite model...")
    # assertion is not working for some reason
    assert tflite_model_path.is_file(), f"{tflite_model_path} is not a file."
    interpreter = _load_tflite_model(tflite_model_path)
    input_shape = interpreter.get_input_details()[0]["shape"]
    my_signature = interpreter.get_signature_runner()
    output_dir.mkdir(exist_ok=True, parents=True)
    assert input_data_path.is_dir(), f"{input_data_path} is not a directory."
    for img_path in Path(input_data_path).glob("*.*[Gg]"):
        img = cv2.imread(str(img_path))
        input_data = _preprocess_input(img, input_shape)
        output_tensor = my_signature(input=input_data / 255)["output"]
        output = _post_process(output_tensor, conf_thres, iou_thres, True)
        plot_images(input_data, output, None, str(output_dir / img_path.name), NAMES)


def _load_tflite_model(tflite_model_path: Path):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    return interpreter


def _preprocess_input(img: Path, input_shape: List[int]) -> np.ndarray:
    _, _, target_h, target_w = input_shape
    img_h, img_w, _ = img.shape
    max_img_size = max(img_h, img_w)
    max_target_size = max(target_h, target_w)
    resize_scale = min( [max_target_size / max_img_size, RESIZE_SCALE] )
    resized_img = cv2.resize(
        img, (int(resize_scale * img_w), int(resize_scale * img_h))
    )
    resized_img_h, resized_img_w, _ = resized_img.shape
    pad_h, pad_w = _get_pad(resized_img_h, target_h), _get_pad(resized_img_w, target_w)
    padded_img = cv2.copyMakeBorder(
        resized_img, *pad_h, *pad_w, cv2.BORDER_CONSTANT, value=PAD_COLOR
    )
    return np.expand_dims(np.moveaxis(padded_img, -1, 0), 0).astype(np.float32)


def _get_pad(img_size: int, target_size: int):
    pad = (target_size - img_size) // 2
    return pad, target_size - img_size - pad


def _post_process(output_tensor, conf_thres, iou_thres, multi_label):
    output_tensor = torch.tensor(output_tensor)
    nms_result = non_max_suppression(
        output_tensor, conf_thres, iou_thres, multi_label=multi_label
    )
    return output_to_target(nms_result)


def parse_args(cli_args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tp",
        "--tflite-model-path",
        type=Path,
        required=True,
        help="Path to the tflite model to be tested.",
    )
    parser.add_argument(
        "-ip",
        "--input-data-path",
        type=Path,
        required=True,
        help="The path to a folder of test images. Allowing a path to a single image is not supported yet.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Path to save the visualization of test results.",
    )
    parser.add_argument(
        "-c",
        "--conf-thres",
        type=float,
        default=0.5,
        help="Filter out detection results with confidence lowe than this value.",
    )
    parser.add_argument(
        "-i",
        "--iou-thres",
        type=float,
        default=0.5,
        help="When two object has iou higher than this value only keep one of them.",
    )
    args = parser.parse_args(cli_args)
    return args


if __name__ == "__main__":
    test_tflite(**vars(parse_args()))
