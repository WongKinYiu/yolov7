import argparse

import onnx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract subgraph from onnx model")
    parser.add_argument('-i','--input_path', help='Path to input onnx model', default="yolov7-4d-trained-640-nms.onnx", required=False)
    parser.add_argument('-o','--output_path', help='Path to output extracted onnx model', default="yolov7-4d-trained-640.onnx", required=False)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    input_names = ["images"]
    output_names = ["/model/model.105/Concat_6_output_0"]
    onnx.utils.extract_model(input_path, output_path, input_names, output_names)