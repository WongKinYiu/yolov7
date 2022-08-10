"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import cv2
import numpy as np

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import colorstr, check_img_size, check_requirements, file_size, set_logging
from utils.torch_utils import select_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')  # ONNX-only
    parser.add_argument('--simplify', action='store_true', help='simplify ONNX model')  # ONNX-only
    parser.add_argument('--export-nms', action='store_true', help='export the nms part in ONNX model')  # ONNX-only, #opt.grid has to be set True for nms export to work
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection
    # img = cv2.imread("/user/a0132471/Files/bit-bucket/pytorch/jacinto-ai-pytest/data/results/datasets/pytorch_coco_mmdet_img_resize640_val2017_5k_yolov5/images/val2017/000000000139.png")
    # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    # img = np.ascontiguousarray(img)
    # img = torch.tensor(img[None,:,:,:], dtype = torch.float32)
    # img /= 255

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not (opt.grid or opt.export_nms) # set Detect() layer grid export
    for _ in range(2):
        y = model(img)  # dry runs
    output_names = None
    if opt.export_nms:
        nms = models.common.NMS(conf=0.01, kpt_label=True)
        nms_export = models.common.NMS_Export(conf=0.01, kpt_label=True)
        y_export = nms_export(y)
        y = nms(y)
        #assert (torch.sum(torch.abs(y_export[0]-y[0]))<1e-6)
        model_nms = torch.nn.Sequential(model, nms_export)
        model_nms.eval()
        output_names = ['detections']

    print(f"\n{colorstr('PyTorch:')} starting from {opt.weights} ({file_size(opt.weights):.1f} MB)")

    # TorchScript export -----------------------------------------------------------------------------------------------
    prefix = colorstr('TorchScript:')
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img, strict=False)
        ts = optimize_for_mobile(ts)  # https://pytorch.org/tutorials/recipes/script_optimized.html
        ts.save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

    # ONNX export ------------------------------------------------------------------------------------------------------
    prefix = colorstr('ONNX:')
    try:
        import onnx

        print(f'{prefix} starting export with onnx {onnx.__version__}...')
        f = opt.weights.replace('.pt', '.onnx')  # filename
        if opt.export_nms:
            torch.onnx.export(model_nms, img, f, verbose=False, opset_version=11, input_names=['images'], output_names=output_names,
                              dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                            'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None)
        else:
            torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'], output_names=output_names,
                              dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                            'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None)


        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if opt.simplify:
            try:
                check_requirements(['onnx-simplifier'])
                import onnxsim

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=opt.dynamic,
                                                     input_shapes={'images': list(img.shape)} if opt.dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

    # CoreML export ----------------------------------------------------------------------------------------------------
    prefix = colorstr('CoreML:')
    try:
        import coremltools as ct

        print(f'{prefix} starting export with coremltools {ct.__version__}...')
        # convert model from torchscript and apply pixel scaling as per detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        model.save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s). Visualize with https://github.com/lutzroeder/netron.')
