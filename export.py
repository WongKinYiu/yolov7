
import argparse
import sys
import time
import warnings

import numpy as np
import os
sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import models
from models.experimental import attempt_load, End2End
from models.experimental_rv import End2EndRVFixedOutput, End2EndRVFixedOutput_TorchScript, End2EndRVFillOutput_TRT, Model_inp_wrapper_2nd  #, End2EndRV,
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from utils.add_nms import RegisterNMS

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true', help='export end2end onnx') # For TRT NMS only not ORT
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    parser.add_argument('--core-ml', action='store_true', help='')
    parser.add_argument('--no-tir-signal', action='store_true', help='')
    parser.add_argument('--include-nms-for-torchscript', action='store_true', help='')

    parser.add_argument('--tir-channel-expansion', action='store_true', help='drc_per_ch_percentile')

    parser.add_argument('--input-channels', type=int, default=1, help='')

    parser.add_argument('--save-path', default='/mnt/Data/hanoch', help='save to project/name')

    parser.add_argument('--flip-channel-from-tf-to-pt', action='store_false', help='')

    parser.add_argument('--reorder-output-class-to-viz-od', action='store_true', help='') #  "person": 0, "car": 1,   "bike": 2,   "animal": 3, "locomotive": 4, "braking_shoe": 5


    opt = parser.parse_args()

    if opt.tir_channel_expansion: # operates over 3 channels
        opt.input_channels = 3

    if opt.tir_channel_expansion and opt.norm_type != 'single_image_percentile_0_1': # operates over 3 channels
        print('Not a good combination')


    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
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
    if opt.flip_channel_from_tf_to_pt:
        if 1:
            path  = '/mnt/Data/hanoch/tir_frames_rois/yolo7_tir_data_all/TIR9_v20_Dec18_Test22C_20181128_041121_FS_210F_0001_5989_AVISHAY_center_roi_220_994.tiff'
            import tifffile
            from utils.datasets import scaling_image
            img = tifffile.imread(path)
            img = scaling_image(img, scaling_type='single_image_percentile_0_1', percentile=0.3)
            img = torch.tensor(img[np.newaxis, :,:, np.newaxis].astype(np.float32)).to(device)
            img.to(device)
        else:
            img = torch.zeros(opt.batch_size, *opt.img_size, opt.input_channels).to(device)  # TF  order (B,W,H,C) => (B,C,W,H) =>
        model = Model_inp_wrapper_2nd(model=model)
    else:
        img = torch.zeros(opt.batch_size, opt.input_channels, *opt.img_size).to(
            device)  # image size(1,3,320,192) iDetection


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
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run
    if opt.include_nms:
        model.model[-1].include_nms = True
        y = None

    if opt.include_nms_for_torchscript:
        # torchscript nms:
        model=End2EndRVFixedOutput_TorchScript(model=model, device=device)
        model = model.eval()
        y = model(img)  # dry run

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        fname = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img, strict=False)
        ts.save(fname)
        print('TorchScript export success, saved as %s' % fname)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    if opt.core_ml:
        # CoreML export
        try:
            import coremltools as ct

            print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
            # convert model from torchscript and apply pixel scaling as per detect.py
            ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])]) # HK@@ TODO modify normalization apparoch or remove from onnx
            bits, mode = (8, 'kmeans_lut') if opt.int8 else (16, 'linear') if opt.fp16 else (32, None)
            if bits < 32:
                if sys.platform.lower() == 'darwin':  # quantization only supported on macOS
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress numpy==1.20 float warning
                        ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
                else:
                    print('quantization only supported on macOS, skipping...')

            fname = opt.weights.replace('.pt', '.mlmodel')  # filename
            ct_model.save(f)
            print('CoreML export success, saved as %s' % f)
        except Exception as e:
            print('CoreML export failure: %s' % e)
                     
    # TorchScript-Lite export
    try:
        print('\nStarting TorchScript-Lite export with torch %s...' % torch.__version__)
        fname = opt.weights.replace('.pt', '.torchscript.ptl')  # filename
        tsl = torch.jit.trace(model, img, strict=False)
        tsl = optimize_for_mobile(tsl)
        tsl._save_for_lite_interpreter(fname)
        print('TorchScript-Lite export success, saved as %s' % fname)
    except Exception as e:
        print('TorchScript-Lite export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        fname = opt.weights.replace('.pt', '.onnx')  # filename
        model.eval()
        output_names = ['classes', 'boxes'] if y is None else ['output']
        dynamic_axes = None
        if opt.dynamic:
            dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
             'output': {0: 'batch', 2: 'y', 3: 'x'}}
        if opt.dynamic_batch:
            opt.batch_size = 'batch'
            dynamic_axes = {
                'images': {
                    0: 'batch',
                }, }
            if opt.end2end and opt.max_wh is None:
                output_axes = {
                    'num_dets': {0: 'batch'},
                    'det_boxes': {0: 'batch'},
                    'det_scores': {0: 'batch'},
                    'det_classes': {0: 'batch'},
                }
            else:
                output_axes = {
                    'output': {0: 'batch'},
                }
            dynamic_axes.update(output_axes)
        if opt.grid:
            if opt.end2end:
                print('\nStarting export end2end onnx model for %s...' % 'TensorRT' if opt.max_wh is None else 'onnxruntime')
                model = End2End(model=model, max_obj=opt.topk_all, iou_thres=opt.iou_thres, score_thres=opt.conf_thres,
                                max_wh=opt.max_wh, device=device, n_classes=len(labels),
                                reorder_output_class_to_viz_od=opt.reorder_output_class_to_viz_od)

                if opt.end2end and opt.max_wh is None:
                    if 1:  # Efficinet NMS TRT Yuval patch
                        model = End2EndRVFillOutput_TRT(model=model, device=device)

                    output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                    shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                              opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
                else: # ORT
                    if 1: # Yuval patch
                        model = End2EndRVFixedOutput(model=model, device=device)
                    #     Output format [X, selected_boxes_xyxy, selected_categories, selected_scores]
                    output_names = ['output']
            else:
                model.model[-1].concat = True

        # Enable the profiler
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            # Run your model's forward pass or any other computations
            t1 = time_synchronized()
            output = model(img)
            t2 = time_synchronized()
            print(f'inference time: ({(1E3 * (t2 - t1)):.1f}ms) ')
        # Disable the profiler and get the report
        print(prof)

        print("onnx:")
        print(img.shape)

        opset_version = 12 #14
        if opt.reorder_output_class_to_viz_od:
            fname = os.path.join(os.path.dirname(fname), os.path.basename(fname).split('onnx')[0] + 'person_cls_0.onnx')

        torch.onnx.export(model, img, fname, verbose=False, opset_version=opset_version, input_names=['images'],
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)

        # Checks
        onnx_model = onnx.load(fname)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        # import onnxruntime as ort
        # cuda = [True if torch.cuda.is_available() else False][0]
        # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        # session = ort.InferenceSession(fname, providers=providers)
        # input_shape = session.get_inputs()[0].shape
        # outname = [i.name for i in session.get_outputs()]
        # inname = [i.name for i in session.get_inputs()]
        # inp = {inname[0]: img.cpu().numpy()}
        # output2 = session.run(outname, inp)

        if opt.end2end and opt.max_wh is None:
            for i in onnx_model.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

        # # Metadata
        # d = {'stride': int(max(model.stride))}
        # for k, v in d.items():
        #     meta = onnx_model.metadata_props.add()
        #     meta.key, meta.value = k, str(v)
        # onnx.save(onnx_model, f)

        if opt.simplify:
            try:
                import onnxsim

                print('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                print(f'Simplifier failure: {e}')

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        onnx.save(onnx_model, fname)
        print('ONNX export success, saved as %s' % fname)

        if opt.max_wh is not None and 0:
            if opt.include_nms:
                print('Registering NMS plugin for ONNX...')
                mo = RegisterNMS(fname)
                mo.register_nms(score_thresh=0.66, nms_thresh=0.6)  #/mnt/Data/hanoch/runs/train/yolov7800_11/weights/epoch_099.onnx
                mo.save(fname)

    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))

"""
--include-nms --device 0 --weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.pt --batch-size 1 --end2end --grid --conf-thres 0.5 --iou-thres 0.6 --simplify --max-wh 640
--include-nms --device 0 --weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.pt --batch-size 1 --end2end --grid --conf-thres 0.1 --iou-thres 0.6 --simplify --max-wh 640
0.1 minimal filtering of proposals
Created NMS plugin 'EfficientNMS_TRT' with attributes: {'plugin_version': '1', 'background_class': -1, 'max_output_boxes': 100, 'score_threshold': 0.66, 'iou_threshold': 0.6, 'score_activation': False, 'box_coding': 0}

issue with default EfficientNMS_TRT where ORT can't load 
https://github.com/microsoft/onnxruntime/issues/16121
When loading model with ORT
onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph: [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from /mnt/Data/hanoch/runs/train/yolov7999/weights/best.onnx failed:This is an invalid model. In Node, ("batched_nms", EfficientNMS_TRT, "", -1) : ("output": tensor(float),) -> ("num_dets": tensor(int32),"det_boxes": tensor(float),"det_scores": tensor(float),"det_classes": tensor(int32),) , Error Node (batched_nms) has input size 1 not in range [min=2, max=3].

pip install onnxruntime-gpu


desired is nms_ort
None instead 640 not taking the TRT but ORT (nms of NVIDIA)

--include-nms --device 0 --weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.pt --batch-size 1 --end2end --grid --conf-thres 0.1 --iou-thres 0.6 --simplify --max-wh 640
# reorder outputs

--include-nms --device 0 --weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.pt --batch-size 1 --end2end --grid --conf-thres 0.1 --iou-thres 0.6 --simplify --max-wh 640 --reorder-output-class-to-viz-od

# 3 classes 
/mnt/Data/hanoch/runs/train/yolov71351/weights/epoch_049.pt
--include-nms --device 0 --weights /mnt/Data/hanoch/runs/train/yolov71351/weights/epoch_049.pt --batch-size 1 --end2end --grid --conf-thres 0.1 --iou-thres 0.6 --simplify --max-wh 640 --reorder-output-class-to-viz-od

"""
