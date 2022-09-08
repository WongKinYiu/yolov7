import argparse
from importlib.util import module_for_loader
import sys
import time
import warnings

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.mobile_optimizer import optimize_for_mobile

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from utils.datasets import CalibratorImages
import tensorrt as trt
import numpy as np
from functools import reduce
from torch2trt.calibration import DatasetCalibrator, TensorBatchDataset

def get_module_by_name(model, access_string):
    names = access_string.split('.')[:-1]
    return reduce(getattr, names, model)

def convert_to_engine(onnx_f, im, sparsify=False, int8=False, half=False, int8_calib_dataset=None, calib_batch_size=4, workspace=28, calib_algo='entropy2', end2end=False, conf_thres=0.45, iou_thres=0.25, max_det=100):
    prefix = 'TensorRT:'
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << workspace
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(onnx_f)
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    print(f'{prefix} Network Description:')
    for inp in inputs:
        print(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
    for out in outputs:
        print(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')
    
    if end2end:
        previous_output = network.get_output(0)
        network.unmark_output(previous_output)
        strides = trt.Dims([1,1,1])
        starts = trt.Dims([0,0,0])
        bs, num_boxes, temp = previous_output.shape
        shapes = trt.Dims([bs, num_boxes, 4])
        boxes = network.add_slice(previous_output, starts, shapes, strides)
        num_classes = temp - 5
        starts[2] = 4
        shapes[2] = 1
        obj_score = network.add_slice(previous_output, starts, shapes, strides)
        starts[2] = 5
        shapes[2] = num_classes
        scores= network.add_slice(previous_output, starts, shapes, strides)
        updated_scores = network.add_elementwise(obj_score.get_output(0), scores.get_output(0), trt.ElementWiseOperation.PROD)
        
        registry = trt.get_plugin_registry()
        assert(registry)
        creator = registry.get_plugin_creator("EfficientNMS_TRT","1")
        assert(creator)
        
        fc = []
        fc.append(trt.PluginField("background_class", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("max_output_boxes", np.array([max_det], dtype=np.int32), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("score_threshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
        fc.append(trt.PluginField("iou_threshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
        fc.append(trt.PluginField("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
        
        fc = trt.PluginFieldCollection(fc)
        nms_layer = creator.create_plugin("nms_layer", fc)
        
        layer = network.add_plugin_v2([boxes.get_output(0), updated_scores.get_output(0)], nms_layer)
        layer.get_output(0).name = "num"
        layer.get_output(1).name = "boxes"
        layer.get_output(2).name = "scores"
        layer.get_output(3).name = "classes"
        for i in range(4):
            network.mark_output(layer.get_output(i))
    
    f = onnx_f.replace('onnx','engine')
    print(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half and not int8 else 32} engine in {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)

    inputs_in = im
    if not isinstance(im, tuple):
        im = (im,)

    if int8:
        if int8_calib_dataset is None:
            int8_calib_dataset = TensorBatchDataset(inputs_in)
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if builder.platform_has_fast_int8:
            print(f'{prefix} building INT8 engine in {f}')
            config.set_flag(trt.BuilderFlag.INT8)
        
        if calib_algo=='entropy2':
            algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
        elif calib_algo == 'entropy':
            algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION
        else:
            algo = trt.CalibrationAlgoType.MINMAX_CALIBRATION
        calibrator = DatasetCalibrator(
            im, int8_calib_dataset, batch_size=calib_batch_size, algorithm=algo
        )

        config.int8_calibrator = calibrator

    if sparsify:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end_trt', action='store_true', help='export end2end trt (include NMS)')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='conf threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--opset', default=12, type=int, help='ONNX opset version')
    parser.add_argument('--fp16', action='store_true', help='TensorRT FP16 half-precision export')
    parser.add_argument('--sparsify', action='store_true', default=False, help='sparsify model')
    parser.add_argument('--prop', type=float, default=0.3, help='sparsification amount')
    parser.add_argument('--struct', default=False, action='store_true', help='structured sparsification')
    parser.add_argument('--int8', action='store_true', help='TensorRT INT8 quantization')
    parser.add_argument('--calibrate', default=False, action='store_true')
    parser.add_argument('--calib-num-images', default=200, type=int, help='number of images to be used for INT8 calibration')
    parser.add_argument('--calib-batch-size', default=4, type=int, help='INT8 calibration batch size')
    parser.add_argument('--calib-algo', default='entropy2', type=str, choices=['entropy','entropy2','minmax'], help='INT8 calibration batch size')
    parser.add_argument('--trt', action='store_true', default=False, help='enable TensorRT optimization')
    parser.add_argument('--workspace', default=28, type=int, help='set workspace size')
    parser.add_argument('--seed', type=int, default=10, help='seed for INT8 calibration')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    
    print(opt)
    set_logging()
    t = time.time()
    
    # Model sparsification
    if opt.sparsify:
        print("Sparsifying model")
        model = torch.load(opt.weights)
        ckpt = {}
        for k in model.keys():
            ckpt[k] = model[k]
        m = model['model'].float().cuda()
        modules = [module[0] for module in m.named_parameters()]
        parameters_to_prune = []
        for module in modules:
            if 'weight' in module:
                obj = get_module_by_name(m, module)
                if opt.struct and not isinstance(obj, nn.BatchNorm2d):
                    prune.ln_structured(obj, name='weight', amount=opt.prop, n=1, dim=0)
                    prune.remove(obj, 'weight')
                else:
                    parameters_to_prune.append((obj, 'weight'))
        if not opt.struct:        
            parameters_to_prune = tuple(parameters_to_prune)
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=opt.prop)
            for param in parameters_to_prune:
                prune.remove(*param)
            
        ckpt['model'] = m
        opt.weights = opt.weights.replace('.pt', '_pruned.pt')
        torch.save(ckpt, opt.weights)
        print(f"Model sparsification successful. Saved as {opt.weights}")

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  
    labels = model.names
    
    # Checks
    gs = int(max(model.stride))  
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device) 

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  
        if isinstance(m, models.common.Conv):  
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    model.model[-1].export = not opt.grid
    
    
    y = model(img)
    
    

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        model.eval()
        
        if opt.grid:
            model.model[-1].concat = True

        torch.onnx.export(model, img, f, verbose=False, opset_version=opt.opset, input_names=['images'],
                          output_names=['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        if opt.simplify:
            try:
                import onnxsim

                print('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                print(f'Simplifier failure: {e}')

        
        onnx.save_model(onnx_model, f)
        onnx.save(onnx_model,f)
        
        print('ONNX export success, saved as %s' % f)
        
    except Exception as e:
        print('ONNX export failure: %s' % e)
    
    # TensorRT export
    if opt.trt:
        try:
            if opt.int8 and opt.calibrate:
                calib_dataset = CalibratorImages('../datasets/coco/val2017/*.jpg', auto=False, num_images=opt.calib_num_images, seed=opt.seed)
            else:
                calib_dataset = None
            convert_to_engine(f, 
                            img,
                            sparsify=opt.sparsify, 
                            half=opt.fp16, 
                            int8=opt.int8, 
                            int8_calib_dataset=calib_dataset, 
                            calib_batch_size=opt.calib_batch_size, 
                            workspace=opt.workspace, 
                            calib_algo=opt.calib_algo, 
                            end2end=opt.end2end_trt, 
                            iou_thres=opt.iou_thres, 
                            conf_thres=opt.conf_thres, 
                            max_det=opt.topk_all)
        except Exception as e:
            print("TensorRT export failure: %s" % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
