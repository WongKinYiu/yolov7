from leap_binder import (preprocess_func, input_encoder, gt_encoder, custom_loss,
                         pred_visualizer, gt_visualizer, metadata_image_path,
                         image_stats, preprocess_func_no_csv, accuracy_f1,
                         preprocess_unlabeled_func_leap)

from code_loader.contract.datasetclasses import DataStateType

# import tensorflow as tf
import numpy as np
from utils.general import non_max_suppression
from leap_utils import bb_array_to_object, draw_image_with_boxes
import torch
from code_loader.contract.visualizer_classes import LeapImageWithBBox
import matplotlib
# matplotlib.use('Tkagg')
import onnx
import onnxruntime as ort

from code_loader.default_metrics import flatten_non_batch_dims
from leap_binder import leap_binder

def check_integration():

    image_index = 1 #100
    leap_binder.check()
    nc = 3
    model_path = '/mnt/Algo/Hanoch/onnx_versions/TIR/tir_od_3_classes/tir_od_3cls_unsqueeze_fixed.onnx' #'/mnt/Data/hanoch/runs/train/yolov71351/weights/epoch_049.onnx'
    if 1:
        res = preprocess_func_no_csv()
    else:
        res = preprocess_func()

    res.append(preprocess_unlabeled_func_leap())

    for train in res:
        train = res[-1]
        for image_index in range(10):
            # image_index = 102
            meta_dat = metadata_image_path(image_index, train)
            # targt_path = '/mnt/Data/hanoch/tir_frames_rois/yolo7_tir_data_all/TIR11_V50_JUL21_Test38D_2021_02_03_16_11_48_FS_210_XGA_3062_5512_LIEL_center_roi_210_3621.tiff'
            # if targt_path != meta_dat['path']:
            #     continue
            inpt = input_encoder(image_index, train)[None, ...]
            if 1:
                cuda = [True if torch.cuda.is_available() else False][0]
                if 0:
                    providers = [('TensorrtExecutionProvider', {'device_id': 0}), 'CUDAExecutionProvider'] if cuda else [
                        'CPUExecutionProvider']
                else:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
                # if isinstance(opt.weights, list):
                #     opt.weights = opt.weights[0]

                session = ort.InferenceSession(model_path, providers=providers)
                input_shape = session.get_inputs()[0].shape
                print('Input shape : ', input_shape)
                outname = [i.name for i in session.get_outputs()]
                inname = [i.name for i in session.get_inputs()]
                inp = {inname[0]: inpt}
                prediction = session.run(outname, inp)[0]
                if train.state != DataStateType.unlabeled:
                    gt = gt_encoder(image_index, train)[None, ...]
                # print(train.data['dataset1'].label_files[0])
                # out2 = train.data['dataset1'].__getitem__(0)
                    image_stats_real = image_stats(gt)
                    loss = custom_loss(prediction, gt)
                    acc = accuracy_f1(prediction, gt)

                    gt_img_with_bbox = gt_visualizer(gt, inpt)
                img_with_bbox = pred_visualizer(prediction, inpt)



if __name__ == '__main__':
    check_integration()
