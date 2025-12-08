import sys
import torch
from utils.torch_utils import select_device, time_synchronized, TracedModel
from tqdm import tqdm
from torchvision.ops import batched_nms
import matplotlib.pyplot as plt
import cv2
import pandas as pd
#%%
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
import argparse
from utils.datasets import create_dataloader, create_folder

from utils.datasets import LoadStreams, LoadImages, scaling_image, LoadImagesAddingNoiseAndLabels, LoadImagesAndLabels, InfiniteDataLoader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.metrics import ap_per_class
from utils.general import box_iou
from utils.plots import plot_one_box
import pickle
import os
from utils.general import xywh2xyxy
from collections import defaultdict

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

# Compute precision, recall, and AP for a single class
def compute_precision_recall_ap(preds, gts, iou_threshold=0.5):
    preds = sorted(preds, key=lambda x: x[2], reverse=True)  # Sort by confidence

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    matched = set()
    thresholds = []

    for i, (pred_box, pred_cls, confidence) in enumerate(preds):
        best_iou = 0
        best_gt_idx = -1
        thresholds.append(confidence)

        for gt_idx, (gt_box, gt_cls) in enumerate(gts):
            if gt_idx in matched:
                continue

            if pred_cls == gt_cls:
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            tp[i] = 1
            matched.add(best_gt_idx)
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / len(gts)

    # Compute AP using the precision-recall curve
    ap = 0
    for i in range(1, len(precision)):
        ap += (recall[i] - recall[i - 1]) * precision[i]

    return precision, recall, ap, np.array(thresholds)

# Compute mAP and plot precision-recall curve
def compute_map(predictions, ground_truths, num_classes,
                iou_threshold=0.5, conf_th=-1,
                min_precision_acceptible=0.9, save_dir='', class_enum=''):
    aps = []
    plt.figure(figsize=(10, 7))

    for cls in range(num_classes):
        preds = [(box, cls_id, conf) for box, cls_id, conf in predictions if cls_id == cls]
        gts = [(box, cls_id) for box, cls_id in ground_truths if cls_id == cls]

        if len(gts) == 0:
            continue

        precision, recall, ap, thresholds = compute_precision_recall_ap(preds, gts, iou_threshold)
        aps.append(ap)
        if 1:
            if class_enum:
                plt.plot(recall, precision, label=f"Class {class_enum[cls]} (AP={ap:.2f})")
            else:
                plt.plot(recall, precision, label=f"Class {cls} (AP={ap:.2f})")

            inds = [i for i in range(0, np.where(precision < min_precision_acceptible)[0][-1].item())]
            print('Pr, Re Th class{}'.format(cls))
            [(precision[i], recall[i], thresholds[i]) for i in inds]
            for i in range(0, np.where(precision < min_precision_acceptible)[0][-1].item(), int(len(inds)/20)):
                print('Th {:.2f}, Pr {:.2f}, Re {:.2f}'.format(thresholds[i], 100*precision[i], 100*recall[i]))
                plt.annotate(f'{thresholds[i]:.2f}',
                             xy=(recall[i], precision[i]),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8)
                # plt.savefig(os.path.join(save_dir, 'precision_recall_thresholds_class_' + str(cls) + '.png'))

        else:
            plt.plot(recall, precision, label=f"Class {cls} (AP={ap:.2f})")
            os.path.join(save_dir, 'precision_recall_class_' + str(cls)+ '.png')

    mAP = np.mean(aps)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (mAP={mAP:.2f}) conf-th=={conf_th:.2f}")
    plt.legend()
    plt.grid()
    plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'map.png'))

    return mAP



def yolobbox_to_xyxy(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return [int(x1), int(y1), int(x2), int(y2)]

coco_lass_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

reshape = False

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def main(opt):

    create_folder(opt.save_path)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in coco_lass_names[:10]]
    pred_tgt_acm = list()
    p_r_iou = 0.5
    niou = 1
    # Model
    cuda = [True if torch.cuda.is_available() else False][0]
    if 0:
        providers = [('TensorrtExecutionProvider', {'device_id': 0}), 'CUDAExecutionProvider'] if cuda else ['CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    if isinstance(opt.weights, list):
        opt.weights = opt.weights[0]

    session = ort.InferenceSession(opt.weights, providers=providers)
    input_shape = session.get_inputs()[0].shape
    print('Input shape : ', input_shape)
    is_new_model = np.array(([True if ix==640 else False for ix in input_shape])).any().item()
    # sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.intra_op_num_threads = 8  # Limit the number of threads
    # sess = ort.InferenceSession(onnx_file, sess_options, providers=["CPUExecutionProvider"])


    test_path = opt.test_files_path
    # imgsz_test = opt.img_size #512 #(512, 512, 3)
    if opt.img_size>-1 and bool(input_shape):
        print('!!!!!!!!!!!!!!!!  model has input size defined in ONNX  overrideen  ? ')
    imgsz_test = np.unique([x for x in input_shape if x >1])[0].item()
    # dataset_image_size = [640, 640]
    batch_size = 16
    gs = 32 #gs = max(int(model.stride.max()), 32)

    uniq_class_subset = ['Car', 'Person']
    coco_mapping = {ix : coco_lass_names.index(x.lower()) for ix, x in  enumerate(uniq_class_subset)}

    opt.device = torch.device("cuda:" + str(opt.device) + "" if torch.cuda.is_available() else "cpu")

    hyp = dict()
    hyp['person_size_small_medium_th'] = 32 * 32
    hyp['car_size_small_medium_th'] = 44 * 44

    hyp['img_percentile_removal'] = 0.3
    hyp['beta'] = 0.3
    hyp['gamma'] = 80  # dummy anyway augmentation is disabled
    hyp['gamma_liklihood'] = 0.01
    hyp['random_pad'] = True
    hyp['copy_paste'] = False
    nc = 2

    images_parent_folder = opt.images_parent_folder
    opt.csv_metadata_path = ''
    print(colorstr('val: '))
    opt.single_cls = False
    opt.input_channels = 1
    opt.tir_channel_expansion = False
    opt.no_tir_signal = False
    detection_res = list()
    reorder_output_class_to_viz_od_dict = {0: 1, 1: 0}

    if opt.detection_no_gt:
        dataloader = LoadImages(images_parent_folder, img_size=imgsz_test, stride=32,
                             scaling_type=opt.norm_type, input_channels=opt.input_channels,
                             no_tir_signal=opt.no_tir_signal,
                             tir_channel_expansion=opt.tir_channel_expansion)
        plot = True
    else:
        if opt.adding_ext_noise:
            hyp['gamma_liklihood'] = 0.5
            scaling_before_mosaic = bool(hyp.get('scaling_before_mosaic', False))
            world_size = 1
            workers = 8
            rank = -1
            image_weights = False
            dataset = LoadImagesAddingNoiseAndLabels(path=test_path, img_size=imgsz_test,
                                                        batch_size=batch_size, stride=gs,  # testloader
                                                        hyp=hyp, cache_images=opt.cache_images, rect=False,
                                                        pad=0.5, prefix=colorstr('val: '),
                                                        rel_path_images=images_parent_folder,
                                                        scaling_type=opt.norm_type,
                                                        input_channels=opt.input_channels,
                                                        num_cls=nc,
                                                        tir_channel_expansion = opt.tir_channel_expansion,
                                                        no_tir_signal = opt.no_tir_signal,
                                                        scaling_before_mosaic = scaling_before_mosaic,
                                                        path_noisy_samples=opt.noise_parent_folder)

            batch_size = min(batch_size, len(dataset))
            nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
            sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
            loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader

            # batch_size = min(batch_size, len(dataset))
            # nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
            # sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
            # loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
            # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()

            dataloader = loader(dataset,
                                batch_size=batch_size,
                                num_workers=nw,
                                sampler=sampler,
                                pin_memory=True,
                                collate_fn= LoadImagesAndLabels.collate_fn)

        else:
            dataloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt,  # testloader
                                                 hyp=hyp, cache=opt.cache_images, rect=False, rank=-1,
                                                 # @@@ rect was True why?
                                                 world_size=1, workers=8,
                                                 pad=0.5, prefix=colorstr('val: '),
                                                 rel_path_images=images_parent_folder, num_cls=nc)[0]

    seen = 0
    t0 = 0

    test_vector_image_resolution = [768, 1024]
    test_image = '/mnt/Data/hanoch/tir_old_tf/ODFrame_prod_onnx_tir_v1_tf_test_vector.png'
    test_vector_ref_detections = { 'BBox': [383.031, 248.378, 39.7324, 87.0982], 'Score': 0.982154,
                                   'Class': 'Person', 'crop_coordination_xy':[256, 128],
                                   'tol_box':0.1, 'tol_conf':0.2, 'detection_threshold': 0.8,
                                   'test_image': test_image}


    det_threshold = opt.conf_thres
    nms_ious_th = opt.iou_thres

    predictions = list()
    ground_truths = list()
    stats = []
    seen = 0
    names = ['car', 'person']

    test_vectors_test = opt.test_vectors_test_old_tir_1p5
    # Batchwise
    for img, targets, paths, shapes in tqdm(dataloader):
        if bool(opt.detection_no_gt):
            paths, img, im0s, _ = img, targets, paths, shapes
            # del targets
            img = torch.from_numpy(img[None, ...]).to(opt.device)

        img = img.cpu().numpy().astype('float32').transpose(0, 2,3,1) # channel last
        nb, _, height, width = img.shape  # batch size, channels, height, width
        with (torch.no_grad()):
            # Run model
            t = time_synchronized()
            outname = [i.name for i in session.get_outputs()]
            inname = [i.name for i in session.get_inputs()]
            # print(inname, outname)
            # image wise inside batch
            for ix, im in enumerate(img):
                seen += 1
                # #############  TEST VECTOR SANITY for OLD MODEL
                if test_vectors_test:
                    im = cv2.imread(test_image)  # BGR
                    if not(opt.no_tir_signal):
                        im = im[:, :, :1]  # channels are duplicated in the source
                        # im = im[:, :, np.newaxis]
                        if reshape:
                            im = cv2.resize(im, (imgsz_test, imgsz_test))[:, :, np.newaxis]
                        else:
                            # im = im[test_vector_ref_detections['crop_coordination'][0]:test_vector_ref_detections['crop_coordination'][0]+512,
                            #      test_vector_ref_detections['crop_coordination'][1]:test_vector_ref_detections['crop_coordination'][1]+512]
                            # im = im*255
                            im = im[test_vector_ref_detections['crop_coordination_xy'][1]:
                                    test_vector_ref_detections['crop_coordination_xy'][1] + 512,
                                     test_vector_ref_detections['crop_coordination_xy'][0]:
                                     test_vector_ref_detections['crop_coordination_xy'][0] + 512]
                        if 0:
                            im = im/255.0
                        im = im.astype('float32')
                else:
                    if opt.norm_type == 'no_norm': # old TIR model given dataloader not test vector
                        # im = im /(2**16 -1)
                        im = im.astype('float32')
                    else:
                        im = im.astype('float32') # New TIR model with specific normalization

                inp = {inname[0]: im[np.newaxis, :, :, :]}
                output = session.run(outname, inp)
                sigmoid_out = output[0]
                # BBOX CV2 is upper x,y and w, h while YOLO is center x,y
                if len(output) == 1:
                    output = output[0]


                if is_new_model:
                    nn_out = output #[batch_id, selected_boxes_xyxy, selected_category, selected_score]
                    if opt.reorder_output_class_to_viz_od:
                        out_all = list()
                        for x in nn_out:
                            out_all.append(
                                np.append(np.append(x[:5], reorder_output_class_to_viz_od_dict[x[5]]), x[6:]))
                        nn_out = np.stack(out_all, axis=0)

                    # new TIR model packing all outputs on the same array
                    scores_above_th_val = nn_out[[nn_out[:, -1] > det_threshold][0], -1]
                    scores_above_th = nn_out[:, -1] > det_threshold

                    bboxes = nn_out[scores_above_th, 1:5]
                    ml_class_id = nn_out[scores_above_th, 5].astype('int')
                else:

                    det_threshold = test_vector_ref_detections['detection_threshold']
                    if opt.conf_thres != det_threshold:
                        det_threshold = opt.conf_thres
                        # print('User changed ONNX old version threshold !!!!!!!!!!!!!!!! ', 10*'!!!#33')
                    bbox_out = output[1]
                    class_id = np.argmax(sigmoid_out, 2)
                    scores = np.max(sigmoid_out, 2)
                    scores_above_th = scores[scores > det_threshold]
                    bboxes = bbox_out[scores>det_threshold]
                    ml_class_id = class_id[scores > det_threshold].astype('int')
                if is_new_model:

                    # preds = [(bboxes_normalized_coord_detected_2_nms[x].numpy(), ml_class_id[x].item(),
                    #           scores_above_th[x].item()) for x in nms_out]
                    #
                    predictions.extend([(bboxes_, ml_class_id_.item(), scores_above_th_.item()) for bboxes_ ,ml_class_id_, scores_above_th_ in zip(bboxes ,ml_class_id, scores_above_th_val)])
                    if isinstance(targets, torch.Tensor):
                        img_gt_lbls = targets[targets[:, 0] == ix, 1].cpu().numpy()
                    else:
                        img_gt_lbls = targets[targets[:, 0] == ix, 1]

                    img_gt_boxes = targets[targets[:, 0] == ix, 2:]  # YOLO bbox are relative coordination (center x,y and w,h ) hence resolution independant
                    pred = torch.tensor([np.append(np.append(bboxes_, scores_above_th_.item()), ml_class_id_.item()) for bboxes_, ml_class_id_, scores_above_th_ in
                                                    zip(bboxes, ml_class_id, scores_above_th_val)])
                    if bool(opt.detection_no_gt):
                        im0s = detection_plot(colors, dataloader, detection_res, im0s, opt, paths, plot, pred,
                                              uniq_class_subset)

                    else:
                        img_gt_boxes_xyxy = [yolobbox_to_xyxy(*yolo_bb, imgsz_test, imgsz_test) for yolo_bb in img_gt_boxes]
                        # img_gt_boxes_xywh = [[x[0], x[1],x[2]-x[0], x[3]-x[1]] for x in img_gt_boxes_xyxy]

                        # Ground truths: [(bbox, class_id)]
                        ground_truths.extend([(x, y) for x, y in zip(img_gt_boxes_xyxy, img_gt_lbls)])
                        # ******************
                        # Like test.py
                        labels = torch.tensor([[y.item()] + x for x, y in zip(img_gt_boxes_xyxy, img_gt_lbls)])
                        nl = len(labels)
                        # if nl == ml_class_id.shape[0]:
                        #     print('prob all TP')
                        #     print('path', paths[seen-1])
                        #     predn[pi, :4]
                        tcls = labels[:, 0].tolist() if nl else []  # target class

                        predn = pred.clone()
                        tbox = labels[:, 1:5]
                        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=opt.device)
                        if nl:
                            detected = []  # target indices
                            tcls_tensor = labels[:, 0]
                            # Per target class
                            for cls in torch.unique(tcls_tensor):
                                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                                # Search for detections
                                if pi.shape[0]:
                                    # Prediction to target ious
                                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                                    # Append detections
                                    detected_set = set()
                                    for j in (ious > p_r_iou).nonzero(as_tuple=False): # iouv[0]=0.5 IOU for dectetions iouv in general are all 0.5:0.05:.. for COCO
                                        d = ti[i[j]]  # detected target
                                        if d.item() not in detected_set:
                                            detected_set.add(d.item())
                                            detected.append(d)
                                            correct[pi[j]] = ious[j] > p_r_iou  # iou_thres is 1xn
                                            if len(detected) == nl:  # all targets already located in image
                                                break
                        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(),
                                      tcls))  # correct @ IOU=0.5 of pred box with target
                        pred_tgt_acm.append({'correct': correct.cpu().numpy(), 'conf': pred[:, 4].cpu().numpy(), 'pred_cls': pred[:, 5].cpu().numpy(), 'tcls': tcls} )


                else:
                    if test_vectors_test: # since no labels file annotations are not resized like dat is implicitly
                        if reshape:
                            bboxes_normalized_coord_detected = [[int(i[0].item() * test_vector_image_resolution[1] / imgsz_test),
                                                  int(i[1].item() * test_vector_image_resolution[0] / imgsz_test),
                                                  int(i[2].item() * test_vector_image_resolution[1] / imgsz_test),
                                                  int(i[3].item() * test_vector_image_resolution[0] / imgsz_test)] for i in bboxes]
                        else:
                            bboxes_normalized_coord_detected = bboxes
                            bboxes_normalized_coord_detected[:, 0] += test_vector_ref_detections['crop_coordination_xy'][0]
                            bboxes_normalized_coord_detected[:, 1] += test_vector_ref_detections['crop_coordination_xy'][1]

                        norm_err = [(b - np.array(test_vector_ref_detections['BBox'])) / np.array(test_vector_ref_detections['BBox'])
                                    for b in bboxes_normalized_coord_detected]

                        good_enough = [np.isclose(b, np.array(test_vector_ref_detections['BBox']), rtol=0.02).all() # 2% relative tolerance
                                       # 2% relative tolerance
                                       for b in bboxes_normalized_coord_detected]

                        good_enough_abs_tol = [np.isclose(b, np.array(test_vector_ref_detections['BBox']), atol=test_vector_ref_detections['tol_box']).all() # 2% relative tolerance
                                       # 2% relative tolerance
                                       for b in bboxes_normalized_coord_detected]

                        if np.array([not (i) for i in good_enough]).any():
                            print('!!!!!!!!!!!!! Heuston we have a problem BBOX relative tolerance')

                        if not((np.isclose(scores_above_th, test_vector_ref_detections['Score'])).any()):
                            print('!!!!!!!!!!!!! Heuston we have a problem conf relative tolerance')

                    else:
                        # bboxes_normalized_coord_detected = [[int(i[0].item() * dataset_image_size[1] / imgsz_test), # GT labels are related to 640*640 while NN output isrelated to 512
                        #                       int(i[1].item() * dataset_image_size[0] / imgsz_test),
                        #                       int(i[2].item() * dataset_image_size[1] / imgsz_test),
                        #                       int(i[3].item() * dataset_image_size[0] / imgsz_test)] for i in bboxes]
                        # bboxes_normalized_coord_detected = bboxes
                        # cv2: xywh to xyxyx
                        bboxes_normalized_coord_detected_2_nms = [[int(i[0].item()), # GT labels are related to 640*640 while NN output isrelated to 512
                                               int(i[1].item()), int(i[0].item() + i[2].item()),
                                               int(i[1].item() + i[3].item())] for i in bboxes]
                        # b = xywh2xyxy(bboxes_normalized_coord_detected.reshape(-1, 4)).ravel().astype(np.int)


                        gt = [[0.852344, 0.082031, 0.126562, 0.107813], # 0
                              [0.685937, 0.079687, 0.021875, 0.081250], # 1
                              [0.922656, 0.067187, 0.051562, 0.050000]] # 0

                        # [yolobbox_to_xyxy(gt[0], gt[1], gt[2], gt[3], 640, 640) for gt in gt]
                        # [[505, 17, 586, 87] = > CaR , [431, 24, 445, 76], [574, 26, 606, 58]]

                        bboxes_normalized_coord_detected_2_nms = torch.tensor(bboxes_normalized_coord_detected_2_nms)
                        bboxes_normalized_coord_detected_2_nms = bboxes_normalized_coord_detected_2_nms.to(torch.float)

                    #     list of detections, on (n,6) tensor per image [xyxy, conf, cls]

                    nms_out = batched_nms(boxes=bboxes_normalized_coord_detected_2_nms,
                                          scores=torch.tensor(scores_above_th),
                                          idxs=torch.tensor(ml_class_id),
                                          iou_threshold=nms_ious_th)
                    preds = [(bboxes_normalized_coord_detected_2_nms[x].numpy(), ml_class_id[x].item(), scores_above_th[x].item()) for x in nms_out]
                    predictions.extend(preds)
                    if not bool(opt.detection_no_gt): # no GT in detectons noly
                        img_gt_lbls = targets[targets[:, 0] == ix, 1].cpu().numpy()
                        img_gt_lbls_in_coco_order = [coco_mapping[ele.item()] for ele in img_gt_lbls]
                        img_gt_boxes = targets[targets[:, 0] == ix, 2:] # YOLO bbox are relative coordination (center x,y and w,h ) hence resolution independant
                        img_gt_boxes_xyxy = [yolobbox_to_xyxy(*yolo_bb, imgsz_test, imgsz_test) for yolo_bb in img_gt_boxes]
                        # img_gt_boxes_xywh = [[x[0], x[1],x[2]-x[0], x[3]-x[1]] for x in img_gt_boxes_xyxy]

                        # Ground truths: [(bbox, class_id)]
                        ground_truths.extend([(x, y) for x, y in zip (img_gt_boxes_xyxy, img_gt_lbls_in_coco_order)])
                    else: # detections only no GT
                        im0s = detection_plot(colors, dataloader, detection_res, im0s, opt, paths, plot, pred,
                                              uniq_class_subset)

                        tag = ''
                        # if opt.adding_ext_noise:
                        #     save_fname = paths.split('/')[-1].split('.')[0] + '_' + str(
                        #         getattr(dataloader, 'frame', 0)) + '_noisy_data.png'  # img.jpg
                        # else:
                        #     save_fname = paths.split('/')[-1].split('.')[0] + '_'+ str(getattr(dataloader, 'frame', 0)) + '.png'  # img.jpg
                        #
                        # im0s = letterbox(im0s, imgsz_test, 32)[0] # reshpae presentation image for debug
                        # im0s = np.repeat(im0s[:, :, np.newaxis], 3, axis=2)  # convert GL to RGB by replication
                        # im0s = scaling_image(im0s, scaling_type='single_image_percentile_0_1')
                        # if im0s.max() <= 1:
                        #     im0s = im0s * 255
                        #
                        # if bool(preds):
                        #     for i, det in enumerate(preds):  # detections per image
                        #         xyxy, cls, conf = det
                        #         label = f'{coco_lass_names[int(cls)]} {conf:.2f}'
                        #         if plot:
                        #             if isinstance(xyxy, list):
                        #                 xyxy = xyxy[0]
                        #             plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=1)
                        #             save_path = os.path.join(opt.save_path, save_fname)
                        #             cv2.imwrite(save_path, im0s)
                        #         detection_res.append({'file': paths.split('/')[-1], 'fname': paths.split('/')[-1].split('_right')[0].split('_left')[0], 'class': cls, 'conf': conf, 'xyxy': xyxy})
                        #
                        # else:
                        #     print(scores.max())
                        #     detection_res.append({'file': paths.split('/')[-1], 'fname': paths.split('/')[-1].split('_right')[0].split('_left')[0], 'class': -1, 'conf': 0, 'xyxy': -1})
                        #
                        # road  : 20250112_173041_FS_50_XGA_Test55A_SY_RD_US_right_roi_375
                        # diluted_by_factor = 100
                        # if seen %diluted_by_factor:
            t0 += time_synchronized() - t
            # out coco 80 classes : [1, 25200, 85] [batch, proposals_3_scales,4_box__coord+1_obj_score + n x classes]
            # Compute loss
            # if compute_loss:
            #     loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

        # t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz_test, imgsz_test, batch_size)  # tuple
        # if 1:
        #     print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
        # Predicted: [(bbox, class_id, confidence)]
    if is_new_model and not bool(opt.detection_no_gt):
        df = pd.DataFrame(pred_tgt_acm)
        df.to_csv(os.path.join(opt.save_path, 'onnx_model_pred_tgt_acm_conf_th_' + str(det_threshold.__format__('.3f')) + '.csv'), index=False)

        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():  # P, R @  # max F1 index
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, v5_metric=False, save_dir=opt.save_path,
                                                  names=names)  # based on correct @ IOU=0.5 of pred box with target
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format

        for i, c in enumerate(ap_class):
            print( (names[c], seen, p[i], r[i], ap[i]))

        # map = compute_map(predictions, ground_truths, num_classes=2, iou_threshold=0.5, conf_th=det_threshold)
    else:
        # classes are confined to coco_lass_names
        iou_threshold = 0.5
        map = compute_map(predictions, ground_truths, num_classes=3,
                          iou_threshold=iou_threshold, conf_th=det_threshold,
                          min_precision_acceptible=0.9, save_dir=opt.save_path,
                          class_enum=coco_lass_names)

        tta_res = dict()
        tta_res['predictions'] = predictions
        tta_res['ground_truths'] = ground_truths
        tta_res['iou_threshold'] = opt.iou_thres
        tta_res['conf_thres'] = opt.conf_thres
        np.savetxt(os.path.join(opt.save_path,"image_preprocessed.csv"), img[0,:,:,0], delimiter=",")
        with open(os.path.join(opt.save_path,  'metadata_for_pre_re_detection_threshold_' + str(det_threshold) + '.pkl'), 'wb') as f:
            pickle.dump(tta_res, f)

    print(f"mAP: {map:.2f}")

    if bool (detection_res):
        df = pd.DataFrame(detection_res)
        df.to_csv(os.path.join(opt.save_path, 'detections_results.csv'))
    return


def detection_plot(colors, dataloader, detection_res, im0s, opt, paths, plot, pred, uniq_class_subset):
    if pred.numel() > 0:
        if opt.adding_ext_noise:
            save_fname = paths.split('/')[-1].split('.')[0] + '_' + str(
                getattr(dataloader, 'frame', 0)) + '_noisy_data.png'  # img.jpg
        else:
            save_fname = paths.split('/')[-1].split('.')[0] + '_' + str(
                getattr(dataloader, 'frame', 0)) + '.png'  # img.jpg
        im0s = np.repeat(im0s[:, :, np.newaxis], 3, axis=2)  # convert GL to RGB by replication
        im0s = scaling_image(im0s, scaling_type='single_image_percentile_0_1')
        if im0s.max() <= 1:
            im0s = im0s * 255

        for i, det in enumerate(pred):  # detections per image
            xyxy = det[:4]
            conf = det[4]
            cls = int(det[5])
            label = f'{uniq_class_subset[int(cls)]} {conf:.2f}'
            if plot:
                if isinstance(xyxy, list):
                    xyxy = xyxy[0]
                plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=1)
                save_path = os.path.join(opt.save_path, save_fname)
                cv2.imwrite(save_path, im0s)
            detection_res.append(
                {'file': paths.split('/')[-1], 'fname': paths.split('/')[-1].split('_right')[0].split('_left')[0],
                 'class': cls, 'conf': conf, 'xyxy': xyxy})

    else:
        detection_res.append(
            {'file': paths.split('/')[-1], 'fname': paths.split('/')[-1].split('_right')[0].split('_left')[0],
             'class': -1, 'conf': 0, 'xyxy': -1})
    return im0s


if __name__ == '__main__':

    print(f"Python version: {sys.version}, {sys.version_info} ")
    print(f"Pytorch version: {torch.__version__} ")

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=-1, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--oldmodel', action='store_true', help='augmented inference')
    parser.add_argument('--norm-type', type=str, default='standardization',
                                        choices=['standardization', 'single_image_0_to_1', 'single_image_mean_std','single_image_percentile_0_255',
                                                 'single_image_percentile_0_1', 'remove+global_outlier_0_1', 'no_norm'],
                                        help='Normalization approach')

    parser.add_argument('--save-path', default='', help='save to project/name')
    parser.add_argument('--test-files-path', type=str, default='/home/hanoch/projects/tir_od/yolov7/tir_od/test_set/Test51a_Test40A_test_set.txt', help='')

    parser.add_argument('--detection-no-gt', action='store_true', help='')

    parser.add_argument('--images-parent-folder', type=str, default='/mnt/Data/hanoch/tir_frames_rois/yolo7_tir_data_all', help='')  # in case --detection-no-gt


    parser.add_argument('--test-vectors-test-old-tir-1p5', action='store_true', help='') # based on CPP test vector

    parser.add_argument('--adding-ext-noise', action='store_true', help='')

    parser.add_argument('--noise-parent-folder', type=str, default='/home/hanoch/projects/tir_frames_rois/marmon_noisy_sy/noise_samples', help='')  # in case --detection-no-gt
    # --test-path   '/home/hanoch/projects/tir_od/yolov7/tir_od/test_set/Test51a_Test40A_test_set_part.txt'  'Test51a_Test40A_test_set_part.txt'

    parser.add_argument('--reorder-output-class-to-viz-od', action='store_true', help='') #  "person": 0, "car": 1,   "bike": 2,   "animal": 3, "locomotive": 4, "braking_shoe": 5

    opt = parser.parse_args()

    main(opt=opt)

    """
    --cache-images --device 0 --weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.onnx --img-size 640 --conf-thres 0.66  --iou-thres 0.6 --norm-type single_image_percentile_0_1 --save-path /mnt/Data/hanoch/runs/train/yolov7999 
    --cache-images --device 0 --weights /mnt/Data/hanoch/tir_old_tf/tir_od_1.5.onnx --img-size 512 --conf-thres 0.8  --iou-thres 0.5 --norm-type no_norm
         '/mnt/Data/hanoch/tir_old_tf/tir_od_1.5.onnx'   # old TIR model

    --cache-images --device 0 --weights /mnt/Data/hanoch/tir_old_tf/tir_od_1.5.onnx --img-size 512 --conf-thres 0.8  --iou-thres 0.5 --norm-type no_norm --save-path /mnt/Data/hanoch/runs/tir_old_1.5 --test-files-path /home/hanoch/projects/tir_od/yolov7/tir_od/test_set/Test51a_Test40A_test_set.txt
    
DEtections only old model  
    --cache-images --device 0 --weights /mnt/Data/hanoch/tir_old_tf/tir_od_1.5.onnx --img-size 512 --conf-thres 0.8  --iou-thres 0.5 --norm-type no_norm --save-path /mnt/Data/hanoch/runs/tir_old_1.5 --images-parent-folder /home/hanoch/projects/tir_frames_rois/marmon_noisy_sy --detection-no-gt
    Reducing th = 0.2
    --cache-images --device 0 --weights /mnt/Data/hanoch/tir_old_tf/tir_od_1.5.onnx --img-size 512 --conf-thres 0.2  --iou-thres 0.5 --norm-type no_norm --save-path /mnt/Data/hanoch/runs/tir_old_1.5 --images-parent-folder /home/hanoch/projects/tir_frames_rois/marmon_noisy_sy --detection-no-gt
    Adding noise 
    --cache-images --device 0 --weights /mnt/Data/hanoch/tir_old_tf/tir_od_1.5.onnx --img-size 512 --conf-thres 0.8  --iou-thres 0.5 --norm-type no_norm --save-path /mnt/Data/hanoch/runs/tir_old_1.5 --test-files-path /home/hanoch/projects/tir_od/yolov7/tir_od/test_set/Test51a_Test40A_test_set.txt --adding-ext-noise
    
    Plotting P/R curve over detections th=0.05
    --cache-images --device 0 --weights /mnt/Data/hanoch/tir_old_tf/tir_od_1.5.onnx --img-size 512 --conf-thres 0.05  --iou-thres 0.5 --norm-type no_norm --save-path /mnt/Data/hanoch/runs/tir_old_1.5 --images-parent-folder /home/hanoch/projects/tir_frames_rois/marmon_noisy_sy --detection-no-gt      

Detection
    --cache-images --device 0 --weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.onnx --img-size 640 --conf-thres 0.66  --iou-thres 0.6 --norm-type single_image_percentile_0_1 --images-parent-folder /home/hanoch/projects/tir_frames_rois/marmon_noisy_sy --save-path /mnt/Data/hanoch/runs/yolov7999_onnx_run --detection-no-gt --adding-ext-noise


    DEtections only New model Yolov7999  with noise addition
    --cache-images --device 0 --weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.onnx --img-size 640 --conf-thres 0.48  --iou-thres 0.6 --norm-type single_image_percentile_0_1 --images-parent-folder /mnt/Data/hanoch/tir_frames_rois/onnx_bm --save-path /mnt/Data/hanoch/runs/yolov7999_onnx_run --detection-no-gt
P/R curve 
    --cache-images --device 0 --weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.onnx --img-size 640 --conf-thres 0.01  --iou-thres 0.6 --norm-type single_image_percentile_0_1  --test-files-path /home/hanoch/projects/tir_od/yolov7/tir_od/test_set/Test51a_Test40A_test_set.txt --save-path /mnt/Data/hanoch/runs/yolov7999_onnx_run/P_R_curve_test_set --adding-ext-noise
    
    DEtections only New model Yolov7999  based on text file name
    --cache-images --device 0 --weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.onnx --img-size 640 --conf-thres 0.48  --iou-thres 0.6 --norm-type single_image_percentile_0_1 --images-parent-folder /mnt/Data/hanoch/tir_frames_rois/onnx_bm --save-path /mnt/Data/hanoch/runs/yolov7999_onnx_run --detection-no-gt
best.person_cls_0.onnx
    --cache-images --device 0 --weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.person_cls_0.onnx --img-size 640 --conf-thres 0.01  --iou-thres 0.6 --norm-type single_image_percentile_0_1  --test-files-path /home/hanoch/projects/tir_od/yolov7/tir_od/test_set/Test51a_Test40A_test_set.txt --save-path /mnt/Data/hanoch/runs/yolov7999_onnx_run/P_R_curve_test_set_reorder_output 
    # Test vector 
    --cache-images --device 0 --weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.person_cls_0.onnx --img-size 640 --conf-thres 0.48  --iou-thres 0.6 --norm-type single_image_percentile_0_1  --images-parent-folder /mnt/Data/hanoch/tir_frames_rois/onnx_bm --save-path /mnt/Data/hanoch/runs/yolov7999_onnx_run --detection-no-gt 

    """
