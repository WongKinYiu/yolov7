import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm
import copy
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix, range_bar_plot, range_p_r_bar_plot
from utils.plots import plot_images, output_to_target, plot_study_txt, append_to_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel
import pandas as pd
from yolo_object_embeddings import ObjectEmbeddingVisualizer

def object_size_to_range(obj_height_pixels: float, focal:int, class_id:int=1):
    class_height_ml = {0:1.5, 1:1.8, 2:4} # car Sedan height = 1.5 m , person height is 1.8m ; locomotive in ML is 4m in SY is 3m
    class_height_sy = {0:1.5, 1:1.8, 2:3} # car Sedan height = 1.5 m , person height is 1.8m ; locomotive in ML is 4m in SY is 3m
    pixel_size = 17e-6
    class_id = int(class_id)
    if focal==50:
        obj_height_m = class_height_sy[class_id]
    elif focal >200:
        obj_height_m = class_height_ml[class_id]
    else:
        raise ValueError('focal length out of range', focal)

    return obj_height_m * focal * 1e-3 / (obj_height_pixels * pixel_size)


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # used for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         **kwargs):
    # Initialize/load model and set device

    hyp = kwargs['hyp']
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories

        if opt.save_path == '':
            save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        else:
            save_dir = Path(increment_path(os.path.join(opt.save_path, Path(opt.project) , opt.name), exist_ok=opt.exist_ok))

        try: # no suduer can fail
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",e)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        if isinstance(compute_loss, bool):  # ovcerside in case injected explicitly from training by OHEM
            if compute_loss:
                trace= False
        if trace:
            model = TracedModel(model, device, imgsz, opt.input_channels)

    if isinstance(compute_loss, bool): # otherside in case injected explicitly from training by OHEM
        if compute_loss:
            from utils.loss import ComputeLoss, ComputeLossOTA
            hyp.update({'cls_pw': 1.0})
            hyp.update({'fl_gamma': 1.5})
            hyp.update({'obj_pw': 1.0})
            hyp.update({'anchor_t': 4.0})
            hyp.update({'cls': 0.5})
            hyp.update({'box': 0.05})
            hyp.update({'obj': 1.0})

            model.hyp = hyp
            compute_loss = ComputeLoss(device=device, model=model)

    #torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA @@ HK : TODO what are the consequences  add :
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader

    embed_analyse = kwargs.get('embed_analyse', False)
    model_name = kwargs.get('model_name', '')
    loss_per_image_acm = list()
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, opt.input_channels, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        hyp = dict()
        hyp['person_size_small_medium_th'] = 32 * 32
        hyp['car_size_small_medium_th'] = 44 * 44

        hyp['img_percentile_removal'] = 0.3
        hyp['beta'] = 0.3
        hyp['gamma'] = 80 # dummy anyway augmentation is disabled
        hyp['gamma_liklihood'] = 0.01
        hyp['random_pad'] = True
        hyp['copy_paste'] = False
        # augment=False explicit no augmentation to test
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, hyp, pad=0.5, augment=False, rect=False, #rect was True  # HK@@@ TODO : why pad =0.5?? only effective in rect=True in test time ? https://github.com/ultralytics/ultralytics/issues/13271
                                       prefix=colorstr(f'{task}: '), rel_path_images=data['path'], num_cls=data['nc'])[0]

        labels = np.concatenate(dataloader.dataset.labels, 0)
        class_labels = torch.tensor(labels[:, 0])  # classes

        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.dump(vars(opt), f, sort_keys=False)
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(hyp, f, sort_keys=False)
        with open(save_dir / 'test_set.txt', 'w') as f:
            for file in dataloader.dataset.img_files:
                f.write(f"{file}\n")




    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc, conf=conf_thres, iou_thres=iou_thres) # HK per conf per iou_thresh
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

    if not training:
        print(100 * '==')
        print('Test set labels {} count : {}'.format(names, torch.bincount(class_labels.long(), minlength=nc) + 1))

    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    # res_all = list()
    predictions_df = pd.DataFrame(columns=[
        'image_id',
        'pred_cls',
        'bbox_x',
        'bbox_y',
        'bbox_w',
        'bbox_h',
        'score'
    ])

    if embed_analyse:
        obj_embed_viz = ObjectEmbeddingVisualizer(model=model, device=device)
        features_acm = torch.empty((0, 1024)) # embedding dim of last scale 1024x20x20
        labels_acm = np.array([])

    stats_all_large, stats_person_medium = [], []
    if dataloader.dataset.use_csv_meta_data_file:

        n_bins_of100m = 20
        bin_size_100 = 100
        bin_size_25 = 50

        range_bins_map = {x.item():[0]*n_bins_of100m for x in pd.unique(dataloader.dataset.df_metadata['sensor_type'])}
        range_bins_precision_all_classes = {x.item():[np.zeros(nc)] *n_bins_of100m for x in pd.unique(dataloader.dataset.df_metadata['sensor_type'])}
        range_bins_recall_all_classes = {x.item():[np.zeros(nc)] *n_bins_of100m for x in pd.unique(dataloader.dataset.df_metadata['sensor_type'])}
        range_bins_support_gt = {x.item():[0]*n_bins_of100m for x in pd.unique(dataloader.dataset.df_metadata['sensor_type'])}
        gt_per_range_bins = {x.item(): [[] for _ in range(n_bins_of100m)] for x in pd.unique(dataloader.dataset.df_metadata['sensor_type'])}# collecting GT labels
        gt_path_per_range_bins = {x.item(): [[] for _ in range(n_bins_of100m)] for x in pd.unique(dataloader.dataset.df_metadata['sensor_type'])}# collecting GT labels
        bin_size_per_sensor = {}

        for sensor_type in pd.unique(dataloader.dataset.df_metadata['sensor_type']):
            exec('stats_all_sensor_type_{}'.format(sensor_type.item()) + '=[]')  # 'stats_all_50'
            exec('stats_all_sensor_type_{}_with_range'.format(sensor_type.item()) + '=[]')  # 'stats_all_50'

            sensor_focal = int(sensor_type.astype('str').split('_')[-1])

            if sensor_focal > bin_size_100:
                bin_size_per_sensor.update({sensor_focal: bin_size_100})
            else:
                bin_size_per_sensor.update({sensor_focal: bin_size_25})

        for daytime in pd.unique(dataloader.dataset.df_metadata['part_in_day']):
            exec('stats_all_time_{}'.format(daytime.lower()) + '=[]')  # 'stats_all_day'

        for weather_condition in pd.unique(dataloader.dataset.df_metadata['weather_condition']):
            if isinstance(weather_condition, str):
                exec('stats_all_weather_condition_{}'.format(weather_condition.lower()) + '=[]')  # 'stats_all_day'

        sensor_type_vars = [key for key in vars().keys() if 'stats_all_sensor_type' in key and not '_with_range' in key]
        time_vars = [key for key in vars().keys() if 'stats_all_time' in key and not '_with_range' in key]
        weather_condition_vars = [key for key in vars().keys() if 'stats_all_weather_condition' in key and not '_with_range' in key]
        # all_cars = list()

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0 c# already done inside dataloader
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=augment)  # inference out [batch, proposals, figures_of] figures_of :(4 coordination, obj conf, cls conf ) and training outputs(batch_size, anchor per scale, x,y dim of scale out 40x40 ,n_classes-conf+1-objectness+4-bbox ) over 3 scales diferent outputs (2,2,80,80,7), (2,2,40,40,7)  : 640/8=40
            t0 += time_synchronized() - t
            # out coco 80 classes : [1, 25200, 85] [batch, proposals_3_scales,4_box__coord+1_obj_score + n x classes]
            # Compute loss
            if compute_loss:
                loss_out = compute_loss([x.float() for x in train_out], targets) # loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
                loss_per_image_acm.append(loss_out[0].detach().cpu())
                loss += loss_out[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized() #NMS
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True) # Does thresholding for class  : list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            # out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=False) # Does thresholding for class  : list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            t1 += time_synchronized() - t

        if trace and embed_analyse and np.sum([x.numel() for x in out])>0: # features are being saved/clone in the trace model version only TODO for others
            features, labels = obj_embed_viz.extract_object_grounded_features(feature_maps=model.features,
                                                           predictions=out,
                                                           image_shape=img.shape)
            features_acm = torch.cat((features_acm, features.detach().cpu()), dim=0)
            labels_acm = np.concatenate((labels_acm, labels), axis=0)

        # Statistics per image
        for si, pred in enumerate(out): # [bbox_coors, objectness_logit, class]

            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            # No predictions in the image but only GT = >FN
            if len(pred) == 0:
                if nl:
                    if dataloader.dataset.use_csv_meta_data_file: # cases where no predictions but still GT are

                        weather_condition = (dataloader.dataset.df_metadata[
                                                 dataloader.dataset.df_metadata['tir_frame_image_file_name'] ==
                                                 str(path).split('/')[-1]]['weather_condition'].item())
                        if isinstance(weather_condition, str):
                            weather_condition = weather_condition.lower()
                            exec([x for x in weather_condition_vars if str(weather_condition) in x][
                                     0] + '.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))')

                        time_in_day = dataloader.dataset.df_metadata[
                            dataloader.dataset.df_metadata['tir_frame_image_file_name'] == str(path).split('/')[-1]][
                            'part_in_day'].item().lower()
                        # eval([x for x in time_vars if str(time_in_day) in x][0]).append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                        exec([x for x in time_vars if str(time_in_day) in x][
                                 0] + '.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))')

                        sensor_type = dataloader.dataset.df_metadata[
                            dataloader.dataset.df_metadata['tir_frame_image_file_name'] == str(path).split('/')[-1]][
                            'sensor_type'].item()

                        gt_range = [(object_size_to_range(obj_height_pixels=h, focal=sensor_type,
                                                          class_id=class_id.numpy().item())) for
                                    class_id, (x, y, w, h) in zip(labels[:, 0].cpu(), labels[:, 1:5].cpu())]

                        # coupling the range cell between any overlapped IOU >TH between pred bbox and GT bbox
                        if 1:
                            gt_range = [_range // 100 for _range in
                                        gt_range]  # gt_range = [_range // 100 for _range in gt_range]
                        else:
                            gt_range = [_range // bin_size_per_sensor[sensor_type] for _range in gt_range]

                        # print('Cars in image', (np.array(tcls)==0).astype('int').sum().item())
                        # all_cars.append((np.array(tcls)==0).astype('int').sum().item()) # all_cars
                        # only update GTs : gt_per_range_bins
                        for gt_lbl, rng_ in zip(labels[:,0], gt_range):
                            if rng_ < n_bins_of100m :
                                gt_per_range_bins[sensor_type][int(rng_.item())].append(int(gt_lbl.item()))  # add to each range bin GT the GT counts
                                gt_path_per_range_bins[sensor_type][int(rng_.item())].append(str(path))
                            else:
                                print('target out of 2000m', gt_lbl, path)
                        exec([x for x in sensor_type_vars if str(sensor_type) in x][0] + '.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))')
                        #         However _with_range holds ranges of pred = > dummt prediction
                        obj_range_m = torch.Tensor()
                        exec([x+'_with_range' for x in sensor_type_vars if str(sensor_type) in x][0] + '.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls, obj_range_m, pred[:, 5].shape[0]*[str(path)]))') # path is replicated to match each prediction TP/FP in the image

                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))    #niou for COCO 0.5:0.05:1
                continue

            # Predictions
            predn = pred.clone() # *xyxy, conf, cls in predn  [x y ,w ,h, conf, cls] taking top 300 after NMS
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                collect_info = list()
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

                    collect_info.append({'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

                    predictions_df = pd.concat([
                        predictions_df,
                        pd.DataFrame({
                            'image_id': [image_id],
                            'pred_cls': [coco91class[int(p[5])] if is_coco else int(p[5])],
                            'bbox_x': [[round(x, 3) for x in b][0]],
                            'bbox_y': [[round(x, 3) for x in b][1]],
                            'bbox_w': [[round(x, 3) for x in b][2]],
                            'bbox_h': [[round(x, 3) for x in b][3]],
                            'score':  [round(p[4], 5)]
                        })
                    ], ignore_index=True)

                    for it in labels.cpu().numpy():
                        # jdict.append({'image_id': image_id,
                        #               'gt_cls': it[0],
                        #               'bbox': [round(x, 3) for x in it[1:]]})

                        predictions_df = pd.concat([
                            predictions_df,
                            pd.DataFrame({
                                'image_id': [image_id],
                                'gt_cls': [it[0]],
                                'bbox': [[round(x, 3) for x in it[1:]]]})
                        ], ignore_index=True)

            # Assign all predictions as incorrect ; pred takes top 300 predictions conf over 10 ious [0.5:0.95:0.05]
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False): # iouv[0]=0.5 IOU for dectetions iouv in general are all 0.5:0.05:.. for COCO
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf_objectness, pcls, tcls) Predicted class is Max-Likelihood among all classes logit and threshol goes over the objectness only
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)) # correct @ IOU=0.5 of pred box with target

            if save_json:
                predictions_df = pd.concat([
                    predictions_df,
                    pd.DataFrame({
                        'image_id': [image_id],
                        'correct': [correct[:, 0].cpu()]
                    })
                ], ignore_index=True)


            if 1: #not training or 1:
                # assert len(pred[:, :4]) == 1
                x, y, w, h = xyxy2xywh(pred[:, :4])[0]##HK BUG !! need to go over all preds see which indexes aligned to which value
                if w * h > hyp['person_size_small_medium_th'] and  w * h <= hyp['car_size_small_medium_th']:
                    stats_person_medium.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                #     [(ix, w.cpu() * h.cpu()) for ix, (x, y, w, h) in enumerate(xyxy2xywh(pred[:, :4])) if w * h > hyp['person_size_small_medium_th'] and  w * h <= hyp['car_size_small_medium_th']]
                if w * h > hyp['car_size_small_medium_th']:
                    stats_all_large.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

                #     sensor type
                if dataloader.dataset.use_csv_meta_data_file:
                    weather_condition = (dataloader.dataset.df_metadata[dataloader.dataset.df_metadata['tir_frame_image_file_name'] == str(path).split('/')[-1]]['weather_condition'].item())
                    if isinstance(weather_condition, str):
                        weather_condition = weather_condition.lower()
                        exec([x for x in weather_condition_vars if str(weather_condition) in x][0] + '.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))')

                    time_in_day = dataloader.dataset.df_metadata[dataloader.dataset.df_metadata['tir_frame_image_file_name'] == str(path).split('/')[-1]]['part_in_day'].item().lower()
                    # eval([x for x in time_vars if str(time_in_day) in x][0]).append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                    exec([x for x in time_vars if str(time_in_day) in x][0] + '.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))')

                    sensor_type = dataloader.dataset.df_metadata[dataloader.dataset.df_metadata['tir_frame_image_file_name'] == str(path).split('/')[-1]]['sensor_type'].item()
                    # print(sensor_type)
                    # obj_range_m = torch.tensor(
                    #     [(object_size_to_range(obj_height_pixels=h.cpu(), focal=sensor_type, class_id=class_id.cpu().numpy().item()))
                    #      for class_id, (x, y, w, h) in zip(pred[:, 5], xyxy2xywh(pred[:, :4]))])
                    gt_range = [(object_size_to_range(obj_height_pixels=h, focal=sensor_type, class_id=class_id.numpy().item())) for
                                class_id, (x, y, w, h) in zip(labels[:, 0].cpu(), labels[:, 1:5].cpu())]

                    # coupling the range cell between any overlapped IOU >TH between pred bbox and GT bbox
                    obj_range_m = list()
                    i = 0
                    for class_id_pred, (x1_p, y1_p, x2_p, y2_p) in zip(pred[:, 5].cpu(), pred[:, :4].cpu()):

                        range_candidate = torch.tensor(
                            object_size_to_range(obj_height_pixels=xyxy2xywh(pred[:, :4])[0][-1].cpu(),
                                                 focal=sensor_type,
                                                 class_id=class_id_pred.cpu().numpy().item()))
                        # Find any IOU overlapped between GT and prediction
                        for class_id_gt, (x1_gt, y1_gt, x2_gt, y2_gt), (xc,yc,w,h) in zip(labels[:, 0].cpu(),
                                                                             xywh2xyxy(labels[:, 1:5].cpu()), labels[:, 1:5].cpu()):
                            ious = box_iou(torch.tensor((x1_gt, y1_gt, x2_gt, y2_gt)).unsqueeze(axis=0), torch.tensor((x1_p, y1_p, x2_p, y2_p)).unsqueeze(axis=0))
                            i += 1
                            if ious > iouv.cpu()[0]: #
                                range_candidate = torch.tensor(
                                    object_size_to_range(obj_height_pixels=h.cpu(),
                                                         focal=sensor_type,
                                                         class_id=class_id_pred.cpu().numpy().item()))
                                break  # the aligned GT/Pred was found no need to iterate more, this is the atmost candidate
                        obj_range_m.append(range_candidate)
                    # else: #ranges = func(sqrt(height*width))
                    #     obj_range_m = torch.tensor([(object_size_to_range(obj_height_pixels=(np.sqrt(h.cpu()*w.cpu())), focal=sensor_type)) for ix, (x, y, w, h) in enumerate(xyxy2xywh(pred[:, :4]))])
                    #     gt_range = [(object_size_to_range(obj_height_pixels=(np.sqrt(h*w)), focal=sensor_type)) for ix, (x, y, w, h) in enumerate(labels[:,1:5].cpu())]
                    #
                    if 1:
                        gt_range = [_range // 100 for _range in gt_range] #gt_range = [_range // 100 for _range in gt_range]
                    else:
                        gt_range = [_range//bin_size_per_sensor[sensor_type] for _range in gt_range]

                    for gt_lbl, rng_ in zip(labels[:,0], gt_range):
                        if rng_ < n_bins_of100m :
                            gt_per_range_bins[sensor_type][int(rng_.item())].append(int(gt_lbl.item()))  # add to each range bin GT the GT counts
                            gt_path_per_range_bins[sensor_type][int(rng_.item())].append(str(path))
                        else:
                            print('target out of 2000m', gt_lbl, path)
                        # (obj_range_m.cpu().reshape(-1))
                    exec([x for x in sensor_type_vars if str(sensor_type) in x][0] + '.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))')
                    exec([x+'_with_range' for x in sensor_type_vars if str(sensor_type) in x][0] + '.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls, obj_range_m, pred[:, 5].shape[0]*[str(path)]))') # path is replicated to match each prediction TP/FP in the image



                    # for daytime in pd.unique(dataloader.dataset.df_metadata['part_in_day']):
                    #     exec('stats_all_part_in_day{}'.format(daytime.lower()) + '=[]')  # 'stats_all_day'



        # Plot images  aa = np.repeat(img[0,:,:,:].cpu().permute(1,2,0).numpy(), 3, axis=2).astype('float32') cv2.imwrite('test/exp40/test_batch88_labels__.jpg', aa*255)
        if (plots and batch_i > 10):
            # conf_thresh_plot = 0.1 # the plot threshold the connfidence
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if trace and embed_analyse:
        embeddings = obj_embed_viz.visualize_object_embeddings(features_acm,
                                                               labels_acm,
                                                               path=save_dir,
                                                               tag=opt.conf_thres)

    if not training or 1:
        stats_person_medium = [np.concatenate(x, 0) for x in zip(*stats_person_medium)]  # to numpy
        stats_all_large = [np.concatenate(x, 0) for x in zip(*stats_all_large)]  # to numpy
        if dataloader.dataset.use_csv_meta_data_file:
            for time_var in time_vars:
                exec('{}=[np.concatenate(x, 0) for x in zip(*{})]'.format(time_var, time_var))  # 'stats_all_50'

            for sensor_type in sensor_type_vars:
                exec('{}=[np.concatenate(x, 0) for x in zip(*{})]'.format(sensor_type, sensor_type))  # 'stats_all_50'
                exec('{}_with_range=[np.concatenate(x, 0) for x in zip(*{}_with_range)]'.format(sensor_type, sensor_type))  # 'stats_all_50'

            for weather_condition in weather_condition_vars:
                exec('{}=[np.concatenate(x, 0) for x in zip(*{})]'.format(weather_condition, weather_condition))  # 'stats_all_50'

    if len(stats) and stats[0].any():
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir,
                            names=names, class_support=nt, tag=model_name) #based on correct @ IOU=0.5 of pred box with target
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()


        # if bool(stats_person_medium):
        #     p_med, r_med, ap_med, f1_med, ap_class_med = ap_per_class(*stats_person_medium, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names, tag='small_objects')
        #     ap50_med, ap_med = ap_med[:, 0], ap_med.mean(1)  # AP@0.5, AP@0.5:0.95
        #     mp_med, mr_med, map50_med, map_med = p_med.mean(), r_med.mean(), ap50_med.mean(), ap_med.mean()
        #     nt_med = np.bincount(stats_person_medium[3].astype(np.int64), minlength=nc)  # number of targets per class
        #
        # if bool(stats_all_large):
        #     p_large, r_large, ap_large, f1_large, ap_class_large = ap_per_class(*stats_all_large, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names, tag='large_objects')
        #     ap50_large, ap_large = ap_large[:, 0], ap_large.mean(1)  # AP@0.5, AP@0.5:0.95
        #     mp_large, mr_large, map50_large, map_large = p_large.mean(), r_large.mean(), ap50_large.mean(), ap_large.mean()
        #     nt_large = np.bincount(stats_all_large[3].astype(np.int64), minlength=nc)  # number of targets per class

        if dataloader.dataset.use_csv_meta_data_file:
            for time_var in time_vars:
                if bool(eval(time_var)):
                    # TODO in the names of classes in labels add the support from nt_, also pass the support of overall and the title name like in the tag but only day /night to the title inside plot_pr_curve()
                    exec("nt_{} = np.bincount({}[3].astype(np.int64), minlength={})".format(time_var, time_var, nc))
                    exec("p_{}, r_{}, ap_{}, f1_{}, ap_class_{} = ap_per_class(*{}, plot={}, v5_metric={}, save_dir={}, names={}, tag={}, class_support=nt_{})".format(time_var,
                                                                time_var, time_var, time_var, time_var, time_var, plots, v5_metric, 'str(save_dir)', names, 'str(time_var)', time_var))

                    exec("ap50_{}, ap_{} = ap_{}[:, 0], ap_{}.mean(1)".format(time_var, time_var, time_var, time_var))
                    exec("mp_{}, mr_{}, map50_{}, map_{} = p_{}.mean(), r_{}.mean(), ap50_{}.mean(), ap_{}.mean()".format(time_var, time_var, time_var, time_var, time_var, time_var, time_var, time_var))

                for weather_condition in weather_condition_vars:
                    exec("nt_{} = np.bincount({}[3].astype(np.int64), minlength={})".format(weather_condition, weather_condition, nc))
                    exec("p_{}, r_{}, ap_{}, f1_{}, ap_class_{} = ap_per_class(*{}, plot={}, v5_metric={}, save_dir={}, names={}, tag={}, class_support=nt_{})".format(weather_condition,
                                    weather_condition, weather_condition, weather_condition, weather_condition, weather_condition, plots, v5_metric, 'str(save_dir)', names, 'str(weather_condition)', weather_condition))

                    exec("ap50_{}, ap_{} = ap_{}[:, 0], ap_{}.mean(1)".format(weather_condition, weather_condition, weather_condition, weather_condition))
                    exec("mp_{}, mr_{}, map50_{}, map_{} = p_{}.mean(), r_{}.mean(), ap50_{}.mean(), ap_{}.mean()".format(weather_condition, weather_condition,
                                                            weather_condition, weather_condition, weather_condition, weather_condition, weather_condition, weather_condition))

            if 0 : #debug
                ranges100_pred = np.array([])
                print('gt class dist per range cell of 100s and sum of GTs ')
                print([(100*(ix+1), np.unique(x, return_counts=True), np.size(x)) for ix, x in enumerate(gt_per_range_bins[210])])
                exec('ranges100_pred={}_with_range[4]//100'.format('stats_all_sensor_type_210'))
                exec('cls_pred={}_with_range[2]'.format('stats_all_sensor_type_210'))
                exec('predicted={}_with_range[0]'.format('stats_all_sensor_type_210'))

                print('True predictions on 210', eval('ranges100_pred.shape'))
                np.array([np.size(x) for ix, x in enumerate(gt_per_range_bins[210])]).T # predictions may be TRue or False
                print('detections preds per range cell')
                np.unique(ranges100_pred, return_counts=True)[1].T

                [np.bincount(x, minlength=2) for ix, x in enumerate(gt_per_range_bins[210])]
                range_bin_ = 1
                np.unique(cls_pred[ranges100_pred == range_bin_], return_counts=True)
                # count of good bad predictions per class
                stats_all_sensor_type_210_with_range[2][ranges100_pred == range_bin_]

                np.unique(stats_all_sensor_type_210_with_range[0][:, 0][ranges100_pred == range_bin_],
                          return_counts=True)

            for sensor_type in sensor_type_vars:
                if bool(eval(sensor_type)):
                    exec("nt_{} = np.bincount({}[3].astype(np.int64), minlength={})".format(sensor_type, sensor_type, nc))

                    exec("p_{}, r_{}, ap_{}, f1_{}, ap_class_{} = ap_per_class(*{}, plot={}, v5_metric={}, save_dir={}, names={}, tag={}, class_support=nt_{})".format(sensor_type,
                                                                sensor_type, sensor_type, sensor_type, sensor_type, sensor_type, plots, v5_metric, 'str(save_dir)', names, 'str(sensor_type)',sensor_type))

                    exec("ap50_{}, ap_{} = ap_{}[:, 0], ap_{}.mean(1)".format(sensor_type, sensor_type, sensor_type, sensor_type))
                    exec("mp_{}, mr_{}, map50_{}, map_{} = p_{}.mean(), r_{}.mean(), ap50_{}.mean(), ap_{}.mean()".format(sensor_type, sensor_type, sensor_type, sensor_type, sensor_type, sensor_type, sensor_type, sensor_type))
                    try:
                        sensor_focal = int(sensor_type.split('_')[-1])
                        if  1 :#sensor_focal > 100: # ML
                            exec('ranges={}_with_range[4]//100'.format(sensor_type))

                            for rng_100 in range(0,n_bins_of100m):
                                nt_stat_list_per_range = np.zeros(nc)
                                r_stat_list_per_range = np.zeros(nc)
                                p_stat_list_per_range = np.zeros(nc)
                                map50_per_range = np.array(0)

                                # ind = np.array([])
                                # exec('ind = np.where(ranges == rng_100)[0]')
                                ind = eval('np.where(ranges == rng_100)[0]')
                                stat_list_per_range = list()
                                if ind.size>0: # if there were detections at that range bin
                                    for ele in range(3):# taking the relevant preds related to the distance bin, since P/R/AP are computed globally vs. all GT/targets then it is compared to all targets
                                        stat_list_per_range.append(eval(sensor_type)[ele][ind])
                                    # GT at thta bin range
                                    if not bool(gt_per_range_bins[sensor_focal][rng_100]): # predictions but no GT => FP=>low Prcesion
                                        map50_per_range = np.array(0)
                                        nt_stat_list_per_range = np.zeros(nc)
                                        r_stat_list_per_range = np.zeros(nc)
                                        p_stat_list_per_range = np.zeros(nc)
                                    else:
                                        stat_list_per_range.append(np.array([x for x in gt_per_range_bins[sensor_focal][rng_100]])) # add all targets/labels
                                        nt_stat_list_per_range = np.bincount(stat_list_per_range[3].astype(np.int64), minlength=nc) # GT count per bin range of classes

                                        p_stat_list_per_range, r_stat_list_per_range, ap_stat_list_per_range, f1_stat_list_per_range, \
                                        ap_class_stat_list_per_range = ap_per_class(*stat_list_per_range, plot=plots, v5_metric=v5_metric,
                                                                        save_dir='', names=names, tag='',
                                                                                    class_support=nt_stat_list_per_range)
                                        ap50_per_range, ap_per_range = ap_stat_list_per_range[:, 0], ap_stat_list_per_range.mean(1)  # AP@0.5, AP@0.5:0.95
                                        mp_per_range, mr_per_range, map50_per_range, map_per_range = p_stat_list_per_range.mean(), r_stat_list_per_range.mean(), ap50_per_range.mean(), ap_per_range.mean()
                                else:# no prediction at this range
                                    r_stat_list_per_range = np.zeros(nc)
                                    p_stat_list_per_range = np.zeros(nc)
                                    fn = len(gt_per_range_bins[sensor_focal][rng_100])
                                    recall = 0  # no TP 0/TP+FN
                                    precision = 0
                                    map50_per_range = np.array(0)
                                    if not bool(gt_per_range_bins[sensor_focal][rng_100]):# there are no GT no pred
                                        nt_stat_list_per_range = np.zeros(nc)  # actual GT
                                    else:
                                        nt_stat_list_per_range = np.array(gt_per_range_bins[sensor_focal][rng_100]).sum()
                                # there are GT but no pred
                                # print(map50_per_range)
                                if bool(stat_list_per_range):
                                    if len(stat_list_per_range) == 4 : # sometimes no GT but only predictions hence can't slice
                                        unique_classes = np.unique(stat_list_per_range[3]) # sometimes the GT classes are subset and doesn;t cover the entire set hance slice the relevant ones
                                    else:
                                        unique_classes = range(len(nt_stat_list_per_range))

                                    range_bins_map[sensor_focal][rng_100] = map50_per_range.item()

                                    g = copy.deepcopy(range_bins_precision_all_classes[sensor_focal][rng_100])
                                    g[unique_classes] = p_stat_list_per_range.reshape(1, -1)
                                    range_bins_precision_all_classes[sensor_focal][rng_100] = g

                                    # range_bins_precision_all_classes[sensor_focal][rng_100][unique_classes] = p_stat_list_per_range # broadcast the count of each calss in case one of the classes are missing
                                    g = copy.deepcopy(range_bins_recall_all_classes[sensor_focal][rng_100])
                                    g[unique_classes] = r_stat_list_per_range.reshape(1, -1)
                                    range_bins_recall_all_classes[sensor_focal][rng_100] = g
                                    # range_bins_recall_all_classes[sensor_focal][rng_100][unique_classes] = r_stat_list_per_range

                                    range_bins_support_gt[sensor_focal][rng_100] = nt_stat_list_per_range[unique_classes].sum().item()
                    except Exception as e:
                        print(f'pycocotools unable to run: {e}')

            # bug_diff = np.array(range_bins_support_gt[210]) - np.array(
            #     [np.size(x) for ix, x in enumerate(gt_per_range_bins[210])]).T
            # In case no Preds than there is no GT count in the if condition anyway if pred=0=>TP=0 than P=R=0
            range_bar_plot(n_bins_of100m, range_bins_map, save_dir, range_bins_support=range_bins_support_gt)
            range_p_r_bar_plot(n_bins_of100m, range_bins_precision_all_classes, range_bins_recall_all_classes,
                               save_dir, range_bins_support=range_bins_support_gt, names=names, conf=opt.conf_thres)
            # for time_var in time_vars:
            #     for sensor_type in sensor_type_vars:

        # nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
        nt_med = torch.zeros(1)
        nt_large = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    # if not training or 1:
        # if bool(stats_person_medium):
        #     try:
        #         print(pf % ('all_medium', seen, nt_med.sum(), mp_med, mr_med, map50_med, map_med))
        #     except Exception as e:
        #         print(e)
        #
        # if bool(stats_all_large):
        #     try:
        #         print(pf % ('all_large', seen, nt_large.sum(), mp_large, mr_large, map50_large, map_large))
        #     except Exception as e:
        #         print(e)

    file_path = os.path.join(save_dir, 'class_stats.txt') #'Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95'
    append_to_txt(file_path, 'all', seen, nt.sum(), mp, mr, map50, map)

    # Print results per class
    if 1 or (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            append_to_txt(file_path, names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i])
        # try:
        #     if bool(stats_person_medium):
        #         for i, c in enumerate(ap_class_med):
        #             print(pf % (names[c]+ '_med', seen, nt_med[c], p_med[i], r_med[i], ap50_med[i], ap_med[i]))
        #             append_to_txt(file_path, names[c] + '_med', seen, nt_med[c], p_med[i], r_med[i], ap50_med[i], ap_med[i])
        # except Exception as e:
        #     print(e)
        #
        # try:
        #     if bool(stats_all_large):
        #         for i, c in enumerate(ap_class_large):
        #             print(pf % (names[c]+ '_large', seen, nt_large[c], p_large[i], r_large[i], ap50_large[i], ap_large[i]))
        #             append_to_txt(file_path, names[c] + '_large', seen, nt_large[c], p_large[i], r_large[i], ap50_large[i], ap_large[i])
        # except Exception as e:
        #     print(e)

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict): # @@ HK TODO:
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        pred_df_file = str(save_dir / f"{w}_predictions.csv")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval1 = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval1.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval1.evaluate()
            eval1.accumulate()
            eval1.summarize()
            map, map50 = eval1.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    if save_json:
        predictions_df.to_csv(pred_df_file, index=False)

    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t, loss_per_image_acm


def sensor_type_breakdown_kpi(gt_per_range_bins, n_bins_of100m, names, nc, plots, range_bins_map, range_bins_support,
                              save_dir, bin_size_per_sensor, v5_metric, **kwargs):

    # print(sensor_type_50)
    # print(sensor_type_210)

    # print(f_sensor_type_50)
    # print(f_sensor_type_210)
    # exec('{}=sensor_type_50'.format(f_sensor_type_50))
    # exec('{}=sensor_type_210'.format(f_sensor_type_210))
    # print([key for key in vars().keys() if 'stats_all_sensor_type' in key and not '_with_range' in key])
    sensor_type_vars = [key for key in vars().keys() if 'stats_all_sensor_type' in key and not '_with_range' in key]

    # exec('{}=sensor_type_50'.format(f_sensor_type_50))
    for sensor_type in sensor_type_vars: #sensor_type_vars:
        if bool(sensor_type):
            exec("nt_{} = np.bincount({}[3].astype(np.int64), minlength={})".format(sensor_type, sensor_type, nc))

            exec(
                "p_{}, r_{}, ap_{}, f1_{}, ap_class_{} = ap_per_class(*{}, plot={}, v5_metric={}, save_dir={}, names={}, tag={}, class_support=nt_{})".format(
                    sensor_type,
                    sensor_type, sensor_type, sensor_type, sensor_type, sensor_type, plots, v5_metric, 'str(save_dir)',
                    names, 'str(sensor_type)', sensor_type))

            exec("ap50_{}, ap_{} = ap_{}[:, 0], ap_{}.mean(1)".format(sensor_type, sensor_type, sensor_type,
                                                                      sensor_type))
            exec("mp_{}, mr_{}, map50_{}, map_{} = p_{}.mean(), r_{}.mean(), ap50_{}.mean(), ap_{}.mean()".format(
                sensor_type, sensor_type, sensor_type, sensor_type, sensor_type, sensor_type, sensor_type, sensor_type))

            sensor_focal = int(sensor_type.split('_')[-1])
            if 1:  # sensor_focal > 100: # ML
                exec('ranges={}_with_range[3]//{}'.format(sensor_type, bin_size_per_sensor[sensor_focal]))
                for rng_100 in range(0, n_bins_of100m):
                    # ind = np.array([])
                    # exec('ind = np.where(ranges == rng_100)[0]')
                    ind = eval('np.where(ranges == rng_100)[0]')
                    stat_list_per_range = list()
                    if ind.any():  # if there were detections at that range bin
                        for ele in range(
                                3):  # taking the relevant preds related to the distance bin, since P/R/AP are computed globally vs. all GT/targets then it is compared to all targets
                            stat_list_per_range.append(eval(sensor_type)[ele][ind])
                        # GT at thta bin range
                        if not bool(
                                gt_per_range_bins[sensor_focal][rng_100]):  # predictions but no GT => FP=>low Prcesion
                            map50_per_range = np.array(0)
                            nt_stat_list_per_range = np.array(0)
                        else:
                            stat_list_per_range.append(np.array(
                                [x for x in gt_per_range_bins[sensor_focal][rng_100]]))  # add all targets/labels
                            nt_stat_list_per_range = np.bincount(stat_list_per_range[3].astype(np.int64), minlength=nc)

                            p_stat_list_per_range, r_stat_list_per_range, ap_stat_list_per_range, f1_stat_list_per_range, \
                                ap_class_stat_list_per_range = ap_per_class(*stat_list_per_range, plot=plots,
                                                                            v5_metric=v5_metric,
                                                                            save_dir='', names=names, tag='',
                                                                            class_support=nt_stat_list_per_range)

                            ap50_per_range, ap_per_range = ap_stat_list_per_range[:, 0], ap_stat_list_per_range.mean(
                                1)  # AP@0.5, AP@0.5:0.95
                            mp_per_range, mr_per_range, map50_per_range, map_per_range = p_stat_list_per_range.mean(), r_stat_list_per_range.mean(), ap50_per_range.mean(), ap_per_range.mean()
                    else:  # no prediction at this range
                        if not bool(gt_per_range_bins[sensor_focal][rng_100]):  # there are GT but no pred
                            nt_stat_list_per_range = np.array(0)
                            fn = len(gt_per_range_bins[sensor_focal][rng_100])
                            recall = 0  # no TP 0/TP+FN
                            precision = 0
                            map50_per_range = np.array(0)
                    print(map50_per_range)
                    range_bins_map[sensor_focal][rng_100] = map50_per_range.item()
                    range_bins_support[sensor_focal][rng_100] = nt_stat_list_per_range.sum().item()
    range_bar_plot(n_bins=17, range_bins=range_bins_map, save_dir=save_dir, range_bins_support=range_bins_support)
    # for time_var in time_vars:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--norm-type', type=str, default='standardization',
                        choices=['standardization', 'single_image_0_to_1', 'single_image_mean_std','single_image_percentile_0_255',
                                 'single_image_percentile_0_1', 'remove+global_outlier_0_1'],
                        help='Normalization approach')

    parser.add_argument('--no-tir-signal', action='store_true', help='')

    parser.add_argument('--tir-channel-expansion', action='store_true', help='drc_per_ch_percentile')

    parser.add_argument('--input-channels', type=int, default=3, help='')

    parser.add_argument('--save-path', default='', help='save to project/name')

    parser.add_argument('--csv-metadata-path', default='', help='save to project/name')

    parser.add_argument('--embed-analyse', action='store_true', help='')


    opt = parser.parse_args()

    if opt.tir_channel_expansion: # operates over 3 channels
        opt.input_channels = 3

    if opt.tir_channel_expansion and opt.norm_type != 'single_image_percentile_0_1': # operates over 3 channels
        print('Not a good combination')

    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()
    hyp = dict()

    model_name = str(opt.weights)[str(opt.weights).find('yolo'):].split('/')[0]

    compute_loss = False
    compute_loss_for_test = False # True
    if compute_loss_for_test:
        compute_loss=True

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             hyp=hyp,
             embed_analyse=opt.embed_analyse,
             model_name=model_name,
             compute_loss=compute_loss)

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
"""

--weights ./yolov7/yolov7.pt --device 0 --batch-size 16 --data data/coco_2_tir.yaml --img-size 640 --conf 0.6 --verbose --save-txt --save-hybrid --norm-type single_image_percentile_0_1
test based on RGB coco model
--weights ./yolov7/yolov7.pt --device 0 --batch-size 64 --data data/coco_2_tir.yaml --img-size 640 --conf 0.25 --verbose --save-txt --norm-type single_image_percentile_0_1 --project test --task train

--weights ./yolov7/yolov7.pt --device 0 --batch-size 64 --data data/tir_od.yaml --img-size 640 --conf 0.25 --verbose --save-txt --norm-type single_image_percentile_0_1 --project test --task val
# Using pretrained model
--weights /mnt/Data/hanoch/runs/train/yolov7434/weights/epoch_099.pt --device 0 --batch-size 4 --data data/tir_od_test_set.yaml --img-size 640 --conf 0.25 --verbose --norm-type single_image_percentile_0_1 --project test --task test
#vbased on 7555 mAP=82.3
--weights /mnt/Data/hanoch/runs/train/yolov7563/weights/best.pt --device 0 --batch-size 16 --data data/tir_od_test_set.yaml --img-size 640 --conf 0.02 --verbose --norm-type single_image_percentile_0_1 --input-channels 1 --project test --task test --iou-thres 0.4

/home/hanoch/projects/tir_od/runs/train/yolov7563/weights


--weights /mnt/Data/hanoch/runs/train/yolov7575/weights/best.pt --device 0 --batch-size 16 --data data/tir_od_test_set.yaml --img-size 640 --conf 0.001 --verbose --norm-type single_image_percentile_0_1 --input-channels 1 --project test --task test --iou-thres 0.6
--weights /home/hanoch/projects/tir_od/runs/gpu02/yolov74/weights --device 0 --batch-size 16 --data data/tir_od_test_set.yaml --img-size 640 --conf 0.001 --verbose --norm-type single_image_percentile_0_1 --input-channels 1 --project test --task test --iou-thres 0.6

3class
/mnt/Data/hanoch/runs/train/yolov71058
--weights /mnt/Data/hanoch/runs/train/yolov71058/weights/best.pt --device 0 --batch-size 16 --data data/tir_od_center_roi_aug_list_train_cls.yaml --img-size 640 --conf 0.02 --verbose --norm-type single_image_percentile_0_1 --input-channels 1 --project test --task val --iou-thres 0.6

tir_tiff_w_center_roi_validation_set_train_cls_usa.txt

# per day/nigh SY/ML
--weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.pt --device 0 --batch-size 16 --data data/tir_od_test_set.yaml --img-size 640 --conf 0.001 --verbose --norm-type single_image_percentile_0_1 --input-channels 1 --project test --task test --iou-thres 0.6 --csv-metadata-path tir_od/tir_center_merged_seq_tiff_last_original_png.xlsx
P/R
--weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.pt --device 0 --batch-size 16 --data data/tir_od_test_set.yaml --img-size 640 --verbose --norm-type single_image_percentile_0_1 --input-channels 1 --project test --task test --iou-thres 0.6 --csv-metadata-path tir_od/tir_center_merged_seq_tiff_last_original_png.xlsx --conf 0.65
mAP:
--weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.pt --device 0 --batch-size 16 --data data/tir_od_test_set.yaml --img-size 640 --verbose --norm-type single_image_percentile_0_1 --input-channels 1 --project test --task test --iou-thres 0.6 --csv-metadata-path tir_od/tir_center_merged_seq_tiff_last_original_png.xlsx --conf 0.01


Fixed wether csv  P/R
--weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.pt --device 0 --batch-size 16 --data data/tir_od_test_set.yaml --img-size 640 --verbose --norm-type single_image_percentile_0_1 --input-channels 1 --project test --task test --csv-metadata-path tir_od/tir_tiff_seq_png_3_class_fixed_whether.xlsx --iou-thres 0.6  --conf 0.65

FOG
--weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.pt --device 0 --batch-size 16 --data data/tir_od_fog_set.yaml --img-size 640 --verbose --norm-type single_image_percentile_0_1 --input-channels 1 --project test --task test --csv-metadata-path tir_od/tir_tiff_seq_png_3_class_fixed_whether.xlsx --conf 0.65 --iou-thres 0.6

Locomotive

--weights /mnt/Data/hanoch/runs/train/yolov71107/weights/best.pt --device 0 --batch-size 16 --data data/tir_od_test_set_3_class_train.yaml --img-size 640 --verbose --norm-type single_image_percentile_0_1 --input-channels 1 --project test --task test --iou-thres 0.6  --conf 0.1 --embed-analyse

--weights /mnt/Data/hanoch/runs/train/yolov71133/weights/best.pt --device 0 --batch-size 16 --data data/tir_od_test_set_3_class_train.yaml --img-size 640 --verbose --norm-type single_image_percentile_0_1 --input-channels 1 --project test --task test --iou-thres 0.6 --conf 0.4
-------  Error analysis  ------------
1st run with conf_th=0.0001 then observe the desired threshold, re-run with the desired threshold abd observe images with bboxes given the deired threshold 


--weights /mnt/Data/hanoch/runs/train/yolov71351/weights/epoch_049.pt --device 0 --batch-size 16 --data data/tir_od_test_set_3_class_train.yaml --img-size 640 --verbose --norm-type single_image_percentile_0_1 --input-channels 1 --project test --task test --csv-metadata-path process_tir/tir_tiff_seq_png_3_class_fixed_whether_copied_dataset_label.xlsx --iou-thres 0.6  --conf 0.1 

"""