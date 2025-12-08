from typing import List, Union

import numpy as np

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import LeapDataType
from code_loader.contract.visualizer_classes import LeapHorizontalBar
from utils.datasets import create_dataloader
from leapcfg.config import CONFIG, hyp, data_dict, data_test_dict
import torch
# from src_tensorleap.infrastructure.logger import leaplogger
from models.yolo import Model
from utils.loss import ComputeLossOTA
# import tensorflow as tf
import yaml

from leap_utils import bb_array_to_object
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from utils.general import non_max_suppression
import os
from code_loader.default_metrics import flatten_non_batch_dims
from utils.general import box_iou, scale_coords, xywh2xyxy
from utils.metrics import ap_per_class, ConfusionMatrix, range_bar_plot, range_p_r_bar_plot
import numpy as np
import torch
from YOLOv7onnx import yolobbox_to_xyxy
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_preprocess, tensorleap_input_encoder, \
    tensorleap_gt_encoder, tensorleap_custom_visualizer, tensorleap_custom_loss, tensorleap_metadata, tensorleap_custom_metric

from code_loader.contract.datasetclasses import DataStateType

from code_loader.inner_leap_binder.leapbinder_decorators import *

from code_loader.contract.enums import MetricDirection
from utils.datasets import LoadImages

nc = 1 if CONFIG['SINGLE_CLS'] else int(data_dict['nc'])  # number of classes
img_shape = 640
print(3)
root = os.path.abspath(os.path.dirname(__file__))
torch_model = Model(os.path.join(root,CONFIG['CFG']), ch=3, nc=nc, anchors=hyp.get('anchors'))
nl = torch_model.model[-1].nl
hyp['box'] *= 3. / nl  # scale to layers
hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
hyp['obj'] *= (CONFIG['IMGSZ'] / img_shape) ** 2 * 3. / nl  # scale to image size and layers
hyp['label_smoothing'] = CONFIG['LABEL_SMOOTHING']
torch_model.nc = nc  # attach number of classes to model
torch_model.hyp = hyp  # attach hyperparameters to model
torch_model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
compute_loss_ota = ComputeLossOTA(torch_model)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
# Preprocess Function
def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes in YOLO format (center_x, center_y, width, height)

    Args:
        box1: [x, y, w, h] - first bounding box
        box2: [x, y, w, h] - second bounding box

    Returns:
        IoU value between 0 and 1
    """
    # Convert center format to corner format
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union area
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def match_predictions_to_targets(predictions, targets, iou_threshold=0.5, conf_th=0):
    """
    Match predictions to targets based on IoU threshold

    Args:
        predictions: numpy array of shape (N, 7) - [batch_idx, x, y, w, h, class_idx, confidence]
        targets: numpy array of shape (M, 6) - [batch_idx, class_idx, x, y, w, h]
        iou_threshold: IoU threshold for positive matches

    Returns:
        matched_pairs: list of tuples (pred_idx, target_idx)
        unmatched_preds: list of prediction indices without matches
        unmatched_targets: list of target indices without matches
    """
    matched_pairs = []
    unmatched_preds = list(range(len(predictions)))
    unmatched_targets = list(range(len(targets)))

    # Group by batch index for efficiency
    pred_batches = {}
    target_batches = {}

    for i, pred in enumerate(predictions):
        batch_idx = int(pred[0])
        if batch_idx not in pred_batches:
            pred_batches[batch_idx] = []
        pred_batches[batch_idx].append(i)

    for i, target in enumerate(targets):
        batch_idx = int(target[0])
        if batch_idx not in target_batches:
            target_batches[batch_idx] = []
        target_batches[batch_idx].append(i)

    # Match within each batch
    for batch_idx in pred_batches.keys():
        if batch_idx not in target_batches:
            continue

        batch_preds = pred_batches[batch_idx]
        batch_targets = target_batches[batch_idx]

        # Calculate IoU matrix for this batch
        iou_matrix = np.zeros((len(batch_preds), len(batch_targets)))

        for i, pred_idx in enumerate(batch_preds):
            pred = predictions[pred_idx]
            pred_box = pred[1:5]  # [x, y, w, h]

            for j, target_idx in enumerate(batch_targets):
                target = targets[target_idx]
                target_box = yolobbox_to_xyxy(*target[2:6], img_shape,img_shape)  # [x, y, w, h]

                # Only consider if classes match
                if int(pred[5]) == int(target[1]):
                    if pred[6] > conf_th:
                        iou_matrix[i, j] = compute_iou(pred_box, target_box)

        # Hungarian matching (simplified greedy approach)
        while True:
            # Find best match
            max_iou = np.max(iou_matrix)
            if max_iou < iou_threshold:
                break

            max_pos = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            pred_idx = batch_preds[max_pos[0]]
            target_idx = batch_targets[max_pos[1]]

            matched_pairs.append((pred_idx, target_idx))
            unmatched_preds.remove(pred_idx)
            unmatched_targets.remove(target_idx)

            # Remove matched prediction and target from consideration
            iou_matrix[max_pos[0], :] = 0
            iou_matrix[:, max_pos[1]] = 0

    return matched_pairs, unmatched_preds, unmatched_targets

def compute_yolo_cross_entropy_loss(predictions, targets, iou_threshold=0.5,
                                    num_classes=80, negative_weight=0.1, conf_th=0):
    """
    Compute cross-entropy loss for YOLO predictions given NMS output

    Args:
        predictions: numpy array of shape (100, 7) - [batch_idx, x, y, w, h, class_idx, confidence]
        targets: numpy array of shape (50, 6) - [batch_idx, class_idx, x, y, w, h]
        iou_threshold: IoU threshold for positive/negative assignment
        num_classes: Number of classes in the dataset
        negative_weight: Weight for negative samples to balance positive/negative ratio

    Returns:
        total_loss: Cross-entropy loss value
        loss_components: Dictionary with breakdown of loss components
    """
    # Convert to torch tensors if numpy arrays
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions).float()
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).float()

    # Convert back to numpy for matching algorithm
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()

    # Match predictions to targets
    matched_pairs, unmatched_preds, unmatched_targets = match_predictions_to_targets(
        pred_np, target_np, iou_threshold, conf_th=conf_th)

    total_loss = 0.0
    positive_loss = 0.0
    negative_loss = 0.0

    # Positive samples (matched predictions)
    if matched_pairs:
        pos_pred_indices = [pair[0] for pair in matched_pairs]
        pos_target_indices = [pair[1] for pair in matched_pairs]

        pos_predictions = predictions[pos_pred_indices]
        pos_targets = targets[pos_target_indices]

        # Extract confidence scores and target classes
        pred_confidences = pos_predictions[:, 6]  # Best class softmax confidence
        target_classes = pos_targets[:, 1].long()  # Target class indices

        # Create one-hot encoded targets for cross-entropy
        # Assuming confidence represents the probability of the predicted class
        pred_classes = pos_predictions[:, 5].long()  # Predicted class indices

        # Create soft targets (confidence as probability for predicted class, 0 for others)
        soft_targets = torch.zeros(len(pos_predictions), num_classes)
        for i, (pred_cls, conf) in enumerate(zip(pred_classes, pred_confidences)):
            soft_targets[i, pred_cls] = conf

        # For positive samples, we want high confidence for correct class
        pos_targets_one_hot = torch.zeros(len(pos_predictions), num_classes)
        pos_targets_one_hot.scatter_(1, target_classes.unsqueeze(1), 1.0)

        # Cross-entropy loss for positive samples
        pos_log_probs = torch.log(soft_targets + 1e-8)
        positive_loss = -torch.sum(pos_targets_one_hot * pos_log_probs) / len(pos_predictions)

    # Negative samples (unmatched predictions)
    if unmatched_preds:
        neg_predictions = predictions[unmatched_preds]
        neg_confidences = neg_predictions[:, 6]

        # For negative samples, we want low confidence (close to uniform distribution)
        uniform_target = torch.ones(len(neg_predictions), num_classes) / num_classes

        # Create soft predictions based on confidence
        neg_pred_classes = neg_predictions[:, 5].long()
        soft_neg_preds = torch.ones(len(neg_predictions), num_classes) * (1 - neg_confidences).unsqueeze(1) / (
                    num_classes - 1)
        for i, (pred_cls, conf) in enumerate(zip(neg_pred_classes, neg_confidences)):
            soft_neg_preds[i, pred_cls] = conf

        # Cross-entropy loss for negative samples
        neg_log_probs = torch.log(soft_neg_preds + 1e-8)
        negative_loss = -torch.sum(uniform_target * neg_log_probs) / len(neg_predictions)
        negative_loss *= negative_weight

    # Penalty for unmatched targets (missed detections) FN = -1*log(0.01) = 4.6  Training Signal is Missing → Implicit FN Penalty
    # The FN leads to a missed learning opportunity: the network gets no gradient signal for that object. This means the model won’t improve for that object class/location on that iteration.     # In some detectors, e.g., YOLO, this is mitigated using objectness confidence loss (penalizing missing detections).
    # For FN where no box overlaps:
    # Cross-entropy loss for classification is zero or absent.
        # No prediction is assigned, so no classification loss is backpropagated.
        # 🧪 Consequence:
    # The model doesn’t "see" the object and doesn't learn from it → gradients don’t flow for that instance.

    missed_detection_penalty = torch.Tensor(np.array((len(unmatched_targets) * 4.6)).astype('float32')) #2.0  # Penalty for each missed target

    total_loss = positive_loss + negative_loss + missed_detection_penalty

    loss_components = {
        'positive_loss': positive_loss.item() if isinstance(positive_loss, torch.Tensor) else positive_loss,
        'negative_loss': negative_loss.item() if isinstance(negative_loss, torch.Tensor) else negative_loss,
        'missed_detection_penalty': missed_detection_penalty,
        'num_positive_pairs': len(matched_pairs),
        'num_negative_samples': len(unmatched_preds),
        'num_missed_targets': len(unmatched_targets)
    }

    return total_loss, loss_components

csv_data_path = 'tir_tiff_seq_png_3_class_fixed_whether_copied_dataset_7l_tleap.xlsx'
def preprocess_func() -> List[PreprocessResponse]:
    # Hyperparameters
    train_path = data_dict['train']
    val_path = data_dict['val']
    test_path = data_test_dict['test']
    train_dataloader, dataset = create_dataloader(train_path, CONFIG['IMGSZ'], 1, CONFIG['GS'],
                                            Namespace(single_cls=CONFIG['SINGLE_CLS'], norm_type='single_image_percentile_0_1',
                                                      input_channels=1, tir_channel_expansion=False,
                                                      no_tir_signal=False, csv_metadata_path=csv_data_path),
                                            num_cls=nc, hyp=hyp, augment=False, cache=CONFIG['CACHE_IMAGES'], rect=False,
                                            rank=-1, world_size=1, workers=CONFIG['WORKERS'],
                                            image_weights=False, quad=CONFIG['QUAD'], prefix='train: ')


    val_dataloader, dataset = create_dataloader(val_path, CONFIG['IMGSZ_TEST'], 1, CONFIG['GS'],\
                                                 Namespace(single_cls=CONFIG['SINGLE_CLS'], norm_type='single_image_percentile_0_1',
                                                      input_channels=1, tir_channel_expansion=False, no_tir_signal=False, csv_metadata_path=csv_data_path),  # testloader
                                                num_cls=nc, hyp=hyp, augment=False, cache=CONFIG['CACHE_IMAGES'], rect=False, rank=-1,
                                                world_size=1, workers=CONFIG['WORKERS'],
                                                prefix='val: ')

    test_dataloader, dataset = create_dataloader(test_path, CONFIG['IMGSZ_TEST'], 1, CONFIG['GS'],\
                                                 Namespace(single_cls=CONFIG['SINGLE_CLS'], norm_type='single_image_percentile_0_1',
                                                      input_channels=1, tir_channel_expansion=False, no_tir_signal=False, csv_metadata_path=csv_data_path),  # testloader
                                                num_cls=nc,hyp=hyp, augment=False, cache=CONFIG['CACHE_IMAGES'], rect=False, rank=-1,
                                                world_size=1, workers=CONFIG['WORKERS'],
                                                prefix='test: ')

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs
    train = PreprocessResponse(length=train_dataloader.dataset.__len__(), data={'dataset1': train_dataloader.dataset})
    val = PreprocessResponse(length=val_dataloader.dataset.__len__(), data={'dataset1': val_dataloader.dataset})
    test = PreprocessResponse(length=test_dataloader.dataset.__len__(), data={'dataset1': test_dataloader.dataset})
    response = [train, val, test]
    return response

@tensorleap_preprocess()
def preprocess_func_no_csv() -> List[PreprocessResponse]:
    # Hyperparameters
    train_path = data_dict['train']
    val_path = data_dict['val']
    test_path = data_test_dict['test']

    train_dataloader, dataset = create_dataloader(train_path, CONFIG['IMGSZ'], 1, CONFIG['GS'],
                                            Namespace(single_cls=CONFIG['SINGLE_CLS'], norm_type='single_image_percentile_0_1',
                                                      input_channels=1, tir_channel_expansion=False,
                                                      no_tir_signal=False, csv_metadata_path=''),
                                                  num_cls=nc,
                                            hyp=hyp, augment=False, cache=CONFIG['CACHE_IMAGES'], rect=False,
                                            rank=-1, world_size=1, workers=CONFIG['WORKERS'],
                                            image_weights=False, quad=CONFIG['QUAD'], prefix='train: ')


    val_dataloader, dataset = create_dataloader(val_path, CONFIG['IMGSZ_TEST'], 1, CONFIG['GS'],\
                                                 Namespace(single_cls=CONFIG['SINGLE_CLS'], norm_type='single_image_percentile_0_1',
                                                      input_channels=1, tir_channel_expansion=False, no_tir_signal=False, csv_metadata_path=''),  # testloader
                                                num_cls=nc,hyp=hyp, augment=False, cache=CONFIG['CACHE_IMAGES'], rect=False, rank=-1,
                                                world_size=1, workers=CONFIG['WORKERS'],
                                                prefix='val: ')

    test_dataloader, dataset = create_dataloader(test_path, CONFIG['IMGSZ_TEST'], 1, CONFIG['GS'],\
                                                 Namespace(single_cls=CONFIG['SINGLE_CLS'], norm_type='single_image_percentile_0_1',
                                                      input_channels=1, tir_channel_expansion=False, no_tir_signal=False, csv_metadata_path=''),  # testloader
                                                num_cls=nc, hyp=hyp, augment=False, cache=CONFIG['CACHE_IMAGES'], rect=False, rank=-1,
                                                world_size=1, workers=CONFIG['WORKERS'],
                                                prefix='test: ')

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs
    train = PreprocessResponse(length=train_dataloader.dataset.__len__(), data={'dataset1': train_dataloader.dataset}, state=DataStateType.training)
    val = PreprocessResponse(length=val_dataloader.dataset.__len__(), data={'dataset1': val_dataloader.dataset}, state=DataStateType.validation)
    test = PreprocessResponse(length=test_dataloader.dataset.__len__(), data={'dataset1': test_dataloader.dataset}, state=DataStateType.test)
    response = [train, val, test]
    return response

@tensorleap_unlabeled_preprocess()
def preprocess_unlabeled_func_leap() -> List[PreprocessResponse]:
    unlabeled_dataset = LoadImages('/mnt/Data/hanoch/tir_frames_rois/unlabeled_210_data_set_tleap.txt',#CONFIG['UNLABELED_DATASET'],
                                   CONFIG['IMGSZ_TEST'], CONFIG['GS'], \
                                  scaling_type='single_image_percentile_0_1',
                                             input_channels=1, tir_channel_expansion=False, no_tir_signal=False,
                                             rel_path_for_list_files=CONFIG['rel_path_for_list_files'], detection_ordering=False)

    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset,
                                                 batch_size=1,
                                                 pin_memory=True,
                                                 collate_fn=unlabeled_dataset.unlabeled_tleap_collate_fn)

    # unlabeled_test = PreprocessResponse(data=test_unlabeled_dataset,
    #                                     state=DataStateType.unlabeled,
    #                                     sample_ids=test_mapped_strings, sample_id_type=str)
    #
    unlabeled = PreprocessResponse(length=len(unlabeled_loader.dataset), data={'dataset1':unlabeled_loader.dataset}, state=DataStateType.unlabeled)
    print('Done...preprocess_unlabeled_func_leap')

    return unlabeled

# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
@tensorleap_input_encoder('image')
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    image = preprocess.data['dataset1'][idx][0]
    if isinstance(image, torch.Tensor):
        return image.permute((1, 2, 0)).numpy().astype('float32')
    else:
        return image.transpose((1, 2, 0)).astype('float32')



# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    #TODO - problem - can't have dynamic GT shape?
    #Remove none-zero gt before picking up
    torch_gt = preprocess.data['dataset1'][idx][1]
    np_gt = np.zeros((CONFIG['MAX_INSTANCE'], 6))
    np_gt[:, 1] = nc + 1
    instances_count = torch_gt.shape[0]
    np_gt[:instances_count, :] = torch_gt[:min(CONFIG['MAX_INSTANCE'], instances_count), :]
    np_gt = np_gt.astype('float32')
    return np_gt


def arc_sigmoid(pred):
    return torch.log(pred / (1 - pred))


def transform_xy(pred, strides):
    x = [(pred[i][..., :2] / strides[i]
          + 0.5
          - torch_model.model[-1]._make_grid(*pred[i].shape[2:4]))/2 for i in range(len(pred))]
    for i in range(len(pred)):
        pred[i][..., :2] = arc_sigmoid(x[i])
    return pred


def transform_wh(pred):
    for i in range(len(pred)):
        pred[i][..., 2:4] =  arc_sigmoid(torch.sqrt(pred[i][..., 2:4] /
                                   torch_model.model[-1].anchor_grid[i]) / 2)
    return pred


def transform_conf(pred):
    for i in range(len(pred)):
        pred[i][..., 4:] = arc_sigmoid(pred[i][..., 4:])
    return pred

# @tensorleap_custom_loss('od_loss')
# def custom_loss_dummy(predictions, targets):
#     err_str = f"-------  custome loss  !!!!!!{predictions.shape}"
#     leaplogger.warning(err_str)
#
#     return np.array(0).astype('float32')

@tensorleap_custom_loss('od_loss')
def custom_loss(predictions, targets):


    # Compute loss
    predictions = predictions[0, :, :]
    loss, components = compute_yolo_cross_entropy_loss(
        predictions[predictions[:,6]>0], targets[0, targets[0, :, 1] <nc, :],
        iou_threshold=0.5,
        num_classes=nc,
        negative_weight=1
    )
    # fp = components['num_negative_samples']
    # fn = components['num_missed_targets']
    # tp = components['num_positive_pairs']
    # precision = tp/(tp+fp)
    # recall = tp/(tp+fn)
    # t = targets[0, targets[0, :, 1] <nc, :]
    # [(t1[1], yolobbox_to_xyxy(*t1[2:6], 640,640)) for t1 in t]
    print(f"Total Loss: {loss:.4f}")
    print("Loss Components:")
    if not isinstance(loss, torch.Tensor):
        print('type(loss)', type(loss))

    for key, value in components.items():
        print(f"  {key}: {value}")
    loss = loss[None, ...]
    loss = loss.numpy()
    return loss

def custom_loss2(prediction, targets):

    # ground_truth, prediction = flatten_non_batch_dims(gt, prediction)
    # return 0
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    seen = 0

    for si, pred in enumerate(prediction):  # [bbox_coors, objectness_logit, class]

        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        # path = Path(paths[si])
        seen += 1

        # No predictions in the image but only GT
        if len(pred) == 0:
            if nl:

                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(),
                              tcls))  # niou for COCO 0.5:0.05:1
            continue

        # Predictions
        predn = pred.clone()  # *xyxy, conf, cls in predn  [x y ,w ,h, conf, cls] taking top 300 after NMS
        scale_coords(img_shape, predn[:, :4], img_shape, img_shape)  # native-space pred

        # Append to text file
        # Assign all predictions as incorrect ; pred takes top 300 predictions conf over 10 ious [0.5:0.95:0.05]
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])
            scale_coords(img_shape, tbox, img_shape, img_shape)  # native-space labels
            # if plots:
            #     confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

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
                    for j in (ious > iouv[0]).nonzero(
                            as_tuple=False):  # iouv[0]=0.5 IOU for dectetions iouv in general are all 0.5:0.05:.. for COCO
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf_objectness, pcls, tcls) Predicted class is Max-Likelihood among all classes logit and threshol goes over the objectness only
        stats.append(
            (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # correct @ IOU=0.5 of pred box with target

        # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, v5_metric=False, save_dir='',
                            names=['car', 'person', 'locomotive'], class_support=nt, tag='tir_od_3classes') #based on correct @ IOU=0.5 of pred box with target
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()



@tensorleap_custom_visualizer('bb_decoder', LeapDataType.ImageWithBBox)
def pred_visualizer(pred, img):
    # err_str = f"-------  prediction shape !!!!!!{pred.shape}"
    # leaplogger.warning(err_str)

    img = img[0,:,:,:]
    # out = non_max_suppression(torch.from_numpy(pred[None, ...]),
    #                           conf_thres=CONFIG['NMS']['CONF_THRESH'],
    #                           iou_thres=CONFIG['NMS']['IOU_THRESH'], multi_label=True)[0].numpy()
    if any(pred[0, :, 6] > 0):
        pred_ = pred[0, pred[0, :, 6] > 0,:]
        pred_ = pred_[:, 1:]
        pred_[:, :4] = pred_[:, :4] / CONFIG['IMGSZ']
        new_order = [0, 1, 2, 3, 5, 4]
        pred_reshaped = pred_[:,new_order]
    else:
        # pred_reshaped = pred[0,0,:6]
        return LeapImageWithBBox((img * 255).astype(np.uint8), 0.0)
    res = bb_array_to_object(pred_reshaped, iscornercoded=True, bg_label=-1, is_gt=False)
    return LeapImageWithBBox((img * 255).astype(np.uint8), res)

@tensorleap_custom_visualizer('bb_gt_decoder', LeapDataType.ImageWithBBox)
def gt_visualizer(gt, img):
    gt_inter = gt[0, gt[0, :, 1] < nc,:]
    gt_inter = np.append(np.ones((gt_inter.shape[0],1)),gt_inter ,axis=1)
    new_order = [3, 4, 5, 6, 0, 2]
    gt_reshaped = gt_inter[:,new_order]
    img = img[0,:,:,:]
    # gt_permuted = np.concatenate([gt[:, 2:], gt[:, :2]], axis=-1)
    res = bb_array_to_object(gt_reshaped, iscornercoded=False, bg_label=-1, is_gt=True)
    return LeapImageWithBBox((img * 255).astype(np.uint8), res)


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
# def metadata_label(idx: int, preprocess: PreprocessResponse) -> int:
#     one_hot_digit = gt_encoder(idx, preprocess)
#     digit = one_hot_digit.argmax()
#     digit_int = int(digit)
#     return digit_int

@tensorleap_metadata('image_path')
def metadata_image_path(idx: int, preprocess: PreprocessResponse) -> dict:
    # index_of_element = int(preprocess.sample_ids.index(idx))
    # print(preprocessing.data[idx][3])
    (_, _, path, _) = preprocess.data['dataset1'].__getitem__(idx)
    response = {'path': path}
    return response

def metadata_image_path_wishful(idx: int, preprocess: PreprocessResponse) -> dict:
    # index_of_element = int(preprocess.sample_ids.index(idx))
    # print(preprocessing.data[idx][3])
    (_, _, path, _) = preprocess.data['dataset1'].__getitem__(idx)
    sensor_type = preprocess.data['dataset1'].df_metadata[
        preprocess.data['dataset1'].df_metadata['tir_frame_image_file_name'] == str(path).split('/')[-1]]['sensor_type'].item()

    time_in_day = preprocess.data['dataset1'].df_metadata[
        preprocess.data['dataset1'].df_metadata['tir_frame_image_file_name'] == str(path).split('/')[-1]][
        'part_in_day'].item()

    weather_condition = preprocess.data['dataset1'].df_metadata[
        preprocess.data['dataset1'].df_metadata['tir_frame_image_file_name'] == str(path).split('/')[-1]][
        'weather_condition'].item()

    response = {'path': path, 'sensor_type':sensor_type, 'time_in_day': time_in_day, 'weather_condition': weather_condition}
    return response

@tensorleap_custom_metric('accuracy_f1', direction=MetricDirection.Upward)
def accuracy_f1(predictions, targets, conf_th=0.1):

    predictions = predictions[0, :, :]
    loss, components = compute_yolo_cross_entropy_loss(
        predictions[predictions[:,6]>0], targets[0, targets[0, :, 1] <nc, :],
        iou_threshold=0.5,
        num_classes=nc,
        negative_weight=1,
        conf_th=conf_th
    )
    fp = components['num_negative_samples']
    fn = components['num_missed_targets']
    tp = components['num_positive_pairs']
    if tp>0:
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2/(1/precision + 1/recall)
    else:
        f1 = 0
    return np.atleast_1d(np.array(f1))

@tensorleap_custom_metric('images_stat')
def image_stats(targets : np.ndarray) -> dict:
    targets_act = targets[0, targets[0, :, 1] < nc, :]
    nt = np.bincount(targets_act[:,1].astype('int'), minlength=nc)
    stat = {'no_objects_cls_{}'.format(c): np.atleast_1d(np.array(i)).astype('float32') for i, c in
            zip(nt, range(nc))}
    if nt[0]==0:
        print('nt', nt)
    box_area = {v: list() for v in range(nc)}
    for targets_ in targets_act:

        target_box = yolobbox_to_xyxy(*targets_[2:6], img_shape, img_shape)
        area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
        box_area[targets_[1]].append(area)

    box_stat = {'class_area_avg_{}'.format(k): np.atleast_1d(np.array(np.array(v).mean().astype('float32'))) if v else np.atleast_1d(np.array(np.float32(0))) for k, v in
                box_area.items()}

    stat.update(box_stat)

    box_stat_min = {'class_area_min_{}'.format(k): np.atleast_1d(np.array(np.array(v).min().astype('float32'))) if v else np.atleast_1d(np.array(np.float32(0))) for k, v in
                box_area.items()}
    stat.update(box_stat_min)

    box_stat_max = {'class_area_max_{}'.format(k): np.atleast_1d(np.array(np.array(v).max().astype('float32'))) if v else np.atleast_1d(np.array(np.float32(0))) for k, v in
                box_area.items()}
    stat.update(box_stat_max)

    if 0:
        stat = dict()
        stat['class_area_0'] = np.array([961.0])[None, ...].astype('float32')
    return stat

#                     sensor_type = dataloader.dataset.df_metadata[dataloader.dataset.df_metadata['tir_frame_image_file_name'] == str(path).split('/')[-1]]['sensor_type'].item()
# (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s))
# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.add_prediction(name='classes', labels=['X', 'Y', 'W', ' H', ' Conf'] + data_dict['names'])

if __name__ == '__main__':
    leap_binder.check()


#