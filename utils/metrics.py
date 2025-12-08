# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from . import general


def fitness(x, w=[0.0, 0.0, 0.1, 0.9]):
    # Model fitness as a weighted combination of metrics
    # w =   # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)

from pathlib import PosixPath

def ap_per_class(tp, conf, pred_cls, target_cls, v5_metric=False, plot=False,
                 save_dir='.', names=(), tag='', class_support:np.ndarray=np.array([])):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10). nx10 due to 10 IOUs over 10 ious [0.5:0.95:0.05]
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    if not isinstance(save_dir, PosixPath):
        save_dir = Path(save_dir)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs {HK: since tp computed as the oned that has IOUs>th with GT hence the other predictions are FP by definition}
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases. interpolate over desired px linearily spaced [0:1] by recall=func(conf). r[ci] are the recall=func(conf=px)
            # Hence px becomes the new conf linearily spaced axis
            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score. same interp for precision=func(conf) = > precision@ px linealiy spaced

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], v5_metric=v5_metric)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        if bool(py):
            plot_pr_curve(px, py, ap, Path(os.path.join(save_dir, 'PR_curve_' + tag + '.png')),
                          names=names, class_support=class_support)
        plot_mc_curve(px, f1, Path(os.path.join(save_dir, 'F1_curve_' + tag + '.png')), ylabel='F1')
        plot_mc_curve(px, p, Path(os.path.join(save_dir, 'P_curve_' + tag + '.png')), ylabel='Precision')
        plot_mc_curve(px, p, Path(os.path.join(save_dir, 'R_curve_' + tag + '.png')), ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision, v5_metric=False):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
        v5_metric: Assume maximum recall to be 1.0, as in YOLOv5, MMDetetion etc.
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    if v5_metric:  # New YOLOv5 metric, same as MMDetection and Detectron2 repositories
        mrec = np.concatenate(([0.], recall, [1.0]))
    else:  # Old YOLOv5 metric, i.e. default YOLOv7 metric
        mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = general.box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted @ th={}'.format(self.conf))
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------

def range_p_r_bar_plot(n_bins:int, range_bins_precision_all_classes:dict, range_bins_recall_all_classes:dict,
                   save_dir:str, bar_width:int = 25, range_bins_support:dict={}, names:str='', conf:float=0):

    for class_graph in range(len(names)):
        x = 100 * np.arange(n_bins) + 100
        for (k_p, v_p), (k_r, v_r) in zip(range_bins_precision_all_classes.items(), range_bins_recall_all_classes.items()):
            plt.figure()
            try:
                bar1 = plt.bar(x-bar_width , [x[class_graph] for x in v_p[:n_bins]], bar_width, color='b', label='p')
                bar2 = plt.bar(x-bar_width//2 , [x[class_graph] for x in v_r[:n_bins]], bar_width, color='g', label='r')
            except Exception as e:
                print(f'pycocotools unable to run: {e}')

            plt.xticks(x, (x/100).astype('int'))

            # if bool(range_bins_support):
            #     for ix, rect in enumerate(bar1):
            #         height = rect.get_height()
            #         plt.text(rect.get_x() + rect.get_width() / 2.0, height, 'p/r', ha='center', va='bottom')

            plt.legend()
            # plt.tight_layout()
            # plt.ylim([0.0, 1.05])
            plt.ylabel('P/R')
            plt.xlabel('Range[x100m]')
            plt.grid()
            plt.title('Sensor {}mm Precision/Recall vs. range. Class: {} conf: {}'.format(k_p, str(names[class_graph]), conf))
            # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.savefig(os.path.join(save_dir, 'p_r_distribution_distance_sensor_' + str(k_p) + '_class_' + str(names[class_graph]) +'.png'), dpi=250)
            plt.clf()
            plt.close()

        pass
def range_bar_plot(n_bins, range_bins, save_dir, bar_width = 25, range_bins_support={}):

    x = 100 * np.arange(n_bins) + 100
    if 0:
        x= x/100
    for k, v in range_bins.items():
        plt.figure()
        bar1 = plt.bar(x , v[:n_bins], bar_width, color='skyblue')
        x_positions = np.arange(len(v))
        # plt.xticks(x_positions, v)
        if 1:
            plt.xticks(x, (x/100).astype('int'))
        if bool(range_bins_support):
            for ix, rect in enumerate(bar1):
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                         f'{range_bins_support[k][ix]:.0f}', ha='center', va='bottom')

        # plt.legend()
        # plt.tight_layout()
        # plt.ylim([0.0, 1.05])
        plt.ylabel('mAP')
        plt.xlabel('Range[x100m]')
        plt.grid()
        plt.title('Sensor {}mm mAP vs. range[x100m]'.format(k))
        # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.savefig(os.path.join(save_dir, 'mAP_distribution_distance_sensor_' + str(k) + '.png'), dpi=250)
        plt.clf()
        plt.close()

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=(),
                  precisions_of_interest=[0.95, 0.9, 0.85],
                  class_support:np.ndarray=np.array([])):

    tag = str(save_dir).split('/')[-1].split('.')[0].split('PR_curve_stats_all_')[-1]
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):

            recall_of_interest_per_class = np.zeros_like(precisions_of_interest)
            try:
                if np.array(precisions_of_interest).min() < np.array(py[:, i]).max(): # make sure that precision has the values in the interest ROI
                    recall_of_interest_per_class = [px[int(np.where(y.reshape(-1) > x)[0][-1])] for x in precisions_of_interest]
                    conf_at_precision_of_iterest = px[::-1][[int(np.where(y.reshape(-1) > x)[0][-1]) for x in precisions_of_interest]]
            except Exception as e:
                print(e)
                print(precisions_of_interest)

            if class_support.any():
                ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f} ({class_support[i]})')  # plot(recall, precision)
            else:
                ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
            ax.plot(recall_of_interest_per_class, precisions_of_interest, '*', color='green')
            for k in range(len(precisions_of_interest)):
                ax.plot(recall_of_interest_per_class[k], precisions_of_interest[k], '*', color='green')
                try:
                    ax.text(x=recall_of_interest_per_class[k], y=precisions_of_interest[k], fontsize=12,
                            s=f"th={conf_at_precision_of_iterest[k]:.2f}")
                except Exception as e:
                    print(f'WARNING: cant plot recall of interest too few data or so: {e}')

                # ax.text(x=0.6, y=precisions_of_interest[i], fontsize=12, s=f" R/P {names[i]}[ {recall_of_interest_per_class[i]:.3f}    {precisions_of_interest[i]:.3f}]")
                # ax.text(x=0.6, y=max(0.9-0.2*i, 0), fontsize=12, s=f" R/P {names[i]}[ {recall_of_interest_per_class[i]:.3f}    {precisions_of_interest[i]:.3f}]")
                if k == 0:
                    ax.text(x=min(0.1 + 0.4 * k, 1), y=max(0.5 - 0.2 * i, 0), fontsize=12,
                            s=f" R@Pr {names[i]}[{recall_of_interest_per_class[k]:.2f};{precisions_of_interest[k]:.2f}]")
                else:
                    ax.text(x=min(0.1 + (0.5 - k*0.1) * k, 1), y=max(0.5 - 0.2 * i, 0), fontsize=12,
                        s=f"[{recall_of_interest_per_class[k]:.2f};{precisions_of_interest[k]:.2f}]")

            print('Class: {} : Thresholds : {} per precision {}'.format(names[i], conf_at_precision_of_iterest,
                                                                        precisions_of_interest))
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    recall_of_interest = np.zeros_like(precisions_of_interest)
    try:
        if np.array(precisions_of_interest).min() < np.array(py.mean(1)).max():
            recall_of_interest = [px[int(np.where(py.mean(1).reshape(-1) > x)[0][-1])] for x in precisions_of_interest]
    except Exception as e:
        print(e)
        print(precisions_of_interest)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean()) # py [ap , num_clases]
    ax.plot(recall_of_interest, precisions_of_interest, '*')
    ax.plot([1, 0], color='navy', linewidth=2, linestyle='--')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid()
    ax.set_title(tag)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid()
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()
