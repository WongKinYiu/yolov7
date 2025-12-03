import argparse
import copy
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
import tifffile
import copy
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, scaling_image, LoadImagesAndLabels
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave #and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.lower().startswith(  # removed HK or source.endswith('.txt')
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = webcam and not source.endswith('.txt')
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size, opt.input_channels)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        resize_or_crop = 'center_crop'
        dataset = LoadImages(source, img_size=imgsz, stride=stride,
                             scaling_type=opt.norm_type, input_channels=opt.input_channels,
                             no_tir_signal=opt.no_tir_signal,
                             tir_channel_expansion=opt.tir_channel_expansion,
                             rel_path_for_list_files=opt.rel_path_for_list_files,
                             resize_or_crop=resize_or_crop)

        dataset_loader = torch.utils.data.DataLoader(dataset,
                            batch_size=1,
                            pin_memory=True,
                                              collate_fn=dataset.unlabeled_collate_fn)


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, opt.input_channels, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset_loader:

        if os.path.basename(path).split('.')[-1] == 'tiff': # im0s only for plotting version
            im0s = np.repeat(im0s[ :, :, np.newaxis], 3, axis=2) # convert GL to RGB by replication
            im0s = scaling_image(im0s, scaling_type=opt.norm_type)
            if im0s.max()<=1:
                im0s = im0s*255

            # im0s = copy.deepcopy(np.uint8(img.transpose(1,2,0) * 255.0))

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                if resize_or_crop == 'center_crop':
                    del_x = (im0.shape[1] - img.shape[-1])//2
                    del_y = (im0.shape[0] - img.shape[-2])//2
                    # det[:, :4] = det[:, :4] + [del_y, del_y, del_x, del_x]
                    det[:, [0, 2]] += del_x  # x padding
                    det[:, [1, 3]] += del_y  # y padding

                elif resize_or_crop == 'resize':
                # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                else:
                    raise

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    print(save_path,os.path.basename(save_path).split('.'))
                    if os.path.basename(save_path).split('.')[-1] == 'tiff':
                        #print('ka')
                        save_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split('.')[0] + '.png')
                        cv2.imwrite(save_path, im0)
                    else:
                        cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--norm-type', type=str, default='standardization',
                        choices=['standardization', 'single_image_0_to_1', 'single_image_mean_std','single_image_percentile_0_255',
                                 'single_image_percentile_0_1', 'remove+global_outlier_0_1'],
                        help='Normalization approach')

    parser.add_argument('--no-tir-signal', action='store_true', help='')

    parser.add_argument('--tir-channel-expansion', action='store_true', help='drc_per_ch_percentile')

    parser.add_argument('--input-channels', type=int, default=1, help='')

    parser.add_argument('--save-path', default='', help='save to project/name')

    parser.add_argument('--rel-path-for-list-files', default='/mnt/Data/hanoch/tir_frames_rois/yolo7_tir_data_all', help='')



    opt = parser.parse_args()

    if opt.tir_channel_expansion: # operates over 3 channels
        opt.input_channels = 3

    if opt.tir_channel_expansion and opt.norm_type != 'single_image_percentile_0_1': # operates over 3 channels
        print('Not a good combination')

    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
#  --source yolov7/tir_od/fog_data_set.txt

"""
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
python -u ./yolov7/detect.py --weights ./yolov7/yolov7.pt --conf 0.25 --img-size 640 --device 0 --save-txt --source /home/hanoch/projects/tir_frames_rois/png/Rotem_test_22c_dec18.png
python -u ./yolov7/detect.py --weights ./yolov7/yolov7.pt --conf 0.25 --img-size 640 --device 0 --save-txt --norm-type single_image_percentile_0_1 --source /home/hanoch/projects/tir_frames_rois/yolo7_tir_data_all/TIR10_v20_Dec18_Test22C_20181127_223533_FS_210F_0001_5500_ROTEM_left_roi_220_4707.tiff
--weights ./yolov7/yolov7.pt --conf 0.25 --img-size 640 --device 0 --save-txt --norm-type single_image_percentile_0_1 --source /home/hanoch/projects/tir_frames_rois/yolo7_tir_data_all/TIR10_v20_Dec18_Test22C_20181127_223533_FS_210F_0001_5500_ROTEM_left_roi_220_4707.tiff
--weights ./yolov7/yolov7.pt --conf 0.25 --img-size 640 --device 0 --save-txt --norm-type single_image_percentile_0_1 --source /home/hanoch/projects/tir_frames_rois/yolo7_tir_data_all/TIR10_V50_OCT21_Test46A_ML_RD_IL_2021_08_05_14_48_05_FS_210_XGA_630_922_DENIS_right_roi_210_881.tiff
--weights ./yolov7/yolov7.pt --conf 0.25 --img-size 640 --device 0 --save-txt --norm-type single_image_percentile_0_1 --source /home/hanoch/projects/tir_frames_rois/yolo7_tir_data_all/TIR135_V80_JUL23_Test55A_SY_RD_US_2023_01_18_07_29_38_FS_50_XGA_0001_3562_Shahar_left_roi_50_1348.tiff

--weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.pt --conf 0.25 --img-size 640 --device 0 --save-txt --norm-type single_image_percentile_0_1 --source /home/hanoch/projects/tir_frames_rois/fog/28_02_2019_16_05_01[1]_04783.tiff

--weights /mnt/Data/hanoch/runs/train/yolov7999/weights/best.pt --img-size 640 --device 0 --save-txt --norm-type single_image_percentile_0_1 --source yolov7/tir_od/fog_data_set.txt --conf 0.65


YOLO model
--weights ./yolov7/yolov7.pt --conf 0.25 --img-size 640 --device 0 --save-txt --norm-type single_image_percentile_0_1 --source /home/hanoch/projects/tir_od/Snipaste_2024-09-15_09-00-58_tir_135_TIR135_V80_JUL23_Test55A_SY_RD_US_2023_01_18_07_29_38_FS_50_XGA_0001_3562_Shahar_left_roi_50_1348.png

--weights /mnt/Data/hanoch/runs/train/yolov7575/weights/best.pt --conf 0.01 --img-size 640 --input-channels 1 --device 0 --save-txt --norm-type single_image_percentile_0_1 --source /home/hanoch/projects/tir_frames_rois/tir_tiff_tiff_files/TIR8_V50_Test19G_Jul20_2018-12-06_13-39-17_FS_50F_0114_6368_ROTEM_right_roi_50_345.tiff

Failed training-set locomotiver
--weights /mnt/Data/hanoch/runs/train/yolov71351/weights/epoch_049.pt --img-size 640 --device 0 --save-txt --norm-type single_image_percentile_0_1 --conf 0.65 --source /mnt/Data/hanoch/tir_frames_rois/yolo7_tir_data_all/TIR28_V90_JAN25_Test52C_2022_10_26_21_01_06_FS_50_XGA_0001_2898_MICHAEL_center_roi_50_1318.tiff 


# unlabeled data
--weights /mnt/Data/hanoch/runs/train/yolov71351/weights/epoch_049.pt --img-size 640 --device 0 --save-txt --norm-type single_image_percentile_0_1 --source yolov7/tir_od/unlabeled_210_data_set.txt --conf 0.65 --rel-path-for-list-files /mnt/Data/hanoch/tir_frames_rois/unlabeled
--weights /mnt/Data/hanoch/runs/train/yolov71351/weights/epoch_049.pt --img-size 640 --device 0 --save-txt --norm-type single_image_percentile_0_1 --source /mnt/Staging_Isilon/1306_data/TIR_snapshots
"""