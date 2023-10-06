import os 
import argparse
import random
import torch
import kwcoco
import glob
import cv2

import ubelt as ub
import numpy as np

from pathlib import Path

from angel_system.data.data_paths import grab_data
from angel_system.data.common.load_data import time_from_name

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, TracedModel
from utils.plots import plot_one_box
from utils.datasets import letterbox


def data_loader(recipes, split):
    training_split = {
        split: []
    }

    for recipe in recipes:
        ( ptg_root,
        recipe_data_dir,
        recipe_activity_config_fn,
        recipe_activity_gt_dir, 
        recipe_ros_bags_dir,
        recipe_training_split,
        recipe_obj_dets_dir,
        recipe_obj_config ) = grab_data(recipe, "gyges")

        training_split = {key: value + recipe_training_split[key] for key, value in training_split.items()}

    print("\nTraining split:")
    for split_name, videos in training_split.items():
        print(f"{split_name}: {len(videos)} videos")
        print([os.path.basename(v) for v in videos])
    print("\n")

    videos = training_split[split]
    return videos

def read_image(image_fn, imgsz, stride, device, half):
    img0 = cv2.imread(image_fn)  # BGR
    assert img0 is not None, 'Image Not Found ' + image_fn

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img0, img

def detect(opt):
    save_path = f"{opt.project}/{opt.name}"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size

    if not opt.no_trace:
        model = TracedModel(model, device, opt.img_size)

    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    dset = kwcoco.CocoDataset()

    # Add categories
    for i, object_label in enumerate(names):
        if object_label == "background":
            continue
        dset.add_category(name=object_label, id=i)

    videos = data_loader(opt.recipes, opt.split)
    for video in videos:
        video_name = os.path.basename(video)
        vid = dset.add_video(name=video_name)

        save_imgs_dir = f"{save_path}/images/{video_name}"
        Path(save_imgs_dir).mkdir(parents=True, exist_ok=True)

        images = glob.glob(f"{video}_extracted/images/*.png")
        for image_fn in ub.ProgIter(images, desc=f"images in {video_name}"):
            fn = os.path.basename(image_fn)
            img0, img = read_image(image_fn, imgsz, stride, device, half)
            height, width = img0.shape[:2]
            frame_num, time = time_from_name(image_fn)

            img_id = dset.add_image(
                file_name=image_fn,
                video_id=vid,
                frame_index=frame_num,
                width=width,
                height=height,
            )

            # Predict
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if not len(det):
                    continue
                
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                for *xyxy, conf, cls_id in reversed(det): # center xy, wh
                    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    norm_xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    cxywh = [norm_xywh[0] * width, norm_xywh[1] * height,
                            norm_xywh[2] * width, norm_xywh[3] * height] # center xy, wh
                    xywh = [cxywh[0] - (cxywh[2] / 2), cxywh[1] - (cxywh[3] / 2),
                            cxywh[2], cxywh[3]]

                    ann = {
                        "area": xywh[2] * xywh[3],
                        "image_id": img_id,
                        "category_id": cls_id,
                        "segmentation": [],
                        "bbox": xywh,
                        "confidence": float(conf),
                    }

                    dset.add_annotation(**ann)

                    # Optionaly draw results
                    if opt.save_img:  # Add bbox to image
                        label = f'{names[int(cls_id)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls_id)], line_thickness=1)
                if opt.save_img:
                    cv2.imwrite(f"{save_imgs_dir}/{fn}", img0)
    
    # Save
    dset.fpath = f"{save_path}/{opt.name}_{opt.split}_obj_results.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved predictions to {dset.fpath}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--recipes',
        type=str,
        nargs='+',
        default='coffee',
        help='Dataset(s)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Data split to run on'
    )
    parser.add_argument(
        '--weights',
        nargs='+',
        type=str,
        default='yolov7.pt',
        help='model.pt path(s)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=1280,
        help='inference size (pixels)'
    )
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.25,
        help='object confidence threshold'
    )
    parser.add_argument(
        '--iou-thres',
        type=float,
        default=0.45,
        help='IOU threshold for NMS'
    )
    parser.add_argument(
        '--device',
        default='',
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
    )
    parser.add_argument(
        '--agnostic-nms',
        action='store_true',
        help='class-agnostic NMS'
    )
    parser.add_argument(
        '--project',
        default='runs/detect',
        help='save results to project/name'
    )
    parser.add_argument(
        '--name',
        default='exp',
        help='save results to project/name'
    )
    parser.add_argument(
        '--no-trace',
        action='store_true',
        help='don`t trace model'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='augmented inference'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        help='filter by class: --class 0, or --class 0 2 3'
    )
    parser.add_argument(
        '--save-img',
        action='store_true',
        help='save results to *.png'
    )

    opt = parser.parse_args()
    print(opt)

    detect(opt)

if __name__ == '__main__':
    main()
