import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from tqdm import tqdm
import os, sys, json

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, save_json, save_img = \
        opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.save_json, opt.save_img

    #save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    #JSON Dict
    jdict = []    

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    try:
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        print(f'\nExperiment Directory: {save_dir}\n')
    except OSError as e:
        sys.exit(f'\nCannot create directory: {e}')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

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
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    for batchi, (path, img, im0s, vid_cap) in enumerate (tqdm(dataset)):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        # t1 = time_synchronized()
        # pred = model(img, augment=opt.augment)[0]
        
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        if save_json:
            num_instances = len(pred[0])  
            image_id = os.path.basename(os.path.normpath(path))   
            image_size = im0s.shape[:2]  

            if num_instances == 0 :
                jdict.append({  'image_name': image_id,
                                'image_size': image_size,
                                'num_instances': num_instances,
                                'labels': [],    #nan_const,
                                'bbox_xyxy': [],   #nan_const,
                                'scores': []    #nan_const
                                })
            
            else:                
                box_dict = []
                score_dict = []
                label_dict = []                    

                for si, pred2 in enumerate(pred):                     
                    box = pred2[:, :4].clone()  # xyxy 
                    box = scale_coords(img.shape[2:], pred2[:, :4], image_size).round()
                    box_xyxy = box.cpu().tolist()  

                    for p, xy in zip(pred2.tolist(), box_xyxy):                        
                        try:
                            names[int(p[5])]
                            box_dict.append(xy)
                            score_dict.append(round(p[4], 5))  
                            label_dict.append(int(p[5]))
                        except:
                            num_instances -= 1
                            continue
                       
                jdict.append({'image_name': image_id,
                            'image_size': image_size,
                            'num_instances': num_instances,
                            'labels': label_dict,
                            'bbox_xyxy': box_dict,
                            'scores': score_dict
                            })
        
        
        if save_img:
            for i, det in enumerate(pred):  # detections per image           
                
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                out = save_dir
                save_path = str(Path(out) / Path(p).name)
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                #print('\nsave path ', save_path)

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    if not save_json:
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        try:
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string
                        except:
                            continue

                    
                    # Write results
                    num_instances = len(det)

                    for *xyxy, conf, cls in det:                    

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                                
                        if save_img or view_img:  # Add bbox to image
                            try:
                                label = '%s %.2f' % (names[int(cls)], conf)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            except:
                                continue

 
                if save_img:
                    cv2.imwrite(save_path, im0)
               
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

    print('Done. (%.3fs)' % (time.time() - t0))



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

    parser.add_argument('--save-json', action='store_true')    #save all preds in a JSON
    parser.add_argument('--save-img', action='store_true') 

    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
