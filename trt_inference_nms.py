import torch
import torch.nn as nn
import tensorrt as trt
from collections import namedtuple, OrderedDict
import numpy as np
from utils.datasets import letterbox, create_dataloader
from utils.general import non_max_suppression, coco80_to_coco91_class, clip_coords, xywh2xyxy, box_iou, scale_coords
from utils.torch_utils import time_synchronized
from utils.metrics import ap_per_class
from tqdm import tqdm
import argparse
import yaml

class TrtModelNMS(nn.Module):
    def __init__(self, weights='yolor_csp.engine', device=torch.device(0)):
        super().__init__()
        self.device = device
        print(f"Loading {weights} for TensorRT inference...")
        Binding = namedtuple('Binding',('name','dtype','shape','data','ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        
        with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        self.fp16 = False
        
        for index in range(self.model.num_bindings):
            name = self.model.get_binding_name(index)
            dtype = trt.nptype(self.model.get_binding_dtype(index))
            if self.model.binding_is_input(index):
                if dtype == np.float16:
                    self.fp16 = True
            shape = tuple(self.context.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.batch_size = self.bindings['images'].shape[0]
        
        
    def forward(self, im):
        b, ch, h, w = im.shape
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        
        nums = self.bindings['num'].data
        boxes = self.bindings['boxes'].data
        scores = self.bindings['scores'].data
        classes = self.bindings['classes'].data

        if(isinstance(nums, np.ndarray)):
            nums = torch.tensor(nums, device=self.device)
        if(isinstance(boxes, np.ndarray)):
            boxes = torch.tensor(boxes, device=self.device)
        if(isinstance(scores, np.ndarray)):
            scores = torch.tensor(scores, device=self.device)
        if(isinstance(classes, np.ndarray)):
            classes = torch.tensor(classes, device=self.device)
        
        return nums, boxes, scores, classes

def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))

def trt_infer(opt):
    model = TrtModelNMS(opt.weights)
    model.eval()
    im = torch.randn(1,3,opt.imgsz,opt.imgsz).cuda()
    loss = torch.zeros(3).cuda()
    nums, boxes, scores, classes = model(im)

    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    nc = int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).cuda()
    niou = iouv.numel()

    path = data['test']
    dataloader = create_dataloader(path, opt.imgsz, 1, 64, opt, pad=0.5, rect=False)[0]

    seen = 0
        
    names = load_classes(opt.names)

    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.

    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to('cuda:0', non_blocking=True)
        img = img.half() if opt.half else img.float()
        img /= 255.0
        targets = targets.cuda()
        nb, _, height, width = img.shape
        whwh = torch.Tensor([width, height, width, height]).cuda()
        with torch.no_grad():
            t = time_synchronized()
            nums, boxes, scores, classes = model(img)
            t0 += time_synchronized() - t
		
        boxes = boxes.squeeze()
        scores= scores.T
        classes = classes.T
        
        output = [torch.cat((boxes, scores, classes), dim=1)]
        
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:,0].tolist() if nl else []
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            clip_coords(pred, (height, width))
            
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).cuda()
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]
                
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
 
                    if pi.shape[0]:
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)

                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  
                                if len(detected) == nl:  
                                    break

            stats.append((correct.cpu(), pred[:,4].cpu(), pred[:,5].cpu(), tcls))
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False)
        ap50, ap = ap[:, 0], ap.mean(1)  
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  
    else:
        nt = torch.zeros(1)

    pf = '%20s' + '%12.3g' * 6  
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (opt.imgsz, opt.imgsz, 1)
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolor_csp.engine')
    parser.add_argument('--data', type=str, default='data/coco_subset.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--half', default=False, action='store_true')
    parser.add_argument('--single_cls', default=False, action='store_true')
    parser.add_argument('--names', type=str, default='data/coco.names')
    opt = parser.parse_args()
    trt_infer(opt)

