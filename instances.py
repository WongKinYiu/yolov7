"""
YOLOV7 instance segmentation network.
This is a modified version of YOLOv7 mask based on
https://github.com/WongKinYiu/yolov7/blob/mask/tools/instance.ipynb.
"""
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


class YOLOv7Mask():
    def __init__(self,
                 weight_path='weights/yolov7-mask.pt',
                 hyp_cfg="data/hyp.scratch.mask.yaml") -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        with open(hyp_cfg, 'r') as f:
            self.hyp = yaml.load(f, Loader=yaml.FullLoader)
        self.weights_path = weight_path
        self.model = self.get_model()

    def get_model(self):
        weights = torch.load(self.weights_path)
        model = weights['model']
        if torch.cuda.is_available():
            model = model.half().to(self.device).eval()
        else:
            _ = model.float().eval().to(self.device).eval()
        return model

    def img_to_tensor(self, img_path):
        image = cv2.imread(img_path)
        image = letterbox(image, 640, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            image = image.half().to(device)
        return image

    def tensor_to_img(self, tensor):
        nimg = tensor[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        return nimg

    def vis_output(self,
                   img_tensor,
                   pred_masks_np,
                   nbboxes,
                   pred_cls,
                   pred_conf,
                   names):
        pnimg = self.tensor_to_img(img_tensor)
        for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
            if conf < 0.25:
                continue
            color = [np.random.randint(255), np.random.randint(
                255), np.random.randint(255)]

            pnimg[one_mask] = pnimg[one_mask] * 0.5 + \
                np.array(color, dtype=np.uint8) * 0.5
            pnimg = cv2.rectangle(
                pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label = '%s %.3f' % (names[int(cls)], conf)
            t_size = cv2.getTextSize(
                label, 0, fontScale=0.5, thickness=1)[0]
            c2 = bbox[0] + t_size[0], bbox[1] - t_size[1] - 3
            pnimg = cv2.rectangle(
                pnimg, (bbox[0], bbox[1]), c2, color, -1, cv2.LINE_AA)  # filled
            pnimg = cv2.putText(pnimg, label, (bbox[0], bbox[1] - 2), 0, 0.5, [
                                255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        return pnimg

    def get_mask(self,
                 img_path,
                 visualize=False,
                 conf_thresh=0.25):
        model = self.model

        image = self.img_to_tensor(img_path)

        output = model(image)
        inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output[
            'bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']

        bases = torch.cat([bases, sem_output], dim=1)
        nb, _, height, width = image.shape
        names = model.names
        pooler_scale = model.pooler_scale
        pooler = ROIPooler(output_size=self.hyp['mask_resolution'], scales=(
            pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)
        output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(
            inf_out, attn, bases, pooler, self.hyp, conf_thres=conf_thresh, iou_thres=0.65, merge=False, mask_iou=None)

        pred, pred_masks = output[0], output_mask[0]
        try:
            bboxes = Boxes(pred[:, :4])
            original_pred_masks = pred_masks.view(-1,
                                                  self.hyp['mask_resolution'], self.hyp['mask_resolution'])
            pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
                original_pred_masks, bboxes, (height, width), threshold=0.5)
            pred_masks_np = pred_masks.detach().cpu().numpy()
            pred_cls = pred[:, 5].detach().cpu().numpy()
            pred_conf = pred[:, 4].detach().cpu().numpy()

            nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)

            if visualize:
                pnimg = self.vis_output(
                    image, pred_masks_np, nbboxes, pred_cls, pred_conf, names)
                pnimg = cv2.cvtColor(pnimg, cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_path.replace('.', '_vis.'), pnimg)

            return pred_masks_np, pred_cls, pred_conf, nbboxes
        except Exception as e:
            print('No mask found')
            return None, None, None, None


if __name__ == '__main__':
    yolov7 = YOLOv7Mask()
    masks, cls, conf, bboxs = yolov7.get_mask(
        "inference/images/horses.jpg", visualize=True)
    print(masks, cls, conf, bboxs)
