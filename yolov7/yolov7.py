import cv2
import numpy as np
import torch
from importlib_resources import files

from yolov7.models.experimental import attempt_load_state_dict
from yolov7.models.yolo import Model
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import scale_coords, non_max_suppression, check_img_size
from yolov7.utils.torch_utils import TracedModel


@torch.no_grad()
class YOLOv7:
    _defaults = {
        'bgr': True,
        'device': 'cuda',
        'conf_thresh': 0.25,
        'nms_thresh': 0.45,
        'model_image_size': 640,
        'max_batch_size': 4,
        'half': True,
        'same_size': True,
        'weights': files('yolov7').joinpath('weights/yolov7_state.pt'),
        'cfg': files('yolov7').joinpath('cfg/deploy/yolov7.yaml'),
        'trace': True,
        'cudnn_benchmark': False,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # update with user overrides

        self.device, self.device_num = self._select_device(self.device)

        model = Model(self.cfg)
        self.model, self.class_names = attempt_load_state_dict(model, self.weights, map_location=torch.device('cpu'))
        self.model.to(self.device)

        self.model_stride = int(self.model.stride.max())  # model stride
        self.model_image_size = check_img_size(self.model_image_size, s=self.model_stride)  # check img_size

        if self.trace:
            self.model = TracedModel(self.model, self.device, self.model_image_size)

        if self.device == torch.device('cpu'):
            self.half = False
        if self.half:
            self.model.half()

        if self.cudnn_benchmark:
            # set True to speed up constant image size inference. Beware: GPU memory hogger
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        # warm up
        self._detect([np.zeros((10, 10, 3), dtype=np.uint8)])
        print('Warmed up!')

    @staticmethod
    def _select_device(device):
        if not device.isnumeric():
            if device.lower() not in ['cpu', 'cuda']:
                raise ValueError(f'Device "{device}" not supported')
            return torch.device(device.lower()), None
        else:
            return torch.device(f'cuda:{device}'), int(device)

    def classname_to_idx(self, classname):
        return self.class_names.index(classname)

    def _detect(self, list_of_imgs):
        if self.bgr:
            list_of_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in list_of_imgs]

        resized = [letterbox(img, new_shape=self.model_image_size, auto=self.same_size, stride=self.model_stride)[0] for img in list_of_imgs]
        images = np.stack(resized, axis=0)
        images = np.divide(images, 255, dtype=np.float32)
        images = np.ascontiguousarray(images.transpose(0, 3, 1, 2))
        input_shapes = [img.shape for img in images]

        batches = []
        for i in range(0, len(images), self.max_batch_size):
            these_imgs = torch.from_numpy(images[i:i+self.max_batch_size])
            if self.half:
                these_imgs = these_imgs.half()
            batches.append(these_imgs)

        if self.device_num is not None
            with torch.cuda.device(self.device_num):
                preds = self._batch_pred(batches)
        else:
            preds = self._batch_pred(batches)

        predictions = torch.cat(preds, dim=0)

        return predictions, input_shapes

    def _batch_pred(self, batches):
        preds = []
        for batch in batches:
            batch = batch.to(self.device)
            features = self.model(batch)[0]
            preds.append(features.detach().cpu())
            del features
        return preds

    def detect_get_box_in(self, images, box_format='ltrb', classes=None, buffer_ratio=0.0):
        '''
        Parameters
        ----------
        images : ndarray or List[ndarray]
            ndarray-like for single image or list of ndarray-like
        box_format : str, optional
            string of characters representing format order, where l = left, t = top, r = right, b = bottom, w = width and h = height
        classes : List[str], optional
            classes to focus on
        buffer_ratio : float, optional
            proportion of buffer around the width and height of the bounding box
        raw : bool, optional
            return raw inferences instead of detections after postprocessing
        
        Returns
        ------
        If one ndarray given, this returns a list (boxes in one image) of tuple (box_infos, score, predicted_class),
        else if a list of ndarray given, this return a list (batch) containing the former as the elements.
        box_infos : List[float]
            according to the given box format
        score : float
            confidence level of prediction
        predicted_class : string
        '''
        single = False
        if isinstance(images, list):
            if len(images) <= 0:
                return None
            else:
                if not all(isinstance(im, np.ndarray) for im in images):
                    raise AssertionError('all images must be np arrays')
        elif isinstance(images, np.ndarray):
            images = [images]
            single = True

        if any(c not in [*'tlbrwh'] for c in box_format):
            raise AssertionError('box_format given is unrecognised!')

        res, input_shapes = self._detect(images)
        frame_shapes = [image.shape for image in images]

        all_dets = self._postprocess(res, input_shapes=input_shapes, frame_shapes=frame_shapes, box_format=box_format, classes=classes, buffer_ratio=buffer_ratio)

        if single:
            return all_dets[0]
        else:
            return all_dets

    def get_detections_dict(self, frames, classes=None, buffer_ratio=0.0):
        '''
        Parameters
        ----------
        frames : List[ndarray]
            list of input images
        classes : List[str], optional
            classes to focus on
        buffer_ratio : float, optional
            proportion of buffer around the width and height of the bounding box

        Returns
        -------
        List[dict]
            list of detections for each frame with keys: label, confidence, t, l, b, r, w, h
        '''

        if frames is None or len(frames) == 0:
            return None
        all_dets = self.detect_get_box_in(frames, box_format='tlbrwh', classes=classes, buffer_ratio=buffer_ratio)
        
        all_detections = []
        for dets in all_dets:
            detections = []
            for tlbrwh, confidence, label in dets:
                top, left, bot, right, width, height = tlbrwh
                detections.append({'label': label, 'confidence': confidence, 't': top, 'l': left, 'b': bot, 'r': right, 'w': width, 'h': height})
            all_detections.append(detections)
        return all_detections

    def _postprocess(self, boxes, input_shapes, frame_shapes, box_format='ltrb', classes=None, buffer_ratio=0.0):
        class_idxs = [self.classname_to_idx(name) for name in classes] if classes is not None else None
        preds = non_max_suppression(boxes, self.conf_thresh, self.nms_thresh, classes=class_idxs)

        detections = []
        for i, frame_bbs in enumerate(preds):
            if frame_bbs is None:
                detections.append([])
                continue

            im_height, im_width, _ = frame_shapes[i]
            
            # Rescale preds from input size to frame size
            frame_bbs[:, :4] = scale_coords(input_shapes[i][1:], frame_bbs[:, :4], frame_shapes[i]).round()

            frame_dets = []
            for *xyxy, cls_conf, cls_id in frame_bbs:
                cls_conf = float(cls_conf)
                cls_name = self.class_names[int(cls_id)]

                left = int(xyxy[0])
                top = int(xyxy[1])
                right = int(xyxy[2])
                bottom = int(xyxy[3])
                
                width = right - left + 1
                height = bottom - top + 1
                width_buffer = width * buffer_ratio
                height_buffer = height * buffer_ratio

                top = max(0.0, top - 0.5*height_buffer)
                left = max(0.0, left - 0.5*width_buffer)
                bottom = min(im_height - 1.0, bottom + 0.5*height_buffer)
                right = min(im_width - 1.0, right + 0.5*width_buffer)

                box_attr = {'t': int(round(top)),
                            'l': int(round(left)),
                            'b': int(round(bottom)),
                            'r': int(round(right)),
                            'w': int(round(width+width_buffer)),
                            'h': int(round(height+height_buffer))}
                box_infos = [box_attr[c] for c in box_format]
                if not len(box_infos) > 0:
                    raise AssertionError('box infos is blank')

                detection = (box_infos, cls_conf, cls_name)
                frame_dets.append(detection)
            detections.append(frame_dets)

        return detections
