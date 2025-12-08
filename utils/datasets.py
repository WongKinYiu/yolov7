# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
import warnings
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.stats import truncnorm
# from torchvision.transforms.functional import adjust_gamma
from skimage.exposure import adjust_gamma
import albumentations as A
import pickle
from copy import deepcopy
#from pycocotools import mask as maskUtils
from torchvision.utils import save_image
from torchvision.ops import roi_pool, roi_align, ps_roi_pool, ps_roi_align

from utils.general import check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, \
    resample_segments, clean_str, check_file
from utils.torch_utils import torch_distributed_zero_first
# @@HK :  pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 resolve h\lib\fbgemm.dll" or one of its dependencies on Windows
# Parameters
eps = 1e-5
check_consistency_labels_meta = True
import pandas as pd
def flatten(lst): return [x for l in lst for x in l]


remove_hard_examples = False  # Bad practice Caveats when removing images bbox-less hence no hard negative examples in it


def clip_boxes_to_border(boxes_array, border):
    """
    Clips bounding boxes to stay within the given border.

    Args:
        boxes_array (np.ndarray): A NumPy array of shape (N, 5),
                                  where N is the number of boxes.
                                  Format: [class_id, x_min, y_min, x_max, y_max].
        border (tuple): Border box as (x_min, y_min, x_max, y_max).

    Returns:
        np.ndarray: Clipped bounding boxes with the same shape (N, 5).
    """
    if boxes_array.shape[1] != 5:
        raise ValueError("The array must have shape (N, 5) with [class_id, x_min, y_min, x_max, y_max]")

    x_min_border, y_min_border, x_max_border, y_max_border = border

    # Copy the original array to avoid modifying input
    clipped_boxes = np.copy(boxes_array)

    # Clip coordinates while keeping class_id unchanged
    clipped_boxes[:, 1] = np.clip(clipped_boxes[:, 1], x_min_border, x_max_border)  # x_min
    clipped_boxes[:, 2] = np.clip(clipped_boxes[:, 2], y_min_border, y_max_border)  # y_min
    clipped_boxes[:, 3] = np.clip(clipped_boxes[:, 3], x_min_border, x_max_border)  # x_max
    clipped_boxes[:, 4] = np.clip(clipped_boxes[:, 4], y_min_border, y_max_border)  # y_max

    return clipped_boxes

help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

def load_csv_xls_2_df(eileen_annot, index_col=False):
    filename, file_extension = os.path.splitext(eileen_annot)
    if file_extension == '.csv':
        df_eilen = pd.read_csv(eileen_annot, index_col=index_col)
    elif file_extension == '.xlsx':
        df_eilen = pd.read_excel(eileen_annot, index_col=index_col, engine='openpyxl')

    return df_eilen

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

# import warnings
# warnings.filterwarnings('error', category=RuntimeWarning)
def scaling_image(img, scaling_type, percentile:float =0.03,
                  beta:float =0.3, roi :tuple=(), img_size: int=640):
    if scaling_type == 'no_norm':
        if bool(roi):
            raise
        img = img

    elif scaling_type == 'standardization': # default by repo
        if bool(roi):
            raise
        img = img/ 255.0

    elif scaling_type =="single_image_0_to_1":
        if bool(roi):
            raise
        max_val = np.max(img.ravel())
        min_val = np.min(img.ravel())
        img = np.double(img - min_val) / (np.double(max_val - min_val)  + eps)
        img = np.minimum(np.maximum(img, 0), 1)

    elif scaling_type == 'single_image_mean_std':
        if bool(roi):
            raise
        img = (img - img.ravel().mean()) / img.ravel().std()

    elif scaling_type == 'single_image_percentile_0_1':
        if bool(roi):
            dw, dh = img_size[1] - roi[1], img_size[0] - roi[0]  # wh padding
            dw /= 2  # divide padding into 2 sides
            dh /= 2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

            if len(img.shape) == 2:
                img_crop = img[bottom:-top, :]
            else:
                img_crop = img[:, bottom:-top, :]

            min_val = np.percentile(img_crop.ravel(), percentile)
            max_val = np.percentile(img_crop.ravel(), 100-percentile)
        else:
            min_val = np.percentile(img.ravel(), percentile)
            max_val = np.percentile(img.ravel(), 100-percentile)
        img = np.double(img - min_val) / (np.double(max_val - min_val) + eps)
        img = np.minimum(np.maximum(img, 0), 1)

    elif scaling_type == 'single_image_percentile_0_255':
        # min_val = np.percentile(img.ravel(), percentile)
        # max_val = np.percentile(img.ravel(), 100 - percentile)
        # img = np.double(img - min_val) / np.double(max_val - min_val)
        # img = np.uint8(np.minimum(np.maximum(img, 0), 1)*255)
        if bool(roi):
            raise
        ImgMin = np.percentile(img, percentile)
        ImgMax = np.percentile(img, 100-percentile)
        ImgDRC = (np.double(img - ImgMin) / (np.double(ImgMax - ImgMin)) * 255 + eps)
        img_temp = (np.uint8(np.minimum(np.maximum(ImgDRC, 0), 255)))
        # img_temp = img_temp / 255.0
        return img_temp


    elif scaling_type == 'remove+global_outlier_0_1':
        if bool(roi):
            raise
        img = np.double(img - img.min()*(beta))/np.double(img.max()*(1-beta) - img.min()*(beta))  # beta in [percentile]
        img = np.double(np.minimum(np.maximum(img, 0), 1))
    elif scaling_type == 'normalization_uint16':
        raise ValueError("normalization norm image method was not imp yet.")
    elif scaling_type == 'normalization':
        raise ValueError("normalization norm image method was not imp yet.")
    else:
        raise ValueError("Unknown norm image method")

    return img


def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='',rel_path_images='', num_cls=-1):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    if augment:
        hyp['gamma_liklihood'] = opt.gamma_aug_prob
        print("", 100 * '==')
        print('gamma_liklihood was overriden by optional value ', opt.gamma_aug_prob)

    with torch_distributed_zero_first(rank):
        scaling_before_mosaic = bool(hyp.get('scaling_before_mosaic', False))

        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      rel_path_images=rel_path_images,
                                      scaling_type=opt.norm_type,
                                      input_channels=opt.input_channels,
                                      num_cls=num_cls,
                                      tir_channel_expansion=opt.tir_channel_expansion,
                                      no_tir_signal=opt.no_tir_signal,
                                      scaling_before_mosaic=scaling_before_mosaic,
                                      csv_metadata_path=opt.csv_metadata_path)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages(Dataset):  # for inference
    def __init__(self, path, img_size=640, stride=32,
                 scaling_type='standardization', img_percentile_removal=0.3, beta=0.3, input_channels=3,
                 tir_channel_expansion=False, no_tir_signal=False,
                 rel_path_for_list_files='', detection_ordering=True, resize_or_crop='resize'):

        self.resize_or_crop = resize_or_crop
        self.detection_ordering = detection_ordering
        self.count = 0
        p = str(Path(path).absolute())  # os-agnostic absolute path
        print('p',p)
        print('rel_path_for_list_files', rel_path_for_list_files)
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            if path.endswith('.txt'):
                files = self.parse_image_file_names(path, rel_path_for_list_files)

            else:
                files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

        self.scaling_type = scaling_type
        self.percentile = img_percentile_removal
        self.beta = beta
        self.input_channels = input_channels
        self.tir_channel_expansion = tir_channel_expansion
        self.is_tir_signal = not (no_tir_signal)

    def parse_image_file_names(self, path, rel_path_for_list_files):
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        if bool(rel_path_for_list_files):
                            f += [os.path.join(rel_path_for_list_files, x.replace('./', '')).rstrip() if x.startswith(
                                './') else x for x
                                  in t]  # local to global path
                        else:
                            f += [x.replace('./', parent).rstrip() if x.startswith('./') else x for x in
                                  t]  # local to global path

                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{p} does not exist')
            self.img_files = sorted(
                [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f' No images found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}\nSee {help_url}')
        return f

    def __iter__(self):
        self.count = 0
        return self
    @staticmethod
    def unlabeled_collate_fn(batch):
        path, img, im0s, vid_cap = zip(*batch)  # transposed
        return path[0], np.array(img[0]), np.array(im0s[0]), np.array([p for p in vid_cap])

    @staticmethod
    def unlabeled_tleap_collate_fn(batch):
        path, img, im0s, vid_cap = zip(*batch)  # transposed
        return path[0], torch.tensor(img[0]), torch.tensor(im0s[0]), np.array([p for p in vid_cap])

    def __getitem__(self, index_dummy):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            # img0 = cv2.imread(path)  # BGR
            # 16bit unsigned
            if os.path.basename(path).split('.')[-1] == 'tiff':
                img0 = cv2.imread(path, -1)
            else:
                img0 = cv2.imread(path)  # BGR

            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # resize_or_crop = 'resize'
        # 'center_crop'
        if self.resize_or_crop == 'resize':
            # Padded resize
            img = letterbox(img0, self.img_size, stride=self.stride)[0]
        elif self.resize_or_crop == 'center_crop':
            img = img0[((img0.shape[0] - self.img_size) // 2): (img0.shape[0] - self.img_size) // 2 + self.img_size, (
                            img0.shape[1] - self.img_size) // 2 : self.img_size + (img0.shape[1] - self.img_size) // 2]
        else:
            raise ValueError(self.resize_or_crop)

        if self.tir_channel_expansion:  # HK @@ according to the paper this CE is a sort of augmentation hence no need to preliminary augment. One of the channels are inversion hence avoid channel inversion aug
            img = np.repeat(img[np.newaxis, :, :], 3, axis=0)  # convert GL to RGB by replication
            img_ce = np.zeros_like(img).astype('float64')

            # CH1 hist equalization
            img_chan = scaling_image(img[0, :, :], scaling_type=self.scaling_type,
                                     percentile=0, beta=self.beta)
            img_ce[0, :, :] = img_chan.astype('float64')

            img_chan = scaling_image(img[1, :, :], scaling_type=self.scaling_type,
                                     percentile=self.percentile, beta=self.beta)

            img_ce[1, :, :] = img_chan.astype('float64')

            img_chan = inversion_aug(img_ce[1, :, :])  # invert the DRC one
            img_ce[2, :, :] = img_chan.astype('float64')
            img = img_ce

        if not self.tir_channel_expansion:
            if self.is_tir_signal:
                img = np.repeat(img[np.newaxis, :, :], self.input_channels, axis=0) #convert GL to RGB by replication
            else:
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

        # print('\n image file', self.img_files[index])
        if 0:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(img.ravel(), bins=128)
            plt.savefig(os.path.join('/home/hanoch/projects/tir_od/outputs', os.path.basename(path).split('.')[0]+ 'pre'))

        file_type = os.path.basename(path).split('.')[-1].lower()

        if (file_type !='tiff' and file_type != 'png'):
            print('!!!!!!!!!!!!!!!!  index : {}  {} unrecognized '.format(index, self.img_files[index]))

        if file_type != 'png':
            img = scaling_image(img, scaling_type=self.scaling_type,
                                percentile=self.percentile, beta=self.beta)
        else:
            img = scaling_image(img,
                                scaling_type='single_image_0_to_1')  # safer in case double standartiozation one before mosaic and her the last one since mosaic is random based occurance

        if 0:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(img.ravel(), bins=128)
            plt.savefig(os.path.join('/home/hanoch/projects/tir_od/outputs', os.path.basename(path).split('.')[0]+ 'post'))

        img = np.ascontiguousarray(img)

        if self.detection_ordering: # unlabeled default ordering
            return path, img, img0, self.cap
        else: # assimilate the same ordering as dataloader of labeled
            return img, 0, path, 0

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            url = eval(s) if s.isnumeric() else s
            if 'youtube.com/' in str(url) or 'youtu.be/' in str(url):  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                url = pafy.new(url).getbest(preftype="mp4").url
            cap = cv2.VideoCapture(url)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', rel_path_images='',
                 scaling_type='standardization', input_channels=3,
                 num_cls=-1, tir_channel_expansion=False, no_tir_signal=False, scaling_before_mosaic=False,
                 csv_metadata_path=''):

        self.scaling_before_mosaic = scaling_before_mosaic
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.scaling_type = scaling_type
        self.percentile = hyp['img_percentile_removal']
        self.beta = hyp['beta']
        self.input_channels = input_channels# in case GL image but NN is RGB hence replicate
        self.tir_channel_expansion = tir_channel_expansion
        self.is_tir_signal = not (no_tir_signal)
        self.random_pad = hyp['random_pad']
        self.batch_size = batch_size

        self.use_csv_meta_data_file = False
        if bool(csv_metadata_path):
            self.csv_meta_data_file = check_file(csv_metadata_path)
            self.use_csv_meta_data_file = True

        if self.hyp['copy_paste'] >0 and self.random_pad:
            raise ValueError('copy_paste and random_pad are mutually exclusive. not supported yet!!')

        #self.albumentations = Albumentations() if augment else None
        self.albumentations_gamma_contrast = Albumentations_gamma_contrast(alb_prob=hyp['gamma_liklihood'],
                                                                           gamma_limit=[hyp['gamma'],
                                                                                        100 + 100-hyp['gamma']])

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        if bool(rel_path_images):
                            f += [os.path.join(rel_path_images, x.replace('./', '')).rstrip() if x.startswith('./') else x for x in t]  # local to global path
                        else:
                            f += [x.replace('./', parent).rstrip() if x.startswith('./') else x for x in t]  # local to global path

                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache HK : cache is only for labels /annotations
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            #if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
            #    cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(num_cls, cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        shapes = list(shapes)
        # #@@HK TODO adding truncation ratio increase here
        # if labels.shape[1] > 5:
        #   labels = labels[:,:5]
        #   self.truncation_ratio = labels[:,5]
        self.labels = list(labels)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        # [l for ix,l in enumerate(self.labels) if (l[:, 1:] >1).all()]
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        if remove_hard_examples:
            if nm > 0: # Caveats when removing images bbox-less hence no hard negative examples in it
                print(100*'/*/')
                print('Remove missing annotations file avoiding unlabeled images that would considered as BG. Before', len(self.labels))
            for ix  in range(len(self.labels) - 1, -1, -1): # safe remove by reverrse iteration #enumerate(self.labels):
                if (self.labels[ix][:, 1:] > 1).any() or self.labels[ix].size < 5:
                    print(self.label_files[ix])
                    del self.labels[ix]
                    del self.img_files[ix]
                    del self.label_files[ix]
                    del shapes[ix]

            print('after',               len(self.labels))

        self.shapes = np.array(shapes, dtype=np.float64)
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)
        self.mosiac_no = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride  #pad=0.5 https://github.com/ultralytics/ultralytics/issues/13271 : @123456dad the padding of 0.5 in the BaseDataset class, which results in resizing an image from 640x640 to 672x672, is primarily for maintaining aspect ratio and providing a buffer to apply various augmentations without losing important features at the edges. This padding can affect model performance, as seen in your observation where the .pt model shows a slightly higher mAP compared to the ONNX model.

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

        if self.use_csv_meta_data_file:
            labels_file_names = [x.split('/')[-1] for x in self.label_files]

            df = load_csv_xls_2_df(self.csv_meta_data_file)
            self.df_metadata = pd.DataFrame(columns=['sensor_type', 'part_in_day', 'weather_condition', 'country', 'train_state', 'tir_frame_image_file_name'])
            # TODO :HK @@ itereate         tqdm(zip(self.img_files, self.label_files) and upon --force-csv-list remove missing entries from the csv in train/test lists!!!
            for ix, fname in enumerate(self.img_files):
                file_name = fname.split('/')[-1]
                if not (df['tir_frame_image_file_name'] == file_name).any():
                        print('File name {} metadata hasnt found !!!'. format(file_name))
                try:
                    self.df_metadata.loc[len(self.df_metadata)] = [df[df['tir_frame_image_file_name'] == file_name]['sensor_type'].item(),
                                                         df[df['tir_frame_image_file_name'] == file_name]['part_in_day'].item(),
                                                         df[df['tir_frame_image_file_name'] == file_name]['weather_condition'].item(),
                                                         df[df['tir_frame_image_file_name'] == file_name]['country'].item(),
                                                         df[df['tir_frame_image_file_name'] == file_name]['train_state'].item(),
                                                         df[df['tir_frame_image_file_name'] == file_name]['tir_frame_image_file_name'].item()]
                    if check_consistency_labels_meta:
                        n_objects_in_img = df[df['tir_frame_image_file_name'] == file_name].Person.item() + \
                                           df[df['tir_frame_image_file_name'] == file_name].Car.item() + \
                                           df[df['tir_frame_image_file_name'] == file_name].Worker.item()+\
                                           df[df['tir_frame_image_file_name'] == file_name].Locomotive.item() +\
                                           df[df['tir_frame_image_file_name'] == file_name].Train.item()

                        entry_in_labels_list = labels_file_names.index(
                            df[df['tir_frame_image_file_name'] == file_name]['tir_frame_label_file_name'].item())
                        n_labesl_in_label_file = len(self.labels[entry_in_labels_list])

                        if n_labesl_in_label_file != n_objects_in_img:
                            print('Discrepancy between num objects in metadata xlsx and annotation file', file_name, n_labesl_in_label_file, n_objects_in_img)
                            print(self.labels[entry_in_labels_list])
                            print(df[df['tir_frame_image_file_name'] == file_name].Person.item(),'Person')
                            print(df[df['tir_frame_image_file_name'] == file_name].Car.item(), 'Car')
                            print(df[df['tir_frame_image_file_name'] == file_name].Worker.item(),'Worker')
                            print(df[df['tir_frame_image_file_name'] == file_name].Locomotive.item(), 'Locomotive')
                            print(df[df['tir_frame_image_file_name'] == file_name].Train.item(), 'Train')

                except Exception as e:
                    print(f'{fname} fname WARNING: Ignoring corrupted image and/or label {file_name}: {e}')


    def resample_ohem(self, top_k_indices):
        if len(top_k_indices.shape) == 2:
            top_k_indices = top_k_indices.reshape(-1)

        self.dataset_was_resampled = True
        self.labels =  [self.labels[i] for i in top_k_indices]
        self.img_files =  [self.img_files[i] for i in top_k_indices]
        self.label_files = [self.label_files[i] for i in top_k_indices]
       # del shapes[ix]
        self.imgs = [self.imgs[i] for i in top_k_indices]
        self.n = len(self.imgs)
        self.shapes = [self.shapes[i] for i in top_k_indices]

        # n = len(shapes)  # number of images
        bi = np.floor(np.arange(self.n) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image

        self.indices = range(self.n)
        self.mosiac_no = 0


    def cache_labels(self, num_cls, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):  # is segment
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                        # if (l[:, 0].max() >= num_cls):
                        #     print('ka', i, l, lb_file, im_file)
                        l = np.array([lbl for lbl in l if lbl[0] < num_cls]) # take only labels index upto num of classes and omit others

                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'  #@@HK TODO adding truncation ratio increase here  : assert l.shape[1] == 6,
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                        assert (l[:, 0].max() < num_cls), 'class label out of range -- invalid' # max label can't be greater than num of labels
                        # print(l[:, 0])


                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')


        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights HK@@ since the fi

        file_type = os.path.basename(self.img_files[index]).split('.')[-1].lower()

        if (file_type !='tiff' and file_type != 'png'):
            print('!!!!!!!!!!!!!!!!  index : {}  {} unrecognized '.format(index, self.img_files[index]))

        if self.is_tir_signal:
            if self.scaling_before_mosaic:
                filling_value = 0.5  # on borders or after perspective  fill with 0.5 in [0 1] equals to 114 in [0 255]
            else:
                filling_value = 0 # on borders or after perspective better to have 0 thermal profile  uint16 based on the DR of the image which is unknown TODO: better find an elegent way
        else:
            filling_value = 114

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic'] and not(self.tir_channel_expansion)
        if mosaic:
            # Load mosaic
            if random.random() < 0.8:
                img, labels = load_mosaic(self, index, filling_value=filling_value, file_type=file_type)
            else:
                img, labels = load_mosaic9(self, index, filling_value=filling_value, file_type=file_type)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']: # since mixup is nested in mosaic first its actually override it
                if random.random() < 0.8:
                    img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1), filling_value=filling_value, file_type=file_type)
                else:
                    img2, labels2 = load_mosaic9(self, random.randint(0, len(self.labels) - 1), filling_value=filling_value, file_type=file_type)
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(img.dtype)#.astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)


        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            # img, ratio, pad = letterbox(img, shape, color=(img.mean(), img.mean(), img.mean()), auto=False, scaleup=self.augment)
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, random_pad=self.random_pad)

            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])


            if self.tir_channel_expansion:  # HK @@ according to the paper this CE is a sort of augmentation hence no need to preliminary augment. One of the channels are inversion hence avoid channel inversion aug
                img = np.repeat(img[np.newaxis, :, :], 3, axis=0)  # convert GL to RGB by replication
                img_ce = np.zeros_like(img).astype('float64')

                # CH1 hist equalization
                img_chan = scaling_image(img[0, :, :], scaling_type=self.scaling_type,
                                         percentile=0, beta=self.beta)
                img_ce[0, :, :] = img_chan.astype('float64')

                img_chan = scaling_image(img[1, :, :], scaling_type=self.scaling_type,
                                         percentile=self.percentile, beta=self.beta)

                img_ce[1, :, :] = img_chan.astype('float64')

                img_chan = inversion_aug(img_ce[1, :, :])  # invert the DRC one
                img_ce[2, :, :] = img_chan.astype('float64')
                img = img_ce

        if self.augment:
            # Augment imagespace
            if not mosaic:
                if hyp['random_perspective']:
                    img, labels = random_perspective(img, labels,
                                                     degrees=hyp['degrees'],
                                                     translate=hyp['translate'],
                                                     scale=hyp['scale'],
                                                     shear=hyp['shear'],
                                                     perspective=hyp['perspective'],
                                                     filling_value=filling_value,
                                                     is_fill_by_mean_img=self.is_tir_signal,
                                                     random_pad=self.random_pad)
                    if np.isnan(img).any():
                        print('img is nan no mosaic after rand perspective')

            if random.random() < hyp['inversion']:
                img = inversion_aug(img)

            if np.isnan(img).any():
                print('img is nan gamma')
            # print("std===",img.std(), img.mean())
            # GL gain/attenuation
            # Squeeze pdf (x-mu)*scl+mu
            #img, labels = self.albumentations(img, labels)
            img = self.albumentations_gamma_contrast(img) # apply RandomBrightnessContrast only since it has buggy response

            if random.random() < hyp['gamma_liklihood']:
                if img.dtype == np.uint16 or img.dtype == np.uint8:
                    img = img/np.iinfo(img.dtype).max
                if (img.max()> 1.0).any():
                    img[img > 1.0] = 1.0
                if (img < 0).any():
                    img[img < 0] = 0

                gamma = np.random.uniform(hyp['gamma'], 200-hyp['gamma']) / 100.0
                img = adjust_gamma(img, gamma, gain=1)

            if np.isnan(img).any():
                print('img is nan gamma')

            if hyp['hsv_h'] > 0 or hyp['hsv_s'] > 0 or hyp['hsv_v'] > 0:
            # Augment colorspace
                augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)
            
            if random.random() < hyp['paste_in']:
                sample_labels, sample_images, sample_masks = [], [], [] 
                while len(sample_labels) < 30: # upto 30 tries to have mosaic of 4 images (anchor + 3 X random)
                    sample_labels_, sample_images_, sample_masks_ = load_samples(self, random.randint(0, len(self.labels) - 1), file_type=file_type)
                    sample_labels += sample_labels_
                    sample_images += sample_images_
                    sample_masks += sample_masks_
                    #print(len(sample_labels))
                    if len(sample_labels) == 0:
                        break
                labels = pastein(img, labels, sample_labels, sample_images, sample_masks)
                # try:
                #
                #     tag='paste_in'
                #     import tifffile
                #     if len(img.shape) == 2:
                #         tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output', 'img_loaded__' + tag +'__' +str(self.img_files[index].split('/')[-1].split('.tiff')[0]) + '.tiff'),
                #                          img[:,:,np.newaxis])
                #     else:
                #         tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output', 'img_loaded__' + tag +'__' +str(self.img_files[index].split('/')[-1].split('.tiff')[0]) + '.tiff'),
                #                          img)
                # except Exception as e:
                #     print(e)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        #     tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od', 'img_ce.tiff'), 255*img.transpose(1,2,0).astype('uint8'))
        if not self.tir_channel_expansion:
            if self.is_tir_signal:
                if len(img.shape) == 2:
                    img = np.repeat(img[np.newaxis, :, :], self.input_channels, axis=0) #convert GL to 3-ch if any RGB by replication
                    # print('Warning , TIR image should be 3dim by now (w,h,1)', 100*'*')
                else:
                    img = np.repeat(img.transpose(2, 0, 1), self.input_channels, axis=0)
            else:
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

            if 0:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.hist(img.ravel(), bins=128)
                plt.savefig(os.path.join('/home/hanoch/projects/tir_od/output', os.path.basename(self.img_files[index]).split('.')[0]+ 'pre_' +str(self.scaling_type)))

            # import tifffile
            # tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output', 'img_loaded_before_scaling_' + '_' +str(str(img.max())) + '_' +str(self.img_files[index].split('/')[-1].split('.tiff')[0]) + '.tiff'),
            #                  (img.transpose(1, 2, 0)))


            # In case moasaic of mixed PNG and TIFF the TIFF is pre scaled while the PNG shouldn;t
            if file_type != 'png':

                # img_size, roi = self.rectangle_res_roi(index)  # HK tried to normalize the image according to the real roi inside the square
                # img = scaling_image(img, scaling_type=self.scaling_type,
                #                     percentile=self.percentile, beta=self.beta,
                #                     roi=roi, img_size=img_size)

                img = scaling_image(img, scaling_type=self.scaling_type,
                                    percentile=self.percentile, beta=self.beta)
            else:
                img = scaling_image(img, scaling_type='single_image_0_to_1') # safer in case double standartiozation one before mosaic and her the last one since mosaic is random based occurance

                # print('ka')
            if 0:
                import matplotlib.pyplot as plt
                # plt.figure()
                plt.hist(img.ravel(), bins=128)
                plt.savefig(os.path.join('/home/hanoch/projects/tir_od/output', os.path.basename(self.img_files[index]).split('.')[0] + '_hist_post_scaling_'+ str(self.scaling_type)))
                # aa1 = np.repeat(img[1,:,:,:].cpu().permute(1,2,0).numpy(), 3, axis=2).astype('float32')
                # cv2.imwrite('test/exp40/test_batch88_labels__1.jpg', aa1*255)
                # aa1 = np.repeat(img.transpose(1,2,0), 3, axis=2).astype('float32')
        # print('\n 1st', img.shape)
        if np.isnan(img).any():
            print('img {} index : {} is nan fin'.format(self.img_files[index], index))
            # raise
        # try:
        #     tag='full_rect'
        #     import tifffile
        #     tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output', 'img_loaded__' + tag +'__' +str(self.img_files[index].split('/')[-1].split('.tiff')[0]) + '.tiff'),
        #                      (img.transpose(1, 2, 0)*2**16).astype('uint16'))
        # except Exception as e:
        #     print(f'\nfailed reading: due to {str(e)}')

        # #
        img = np.ascontiguousarray(img)
        # print('\n 2nd', img.shape)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    def rectangle_res_roi(self, index):
        img_orig, _, _ = load_image(self, index)
        loaded_img_shape = img_orig.shape[:2]
        new_shape = self.img_size
        if isinstance(self.img_size, int):  # if list then the 2d dim is embedded
            new_shape = (new_shape, new_shape)
        if new_shape != loaded_img_shape:
            roi = loaded_img_shape
            img_size = new_shape
        else:  # don't do nothing normaliza the entire image
            roi = ()
            img_size = loaded_img_shape
        if self.rect:
            raise ValueError('not supported')
        return img_size, roi

    # Labels : When it comes to annotations, YOLOv8 uses relative coordinates rather than absolute pixel values for the
    # bounding box positions. This means that the labels are in the range of 0 to 1 relative to the image width and height.
    # Consequently, these labels will remain consistent regardless of image resizing. Hence, you do not need to change,
    # adjust or resize the annotations or labels when the images are resized during training. The model will handle this
    # process automatically.


    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4

class LoadImagesAddingNoiseAndLabels(LoadImagesAndLabels):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', rel_path_images='',
                 scaling_type='standardization', input_channels=3,
                 num_cls=-1, tir_channel_expansion=False, no_tir_signal=False, scaling_before_mosaic=False,
                 path_noisy_samples=''):

        super(LoadImagesAddingNoiseAndLabels, self).__init__(path, img_size=img_size, batch_size=batch_size, augment=augment, hyp=hyp,
                                                             rect=rect, image_weights=image_weights,
                                                         cache_images=cache_images, single_cls=single_cls, stride=stride, pad=pad, prefix=prefix, rel_path_images=rel_path_images,
                                                         scaling_type=scaling_type, input_channels=input_channels,
                                                         num_cls=-1, tir_channel_expansion=tir_channel_expansion, no_tir_signal=no_tir_signal, scaling_before_mosaic=scaling_before_mosaic)
        self.path_noisy_samples=path_noisy_samples

        self.noise_filenames = [os.path.join(self.path_noisy_samples, x) for x in os.listdir(self.path_noisy_samples)
                     if x.endswith('tiff')]

        self.recorded_noise = False
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        file_type = os.path.basename(self.img_files[index]).split('.')[-1].lower()

        if (file_type != 'tiff' and file_type != 'png'):
            print('!!!!!!!!!!!!!!!!  index : {}  {} unrecognized '.format(index, self.img_files[index]))

        if self.is_tir_signal:
            if self.scaling_before_mosaic:
                filling_value = 0.5  # on borders or after perspective  fill with 0.5 in [0 1] equals to 114 in [0 255]
            else:
                filling_value = 0  # on borders or after perspective better to have 0 thermal profile  uint16 based on the DR of the image which is unknown TODO: better find an elegent way
        else:
            filling_value = 114

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic'] and not (self.tir_channel_expansion)
        if mosaic:
            # Load mosaic
            if random.random() < 0.8:
                img, labels = load_mosaic(self, index, filling_value=filling_value, file_type=file_type)
            else:
                img, labels = load_mosaic9(self, index, filling_value=filling_value, file_type=file_type)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                if random.random() < 0.8:
                    img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1),
                                                filling_value=filling_value, file_type=file_type)
                else:
                    img2, labels2 = load_mosaic9(self, random.randint(0, len(self.labels) - 1),
                                                 filling_value=filling_value, file_type=file_type)
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(img.dtype)  # .astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)


        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            # img, ratio, pad = letterbox(img, shape, color=(img.mean(), img.mean(), img.mean()), auto=False, scaleup=self.augment)
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, random_pad=self.random_pad)

            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.tir_channel_expansion:  # HK @@ according to the paper this CE is a sort of augmentation hence no need to preliminary augment. One of the channels are inversion hence avoid channel inversion aug
                img = np.repeat(img[np.newaxis, :, :], 3, axis=0)  # convert GL to RGB by replication
                img_ce = np.zeros_like(img).astype('float64')

                # CH1 hist equalization
                img_chan = scaling_image(img[0, :, :], scaling_type=self.scaling_type,
                                         percentile=0, beta=self.beta)
                img_ce[0, :, :] = img_chan.astype('float64')

                img_chan = scaling_image(img[1, :, :], scaling_type=self.scaling_type,
                                         percentile=self.percentile, beta=self.beta)

                img_ce[1, :, :] = img_chan.astype('float64')

                img_chan = inversion_aug(img_ce[1, :, :])  # invert the DRC one
                img_ce[2, :, :] = img_chan.astype('float64')
                img = img_ce

        if self.augment:
            # Augment imagespace
            if not mosaic:
                if hyp['random_perspective']:
                    img, labels = random_perspective(img, labels,
                                                     degrees=hyp['degrees'],
                                                     translate=hyp['translate'],
                                                     scale=hyp['scale'],
                                                     shear=hyp['shear'],
                                                     perspective=hyp['perspective'],
                                                     filling_value=filling_value,
                                                     is_fill_by_mean_img=self.is_tir_signal,
                                                     random_pad=self.random_pad)
                    if np.isnan(img).any():
                        print('img is nan no mosaic after rand perspective')

            if random.random() < hyp['inversion']:
                img = inversion_aug(img)

            if np.isnan(img).any():
                print('img is nan gamma')
            # print("std===",img.std(), img.mean())
            # GL gain/attenuation
            # Squeeze pdf (x-mu)*scl+mu
            # img, labels = self.albumentations(img, labels)
            img = self.albumentations_gamma_contrast(
                img)  # apply RandomBrightnessContrast only since it has buggy response

            if random.random() < hyp['gamma_liklihood']:
                if img.dtype == np.uint16 or img.dtype == np.uint8:
                    img = img / np.iinfo(img.dtype).max
                if (img.max() > 1.0).any():
                    img[img > 1.0] = 1.0
                if (img < 0).any():
                    img[img < 0] = 0

                gamma = np.random.uniform(hyp['gamma'], 200 - hyp['gamma']) / 100.0
                img = adjust_gamma(img, gamma, gain=1)

            if np.isnan(img).any():
                print('img is nan gamma')

            if hyp['hsv_h'] > 0 or hyp['hsv_s'] > 0 or hyp['hsv_v'] > 0:
                # Augment colorspace
                augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

            if random.random() < hyp['paste_in']:
                sample_labels, sample_images, sample_masks = [], [], []
                while len(sample_labels) < 30:  # upto 30 tries to have mosaic of 4 images (anchor + 3 X random)
                    sample_labels_, sample_images_, sample_masks_ = load_samples(self, random.randint(0,
                                                                                                      len(self.labels) - 1),
                                                                                 file_type=file_type)
                    sample_labels += sample_labels_
                    sample_images += sample_images_
                    sample_masks += sample_masks_
                    # print(len(sample_labels))
                    if len(sample_labels) == 0:
                        break
                labels = pastein(img, labels, sample_labels, sample_images, sample_masks)
                # try:
                #
                #     tag='paste_in'
                #     import tifffile
                #     if len(img.shape) == 2:
                #         tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output', 'img_loaded__' + tag +'__' +str(self.img_files[index].split('/')[-1].split('.tiff')[0]) + '.tiff'),
                #                          img[:,:,np.newaxis])
                #     else:
                #         tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output', 'img_loaded__' + tag +'__' +str(self.img_files[index].split('/')[-1].split('.tiff')[0]) + '.tiff'),
                #                          img)
                # except Exception as e:
                #     print(e)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        #     tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od', 'img_ce.tiff'), 255*img.transpose(1,2,0).astype('uint8'))
        if not self.tir_channel_expansion:
            if self.is_tir_signal:
                if len(img.shape) == 2:
                    img = np.repeat(img[np.newaxis, :, :], self.input_channels,
                                    axis=0)  # convert GL to 3-ch if any RGB by replication
                    # print('Warning , TIR image should be 3dim by now (w,h,1)', 100*'*')
                else:
                    img = np.repeat(img.transpose(2, 0, 1), self.input_channels, axis=0)
            else:
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

            if 0:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.hist(img.ravel(), bins=128)
                plt.savefig(os.path.join('/home/hanoch/projects/tir_od/output',
                                         os.path.basename(self.img_files[index]).split('.')[0] + 'pre_' + str(
                                             self.scaling_type)))

            # import tifffile
            # tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output', 'img_loaded_before_scaling_' + '_' +str(str(img.max())) + '_' +str(self.img_files[index].split('/')[-1].split('.tiff')[0]) + '.tiff'),
            #                  (img.transpose(1, 2, 0)))

            # In case moasaic of mixed PNG and TIFF the TIFF is pre scaled while the PNG shouldn;t
            if file_type != 'png':

                # img_size, roi = self.rectangle_res_roi(index)  # HK tried to normalize the image according to the real roi inside the square
                # img = scaling_image(img, scaling_type=self.scaling_type,
                #                     percentile=self.percentile, beta=self.beta,
                #                     roi=roi, img_size=img_size)
                if self.recorded_noise:
                    index_noise_same = np.random.randint(0, len(self.noise_filenames ))
                    img_noise_path = self.noise_filenames[index_noise_same]
                    img_noise = cv2.imread(img_noise_path, -1)

                    if len(img.shape) == 3:
                        shape_tup = img.shape[1:]
                    else:
                        shape_tup = img.shape

                    img_noise = letterbox(img_noise, shape_tup, 32)[0]  # reshpae presentation image for debug

                    img_noise = img_noise[np.newaxis, :, :]  # (640,640, 1)
                else:
                    min_val = np.percentile(img.ravel(), 0.5)
                    max_val = np.percentile(img.ravel(), 100 - 0.5)
                    density_per_scanline = int(0.5 + img.shape[1]*4/128)
                    pattern_len = 3
                    img_noise = np.zeros_like(img).astype('uint16')
                    for row in range (img.shape[1]):
                        pattern_location = random.choices(range(img.shape[1] - pattern_len), k=density_per_scanline)
                        pattern_location.sort()
                        noise_amp = np.random.randint(0, max_val- min_val, len(pattern_location))
                        for ix, noise_patt in enumerate(pattern_location):
                            img_noise[0,row,noise_patt:noise_patt+pattern_len] = np.array(pattern_len*[noise_amp[ix]])




                img = img + img_noise

                img = scaling_image(img, scaling_type=self.scaling_type,
                                    percentile=self.percentile, beta=self.beta)
            else:
                img = scaling_image(img,
                                    scaling_type='single_image_0_to_1')  # safer in case double standartiozation one before mosaic and her the last one since mosaic is random based occurance

                # print('ka')
            if 0:
                import matplotlib.pyplot as plt
                # plt.figure()
                plt.hist(img.ravel(), bins=128)
                plt.savefig(os.path.join('/home/hanoch/projects/tir_od/output',
                                         os.path.basename(self.img_files[index]).split('.')[
                                             0] + '_hist_post_scaling_' + str(self.scaling_type)))
                # aa1 = np.repeat(img[1,:,:,:].cpu().permute(1,2,0).numpy(), 3, axis=2).astype('float32')
                # cv2.imwrite('test/exp40/test_batch88_labels__1.jpg', aa1*255)
                # aa1 = np.repeat(img.transpose(1,2,0), 3, axis=2).astype('float32')
        # print('\n 1st', img.shape)
        if np.isnan(img).any():
            print('img {} index : {} is nan fin'.format(self.img_files[index], index))
            # raise
        # try:
        #     tag='full_rect'
        #     import tifffile
        #     tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output', 'img_loaded__' + tag +'__' +str(self.img_files[index].split('/')[-1].split('.tiff')[0]) + '.tiff'),
        #                      (img.transpose(1, 2, 0)*2**16).astype('uint16'))
        # except Exception as e:
        #     print(f'\nfailed reading: due to {str(e)}')

        # #
        img = np.ascontiguousarray(img)
        # print('\n 2nd', img.shape)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        #16bit unsigned
        if os.path.basename(path).split('.')[-1] == 'tiff':
            img = cv2.imread(path, -1)
            img = img[:, :, np.newaxis] # (640,640, 1)
        else:
            img = cv2.imread(path)  # BGR
            if self.is_tir_signal:
                img = img[:,:,0] # channels are duplicated in the source
                img = img[:, :, np.newaxis]
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def inversion_aug(img):
    if img.dtype == np.uint16 or img.dtype == np.uint8:
        img = np.iinfo(img.dtype).max - img
        return img
    elif img.dtype == np.float32 or img.dtype == np.float64:
        img = 1.0 - img
        return img
    else:
        raise ValueError("image type is not supported (int8, UINT16) {}".format(img.dtype))


def hist_equalize(img, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB

def load_mosaic(self, index, filling_value, file_type='tiff'):
    # loads images in a 4-mosaic
    self.mosiac_no += 1
    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices

    # debug = True
    # if debug:
    #     indices[1] = 17106 #TIR2_V60_Jan21_Test51D_ML_RD_IL_26_12_2021_15_50_27_FS_210_XGA_0001_0100_ROTEM_left_roi_210_85

    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)
        if self.scaling_before_mosaic:
            if file_type == 'png':
                img = scaling_image(img, scaling_type='single_image_0_to_1')
            else:
                img = scaling_image(img, scaling_type=self.scaling_type,
                                    percentile=self.percentile, beta=self.beta)

        # import tifffile
        # tifffile.imwrite(os.path.join('/mnt/Data/hanoch/runs/output',
        #                               str(self.mosiac_no) + 'img_mosiac_' + '_crop_no_' + str(i) + '_'\
        #                               +str(self.img_files[index].split('/')[-1].split('.tiff')[0]) +'.tiff'),
        #                                 img)

        # place img in img4
        if i == 0:  # top left
            if self.is_tir_signal:
                img4 = init_image_plane(self, img, s, n_div=2)
            else:
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc # dest image
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h # src image
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels = clip_boxes_to_border(labels, (x1a, y1a, x2a, y2a))
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    #img4, labels4, segments4 = remove_background(img4, labels4, segments4)
    #sample_segments(img4, labels4, segments4, probability=self.hyp['copy_paste'])
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, probability=self.hyp['copy_paste']) # mainly for instance segmentation ??!! #@@HK
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border,
                                       filling_value=filling_value,
                                       is_fill_by_mean_img=self.is_tir_signal)# mosaic has its own random padding hence no need to support inside perspective (scaling)
                                         # border to remove

    # import tifffile
    # tifffile.imwrite(os.path.join('/mnt/Data/hanoch/runs/output',
    #                               str(self.mosiac_no) + '_img_mosaic' + '_' + str(
    #                                   self.img_files[indices[0]].split('/')[-1].split('.tiff')[0]) +'.tiff'),
    #                                 img4)
            # tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output',
            #                               'img_mosiac_' + str(self.mosiac_no) + '_' + str(
            #                                   self.img_files[indices[1]].split('/')[-1].split('.tiff')[0]) + '.tiff'), img)
            # tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output',
            #                               'img_mosiac_' + str(self.mosiac_no) + '_' + str(
            #                                   self.img_files[indices[2]].split('/')[-1].split('.tiff')[0]) + '.tiff'), img)
            # tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output',
            #                               'img_mosiac_' + str(self.mosiac_no) + '_' + str(
            #                                   self.img_files[indices[3]].split('/')[-1].split('.tiff')[0]) + '.tiff'), img)

    return img4, labels4


def load_mosaic9(self, index, filling_value, file_type='tiff'):
    # loads images in a 9-mosaic

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)
        if self.scaling_before_mosaic:
            if file_type == 'png':
                img = scaling_image(img, scaling_type='single_image_0_to_1')
            else:
                img = scaling_image(img, scaling_type=self.scaling_type,
                                    percentile=self.percentile, beta=self.beta)

        # place img in img9
        if i == 0:  # center
            if self.is_tir_signal:
                img9 = init_image_plane(self, img, s, n_div=3)
            else:
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles

            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    #img9, labels9, segments9 = remove_background(img9, labels9, segments9)
    img9, labels9, segments9 = copy_paste(img9, labels9, segments9, probability=self.hyp['copy_paste']) # mainly for instance segmentation ??!! #@@HK

    # Perspective transformation can create holes in thermal better fill w/o reflection

    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border,
                                       filling_value=filling_value,
                                       is_fill_by_mean_img=self.is_tir_signal)

    return img9, labels9


def load_samples(self, index, file_type='tiff'):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        if self.scaling_before_mosaic:
            if file_type == 'png':
                img = scaling_image(img, scaling_type='single_image_0_to_1')
            else:
                img = scaling_image(img, scaling_type=self.scaling_type,
                                    percentile=self.percentile, beta=self.beta)


        # place img in img4
        if i == 0:  # top left
            if self.is_tir_signal:
                img4 = init_image_plane(self, img, s, n_div=2)
            else:
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles  # base image with 4 tiles

            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    #img4, labels4, segments4 = remove_background(img4, labels4, segments4)
    sample_labels, sample_images, sample_masks = sample_segments(img4, labels4, segments4, probability=0.5)

    return sample_labels, sample_images, sample_masks

def init_random_image_plane(img, s, n_div=1):
    if img.dtype == np.uint16:
        std_ = 500
        lower = 0
        upper = 2 ** 16 - 1
        filling_value = img.mean()
        # img4 = np.random.normal(img.mean(), std_, (s * 2, s * 2, img.shape[2])).astype(img.dtype) # Random can goes beyond UINT16 and would be wrapped arround which is also random so OK
    elif img.dtype == np.uint8:
        std_ = 15
        filling_value = 114
        lower = 0
        upper = 255
        # img4 = truncnorm.rvs((lower - filling_value) / std_, (upper - filling_value) / std_, loc=filling_value,
        #               scale=std_, size=(s * 2, s * 2)).astype(img.dtype) # bounded random number
    else:
        std_ = 0.05
        filling_value = 0.5
        lower = 0
        upper = 1
        # img4 = truncnorm.rvs((0 - filling_value) / std_, (1 - filling_value) / std_, loc=filling_value,
        #               scale=std_, size=(s * 2, s * 2)).astype(img.dtype) # bounded random number
        # img4 = np.random.normal(img.mean(), std_, (s * 2, s * 2, img.shape[2]))

    if len(img.shape) == 3:
        siz = s * n_div, s * n_div, img.shape[2]
    else:
        siz = s * n_div, s * n_div
    img4 = truncnorm.rvs((lower - filling_value) / std_, (upper - filling_value) / std_,
                         loc=img.mean(), scale=std_, size=(siz)).astype(img.dtype)  # bounded random number

    return img4

def init_image_plane(self, img, s, n_div=2):
    if self.random_pad:
        img4 = init_random_image_plane(img=img, s=s, n_div=n_div)
    else:
        img4 = np.full((s * n_div, s * n_div, img.shape[2]), img.mean(),
                       dtype=img.dtype)  # base image with 4 tiles fill with 0.5 in [0 1] equals to 114 in [0 255]
    img4 = img4[:s*n_div, :s*n_div]  # in case rectangle shape, AR>1, than crop the padding plane according to the right final shape
    return img4


def copy_paste(img, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        im_new = np.zeros(img.shape, np.uint8)
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        img[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img, labels, segments


def remove_background(img, labels, segments):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    h, w, c = img.shape  # height, width, channels
    im_new = np.zeros(img.shape, np.uint8)
    img_new = np.ones(img.shape, np.uint8) * 114
    raise ValueError('uint8 cast dosnot comply with TIR uint 16')
    for j in range(n):
        cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)
        
        i = result > 0  # pixels to replace
        img_new[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img_new, labels, segments


def sample_segments(img, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    sample_labels = []
    sample_images = []
    sample_masks = []
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = l[1].astype(int).clip(0,w-1), l[2].astype(int).clip(0,h-1), l[3].astype(int).clip(0,w-1), l[4].astype(int).clip(0,h-1) 
            
            #print(box)
            if (box[2] <= box[0]) or (box[3] <= box[1]):
                continue
            
            sample_labels.append(l[0])
            
            mask = np.zeros(img.shape, np.uint8)
            
            cv2.drawContours(mask, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
            sample_masks.append(mask[box[1]:box[3],box[0]:box[2],:])
            
            result = cv2.bitwise_and(src1=img, src2=mask)
            i = result > 0  # pixels to replace
            mask[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
            #print(box)
            sample_images.append(mask[box[1]:box[3],box[0]:box[2],:])

    return sample_labels, sample_images, sample_masks


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, stride=32, random_pad=False):
    # Resize and pad image while meeting stride-multiple constraints i.e. 32
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if random_pad and dh>0: # recatangle image with padding is expected
        img_plane = init_random_image_plane(img, s=max(img.shape), n_div=1)
        # img_plane = img_plane[:img.shape[0], :img.shape[1]]  # in case rectangle shape, AR>1, than crop the padding plane according to the right final shape
        img_plane[bottom:-top, :] = img
        img = img_plane
        # img[:bottom, :] = img_plane[:bottom, :]
        # img[-top:, :] = img_plane[-top:, :]
    else:
        n_ch = img.shape[-1]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        if n_ch == 1 and len(img.shape) == 2: # fixing bug in cv2 where n_ch==1 no explicit consideration
            img = img[..., None]
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0), filling_value=114, is_fill_by_mean_img=False,
                       random_pad=False):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1.1 + scale) #@@HK TODO why not symetric
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if is_fill_by_mean_img:
            filling_value = int(img.mean()+1) # filling value can be only an integer hance when scaling before mosaic signal is [0,1] then in the random perspective the posibilities for filling values are 0 or 1
        n_ch = img.shape[-1]

        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(filling_value, filling_value, filling_value))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(filling_value, filling_value, filling_value))

        if n_ch == 1 and len(img.shape) == 2: # fixing bug in cv2 where n_ch==1 no explicit consideration
            img = img[..., None]

        # import tifffile
        # unique_run_name = str(int(time.time_ns()))
        #
        # tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output', str(unique_run_name) + '_' + 'img_projective_before_pad' + str(s) + '_' +  str(T[0, 2]) + '_'  + str(T[1, 2]) + '.tiff'),
        #                  img[:,:,np.newaxis])

        if random_pad:

            pad_w = int((width - np.round(width * s)) // 2)
            pad_h = int((height - np.round(height * s)) // 2)
            img_plane = init_random_image_plane(img, s=max(img.shape), n_div=1)
            img_plane = img_plane[:img.shape[0], :img.shape[1]] # in case rectangle shape, AR>1, than crop the padding plane according to the right final shape

            if pad_w + int(T[0, 2] - width/2) >0:
                # Left padding
                # img[:, :pad_w + max(0,int(T[0, 2] - width/2))] = img_plane[:, :pad_w + max(0,int(T[0, 2] - width/2))]
                img[:, :pad_w + int(T[0, 2] - width / 2)] = img_plane[:, :pad_w + int(T[0, 2] - width / 2)]
            # padding form left can be done even if trans goes right handed beyonf the resolution over the canvas 1280
            img[:, width - pad_w + int(T[0, 2] - width/2):] = img_plane[:, width - pad_w + int(T[0, 2] - width/2):]

            if pad_h + int(T[1, 2] - height / 2) > 0:
                img[:pad_h + int(T[1, 2] - height/2), :] = img_plane[:pad_h + int(T[1, 2] - height/2), :]

            # img[height-pad_h + max(0,int(T[1, 2] - height/2)):, :] = img_plane[height-pad_h + max(0,int(T[1, 2] - height/2)):, :]
            img[height-pad_h + int(T[1, 2] - height/2):, :] = img_plane[height-pad_h + int(T[1, 2] - height/2):, :]

            # print(pad_w + int(T[0, 2] - width/2), width - pad_w + int(T[0, 2] - width/2), pad_h + int(T[1, 2] - height/2), height-pad_h + int(T[1, 2] - height/2) ,str(T[0, 2]) + '_'+ str(s))

        # import tifffile
        # pad_w = int((width - np.round(width * s)) // 2)
        # pad_h = int((height - np.round(height * s)) // 2)
        # tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/output', 'img_projective_' + str(s) + '_' +  str(T[0, 2]) + '_' + str(pad_w) + '_' + str(T[1, 2]) + '_' + str(pad_h) +'.tiff'),
        #                  img[:,:,np.newaxis])

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def bbox_ioa(box1, box2):
    # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # Intersection over box2 area
    return inter_area / box2_area
    

def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels
    

def pastein(image, labels, sample_labels, sample_images, sample_masks):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6  # image size fraction
    for s in scales:
        if random.random() < 0.2:
            continue
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)   
        
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        if len(labels):
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area     
        else:
            ioa = np.zeros(1)
        
        if (ioa < 0.30).all() and len(sample_labels) and (xmax > xmin+20) and (ymax > ymin+20):  # allow 30% obscuration of existing labels
            sel_ind = random.randint(0, len(sample_labels)-1)
            #print(len(sample_labels))
            #print(sel_ind)
            #print((xmax-xmin, ymax-ymin))
            #print(image[ymin:ymax, xmin:xmax].shape)
            #print([[sample_labels[sel_ind], *box]])
            #print(labels.shape)
            hs, ws, cs = sample_images[sel_ind].shape
            r_scale = min((ymax-ymin)/hs, (xmax-xmin)/ws)
            r_w = int(ws*r_scale)
            r_h = int(hs*r_scale)
            
            if (r_w > 10) and (r_h > 10):
                r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                temp_crop = image[ymin:ymin+r_h, xmin:xmin+r_w]
                m_ind = r_mask > 0
                if m_ind.astype(np.int32).sum() > 60:
                    temp_crop[m_ind] = r_image[m_ind]
                    #print(sample_labels[sel_ind])
                    #print(sample_images[sel_ind].shape)
                    #print(temp_crop.shape)
                    box = np.array([xmin, ymin, xmin+r_w, ymin+r_h], dtype=np.float32)
                    if len(labels):
                        labels = np.concatenate((labels, [[sample_labels[sel_ind], *box]]), 0)
                    else:
                        labels = np.array([[sample_labels[sel_ind], *box]])
                              
                    image[ymin:ymin+r_h, xmin:xmin+r_w] = temp_crop

    return labels


import albumentations as A

class Albumentations_gamma_contrast:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, alb_prob=0.01, gamma_limit=[80, 120]):
        self.transform = None

        self.transform = A.Compose([
            # A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=alb_prob), #Contrast adjustment: x' = clip((x - mean) * (1 + a) + mean) ; x'' = clip(x' * (1 + β))
            ])# A.RandomGamma(gamma_limit=gamma_limit, p=alb_prob)])
            # A.Blur(p=0.01),
            # A.MedianBlur(p=0.01),
            # A.ToGray(p=0.01),
            # A.ImageCompression(quality_lower=75, p=0.01),],
            # bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

            #logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))

    def __call__(self, im, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im)  # transformed
            im = new['image']
        return im

class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        import albumentations as A

        self.transform = A.Compose([
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.01),
            A.RandomGamma(gamma_limit=[80, 120], p=0.01),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),

            A.ToGray(p=0.01),
            A.ImageCompression(quality_lower=75, p=0.01),],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

            #logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../coco'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../coco/'):  # from utils.datasets import *; extract_boxes('../coco128')
    # Convert detection dataset into classification dataset, with one directory per class

    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            # image
            raise # not aligned to TIR 1 channel signal
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../coco', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit('../coco')
    Arguments
        path:           Path to images directory
        weights:        Train, val, test weights (list)
        annotated_only: Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in img_formats], [])  # image files only
    n = len(files)  # number of files
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path / x).unlink() for x in txt if (path / x).exists()]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path / txt[i], 'a') as f:
                f.write(str(img) + '\n')  # add image to txt file
    
    
def load_segmentations(self, index):
    key = '/work/handsomejw66/coco17/' + self.img_files[index]
    #print(key)
    # /work/handsomejw66/coco17/
    return self.segs[key]


from torch.utils.data import DataLoader

def reset_dataloader_batch_size(dataloader, ohem_batch_size, disable_augment=True):
    dataloader_ohem = DataLoader(dataloader.dataset, batch_size=ohem_batch_size,
                                 num_workers=dataloader.num_workers,
                                 sampler=dataloader.sampler,
                                 pin_memory=dataloader.pin_memory,
                                 collate_fn=dataloader.collate_fn)

    if disable_augment:
        dataloader_ohem.dataset.mosaic = False
        dataloader_ohem.dataset.augment = False

    return dataloader_ohem
