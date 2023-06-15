import base64
import io
import kserve
import os

from detect import make_parser, detect
from kserve_wrapper.modelpath import modelpath_join
from model_singleton import ModelSingleton
from PIL import Image
from typing import Dict

class YoloV7Model(kserve.Model):
    """ YoloV7 implements paper - YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors
    see the orignal link: https://github.com/WongKinYiu/yolov7
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.cfg = None
        self.device = self._detect_device()
        self.ms = None

    def _detect_device(self) -> str:
        if "YOLOV7_DEVICE" not in os.environ:
            return "" # empty string for auto detect
        return os.environ["YOLOV7_DEVICE"]

    def load(self):
        weights = modelpath_join("yolov7.pt")
        self.ms = ModelSingleton(weights, self.device)

        self.ready = True
        return self.ready

    def _query_ts(self) -> int:
        import datetime

        now = datetime.datetime.utcnow()
        return int(now.timestamp()*1000.0)

    def _ensure_folders_exists(self, strs: []):
        for astr in strs:
            basedir = os.path.dirname(astr)
            os.makedirs(basedir, exist_ok=True)

    def _load_label_txt(self, path: str) -> []:
        if not os.path.exists(path):
            return None

        import csv

        labels_list = []
        with open(path, 'r') as file:
            csvreader = csv.reader(file, delimiter=' ')
            for row in csvreader:
                if not ''.join(row).strip():
                    continue
                labels_list.append(self._make_label_dict(row))
        return labels_list

    def _make_label_dict(self, csvrow):
        # all string
        return {
            "label_index": csvrow[0],
            "x": csvrow[1],
            "y": csvrow[2],
            "w": csvrow[3],
            "h": csvrow[4],
        }

    def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        inputs = request["instances"]
        # request is wrapped the following format
        # {
        #   "instances": [
        #     {
        #       "image_bytes": {
        #           "b64": "<b64-encoded>",
        #       },
        #       "key": "somekeys",
        #     },
        #   ],
        # }
        # and response is wrapped into the following
        # {
        #  "predictions: [
        #    {
        #      "image_bytes": {
        #          "b64": "<b64-encoded>",
        #      },
        #      "label_and_positions: [
        #        { "label_index": '17', "x": '0.534282', "y": '0.519531', "w":
        #        '0.111255', "h": '0.21875'},
        #      ],
        #      "key": "somekeys",
        #      "type": "yolov7-object-detector",
        #    },
        #  ]
        # }

        ts = self._query_ts()
        ts_str = "{:010d}".format(ts)
        query_file_path = "/tmp/query/{}/input.jpg".format(ts_str)
        detect_label_txt_path = "/tmp/kserve/{}/labels/input.txt".format(ts_str)
        detect_photo_path = "/tmp/kserve/{}/input.jpg".format(ts_str)

        self._ensure_folders_exists([query_file_path, detect_label_txt_path,
            detect_photo_path])

        data = inputs[0]["image_bytes"]["b64"]
        key = inputs[0]["key"]
        raw_img_data = base64.b64decode(data)
        input_image = Image.open(io.BytesIO(raw_img_data))
        input_image.save(query_file_path)


        # generate online argument for this simple query
        runtime_argparse = make_parser().parse_args([
            "--weights",
            "/mnt/models/yolov7.pt",
            "--conf",
            "0.25",
            "--img-size",
            "640",
            "--source",
            query_file_path,
            "--project",
            "/tmp/kserve",
            "--name",
            ts_str,
            "--save-txt",
            "--exist-ok",
         ])
        detect(runtime_argparse, True)

        # convert image to b64
        grid_result_img = Image.open(detect_photo_path)
        rgb_im = grid_result_img.convert('RGB')

        buffered = io.BytesIO()
        rgb_im.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {
                "predictions": [
                {
                    "image_bytes": {
                        "b64": img_str,
                    },
                    "label_and_position": self._load_label_txt(detect_label_txt_path),
                    "key": key,
                    "type": "yolov7-object-detector",
                },
            ]
        }
