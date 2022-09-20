import argparse
import json
import math
import yaml
import pathlib
import io
import threading
import queue

import boto3

from train import train, select_device

s3 = boto3.client("s3")

def load_jsonl(raw_str):
    content = []
    for line in raw_str.split("\n"):
        content.append(json.loads(line))
    return content


def parse_opt():
    parser = argparse.ArgumentParser()
    # Dataset Fetcher
    parser.add_argument("--bucket", help="input manifest/ label bucket name")
    parser.add_argument("--image-bucket", help="input image bucket (if not same as --bucket)")
    parser.add_argument("--key", help="S3 key")
    parser.add_argument(
        "--holdout-size", 
        type=float, 
        default=0.2, 
        help="proportion of the input dataset to hold out for testing and validation (0.0 < n < 1.0)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="MLFlow Experiment Name"
    )

    # Training Args
    parser.add_argument(
        "--weights", type=str, default="yolo7.pt", help="initial weights path"
    )
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument(
        "--hyp",
        type=str,
        default="data/hyp.scratch.p5.yaml",
        help="hyperparameters path",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--batch-size", type=int, default=16, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="[train, test] image sizes",
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume most recent training",
    )
    parser.add_argument(
        "--nosave", action="store_true", help="only save final checkpoint"
    )
    parser.add_argument("--notest", action="store_true", help="only test final epoch")
    parser.add_argument(
        "--noautoanchor", action="store_true", help="disable autoanchor check"
    )
    parser.add_argument("--evolve", action="store_true", help="evolve hyperparameters")
    parser.add_argument(
        "--cache-images", action="store_true", help="cache images for faster training"
    )
    parser.add_argument(
        "--image-weights",
        action="store_true",
        help="use weighted image selection for training",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--multi-scale", action="store_true", help="vary img-size +/- 50%%"
    )
    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="train multi-class data as single-class",
    )
    parser.add_argument(
        "--adam", action="store_true", help="use torch.optim.Adam() optimizer"
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use SyncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="DDP parameter, do not modify"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="maximum number of dataloader workers"
    )
    parser.add_argument("--project", default="runs/train", help="save to project/name")
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--linear-lr", action="store_true", help="linear LR")
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon"
    )
    parser.add_argument(
        "--upload_dataset",
        action="store_true",
        help="Upload dataset as W&B artifact table",
    )
    parser.add_argument(
        "--bbox_interval",
        type=int,
        default=-1,
        help="Set bounding-box image logging interval for W&B",
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=-1,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument(
        "--artifact_alias",
        type=str,
        default="latest",
        help="version of dataset artifact to be used",
    )
    parser.add_argument(
        "--freeze",
        nargs="+",
        type=int,
        default=[0],
        help="Freeze layers: backbone of yolov7=50, first3=0 1 2",
    )
    opt = parser.parse_args()
    if opt.image_bucket is None:
        opt.image_bucket = opt.bucket
    return opt


def main(opt):
    bucket = opt.bucket
    image_bucket = opt.image_bucket
    key = opt.key
    name = opt.name

    with io.BytesIO() as buff:
        s3.download_fileobj(bucket, key, buff)
        buff.seek(0,0)
        manifest_content = buff.read().decode("utf-8")
    manifest = load_jsonl(manifest_content)

    labels_for_image = {}

    idx_to_label = {}
    label_to_idx = {}
    next_idx = 0

    for line in manifest:
        labels = line.get("labels")
        for label in labels:
            labels_for_image[key] = labels
            c = label.get("class")
            if c not in label_to_idx:
                idx_to_label[next_idx] = c
                label_to_idx[c] = next_idx
                next_idx += 1

    num_classes = len(idx_to_label)

    exp_name_path = pathlib.Path(name)

    splits = ("train", "val", "test",)
    split_sizes = {name: 0 for name in splits}
    holdout_size = math.floor(opt.holdout_size * len(manifest))
    split_sizes["val"] = holdout_size // 2
    split_sizes["test"] = holdout_size // 2
    split_sizes["train"] = len(manifest) - split_sizes["val"] - split_sizes["test"]
    split_manifest = {split_name: [] for split_name in splits}

    for split_name in splits:
        while len(split_manifest[split_name]) < split_sizes[split_name]:
            split_manifest[split_name].append(manifest.pop())
        split_path = exp_name_path / split_name

        images_path = split_path / "images"
        images_path.mkdir(parents=True)

        labels_path = split_path / "labels"
        labels_path.mkdir(parents=True)

        yaml_path = pathlib.Path("data") / f"{name}.yaml"

        images_to_download = queue.Queue()
        labels_for_image = {}
        
        def save_img(key):
            this_image_path = images_path / key
            this_image_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(image_bucket, key, str(this_image_path))
            return

        def save_img_worker():
            while True:
                unsaved_img = images_to_download.get()
                save_img(unsaved_img)
                images_to_download.task_done()

        # download all the files
        threading.Thread(target=save_img_worker, daemon=True).start()
        for item in split_manifest[split_name]:
            key = item["s3Key"]
            labels = item["data"]
            images_to_download.put(key)
            labels_for_image[key] = labels
        images_to_download.join()

        # write the annotations
        for image, labels in labels_for_image.items():
            # assume [{ cx, cy, w, h, class, img_width, img_height }, ...]
            out = ""
            for label in labels:
                c = label_to_idx[label["class"]]
                img_width = label.get("img_width")
                img_height = label.get("img_height")
                # Compute YOLO format BBox from our internal annotation format
                x_center_norm = float(label["bbox_cx"]) / float(img_width)
                y_center_norm = float(label["bbox_cy"]) / float(img_height)
                bbox_width_norm = float(label["bbox_width"]) / float(img_width)
                bbhox_height_norm = float(label["bbox_height"]) / float(img_height)
                # And write them out to a file
                out += "{} {} {} {} {}\n".format(
                    c, x_center_norm, y_center_norm, bbox_width_norm, bbhox_height_norm
                )
            label_path = labels_path / pathlib.Path(image).with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.touch()
            with open(label_path, "w") as f:
                f.write(out)

    training_in = {
        "train": str(exp_name_path / "train"),
        "val": str(exp_name_path / "val"),
        "test": str(exp_name_path / "test"),
        "nc": num_classes,
        "names": idx_to_label,
    }

    yaml_out = yaml.dump(training_in)
    with open(yaml_path, "w") as f:
        f.write(yaml_out)

    # Shim in the yaml.data that we just generated on the fly from our Label Manifest from S3
    opt.data = yaml_path

    device = select_device(opt.device, batch_size=opt.batch_size)
    with open(opt.hyp, "r") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader) 

    # Delete duplicate argparse options
    # `Bucket` is used by both train.py (for some Google Cloud Storage we aren't accessing) 
    # and driver.py (for some S3 storage we are), so lets delete it before passing our args to train()
    del opt.bucket
    train(hyp=hyp, opt=opt, device=device, tb_writer=None)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)