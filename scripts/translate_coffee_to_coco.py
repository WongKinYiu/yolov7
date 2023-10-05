import json
import os


with open("/data/PTG/cooking/annotations/coffee/berkeley/2022-11-05_whole/fine-tuning_new_obj_labels.mscoco.json") as f:
    truth = json.load(f)


for im in truth['images']:
    im_id = im['id']
    im_fname = im['file_name']

    # Get all objects for this image
    im_objects = [ann for ann in truth['annotations'] if ann['image_id'] == im_id]

    im_fpath = "/data/standard_datasets/coffee_coco/labels/train/" + os.path.splitext(os.path.basename(im_fname))[0] + ".txt"

    with open(im_fpath, "w") as text_file:
        for obj in im_objects:
            x,y,w,h = obj['bbox']
            x /= 1280.0
            w /= 1280.0
            y /= 720.0
            h /= 720.0
            text_file.write(f"{obj['category_id']} ")
            text_file.write(f"{x} {y} {x+w} {y} {x+w} {y+h} {x} {y+h}\n")


