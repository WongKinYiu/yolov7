# This is a modified version of a script found here:
# https://gist.github.com/wfng92/c77c822dad23b919548049d21d4abbb8

import xml.etree.ElementTree as ET
import glob
import os
import json
from distutils.dir_util import copy_tree

if not os.path.isdir("./data/RDD2022/Norway_split"):
    # split dataset for Norway into training and validation sets
    
    os.makedirs("./data/RDD2022/Norway_split/train/labels")
    os.makedirs("./data/RDD2022/Norway_split/train/images")
    os.makedirs("./data/RDD2022/Norway_split/train/annotations/xmls")
    os.makedirs("./data/RDD2022/Norway_split/val/labels")
    os.makedirs("./data/RDD2022/Norway_split/val/images")
    os.makedirs("./data/RDD2022/Norway_split/val/annotations/xmls")

    from_xmls = "./data/RDD2022/Norway/train/annotations/xmls"
    from_images = "./data/RDD2022/Norway/train/images"
    val_images = "./data/RDD2022/Norway_split/val/images"
    val_xmls = "./data/RDD2022/Norway_split/val/annotations/xmls"
    train_images = "./data/RDD2022/Norway_split/train/images"
    train_xmls = "./data/RDD2022/Norway_split/train/annotations/xmls"

    xml_files = glob.glob(os.path.join(from_xmls, '*.xml'))
    train = True
    for fil in xml_files:
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(from_images, f"{filename}.jpg")):
            print(f"{filename} image does not exist!")
            continue
        if train:
            os.replace(os.path.join(from_xmls, f"{filename}.xml"), os.path.join(train_xmls, f"{filename}.xml"))
            os.replace(os.path.join(from_images, f"{filename}.jpg"), os.path.join(train_images, f"{filename}.jpg"))
        else:
            os.replace(os.path.join(from_xmls, f"{filename}.xml"), os.path.join(val_xmls, f"{filename}.xml"))
            os.replace(os.path.join(from_images, f"{filename}.jpg"), os.path.join(val_images, f"{filename}.jpg"))
        train = not train


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]

classes = ['D00', 'D10', 'D20', 'D40']
input_dirs = [
    "./data/RDD2022/Norway_split/val/annotations/xmls/",
    "./data/RDD2022/Norway_split/train/annotations/xmls/",
]
output_dirs = [
    "./data/RDD2022/Norway_split/val/labels/",
    "./data/RDD2022/Norway_split/train/labels/",
]
image_dirs = [
    "./data/RDD2022/Norway_split/val/images/",
    "./data/RDD2022/Norway_split/train/images/",
]

def check_bbox(bbox, im_width, im_height):
    return bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] <= im_width and bbox[3] <= im_height


for i in range(len(input_dirs)):
    input_dir = input_dirs[i]
    output_dir = output_dirs[i]
    image_dir = image_dirs[i]

    if not os.path.isdir(output_dir):
        # create the labels folder (output directory) 
        os.mkdir(output_dir)

    # identify all the xml files in the annotations folder (input directory)
    files = glob.glob(os.path.join(input_dir, '*.xml'))
    # loop through each 
    for fil in files:
        try:
            basename = os.path.basename(fil)
            filename = os.path.splitext(basename)[0]
            # check if the label contains the corresponding image file
            if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
                print(f"{filename} image does not exist!")
                continue
            
            result = []
    
            # parse the content of the xml file
        
            tree = ET.parse(fil)
            root = tree.getroot()
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)

            for obj in root.findall('object'):
                label = obj.find("name").text
                # check for new classes and append to list
                if label not in classes:
                    continue # skip unknown classes
                index = classes.index(label)
                pil_bbox = [int(float(x.text)) for x in obj.find("bndbox")]
                is_valid = check_bbox(pil_bbox, width, height)
                if not is_valid:
                    continue    # skip boxes that go out of bounds
                yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
                # convert data to string
                bbox_string = " ".join([str(x) for x in yolo_bbox])
                result.append(f"{index} {bbox_string}")

            if result:
                # generate a YOLO format text file for each xml file
                with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(result))
        except:
            print("there was a problem with parsing the file: ", fil)

    # generate the classes file as reference
    with open('classes.txt', 'w', encoding='utf8') as f:
        f.write(json.dumps(classes))