# Imports
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Set seed
random.seed(0)

# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"airplane": 0,
                            "drone": 1}
class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

def plot_bounding_box(image, annotation_list, annotation_file):
    annotations = np.array(annotation_list)
    
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    
    # Rescale
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    # xywh to xyxy
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    
    save_path = "_".join(".".join(annotation_file.split('.')[:-1]).replace("labels", "labeled_images").split('_')[:-1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = annotation_file.split('_')[-1].replace("txt", "png")
    plt.savefig(save_path + "/" + file_name)
    
# Get any random annotation file 
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to data folder')
paths = parser.parse_args()
data_path = paths.data_path + "/"

annotations = os.listdir(data_path + "labels/")
for annotation_file in os.listdir(data_path + "labels/"):
    annotation_file = data_path + "labels/" + annotation_file
    
    with open(annotation_file, "r") as file:
        annotation_list = file.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x ] for x in annotation_list]

    #Get the corresponding image file
    image_file = annotation_file.replace("labels", "images").replace("txt", "jpg")
    assert os.path.exists(image_file)

    #Load the image
    image = Image.open(image_file)
    print(image_file)

    #Plot the Bounding Box
    if len(annotation_list) > 0:
        plot_bounding_box(image, annotation_list, annotation_file)