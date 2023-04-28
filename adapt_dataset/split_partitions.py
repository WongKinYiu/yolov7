import os
from pathlib import Path
import random
import shutil

def copy_images(images_list, partition):
    for image in images_list:
        for x in range(3):
            for y in range(2):
                image_with_extension = f'{image}_{x}_{y}.png'
                label_with_extension = f'{Path(image).stem}_{x}_{y}.txt'

                image_path = os.path.join(images_folder, image_with_extension)
                label_path = os.path.join(labels_folder, label_with_extension)

                destination_images = os.path.join(dest_folder, partition, 'images')
                destination_labels = os.path.join(dest_folder, partition, 'labels')

                if not os.stat(label_path).st_size == 0:
                    dest_image_path = os.path.join(destination_images, image_with_extension)
                    dest_label_path = os.path.join(destination_labels, label_with_extension)

                    dest_image_dir = os.path.dirname(dest_image_path)
                    if not os.path.exists(dest_image_dir):
                        os.makedirs(dest_image_dir)

                    dest_label_dir = os.path.dirname(dest_label_path)
                    if not os.path.exists(dest_label_dir):
                        os.makedirs(dest_label_dir)

                    shutil.copyfile(image_path, dest_image_path)
                    shutil.copyfile(label_path, dest_label_path)

train = 0.8
val = 0.1
test = 0.1

images_folder = 'H:\PatoUTN\pap\CROC original\croc_cortado'
labels_folder = 'H:\PatoUTN\pap\CROC original\labels_cortado_1clase'
dest_folder = 'H:\PatoUTN\pap\CROC original\split_1_clase'

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

images = os.listdir(images_folder)

# get parent images
parent_images = {}

for image in images:
    key = image.split('_')[0]
    if key not in parent_images:
        parent_images[key] = key

parent_images = list(parent_images.values())
random.Random(4).shuffle(parent_images)

total = len(parent_images)
train_images_len = int(total * train)
test_images_len = int(total * test)
val_images_len = int(total * val)


train_images = parent_images[:train_images_len]
test_images = parent_images[train_images_len:train_images_len+test_images_len]
val_images = parent_images[train_images_len+test_images_len:]

copy_images(train_images, 'train')
copy_images(test_images, 'test')
copy_images(val_images, 'val')


