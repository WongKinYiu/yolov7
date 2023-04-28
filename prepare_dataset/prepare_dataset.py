import os
from pathlib import Path
import random
import shutil
import json
import cv2

def write_labels(i, j, labels, image, image_partition):
    folder_name = f'{subimages_path}/{image_partition}/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    file_name = f'{folder_name}/{image["image_name"].split(".")[0]}_{i}_{j}.txt'
    annotation_file = open(file_name, 'w')
    annotation_file.writelines(line + '\n' for line in labels)
    annotation_file.close()

def divide_image_into_subimages(img, x, y, s, id_imagen, i, j, image_partition):
    x = int(x)
    y = int(y)
    cropped_img = img[y:y+s, x:x+s]
    
    folder_name = f'{subimages_path}/{image_partition}/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    cv2.imwrite(f'{folder_name}{id_imagen}_{i}_{j}.png', cropped_img)

def correctSize(xx, yy):
  corrected_x = width_in_image
  corrected_y = height_in_image
  if xx + width_in_image > 1:
    corrected_x = width_in_image - ((xx + width_in_image) - 1)
  
  if yy + height_in_image > 1:
    corrected_y = height_in_image - ((yy + height_in_image) - 1)

  return corrected_x, corrected_y

split_original_images = True
extract_cells = True
create_subimages = True

original_images_path = 'H:/PatoUTN/pap/CROC original/base'
original_images_splitted_path = 'H:/PatoUTN/pap/CROC original/prepared_dataset/full_size'
cells_images_path = 'H:/PatoUTN/pap/CROC original/prepared_dataset/cells'
subimages_path = 'H:/PatoUTN/pap/CROC original/prepared_dataset/subimages'

# Split images into test/train/val
train = 0.8
val = 0.1
test = 0.1

if not os.path.exists(original_images_splitted_path):
    os.makedirs(original_images_splitted_path)

images = os.listdir(original_images_path)
random.Random(4).shuffle(images)

total = len(images)

train_images_len = int(total * train)
test_images_len = int(total * test)
val_images_len = int(total * val)

train_images = images[:train_images_len]
test_images = images[train_images_len:train_images_len+test_images_len]
val_images = images[train_images_len+test_images_len:]

if split_original_images:
    for image_type, image_names in [("train", train_images), ("test", test_images), ("val", val_images)]:
        for image_name in image_names:
            dst_folder_path = os.path.join(original_images_splitted_path, image_type)
            os.makedirs(dst_folder_path, exist_ok=True)
            dst_path = os.path.join(dst_folder_path, image_name)
            src_path = os.path.join(original_images_path, image_name)
            shutil.copy(src_path, dst_path)

# Extract cells from splitted images and create subimages

filename = 'H:/PatoUTN/pap/CROC original/classifications.json'
f = open (filename, "r")
data = json.loads(f.read())

width = 90
height = 90
size = 640

width_in_image = width / size
height_in_image = height / size

for image in data:
    image_name = os.path.join(original_images_path, image['image_name'])
    image_name_only, image_extension = os.path.splitext(image['image_name'])
    cv_img = cv2.imread(image_name)

    img_width = cv_img.shape[1]
    img_height = cv_img.shape[0]

    if image['image_name'] in train_images:
        image_partition = 'train'
    elif image['image_name'] in test_images:
        image_partition = 'test'
    elif image['image_name'] in val_images:
        image_partition = 'val'
    else:
        raise Exception("error")

    file_content_0_0 = []
    file_content_0_1 = []
    file_content_1_0 = []
    file_content_1_1 = []
    file_content_2_0 = []
    file_content_2_1 = []

    for cell in image['classifications']:
        x = int(cell['nucleus_x'])
        y = int(cell['nucleus_y'])

        img_class = cell['bethesda_system']

        if extract_cells:
            from_y = int(y-height/2)
            to_y = int(y+height/2)

            from_x = int(x-width/2)
            to_x = int(x+width/2)

            if from_y < 0:
                to_y = to_y + abs(from_y)
                from_y = 0

            if from_x < 0:
                to_x = to_x + abs(from_x)
                from_x = 0

            if to_y > cv_img.shape[0]:
                from_y = from_y - (to_y - cv_img.shape[0])
                to_y = cv_img.shape[0]

            if to_x > cv_img.shape[1]:
                from_x = from_x - (to_x - cv_img.shape[1])
                to_x = cv_img.shape[1]

            cell_image = cv_img[from_y:to_y, from_x:to_x]

            class_folder = os.path.join(cells_images_path, image_partition, img_class)

            if not os.path.exists(class_folder):
                os.makedirs(class_folder, exist_ok=True)

            cv2.imwrite(os.path.join(class_folder, f"{image_name_only}_{x}_{y}{image_extension}"), cell_image)

        if create_subimages:
            if x < size and y < size:
                xInImage = x / size
                yInImage = y / size
                corrected_w, corrected_y = correctSize(xInImage, yInImage)
                file_content_0_0.append(f'{img_class} {xInImage} {yInImage} {corrected_w} {corrected_y}')
            if x < size and y > img_height - size:
                xInImage = x / size
                yInImage = (y - (img_height - size)) / size
                corrected_w, corrected_y = correctSize(xInImage, yInImage)
                file_content_0_1.append(f'{img_class} {xInImage} {yInImage} {corrected_w} {corrected_y}')
            
            if x < (img_width / 2) + (size / 2) and x > (img_width / 2) - (size / 2) and y < size:
                xInImage = (x - ((img_width / 2) - (size / 2))) / size
                yInImage = y / size
                corrected_w, corrected_y = correctSize(xInImage, yInImage)
                file_content_1_0.append(f'{img_class} {xInImage} {yInImage} {corrected_w} {corrected_y}')
            if x < (img_width / 2) + (size / 2) and x > (img_width / 2) - (size / 2) and y > img_height - size:
                xInImage = (x - ((img_width / 2) - (size / 2))) / size
                yInImage = (y - (img_height - size)) / size
                corrected_w, corrected_y = correctSize(xInImage, yInImage)
                file_content_1_1.append(f'{img_class} {xInImage} {yInImage} {corrected_w} {corrected_y}')

            if x > img_width - size and y < size:
                xInImage = (x - (img_width - size)) / size
                yInImage = y / size
                corrected_w, corrected_y = correctSize(xInImage, yInImage)
                file_content_2_0.append(f'{img_class} {xInImage} {yInImage} {corrected_w} {corrected_y}')
            if x > img_width - size and y > img_height - size:
                xInImage = (x - (img_width - size)) / size
                yInImage = (y - (img_height - size)) / size
                corrected_w, corrected_y = correctSize(xInImage, yInImage)
                file_content_2_1.append(f'{img_class} {xInImage} {yInImage} {corrected_w} {corrected_y}')

    if create_subimages:
        write_labels(0, 0, file_content_0_0, image, image_partition)
        write_labels(0, 1, file_content_0_1, image, image_partition)
        write_labels(1, 0, file_content_1_0, image, image_partition)
        write_labels(1, 1, file_content_1_1, image, image_partition)
        write_labels(2, 0, file_content_2_0, image, image_partition)
        write_labels(2, 1, file_content_2_1, image, image_partition)

        imagen_id = Path(image['image_name']).stem
        rows, cols, _ = cv_img.shape
        divide_image_into_subimages(cv_img, 0, 0, size, imagen_id, 0, 0, image_partition)
        divide_image_into_subimages(cv_img, 0, rows - size, size, imagen_id, 0, 1, image_partition)

        divide_image_into_subimages(cv_img, (cols / 2) - (size / 2), 0, size, imagen_id, 1, 0, image_partition)
        divide_image_into_subimages(cv_img, (cols / 2) - (size / 2), rows - size, size, imagen_id, 1, 1, image_partition)

        divide_image_into_subimages(cv_img, cols - size, 0, size, imagen_id, 2, 0, image_partition)
        divide_image_into_subimages(cv_img, cols - size, rows - size, size, imagen_id, 2, 1, image_partition)

f.close()