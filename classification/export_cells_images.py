import json
import cv2
import os

source = "H:\\PatoUTN\\pap\\CROC original\\base"
dest = "H:\\PatoUTN\\pap\\CROC original\\imgs_for_classification"

if not os.path.exists(dest):
  os.mkdir(dest)

# JSON file
filename = 'H:\\PatoUTN\\pap\\CROC original\\classifications.json'
f = open (filename, "r")
data = json.loads(f.read())

img_width = 1376
img_height = 1020

size = 640

width = 90
height = 90

for image in data:
  image_name = os.path.join(source, image['image_name'])

  image_name_only, image_extension = os.path.splitext(image['image_name'])

  cv_img = cv2.imread(image_name)

  for cell in image['classifications']:
    x = int(cell['nucleus_x'])
    y = int(cell['nucleus_y'])

    img_class = cell['bethesda_system']

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

    class_folder = os.path.join(dest, img_class)

    if not os.path.exists(class_folder):
      os.mkdir(class_folder)

    cv2.imwrite(os.path.join(class_folder, f"{image_name_only}_{x}_{y}{image_extension}"), cell_image)

# Closing JSON file
f.close()


