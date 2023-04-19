import json

def write_file(i, j, labels):
  file_name = f'H:\\PatoUTN\\pap\\CROC original\\labels_cortado\\{image["image_name"].split(".")[0]}_{i}_{j}.txt'
  annotation_file = open(file_name, 'w')
  annotation_file.writelines(line + '\n' for line in labels)
  annotation_file.close()

def correctSize(xx, yy):
  corrected_x = width_in_image
  corrected_y = height_in_image
  # if xx + width_in_image > 1:
  #   corrected_x = width_in_image - ((xx + width_in_image) - 1)
  
  # if yy + height_in_image > 1:
  #   corrected_y = height_in_image - ((yy + height_in_image) - 1)

  return corrected_x, corrected_y

classes = {
    "Negative for intraepithelial lesion" : 0,
    "ASC-US" : 1,
    "ASC-H" : 2,
    "LSIL" : 3,
    "HSIL" : 4,
    "SCC" : 5,
}

# JSON file
filename = 'H:\\PatoUTN\\pap\\CROC original\\classifications.json'
f = open (filename, "r")
data = json.loads(f.read())

img_width = 1376
img_height = 1020

size = 640

width = 90
height = 90

width_in_image = width / size
height_in_image = height / size

for image in data:

    # if not image['image_name'] == 'fb5e83755f682922aee859fd52013c36.png':
    #   continue

    file_content_0_0 = []
    file_content_0_1 = []
    file_content_1_0 = []
    file_content_1_1 = []
    file_content_2_0 = []
    file_content_2_1 = []

    for cell in image['classifications']:
      cell_class = classes[cell["bethesda_system"]]
      x = int(cell['nucleus_x'])
      y = int(cell['nucleus_y'])

      if x < size and y < size:
        xInImage = x / size
        yInImage = y / size

        corrected_w, corrected_y = correctSize(xInImage, yInImage)

        file_content_0_0.append(f'{cell_class} {xInImage} {yInImage} {corrected_w} {corrected_y}')
      if x < size and y > img_height - size:
        xInImage = x / size
        yInImage = (y - (img_height - size)) / size
        corrected_w, corrected_y = correctSize(xInImage, yInImage)
        file_content_0_1.append(f'{cell_class} {xInImage} {yInImage} {corrected_w} {corrected_y}')
      
      if x < (img_width / 2) + (size / 2) and x > (img_width / 2) - (size / 2) and y < size:
        xInImage = (x - ((img_width / 2) - (size / 2))) / size
        yInImage = y / size
        corrected_w, corrected_y = correctSize(xInImage, yInImage)
        file_content_1_0.append(f'{cell_class} {xInImage} {yInImage} {corrected_w} {corrected_y}')
      if x < (img_width / 2) + (size / 2) and x > (img_width / 2) - (size / 2) and y > img_height - size:
        xInImage = (x - ((img_width / 2) - (size / 2))) / size
        yInImage = (y - (img_height - size)) / size
        corrected_w, corrected_y = correctSize(xInImage, yInImage)
        file_content_1_1.append(f'{cell_class} {xInImage} {yInImage} {corrected_w} {corrected_y}')

      if x > img_width - size and y < size:
        xInImage = (x - (img_width - size)) / size
        yInImage = y / size
        corrected_w, corrected_y = correctSize(xInImage, yInImage)
        file_content_2_0.append(f'{cell_class} {xInImage} {yInImage} {corrected_w} {corrected_y}')
      if x > img_width - size and y > img_height - size:
        xInImage = (x - (img_width - size)) / size
        yInImage = (y - (img_height - size)) / size
        corrected_w, corrected_y = correctSize(xInImage, yInImage)
        file_content_2_1.append(f'{cell_class} {xInImage} {yInImage} {corrected_w} {corrected_y}')

    # write files with annotations for the image
    write_file(0, 0, file_content_0_0)
    write_file(0, 1, file_content_0_1)
    write_file(1, 0, file_content_1_0)
    write_file(1, 1, file_content_1_1)
    write_file(2, 0, file_content_2_0)
    write_file(2, 1, file_content_2_1)

# Closing JSON file
f.close()


