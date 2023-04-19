import cv2
import numpy as np
import math
import os
from pathlib import Path

img_source = 'H:\\PatoUTN\\pap\\CROC original\\base'
size = 640

def recortar(img, x, y, s, id_imagen, i, j):
    # Recortar la imagen
    x = int(x)
    y = int(y)
    crop_img = img[y:y+s, x:x+s]
    # Guardar la imagen recortada
    cv2.imwrite(f'H:\\PatoUTN\\pap\\CROC original\\croc_cortado\\{id_imagen}_{i}_{j}.png', crop_img)

for imagen in os.listdir(img_source):
    imagen = os.path.join(img_source, imagen)
    imagen_id = Path(imagen).stem
    img = cv2.imread(imagen)
    rows, cols, _ = img.shape

    recortar(img, 0, 0, size, imagen_id, 0, 0)
    recortar(img, 0, rows - size, size, imagen_id, 0, 1)

    recortar(img, (cols / 2) - (size / 2), 0, size, imagen_id, 1, 0)
    recortar(img, (cols / 2) - (size / 2), rows - size, size, imagen_id, 1, 1)

    recortar(img, cols - size, 0, size, imagen_id, 2, 0)
    recortar(img, cols - size, rows - size, size, imagen_id, 2, 1)