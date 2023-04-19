import cv2
import numpy as np
import math

img = cv2.imread('preprocessing/imagen.png')

size = 640

rows, cols, _ = img.shape
num_cols = math.ceil(cols / size) # 3
num_rows = math.ceil(rows / size) # 2

def recortar(img, x, y, s, i, j):
    # Recortar la imagen
    x = int(x)
    y = int(y)
    crop_img = img[y:y+s, x:x+s]
    # Guardar la imagen recortada
    cv2.imwrite(f'preprocessing/imagen_{i}_{j}.png', crop_img)



recortar(img, 0, 0, size, 0, 0)
recortar(img, 0, rows - size, size, 0, 1)

recortar(img, (cols / 2) - (size / 2), 0, size, 1, 0)
recortar(img, (cols / 2) - (size / 2), rows - size, size, 1, 1)

recortar(img, cols - size, 0, size, 2, 0)
recortar(img, cols - size, rows - size, size, 1, 1)