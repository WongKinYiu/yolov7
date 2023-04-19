import cv2
import numpy as np
import math

# Cargar imagen original
img = cv2.imread('preprocessing/imagen.png')

# Tamaño deseado de las imágenes recortadas
size = 640

# Cantidad de imágenes en horizontal y vertical
rows, cols, _ = img.shape
num_cols = math.ceil(cols / size) # 3
num_rows = math.ceil(rows / size) # 2

# Ancho y alto de la imagen recortada
width = size
height = size

# Recortar imagen y rellenar con píxeles blancos si es necesario
for i in range(num_rows):
    for j in range(num_cols):
        # Coordenadas del rectángulo a recortar
        x = j * size
        y = i * size
        w = min(width, img.shape[1] - x)
        h = min(height, img.shape[0] - y)
        
        # Recortar la imagen
        crop_img = img[y:y+h, x:x+w]
        
        # Rellenar con píxeles blancos si la imagen no es del tamaño deseado
        if crop_img.shape[0] != height:
            extra_pixels_y = height - crop_img.shape[0]
            crop_img = cv2.copyMakeBorder(crop_img, 0, extra_pixels_y, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        if crop_img.shape[1] != width:
            extra_pixels_x = width - crop_img.shape[1]
            crop_img = cv2.copyMakeBorder(crop_img, 0, 0, 0, extra_pixels_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # Guardar la imagen recortada
        cv2.imwrite(f'preprocessing/imagen_{i}_{j}.png', crop_img)