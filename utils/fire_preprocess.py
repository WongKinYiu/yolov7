import cv2
import numpy as np

def fire_preprocess(img):
    """
    Fire & smoke specific preprocessing
    Input: BGR image (OpenCV)
    Output: Enhanced BGR image
    """

    # 1. Convert to LAB color space for contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. Apply CLAHE on L channel (contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # 3. Merge channels back
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Slight brightness & contrast boost
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=15)

    return img
