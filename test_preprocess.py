import cv2
from utils.fire_preprocess import fire_preprocess

# Read test image
img = cv2.imread("test_images/fire_1.jpg")

if img is None:
    print("Error: Image not found!")
    exit()

# Apply preprocessing
out = fire_preprocess(img)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Preprocessed", out)

cv2.waitKey(0)
cv2.destroyAllWindows()
