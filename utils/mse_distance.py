import os
import numpy as np
import cv2  # OpenCV supports TIFF in uint16
from glob import glob

# Paths
folder1 = r"C:\Users\hanoch\Downloads\decompressed_mvk"
folder2 = r"C:\Users\hanoch\Downloads\OneDrive_2_26-08-2025"

def normalized_mse(img1, img2):
    """Compute normalized MSE between two uint16 images."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    norm = np.mean(img1 ** 2)  # normalization by signal power

    return mse / (norm + 1e-12)  # avoid divide by zero

# Collect TIFF files
files1 = {os.path.basename(f): f for f in glob(os.path.join(folder1, "*.tif"))}
files2 = {os.path.basename(f): f for f in glob(os.path.join(folder2, "*.tif"))}

common_files = sorted(set(files1.keys()) & set(files2.keys()))

results = {}

for fname in common_files:
    img1 = cv2.imread(files1[fname], cv2.IMREAD_UNCHANGED)  # preserve uint16
    img2 = cv2.imread(files2[fname], cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        print(f"⚠️ Could not read {fname}")
        continue

    if img1.shape != img2.shape:
        print(f"⚠️ Shape mismatch for {fname}: {img1.shape} vs {img2.shape}")
        continue

    nmse = normalized_mse(img1, img2)
    results[fname] = nmse
    print(f"{fname}: NMSE = {nmse:.6f}")

# Optional: compute average
if results:
    avg_nmse = np.mean(list(results.values()))
    print("\n✅ Average NMSE:", avg_nmse)
else:
    print("No matching TIFF files found.")
