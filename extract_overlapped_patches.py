import numpy as np
import cv2


def extract_overlapped_patches(overlap_prop,from_path, to_path='', patch_height=640, patch_width=640):
    """Crop patches of a given image with a specified overlap proportion. The overlap proportion
    must be a number between 0 and 1."""
    img = cv2.imread(from_path)

    h_overlap = int(patch_height * overlap_prop)
    w_overlap = int(patch_width * overlap_prop)
    
    true_w_overlap = 0
    true_h_overlap = 0

    img_height, img_width, _ = img.shape

    if overlap_prop < 0 or overlap_prop >= 1:
        raise ValueError("Overlap proportion must be a number between 0 (inclusive) and 1 (exclusive)")
    elif img_height < patch_height:
        raise ValueError("Patch height can't be bigger than image height")
    elif img_width < patch_width:
        raise ValueError("Patch width can't be bigger than image width")

    cy = [] # List of the y-axis coordinates of each patch centre
    acum_height = 0
    i = 0

    # Check if given overlap make patches cover the hole image in Y axis
    while acum_height < img_height:
        cyi = 0
        if len(cy) == 0:
            cyi = patch_height / 2
        else:
            cyi = cy[i-1] + patch_height - h_overlap
        cy.append(cyi)
        acum_height = cyi + patch_height / 2
        i += 1

    if acum_height > img_height:
        height_dif = acum_height - img_height
        extra_overlap = height_dif / (len(cy)-1)
        true_h_overlap = h_overlap + extra_overlap
        # Change overlap for one that make patches cover the whole image
        for j  in range(1, len(cy)):
            cy[j] = cy[j-1] + patch_height - true_h_overlap

    # Same process but for X axis
    cx = [] # List of the x-axis coordinates of each patch centre
    acum_width = 0
    i = 0

    while acum_width < img_width:
        cxi = 0
        if len(cx) == 0:
            cxi = patch_width / 2
        else:
            cxi = cx[i-1] + patch_width - w_overlap
        cx.append(cxi)
        acum_width = cxi + patch_width / 2
        i += 1

    if acum_width > img_width:
        width_dif = acum_width - img_width
        extra_overlap = width_dif / (len(cx)-1)
        true_w_overlap = w_overlap + extra_overlap
        for j in range(1, len(cx)):
            cx[j] = cx[j-1] + patch_width - true_w_overlap
    
    # Create patches
    for j in range(len(cy)):
        for k in range(len(cx)):
            y_inf = int(cy[j] - patch_height / 2)
            x_inf = int(cx[k] - patch_width / 2)
            patch_img = img[y_inf:y_inf+patch_height, x_inf:x_inf+patch_width]

            cv2.imwrite(to_path + '/' + f'patch_{j}_{k}.png', patch_img)

    return true_h_overlap/patch_height, true_w_overlap/patch_width


if __name__ == '__main__':
    imageName = 'YourImageName'
    print(extract_overlapped_patches(0.2, 'YourImage.png', imageName))
        


