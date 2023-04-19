import cv2


def extract_overlapped_patches(overlap_prop, from_path,
                               patch_height=640, patch_width=640, save=False, to_path=''):
    """
    Crop patches of a given image with a specified overlap proportion. The overlap 
    proportion must be a number between 0 and 1. If the given overlap doesn't make patches 
    cover the whole image, a bigger one will be used.
    Returns:
    dict: Key contains a tuple with x and y coordinates of the patch's center, value contains
    a cv2 image of the patch.
    """

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

    patch_centers_y, true_h_overlap = calculate_patch_coords(patch_height, img_height, h_overlap)
    patch_centers_x, true_w_overlap = calculate_patch_coords(patch_width, img_width, w_overlap)
    
    patches = {}
    # Create patches
    for j in range(len(patch_centers_y)):
        for k in range(len(patch_centers_x)):
            y_inf = int(patch_centers_y[j] - patch_height / 2)
            x_inf = int(patch_centers_x[k] - patch_width / 2)
            patch_img = img[y_inf:y_inf+patch_height, x_inf:x_inf+patch_width]

            # Add patch with its center coordinates to the image
            patches[(patch_centers_x[k], patch_centers_y[j])] = patch_img
            
            if save == True:
                if to_path == '':
                    cv2.imwrite(f'patch_{j}_{k}.png', patch_img)
                else:
                    cv2.imwrite(to_path + '/' + f'patch_{j}_{k}.png', patch_img)

    return patches


def calculate_patch_coords(patch_size, img_size, overlap):
    cc = [] # Patch center coordinates of a specified axis
    acum_size = 0
    i = 0
    while acum_size < img_size:
        ci = 0
        if len(cc) == 0:
            ci = patch_size / 2
        else:
            ci = cc[i-1] + patch_size - overlap
        cc.append(ci)
        acum_size = ci + patch_size / 2
        i += 1

    if acum_size > img_size:
        size_dif = acum_size - img_size
        extra_overlap = size_dif / (len(cc)-1)
        true_overlap = overlap + extra_overlap
        # Change overlap for one that make patches cover the whole image
        for j  in range(1, len(cc)):
            cc[j] = cc[j-1] + patch_size - true_overlap

    return cc, true_overlap


if __name__ == '__main__':
    patches = extract_overlapped_patches(0.2, 'YourImage.png')