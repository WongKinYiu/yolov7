import os
import cv2
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
import random
import shutil


def create_2_class_dataset(src_folder, dst_folder):
    """Join 6 class dataset to make one with classes: Normal, Altered"""
    for split in os.listdir(src_folder):
        src_split_folder = os.path.join(src_folder, split)
        dst_split_folder = os.path.join(dst_folder, split)
        shutil.copytree(os.path.join(src_split_folder, 'Negative for intraepithelial lesion'),
                        os.path.join(dst_split_folder, 'Normal'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(src_split_folder, 'ASC-H'),
                        os.path.join(dst_split_folder, 'Altered'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(src_split_folder, 'ASC-US'),
                        os.path.join(dst_split_folder, 'Altered'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(src_split_folder, 'HSIL'),
                        os.path.join(dst_split_folder, 'Altered'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(src_split_folder, 'LSIL'),
                        os.path.join(dst_split_folder, 'Altered'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(src_split_folder, 'SCC'),
                        os.path.join(dst_split_folder, 'Altered'), dirs_exist_ok=True)

def create_3_class_dataset(src_folder, dst_folder):
    """Join 6 class dataset to make one with classes: Normal, Low-Grade, High-Grade"""
    for split in os.listdir(src_folder):
        src_split_folder = os.path.join(src_folder, split)
        dst_split_folder = os.path.join(dst_folder, split)
        shutil.copytree(os.path.join(src_split_folder, 'Negative for intraepithelial lesion'),
                        os.path.join(dst_split_folder, 'Normal'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(src_split_folder, 'ASC-H'),
                        os.path.join(dst_split_folder, 'High-Grade'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(src_split_folder, 'ASC-US'),
                        os.path.join(dst_split_folder, 'Low-Grade'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(src_split_folder, 'HSIL'),
                        os.path.join(dst_split_folder, 'High-Grade'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(src_split_folder, 'LSIL'),
                        os.path.join(dst_split_folder, 'Low-Grade'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(src_split_folder, 'SCC'),
                        os.path.join(dst_split_folder, 'High-Grade'), dirs_exist_ok=True)   


def balance_2_class(src_folder, dst_folder, downsample_amount=0):
    """Joins classes by normal and altered, downsamples biggest class and
    augments the other"""
    create_2_class_dataset(src_folder, dst_folder)
    train_folder = os.path.join(dst_folder, 'train')
    normal_folder = os.path.join(train_folder, 'Normal')
    altered_folder = os.path.join(train_folder, 'Altered')
    
    if len(os.listdir(normal_folder)) > len(os.listdir(altered_folder)):
        downsample(normal_folder, downsample_amount)
        img_diff = len(os.listdir(normal_folder)) - len(os.listdir(altered_folder))
        augment_sample(altered_folder, img_diff)
    elif len(os.listdir(normal_folder)) < len(os.listdir(altered_folder)):
        downsample(altered_folder, downsample_amount)
        img_diff = len(os.listdir(altered_folder)) - len(os.listdir(normal_folder))
        augment_sample(normal_folder, img_diff)

def balance_3_class(src_folder, dst_folder, downsample_amount=0):
    """Joins classes by Normal, Low-Grade and High-grade. Downsamples normal
    class and augments the others."""
    create_3_class_dataset(src_folder, dst_folder)
    train_folder = os.path.join(dst_folder, 'train')
    normal_folder = os.path.join(train_folder, 'Normal')
    normal_images = os.listdir(normal_folder)
    
    to_be_removed = []
    # Downsample Normal class
    downsample(normal_folder, downsample_amount)

    # Augment Low and High Grade classes
    low_folder = os.path.join(train_folder, 'Low-Grade')
    img_diff = len(os.listdir(normal_folder)) - len(os.listdir(low_folder))
    if img_diff >= 0:
        augment_sample(low_folder, img_diff)
    else:
        raise ValueError('Normal Downsample amount is too big')

    high_folder = os.path.join(train_folder, 'High-Grade')
    img_diff = len(os.listdir(normal_folder)) - len(os.listdir(high_folder))
    if img_diff >= 0:
        augment_sample(high_folder, img_diff)
    else:
        raise ValueError('Normal Downsample amount is too big')
   

def balance_6_class(src_folder, dst_folder, downsample_amount=0):
    """Downsample biggest class and augment the others"""
    shutil.copytree(src_folder, dst_folder)
    
    train_folder = os.path.join(dst_folder, 'train')

    imgs_class_count = {}
    maxCaseAmount = 0
    maxSizeClass = ''

    # Get biggest class and store amounts in a dict
    for type in os.listdir(train_folder):
        imgs_class_count[type] = len(os.listdir(os.path.join(train_folder, type)))
        if imgs_class_count[type] > maxCaseAmount:
            maxSizeClass = type
            maxCaseAmount = imgs_class_count[type]
    
    # Downsample biggest class
    downsample(os.path.join(train_folder, maxSizeClass), downsample_amount)
    imgs_class_count[maxSizeClass] -= downsample_amount
    maxCaseAmount -= downsample_amount
    
    # Balance small classes choosing random samples to be randomly augmented
    for type in imgs_class_count.keys():
        type_path = os.path.join(train_folder, type)
        original_images = os.listdir(type_path)
        if imgs_class_count[type] < maxCaseAmount:
            img_diff = maxCaseAmount - imgs_class_count[type]
            # Avoid doing augmentation of the same image if it is possible
            if imgs_class_count[type] >= img_diff:
                sample = random.sample(range(0, imgs_class_count[type]), img_diff)
                for rnd in sample:
                    img_name = original_images[rnd]
                    img = cv2.imread(os.path.join(type_path, img_name))
                    ag_img = augment_operations(random.randint(1,10), img)
                    cv2.imwrite(os.path.join(type_path, 'ag_' + img_name), ag_img)

            else:
                ag_count = 0
                # Here the images must be augmented more than once
                for img_name in original_images:
                    img = cv2.imread(os.path.join(type_path, img_name))
                    ag_img = augment_operations(random.randint(1,10), img)
                    cv2.imwrite(os.path.join(type_path, f'ag_{ag_count}_' + img_name), ag_img)
                    ag_count += 1
                img_diff -= len(original_images)
                # After each image is augmented, we augment a original ohe randomly
                for i in range(img_diff):
                    img_name = random.choice(original_images)
                    img = cv2.imread(os.path.join(type_path, img_name))
                    ag_img = augment_operations(random.randint(1,10), img)
                    cv2.imwrite(os.path.join(type_path, f'ag_{ag_count}_' + img_name), ag_img)
                    ag_count += 1   


def augment_sample(img_folder, augment_amount):
    """Augment by a random operation a random sample"""
    images = os.listdir(img_folder)
    img_amount = len(images)
    for rnd in random.sample(range(img_amount), augment_amount):
        img_name = images[rnd]
        img = cv2.imread(os.path.join(img_folder, img_name))
        ag_img = augment_operations(random.randint(1,10), img)
        cv2.imwrite(os.path.join(img_folder, 'ag_' + img_name), ag_img)


def downsample(img_folder, downsample_amount):
    images = os.listdir(img_folder)
    img_amount = len(images)
    to_be_removed = []
    for rnd in random.sample(range(img_amount), downsample_amount):
        to_be_removed.append(images[rnd])
    for image in to_be_removed:
        os.remove(os.path.join(img_folder, image))


def augment_operations(operation, img):
    # rotacionar
    if(operation == 1):
        nova_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif(operation == 2):
        nova_img = cv2.rotate(img, cv2.ROTATE_180)
    elif(operation == 3):
        nova_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # espelhar
    elif(operation == 4):
        nova_img= cv2.flip(img, 1)
    elif(operation == 5):
        img_rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        nova_img = cv2.flip(img_rotate_90, 1)
    elif(operation == 6):
        img_rotate_180 = cv2.rotate(img, cv2.ROTATE_180)
        nova_img = cv2.flip(img_rotate_180, 1)
    elif(operation == 7):
        img_rotate_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        nova_img = cv2.flip(img_rotate_270, 1)
    elif(operation == 8):
        sigma = 0.05 
        noisy = random_noise(img, var=sigma**2)
        nova_img = noisy
        nova_img = nova_img * 255
    elif(operation == 9):
        sigma = 0.005 
        noisy = random_noise(img, var=sigma**2)
        nova_img = denoise_tv_chambolle(noisy, weight=0.05)
        nova_img = nova_img * 255
    elif(operation == 10):
        sigma = 0.005 
        noisy = random_noise(img, var=sigma**2)
        nova_img = denoise_bilateral(noisy, sigma_color=0.01, sigma_spatial=5, channel_axis=-1)
        nova_img = nova_img * 255
    return nova_img


if __name__ == '__main__':
    src_folder = 'E:\\MLPathologyProject\\pap\\CRIC\\prepared_dataset\\cells'
    dst_folder = 'E:\\MLPathologyProject\\pap\\CRIC\\prepared_dataset\\cells_3_class_balanced'
    balance_3_class(src_folder, dst_folder, 3010)
