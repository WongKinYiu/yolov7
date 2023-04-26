import os
import random
import shutil

# Set the source folder path, destination folder path, and train/test/validation ratios
src_folder = "H:\PatoUTN\pap\CROC original\imgs_for_classification"
dst_folder = "H:\PatoUTN\pap\CROC original\imgs_for_classification_split"
train_ratio = 0.8  # 80% of images will be used for training
test_ratio = 0.1   # 10% of images will be used for testing
val_ratio = 0.1    # 10% of images will be used for validation

# Create the destination folders
train_folder = os.path.join(dst_folder, "train")
test_folder = os.path.join(dst_folder, "test")
val_folder = os.path.join(dst_folder, "validation")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get the list of all class folders in the source folder
all_classes = os.listdir(src_folder)

for class_name in all_classes:
    # Create destination folders for the current class
    class_train_folder = os.path.join(train_folder, class_name)
    class_test_folder = os.path.join(test_folder, class_name)
    class_val_folder = os.path.join(val_folder, class_name)
    os.makedirs(class_train_folder, exist_ok=True)
    os.makedirs(class_test_folder, exist_ok=True)
    os.makedirs(class_val_folder, exist_ok=True)

    # Get the list of all images in the source class folder
    class_folder = os.path.join(src_folder, class_name)
    all_images = os.listdir(class_folder)

    # Group the images by big image name
    big_images = {}
    for image in all_images:
        big_image_name = image.split("_")[0]
        if big_image_name not in big_images:
            big_images[big_image_name] = []
        big_images[big_image_name].append(image)

    # Shuffle the big images
    big_image_names = list(big_images.keys())
    random.shuffle(big_image_names)

    # Split the big images into train, test, and validation sets
    num_images = len(big_image_names)
    num_train_images = int(num_images * train_ratio)
    num_test_images = int(num_images * test_ratio)
    num_val_images = num_images - num_train_images - num_test_images
    train_images = big_image_names[:num_train_images]
    test_images = big_image_names[num_train_images:num_train_images+num_test_images]
    val_images = big_image_names[num_train_images+num_test_images:]

    # Copy the images to the destination folders
    for image_type, image_names in [("train", train_images), ("test", test_images), ("validation", val_images)]:
        for big_image_name in image_names:
            images_to_copy = big_images[big_image_name]
            for image in images_to_copy:
                src_path = os.path.join(class_folder, image)
                dst_folder_path = os.path.join(dst_folder, image_type, class_name)
                os.makedirs(dst_folder_path, exist_ok=True)
                dst_path = os.path.join(dst_folder_path, image)
                shutil.copy(src_path, dst_path)
