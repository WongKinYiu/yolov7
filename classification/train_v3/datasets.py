import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
# Required constants.

IMAGE_SIZE = 90 # Image size of resize when applying transforms.
NUM_WORKERS = 4 # Number of parallel processes for data preparation.

# Training transforms
def get_train_transform(IMAGE_SIZE, pretrained):
    train_transform = transforms.Compose([
        #transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_transform

# Validation transforms
def get_valid_transform(IMAGE_SIZE, pretrained):
    valid_transform = transforms.Compose([
        #transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return valid_transform

# Test transforms
def get_test_transform(IMAGE_SIZE, pretrained):
    test_transform = transforms.Compose([
        #transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return test_transform

# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize


def get_datasets(pretrained, ROOT_DIR):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        os.path.join(ROOT_DIR,'train'), 
        transform=(get_train_transform(IMAGE_SIZE, pretrained))
    )
    
    dataset_valid = datasets.ImageFolder(
        os.path.join(ROOT_DIR,'val'), 
        transform=(get_valid_transform(IMAGE_SIZE, pretrained))
    )
    dataset_test = datasets.ImageFolder(
        os.path.join(ROOT_DIR,'test'), 
        transform=(get_test_transform(IMAGE_SIZE, pretrained))
    )

    return dataset_train, dataset_valid, dataset_test, dataset_train.classes


def get_data_loaders(dataset_train, dataset_valid, dataset_test, BATCH_SIZE):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    # Batch size for the test dataset is 1 so the test function can
    # retrieve the filename of missclasified images
    test_loader = DataLoader(
        dataset_test, batch_size=1, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader, test_loader