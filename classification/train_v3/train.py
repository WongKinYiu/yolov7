"""
train.py
Modified script from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
Modify paths and hyperparams at the end of script
"""

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import build_model
from datasets import get_datasets, get_data_loaders
from utils import save_model, conf_matrix_report, missclasiffied_subclasses
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
import os


def main(epochs, lr, batch_size,pretrained, model_name, dataset_path, dest_path,
          fine_tune, label, use_weight_balance):
    """Main training function. Trains the model and saves the best epoch."""
    # Create output folder
    if label == '':
        dir_name = f'{os.path.basename(dataset_path)}_{model_name}'
    else:
        dir_name = f'{os.path.basename(dataset_path)}_{model_name}_{label}'
    os.mkdir(os.path.join(dest_path, dir_name))
    dest_path = os.path.join(dest_path, dir_name)   

    # Define Tensorboard writer
    writer = SummaryWriter(dest_path)
    # Add hyperparameters to Tensorboard
    hparams = f'Epochs: {epochs}\nLearning rate: {lr}\n' \
    f'Batch_size: {batch_size}\n' \
    f'Pretrained: {pretrained}\nModel name: {model_name}'
    writer.add_text('Main Info', hparams)

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_test, dataset_classes = get_datasets(pretrained, dataset_path)
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Number of test images: {len(dataset_test)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    train_loader, valid_loader, test_loader = get_data_loaders(dataset_train, 
                                                               dataset_valid,
                                                               dataset_test,
                                                               batch_size)
    # Learning_parameters.
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    model = build_model(
        model=model_name,
        pretrained=pretrained, 
        fine_tune=fine_tune, 
        num_classes=len(dataset_classes)
    ).to(device)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    if use_weight_balance:
        class_counts = torch.zeros(len(dataset_classes))
        for item in dataset_train.targets:
            class_counts[item] += 1

        class_weights = 1 - (class_counts / len(dataset_train.targets))
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()


    criterion = nn.CrossEntropyLoss()

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion,
                                                device)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                    criterion, device,
                                                    dest_path)
        
        # Write loss and accuracy to Tensorboard
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', train_epoch_acc, epoch)
        writer.add_scalar('Loss/valid', valid_epoch_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_epoch_acc, epoch)

        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        time.sleep(5)
    
    test_loss, test_acc, predictions, targets, miss_paths = test(model, 
                                                                 test_loader,  
                                criterion, device, dest_path, dataset_classes)
    
    # Write loss and accuracy to Tensorboard
    writer.add_scalar('Loss/test', test_loss)
    writer.add_scalar('Accuracy/test', test_acc)

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')

    # Create matrix and classification report and add it to Tensorboard
    conf_matrix_report(predictions, targets, writer, dataset_train)
    # Save missclasified images data
    missclasiffied_subclasses(miss_paths, writer)
    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, dest_path)

    print('TRAINING COMPLETE')


# Training function.
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_correct = 0
    train_running_loss = 0.0
    counter = 0

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(image)

        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        # Backpropagation
        loss.backward()

        # Update the weights.
        optimizer.step()

        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = train_running_correct / len(trainloader.dataset)
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion, device, dest_path):
    model.eval()
    print('Validation')
    valid_running_correct = 0
    valid_running_loss = 0.0
    counter = 0
    best_valid_acc = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(image)

            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            epoch_running_correct += (preds == labels).sum().item()
     
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = valid_running_correct / len(testloader.dataset)

    # Save best model based on validation accuracy
    if epoch_acc > best_valid_acc:
        torch.save(model.state_dict(), os.path.join(dest_path, 'model.pt'))
        best_valid_acc = epoch_acc

    return epoch_loss, epoch_acc


# Test function.
def test(model, testloader, criterion, device, dest_path, dataset_classes):
    model.load_state_dict(torch.load(os.path.join(dest_path, 'model.pt')))
    model.eval()
    print('Test')
   
    epoch_running_correct = 0
    epoch_running_loss = 0.0
    counter = 0
    predictions = []
    targets = []
    miss_paths = pd.DataFrame(columns=['True', 'Path'])

    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader, 0):
            images = images.to(device)
            labels = labels.to(device)
            counter += 1

            # Forward pass
            outputs = model(images)

            # Calculate the loss.
            loss = criterion(outputs, labels)
            epoch_running_loss += loss.item()
            

            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            epoch_running_correct += (preds == labels).sum().item()

            # Append predictions and targets for classification report 
            # and confusion matrix
            predictions.append(preds)
            targets.append(labels)

            # Fill dataframe with missclassified images paths
            if testloader.batch_size == 1 and preds != labels:
                filename, _ = testloader.dataset.samples[i]
                true_class = dataset_classes[labels.item()]
                new_row = {'True': true_class, 'Path': filename}
                miss_paths.loc[len(miss_paths)] = new_row               
               
    # Loss and accuracy for the complete epoch.
    epoch_loss = epoch_running_loss / counter
    epoch_acc = epoch_running_correct / len(testloader.dataset)
    return epoch_loss, epoch_acc, predictions, targets, miss_paths


if __name__ == '__main__':
    epochs = 30
    lr = 0.001
    batch_size = 16
    pretrained = True
    model_name = 'resnet18'
    fine_tune = True
    label = ''
    dataset_path = '/shared/PatoUTN/PAP/Datasets/cells_2_class_balanced'
    dest_path = '/shared/PatoUTN/PAP/Entrenamientos'
    use_weight_balance = True
    main(epochs, lr, batch_size, pretrained, model_name, dataset_path, 
         dest_path, fine_tune, label, use_weight_balance)
