import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import build_model
from datasets import get_datasets, get_data_loaders
from utils import save_model, save_plots

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import os
from pandas import DataFrame

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=20,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-pt', '--pretrained', action='store_true',
    help='Whether to use pretrained weights or not'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.0001,
    help='Learning rate for training the model'
)
args = vars(parser.parse_args())

# Define device to use (CPU or GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Tensorboard writer
writer = SummaryWriter()

# Training function.
def train(model, trainloader, optimizer, criterion, epoch, writer):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total_step = len(trainloader)

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
        train_total += labels.size(0)
        train_running_correct += (preds == labels).sum().item()

        # Print statistics
        train_loss += loss.item() * image.size(0)
        train_loss = train_loss / train_total
        train_acc = 100 * train_running_correct / train_total

        # Write loss and accuracy to Tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch * total_step + i)
        writer.add_scalar('Accuracy/train', train_acc, epoch * total_step + i)

        if i == 0 or (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}], Step [{i+1}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}')
            #print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}')
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion, epoch, writer):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
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
            valid_total += labels.size(0)
            valid_running_correct += (preds == labels).sum().item()

            # Print statistics
            valid_loss += loss.item() * image.size(0)
        
        valid_loss = valid_loss / valid_total
        valid_acc = 100 * valid_running_correct / valid_total

        #print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.2f}')
        print(f'Epoch [{epoch+1}], Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.2f}')

        # Write loss and accuracy to Tensorboard
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)

        # Save best model based on validation accuracy
        if valid_acc > best_valid_acc:
            torch.save(model.state_dict(), 'resnet_model.pt')
            best_valid_acc = valid_acc

    # Append losses and accuracies for plotting
    #train_losses.append(train_loss)
    #valid_losses.append(valid_loss)
    #train_accs.append(train_acc)
    #valid_accs.append(valid_acc)
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


# Test function.
def test(model, testloader, criterion, epoch, writer):
    model.load_state_dict(torch.load('resnet_model.pt'))
    model.eval()
    print('Test')
    test_loss = 0
    test_running_loss = 0.0
    test_running_correct = 0
    counter = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(image)

            # Calculate the loss.
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()


            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_running_correct += (preds == labels).sum().item()

            # Append predictions and targets for classification report and confusion matrix
            predictions.append(preds)
            targets.append(labels)

            # Print statistics
            test_loss += loss.item() * image.size(0)
        test_loss = test_loss / test_total
        test_acc = 100 * test_running_correct / test_total
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')

        # Write loss and accuracy to Tensorboard
        writer.add_scalar('Loss/test', test_loss, args['epochs'])
        writer.add_scalar('Accuracy/test', test_acc, args['epochs'])
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = test_running_loss / counter
    epoch_acc = 100. * (test_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc, predictions, targets


def metricas(predictions, targets, writer, train_dataset):
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    conf_matrix = confusion_matrix(targets.cpu().numpy(), predictions.cpu().numpy())

    df_cm = DataFrame(conf_matrix , index=train_dataset.class_to_idx, columns=train_dataset.class_to_idx)
    class_report = classification_report(targets.cpu().numpy(), predictions.cpu().numpy(), target_names=train_dataset.class_to_idx)
    print(f'Confusion matrix:\n{conf_matrix}')
    print(f'Classification report:\n{class_report}')

    fig = plt.figure()
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

    writer.add_figure('confussion_matrix', fig)

    # save class report
    with open('classification_report.txt', 'w') as f:
        f.write(class_report)
    return



if __name__ == '__main__':
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_test, dataset_classes = get_datasets(args['pretrained'])
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Number of test images: {len(dataset_test)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    train_loader, valid_loader, test_loader = get_data_loaders(dataset_train, dataset_valid)
    # Learning_parameters. 
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    model = build_model(
        pretrained=args['pretrained'], 
        fine_tune=True, 
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
    criterion = nn.CrossEntropyLoss()
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion, epoch, writer)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                    criterion, epoch, writer)
        
        
        
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        time.sleep(5)
    
    test_loss, test_acc, predictions, targets = test(model, test_loader,  
                                criterion, epoch, writer)
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

    metricas(predictions, targets, writer, dataset_train)
    
    
    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, args['pretrained'])
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, args['pretrained'])
    print('TRAINING COMPLETE')