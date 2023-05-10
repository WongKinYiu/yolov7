import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import os
from pandas import DataFrame

# Define hyperparameters
num_classes = 6
num_epochs = 1
batch_size = 64
learning_rate = 0.001
use_wieght_balance = True

# Define device to use (CPU or GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms for data augmentation
transform_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
base_folder = 'H:/PatoUTN/pap/CROC original/prepared_dataset/cells'
train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(base_folder, 'train'), transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(base_folder, 'test'), transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

valid_dataset = torchvision.datasets.ImageFolder(root=os.path.join(base_folder, 'val'), transform=transform_test)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Define model
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.to(device)

# Define loss function and optimizer
if use_wieght_balance:
    class_counts = torch.zeros(num_classes)
    for item in train_dataset.targets:
        class_counts[item] += 1

    class_weights = 1 - (class_counts / len(train_dataset.targets))
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Define Tensorboard writer
writer = SummaryWriter()

# Training loop
total_step = len(train_loader)
train_losses = []
valid_losses = []
train_accs = []
valid_accs = []
best_valid_acc = 0
for epoch in range(num_epochs):
    train_loss = 0
    train_total = 0
    train_correct = 0
    valid_loss = 0
    valid_total = 0
    valid_correct = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Print statistics
        train_loss += loss.item() * images.size(0)
        train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total

        # Write loss and accuracy to Tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch * total_step + i)
        writer.add_scalar('Accuracy/train', train_acc, epoch * total_step + i)

        if i == 0 or (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}')
            

    # Validate the model on the validation set
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

            # Print statistics
            valid_loss += loss.item() * images.size(0)
        valid_loss = valid_loss / valid_total
        valid_acc = 100 * valid_correct / valid_total
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.2f}')

        # Write loss and accuracy to Tensorboard
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)

        # Save best model based on validation accuracy
        if valid_acc > best_valid_acc:
            torch.save(model.state_dict(), 'resnet_model.pt')
            best_valid_acc = valid_acc

    # Append losses and accuracies for plotting
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)

# Evaluate the model on the test set
model.load_state_dict(torch.load('resnet_model.pt'))
model.eval()
test_loss = 0
test_total = 0
test_correct = 0
predictions = []
targets = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Calculate test accuracy
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        # Append predictions and targets for classification report and confusion matrix
        predictions.append(predicted)
        targets.append(labels)

        # Print statistics
        test_loss += loss.item() * images.size(0)
    test_loss = test_loss / test_total
    test_acc = 100 * test_correct / test_total
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')

    # Write loss and accuracy to Tensorboard
    writer.add_scalar('Loss/test', test_loss, num_epochs)
    writer.add_scalar('Accuracy/test', test_acc, num_epochs)

# Print classification report and confusion matrix
predictions = torch.cat(predictions, dim=0)
targets = torch.cat(targets, dim=0)
conf_matrix = confusion_matrix(targets.cpu().numpy(), predictions.cpu().numpy())
df_cm = DataFrame(conf_matrix , index=train_dataset.class_to_idx, columns=train_dataset.class_to_idx)
class_report = classification_report(targets.cpu().numpy(), predictions.cpu().numpy(), target_names=train_dataset.class_to_idx)
print(f'Confusion matrix:\n{conf_matrix}')
print(f'Classification report:\n{class_report}')

# Plot the confusion matrix as an image
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