import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import  Dataset
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


import os

# Set a custom cache directory
os.environ['TORCH_HOME'] = './'

# Define your image folder and dataframe
image_folder = 'Images'
df = pd.read_csv('BrainTumor.csv')[['Image', 'Class']]
# Split your data into train, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=1/9, random_state=42)

from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None, target_transform=None):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.dataframe.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)
        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# Define image transformations
input_size = 224  # EfficientNetB0 input size
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Create datasets and dataloaders
train_dataset = CustomDataset(train_df, image_folder, transform=transform)
val_dataset = CustomDataset(val_df, image_folder, transform=transform)
test_dataset = CustomDataset(test_df, image_folder, transform=transform)

batch_size = 32  # Adjust batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



# Define the EfficientNetB0 model
model = models.efficientnet_b0(pretrained=True)  # Load pre-trained weights
num_classes = 2  # Assuming you have 2 classes (0 and 1)

# Replace the classifier with a new one
num_ftrs = model.classifier[1].in_features  # Get the number of input features for the classifier
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),  # Add an optional hidden layer if needed
    nn.ReLU(),
    nn.Dropout(0.5),  # Add dropout for regularization
    nn.Linear(512, num_classes)
)

# Set device (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('We are using:',device)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate

# Create data loaders
batch_size = 32  # Adjust batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Training loop with progress bar
num_epochs = 10  # You can adjust the number of epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Use tqdm for a progress bar
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.update(1)  # Update the progress bar

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Validation loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy}%")

# Testing loop
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy}%")
