import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

data_dir = "brain_tumor_dataset"
classes = ['yes', 'no']
batch_size = 32


# Set TORCH_HOME to a writable directory
os.environ['TORCH_HOME'] = '~/torch'

# Step 2: Prepare the dataseta
class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Dynamically split data
        for label, class_name in enumerate(classes):
            class_dir = os.path.join(data_dir, class_name)
            filenames = os.listdir(class_dir)
            np.random.shuffle(filenames) # Shuffle each time
            split = int(0.8 * len(filenames))
            if train:
                filenames = filenames[:split]
            else:
                filenames = filenames[split:]
            
            for filename in filenames:
                self.images.append(os.path.join(class_dir, filename))
                self.labels.append(label)
        
        # Shuffle dataset
        temp = list(zip(self.images, self.labels))
        np.random.shuffle(temp)
        self.images, self.labels = zip(*temp)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

# Step 3: Define data transformations
transform = transforms.Compose([
    transforms.Resize(299),  # Inception V3 expects 299x299 images
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Step 4 & 5: Load the dataset and create data loaders
train_dataset = BrainTumorDataset(data_dir, transform=transform, train=True)
test_dataset = BrainTumorDataset(data_dir, transform=transform, train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 6: Load and modify the Inception V3 model
from torchvision.models import inception_v3, Inception_V3_Weights

weights = Inception_V3_Weights.IMAGENET1K_V1
model = inception_v3(weights=weights, aux_logits=True)  # Ensure aux_logits is True if you want to use the auxiliary output

# Modify the final layer for binary classification
model.fc = torch.nn.Linear(model.fc.in_features, 2)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Steps 7 & 8 would involve training the model and evaluating its performance
# Note: This is a simplified example, and training/evaluation steps are not included here.



import torch
from tqdm import tqdm

# Assuming the previous code has been executed, and model, train_loader, and test_loader are already defined

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Running on: ",device)
def train_model(model, train_loader, loss_fn, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Handle the model's output
            outputs = model(images)
            if isinstance(outputs, torch.Tensor):
                loss = loss_fn(outputs, labels)
            else:
                # If the model returns a tuple (main output, aux output), unpack it
                output, aux_output = outputs
                loss1 = loss_fn(output, labels)
                loss2 = loss_fn(aux_output, labels)
                loss = loss1 + 0.4 * loss2  # Weighted sum of the main loss and the auxiliary loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)  # Use main output for accuracy calculation
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")


def evaluate_model(model, test_loader):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    
    avg_loss = running_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Training the model
num_epochs = 10  # You can adjust this according to your needs
train_model(model, train_loader, loss_fn, optimizer, num_epochs=num_epochs)

# Evaluating the model
evaluate_model(model, test_loader)
