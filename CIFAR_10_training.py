import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Callable, Dict, Tuple
import numpy as np
from opacus import GradSampleModule
from opacus import PrivacyEngine
from dp_optimizer import DPMFSGD


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
    transforms.Normalize((0.5,), (0.5,))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

# Define the split sizes
train_size = int(0.8 * len(full_train_dataset))  # 80% for training
test_size = len(full_train_dataset) - train_size  # 20% for testing

# Split the dataset
train_dataset, test_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # First block: Conv(32) -> ReLU -> Conv(32) -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second block: Conv(64) -> ReLU -> Conv(64) -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third block: Conv(128) -> ReLU -> Conv(128) -> ReLU -> MaxPool
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer (Dense Layer): 128 * 4 * 4 -> 10 classes
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        # Second block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Third block
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        # Flatten
        x = self.flatten(x)

        # Fully connected layer
        x = self.fc(x)

        return x


import torch
import torchvision
import random
from torch.utils.data import DataLoader, Subset


def train(model, train_dataset, criterion, optimizer, epochs, batch_size, device):
    model.train()

    for epoch in range(epochs):
      print(f"Epoch {epoch + 1}/{epochs}")

      for inputs, targets in train_loader:

        inputs, targets = inputs.to(device), targets.to(device)
        total_samples = inputs.size(0)

        microbatch_size = 256

        # Process in microbatches
        for start in range(0, total_samples, microbatch_size):
          end = start + microbatch_size
          micro_inputs = inputs[start:end]
          micro_targets = targets[start:end]

          optimizer.zero_microbatch_grad()
          outputs = model(micro_inputs)
          loss = criterion(outputs, micro_targets)
          loss.backward()
          optimizer.microbatch_step()

        # Step update after all microbatches
        optimizer.step()

      # Evaluate on test set after each epoch
      test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
      print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
      print('___________________________________________________________')

# Test function
def evaluate(model, test_loader, criterion, device):
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)


    epoch_loss = running_loss / len(test_loader)
    accuracy = 100 * total_correct / total_samples
    return epoch_loss, accuracy


def run_experiments(num_iterations=100, b_min_sep=10, epochs=10, clip_norm=1, lr=0.001, momentum=0, weight_decay=0.999, sigma=1, batch_size=512,  bandwidth=5, factorization_type='band', MF_coef=None, use_amplification=False):
    model = GradSampleModule(CNNModel())
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize DP-MF-SGD optimizer
    optimizer = DPMFSGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, noise_multiplier=sigma, l2_norm_clip=clip_norm, batch_size=batch_size,
                        iterations_number=num_iterations, b_min_sep=b_min_sep, band_width=bandwidth, device=device, factorization_type=factorization_type, MF_coef=MF_coef, use_amplification=use_amplification)

    train(model, train_dataset, criterion, optimizer, epochs, batch_size, device)
     
