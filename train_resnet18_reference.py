
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


# ==========================================
# 1. SETTINGS
# ==========================================

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_PER_CLASS = 1000
TEST_PER_CLASS = 200
LEARNING_RATE = 0.001

OUTPUT_DIR = "resnet18_reference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("Training ResNet-18 on CIFAR-10")
print("Device:", device)
print("Training images per class:", TRAIN_PER_CLASS)
print("Epochs:", EPOCHS)


# ==========================================
# 2. DATA TRANSFORMS
# ==========================================

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


# ==========================================
# 3. LOAD CIFAR-10
# ==========================================

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=test_transform
)


# ==========================================
# 4. SELECT BALANCED SUBSETS
# ==========================================

def select_indices(dataset, images_per_class):
    rng = random.Random(SEED)
    selected = []

    for class_id in range(10):
        candidates = [
            i for i, label in enumerate(dataset.targets)
            if label == class_id
        ]

        rng.shuffle(candidates)
        selected.extend(candidates[:images_per_class])

    return selected


train_indices = select_indices(
    train_dataset, TRAIN_PER_CLASS
)

test_indices = select_indices(
    test_dataset, TEST_PER_CLASS
)

train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training samples:", len(train_subset))
print("Testing samples:", len(test_subset))


# ==========================================
# 5. CREATE RESNET-18
# ==========================================

model = resnet18(weights=None)

# Adapt the first convolution for 32x32 images.
model.conv1 = nn.Conv2d(
    3, 64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

model.maxpool = nn.Identity()

# CIFAR-10 has 10 output classes.
model.fc = nn.Linear(
    model.fc.in_features, 10
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 6. EVALUATION FUNCTION
# ==========================================

def evaluate_model():
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    average_loss = total_loss / total

    return accuracy, average_loss


# ==========================================
# 7. TRAIN THE MODEL
# ==========================================

best_accuracy = 0.0
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(images)

        # Calculate loss.
        loss = criterion(outputs, labels)

        # Backpropagation.
        loss.backward()

        # Update model weights.
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100.0 * correct / total
    train_loss = running_loss / total

    test_accuracy, test_loss = evaluate_model()

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2f}%")

    # Save the best model based on test accuracy.
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy

        torch.save(
            model.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "resnet18_cifar10_reference.pth"
            )
        )

        print("Best model saved!")


# ==========================================
# 8. FINISH
# ==========================================

elapsed = (time.time() - start_time) / 60

print("\n" + "=" * 45)
print("REFERENCE MODEL TRAINING COMPLETED")
print("=" * 45)
print(f"Best test accuracy: {best_accuracy:.2f}%")
print(f"Runtime: {elapsed:.2f} minutes")
print("Saved in:", OUTPUT_DIR)
print("Replay memory used: No")
