
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

# 1. Use CPU
device = torch.device("cpu")
print("Using device:", device)

# 2. Prepare CIFAR-10 images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

# 3. Load dataset
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

# Task 1: airplane (0) and automobile (1)
task_classes = [0, 1]

train_indices = [
    i for i, label in enumerate(train_dataset.targets)
    if label in task_classes
][:1000]

test_indices = [
    i for i, label in enumerate(test_dataset.targets)
    if label in task_classes
][:400]

train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

train_loader = DataLoader(
    train_subset, batch_size=32, shuffle=True, num_workers=0
)

test_loader = DataLoader(
    test_subset, batch_size=32, shuffle=False, num_workers=0
)

print("Task 1 classes: airplane and automobile")
print("Training images:", len(train_subset))
print("Testing images:", len(test_subset))

# 4. Create ResNet-18
model = resnet18(weights=None)

# Adapt ResNet-18 for 32x32 CIFAR-10 images
model.conv1 = nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# 5. Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Train for one epoch
model.train()
total_loss = 0.0
correct = 0
total = 0

for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    predictions = outputs.argmax(dim=1)
    correct += (predictions == labels).sum().item()
    total += labels.size(0)

print("\nTraining completed!")
print("Average batch loss:", round(total_loss / len(train_loader), 4))
print("Training accuracy:", round(100 * correct / total, 2), "%")

# 7. Evaluate on test images
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Task-aware evaluation: consider only classes 0 and 1
        task_outputs = outputs[:, task_classes]
        predictions = task_outputs.argmax(dim=1)

        # Convert local predictions (0, 1) to original CIFAR-10 labels
        predicted_labels = torch.tensor(
            [task_classes[p.item()] for p in predictions]
        )

        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print("Task 1 test accuracy:", round(accuracy, 2), "%")

# 8. Save model
torch.save(model.state_dict(), "resnet18_task1.pth")
print("Model saved as resnet18_task1.pth")
