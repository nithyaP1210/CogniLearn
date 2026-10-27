
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
import csv
import os

# 1. Configuration
device = torch.device("cpu")
print("Using device:", device)

batch_size = 32
learning_rate = 0.001
epochs_per_task = 1

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Five sequential tasks
tasks = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]
]

# 2. Prepare dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

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

# 3. Create ResNet-18
model = resnet18(weights=None)

model.conv1 = nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. Select a small subset for each task
def get_task_loader(dataset, classes, training):
    targets = dataset.targets

    # Select at most 500 training images per class,
    # or 200 test images per class.
    limit_per_class = 500 if training else 200

    indices = []

    for class_id in classes:
        class_indices = [
            i for i, label in enumerate(targets)
            if label == class_id
        ]
        indices.extend(class_indices[:limit_per_class])

    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=0
    )

# 5. Evaluate a task using only its two output classes
def evaluate_task(task_classes):
    model.eval()
    correct = 0
    total = 0

    test_loader = get_task_loader(
        test_dataset, task_classes, training=False
    )

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Task-aware evaluation:
            # compare only the two classes in this task.
            selected_outputs = outputs[:, task_classes]
            local_predictions = selected_outputs.argmax(dim=1)

            predicted_labels = torch.tensor(
                [task_classes[p.item()] for p in local_predictions],
                dtype=torch.long
            )

            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total

# 6. Train tasks sequentially
results = []
accuracy_history = []

for task_number, task_classes in enumerate(tasks, start=1):
    print("\n" + "=" * 45)
    print("Training Task", task_number)
    print("Classes:", [class_names[c] for c in task_classes])
    print("=" * 45)

    train_loader = get_task_loader(
        train_dataset, task_classes, training=True
    )

    model.train()
    total_loss = 0.0
    batches = 0

    for epoch in range(epochs_per_task):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        print(
            "Epoch", epoch + 1,
            "| Average batch loss:",
            round(total_loss / batches, 4)
        )

    # Evaluate all tasks seen so far after this task
    current_accuracies = {}

    for previous_task_index in range(task_number):
        previous_classes = tasks[previous_task_index]
        accuracy = evaluate_task(previous_classes)

        current_accuracies[previous_task_index + 1] = accuracy

        print(
            "Task", previous_task_index + 1,
            "accuracy:", round(accuracy, 2), "%"
        )

    accuracy_history.append(current_accuracies)

    results.append({
        "after_task": task_number,
        "task_classes": ", ".join(
            class_names[c] for c in task_classes
        ),
        "average_batch_loss": round(total_loss / max(batches, 1), 4),
        "task_1_accuracy": round(current_accuracies.get(1, 0), 2),
        "task_2_accuracy": round(current_accuracies.get(2, 0), 2),
        "task_3_accuracy": round(current_accuracies.get(3, 0), 2),
        "task_4_accuracy": round(current_accuracies.get(4, 0), 2),
        "task_5_accuracy": round(current_accuracies.get(5, 0), 2)
    })

# 7. Save final model
torch.save(model.state_dict(), "resnet18_sequential_baseline.pth")

# 8. Save accuracy results to CSV
with open("baseline_results.csv", "w", newline="") as file:
    fieldnames = [
        "after_task", "task_classes", "average_batch_loss",
        "task_1_accuracy", "task_2_accuracy",
        "task_3_accuracy", "task_4_accuracy",
        "task_5_accuracy"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 45)
print("SEQUENTIAL TRAINING COMPLETED")
print("Model saved: resnet18_sequential_baseline.pth")
print("Results saved: baseline_results.csv")
print("=" * 45)

# 9. Calculate a simple forgetting measure
print("\nForgetting analysis:")

for task_index in range(4):
    task_number = task_index + 1

    # Accuracy immediately after learning this task
    initial_accuracy = accuracy_history[task_index][task_number]

    # Accuracy after learning all five tasks
    final_accuracy = accuracy_history[-1][task_number]

    forgetting = max(0.0, initial_accuracy - final_accuracy)

    print(
        "Task", task_number,
        "| Accuracy after learning:",
        round(initial_accuracy, 2), "%",
        "| Final accuracy:",
        round(final_accuracy, 2), "%",
        "| Forgetting:",
        round(forgetting, 2), "percentage points"
    )
