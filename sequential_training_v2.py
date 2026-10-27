
import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================

SEED = 42
BATCH_SIZE = 32
EPOCHS_PER_TASK = 2
TRAIN_PER_CLASS = 1000
TEST_PER_CLASS = 200
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

print("=" * 55)
print("COGNILEARN - RESNET18 BASELINE V2")
print("=" * 55)
print("Device:", device)
print("Random seed:", SEED)
print("Epochs per task:", EPOCHS_PER_TASK)
print("Training images per class:", TRAIN_PER_CLASS)
print("Testing images per class:", TEST_PER_CLASS)

start_time = time.time()

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

tasks = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]
]

# ==========================================
# 2. LOAD CIFAR-10
# ==========================================

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

print("\nCIFAR-10 loaded successfully.")

# ==========================================
# 3. CREATE TASK DATA LOADERS
# ==========================================

def get_task_loader(dataset, classes, training):
    limit = TRAIN_PER_CLASS if training else TEST_PER_CLASS
    indices = []

    for class_id in classes:
        class_indices = [
            i for i, label in enumerate(dataset.targets)
            if label == class_id
        ]
        indices.extend(class_indices[:limit])

    subset = Subset(dataset, indices)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=training,
        num_workers=0,
        generator=generator
    )

# ==========================================
# 4. CREATE RESNET-18
# ==========================================

model = resnet18(weights=None)

# Adapt the architecture to CIFAR-10 images
model.conv1 = nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1,
    bias=False
)

model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# One optimizer is maintained across sequential tasks.
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("ResNet-18 created.")
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

# ==========================================
# 5. EVALUATE ONE TASK
# ==========================================

def evaluate_task(task_classes):
    model.eval()

    correct = 0
    total = 0

    loader = get_task_loader(
        test_dataset, task_classes, training=False
    )

    class_tensor = torch.tensor(task_classes)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Task-aware evaluation:
            # restrict predictions to this task's two classes.
            selected_outputs = outputs[:, task_classes]
            local_predictions = selected_outputs.argmax(dim=1)

            predicted_labels = class_tensor[local_predictions]

            correct += (
                predicted_labels == labels
            ).sum().item()

            total += labels.size(0)

    return 100.0 * correct / total

# ==========================================
# 6. SEQUENTIAL TRAINING
# ==========================================

# accuracy_matrix[stage][task] = accuracy percentage
accuracy_matrix = []
stage_losses = []

for stage, task_classes in enumerate(tasks, start=1):

    print("\n" + "=" * 55)
    print("LEARNING TASK", stage)
    print("Classes:", [class_names[c] for c in task_classes])
    print("=" * 55)

    loader = get_task_loader(
        train_dataset, task_classes, training=True
    )

    for epoch in range(EPOCHS_PER_TASK):
        model.train()

        total_loss = 0.0
        total_batches = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        average_loss = total_loss / total_batches

        print(
            f"Epoch {epoch + 1}/{EPOCHS_PER_TASK}"
            f" | Average loss: {average_loss:.4f}"
        )

    stage_losses.append(average_loss)

    # Evaluate every task learned so far.
    stage_results = {}

    for task_index in range(stage):
        task_number = task_index + 1
        accuracy = evaluate_task(tasks[task_index])

        stage_results[task_number] = accuracy

        print(
            f"Task {task_number} accuracy: "
            f"{accuracy:.2f}%"
        )

    accuracy_matrix.append(stage_results)

    average_accuracy = np.mean(list(stage_results.values()))

    print(
        f"Average accuracy across {stage} learned tasks: "
        f"{average_accuracy:.2f}%"
    )

# ==========================================
# 7. SAVE ACCURACY MATRIX
# ==========================================

matrix_file = "baseline_accuracy_matrix_v2.csv"

with open(matrix_file, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "After learning",
        "Task 1", "Task 2", "Task 3", "Task 4", "Task 5",
        "Average accuracy on learned tasks"
    ])

    for stage_index, stage_results in enumerate(
        accuracy_matrix, start=1
    ):
        values = [
            round(stage_results[t], 2)
            if t in stage_results else ""
            for t in range(1, 6)
        ]

        average = np.mean(list(stage_results.values()))

        writer.writerow([
            stage_index,
            *values,
            round(float(average), 2)
        ])

# ==========================================
# 8. CALCULATE FINAL FORGETTING
# ==========================================

print("\n" + "=" * 55)
print("FORGETTING ANALYSIS")
print("=" * 55)

forgetting_values = []

for task_index in range(4):
    task_number = task_index + 1

    observed_accuracies = [
        results[task_number]
        for results in accuracy_matrix[task_index:]
        if task_number in results
    ]

    best_accuracy = max(observed_accuracies)
    final_accuracy = accuracy_matrix[-1][task_number]

    forgetting = best_accuracy - final_accuracy
    forgetting_values.append(forgetting)

    print(
        f"Task {task_number}: "
        f"Best accuracy = {best_accuracy:.2f}%, "
        f"Final accuracy = {final_accuracy:.2f}%, "
        f"Forgetting = {forgetting:.2f} percentage points"
    )

# ==========================================
# 9. PLOT ACCURACY MATRIX
# ==========================================

matrix = np.full((5, 5), np.nan)

for stage_index, results in enumerate(accuracy_matrix):
    for task_number, accuracy in results.items():
        matrix[stage_index, task_number - 1] = accuracy

masked_matrix = np.ma.masked_invalid(matrix)

plt.figure(figsize=(8, 6))

image = plt.imshow(
    masked_matrix,
    cmap="YlGnBu",
    vmin=0,
    vmax=100,
    aspect="auto"
)

plt.colorbar(image, label="Task accuracy (%)")

plt.xticks(range(5), [
    "Task 1", "Task 2", "Task 3", "Task 4", "Task 5"
])

plt.yticks(range(5), [
    "After Task 1",
    "After Task 2",
    "After Task 3",
    "After Task 4",
    "After Task 5"
])

for row in range(5):
    for col in range(5):
        if not np.isnan(matrix[row, col]):
            plt.text(
                col, row,
                f"{matrix[row, col]:.1f}%",
                ha="center", va="center"
            )

plt.title("CogniLearn: Sequential Baseline Accuracy")
plt.xlabel("Evaluated task")
plt.ylabel("Training stage")
plt.tight_layout()
plt.savefig("baseline_accuracy_matrix_v2.png", dpi=300)
plt.close()

# ==========================================
# 10. SAVE MODEL AND CONFIGURATION
# ==========================================

torch.save(
    model.state_dict(),
    "resnet18_baseline_v2.pth"
)

with open("baseline_config_v2.txt", "w") as file:
    file.write("CogniLearn ResNet-18 Baseline V2\n")
    file.write(f"Seed: {SEED}\n")
    file.write("Device: CPU\n")
    file.write("Dataset: CIFAR-10\n")
    file.write("Model initialization: random (weights=None)\n")
    file.write(f"Epochs per task: {EPOCHS_PER_TASK}\n")
    file.write(f"Batch size: {BATCH_SIZE}\n")
    file.write(f"Learning rate: {LEARNING_RATE}\n")
    file.write(f"Training images per class: {TRAIN_PER_CLASS}\n")
    file.write(f"Test images per class: {TEST_PER_CLASS}\n")
    file.write("Evaluation: task-aware\n")
    file.write("Replay memory: not used\n")

elapsed = time.time() - start_time

print("\n" + "=" * 55)
print("BASELINE V2 COMPLETED")
print("=" * 55)
print(f"Total runtime: {elapsed / 60:.2f} minutes")
print("Saved: resnet18_baseline_v2.pth")
print("Saved: baseline_accuracy_matrix_v2.csv")
print("Saved: baseline_accuracy_matrix_v2.png")
print("Saved: baseline_config_v2.txt")
print("Replay memory used: No")
