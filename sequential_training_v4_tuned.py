
import os
import csv
import json
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

# ==================================================
# 1. EXPERIMENT SETTINGS
# ==================================================

SEED = 42
BATCH_SIZE = 32
EPOCHS_PER_TASK = 5
TRAIN_PER_CLASS = 1000
TEST_PER_CLASS = 200
LEARNING_RATE = 0.001

# Separate folder to preserve original V3 results.
OUTPUT_DIR = "baseline_v4_tuned_results"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

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

print("=" * 60)
print("COGNILEARN - RESNET18 BASELINE V4 TUNED EXPERIMENT")
print("=" * 60)
print("Device:", device)
print("Seed:", SEED)
print("Epochs per task:", EPOCHS_PER_TASK)
print("Training images per class:", TRAIN_PER_CLASS)
print("Test images per class:", TEST_PER_CLASS)

start_time = time.time()

# ==================================================
# 2. LOAD CIFAR-10
# ==================================================

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

print("\nDataset loaded successfully.")

# ==================================================
# 3. CREATE REPRODUCIBLE, BALANCED SUBSETS
# ==================================================

def select_indices(dataset, classes, limit):
    rng = random.Random(SEED)
    selected = []

    for class_id in classes:
        candidates = [
            i for i, label in enumerate(dataset.targets)
            if label == class_id
        ]

        # Select a reproducible subset for each class.
        rng.shuffle(candidates)
        selected.extend(candidates[:limit])

    return selected


def get_loader(dataset, classes, training):
    limit = TRAIN_PER_CLASS if training else TEST_PER_CLASS

    indices = select_indices(dataset, classes, limit)
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


# ==================================================
# 4. CREATE RESNET-18
# ==================================================

model = resnet18(weights=None)

# Adapt ResNet-18 to small CIFAR-10 images.
model.conv1 = nn.Conv2d(
    3, 64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("ResNet-18 initialized from scratch.")
print(
    "Total parameters:",
    sum(p.numel() for p in model.parameters())
)

# ==================================================
# 5. EVALUATION FUNCTION WITH DIAGNOSTICS
# ==================================================

def evaluate(classes_to_test, allowed_classes, debug=False):
    """
    Evaluate test examples belonging to classes_to_test.

    Predictions are restricted to allowed_classes.

    Task-aware evaluation:
        allowed_classes contains the two classes of one task.

    Class-incremental evaluation:
        allowed_classes contains every class seen so far.

    If debug=True, print the number of predictions for
    each allowed class.
    """

    model.eval()

    loader = get_loader(
        test_dataset,
        classes_to_test,
        training=False
    )

    allowed_tensor = torch.tensor(
        allowed_classes,
        dtype=torch.long
    )

    correct = 0
    total = 0
    prediction_counts = torch.zeros(10, dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Keep only the allowed class logits.
            selected_outputs = outputs[:, allowed_classes]

            # Find the winning position among allowed classes.
            local_predictions = selected_outputs.argmax(dim=1)

            # Convert positions back to original CIFAR-10 labels.
            predicted_labels = allowed_tensor[local_predictions]

            correct += (
                predicted_labels == labels
            ).sum().item()

            total += labels.size(0)

            # Count how often each original class was predicted.
            prediction_counts += torch.bincount(
                predicted_labels.cpu(),
                minlength=10
            )

    accuracy = 100.0 * correct / total if total > 0 else 0.0

    if debug:
        print(
            "    Diagnostic - allowed classes:",
            [class_names[c] for c in allowed_classes]
        )

        print(
            "    Diagnostic - actual test classes:",
            [class_names[c] for c in classes_to_test]
        )

        print("    Diagnostic - predicted class counts:")

        for class_id in allowed_classes:
            print(
                f"      {class_names[class_id]}: "
                f"{prediction_counts[class_id].item()}"
            )

        print(
            f"    Diagnostic - accuracy: {accuracy:.2f}%"
        )

    return accuracy


# Stores accuracy after each training stage.
task_aware_history = []
class_incremental_history = []
stage_losses = []

# ==================================================
# 6. SEQUENTIAL TRAINING
# ==================================================

for stage, current_task in enumerate(tasks, start=1):

    print("\n" + "=" * 60)
    print("TRAINING TASK", stage)
    print("Classes:", [class_names[c] for c in current_task])
    print("=" * 60)

    train_loader = get_loader(
        train_dataset,
        current_task,
        training=True
    )

    # Train the same model across all tasks.
    # Do not reinitialize the model between tasks.
    for epoch in range(EPOCHS_PER_TASK):

        model.train()

        running_loss = 0.0
        batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        epoch_loss = running_loss / batches

        print(
            f"Epoch {epoch + 1}/{EPOCHS_PER_TASK}"
            f" | Average loss: {epoch_loss:.4f}"
        )

    stage_losses.append(epoch_loss)

    seen_classes = [
        class_id
        for task in tasks[:stage]
        for class_id in task
    ]

    task_results = {}
    incremental_results = {}

    # Evaluate every task learned so far.
    for task_index in range(stage):

        task_number = task_index + 1
        task_classes = tasks[task_index]

        # Task-aware evaluation:
        # the model chooses between the task's two classes.
        task_accuracy = evaluate(
            classes_to_test=task_classes,
            allowed_classes=task_classes
        )

        task_results[task_number] = task_accuracy

        # Class-incremental evaluation:
        # the model chooses among all classes seen so far.
        # Diagnostics are enabled here.
        incremental_accuracy = evaluate(
            classes_to_test=task_classes,
            allowed_classes=seen_classes,
            debug=True
        )

        incremental_results[task_number] = incremental_accuracy

        print(
            f"Task {task_number} | "
            f"Task-aware: {task_accuracy:.2f}% | "
            f"Class-incremental: {incremental_accuracy:.2f}%"
        )

    task_aware_history.append(task_results)
    class_incremental_history.append(incremental_results)

    # Evaluate all classes encountered so far together.
    all_seen_accuracy = evaluate(
        classes_to_test=seen_classes,
        allowed_classes=seen_classes
    )

    print(
        f"Overall class-incremental accuracy after Task {stage}: "
        f"{all_seen_accuracy:.2f}%"
    )

    # Save a checkpoint after each task.
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"after_task_{stage}.pth"
    )

    torch.save({
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "task_aware_history": task_aware_history,
        "class_incremental_history": class_incremental_history,
        "seed": SEED
    }, checkpoint_path)

    print("Checkpoint saved:", checkpoint_path)


# ==================================================
# 7. SAVE ACCURACY MATRICES
# ==================================================

def save_matrix(filename, history):

    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "After task",
            "Task 1",
            "Task 2",
            "Task 3",
            "Task 4",
            "Task 5",
            "Average accuracy on learned tasks"
        ])

        for stage_index, results in enumerate(history, start=1):

            values = [
                round(results[t], 2) if t in results else ""
                for t in range(1, 6)
            ]

            average = np.mean(list(results.values()))

            writer.writerow([
                stage_index,
                *values,
                round(float(average), 2)
            ])

    print("Saved:", path)


save_matrix(
    "task_aware_accuracy.csv",
    task_aware_history
)

save_matrix(
    "class_incremental_accuracy.csv",
    class_incremental_history
)


# ==================================================
# 8. FORGETTING ANALYSIS
# ==================================================

forgetting_results = []

print("\n" + "=" * 60)
print("TASK-AWARE FORGETTING ANALYSIS")
print("=" * 60)

for task_index in range(4):

    task_number = task_index + 1

    observed = [
        results[task_number]
        for results in task_aware_history[task_index:]
        if task_number in results
    ]

    best_accuracy = max(observed)
    final_accuracy = task_aware_history[-1][task_number]

    forgetting = best_accuracy - final_accuracy
    forgetting_results.append(forgetting)

    print(
        f"Task {task_number}: "
        f"Best={best_accuracy:.2f}%, "
        f"Final={final_accuracy:.2f}%, "
        f"Forgetting={forgetting:.2f} percentage points"
    )


# ==================================================
# 9. SAVE SUMMARY METRICS
# ==================================================

final_class_incremental = class_incremental_history[-1]
final_task_aware = task_aware_history[-1]

# This uses 200 test images per class (2,000 total),
# not the complete 10,000-image CIFAR-10 test set.
final_overall_accuracy = evaluate(
    classes_to_test=list(range(10)),
    allowed_classes=list(range(10))
)

summary = {
    "seed": SEED,
    "device": str(device),
    "model": "ResNet-18",
    "initialization": "random; weights=None",
    "tasks": len(tasks),
    "epochs_per_task": EPOCHS_PER_TASK,
    "training_images_per_class": TRAIN_PER_CLASS,
    "test_images_per_class": TEST_PER_CLASS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "replay_memory_used": False,
    "final_all_class_accuracy_percent": round(
        final_overall_accuracy, 2
    ),
    "final_task_aware_accuracies_percent": {
        str(k): round(v, 2)
        for k, v in final_task_aware.items()
    },
    "final_class_incremental_task_accuracies_percent": {
        str(k): round(v, 2)
        for k, v in final_class_incremental.items()
    },
    "forgetting_percentage_points": {
        str(i + 1): round(v, 2)
        for i, v in enumerate(forgetting_results)
    },
    "runtime_minutes": round(
        (time.time() - start_time) / 60, 2
    )
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "summary_metrics.json"
)

with open(summary_path, "w") as file:
    json.dump(summary, file, indent=4)

# Save the final model separately.
torch.save(
    model.state_dict(),
    os.path.join(
        OUTPUT_DIR,
        "resnet18_baseline_final.pth"
    )
)


# ==================================================
# 10. PLOT ACCURACY MATRIX
# ==================================================

matrix = np.full((5, 5), np.nan)

for stage_index, results in enumerate(class_incremental_history):
    for task_number, accuracy in results.items():
        matrix[stage_index, task_number - 1] = accuracy

plt.figure(figsize=(9, 6))

masked = np.ma.masked_invalid(matrix)

plot = plt.imshow(
    masked,
    cmap="YlGnBu",
    vmin=0,
    vmax=100,
    aspect="auto"
)

plt.colorbar(
    plot,
    label="Class-incremental accuracy (%)"
)

plt.xticks(
    range(5),
    [f"Task {i}" for i in range(1, 6)]
)

plt.yticks(
    range(5),
    [f"After Task {i}" for i in range(1, 6)]
)

for row in range(5):
    for col in range(5):
        if not np.isnan(matrix[row, col]):
            plt.text(
                col,
                row,
                f"{matrix[row, col]:.1f}%",
                ha="center",
                va="center"
            )

plt.title(
    "CogniLearn Baseline V4: Class-Incremental Accuracy"
)

plt.xlabel("Evaluated task")
plt.ylabel("Training stage")
plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "class_incremental_accuracy.png"
)

plt.savefig(plot_path, dpi=300)
plt.close()

print("\n" + "=" * 60)
print("BASELINE V4 TUNED EXPERIMENT COMPLETED")
print("=" * 60)
print(
    "Final all-class accuracy:",
    round(final_overall_accuracy, 2),
    "%"
)
print("Runtime:", summary["runtime_minutes"], "minutes")
print("Results folder:", OUTPUT_DIR)
print("Replay memory used: No")
print("All checkpoints and summary metrics saved.")
