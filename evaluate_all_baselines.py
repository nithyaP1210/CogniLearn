
import os
import csv
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset

# ---------------- CONFIGURATION ----------------
SEED = 42
BATCH_SIZE = 64
TEST_PER_CLASS = 200
DEVICE = torch.device("cpu")

TASKS = [
    [0, 1],  # airplane, automobile
    [2, 3],  # bird, cat
    [4, 5],  # deer, dog
    [6, 7],  # frog, horse
    [8, 9],  # ship, truck
]

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ---------------- DATASET ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Select the same seeded 200 test images per class.
rng = random.Random(SEED)
indices_by_class = {c: [] for c in range(10)}

for index, (_, label) in enumerate(test_dataset):
    if len(indices_by_class[label]) < TEST_PER_CLASS:
        indices_by_class[label].append(index)

# Shuffle each class independently and take the same fixed subset.
selected_indices = []
for class_id in range(10):
    candidates = indices_by_class[class_id]
    rng.shuffle(candidates)
    selected_indices.extend(candidates[:TEST_PER_CLASS])

test_subset = Subset(test_dataset, selected_indices)
test_loader = DataLoader(
    test_subset, batch_size=BATCH_SIZE, shuffle=False
)

# ---------------- MODEL ----------------
def create_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def load_weights(model, path):
    checkpoint = torch.load(
        path, map_location=DEVICE, weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()
    return model

# ---------------- EVALUATION ----------------
def evaluate(model, seen_classes, task_classes=None):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            mask = torch.zeros_like(labels, dtype=torch.bool)

            for c in seen_classes:
                mask |= labels == c

            images = images[mask]
            labels = labels[mask]

            if len(labels) == 0:
                continue

            logits = model(images)

            if task_classes is None:
                # Class-incremental: choose among all seen classes.
                logits = logits[:, seen_classes]
                predictions = logits.argmax(dim=1)
                predictions = torch.tensor(
                    seen_classes, dtype=torch.long
                )[predictions]
            else:
                # Task-aware: choose only between this task's classes.
                logits = logits[:, task_classes]
                predictions = torch.tensor(
                    task_classes, dtype=torch.long
                )[logits.argmax(dim=1)]

            correct += (predictions == labels).sum().item()
            total += len(labels)

    return 100.0 * correct / total if total else 0.0

def evaluate_checkpoint(path, version, stage, rows):
    model = create_model()
    load_weights(model, path)

    seen_classes = [c for task in TASKS[:stage] for c in task]

    for task_index, task_classes in enumerate(TASKS[:stage]):
        task_aware = evaluate(
            model, seen_classes, task_classes=task_classes
        )
        class_incremental = evaluate(
            model, seen_classes, task_classes=None
        )

        rows.append({
            "version": version,
            "stage": stage,
            "evaluated_task": task_index + 1,
            "task_aware_accuracy_percent": round(task_aware, 2),
            "class_incremental_accuracy_percent": round(
                class_incremental, 2
            )
        })

# ---------------- RUN ----------------
os.makedirs("uniform_baseline_evaluation", exist_ok=True)
rows = []

# V1 and V2 have final checkpoints only.
final_models = [
    ("V1", "resnet18_sequential_baseline.pth"),
    ("V2", "resnet18_baseline_v2.pth"),
]

for version, path in final_models:
    if os.path.exists(path):
        print(f"Evaluating {version} final checkpoint...")
        evaluate_checkpoint(path, version, 5, rows)
    else:
        print(f"Missing checkpoint: {path}")

# V3 and V4 have per-stage checkpoints.
stage_models = [
    ("V3", "baseline_v3_diagnostic_results/checkpoints"),
    ("V4", "baseline_v4_tuned_results/checkpoints"),
]

for version, folder in stage_models:
    for stage in range(1, 6):
        path = os.path.join(folder, f"after_task_{stage}.pth")

        if not os.path.exists(path):
            print(f"Missing checkpoint: {path}")
            continue

        print(f"Evaluating {version}, stage {stage}...")
        model = create_model()

        checkpoint = torch.load(
            path, map_location=DEVICE, weights_only=False
        )
        if isinstance(checkpoint, dict):
            state = checkpoint.get(
                "model_state_dict",
                checkpoint.get("state_dict", checkpoint)
            )
        else:
            state = checkpoint

        model.load_state_dict(state)
        model.eval()

        seen_classes = [
            c for task in TASKS[:stage] for c in task
        ]

        for task_index, task_classes in enumerate(TASKS[:stage]):
            task_aware = evaluate(
                model, seen_classes, task_classes=task_classes
            )
            class_incremental = evaluate(
                model, seen_classes, task_classes=None
            )

            rows.append({
                "version": version,
                "stage": stage,
                "evaluated_task": task_index + 1,
                "task_aware_accuracy_percent": round(task_aware, 2),
                "class_incremental_accuracy_percent": round(
                    class_incremental, 2
                )
            })

output_path = "uniform_baseline_evaluation/all_baseline_results.csv"

with open(output_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=[
        "version",
        "stage",
        "evaluated_task",
        "task_aware_accuracy_percent",
        "class_incremental_accuracy_percent"
    ])
    writer.writeheader()
    writer.writerows(rows)

print("\nEvaluation completed.")
print("Results saved to:", output_path)
