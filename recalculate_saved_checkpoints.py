"""Re-evaluate existing CogniLearn V3/V4 checkpoints on consistent test subsets.

Place this file in the CogniLearn project folder and run:
    python recalculate_saved_checkpoints.py

It does NOT retrain the models and does NOT overwrite the original results.
It writes corrected evaluation outputs to new *_corrected folders.
"""
import os
import csv
import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

SEED = 42
BATCH_SIZE = 32
TEST_PER_CLASS = 200
DEVICE = torch.device("cpu")
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


def stable_indices(dataset, classes, limit):
    """Return identical selected test indices for each class in every evaluation."""
    selected = []
    for class_id in classes:
        rng = random.Random(SEED + class_id)
        candidates = [
            i for i, label in enumerate(dataset.targets)
            if int(label) == class_id
        ]
        rng.shuffle(candidates)
        selected.extend(candidates[:limit])
    return selected


def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)


def evaluate(model, dataset, classes_to_test, allowed_classes):
    indices = stable_indices(dataset, classes_to_test, TEST_PER_CLASS)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    allowed_tensor = torch.tensor(allowed_classes, dtype=torch.long)
    correct = 0
    total = 0
    pred_counts = torch.zeros(10, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            selected_outputs = outputs[:, allowed_classes]
            local_predictions = selected_outputs.argmax(dim=1)
            predictions = allowed_tensor[local_predictions]
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            pred_counts += torch.bincount(predictions.cpu(), minlength=10)
    return (100.0 * correct / total if total else 0.0), pred_counts.tolist(), total


def evaluate_experiment(folder, output_folder, label):
    print("\n" + "=" * 65)
    print(f"Re-evaluating {label} checkpoints: {folder}")
    print("=" * 65)
    if not os.path.isdir(folder):
        print(f"SKIP: folder not found: {folder}")
        return
    checkpoint_dir = os.path.join(folder, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        print(f"SKIP: checkpoint folder not found: {checkpoint_dir}")
        return
    
    os.makedirs(output_folder, exist_ok=True)

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=TRANSFORM
    )

    matrix = []
    task_aware_rows = []
    summary = []

    for stage in range(1, len(TASKS) + 1):
        checkpoint_path = os.path.join(
            checkpoint_dir, f"after_task_{stage}.pth"
        )

        if not os.path.isfile(checkpoint_path):
            print(f"Missing checkpoint: {checkpoint_path}")
            continue

        checkpoint = torch.load(
            checkpoint_path, map_location=DEVICE, weights_only=False
        )
        model = make_model()
        model.load_state_dict(checkpoint["model_state_dict"])

        seen_classes = [
            c for task in TASKS[:stage] for c in task
        ]

        stage_results = []
        for task_number, task_classes in enumerate(TASKS[:stage], start=1):
            accuracy, counts, total = evaluate(
                model, test_dataset, task_classes, seen_classes
            )
            stage_results.append(accuracy)
            task_aware_accuracy, _, _ = evaluate(
                model, test_dataset, task_classes, task_classes
            )

            task_aware_rows.append({
                "stage": stage,
                "tested_task": task_number,
                "class_incremental_accuracy_percent": round(accuracy, 2),
                "task_aware_accuracy_percent": round(task_aware_accuracy, 2),
                "samples": total,
            })

        # Accuracy across all classes seen up to this stage.
        overall_accuracy, pred_counts, total = evaluate(
            model, test_dataset, seen_classes, seen_classes
        )

        matrix.append(stage_results + [None] * (len(TASKS) - len(stage_results)))

        summary.append({
            "stage": stage,
            "seen_classes": ",".join(CLASS_NAMES[c] for c in seen_classes),
            "class_incremental_accuracy_percent": round(overall_accuracy, 2),
            "samples": total,
            "checkpoint": checkpoint_path,
        })

        print(
            f"Stage {stage}: class-incremental accuracy "
            f"{overall_accuracy:.2f}% ({total} samples)"
        )
        print("Predictions by class:", {
            CLASS_NAMES[i]: n for i, n in enumerate(pred_counts) if n
        })

    with open(os.path.join(output_folder, "summary_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    with open(os.path.join(output_folder, "task_accuracy.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=task_aware_rows[0].keys())
        writer.writeheader()
        writer.writerows(task_aware_rows)

    print(f"Saved evaluation files in: {output_folder}")


if __name__ == "__main__":
    evaluate_experiment(
        "baseline_v3_diagnostic_results",
        "baseline_v3_corrected_evaluation",
        "V3 diagnostic baseline",
    )
    evaluate_experiment(
        "baseline_v4_tuned_results",
        "baseline_v4_corrected_evaluation",
        "V4 tuned baseline",
    )
