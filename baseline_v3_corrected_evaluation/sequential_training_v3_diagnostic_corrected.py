
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
EPOCHS_PER_TASK = 2
TRAIN_PER_CLASS = 1000
TEST_PER_CLASS = 200
LEARNING_RATE = 0.001

OUTPUT_DIR = "baseline_v3_corrected_training_results"
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
print("COGNILEARN - RESNET18 BASELINE V3")
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
    """Select the same reproducible examples for each class every time.