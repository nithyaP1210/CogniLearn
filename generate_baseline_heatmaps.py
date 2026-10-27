
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

RESULTS = {
    "V3 Diagnostic Baseline": "baseline_v3_corrected_evaluation",
    "V4 Tuned Baseline": "baseline_v4_corrected_evaluation",
}

TASKS = [
    "Task 1\nAirplane, Auto",
    "Task 2\nBird, Cat",
    "Task 3\nDeer, Dog",
    "Task 4\nFrog, Horse",
    "Task 5\nShip, Truck",
]


def load_matrix(folder):
    path = os.path.join(folder, "task_accuracy.csv")

    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))

    matrix = np.full((5, 5), np.nan)

    for row in rows:
        stage = int(row["stage"]) - 1
        task = int(row["tested_task"]) - 1

        # Class-incremental accuracy for this task after this stage
        matrix[stage, task] = float(
            row["class_incremental_accuracy_percent"]
        )

    return matrix


for title, folder in RESULTS.items():
    if not os.path.isfile(os.path.join(folder, "task_accuracy.csv")):
        print(f"CSV not found in {folder}")
        continue

    matrix = load_matrix(folder)

    fig, ax = plt.subplots(figsize=(9, 6))
    masked = np.ma.masked_invalid(matrix)

    image = ax.imshow(masked, cmap="YlOrRd", vmin=0, vmax=100)

    ax.set_title(f"{title}\nClass-Incremental Accuracy by Task")
    ax.set_xlabel("Task being evaluated")
    ax.set_ylabel("Model checkpoint after learning")

    ax.set_xticks(range(5))
    ax.set_xticklabels(TASKS)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"After Task {i}" for i in range(1, 6)])

    for r in range(5):
        for c in range(5):
            if not np.isnan(matrix[r, c]):
                ax.text(
                    c, r, f"{matrix[r, c]:.1f}%",
                    ha="center", va="center", color="black"
                )

    fig.colorbar(image, ax=ax, label="Accuracy (%)")
    fig.tight_layout()

    output_path = os.path.join(folder, "class_incremental_heatmap.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heatmap: {output_path}")

print("Heatmap generation complete.")
