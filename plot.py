
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 1. Project folder
PROJECT_DIR = Path(r"C:\Users\USER\Desktop\CogniLearn")

# 2. Find the existing evaluation CSV anywhere inside the project
csv_matches = list(PROJECT_DIR.rglob("all_baseline_results.csv"))

if not csv_matches:
    print("ERROR: all_baseline_results.csv was not found.")
    print("Searched inside:", PROJECT_DIR)
    print("Please check whether the evaluation CSV exists.")
    raise SystemExit

csv_path = csv_matches[0]
print("Using results file:", csv_path)

# 3. Read the evaluation results
df = pd.read_csv(csv_path)

required_columns = {
    "version",
    "stage",
    "class_incremental_accuracy_percent"
}

if not required_columns.issubset(df.columns):
    print("ERROR: The CSV does not contain the required columns.")
    print("Columns found:", list(df.columns))
    raise SystemExit

# 4. Create the output folder inside CogniLearn
output_folder = PROJECT_DIR / "baseline_plots"
output_folder.mkdir(parents=True, exist_ok=True)

# 5. Line graph: accuracy as more tasks are learned
plt.figure(figsize=(9, 5))

for version in ["V3", "V4"]:
    data = df[df["version"] == version].copy()

    # The class-incremental score repeats for each evaluated task.
    data = data.groupby("stage", as_index=False)[
        "class_incremental_accuracy_percent"
    ].first()

    if not data.empty:
        plt.plot(
            data["stage"],
            data["class_incremental_accuracy_percent"],
            marker="o",
            label=version
        )

plt.title("Class-Incremental Accuracy Across Tasks")
plt.xlabel("Number of Tasks Learned")
plt.ylabel("Accuracy (%)")
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_folder / "1_accuracy_by_stage.png", dpi=300)
plt.close()

# 6. Bar graph: final accuracy of each version
max_stage = df.groupby("version")["stage"].max()

final_data = df[
    df.apply(
        lambda row: row["stage"] == max_stage.get(row["version"]),
        axis=1
    )
].copy()

final_data = final_data.groupby("version", as_index=False)[
    "class_incremental_accuracy_percent"
].first()

plt.figure(figsize=(8, 5))
bars = plt.bar(
    final_data["version"],
    final_data["class_incremental_accuracy_percent"]
)

plt.title("Final Accuracy Comparison")
plt.xlabel("Model Version")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(output_folder / "2_final_accuracy_comparison.png", dpi=300)
plt.close()

print("\nPlots created successfully!")
print("Output folder:", output_folder)
print("1_accuracy_by_stage.png")
print("2_final_accuracy_comparison.png")
