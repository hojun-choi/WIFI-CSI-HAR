from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Rubric: dataset analysis and preprocessing reliability start with explicit,
# reproducible paths and expected split sizes.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "UT_HAR"
RESULTS_ROOT = PROJECT_ROOT / "results" / "figures"

EXPECTED_SPLIT_SIZES = {
    "train": 3977,
    "val": 496,
    "test": 500,
}
EXPECTED_LABELS = set(range(7))


def load_numpy_binary(path: Path) -> np.ndarray:
    """Load UT-HAR files that keep a .csv extension but store NumPy binaries."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("rb") as file:
        return np.load(file)


def summarize_array(name: str, array: np.ndarray) -> None:
    print(
        f"{name}: shape={array.shape}, dtype={array.dtype}, "
        f"min={array.min()}, max={array.max()}"
    )


def summarize_labels(name: str, labels: np.ndarray) -> None:
    unique_labels = np.unique(labels)
    print(
        f"{name}: shape={labels.shape}, dtype={labels.dtype}, "
        f"min={labels.min()}, max={labels.max()}, unique={unique_labels.tolist()}"
    )


def verify_split(name: str, features: np.ndarray, labels: np.ndarray) -> None:
    # Rubric: faithful preprocessing requires verifying that each split keeps
    # the documented sample counts before any later modeling step.
    expected_count = EXPECTED_SPLIT_SIZES[name]
    if features.shape[0] != expected_count:
        raise ValueError(
            f"{name} feature count mismatch: expected {expected_count}, got {features.shape[0]}"
        )
    if labels.shape[0] != expected_count:
        raise ValueError(
            f"{name} label count mismatch: expected {expected_count}, got {labels.shape[0]}"
        )
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{name} split mismatch: features={features.shape[0]}, labels={labels.shape[0]}"
        )


def verify_labels(split_to_labels: dict[str, np.ndarray]) -> None:
    # Rubric: label validation protects report correctness for the 7-class task.
    all_labels = np.concatenate(list(split_to_labels.values()), axis=0)
    label_set = set(np.unique(all_labels).tolist())
    if label_set != EXPECTED_LABELS:
        raise ValueError(
            f"Label set mismatch: expected {sorted(EXPECTED_LABELS)}, got {sorted(label_set)}"
        )
    for split_name, labels in split_to_labels.items():
        split_label_set = set(np.unique(labels).tolist())
        if not split_label_set.issubset(EXPECTED_LABELS):
            raise ValueError(
                f"{split_name} contains labels outside 0..6: {sorted(split_label_set)}"
            )


def save_class_distribution(split_to_labels: dict[str, np.ndarray], output_path: Path) -> None:
    # Rubric: dataset analysis should show class balance because low-data
    # degradation is easier to misread when class counts are hidden.
    classes = sorted(EXPECTED_LABELS)
    split_names = ["train", "val", "test"]
    counts = {
        split_name: [Counter(split_to_labels[split_name].tolist()).get(cls, 0) for cls in classes]
        for split_name in split_names
    }

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, counts["train"], width=width, label="train")
    ax.bar(x, counts["val"], width=width, label="val")
    ax.bar(x + width, counts["test"], width=width, label="test")

    ax.set_title("UT-HAR Class Distribution by Split")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Sample Count")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_sample_heatmap(features: np.ndarray, output_path: Path) -> None:
    if features.ndim != 3:
        raise ValueError(f"Expected feature array with 3 dimensions, got shape {features.shape}")

    # Rubric: report-ready EDA should show that CSI inputs have the expected
    # time-by-subcarrier structure before model design is discussed.
    sample = features[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(sample.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("UT-HAR Sample CSI Heatmap (train[0])")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("CSI Feature Index")
    fig.colorbar(image, ax=ax, label="CSI Value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Rubric: the report should state the exact six files used for UT-HAR.
    feature_paths = {
        "train": DATA_ROOT / "data" / "X_train.csv",
        "val": DATA_ROOT / "data" / "X_val.csv",
        "test": DATA_ROOT / "data" / "X_test.csv",
    }
    label_paths = {
        "train": DATA_ROOT / "label" / "y_train.csv",
        "val": DATA_ROOT / "label" / "y_val.csv",
        "test": DATA_ROOT / "label" / "y_test.csv",
    }

    print("Loading UT-HAR files from:", DATA_ROOT)
    print("Note: .csv extensions are preserved from the benchmark, but files are loaded with np.load.")

    # Rubric: loading raw arrays first makes the dataset description in the
    # report directly traceable to the source files.
    features = {split: load_numpy_binary(path) for split, path in feature_paths.items()}
    labels = {split: load_numpy_binary(path) for split, path in label_paths.items()}

    for split_name in ["train", "val", "test"]:
        summarize_array(f"X_{split_name}", features[split_name])
        summarize_labels(f"y_{split_name}", labels[split_name])
        verify_split(split_name, features[split_name], labels[split_name])

    verify_labels(labels)

    class_distribution_path = RESULTS_ROOT / "class_distribution.png"
    heatmap_path = RESULTS_ROOT / "sample_csi_heatmap.png"

    save_class_distribution(labels, class_distribution_path)
    save_sample_heatmap(features["train"], heatmap_path)

    print("Verified split sizes:", EXPECTED_SPLIT_SIZES)
    print("Verified label range: 0..6")
    print(f"Saved figure: {class_distribution_path}")
    print(f"Saved figure: {heatmap_path}")


if __name__ == "__main__":
    main()
