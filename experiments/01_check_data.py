from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.uthar_labels import UT_HAR_CLASS_NAMES, UT_HAR_CLASS_ORDER
from src.visualization import (
    save_sample_csi_heatmap,
    save_sample_csi_lineplot,
    save_sample_heatmap_by_activity,
    save_split_size_summary,
    save_uthar_class_distribution_by_activity,
)

DATA_ROOT = PROJECT_ROOT / "data" / "UT_HAR"
RESULTS_ROOT = PROJECT_ROOT / "results" / "figures"

EXPECTED_SPLIT_SIZES = {
    "train": 3977,
    "val": 496,
    "test": 500,
}
ORIGINAL_README_SPLIT_INFO = {
    "train": 3977,
    "test_total": 996,
}
EXPECTED_LABELS = set(UT_HAR_CLASS_ORDER)


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
    unique_names = [UT_HAR_CLASS_NAMES[int(label)] for label in unique_labels]
    print(
        f"{name}: shape={labels.shape}, dtype={labels.dtype}, "
        f"min={labels.min()}, max={labels.max()}, unique={unique_labels.tolist()} ({unique_names})"
    )


def verify_split(name: str, features: np.ndarray, labels: np.ndarray) -> None:
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
    if features.ndim != 3 or tuple(features.shape[1:]) != (250, 90):
        raise ValueError(
            f"{name} feature shape mismatch: expected (*, 250, 90), got {features.shape}"
        )


def verify_labels(split_to_labels: dict[str, np.ndarray]) -> None:
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


def class_counts_for_split(labels: np.ndarray) -> list[int]:
    counter = Counter(labels.tolist())
    return [int(counter.get(class_id, 0)) for class_id in UT_HAR_CLASS_ORDER]


def first_sample_per_activity(features: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    samples: dict[int, np.ndarray] = {}
    for sample, label in zip(features, labels):
        class_id = int(label)
        if class_id not in samples:
            samples[class_id] = sample
        if len(samples) == len(UT_HAR_CLASS_ORDER):
            break
    return samples


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

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
    print("UT-HAR class names:")
    for class_id in UT_HAR_CLASS_ORDER:
        print(f"- {class_id}: {UT_HAR_CLASS_NAMES[class_id]}")

    features = {split: load_numpy_binary(path) for split, path in feature_paths.items()}
    labels = {split: load_numpy_binary(path) for split, path in label_paths.items()}

    for split_name in ["train", "val", "test"]:
        summarize_array(f"X_{split_name}", features[split_name])
        summarize_labels(f"y_{split_name}", labels[split_name])
        verify_split(split_name, features[split_name], labels[split_name])

    verify_labels(labels)

    split_sizes = {split_name: int(features[split_name].shape[0]) for split_name in ["train", "val", "test"]}
    split_to_counts = {
        split_name: class_counts_for_split(labels[split_name])
        for split_name in ["train", "val", "test"]
    }
    per_activity_samples = first_sample_per_activity(features["train"], labels["train"])

    class_distribution_path = RESULTS_ROOT / "class_distribution_by_activity.png"
    split_summary_path = RESULTS_ROOT / "split_size_summary.png"
    heatmap_path = RESULTS_ROOT / "sample_csi_heatmap.png"
    lineplot_path = RESULTS_ROOT / "sample_csi_lineplot.png"
    heatmap_grid_path = RESULTS_ROOT / "sample_heatmap_by_activity.png"

    save_uthar_class_distribution_by_activity(split_to_counts, class_distribution_path)
    save_split_size_summary(split_sizes, split_summary_path)
    save_sample_csi_heatmap(features["train"][0], heatmap_path)
    save_sample_csi_lineplot(features["train"][0], lineplot_path)
    save_sample_heatmap_by_activity(per_activity_samples, heatmap_grid_path)

    print("Verified split sizes:", EXPECTED_SPLIT_SIZES)
    print(
        "Original UT-HAR README summary: "
        f"train={ORIGINAL_README_SPLIT_INFO['train']}, "
        f"test_total={ORIGINAL_README_SPLIT_INFO['test_total']}"
    )
    print("Actual loaded split sizes:", split_sizes)
    print("Verified sample shape: (250, 90)")
    print("Interpretation: one sample = 250 CSI frame indices x 90 CSI features")
    print("Time note: timestep is CSI frame index, not directly seconds.")
    print("If sampling_rate is fs Hz, one sample duration is 250 / fs seconds.")
    print("100Hz conversion is illustrative only and not confirmed ground truth.")
    print(f"Saved figure: {class_distribution_path}")
    print(f"Saved figure: {split_summary_path}")
    print(f"Saved figure: {heatmap_path}")
    print(f"Saved figure: {lineplot_path}")
    print(f"Saved figure: {heatmap_grid_path}")


if __name__ == "__main__":
    main()
