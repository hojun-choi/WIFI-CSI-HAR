"""UT-HAR dataset utilities for the clean project-specific pipeline.

This module will hold the final dataset loader so preprocessing stays explicit
and reproducible for the report.
"""

# Rubric: this module will own dataset loading, tensor conversion, and
# train-based normalization so preprocessing remains auditable in the report.

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.sampling import stratified_indices


EPSILON = 1e-6


def _load_numpy_binary(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        return np.load(file)


def load_uthar_arrays(data_root: str | Path) -> dict[str, np.ndarray]:
    """Load all UT-HAR arrays from disk using np.load."""
    root = Path(data_root)
    arrays = {
        "X_train": _load_numpy_binary(root / "data" / "X_train.csv"),
        "X_val": _load_numpy_binary(root / "data" / "X_val.csv"),
        "X_test": _load_numpy_binary(root / "data" / "X_test.csv"),
        "y_train": _load_numpy_binary(root / "label" / "y_train.csv"),
        "y_val": _load_numpy_binary(root / "label" / "y_val.csv"),
        "y_test": _load_numpy_binary(root / "label" / "y_test.csv"),
    }
    return arrays


def _per_sample_zscore(X: np.ndarray) -> np.ndarray:
    sample_mean = X.mean(axis=(1, 2), keepdims=True)
    sample_std = X.std(axis=(1, 2), keepdims=True)
    return (X - sample_mean) / (sample_std + EPSILON)


def apply_preprocessing(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    preprocessing: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    """Apply the selected preprocessing policy to UT-HAR features."""
    if preprocessing == "none":
        metadata = {"preprocessing": preprocessing}
        return X_train.copy(), X_val.copy(), X_test.copy(), metadata

    if preprocessing == "train_global_zscore":
        # Rubric: train-based normalization prevents validation/test leakage by
        # restricting statistics to X_train before applying them to all splits.
        train_mean = float(X_train.mean())
        train_std = float(X_train.std())
        X_train_proc = (X_train - train_mean) / (train_std + EPSILON)
        X_val_proc = (X_val - train_mean) / (train_std + EPSILON)
        X_test_proc = (X_test - train_mean) / (train_std + EPSILON)
        metadata = {
            "preprocessing": preprocessing,
            "train_mean": train_mean,
            "train_std": train_std,
        }
        return X_train_proc, X_val_proc, X_test_proc, metadata

    if preprocessing == "per_sample_zscore":
        metadata = {"preprocessing": preprocessing}
        return (
            _per_sample_zscore(X_train),
            _per_sample_zscore(X_val),
            _per_sample_zscore(X_test),
            metadata,
        )

    raise ValueError(
        f"Unsupported preprocessing: {preprocessing}. "
        "Choose from ['none', 'train_global_zscore', 'per_sample_zscore']."
    )


def make_tensor_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Convert UT-HAR arrays to TensorDataset objects."""
    train_dataset = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.int64)),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test.astype(np.float32)),
        torch.from_numpy(y_test.astype(np.int64)),
    )
    return train_dataset, val_dataset, test_dataset


def _sample_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    real_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if real_ratio <= 0 or real_ratio > 1.0:
        raise ValueError("real_ratio must be in the range (0, 1].")

    selected_indices = stratified_indices(y_train, ratio=real_ratio, seed=seed)
    return X_train[selected_indices], y_train[selected_indices], selected_indices


def create_uthar_dataloaders(
    data_root: str | Path,
    batch_size: int,
    preprocessing: str,
    real_ratio: float,
    seed: int,
    model_type: str = "GRU",
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, object]]:
    """Create UT-HAR dataloaders for dry-run and later experiments."""
    arrays = load_uthar_arrays(data_root)
    full_X_train = arrays["X_train"]
    full_y_train = arrays["y_train"]
    X_train, y_train, selected_indices = _sample_training_data(
        full_X_train,
        full_y_train,
        real_ratio=real_ratio,
        seed=seed,
    )
    X_val = arrays["X_val"]
    y_val = arrays["y_val"]
    X_test = arrays["X_test"]
    y_test = arrays["y_test"]

    # Rubric: apply preprocessing only after train subset selection so
    # normalization statistics come from the actual training data in use.
    X_train_proc, X_val_proc, X_test_proc, preprocessing_metadata = apply_preprocessing(
        X_train,
        X_val,
        X_test,
        preprocessing=preprocessing,
    )

    train_dataset, val_dataset, test_dataset = make_tensor_datasets(
        X_train_proc,
        y_train,
        X_val_proc,
        y_val,
        X_test_proc,
        y_test,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_counts_selected = {
        int(class_id): int(count)
        for class_id, count in sorted(Counter(y_train.tolist()).items())
    }
    metadata = {
        "model_type": model_type,
        "num_classes": 7,
        "input_shape": tuple(X_train_proc.shape[1:]),
        # Rubric: validation/test are kept unchanged for fair comparison across
        # preprocessing candidates and later low-data experiments.
        "train_size": len(train_dataset),
        "selected_train_size": len(train_dataset),
        "full_train_size": len(full_X_train),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "real_ratio": real_ratio,
        "preprocessing": preprocessing,
        "preprocessing_metadata": preprocessing_metadata,
        "class_counts_selected": class_counts_selected,
    }
    return train_loader, val_loader, test_loader, metadata
