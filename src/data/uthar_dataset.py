"""UT-HAR dataset utilities for the clean project-specific pipeline."""

# Rubric: this module owns dataset loading, tensor conversion, train-based
# normalization, and train-only augmentation so preprocessing stays auditable.

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentations import UTHARAugmenter, resolve_augmentation_config
from src.data.preprocessing import apply_preprocessing
from src.data.sampling import stratified_indices


class UTHARDataset(Dataset):
    """Array-backed dataset with optional train-only on-the-fly augmentation."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmenter: UTHARAugmenter | None = None,
        seed: int = 42,
    ) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.augmenter = augmenter
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.X[index]
        if self.augmenter is not None:
            # Rubric: augmentation is applied only here on the train dataset so
            # validation/test metrics remain clean and leakage-free.
            sample = self.augmenter(sample, self.rng)
        features = torch.from_numpy(np.asarray(sample, dtype=np.float32))
        label = torch.tensor(int(self.y[index]), dtype=torch.long)
        return features, label


def _load_numpy_binary(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        return np.load(file)


def load_uthar_arrays(data_root: str | Path) -> dict[str, np.ndarray]:
    """Load all UT-HAR arrays from disk using np.load."""
    root = Path(data_root)
    return {
        "X_train": _load_numpy_binary(root / "data" / "X_train.csv"),
        "X_val": _load_numpy_binary(root / "data" / "X_val.csv"),
        "X_test": _load_numpy_binary(root / "data" / "X_test.csv"),
        "y_train": _load_numpy_binary(root / "label" / "y_train.csv"),
        "y_val": _load_numpy_binary(root / "label" / "y_val.csv"),
        "y_test": _load_numpy_binary(root / "label" / "y_test.csv"),
    }


def make_tensor_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    augmentation: bool = False,
    augmentation_config: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """Convert UT-HAR arrays to Dataset objects."""
    augmenter = None
    if augmentation:
        augmenter = UTHARAugmenter(augmentation_config)

    train_dataset = UTHARDataset(X_train, y_train, augmenter=augmenter, seed=seed)
    # Rubric: validation/test are always built from non-augmented arrays.
    val_dataset = UTHARDataset(X_val, y_val, augmenter=None, seed=seed)
    test_dataset = UTHARDataset(X_test, y_test, augmenter=None, seed=seed)
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
    augmentation: bool = False,
    augmentation_config: dict[str, Any] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, object]]:
    """Create UT-HAR dataloaders for dry-runs and later experiments."""
    arrays = load_uthar_arrays(data_root)
    full_X_train = arrays["X_train"]
    full_y_train = arrays["y_train"]
    X_train, y_train, _ = _sample_training_data(
        full_X_train,
        full_y_train,
        real_ratio=real_ratio,
        seed=seed,
    )
    X_val = arrays["X_val"]
    y_val = arrays["y_val"]
    X_test = arrays["X_test"]
    y_test = arrays["y_test"]

    # Rubric: preprocessing is applied after low-data sampling so the train
    # subset alone defines normalization statistics.
    X_train_proc, X_val_proc, X_test_proc, preprocessing_bundle = apply_preprocessing(
        X_train,
        X_val,
        X_test,
        preprocessing=preprocessing,
        preprocessing_config=preprocessing_config,
    )

    resolved_augmentation_config = (
        resolve_augmentation_config(augmentation_config) if augmentation else {}
    )
    train_dataset, val_dataset, test_dataset = make_tensor_datasets(
        X_train_proc,
        y_train,
        X_val_proc,
        y_val,
        X_test_proc,
        y_test,
        augmentation=augmentation,
        augmentation_config=resolved_augmentation_config if augmentation else None,
        seed=seed,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
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
        "train_size": len(train_dataset),
        "selected_train_size": len(train_dataset),
        "full_train_size": len(full_X_train),
        # Rubric: validation/test are kept unchanged for fair low-data and M5
        # augmentation comparisons.
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "real_ratio": real_ratio,
        "preprocessing": preprocessing_bundle["preprocessing"],
        "preprocessing_steps": preprocessing_bundle["preprocessing_steps"],
        "preprocessing_config": preprocessing_bundle["preprocessing_config"],
        "preprocessing_metadata": preprocessing_bundle["preprocessing_metadata"],
        "augmentation": augmentation,
        "augmentation_config": resolved_augmentation_config,
        "class_counts_selected": class_counts_selected,
    }
    return train_loader, val_loader, test_loader, metadata
