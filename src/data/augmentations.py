"""Train-only augmentation utilities for low-data recovery experiments.

Keeping augmentation separate makes it easier to prove that validation and test
data remain untouched, which prevents evaluation leakage.
"""

# Rubric: train-only augmentation prevents evaluation leakage and supports a
# defensible comparison between augmented and non-augmented low-data settings.

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from src.data.sampling import stratified_indices


DEFAULT_AUGMENTATION_CONFIG: dict[str, Any] = {
    "gaussian_noise_std": 0.02,
    "time_mask_ratio": 0.10,
    "subcarrier_mask_ratio": 0.10,
    "max_time_shift": 5,
    "scaling_min": 0.9,
    "scaling_max": 1.1,
    "use_gaussian_noise": True,
    "use_time_mask": True,
    "use_subcarrier_mask": True,
    "use_time_shift": True,
    "use_scaling": True,
}


def resolve_augmentation_config(
    augmentation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge user overrides with the project defaults."""
    resolved = deepcopy(DEFAULT_AUGMENTATION_CONFIG)
    if augmentation_config:
        resolved.update(augmentation_config)
    return resolved


def _mask_length(size: int, ratio: float) -> int:
    if ratio <= 0:
        return 0
    return max(1, int(round(size * ratio)))


def _shift_with_zero_pad(sample: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return sample

    shifted = np.zeros_like(sample)
    if shift > 0:
        shifted[shift:] = sample[:-shift]
    else:
        shifted[:shift] = sample[-shift:]
    return shifted


class UTHARAugmenter:
    """Apply modest CSI-plausible augmentations to a single UT-HAR sample."""

    def __init__(self, augmentation_config: dict[str, Any] | None = None) -> None:
        self.config = resolve_augmentation_config(augmentation_config)

    def __call__(self, sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Rubric: augmentation is for robustness under limited real data, not
        # for altering validation/test evaluation conditions.
        augmented = np.asarray(sample, dtype=np.float32).copy()

        if self.config.get("use_scaling", True):
            scale = rng.uniform(
                float(self.config["scaling_min"]),
                float(self.config["scaling_max"]),
            )
            augmented *= np.float32(scale)

        if self.config.get("use_time_shift", True):
            max_shift = int(self.config["max_time_shift"])
            if max_shift > 0:
                shift = int(rng.integers(-max_shift, max_shift + 1))
                augmented = _shift_with_zero_pad(augmented, shift)

        if self.config.get("use_time_mask", True):
            time_mask_len = _mask_length(
                augmented.shape[0],
                float(self.config["time_mask_ratio"]),
            )
            if time_mask_len > 0:
                start = int(rng.integers(0, max(1, augmented.shape[0] - time_mask_len + 1)))
                augmented[start : start + time_mask_len, :] = 0.0

        if self.config.get("use_subcarrier_mask", True):
            feature_mask_len = _mask_length(
                augmented.shape[1],
                float(self.config["subcarrier_mask_ratio"]),
            )
            if feature_mask_len > 0:
                start = int(rng.integers(0, max(1, augmented.shape[1] - feature_mask_len + 1)))
                augmented[:, start : start + feature_mask_len] = 0.0

        if self.config.get("use_gaussian_noise", True):
            noise_std = float(self.config["gaussian_noise_std"])
            if noise_std > 0:
                noise = rng.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)
                augmented = augmented + noise

        return augmented.astype(np.float32, copy=False)


def make_augmented_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    augmentation_add_ratio: float,
    seed: int,
    augmentation_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create offline appended synthetic training samples from the selected train subset."""
    X_real = np.asarray(X_train, dtype=np.float32)
    y_real = np.asarray(y_train, dtype=np.int64)

    if augmentation_add_ratio < 0:
        raise ValueError("augmentation_add_ratio must be non-negative.")

    resolved_config = resolve_augmentation_config(augmentation_config)
    metadata = {
        "augmentation_mode": "offline_append",
        "real_train_size": int(len(y_real)),
        "synthetic_train_size": 0,
        "effective_train_size": int(len(y_real)),
        "augmentation_add_ratio": float(augmentation_add_ratio),
        "augmentation_config": resolved_config,
        "synthetic_generation_seed": int(seed),
    }

    if augmentation_add_ratio == 0.0 or len(y_real) == 0:
        return X_real, y_real, metadata

    augmenter = UTHARAugmenter(resolved_config)
    rng = np.random.default_rng(seed)
    synthetic_X_chunks: list[np.ndarray] = []
    synthetic_y_chunks: list[np.ndarray] = []

    full_copies = int(np.floor(augmentation_add_ratio))
    remainder_ratio = float(augmentation_add_ratio - full_copies)

    def _augment_subset(subset_X: np.ndarray, subset_y: np.ndarray) -> None:
        if len(subset_y) == 0:
            return
        augmented_samples = [
            augmenter(np.asarray(sample, dtype=np.float32), rng) for sample in subset_X
        ]
        synthetic_X_chunks.append(np.asarray(augmented_samples, dtype=np.float32))
        synthetic_y_chunks.append(np.asarray(subset_y, dtype=np.int64))

    for _ in range(full_copies):
        _augment_subset(X_real, y_real)

    if remainder_ratio > 0:
        selected_indices = stratified_indices(y_real, ratio=remainder_ratio, seed=seed + full_copies + 1)
        _augment_subset(X_real[selected_indices], y_real[selected_indices])

    if not synthetic_X_chunks:
        return X_real, y_real, metadata

    X_synthetic = np.concatenate(synthetic_X_chunks, axis=0).astype(np.float32, copy=False)
    y_synthetic = np.concatenate(synthetic_y_chunks, axis=0).astype(np.int64, copy=False)
    X_effective = np.concatenate([X_real, X_synthetic], axis=0).astype(np.float32, copy=False)
    y_effective = np.concatenate([y_real, y_synthetic], axis=0).astype(np.int64, copy=False)

    metadata["synthetic_train_size"] = int(len(y_synthetic))
    metadata["effective_train_size"] = int(len(y_effective))
    return X_effective, y_effective, metadata
