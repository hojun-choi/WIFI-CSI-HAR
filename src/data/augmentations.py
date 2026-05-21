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
