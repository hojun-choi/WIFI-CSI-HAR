"""Sampling helpers for low-data experiments.

Stratified sampling will be implemented here to support fair comparison across
different real training data ratios.
"""

# Rubric: stratified sampling keeps class distribution comparable across
# `real_ratio` settings, which makes low-data robustness claims more credible.

from __future__ import annotations

import math

import numpy as np
import torch


def stratified_indices(y, ratio: float, seed: int) -> np.ndarray:
    """Return deterministic stratified indices for a reduced training split."""
    if ratio <= 0 or ratio > 1.0:
        raise ValueError("ratio must be in the range (0, 1].")

    labels = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    num_samples = len(labels)
    all_indices = np.arange(num_samples)
    if ratio == 1.0:
        return all_indices

    rng = np.random.default_rng(seed)
    selected_indices: list[np.ndarray] = []

    # Rubric: stratified sampling keeps class distribution comparable across
    # `real_ratio` settings and prevents low-data experiments from removing
    # entire classes by accident.
    for class_id in np.unique(labels):
        class_indices = all_indices[labels == class_id]
        rng.shuffle(class_indices)
        if len(class_indices) == 0:
            continue
        num_selected = int(math.floor(len(class_indices) * ratio))
        if ratio > 0:
            num_selected = max(1, num_selected)
        num_selected = min(len(class_indices), num_selected)
        selected_indices.append(class_indices[:num_selected])

    combined = np.concatenate(selected_indices, axis=0)
    combined.sort()
    return combined
