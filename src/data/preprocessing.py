"""Preprocessing helpers for UT-HAR arrays with auditable train-only fitting."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.signal import savgol_filter


EPSILON = 1e-6

RAW_ALIASES = {"none", "raw"}
SUPPORTED_PREPROCESSING_STEPS = [
    "none",
    "train_global_zscore",
    "per_sample_zscore",
    "minmax_scaling",
    "robust_scaling",
    "savgol_smoothing",
    "moving_average_smoothing",
    "train_featurewise_zscore",
    "per_sample_featurewise_zscore",
]

DEFAULT_PREPROCESSING_CONFIG: dict[str, Any] = {
    "savgol_smoothing": {
        "window_length": 11,
        "polyorder": 2,
        "mode": "interp",
    },
    "moving_average_smoothing": {
        "window_size": 5,
        "padding_mode": "edge",
    },
}


def _as_float32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def _validate_uthar_shape(array: np.ndarray, name: str) -> None:
    if array.ndim != 3 or array.shape[1:] != (250, 90):
        raise ValueError(f"{name} must have shape (N, 250, 90), got {array.shape}")


def _canonicalize_step_name(step: str) -> str:
    normalized = step.strip()
    if not normalized:
        raise ValueError("Empty preprocessing step is not allowed.")
    lowered = normalized.lower()
    if lowered in RAW_ALIASES:
        return "none"
    if lowered not in SUPPORTED_PREPROCESSING_STEPS:
        raise ValueError(
            f"Unsupported preprocessing step: {step}. "
            f"Choose from {SUPPORTED_PREPROCESSING_STEPS}."
        )
    return lowered


def parse_preprocessing_pipeline(preprocessing: str) -> list[str]:
    if preprocessing is None:
        raise ValueError("preprocessing must be a non-empty string.")
    steps = [_canonicalize_step_name(step) for step in preprocessing.split("+")]
    return steps if steps else ["none"]


def resolve_preprocessing_config(
    preprocessing_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = deepcopy(DEFAULT_PREPROCESSING_CONFIG)
    if preprocessing_config:
        for key, value in preprocessing_config.items():
            if isinstance(value, dict) and isinstance(resolved.get(key), dict):
                merged = deepcopy(resolved[key])
                merged.update(value)
                resolved[key] = merged
            else:
                resolved[key] = deepcopy(value)
    return resolved


def _per_sample_zscore(X: np.ndarray) -> np.ndarray:
    sample_mean = X.mean(axis=(1, 2), keepdims=True)
    sample_std = X.std(axis=(1, 2), keepdims=True)
    return _as_float32((X - sample_mean) / (sample_std + EPSILON))


def _per_sample_featurewise_zscore(X: np.ndarray) -> np.ndarray:
    sample_mean = X.mean(axis=1, keepdims=True)
    sample_std = X.std(axis=1, keepdims=True)
    return _as_float32((X - sample_mean) / (sample_std + EPSILON))


def _moving_average_1d(values: np.ndarray, window_size: int, padding_mode: str) -> np.ndarray:
    if window_size <= 1:
        return values.astype(np.float32, copy=False)

    left = window_size // 2
    right = window_size - 1 - left
    padded = np.pad(values, (left, right), mode=padding_mode)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32, copy=False)


def _apply_moving_average(
    X: np.ndarray,
    *,
    window_size: int,
    padding_mode: str,
) -> np.ndarray:
    smoothed = np.empty_like(X, dtype=np.float32)
    for sample_index in range(X.shape[0]):
        for feature_index in range(X.shape[2]):
            smoothed[sample_index, :, feature_index] = _moving_average_1d(
                X[sample_index, :, feature_index],
                window_size=window_size,
                padding_mode=padding_mode,
            )
    return smoothed


def _fit_step(
    step: str,
    X_train: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    if step == "none":
        return {"step": step, "type": "identity", "fit_on": "none"}

    if step == "train_global_zscore":
        return {
            "step": step,
            "type": "train_statistics",
            "fit_on": "selected_train_split_only",
            "train_mean": float(X_train.mean()),
            "train_std": float(X_train.std()),
        }

    if step == "per_sample_zscore":
        return {"step": step, "type": "per_sample", "fit_on": "none"}

    if step == "minmax_scaling":
        return {
            "step": step,
            "type": "train_statistics",
            "fit_on": "selected_train_split_only",
            "train_min": float(X_train.min()),
            "train_max": float(X_train.max()),
        }

    if step == "robust_scaling":
        train_median = float(np.median(X_train))
        q1 = float(np.percentile(X_train, 25))
        q3 = float(np.percentile(X_train, 75))
        return {
            "step": step,
            "type": "train_statistics",
            "fit_on": "selected_train_split_only",
            "train_median": train_median,
            "train_iqr": q3 - q1,
            "train_q1": q1,
            "train_q3": q3,
        }

    if step == "savgol_smoothing":
        step_config = deepcopy(config.get(step, {}))
        window_length = int(step_config.get("window_length", 11))
        polyorder = int(step_config.get("polyorder", 2))
        mode = str(step_config.get("mode", "interp"))
        if window_length <= 0 or window_length % 2 == 0:
            raise ValueError("savgol_smoothing window_length must be a positive odd integer.")
        if polyorder < 0 or polyorder >= window_length:
            raise ValueError("savgol_smoothing polyorder must be in [0, window_length).")
        return {
            "step": step,
            "type": "deterministic_filter",
            "fit_on": "none",
            "window_length": window_length,
            "polyorder": polyorder,
            "mode": mode,
        }

    if step == "moving_average_smoothing":
        step_config = deepcopy(config.get(step, {}))
        window_size = int(step_config.get("window_size", 5))
        padding_mode = str(step_config.get("padding_mode", "edge"))
        if window_size <= 0:
            raise ValueError("moving_average_smoothing window_size must be positive.")
        return {
            "step": step,
            "type": "deterministic_filter",
            "fit_on": "none",
            "window_size": window_size,
            "padding_mode": padding_mode,
        }

    if step == "train_featurewise_zscore":
        train_mean = X_train.mean(axis=(0, 1), keepdims=True)
        train_std = X_train.std(axis=(0, 1), keepdims=True)
        return {
            "step": step,
            "type": "train_feature_statistics",
            "fit_on": "selected_train_split_only",
            "train_mean": train_mean.astype(np.float32).reshape(-1).tolist(),
            "train_std": train_std.astype(np.float32).reshape(-1).tolist(),
        }

    if step == "per_sample_featurewise_zscore":
        return {"step": step, "type": "per_sample_featurewise", "fit_on": "none"}

    raise ValueError(f"Unsupported preprocessing step: {step}")


def _apply_step(X: np.ndarray, step_metadata: dict[str, Any]) -> np.ndarray:
    step = str(step_metadata["step"])
    X = _as_float32(X)

    if step == "none":
        return X.copy()

    if step == "train_global_zscore":
        train_mean = float(step_metadata["train_mean"])
        train_std = float(step_metadata["train_std"])
        return _as_float32((X - train_mean) / (train_std + EPSILON))

    if step == "per_sample_zscore":
        return _per_sample_zscore(X)

    if step == "minmax_scaling":
        train_min = float(step_metadata["train_min"])
        train_max = float(step_metadata["train_max"])
        return _as_float32((X - train_min) / ((train_max - train_min) + EPSILON))

    if step == "robust_scaling":
        train_median = float(step_metadata["train_median"])
        train_iqr = float(step_metadata["train_iqr"])
        return _as_float32((X - train_median) / (train_iqr + EPSILON))

    if step == "savgol_smoothing":
        return _as_float32(
            savgol_filter(
                X,
                window_length=int(step_metadata["window_length"]),
                polyorder=int(step_metadata["polyorder"]),
                axis=1,
                mode=str(step_metadata["mode"]),
            )
        )

    if step == "moving_average_smoothing":
        return _apply_moving_average(
            X,
            window_size=int(step_metadata["window_size"]),
            padding_mode=str(step_metadata["padding_mode"]),
        )

    if step == "train_featurewise_zscore":
        train_mean = np.asarray(step_metadata["train_mean"], dtype=np.float32).reshape(1, 1, -1)
        train_std = np.asarray(step_metadata["train_std"], dtype=np.float32).reshape(1, 1, -1)
        return _as_float32((X - train_mean) / (train_std + EPSILON))

    if step == "per_sample_featurewise_zscore":
        return _per_sample_featurewise_zscore(X)

    raise ValueError(f"Unsupported preprocessing step: {step}")


def apply_preprocessing(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    preprocessing: str,
    preprocessing_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply a single preprocessing step or `+` pipeline to UT-HAR arrays."""
    X_train_proc = _as_float32(X_train).copy()
    X_val_proc = _as_float32(X_val).copy()
    X_test_proc = _as_float32(X_test).copy()
    _validate_uthar_shape(X_train_proc, "X_train")
    _validate_uthar_shape(X_val_proc, "X_val")
    _validate_uthar_shape(X_test_proc, "X_test")

    steps = parse_preprocessing_pipeline(preprocessing)
    resolved_config = resolve_preprocessing_config(preprocessing_config)
    step_metadata_list: list[dict[str, Any]] = []

    for step in steps:
        fitted_step = _fit_step(step, X_train_proc, resolved_config)
        X_train_proc = _apply_step(X_train_proc, fitted_step)
        X_val_proc = _apply_step(X_val_proc, fitted_step)
        X_test_proc = _apply_step(X_test_proc, fitted_step)
        step_metadata_list.append(fitted_step)

    metadata = {
        "preprocessing": "+".join(steps),
        "preprocessing_steps": steps,
        "preprocessing_config": resolved_config,
        "preprocessing_metadata": {
            "pipeline": "+".join(steps),
            "steps": step_metadata_list,
            "fit_rule": "train-statistics transforms fit only on selected train split",
            "deterministic_filter_rule": (
                "deterministic smoothing/filtering steps are applied consistently "
                "to train/val/test"
            ),
        },
    }
    return X_train_proc, X_val_proc, X_test_proc, metadata
