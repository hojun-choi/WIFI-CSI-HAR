"""Shared helpers for preprocessing ranking and final selection."""

from __future__ import annotations

import json

import pandas as pd


SIMPLICITY_ORDER = [
    "none",
    "train_global_zscore",
    "per_sample_zscore",
    "minmax_scaling",
    "robust_scaling",
    "train_featurewise_zscore",
    "per_sample_featurewise_zscore",
    "moving_average_smoothing",
    "savgol_smoothing",
]


def preprocessing_simplicity_rank(preprocessing: str) -> tuple[int, int, int]:
    normalized = str(preprocessing)
    if "+" not in normalized:
        try:
            return (0, SIMPLICITY_ORDER.index(normalized), 0)
        except ValueError:
            return (0, len(SIMPLICITY_ORDER) + 10, 0)

    steps = [step.strip() for step in normalized.split("+") if step.strip()]
    component_ranks = []
    for step in steps:
        try:
            component_ranks.append(SIMPLICITY_ORDER.index(step))
        except ValueError:
            component_ranks.append(len(SIMPLICITY_ORDER) + 10)
    return (1, len(steps), sum(component_ranks))


def apply_single_seed_ranking(frame: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, pd.Series]:
    ranked = frame.copy()
    for column in ["val_macro_f1", "test_macro_f1", "val_accuracy", "test_accuracy"]:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked["simplicity_rank"] = ranked["preprocessing"].map(preprocessing_simplicity_rank)
    ranked = ranked.sort_values(
        by=["val_macro_f1", "val_accuracy", "simplicity_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_val = float(ranked.iloc[0]["val_macro_f1"])
    within_tolerance = ranked[ranked["val_macro_f1"] >= best_val - tolerance].copy()
    selected = within_tolerance.sort_values(
        by=["simplicity_rank", "val_macro_f1", "val_accuracy"],
        ascending=[True, False, False],
    ).iloc[0]
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked, selected


def apply_stability_ranking(
    frame: pd.DataFrame,
    close_tolerance: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    ranked = frame.copy()
    for column in [
        "mean_val_macro_f1",
        "std_val_macro_f1",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_val_accuracy",
        "mean_test_accuracy",
        "mean_val_test_macro_f1_gap",
    ]:
        if column in ranked.columns:
            ranked[column] = pd.to_numeric(ranked[column], errors="coerce")

    ranked["simplicity_rank"] = ranked["preprocessing"].map(preprocessing_simplicity_rank)
    raw_best = ranked.sort_values(
        by=["mean_val_macro_f1", "std_val_macro_f1", "simplicity_rank"],
        ascending=[False, True, True],
    ).iloc[0]
    best_mean = float(raw_best["mean_val_macro_f1"])

    ranked["within_close_tolerance"] = ranked["mean_val_macro_f1"] >= best_mean - close_tolerance
    close_candidates = ranked[ranked["within_close_tolerance"]].copy()
    selected = close_candidates.sort_values(
        by=["std_val_macro_f1", "simplicity_rank", "mean_val_macro_f1"],
        ascending=[True, True, False],
    ).iloc[0]

    close_ranked = close_candidates.sort_values(
        by=["std_val_macro_f1", "simplicity_rank", "mean_val_macro_f1"],
        ascending=[True, True, False],
    )
    remaining_ranked = ranked[~ranked["within_close_tolerance"]].sort_values(
        by=["mean_val_macro_f1", "std_val_macro_f1", "simplicity_rank"],
        ascending=[False, True, True],
    )
    final_ranked = pd.concat([close_ranked, remaining_ranked], ignore_index=True)
    final_ranked.insert(0, "selection_rank", range(1, len(final_ranked) + 1))
    return final_ranked, selected, raw_best


def serialize_seed_list(seeds: list[int]) -> str:
    return json.dumps(list(seeds))
