"""Reusable plotting helpers for report and presentation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir


def _prepare_frame(df) -> pd.DataFrame:
    return pd.DataFrame(df).copy()


def _compute_zoom_limits(values: np.ndarray, y_zoom: bool) -> tuple[float, float] | None:
    numeric_values = np.asarray(values, dtype=float)
    if numeric_values.size == 0 or not y_zoom:
        return None

    y_min = max(0.0, float(numeric_values.min()) - 0.01)
    y_max = min(1.0, float(numeric_values.max()) + 0.005)
    if y_max <= y_min:
        y_max = min(1.0, y_min + 0.01)
    return y_min, y_max


def _rotate_x_labels_if_needed(ax, labels: list[str]) -> None:
    if any(len(label) > 8 for label in labels):
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")


def _annotate_bars(ax, bars, values: list[float]) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_metric_bar(
    df,
    x_col,
    y_col,
    output_path,
    title,
    xlabel,
    ylabel,
    y_zoom=True,
    value_labels=True,
    sort_by=None,
) -> None:
    """Save a bar chart tailored for high-score metric comparison."""
    ensure_dir(Path(output_path).parent)
    frame = _prepare_frame(df)
    if sort_by is not None:
        frame = frame.sort_values(sort_by, ascending=False)

    labels = frame[x_col].astype(str).tolist()
    values = frame[y_col].astype(float).to_numpy()
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(positions, values)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    zoom_limits = _compute_zoom_limits(values, y_zoom=y_zoom)
    if zoom_limits is not None:
        ax.set_ylim(*zoom_limits)

    if value_labels:
        _annotate_bars(ax, bars, values.tolist())

    _rotate_x_labels_if_needed(ax, labels)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_metric_bar(
    df,
    x_col,
    metric_cols,
    output_path,
    title,
    xlabel,
    ylabel,
    y_zoom=True,
    value_labels=True,
) -> None:
    """Save grouped metric bars for side-by-side validation/test comparison."""
    ensure_dir(Path(output_path).parent)
    frame = _prepare_frame(df)
    labels = frame[x_col].astype(str).tolist()
    positions = np.arange(len(labels))
    width = 0.8 / max(1, len(metric_cols))

    fig, ax = plt.subplots(figsize=(9, 5))
    all_values: list[float] = []
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, metric_col in enumerate(metric_cols):
        values = frame[metric_col].astype(float).to_numpy()
        all_values.extend(values.tolist())
        offset = (idx - (len(metric_cols) - 1) / 2.0) * width
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        bars = ax.bar(positions + offset, values, width=width, label=metric_col, color=color)
        if value_labels:
            _annotate_bars(ax, bars, values.tolist())

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    zoom_limits = _compute_zoom_limits(np.asarray(all_values, dtype=float), y_zoom=y_zoom)
    if zoom_limits is not None:
        ax.set_ylim(*zoom_limits)

    _rotate_x_labels_if_needed(ax, labels)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_lines(df, x_col, y_col, group_col, output_path, title, xlabel, ylabel) -> None:
    """Save a grouped line chart for report-ready trend comparison."""
    ensure_dir(Path(output_path).parent)
    frame = _prepare_frame(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    for group_value, group_df in frame.groupby(group_col):
        ordered = group_df.sort_values(x_col)
        ax.plot(
            ordered[x_col].astype(float),
            ordered[y_col].astype(float),
            marker="o",
            label=str(group_value),
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, labels, output_path, title) -> None:
    """Save a confusion matrix figure for report and presentation use."""
    ensure_dir(Path(output_path).parent)
    matrix = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.colorbar(image, ax=ax)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_gap_from_one(
    df,
    x_col,
    metric_col,
    output_path,
    title,
    ylabel="1 - score",
    use_log=False,
) -> None:
    """Save a supplementary gap plot when bounded scores are visually too similar."""
    ensure_dir(Path(output_path).parent)
    frame = _prepare_frame(df)
    labels = frame[x_col].astype(str).tolist()
    scores = frame[metric_col].astype(float).to_numpy()
    gaps = 1.0 - scores
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(positions, gaps)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    if use_log:
        ax.set_yscale("log")
    _annotate_bars(ax, bars, gaps.tolist())
    _rotate_x_labels_if_needed(ax, labels)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_baseline_original_epoch_plots(csv_path, output_dir) -> list[Path]:
    """Generate zoomed and supplementary baseline figures from the original-epoch CSV."""
    frame = pd.read_csv(csv_path)
    output_root = Path(output_dir)
    ensure_dir(output_root)

    created_files = [
        output_root / "baseline_original_epoch_val_test_macro_f1_zoomed.png",
        output_root / "baseline_original_epoch_val_test_accuracy_zoomed.png",
        output_root / "baseline_original_epoch_test_macro_f1_zoomed.png",
        output_root / "baseline_original_epoch_test_accuracy_zoomed.png",
        output_root / "baseline_original_epoch_macro_f1_gap.png",
        output_root / "baseline_original_epoch_accuracy_gap.png",
    ]

    plot_grouped_metric_bar(
        frame,
        x_col="model",
        metric_cols=["val_macro_f1", "test_macro_f1"],
        output_path=created_files[0],
        title="UT-HAR Full-data Baseline (Validation vs Test Macro F1)",
        xlabel="Model",
        ylabel="Macro F1",
        y_zoom=True,
        value_labels=True,
    )
    plot_grouped_metric_bar(
        frame,
        x_col="model",
        metric_cols=["val_accuracy", "test_accuracy"],
        output_path=created_files[1],
        title="UT-HAR Full-data Baseline (Validation vs Test Accuracy)",
        xlabel="Model",
        ylabel="Accuracy",
        y_zoom=True,
        value_labels=True,
    )
    plot_metric_bar(
        frame,
        x_col="model",
        y_col="test_macro_f1",
        output_path=created_files[2],
        title="UT-HAR Full-data Baseline (Test Macro F1)",
        xlabel="Model",
        ylabel="Test Macro F1",
        y_zoom=True,
        value_labels=True,
        sort_by="test_macro_f1",
    )
    plot_metric_bar(
        frame,
        x_col="model",
        y_col="test_accuracy",
        output_path=created_files[3],
        title="UT-HAR Full-data Baseline (Test Accuracy)",
        xlabel="Model",
        ylabel="Test Accuracy",
        y_zoom=True,
        value_labels=True,
        sort_by="test_accuracy",
    )
    plot_metric_gap_from_one(
        frame,
        x_col="model",
        metric_col="test_macro_f1",
        output_path=created_files[4],
        title="UT-HAR Full-data Baseline (Gap to 1.0 for Test Macro F1, lower is better)",
    )
    plot_metric_gap_from_one(
        frame,
        x_col="model",
        metric_col="test_accuracy",
        output_path=created_files[5],
        title="UT-HAR Full-data Baseline (Gap to 1.0 for Test Accuracy, lower is better)",
    )
    return created_files


def save_preprocessing_ablation_plot(csv_path, output_path) -> list[Path]:
    """Create zoomed preprocessing ablation figures from the saved CSV output."""
    frame = pd.read_csv(csv_path)
    main_output = Path(output_path)
    grouped_output = main_output.parent / "preprocessing_ablation_val_test_macro_f1_zoomed.png"

    plot_metric_bar(
        frame,
        x_col="preprocessing",
        y_col="val_macro_f1",
        output_path=main_output,
        title="Preprocessing Ablation on UT-HAR (Validation Macro F1)",
        xlabel="Preprocessing",
        ylabel="Validation Macro F1",
        y_zoom=True,
        value_labels=True,
        sort_by="val_macro_f1",
    )
    plot_grouped_metric_bar(
        frame,
        x_col="preprocessing",
        metric_cols=["val_macro_f1", "test_macro_f1"],
        output_path=grouped_output,
        title="Preprocessing Ablation on UT-HAR (Validation vs Test Macro F1)",
        xlabel="Preprocessing",
        ylabel="Macro F1",
        y_zoom=True,
        value_labels=True,
    )
    return [main_output, grouped_output]
