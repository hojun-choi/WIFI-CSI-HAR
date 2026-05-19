"""Reusable plotting helpers for report and presentation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir


def plot_metric_bar(df, x_col, y_col, output_path, title, xlabel, ylabel) -> None:
    """Save a simple bar chart for report-ready experiment comparison."""
    ensure_dir(Path(output_path).parent)
    frame = pd.DataFrame(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(frame[x_col].astype(str), frame[y_col].astype(float))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if frame[x_col].astype(str).map(len).max() > 8:
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_lines(df, x_col, y_col, group_col, output_path, title, xlabel, ylabel) -> None:
    """Save a grouped line chart for report-ready trend comparison."""
    ensure_dir(Path(output_path).parent)
    frame = pd.DataFrame(df)
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


def save_preprocessing_ablation_plot(csv_path, output_path) -> None:
    """Create the preprocessing ablation comparison plot from its CSV output."""
    frame = pd.read_csv(csv_path)
    plot_metric_bar(
        frame,
        x_col="preprocessing",
        y_col="val_macro_f1",
        output_path=output_path,
        title="Preprocessing Ablation on UT-HAR (Validation Macro F1)",
        xlabel="Preprocessing",
        ylabel="Validation Macro F1",
    )
