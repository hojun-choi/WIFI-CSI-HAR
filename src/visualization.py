"""Reusable plotting helpers for report and presentation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.uthar_labels import UT_HAR_CLASS_NAMES, UT_HAR_CLASS_ORDER
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


def _annotate_line_points(ax, x_values: list[float], y_values: list[float]) -> None:
    for x_value, y_value in zip(x_values, y_values):
        ax.text(
            x_value,
            y_value,
            f"{y_value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _normalize_multiline_text(text: str) -> str:
    return text.replace("\\n", "\n")


def _annotate_line_points_staggered(
    ax,
    x_values: list[float],
    y_values: list[float],
    *,
    series_index: int = 0,
    fontsize: int = 8,
) -> None:
    offsets = [(0, 8), (0, -10), (-10, 8), (10, 8), (-12, -10), (12, -10)]
    for point_index, (x_value, y_value) in enumerate(zip(x_values, y_values)):
        dx, dy = offsets[(series_index + point_index) % len(offsets)]
        va = "bottom" if dy >= 0 else "top"
        ax.annotate(
            f"{float(y_value):.4f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            va=va,
            fontsize=fontsize,
        )


def _annotate_integer_bars(ax, bars, values: list[int]) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(int(value)),
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


def save_sample_csi_lineplot(
    sample: np.ndarray,
    output_path,
    feature_indices: list[int] | None = None,
) -> Path:
    """Save a waveform-like CSI view for report-friendly time-series interpretation."""
    ensure_dir(Path(output_path).parent)
    sample_array = np.asarray(sample)
    if sample_array.ndim != 2:
        raise ValueError(f"Expected a 2D sample array, got shape {sample_array.shape}")

    feature_count = sample_array.shape[1]
    if feature_indices is None:
        feature_indices = np.linspace(0, feature_count - 1, num=3, dtype=int).tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    time_steps = np.arange(sample_array.shape[0])
    for feature_index in feature_indices:
        if feature_index < 0 or feature_index >= feature_count:
            raise ValueError(
                f"feature index {feature_index} is out of range for sample width {feature_count}"
            )
        ax.plot(
            time_steps,
            sample_array[:, feature_index],
            label=f"feature {feature_index}",
        )

    ax.set_title("UT-HAR Sample CSI Line Plot (train[0])")
    ax.set_xlabel("timestep / CSI frame index")
    ax.set_ylabel("CSI amplitude/value")
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def save_uthar_class_distribution_by_activity(
    split_to_counts: dict[str, list[int]],
    output_path,
) -> Path:
    ensure_dir(Path(output_path).parent)
    activity_labels = [UT_HAR_CLASS_NAMES[class_id] for class_id in UT_HAR_CLASS_ORDER]
    split_names = [split_name for split_name in ["train", "val", "test"] if split_name in split_to_counts]
    positions = np.arange(len(activity_labels))
    width = 0.8 / max(1, len(split_names))
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, split_name in enumerate(split_names):
        counts = [int(value) for value in split_to_counts[split_name]]
        offset = (idx - (len(split_names) - 1) / 2.0) * width
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        bars = ax.bar(positions + offset, counts, width=width, label=split_name, color=color)
        _annotate_integer_bars(ax, bars, counts)

    ax.set_title("UT-HAR Class Distribution by Activity and Split")
    ax.set_xlabel("Activity")
    ax.set_ylabel("Sample Count")
    ax.set_xticks(positions)
    ax.set_xticklabels(activity_labels)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def save_split_size_summary(split_sizes: dict[str, int], output_path) -> Path:
    ensure_dir(Path(output_path).parent)
    labels = list(split_sizes.keys())
    values = [int(split_sizes[label]) for label in labels]
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(positions, values)
    _annotate_integer_bars(ax, bars, values)
    ax.set_title("UT-HAR Loaded Split Size Summary")
    ax.set_xlabel("Split")
    ax.set_ylabel("Sample Count")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def save_sample_csi_heatmap(sample: np.ndarray, output_path) -> Path:
    ensure_dir(Path(output_path).parent)
    sample_array = np.asarray(sample)
    if sample_array.ndim != 2:
        raise ValueError(f"Expected a 2D sample array, got shape {sample_array.shape}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    image = ax.imshow(sample_array, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("UT-HAR Sample CSI Heatmap (one sample = 250 CSI frames x 90 CSI features)")
    ax.set_xlabel("CSI feature index")
    ax.set_ylabel("timestep / CSI frame index")
    fig.colorbar(image, ax=ax, label="CSI amplitude/value")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def save_sample_heatmap_by_activity(
    samples_by_class: dict[int, np.ndarray],
    output_path,
) -> Path:
    ensure_dir(Path(output_path).parent)
    ordered_classes = [class_id for class_id in UT_HAR_CLASS_ORDER if class_id in samples_by_class]
    if not ordered_classes:
        raise ValueError("samples_by_class must contain at least one class sample.")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes_flat = axes.flatten()
    image = None

    for axis in axes_flat:
        axis.axis("off")

    for axis, class_id in zip(axes_flat, ordered_classes):
        sample = np.asarray(samples_by_class[class_id])
        image = axis.imshow(sample, aspect="auto", origin="lower", cmap="viridis")
        axis.set_title(UT_HAR_CLASS_NAMES[class_id], fontsize=10)
        axis.set_xlabel("CSI feature index")
        axis.set_ylabel("timestep / CSI frame index")
        axis.axis("on")

    if image is not None:
        fig.colorbar(image, ax=axes_flat.tolist(), shrink=0.8, label="CSI amplitude/value")
    fig.suptitle("UT-HAR Sample CSI Heatmap by Activity", fontsize=13)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


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


def save_final_preprocessing_plots(csv_path, output_dir) -> list[Path]:
    """Generate official final-workflow preprocessing comparison figures."""
    frame = pd.read_csv(csv_path)
    output_root = Path(output_dir)
    ensure_dir(output_root)

    frame = frame.copy()
    if "preprocessing_group" in frame.columns:
        frame["preprocessing_label"] = frame.apply(
            lambda row: (
                f"{row['preprocessing_group']}:{row['preprocessing']}"
                if str(row["preprocessing_group"]) != "single"
                else str(row["preprocessing"])
            ),
            axis=1,
        )
    else:
        frame["preprocessing_label"] = frame["preprocessing"].astype(str)

    frame = frame.sort_values(by=["val_macro_f1", "test_macro_f1"], ascending=[False, False])

    created_files = [
        output_root / "final_preprocessing_val_macro_f1.png",
        output_root / "final_preprocessing_val_test_macro_f1.png",
        output_root / "final_preprocessing_accuracy.png",
    ]

    plot_metric_bar(
        frame,
        x_col="preprocessing_label",
        y_col="val_macro_f1",
        output_path=created_files[0],
        title="Official Final Workflow Preprocessing Comparison (Validation Macro F1)",
        xlabel="Preprocessing",
        ylabel="Validation Macro F1",
        y_zoom=True,
        value_labels=True,
        sort_by="val_macro_f1",
    )
    plot_grouped_metric_bar(
        frame,
        x_col="preprocessing_label",
        metric_cols=["val_macro_f1", "test_macro_f1"],
        output_path=created_files[1],
        title="Official Final Workflow Preprocessing Comparison (Validation vs Test Macro F1)",
        xlabel="Preprocessing",
        ylabel="Macro F1",
        y_zoom=True,
        value_labels=True,
    )
    plot_grouped_metric_bar(
        frame,
        x_col="preprocessing_label",
        metric_cols=["val_accuracy", "test_accuracy"],
        output_path=created_files[2],
        title="Official Final Workflow Preprocessing Comparison (Validation vs Test Accuracy)",
        xlabel="Preprocessing",
        ylabel="Accuracy",
        y_zoom=True,
        value_labels=True,
    )

    if frame["model"].astype(str).nunique() > 1:
        by_model_output = output_root / "final_preprocessing_by_model_macro_f1.png"
        fig, ax = plt.subplots(figsize=(10, 5.5))
        labels = list(dict.fromkeys(frame["preprocessing_label"].astype(str).tolist()))
        positions = np.arange(len(labels))
        for model_name, model_df in frame.groupby("model"):
            ordered = (
                model_df.set_index("preprocessing_label")
                .reindex(labels)
                .dropna(subset=["val_macro_f1"])
                .reset_index()
            )
            if ordered.empty:
                continue
            x_positions = [labels.index(str(label)) for label in ordered["preprocessing_label"]]
            y_values = ordered["val_macro_f1"].astype(float).to_numpy()
            ax.plot(x_positions, y_values, marker="o", label=str(model_name))

        ax.set_title("Official Final Workflow Preprocessing Comparison by Model (Validation Macro F1)")
        ax.set_xlabel("Preprocessing")
        ax.set_ylabel("Validation Macro F1")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        _rotate_x_labels_if_needed(ax, labels)
        zoom_limits = _compute_zoom_limits(frame["val_macro_f1"].astype(float).to_numpy(), y_zoom=True)
        if zoom_limits is not None:
            ax.set_ylim(*zoom_limits)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.savefig(by_model_output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        created_files.append(by_model_output)

    return created_files


def save_preprocessing_stability_plots(csv_path, output_dir) -> list[Path]:
    """Generate multi-seed preprocessing stability figures from the summary CSV."""
    frame = pd.read_csv(csv_path).copy()
    output_root = Path(output_dir)
    ensure_dir(output_root)

    numeric_columns = [
        "mean_val_macro_f1",
        "std_val_macro_f1",
        "mean_test_macro_f1",
        "std_test_macro_f1",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values(by=["selection_rank", "mean_val_macro_f1"], ascending=[True, False])
    labels = frame["preprocessing"].astype(str).tolist()
    positions = np.arange(len(labels))
    created_files = [
        output_root / "final_preprocessing_stability_mean_val_macro_f1.png",
        output_root / "final_preprocessing_stability_val_test_macro_f1.png",
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    values = frame["mean_val_macro_f1"].astype(float).to_numpy()
    errors = frame["std_val_macro_f1"].astype(float).to_numpy()
    bars = ax.bar(
        positions,
        values,
        width=0.62,
        alpha=0.72,
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    )
    ax.errorbar(
        positions,
        values,
        yerr=errors,
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=1.4,
        capsize=6,
        capthick=1.4,
        markersize=4,
        zorder=4,
    )
    ax.set_title("Preprocessing Stability Check (Mean Validation Macro F1 with std across seeds)")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel("Mean Validation Macro F1")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    _rotate_x_labels_if_needed(ax, labels)
    lower = max(0.0, float(np.min(values - errors)) - 0.01)
    upper = min(1.0, float(np.max(values + errors)) + 0.012)
    if upper <= lower:
        upper = min(1.0, lower + 0.02)
    ax.set_ylim(lower, upper)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=1)
    for bar, value, error in zip(bars, values, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + error + 0.0025,
            _normalize_multiline_text(f"{value:.4f}\\n+-{error:.4f}"),
            ha="center",
            va="bottom",
            fontsize=8,
            zorder=5,
        )
    fig.savefig(created_files[0], dpi=200, bbox_inches="tight")
    plt.close(fig)

    plot_grouped_metric_bar(
        frame,
        x_col="preprocessing",
        metric_cols=["mean_val_macro_f1", "mean_test_macro_f1"],
        output_path=created_files[1],
        title="Preprocessing Stability Check (Mean Validation vs Test Macro F1)",
        xlabel="Preprocessing",
        ylabel="Macro F1",
        y_zoom=True,
        value_labels=True,
    )
    return created_files


def save_low_data_plots(csv_path, output_dir) -> list[Path]:
    """Generate low-data robustness figures for report and presentation use."""
    frame = pd.read_csv(csv_path)
    output_root = Path(output_dir)
    ensure_dir(output_root)

    ordered_ratios = [1.0, 0.5, 0.25, 0.1]
    frame = frame.copy()
    frame["real_ratio"] = frame["real_ratio"].astype(float)
    frame["real_ratio_label"] = frame["real_ratio"].map(lambda value: f"{value:g}")

    created_files = [
        output_root / "low_data_macro_f1.png",
        output_root / "low_data_accuracy.png",
        output_root / "low_data_degradation_macro_f1.png",
        output_root / "low_data_degradation_accuracy.png",
    ]

    def _plot_line(metric_col: str, output_path: Path, title: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for model_name, model_df in frame.groupby("model"):
            ratio_to_value = {
                float(row["real_ratio"]): float(row[metric_col])
                for _, row in model_df.iterrows()
                if pd.notna(row[metric_col])
            }
            x_values = [ratio for ratio in ordered_ratios if ratio in ratio_to_value]
            if not x_values:
                continue
            y_values = [ratio_to_value[ratio] for ratio in x_values]
            ax.plot(x_values, y_values, marker="o", label=str(model_name))

        ax.set_title(title)
        ax.set_xlabel("Real Training Data Ratio")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ordered_ratios)
        ax.set_xticklabels([f"{ratio:g}" for ratio in ordered_ratios])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    _plot_line(
        metric_col="test_macro_f1",
        output_path=created_files[0],
        title="UT-HAR Low-data Robustness (Test Macro F1)",
        ylabel="Test Macro F1",
    )
    _plot_line(
        metric_col="test_accuracy",
        output_path=created_files[1],
        title="UT-HAR Low-data Robustness (Test Accuracy)",
        ylabel="Test Accuracy",
    )

    if frame["macro_f1_drop"].notna().any():
        _plot_line(
            metric_col="macro_f1_drop",
            output_path=created_files[2],
            title="Macro F1 Drop from Full-data Baseline",
            ylabel="Macro F1 Drop",
        )
    else:
        created_files.pop(2)

    if frame["accuracy_drop"].notna().any():
        _plot_line(
            metric_col="accuracy_drop",
            output_path=created_files[-1],
            title="Accuracy Drop from Full-data Baseline",
            ylabel="Accuracy Drop",
        )
    else:
        created_files = [path for path in created_files if path.name != "low_data_degradation_accuracy.png"]

    return created_files


def save_final_low_data_plots(csv_path, output_dir) -> list[Path]:
    """Generate official final-workflow low-data robustness figures."""
    frame = pd.read_csv(csv_path).copy()
    output_root = Path(output_dir)
    ensure_dir(output_root)

    required_columns = ["model", "real_ratio", "test_macro_f1", "test_accuracy"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        print(
            "Warning: could not generate official low-data figures because required columns are missing: "
            f"{missing_columns}"
        )
        return []

    numeric_columns = [
        "real_ratio",
        "test_macro_f1",
        "test_accuracy",
        "macro_f1_drop",
        "macro_f1_retention",
        "accuracy_drop",
        "accuracy_retention",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    ordered_ratios = [1.0, 0.5, 0.25, 0.1]
    ratio_labels = {1.0: "100%", 0.5: "50%", 0.25: "25%", 0.1: "10%"}
    models_in_order = ["ResNet18", "LeNet", "ResNet101"]
    present_models = [model for model in models_in_order if model in frame["model"].astype(str).unique()]
    if not present_models:
        present_models = sorted(frame["model"].astype(str).unique().tolist())

    created_files: list[Path] = []

    def _plot_line_metric(
        metric_col: str,
        output_path: Path,
        title: str,
        ylabel: str,
        *,
        add_reference_one: bool = False,
    ) -> None:
        if metric_col not in frame.columns:
            print(f"Warning: skipped {output_path.name} because {metric_col} is missing.")
            return
        fig, ax = plt.subplots(figsize=(8.8, 5.3))
        all_values: list[float] = []
        for model_index, model_name in enumerate(present_models):
            model_df = frame[frame["model"].astype(str) == model_name].copy()
            ratio_to_value = {
                float(row["real_ratio"]): float(row[metric_col])
                for _, row in model_df.iterrows()
                if pd.notna(row.get(metric_col))
            }
            x_values = [ratio for ratio in ordered_ratios if ratio in ratio_to_value]
            if not x_values:
                continue
            y_values = [ratio_to_value[ratio] for ratio in x_values]
            all_values.extend(y_values)
            ax.plot(x_values, y_values, marker="o", label=model_name)
            _annotate_line_points_staggered(
                ax,
                x_values,
                y_values,
                series_index=model_index,
                fontsize=7,
            )

        if add_reference_one:
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle=":")
        ax.set_title(_normalize_multiline_text(title))
        ax.set_xlabel("Real training data ratio")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ordered_ratios)
        ax.set_xticklabels([ratio_labels[ratio] for ratio in ordered_ratios])
        zoom_limits = _compute_zoom_limits(np.asarray(all_values, dtype=float), y_zoom=True)
        if zoom_limits is not None and metric_col in {"test_macro_f1", "test_accuracy", "macro_f1_retention"}:
            lower, upper = zoom_limits
            ax.set_ylim(lower - 0.005, upper + 0.012)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left" if metric_col == "macro_f1_retention" else "best")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        created_files.append(output_path)

    _plot_line_metric(
        metric_col="test_macro_f1",
        output_path=output_root / "final_low_data_macro_f1_by_ratio.png",
        title="Low-data robustness: Raw test Macro F1 by real training data ratio",
        ylabel="Raw test Macro F1",
    )
    _plot_line_metric(
        metric_col="test_accuracy",
        output_path=output_root / "final_low_data_accuracy_by_ratio.png",
        title="Low-data robustness: Raw test Accuracy by real training data ratio",
        ylabel="Raw test Accuracy",
    )
    _plot_line_metric(
        metric_col="macro_f1_retention",
        output_path=output_root / "final_low_data_macro_f1_retention_by_ratio.png",
        title=(
            "Normalized Macro F1 retention by real training data ratio\n"
            "(100% = 1.0 by definition for each model's own full-data baseline)"
        ),
        ylabel="Normalized Macro F1 retention (vs each model's 100% baseline)",
        add_reference_one=True,
    )
    _plot_line_metric(
        metric_col="macro_f1_drop",
        output_path=output_root / "final_low_data_macro_f1_drop_by_ratio.png",
        title="Macro F1 drop under reduced training data",
        ylabel="Macro F1 drop",
    )

    if "test_macro_f1" in frame.columns:
        subset = frame[frame["real_ratio"].isin([0.25, 0.1])].copy()
        if not subset.empty:
            subset["ratio_label"] = subset["real_ratio"].map(ratio_labels)
            positions = np.arange(len(present_models))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8.5, 5))
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
            for idx, ratio in enumerate([0.25, 0.1]):
                ratio_df = (
                    subset[subset["real_ratio"] == ratio]
                    .set_index("model")
                    .reindex(present_models)
                    .reset_index()
                )
                values = ratio_df["test_macro_f1"].astype(float).fillna(np.nan).to_numpy()
                offset = (idx - 0.5) * width
                color = color_cycle[idx % len(color_cycle)] if color_cycle else None
                bars = ax.bar(
                    positions + offset,
                    values,
                    width=width,
                    label=ratio_labels[ratio],
                    color=color,
                )
                for bar, value in zip(bars, values):
                    if np.isnan(value):
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        f"{float(value):.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            ax.set_title("Low-data robustness summary at 25% and 10% (raw test Macro F1)")
            ax.set_xlabel("Model")
            ax.set_ylabel("Raw test Macro F1")
            ax.set_xticks(positions)
            ax.set_xticklabels(present_models)
            ax.legend()
            fig.savefig(output_root / "final_low_data_25_10_summary.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            created_files.append(output_root / "final_low_data_25_10_summary.png")

    return created_files


def save_final_augmentation_plots(csv_path, output_dir) -> list[Path]:
    """Generate official final-workflow augmentation recovery figures."""
    frame = pd.read_csv(csv_path).copy()
    output_root = Path(output_dir)
    ensure_dir(output_root)

    required_columns = [
        "model",
        "real_ratio",
        "augmentation_gain_macro_f1",
        "augmentation_gain_accuracy",
        "test_macro_f1",
        "test_accuracy",
    ]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        print(
            "Warning: could not generate official augmentation figures because required columns are missing: "
            f"{missing_columns}"
        )
        return []

    numeric_columns = [
        "real_ratio",
        "augmentation_gain_macro_f1",
        "augmentation_gain_accuracy",
        "test_macro_f1",
        "test_accuracy",
        "no_aug_test_macro_f1",
        "no_aug_test_accuracy",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    ordered_ratios = [0.5, 0.25, 0.1]
    ratio_labels = {0.5: "50%", 0.25: "25%", 0.1: "10%"}
    models_in_order = ["ResNet18", "LeNet", "ResNet101"]
    present_models = [model for model in models_in_order if model in frame["model"].astype(str).unique()]
    if not present_models:
        present_models = sorted(frame["model"].astype(str).unique().tolist())

    created_files: list[Path] = []

    def _plot_gain_metric(
        metric_col: str,
        output_path: Path,
        title: str,
        ylabel: str,
    ) -> None:
        if metric_col not in frame.columns:
            print(f"Warning: skipped {output_path.name} because {metric_col} is missing.")
            return
        fig, ax = plt.subplots(figsize=(8.5, 5))
        all_values: list[float] = []
        for model_name in present_models:
            model_df = frame[frame["model"].astype(str) == model_name].copy()
            ratio_to_value = {
                float(row["real_ratio"]): float(row[metric_col])
                for _, row in model_df.iterrows()
                if pd.notna(row.get(metric_col))
            }
            x_values = [ratio for ratio in ordered_ratios if ratio in ratio_to_value]
            if not x_values:
                continue
            y_values = [ratio_to_value[ratio] for ratio in x_values]
            all_values.extend(y_values)
            ax.plot(x_values, y_values, marker="o", label=model_name)
            _annotate_line_points(ax, x_values, y_values)

        ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("Real training data ratio")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ordered_ratios)
        ax.set_xticklabels([ratio_labels[ratio] for ratio in ordered_ratios])
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            padding = max(0.01, (y_max - y_min) * 0.15 if y_max != y_min else 0.01)
            ax.set_ylim(y_min - padding, y_max + padding)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        created_files.append(output_path)

    _plot_gain_metric(
        metric_col="augmentation_gain_macro_f1",
        output_path=output_root / "final_augmentation_gain_macro_f1_by_ratio.png",
        title="Offline appended augmentation: Macro F1 gain by train ratio",
        ylabel="Augmentation gain (Macro F1)",
    )
    _plot_gain_metric(
        metric_col="augmentation_gain_accuracy",
        output_path=output_root / "final_augmentation_gain_accuracy_by_ratio.png",
        title="Offline appended augmentation: Accuracy gain by train ratio",
        ylabel="Augmentation gain (Accuracy)",
    )

    if "no_aug_test_macro_f1" in frame.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        positions = np.arange(len(ordered_ratios))
        width = 0.12 if len(present_models) > 2 else 0.18
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for idx, model_name in enumerate(present_models):
            model_df = frame[frame["model"].astype(str) == model_name].copy()
            ordered = (
                model_df.set_index("real_ratio")
                .reindex(ordered_ratios)
                .reset_index()
            )
            aug_values = ordered["test_macro_f1"].astype(float).to_numpy()
            no_aug_values = ordered["no_aug_test_macro_f1"].astype(float).to_numpy()
            center = (idx - (len(present_models) - 1) / 2.0) * (2 * width + 0.03)
            color = color_cycle[idx % len(color_cycle)] if color_cycle else None
            bars_no_aug = ax.bar(
                positions + center - width / 2.0,
                no_aug_values,
                width=width,
                label=f"{model_name} no aug",
                color=color,
                alpha=0.45,
            )
            bars_aug = ax.bar(
                positions + center + width / 2.0,
                aug_values,
                width=width,
                label=f"{model_name} aug",
                color=color,
            )
            for bars, values in [(bars_no_aug, no_aug_values), (bars_aug, aug_values)]:
                for bar, value in zip(bars, values):
                    if np.isnan(value):
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        f"{float(value):.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        ax.set_title("Real data + synthetic augmentation: Macro F1 with and without augmentation")
        ax.set_xlabel("Real training data ratio")
        ax.set_ylabel("Test Macro F1")
        ax.set_xticks(positions)
        ax.set_xticklabels([ratio_labels[ratio] for ratio in ordered_ratios])
        ax.legend(ncol=2, fontsize=8)
        fig.savefig(
            output_root / "final_augmentation_macro_f1_aug_vs_no_aug.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
        created_files.append(output_root / "final_augmentation_macro_f1_aug_vs_no_aug.png")
    else:
        print(
            "Warning: skipped final_augmentation_macro_f1_aug_vs_no_aug.png because "
            "no_aug_test_macro_f1 is missing."
        )

    subset = frame[frame["real_ratio"].isin([0.25, 0.1])].copy()
    if not subset.empty and "augmentation_gain_macro_f1" in subset.columns:
        subset["ratio_label"] = subset["real_ratio"].map(ratio_labels)
        positions = np.arange(len(present_models))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8.5, 5))
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for idx, ratio in enumerate([0.25, 0.1]):
            ratio_df = (
                subset[subset["real_ratio"] == ratio]
                .set_index("model")
                .reindex(present_models)
                .reset_index()
            )
            values = ratio_df["augmentation_gain_macro_f1"].astype(float).fillna(np.nan).to_numpy()
            offset = (idx - 0.5) * width
            color = color_cycle[idx % len(color_cycle)] if color_cycle else None
            bars = ax.bar(
                positions + offset,
                values,
                width=width,
                label=ratio_labels[ratio],
                color=color,
            )
            for bar, value in zip(bars, values):
                if np.isnan(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{float(value):.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
        ax.set_title("Offline appended augmentation summary at 25% and 10%")
        ax.set_xlabel("Model")
        ax.set_ylabel("Augmentation gain (Macro F1)")
        ax.set_xticks(positions)
        ax.set_xticklabels(present_models)
        ax.legend()
        fig.savefig(output_root / "final_augmentation_25_10_summary.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        created_files.append(output_root / "final_augmentation_25_10_summary.png")

    if "augmentation_gain_macro_f1" in frame.columns:
        pivot = (
            frame.pivot_table(
                index="model",
                columns="real_ratio",
                values="augmentation_gain_macro_f1",
                aggfunc="mean",
            )
            .reindex(index=present_models)
        )
        ordered_heatmap_cols = [ratio for ratio in ordered_ratios if ratio in pivot.columns]
        if ordered_heatmap_cols:
            pivot = pivot[ordered_heatmap_cols]
            fig, ax = plt.subplots(figsize=(7.5, 3.8))
            image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
            ax.set_title("Offline appended augmentation gain heatmap (Macro F1)")
            ax.set_xlabel("Real training data ratio")
            ax.set_ylabel("Model")
            ax.set_xticks(np.arange(len(ordered_heatmap_cols)))
            ax.set_xticklabels([ratio_labels[ratio] for ratio in ordered_heatmap_cols])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index.tolist())
            for row_idx in range(pivot.shape[0]):
                for col_idx in range(pivot.shape[1]):
                    value = pivot.iat[row_idx, col_idx]
                    if np.isnan(value):
                        continue
                    ax.text(col_idx, row_idx, f"{float(value):.4f}", ha="center", va="center", fontsize=8)
            fig.colorbar(image, ax=ax, label="Augmentation gain (Macro F1)")
            fig.savefig(output_root / "final_augmentation_gain_heatmap.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            created_files.append(output_root / "final_augmentation_gain_heatmap.png")

    return created_files


def save_final_augmentation_ablation_plots(csv_path, output_dir) -> list[Path]:
    """Generate official augmentation_add_ratio ablation figures."""
    frame = pd.read_csv(csv_path).copy()
    output_root = Path(output_dir)
    ensure_dir(output_root)

    required_columns = [
        "model",
        "real_ratio",
        "augmentation_add_ratio",
        "test_macro_f1",
        "augmentation_gain_macro_f1",
    ]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        print(
            "Warning: could not generate augmentation ablation figures because required columns are missing: "
            f"{missing_columns}"
        )
        return []

    numeric_columns = [
        "real_ratio",
        "augmentation_add_ratio",
        "test_macro_f1",
        "augmentation_gain_macro_f1",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    ordered_models = ["ResNet18", "LeNet", "ResNet101"]
    present_models = [model for model in ordered_models if model in frame["model"].astype(str).unique()]
    if not present_models:
        present_models = sorted(frame["model"].astype(str).unique().tolist())
    ordered_ratios = [0.5, 0.25, 0.1]
    ordered_add_ratios = sorted(frame["augmentation_add_ratio"].dropna().unique().tolist())
    ratio_labels = {0.5: "50%", 0.25: "25%", 0.1: "10%"}

    created_files: list[Path] = []

    def _condition_label(row: pd.Series) -> str:
        ratio = float(row["real_ratio"])
        ratio_label = ratio_labels.get(ratio, f"{ratio:g}")
        return f"{row['model']} ({ratio_label})"

    frame["condition_label"] = frame.apply(_condition_label, axis=1)
    ordered_condition_labels = [
        f"{model} ({ratio_labels.get(ratio, f'{ratio:g}')})"
        for model in present_models
        for ratio in ordered_ratios
        if f"{model} ({ratio_labels.get(ratio, f'{ratio:g}')})" in frame["condition_label"].tolist()
    ]
    add_ratio_labels = {ratio: f"add={ratio:g}" for ratio in ordered_add_ratios}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    metric_pivot = (
        frame.pivot_table(
            index="condition_label",
            columns="augmentation_add_ratio",
            values="test_macro_f1",
            aggfunc="mean",
        )
        .reindex(index=ordered_condition_labels)
    )
    available_metric_cols = [ratio for ratio in ordered_add_ratios if ratio in metric_pivot.columns]
    if available_metric_cols:
        metric_pivot = metric_pivot[available_metric_cols]
        positions = np.arange(len(metric_pivot.index))
        width = 0.8 / max(1, len(available_metric_cols))
        fig, ax = plt.subplots(figsize=(11.5, 5.8))
        for idx, add_ratio in enumerate(available_metric_cols):
            offset = (idx - (len(available_metric_cols) - 1) / 2.0) * width
            values = metric_pivot[add_ratio].astype(float).to_numpy()
            color = color_cycle[idx % len(color_cycle)] if color_cycle else None
            bars = ax.bar(
                positions + offset,
                values,
                width=width,
                label=add_ratio_labels[add_ratio],
                color=color,
                alpha=0.82,
                edgecolor="black",
                linewidth=0.6,
            )
            for bar, value in zip(bars, values):
                if np.isnan(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{float(value):.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_title("Offline appended augmentation ablation: raw test Macro F1 by add ratio")
        ax.set_xlabel("Model and real training data ratio")
        ax.set_ylabel("Raw test Macro F1")
        ax.set_xticks(positions)
        ax.set_xticklabels(metric_pivot.index.tolist())
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(title="Synthetic ratio", loc="upper right", fontsize=8)
        output_path = output_root / "final_augmentation_ablation_macro_f1_by_add_ratio.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        created_files.append(output_path)

    gain_pivot = (
        frame.pivot_table(
            index="condition_label",
            columns="augmentation_add_ratio",
            values="augmentation_gain_macro_f1",
            aggfunc="mean",
        )
        .reindex(index=ordered_condition_labels)
    )
    available_gain_cols = [ratio for ratio in ordered_add_ratios if ratio in gain_pivot.columns]
    if available_gain_cols:
        gain_pivot = gain_pivot[available_gain_cols]
        positions = np.arange(len(gain_pivot.index))
        width = 0.8 / max(1, len(available_gain_cols))
        fig, ax = plt.subplots(figsize=(11.5, 5.8))
        all_gain_values = gain_pivot.to_numpy(dtype=float).flatten()
        valid_gain_values = all_gain_values[~np.isnan(all_gain_values)]
        for idx, add_ratio in enumerate(available_gain_cols):
            offset = (idx - (len(available_gain_cols) - 1) / 2.0) * width
            values = gain_pivot[add_ratio].astype(float).to_numpy()
            color = color_cycle[idx % len(color_cycle)] if color_cycle else None
            bars = ax.bar(
                positions + offset,
                values,
                width=width,
                label=add_ratio_labels[add_ratio],
                color=color,
                alpha=0.82,
                edgecolor="black",
                linewidth=0.6,
            )
            for bar, value in zip(bars, values):
                if np.isnan(value):
                    continue
                va = "bottom" if value >= 0 else "top"
                y_text = value + 0.01 if value >= 0 else value - 0.015
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_text,
                    f"{float(value):.4f}",
                    ha="center",
                    va=va,
                    fontsize=7,
                )
                if float(value) <= -0.2:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        min(value, 0) - 0.05,
                        "collapse",
                        ha="center",
                        va="top",
                        fontsize=7,
                        color="darkred",
                        rotation=90,
                    )
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
        if valid_gain_values.size:
            lower = float(valid_gain_values.min()) - 0.08
            upper = float(valid_gain_values.max()) + 0.08
            ax.set_ylim(lower, upper)
        ax.set_title("Offline appended augmentation ablation: Macro F1 gain vs same-ratio F4 baseline")
        ax.set_xlabel("Model and real training data ratio")
        ax.set_ylabel("Augmentation gain (Macro F1)")
        ax.set_xticks(positions)
        ax.set_xticklabels(gain_pivot.index.tolist())
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(title="Synthetic ratio", loc="upper right", fontsize=8)
        output_path = output_root / "final_augmentation_ablation_gain_by_add_ratio.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        created_files.append(output_path)

    best_rows = (
        frame.sort_values(
            by=["condition_label", "test_macro_f1", "augmentation_gain_macro_f1"],
            ascending=[True, False, False],
        )
        .groupby("condition_label", as_index=False)
        .first()
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    positions = np.arange(len(best_rows))
    best_add_ratios = best_rows["augmentation_add_ratio"].astype(float).tolist()
    best_values = best_rows["test_macro_f1"].astype(float).to_numpy()
    bar_colors = []
    for add_ratio in best_add_ratios:
        idx = available_gain_cols.index(add_ratio) if add_ratio in available_gain_cols else 0
        bar_colors.append(color_cycle[idx % len(color_cycle)] if color_cycle else None)
    bars = ax.bar(
        positions,
        best_values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
    )
    labels = best_rows["condition_label"].astype(str).tolist()
    ax.set_title("Best augmentation_add_ratio by condition (ranked by raw test Macro F1)")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Best raw test Macro F1")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for bar, value, add_ratio in zip(bars, best_values, best_add_ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            _normalize_multiline_text(
                f"{float(value):.4f}\\nbest add_ratio={float(add_ratio):g}"
            ),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    output_path = output_root / "final_augmentation_ablation_best_add_ratio_by_condition.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    created_files.append(output_path)

    pivot = gain_pivot.copy()
    ordered_heatmap_cols = [ratio for ratio in ordered_add_ratios if ratio in pivot.columns]
    if ordered_heatmap_cols:
        pivot = pivot[ordered_heatmap_cols]
        max_abs = np.nanmax(np.abs(pivot.to_numpy(dtype=float)))
        color_limit = max(0.05, float(max_abs)) if not np.isnan(max_abs) else 0.05
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        image = ax.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
        )
        ax.set_title("Offline appended augmentation ablation heatmap (Macro F1 gain vs same-ratio baseline)")
        ax.set_xlabel("augmentation_add_ratio")
        ax.set_ylabel("Model + real_ratio")
        ax.set_xticks(np.arange(len(ordered_heatmap_cols)))
        ax.set_xticklabels([f"{ratio:g}" for ratio in ordered_heatmap_cols])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        for row_idx in range(pivot.shape[0]):
            for col_idx in range(pivot.shape[1]):
                value = pivot.iat[row_idx, col_idx]
                if np.isnan(value):
                    continue
                label = f"{float(value):.4f}"
                if float(value) <= -0.2:
                    label += "\ncollapse"
                ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="Augmentation gain (Macro F1)")
        output_path = output_root / "final_augmentation_ablation_heatmap.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        created_files.append(output_path)

    return created_files


def build_final_augmentation_ablation_summary_by_add_ratio(
    results_csv_path,
    summary_csv_path,
) -> tuple[pd.DataFrame, Path]:
    """Aggregate F5.1 condition-level results by augmentation_add_ratio."""
    results_path = Path(results_csv_path)
    summary_path = Path(summary_csv_path)
    ensure_dir(summary_path.parent)

    frame = pd.read_csv(results_path).copy()
    required_columns = [
        "augmentation_add_ratio",
        "model",
        "real_ratio",
        "test_macro_f1",
        "test_accuracy",
        "augmentation_gain_macro_f1",
        "augmentation_gain_accuracy",
    ]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Could not build augmentation ratio ablation summary because required columns are "
            f"missing: {missing_columns}"
        )

    numeric_columns = [
        "augmentation_add_ratio",
        "real_ratio",
        "test_macro_f1",
        "test_accuracy",
        "augmentation_gain_macro_f1",
        "augmentation_gain_accuracy",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    summary_records: list[dict[str, object]] = []
    for add_ratio, group_df in frame.groupby("augmentation_add_ratio", dropna=True):
        clean_group = group_df.copy()
        positive_mask = clean_group["augmentation_gain_macro_f1"] > 0
        source_counts = (
            clean_group["source"].astype(str).value_counts() if "source" in clean_group.columns else pd.Series(dtype=int)
        )
        summary_records.append(
            {
                "augmentation_add_ratio": float(add_ratio),
                "n_conditions": int(len(clean_group)),
                "models_covered": ", ".join(sorted(clean_group["model"].dropna().astype(str).unique().tolist())),
                "real_ratios_covered": ", ".join(
                    f"{float(ratio):g}"
                    for ratio in sorted(clean_group["real_ratio"].dropna().astype(float).unique().tolist(), reverse=True)
                ),
                "mean_test_macro_f1": float(clean_group["test_macro_f1"].mean()),
                "std_test_macro_f1": float(clean_group["test_macro_f1"].std(ddof=1)),
                "min_test_macro_f1": float(clean_group["test_macro_f1"].min()),
                "max_test_macro_f1": float(clean_group["test_macro_f1"].max()),
                "median_test_macro_f1": float(clean_group["test_macro_f1"].median()),
                "mean_test_accuracy": float(clean_group["test_accuracy"].mean()),
                "std_test_accuracy": float(clean_group["test_accuracy"].std(ddof=1)),
                "mean_augmentation_gain_macro_f1": float(clean_group["augmentation_gain_macro_f1"].mean()),
                "std_augmentation_gain_macro_f1": float(clean_group["augmentation_gain_macro_f1"].std(ddof=1)),
                "min_augmentation_gain_macro_f1": float(clean_group["augmentation_gain_macro_f1"].min()),
                "max_augmentation_gain_macro_f1": float(clean_group["augmentation_gain_macro_f1"].max()),
                "median_augmentation_gain_macro_f1": float(clean_group["augmentation_gain_macro_f1"].median()),
                "mean_augmentation_gain_accuracy": float(clean_group["augmentation_gain_accuracy"].mean()),
                "std_augmentation_gain_accuracy": float(clean_group["augmentation_gain_accuracy"].std(ddof=1)),
                "positive_gain_count": int(positive_mask.sum()),
                "zero_or_negative_gain_count": int((~positive_mask).sum()),
                "positive_gain_rate": float(positive_mask.mean()),
                "reused_final_f5_count": int(source_counts.get("reused_final_f5", 0)),
                "trained_ablation_count": int(source_counts.get("trained_ablation", 0)),
                "existing_ablation_count": int(source_counts.get("existing_ablation", 0)),
            }
        )

    summary_frame = pd.DataFrame(summary_records).sort_values(
        by=["augmentation_add_ratio"], ascending=[True]
    ).reset_index(drop=True)
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8")
    return summary_frame, summary_path


def save_final_augmentation_ablation_summary_plots(summary_csv_path, output_dir) -> list[Path]:
    """Generate aggregate F5.1 summary plots grouped only by augmentation_add_ratio."""
    frame = pd.read_csv(summary_csv_path).copy()
    output_root = Path(output_dir)
    ensure_dir(output_root)

    required_columns = [
        "augmentation_add_ratio",
        "n_conditions",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_augmentation_gain_macro_f1",
        "std_augmentation_gain_macro_f1",
        "positive_gain_rate",
        "mean_test_accuracy",
        "std_test_accuracy",
    ]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        print(
            "Warning: could not generate augmentation ablation aggregate figures because required "
            f"columns are missing: {missing_columns}"
        )
        return []

    for column in required_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    ordered = frame.sort_values(by=["augmentation_add_ratio"], ascending=[True]).reset_index(drop=True)
    add_labels = [f"add={float(value):g}" for value in ordered["augmentation_add_ratio"].tolist()]
    positions = np.arange(len(ordered))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    created_files: list[Path] = []

    def _bar_with_error(
        ax,
        x_positions,
        values,
        errors,
        *,
        width: float,
        label: str,
        color,
    ):
        return ax.bar(
            x_positions,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            yerr=errors,
            error_kw={"elinewidth": 1.3, "capsize": 5, "capthick": 1.3, "ecolor": "black"},
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    width = 0.34
    macro_values = ordered["mean_test_macro_f1"].astype(float).to_numpy()
    macro_errors = ordered["std_test_macro_f1"].astype(float).fillna(0.0).to_numpy()
    gain_values = ordered["mean_augmentation_gain_macro_f1"].astype(float).to_numpy()
    gain_errors = ordered["std_augmentation_gain_macro_f1"].astype(float).fillna(0.0).to_numpy()
    macro_bars = _bar_with_error(
        axes[0],
        positions,
        macro_values,
        macro_errors,
        width=0.55,
        label="mean_test_macro_f1",
        color=(color_cycle[0] if color_cycle else None),
    )
    gain_bars = _bar_with_error(
        axes[1],
        positions,
        gain_values,
        gain_errors,
        width=0.55,
        label="mean_augmentation_gain_macro_f1",
        color=(color_cycle[1] if len(color_cycle) > 1 else None),
    )
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    for ax, bars, values, errors, ylabel in [
        (axes[0], macro_bars, macro_values, macro_errors, "Mean test Macro F1"),
        (axes[1], gain_bars, gain_values, gain_errors, "Mean augmentation gain (Macro F1)"),
    ]:
        for bar, value, error in zip(bars, values, errors):
            y_text = value + error + (0.01 if ylabel.startswith("Mean test") else 0.015)
            va = "bottom"
            if ylabel.startswith("Mean augmentation") and value < 0:
                y_text = value - error - 0.03
                va = "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_text,
                f"{float(value):.4f}",
                ha="center",
                va=va,
                fontsize=8,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(add_labels)
        ax.set_xlabel("augmentation_add_ratio")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[0].set_title("Aggregate raw test Macro F1 across all 9 F5.1 conditions")
    axes[1].set_title("Aggregate Macro F1 gain across all 9 F5.1 conditions")
    fig.suptitle("F5.1 aggregate summary by augmentation_add_ratio (3 models x 3 real_ratios)")
    summary_macro_path = output_root / "final_augmentation_ablation_add_ratio_summary_macro_f1.png"
    fig.savefig(summary_macro_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    created_files.append(summary_macro_path)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    bars = _bar_with_error(
        ax,
        positions,
        gain_values,
        gain_errors,
        width=0.58,
        label="mean_augmentation_gain_macro_f1",
        color=(color_cycle[1] if len(color_cycle) > 1 else None),
    )
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    y_min = float(np.min(gain_values - gain_errors))
    y_max = float(np.max(gain_values + gain_errors))
    lower_padding = max(0.06, (y_max - y_min) * 0.18 if y_max != y_min else 0.08)
    upper_padding = max(0.08, (y_max - y_min) * 0.28 if y_max != y_min else 0.10)
    ax.set_ylim(y_min - lower_padding, y_max + upper_padding)
    for bar, value, error, rate, count in zip(
        bars,
        gain_values,
        gain_errors,
        ordered["positive_gain_rate"].astype(float).to_numpy(),
        ordered["n_conditions"].astype(int).to_numpy(),
    ):
        positive_count = int(round(rate * count))
        x_center = bar.get_x() + bar.get_width() / 2.0
        primary_y = value + error + 0.02 if value >= 0 else value - error - 0.035
        secondary_y = primary_y + 0.05 if value >= 0 else primary_y - 0.055
        primary_va = "bottom" if value >= 0 else "top"
        secondary_va = "bottom" if value >= 0 else "top"
        ax.text(
            x_center,
            primary_y,
            f"{float(value):.4f}",
            ha="center",
            va=primary_va,
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        ax.text(
            x_center,
            secondary_y,
            f"{positive_count}/{count} positive",
            ha="center",
            va=secondary_va,
            fontsize=7,
            color="dimgray",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        if value < 0 and abs(value) > 0.2:
            ax.text(
                x_center,
                value - error - 0.10,
                "collapse risk",
                ha="center",
                va="top",
                fontsize=7,
                color="darkred",
                fontweight="bold",
            )
        elif value < 0:
            ax.text(
                x_center,
                value - error - 0.02,
                "negative",
                ha="center",
                va="top",
                fontsize=7,
                color="dimgray",
            )
    ax.set_title("Aggregate augmentation gain by add ratio across all 9 conditions")
    ax.set_xlabel("augmentation_add_ratio")
    ax.set_ylabel("Mean augmentation gain (Macro F1)")
    ax.set_xticks(positions)
    ax.set_xticklabels(add_labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    summary_gain_path = output_root / "final_augmentation_ablation_add_ratio_summary_gain.png"
    fig.savefig(summary_gain_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    created_files.append(summary_gain_path)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    rate_values = ordered["positive_gain_rate"].astype(float).to_numpy()
    bars = ax.bar(
        positions,
        rate_values,
        width=0.58,
        color=(color_cycle[2] if len(color_cycle) > 2 else None),
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, rate, count in zip(
        bars,
        rate_values,
        ordered["n_conditions"].astype(int).to_numpy(),
    ):
        positive_count = int(round(rate * count))
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            _normalize_multiline_text(f"{rate * 100:.1f}%\n{positive_count}/{count}"),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0.0, 1.12)
    ax.set_title("Positive-gain rate by augmentation_add_ratio (all 9 conditions)")
    ax.set_xlabel("augmentation_add_ratio")
    ax.set_ylabel("Positive gain rate")
    ax.set_xticks(positions)
    ax.set_xticklabels(add_labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    summary_positive_path = output_root / "final_augmentation_ablation_add_ratio_positive_rate.png"
    fig.savefig(summary_positive_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    created_files.append(summary_positive_path)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    accuracy_values = ordered["mean_test_accuracy"].astype(float).to_numpy()
    accuracy_errors = ordered["std_test_accuracy"].astype(float).fillna(0.0).to_numpy()
    bars = _bar_with_error(
        ax,
        positions,
        accuracy_values,
        accuracy_errors,
        width=0.58,
        label="mean_test_accuracy",
        color=(color_cycle[3] if len(color_cycle) > 3 else None),
    )
    for bar, value, error in zip(bars, accuracy_values, accuracy_errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + error + 0.01,
            f"{float(value):.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Aggregate raw test Accuracy by augmentation_add_ratio (all 9 conditions)")
    ax.set_xlabel("augmentation_add_ratio")
    ax.set_ylabel("Mean test Accuracy")
    ax.set_xticks(positions)
    ax.set_xticklabels(add_labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    summary_accuracy_path = output_root / "final_augmentation_ablation_add_ratio_accuracy_summary.png"
    fig.savefig(summary_accuracy_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    created_files.append(summary_accuracy_path)

    return created_files


def save_augmentation_recovery_plots(aug_csv_path, low_data_csv_path, output_dir) -> list[Path]:
    """Generate M5 figures by comparing augmentation against M4 no-aug results."""
    aug_frame = pd.read_csv(aug_csv_path)
    low_data_frame = pd.read_csv(low_data_csv_path) if Path(low_data_csv_path).exists() else pd.DataFrame()
    output_root = Path(output_dir)
    ensure_dir(output_root)

    created_files = [
        output_root / "augmentation_recovery_macro_f1.png",
        output_root / "augmentation_recovery_accuracy.png",
        output_root / "augmentation_gain_macro_f1.png",
        output_root / "augmentation_gain_accuracy.png",
    ]

    ordered_ratios = [0.5, 0.25, 0.1]

    def _ordered_series(frame: pd.DataFrame, metric_col: str, model_name: str) -> tuple[list[float], list[float]]:
        model_df = frame[frame["model"] == model_name].copy()
        if model_df.empty or metric_col not in model_df.columns:
            return [], []
        ratio_to_value = {
            float(row["real_ratio"]): float(row[metric_col])
            for _, row in model_df.iterrows()
            if pd.notna(row.get(metric_col))
        }
        x_values = [ratio for ratio in ordered_ratios if ratio in ratio_to_value]
        y_values = [ratio_to_value[ratio] for ratio in x_values]
        return x_values, y_values

    def _plot_recovery(metric_col: str, output_path: Path, title: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        all_models = sorted(set(aug_frame["model"].astype(str).tolist()))
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

        for idx, model_name in enumerate(all_models):
            color = color_cycle[idx % len(color_cycle)] if color_cycle else None
            x_no_aug, y_no_aug = _ordered_series(low_data_frame, metric_col, model_name)
            x_aug, y_aug = _ordered_series(aug_frame, metric_col, model_name)
            if x_no_aug:
                ax.plot(
                    x_no_aug,
                    y_no_aug,
                    marker="o",
                    linestyle="--",
                    color=color,
                    label=f"{model_name} (no aug)",
                )
            if x_aug:
                ax.plot(
                    x_aug,
                    y_aug,
                    marker="o",
                    linestyle="-",
                    color=color,
                    label=f"{model_name} (aug)",
                )

        ax.set_title(title)
        ax.set_xlabel("Real Training Data Ratio")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ordered_ratios)
        ax.set_xticklabels([f"{ratio:g}" for ratio in ordered_ratios])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(ncol=2, fontsize=9)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_gain(metric_col: str, output_path: Path, title: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for model_name, model_df in aug_frame.groupby("model"):
            ratio_to_value = {
                float(row["real_ratio"]): float(row[metric_col])
                for _, row in model_df.iterrows()
                if metric_col in row and pd.notna(row[metric_col])
            }
            x_values = [ratio for ratio in ordered_ratios if ratio in ratio_to_value]
            if not x_values:
                continue
            y_values = [ratio_to_value[ratio] for ratio in x_values]
            ax.plot(x_values, y_values, marker="o", label=str(model_name))

        ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("Real Training Data Ratio")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ordered_ratios)
        ax.set_xticklabels([f"{ratio:g}" for ratio in ordered_ratios])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    _plot_recovery(
        metric_col="test_macro_f1",
        output_path=created_files[0],
        title="Augmentation Recovery on UT-HAR (Test Macro F1)",
        ylabel="Test Macro F1",
    )
    _plot_recovery(
        metric_col="test_accuracy",
        output_path=created_files[1],
        title="Augmentation Recovery on UT-HAR (Test Accuracy)",
        ylabel="Test Accuracy",
    )
    _plot_gain(
        metric_col="augmentation_gain_macro_f1",
        output_path=created_files[2],
        title="Augmentation Gain on UT-HAR (Macro F1, positive is better)",
        ylabel="Augmentation Gain (Macro F1)",
    )
    _plot_gain(
        metric_col="augmentation_gain_accuracy",
        output_path=created_files[3],
        title="Augmentation Gain on UT-HAR (Accuracy, positive is better)",
        ylabel="Augmentation Gain (Accuracy)",
    )
    return created_files
