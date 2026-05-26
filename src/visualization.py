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
