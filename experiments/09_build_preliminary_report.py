from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_selection import load_benchmark_results
from src.data.uthar_labels import UT_HAR_CLASS_NAMES, UT_HAR_CLASS_ORDER
from src.utils import ensure_dir


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _format_float(value: str | float | None, digits: int = 4) -> str:
    if value in ("", None):
        return "-"
    if isinstance(value, float) and pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _figure_markdown(root: Path, relative_path: str, caption: str) -> str:
    figure_path = root / relative_path
    if figure_path.exists():
        report_relative = Path("..") / Path(relative_path)
        return f"![{caption}]({report_relative.as_posix()})"
    return f"Figure not generated yet: {relative_path}"


def _figure_gallery(root: Path, figures: list[tuple[str, str]]) -> str:
    return "\n\n".join(_figure_markdown(root, path, caption) for path, caption in figures)


def _figure_row_html(root: Path, figures: list[tuple[str, str]]) -> str:
    cells: list[str] = []
    for relative_path, caption in figures:
        figure_path = root / relative_path
        if figure_path.exists():
            report_relative = (Path("..") / Path(relative_path)).as_posix()
            image_html = (
                f'<img src="{report_relative}" alt="{caption}" width="100%"><br>'
                f"<sub>{caption}</sub>"
            )
        else:
            image_html = f"<sub>Figure not generated yet: {relative_path}</sub>"
        cells.append(f"<td align=\"center\" valign=\"top\" width=\"33%\">{image_html}</td>")
    return "<table><tr>" + "".join(cells) + "</tr></table>"


def _dataset_section(root: Path) -> str:
    activity_names = ", ".join(UT_HAR_CLASS_NAMES[class_id] for class_id in UT_HAR_CLASS_ORDER)
    figures = [
        ("results/figures/class_distribution_by_activity.png", "UT-HAR class distribution by activity"),
        ("results/figures/split_size_summary.png", "UT-HAR split size summary"),
        ("results/figures/sample_csi_heatmap.png", "UT-HAR sample CSI heatmap"),
        ("results/figures/sample_csi_lineplot.png", "UT-HAR sample CSI line plot"),
        ("results/figures/sample_heatmap_by_activity.png", "UT-HAR sample heatmap by activity"),
    ]
    lines = [
        "## Dataset and Labels",
        "",
        "- dataset: `UT-HAR`",
        "- array shape in this repo: `(N, 250, 90)`",
        "- one sample = `250 CSI frame indices x 90 CSI features`",
        f"- activity labels: {activity_names}",
        "- timestep is `CSI frame index`, not directly seconds.",
        "- if `sampling_rate = fs` Hz, one sample duration is `250 / fs` seconds.",
        "- `100Hz` conversion is illustrative only and not confirmed ground truth.",
        "",
        _figure_gallery(root, figures),
    ]
    return "\n".join(lines) + "\n"


def _current_official_status(
    benchmark_exists: bool,
    f2_exists: bool,
    f3_exists: bool,
    f4_exists: bool,
    f5_exists: bool,
    f51_exists: bool,
    selected_preprocessing: str | None,
) -> str:
    lines = [
        "## Current Official Status",
        "",
        f"- F1 benchmark completed: {'yes' if benchmark_exists else 'no'}",
        f"- F2 preprocessing comparison completed: {'yes' if f2_exists else 'no'}",
        f"- F3 multi-seed stability check completed: {'yes' if f3_exists else 'no'}",
        (
            f"- Final preprocessing selected: `{selected_preprocessing}`"
            if selected_preprocessing
            else "- Final preprocessing selected: not available yet"
        ),
        f"- F4 low-data robustness: {'completed' if f4_exists else 'not completed yet'}",
        f"- F5 augmentation recovery: {'completed' if f5_exists else 'not completed yet'}",
        f"- F5.1 augmentation add ratio ablation: {'completed' if f51_exists else 'not completed yet'}",
    ]
    return "\n".join(lines) + "\n"


def _benchmark_section(root: Path) -> str:
    results_path = root / "results" / "metrics" / "final_benchmark_results.csv"
    if not results_path.exists():
        return """## F1. Original Benchmark Full Run

F1 original benchmark full run has not been completed yet.
"""

    ordered = load_benchmark_results(results_path)
    if ordered.empty:
        return """## F1. Original Benchmark Full Run

F1 original benchmark full run results exist, but no supported benchmark rows are available yet.
"""

    lines = [
        "## F1. Original Benchmark Full Run",
        "",
        "- selection metric = validation Macro F1",
        "- test Macro F1 is confirmation only",
        "",
        "| rank | model | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy | preprocessing | training_mode |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for index, (_, row) in enumerate(ordered.iterrows(), start=1):
        lines.append(
            f"| {index} | {row['model']} | {_format_float(row['val_macro_f1'])} | "
            f"{_format_float(row['test_macro_f1'])} | {_format_float(row['val_accuracy'])} | "
            f"{_format_float(row['test_accuracy'])} | {row['preprocessing']} | {row['training_mode']} |"
        )
    return "\n".join(lines) + "\n"


def _f2_table(frame: pd.DataFrame) -> str:
    lines = [
        "| rank | preprocessing | model | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for index, (_, row) in enumerate(frame.iterrows(), start=1):
        lines.append(
            f"| {index} | {row['preprocessing']} | {row['model']} | "
            f"{_format_float(row['val_macro_f1'])} | {_format_float(row['test_macro_f1'])} | "
            f"{_format_float(row['val_accuracy'])} | {_format_float(row['test_accuracy'])} |"
        )
    return "\n".join(lines)


def _f2_f3_figure_gallery(root: Path) -> str:
    figures = [
        ("results/figures/final_preprocessing_val_macro_f1.png", "Final preprocessing validation Macro F1"),
        ("results/figures/final_preprocessing_val_test_macro_f1.png", "Final preprocessing validation and test Macro F1"),
        ("results/figures/final_preprocessing_accuracy.png", "Final preprocessing accuracy comparison"),
        ("results/figures/final_preprocessing_stability_mean_val_macro_f1.png", "Preprocessing stability mean validation Macro F1"),
        ("results/figures/final_preprocessing_stability_val_test_macro_f1.png", "Preprocessing stability validation and test Macro F1"),
    ]
    return _figure_gallery(root, figures)


def _f2_section(root: Path, decision_exists: bool) -> str:
    single_frame = _read_frame(root / "results" / "metrics" / "final_preprocessing_results.csv")
    combination_frame = _read_frame(
        root / "results" / "metrics" / "final_preprocessing_combination_results.csv"
    )

    if single_frame.empty and combination_frame.empty:
        return """## F2. Preprocessing Comparison

F2 preprocessing comparison has not been completed yet.
"""

    lines = [
        "## F2. Preprocessing Comparison",
        "",
        "- F2 compares preprocessing candidates with the F1 benchmark rank 1 model.",
        "- Validation Macro F1 is the selection metric.",
        "- Test Macro F1 is confirmation only.",
        "",
    ]

    if not single_frame.empty:
        ranked_single = single_frame.copy()
        for column in ["val_macro_f1", "test_macro_f1", "val_accuracy", "test_accuracy"]:
            ranked_single[column] = pd.to_numeric(ranked_single[column], errors="coerce")
        ranked_single = ranked_single.sort_values(
            by=["val_macro_f1", "test_macro_f1", "val_accuracy"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        best_single = ranked_single.iloc[0]
        lines.extend(
            [
                "### F2 Single-method Results",
                "",
                _f2_table(ranked_single),
                "",
                f"- Best single-method candidate in F2 by validation Macro F1: `{best_single['preprocessing']}`",
                "- Final preprocessing is not selected from F2 single results alone. F3 multi-seed stability check determines the final preprocessing.",
                "",
            ]
        )

    if not combination_frame.empty:
        ranked_combination = combination_frame.copy()
        for column in ["val_macro_f1", "test_macro_f1", "val_accuracy", "test_accuracy"]:
            ranked_combination[column] = pd.to_numeric(ranked_combination[column], errors="coerce")
        ranked_combination = ranked_combination.sort_values(
            by=["val_macro_f1", "test_macro_f1", "val_accuracy"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        lines.extend(
            [
                "### F2 Combination Preprocessing Results",
                "",
                _f2_table(ranked_combination.head(min(5, len(ranked_combination)))),
                "",
                "- These top combination candidates were forwarded into the F3 multi-seed stability check.",
                "",
            ]
        )

    lines.extend(
        [
            _f2_f3_figure_gallery(root),
            "",
            (
                "- `docs/final_preprocessing_decision.md` exists."
                if decision_exists
                else "- `docs/final_preprocessing_decision.md` has not been created yet."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _f3_table(frame: pd.DataFrame) -> str:
    lines = [
        "| rank | preprocessing | mean_val_macro_f1 | std_val_macro_f1 | mean_test_macro_f1 | std_test_macro_f1 | mean_val_test_macro_f1_gap | num_seeds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        rank_value = row["selection_rank"] if "selection_rank" in row else ""
        lines.append(
            f"| {int(rank_value)} | {row['preprocessing']} | {_format_float(row['mean_val_macro_f1'])} | "
            f"{_format_float(row['std_val_macro_f1'])} | {_format_float(row['mean_test_macro_f1'])} | "
            f"{_format_float(row['std_test_macro_f1'])} | {_format_float(row['mean_val_test_macro_f1_gap'])} | "
            f"{int(row['num_seeds'])} |"
        )
    return "\n".join(lines)


def _selected_stability_row(stability_frame: pd.DataFrame) -> pd.Series | None:
    if stability_frame.empty:
        return None
    frame = stability_frame.copy()
    if "selection_rank" in frame.columns:
        frame["selection_rank"] = pd.to_numeric(frame["selection_rank"], errors="coerce")
        frame = frame.sort_values(by=["selection_rank"], ascending=[True])
    else:
        frame["mean_val_macro_f1"] = pd.to_numeric(frame["mean_val_macro_f1"], errors="coerce")
        frame = frame.sort_values(by=["mean_val_macro_f1"], ascending=[False])
    return frame.iloc[0]


def _f3_section(root: Path, stability_frame: pd.DataFrame, decision_exists: bool) -> str:
    if stability_frame.empty:
        return """## F3. Multi-seed Preprocessing Stability Check

F3 multi-seed preprocessing stability check has not been completed yet.
"""

    frame = stability_frame.copy()
    numeric_columns = [
        "selection_rank",
        "mean_val_macro_f1",
        "std_val_macro_f1",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_val_test_macro_f1_gap",
        "num_seeds",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(by=["selection_rank"], ascending=[True]).reset_index(drop=True)
    selected_row = _selected_stability_row(frame)

    selected_summary = ""
    if selected_row is not None:
        seeds_value = str(selected_row["seeds"]).strip("[]")
        seeds_value = ", ".join(part.strip() for part in seeds_value.split(",") if part.strip())
        selected_summary = "\n".join(
            [
                f"- final selected preprocessing = `{selected_row['preprocessing']}`",
                f"- model = `{selected_row['model']}`",
                f"- seeds = {seeds_value}",
                f"- mean_val_macro_f1 = {_format_float(selected_row['mean_val_macro_f1'])}",
                f"- std_val_macro_f1 = {_format_float(selected_row['std_val_macro_f1'])}",
                f"- mean_test_macro_f1 = {_format_float(selected_row['mean_test_macro_f1'])}",
                f"- std_test_macro_f1 = {_format_float(selected_row['std_test_macro_f1'])}",
            ]
        )

    lines = [
        "## F3. Multi-seed Preprocessing Stability Check",
        "",
        "- F3 was added because single-seed F2 showed close preprocessing candidates.",
        "- Final selection is based on mean validation Macro F1 across seeds.",
        "- Stability is used only as a meaningful tie-break.",
        "- Test Macro F1 is confirmation only.",
        "",
        _f3_table(frame),
        "",
        "### Final Preprocessing Decision",
        "",
        selected_summary,
        "",
        (
            "- `docs/final_preprocessing_decision.md` exists and records the official final preprocessing decision."
            if decision_exists
            else "- `docs/final_preprocessing_decision.md` has not been created yet."
        ),
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


def _low_data_figure_gallery(root: Path) -> str:
    figures = [
        ("results/figures/final_low_data_macro_f1_by_ratio.png", "Low-data robustness raw test Macro F1 by train ratio"),
        ("results/figures/final_low_data_accuracy_by_ratio.png", "Low-data robustness raw test accuracy by train ratio"),
        ("results/figures/final_low_data_macro_f1_retention_by_ratio.png", "Normalized Macro F1 retention under reduced training data"),
        ("results/figures/final_low_data_macro_f1_drop_by_ratio.png", "Macro F1 drop under reduced training data"),
        ("results/figures/final_low_data_25_10_summary.png", "Low-data robustness summary at 25% and 10% (raw test Macro F1)"),
    ]
    return _figure_gallery(root, figures)


def _f4_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    numeric_columns = ["real_ratio", "test_macro_f1", "test_accuracy", "macro_f1_retention"]
    working = frame.copy()
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    lines = ["### F4 Interpretation", ""]
    ratio_025 = working[working["real_ratio"] == 0.25].copy()
    ratio_01 = working[working["real_ratio"] == 0.1].copy()

    if not ratio_025.empty:
        best_025 = ratio_025.sort_values(by=["test_macro_f1"], ascending=[False]).iloc[0]
        lines.append(
            f"- At 25% training data, the strongest model by test Macro F1 is `{best_025['model']}` "
            f"({_format_float(best_025['test_macro_f1'])})."
        )
    if not ratio_01.empty:
        best_01 = ratio_01.sort_values(by=["test_macro_f1"], ascending=[False]).iloc[0]
        lines.append(
            f"- At 10% training data, the strongest model by test Macro F1 is `{best_01['model']}` "
            f"({_format_float(best_01['test_macro_f1'])})."
        )
        min_row = ratio_01.sort_values(by=["test_macro_f1"], ascending=[True]).iloc[0]
        if float(min_row["test_macro_f1"]) < 0.2:
            lines.append(
                f"- `{min_row['model']}` collapses sharply at 10%, with test Macro F1 dropping to "
                f"{_format_float(min_row['test_macro_f1'])}."
            )

    ratio_05 = working[working["real_ratio"] == 0.5].copy()
    if not ratio_05.empty and ratio_05["test_macro_f1"].notna().all():
        spread_05 = float(ratio_05["test_macro_f1"].max() - ratio_05["test_macro_f1"].min())
        if spread_05 < 0.03:
            lines.append("- At 50% training data, all benchmark top3 models remain relatively stable.")
        else:
            lines.append("- At 50% training data, model differences are already visible.")

    if not ratio_025.empty and ratio_025["test_macro_f1"].notna().all():
        spread_025 = float(ratio_025["test_macro_f1"].max() - ratio_025["test_macro_f1"].min())
        if spread_025 >= 0.03:
            lines.append("- At 25%, model differences become clearer.")

    if "macro_f1_retention" in working.columns and ratio_01 is not None and not ratio_01.empty:
        best_retention = ratio_01.sort_values(by=["macro_f1_retention"], ascending=[False]).iloc[0]
        lines.append(
            f"- At 10%, the best Macro F1 retention belongs to `{best_retention['model']}` "
            f"({_format_float(best_retention['macro_f1_retention'])})."
        )
        lines.append(
            "- The raw Macro F1 figure and the retention figure should be read differently: "
            "raw Macro F1 shows actual performance, while retention is normalized to each model's own 100% baseline."
        )
        lines.append(
            "- In the retention plot, 100% training data appears as 1.0 by definition because each model is divided by its own full-data result."
        )

    return "\n".join(lines) + "\n"


def _low_data_section(root: Path) -> str:
    frame = _read_frame(root / "results" / "metrics" / "final_low_data_results.csv")
    if frame.empty:
        return """## F4. Low-data Robustness

F4 low-data robustness has not been completed yet.
"""

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
    ordered = frame.sort_values(by=["model", "real_ratio"], ascending=[True, False]).reset_index(drop=True)

    lines = [
        "## F4. Low-data Robustness",
        "",
        "| model | real_ratio | test_macro_f1 | test_accuracy | macro_f1_drop | macro_f1_retention | accuracy_drop | accuracy_retention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['model']} | {_format_float(row['real_ratio'])} | {_format_float(row['test_macro_f1'])} | "
            f"{_format_float(row['test_accuracy'])} | {_format_float(row.get('macro_f1_drop'))} | "
            f"{_format_float(row.get('macro_f1_retention'))} | {_format_float(row.get('accuracy_drop'))} | "
            f"{_format_float(row.get('accuracy_retention'))} |"
        )

    lines.extend(
        [
            "",
            _f4_interpretation(ordered),
            _low_data_figure_gallery(root),
        ]
    )
    return "\n".join(lines) + "\n"


def _augmentation_figure_gallery(root: Path) -> str:
    figures = [
        (
            "results/figures/final_augmentation_gain_macro_f1_by_ratio.png",
            "Offline appended augmentation Macro F1 gain by train ratio",
        ),
        (
            "results/figures/final_augmentation_gain_accuracy_by_ratio.png",
            "Offline appended augmentation accuracy gain by train ratio",
        ),
        (
            "results/figures/final_augmentation_macro_f1_aug_vs_no_aug.png",
            "Real data plus synthetic augmentation versus no-augmentation Macro F1",
        ),
        (
            "results/figures/final_augmentation_25_10_summary.png",
            "Offline appended augmentation summary at 25% and 10%",
        ),
        (
            "results/figures/final_augmentation_gain_heatmap.png",
            "Offline appended augmentation gain heatmap",
        ),
    ]
    return _figure_gallery(root, figures)


def _augmentation_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty or "augmentation_gain_macro_f1" not in frame.columns:
        return ""

    working = frame.copy()
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
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    valid_gains = working["augmentation_gain_macro_f1"].dropna()
    if valid_gains.empty:
        return ""

    positive_count = int((valid_gains > 0).sum())
    negative_count = int((valid_gains < 0).sum())
    zero_count = int((valid_gains == 0).sum())
    largest_positive = working.sort_values(by=["augmentation_gain_macro_f1"], ascending=[False]).iloc[0]
    largest_negative = working.sort_values(by=["augmentation_gain_macro_f1"], ascending=[True]).iloc[0]

    lines = [
        "### F5 Interpretation",
        "",
        "- F5 evaluates offline appended synthetic augmentation against the F4 no-augmentation baseline.",
        "- Synthetic samples are generated only from the selected train subset. Validation/test are never augmented.",
        (
            f"- Positive Macro F1 gain rows: {positive_count}; "
            f"negative rows: {negative_count}; zero rows: {zero_count}."
        ),
        (
            f"- Largest positive Macro F1 gain: `{largest_positive['model']}` at "
            f"`real_ratio={float(largest_positive['real_ratio']):g}` "
            f"({_format_float(largest_positive['augmentation_gain_macro_f1'])})."
        ),
        (
            f"- Largest negative Macro F1 gain: `{largest_negative['model']}` at "
            f"`real_ratio={float(largest_negative['real_ratio']):g}` "
            f"({_format_float(largest_negative['augmentation_gain_macro_f1'])})."
        ),
    ]

    if positive_count > negative_count:
        lines.append("- Overall, augmentation improves more model/ratio pairs than it degrades.")
    elif negative_count > positive_count:
        lines.append(
            "- Overall, the current train-only augmentation policy does not consistently recover low-data performance."
        )
    else:
        lines.append("- Overall, augmentation shows mixed effects across model/ratio pairs.")

    if "test_macro_f1" in working.columns and "no_aug_test_macro_f1" in working.columns:
        low_absolute = working[
            (working["augmentation_gain_macro_f1"] > 0) & (working["test_macro_f1"] < 0.2)
        ]
        if not low_absolute.empty:
            lines.append(
                "- Positive gains should be interpreted carefully when the absolute Macro F1 remains low."
            )

    return "\n".join(lines) + "\n"


def _augmentation_section(root: Path) -> str:
    frame = _read_frame(root / "results" / "metrics" / "final_augmentation_results.csv")
    if frame.empty:
        return """## F5. Augmentation Recovery

F5 augmentation recovery has not been completed yet under the current offline appended synthetic-data design.
"""

    numeric_columns = [
        "real_ratio",
        "test_macro_f1",
        "test_accuracy",
        "no_aug_test_macro_f1",
        "no_aug_test_accuracy",
        "augmentation_gain_macro_f1",
        "augmentation_gain_accuracy",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    ordered = frame.sort_values(by=["model", "real_ratio"], ascending=[True, False]).reset_index(drop=True)
    lines = [
        "## F5. Augmentation Recovery",
        "",
        "- This official F5 design uses offline appended synthetic samples, not on-the-fly augmentation.",
        "- Synthetic samples are generated only from the selected real train subset. Validation/test are never augmented.",
        "",
        "| model | real_ratio | selected_real_train_size | synthetic_train_size | effective_train_size | augmentation_add_ratio | no_aug_test_macro_f1 | test_macro_f1 | no_aug_test_accuracy | test_accuracy | augmentation_gain_macro_f1 | augmentation_gain_accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['model']} | {_format_float(row['real_ratio'])} | "
            f"{_format_float(row.get('selected_real_train_size'), digits=0)} | "
            f"{_format_float(row.get('synthetic_train_size'), digits=0)} | "
            f"{_format_float(row.get('effective_train_size'), digits=0)} | "
            f"{_format_float(row.get('augmentation_add_ratio'))} | "
            f"{_format_float(row.get('no_aug_test_macro_f1'))} | {_format_float(row['test_macro_f1'])} | "
            f"{_format_float(row.get('no_aug_test_accuracy'))} | {_format_float(row['test_accuracy'])} | "
            f"{_format_float(row.get('augmentation_gain_macro_f1'))} | "
            f"{_format_float(row.get('augmentation_gain_accuracy'))} |"
        )
    lines.extend(
        [
            "",
            _augmentation_interpretation(ordered),
            _augmentation_figure_gallery(root),
        ]
    )
    return "\n".join(lines) + "\n"


def _augmentation_ablation_figure_gallery(root: Path) -> str:
    figures = [
        (
            "results/figures/final_augmentation_ablation_macro_f1_by_add_ratio.png",
            "Augmentation ablation raw test Macro F1 by add ratio",
        ),
        (
            "results/figures/final_augmentation_ablation_gain_by_add_ratio.png",
            "Augmentation add ratio ablation Macro F1 gain",
        ),
        (
            "results/figures/final_augmentation_ablation_best_add_ratio_by_condition.png",
            "Best augmentation_add_ratio by condition",
        ),
        (
            "results/figures/final_augmentation_ablation_heatmap.png",
            "Augmentation add ratio ablation heatmap",
        ),
    ]
    return _figure_gallery(root, figures)


def _augmentation_ablation_aggregate_figure_gallery(root: Path) -> str:
    figures = [
        (
            "results/figures/final_augmentation_ablation_add_ratio_summary_macro_f1.png",
            "Aggregate F5.1 raw test Macro F1 and gain by augmentation_add_ratio",
        ),
        (
            "results/figures/final_augmentation_ablation_add_ratio_summary_gain.png",
            "Aggregate F5.1 Macro F1 gain by augmentation_add_ratio",
        ),
        (
            "results/figures/final_augmentation_ablation_add_ratio_positive_rate.png",
            "Aggregate F5.1 positive-gain rate by augmentation_add_ratio",
        ),
        (
            "results/figures/final_augmentation_ablation_add_ratio_accuracy_summary.png",
            "Aggregate F5.1 raw test Accuracy by augmentation_add_ratio",
        ),
    ]
    return _figure_gallery(root, figures)


def _augmentation_ablation_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty or "augmentation_gain_macro_f1" not in frame.columns:
        return ""

    working = frame.copy()
    for column in [
        "real_ratio",
        "augmentation_add_ratio",
        "selected_real_train_size",
        "synthetic_train_size",
        "effective_train_size",
        "no_aug_test_macro_f1",
        "test_macro_f1",
        "augmentation_gain_macro_f1",
    ]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    lines = ["### F5.1 Interpretation", ""]
    gain_counts = (
        working.assign(
            gain_sign=working["augmentation_gain_macro_f1"].apply(
                lambda value: "positive" if value > 0 else ("negative" if value < 0 else "zero")
            )
        )
        .groupby("augmentation_add_ratio")["gain_sign"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for add_ratio, row in gain_counts.sort_index().iterrows():
        lines.append(
            f"- augmentation_add_ratio={float(add_ratio):g}: "
            f"positive {int(row.get('positive', 0))}, "
            f"negative {int(row.get('negative', 0))}, "
            f"zero {int(row.get('zero', 0))}."
        )

    mean_gain = (
        working.groupby("augmentation_add_ratio", as_index=False)["augmentation_gain_macro_f1"]
        .mean()
        .sort_values(by=["augmentation_add_ratio"], ascending=[True])
    )
    for _, row in mean_gain.iterrows():
        lines.append(
            f"- Mean augmentation_gain_macro_f1 at augmentation_add_ratio={float(row['augmentation_add_ratio']):g}: "
            f"{_format_float(row['augmentation_gain_macro_f1'])}."
        )

    mean_test = (
        working.groupby("augmentation_add_ratio", as_index=False)["test_macro_f1"]
        .mean()
        .sort_values(by=["augmentation_add_ratio"], ascending=[True])
    )
    for _, row in mean_test.iterrows():
        lines.append(
            f"- Mean test_macro_f1 at augmentation_add_ratio={float(row['augmentation_add_ratio']):g}: "
            f"{_format_float(row['test_macro_f1'])}."
        )

    best_rows = (
        working.sort_values(
            by=["model", "real_ratio", "test_macro_f1", "augmentation_gain_macro_f1"],
            ascending=[True, True, False, False],
        )
        .groupby(["model", "real_ratio"], as_index=False)
        .first()
    )
    best_summary = ", ".join(
        f"{row['model']}@{float(row['real_ratio']):g}->{float(row['augmentation_add_ratio']):g}"
        for _, row in best_rows.iterrows()
    )
    lines.append(f"- Best augmentation_add_ratio by condition: {best_summary}.")

    best_add_counts = best_rows["augmentation_add_ratio"].value_counts().sort_index()
    if not best_add_counts.empty:
        most_frequent_add_ratio = float(best_add_counts.idxmax())
        lines.append(
            f"- augmentation_add_ratio={most_frequent_add_ratio:g} is most often best across model/ratio conditions."
        )
    if not mean_gain.empty:
        best_mean_row = mean_gain.sort_values(by=["augmentation_gain_macro_f1"], ascending=[False]).iloc[0]
        worst_mean_row = mean_gain.sort_values(by=["augmentation_gain_macro_f1"], ascending=[True]).iloc[0]
        lines.append(
            f"- The most stable average gain appears at augmentation_add_ratio={float(best_mean_row['augmentation_add_ratio']):g}, "
            f"while augmentation_add_ratio={float(worst_mean_row['augmentation_add_ratio']):g} is weakest on average."
        )

    severe_collapse = working.sort_values(by=["test_macro_f1", "augmentation_gain_macro_f1"], ascending=[True, True]).iloc[0]
    if float(severe_collapse["augmentation_gain_macro_f1"]) < -0.2:
        lines.append(
            f"- Severe collapse case: `{severe_collapse['model']}` at `real_ratio={float(severe_collapse['real_ratio']):g}` "
            f"with augmentation_add_ratio={float(severe_collapse['augmentation_add_ratio']):g} drops to "
            f"test_macro_f1={_format_float(severe_collapse['test_macro_f1'])}."
        )

    lines.append("- Synthetic data is not simply the more the better; augmentation strength should be tuned by model and real_ratio.")

    return "\n".join(lines) + "\n"


def _augmentation_ablation_aggregate_table(frame: pd.DataFrame) -> str:
    lines = [
        "| augmentation_add_ratio | n_conditions | mean_test_macro_f1 | std_test_macro_f1 | mean_augmentation_gain_macro_f1 | std_augmentation_gain_macro_f1 | positive_gain_rate | positive_gain_count | source mix |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in frame.iterrows():
        source_mix = (
            f"reused={int(pd.to_numeric(row.get('reused_final_f5_count'), errors='coerce') or 0)}, "
            f"trained={int(pd.to_numeric(row.get('trained_ablation_count'), errors='coerce') or 0)}, "
            f"existing={int(pd.to_numeric(row.get('existing_ablation_count'), errors='coerce') or 0)}"
        )
        lines.append(
            f"| {_format_float(row['augmentation_add_ratio'])} | {int(row['n_conditions'])} | "
            f"{_format_float(row['mean_test_macro_f1'])} | {_format_float(row['std_test_macro_f1'])} | "
            f"{_format_float(row['mean_augmentation_gain_macro_f1'])} | {_format_float(row['std_augmentation_gain_macro_f1'])} | "
            f"{_format_float(float(row['positive_gain_rate']) * 100.0, digits=1)}% | "
            f"{int(row['positive_gain_count'])} | {source_mix} |"
        )
    return "\n".join(lines)


def _augmentation_ablation_aggregate_interpretation(summary_frame: pd.DataFrame) -> str:
    if summary_frame.empty:
        return ""
    working = summary_frame.copy()
    numeric_columns = [
        "augmentation_add_ratio",
        "n_conditions",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_augmentation_gain_macro_f1",
        "std_augmentation_gain_macro_f1",
        "positive_gain_rate",
        "positive_gain_count",
        "zero_or_negative_gain_count",
    ]
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.sort_values(by=["augmentation_add_ratio"], ascending=[True]).reset_index(drop=True)

    best_mean_macro = working.sort_values(
        by=["mean_test_macro_f1", "mean_augmentation_gain_macro_f1"],
        ascending=[False, False],
    ).iloc[0]
    best_mean_gain = working.sort_values(
        by=["mean_augmentation_gain_macro_f1", "mean_test_macro_f1"],
        ascending=[False, False],
    ).iloc[0]
    best_positive_rate = working.sort_values(
        by=["positive_gain_rate", "mean_augmentation_gain_macro_f1"],
        ascending=[False, False],
    ).iloc[0]
    worst_mean_gain = working.sort_values(
        by=["mean_augmentation_gain_macro_f1", "std_augmentation_gain_macro_f1"],
        ascending=[True, False],
    ).iloc[0]
    highest_std = working.sort_values(
        by=["std_augmentation_gain_macro_f1", "std_test_macro_f1"],
        ascending=[False, False],
    ).iloc[0]

    lines = [
        "### F5.1 Aggregate Interpretation",
        "",
        "- These aggregate statistics collapse the full F5.1 grid into three augmentation_add_ratio groups, each covering up to 9 model/real_ratio conditions.",
        f"- Highest mean_test_macro_f1: augmentation_add_ratio={float(best_mean_macro['augmentation_add_ratio']):g} ({_format_float(best_mean_macro['mean_test_macro_f1'])}).",
        f"- Highest mean_augmentation_gain_macro_f1: augmentation_add_ratio={float(best_mean_gain['augmentation_add_ratio']):g} ({_format_float(best_mean_gain['mean_augmentation_gain_macro_f1'])}).",
        f"- Best positive_gain_rate: augmentation_add_ratio={float(best_positive_rate['augmentation_add_ratio']):g} "
        f"({_format_float(float(best_positive_rate['positive_gain_rate']) * 100.0, digits=1)}%, "
        f"{int(best_positive_rate['positive_gain_count'])}/{int(best_positive_rate['n_conditions'])}).",
    ]
    lines.append(
        "- This should not be read as a clear augmentation success. The aggregate view shows limited and condition-dependent benefit rather than a robust universal fix."
    )
    lines.append(
        "- augmentation_add_ratio=0.5 is the most stable aggregate choice: it has the highest mean test Macro F1, the highest mean Macro F1 gain, and it ties for the best positive-gain rate."
    )
    lines.append(
        "- augmentation_add_ratio=1.0 is partially useful, but it is not the aggregate best and it improves only 5 of 9 conditions, the same positive-gain count as 0.5."
    )
    if float(highest_std["augmentation_add_ratio"]) == 2.0 or float(worst_mean_gain["augmentation_add_ratio"]) == 2.0:
        lines.append(
            "- Higher synthetic ratio is not consistently better. The aggregate summary shows that augmentation_add_ratio=2.0 has the weakest average gain and the highest instability, indicating that over-augmentation can harm robustness."
        )
    else:
        lines.append(
            "- Higher synthetic ratio is not consistently better across the 3x3 condition grid."
        )
    lines.append(
        f"- The most unstable aggregate setting is augmentation_add_ratio={float(highest_std['augmentation_add_ratio']):g} "
        f"(std_augmentation_gain_macro_f1={_format_float(highest_std['std_augmentation_gain_macro_f1'])})."
    )
    lines.append(
        "- Positive-gain rate is only 5/9 for both 0.5 and 1.0, so even the better settings should be interpreted as partial recovery, not broad success."
    )
    lines.append(
        "- The negative result is informative: simple offline_append synthetic augmentation can help some weak or collapsed conditions, but it does not generalize as a consistently reliable solution for CSI HAR low-data robustness."
    )
    lines.append(
        "- A likely reason is that CSI is not image-like data. Naive synthetic transforms may fail to preserve physically meaningful channel structure, and too much synthetic data can distort the training distribution."
    )
    lines.append(
        "- Future work should prioritize physics-aware augmentation, class-conditional augmentation, model-specific augmentation strength, and signal-structure-aware or environment-aware synthetic generation."
    )
    return "\n".join(lines) + "\n"


def _augmentation_ablation_section(root: Path) -> str:
    frame = _read_frame(root / "results" / "metrics" / "final_augmentation_ratio_ablation_results.csv")
    if frame.empty:
        return """## F5.1 Augmentation Add Ratio Ablation

F5.1 augmentation_add_ratio ablation has not been run yet.
"""

    for column in [
        "real_ratio",
        "augmentation_add_ratio",
        "selected_real_train_size",
        "synthetic_train_size",
        "effective_train_size",
        "no_aug_test_macro_f1",
        "test_macro_f1",
        "augmentation_gain_macro_f1",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    ordered = frame.sort_values(
        by=["model", "real_ratio", "augmentation_add_ratio"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    lines = [
        "## F5.1 Augmentation Add Ratio Ablation",
        "",
        "- This ablation compares augmentation_add_ratio=0.5, 1.0, 2.0 under the same offline_append design.",
        "- It reuses the official final preprocessing: `moving_average_smoothing+minmax_scaling`.",
        "- It uses benchmark top3 models: `ResNet18`, `LeNet`, `ResNet101`.",
        "- It keeps the same real-ratio settings: `0.5`, `0.25`, `0.1`.",
        "- Each augmented result is compared only against the F4 no-augmentation baseline with the same model, same real_ratio, and same preprocessing.",
        "- `augmentation_add_ratio=1.0` rows come from the completed official F5 run when `source=reused_final_f5`.",
        "- `augmentation_add_ratio=0.5` and `2.0` rows are ablation-specific runs when `source=trained_ablation`.",
        "",
        "| model | real_ratio | augmentation_add_ratio | selected_real_train_size | synthetic_train_size | effective_train_size | no_aug_test_macro_f1 | test_macro_f1 | augmentation_gain_macro_f1 | source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['model']} | {_format_float(row['real_ratio'])} | {_format_float(row['augmentation_add_ratio'])} | "
            f"{_format_float(row.get('selected_real_train_size'), digits=0)} | "
            f"{_format_float(row.get('synthetic_train_size'), digits=0)} | "
            f"{_format_float(row.get('effective_train_size'), digits=0)} | "
            f"{_format_float(row.get('no_aug_test_macro_f1'))} | {_format_float(row.get('test_macro_f1'))} | "
            f"{_format_float(row.get('augmentation_gain_macro_f1'))} | {row.get('source', '-')} |"
        )
    lines.extend(
        [
            "",
            _augmentation_ablation_interpretation(ordered),
            _augmentation_ablation_figure_gallery(root),
        ]
    )
    return "\n".join(lines) + "\n"


def _augmentation_ablation_aggregate_section(root: Path) -> str:
    summary_frame = _read_frame(
        root / "results" / "metrics" / "final_augmentation_ratio_ablation_summary_by_add_ratio.csv"
    )
    if summary_frame.empty:
        return """### F5.1 Aggregate Summary by augmentation_add_ratio

F5.1 aggregate summary by augmentation_add_ratio has not been generated yet.
"""

    ordered = summary_frame.copy()
    for column in [
        "augmentation_add_ratio",
        "n_conditions",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_augmentation_gain_macro_f1",
        "std_augmentation_gain_macro_f1",
        "positive_gain_rate",
        "positive_gain_count",
        "zero_or_negative_gain_count",
        "reused_final_f5_count",
        "trained_ablation_count",
        "existing_ablation_count",
    ]:
        if column in ordered.columns:
            ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    ordered = ordered.sort_values(by=["augmentation_add_ratio"], ascending=[True]).reset_index(drop=True)

    lines = [
        "### F5.1 Aggregate Summary by augmentation_add_ratio",
        "",
        "- This aggregate layer groups only by `augmentation_add_ratio` and summarizes the full 3 models x 3 real_ratios grid.",
        "- Each row below aggregates across up to 9 condition rows from the official F5.1 CSV.",
        "",
        _augmentation_ablation_aggregate_table(ordered),
        "",
        _augmentation_ablation_aggregate_interpretation(ordered),
        _figure_markdown(
            root,
            "results/figures/final_augmentation_ablation_add_ratio_summary_macro_f1.png",
            "Aggregate F5.1 raw test Macro F1 and gain by augmentation_add_ratio",
        ),
        "",
        _figure_row_html(
            root,
            [
                (
                    "results/figures/final_augmentation_ablation_add_ratio_summary_gain.png",
                    "Aggregate F5.1 Macro F1 gain by augmentation_add_ratio",
                ),
                (
                    "results/figures/final_augmentation_ablation_add_ratio_positive_rate.png",
                    "Aggregate F5.1 positive-gain rate by augmentation_add_ratio",
                ),
                (
                    "results/figures/final_augmentation_ablation_add_ratio_accuracy_summary.png",
                    "Aggregate F5.1 raw test Accuracy by augmentation_add_ratio",
                ),
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def _next_step_section(f51_exists: bool) -> str:
    if f51_exists:
        return """## Next Step

F5.1 is complete, so the next step is final report regeneration and packaging from the official F1-F5.1 outputs.

Command:

```powershell
python experiments/09_build_preliminary_report.py
```
"""

    return """## Next Step

F5.1 augmentation_add_ratio ablation should be run before final packaging.

Command:

```powershell
python -u experiments/17_run_augmentation_ratio_ablation.py --use-benchmark-top3 --preprocessing moving_average_smoothing+minmax_scaling --augmentation-add-ratios 0.5 1.0 2.0 --seed 42 --batch-size 64 2>&1 | Tee-Object -FilePath logs\\final_augmentation_ratio_ablation_top3.log
```

OOM fallback:

```powershell
python -u experiments/17_run_augmentation_ratio_ablation.py --use-benchmark-top3 --preprocessing moving_average_smoothing+minmax_scaling --augmentation-add-ratios 0.5 1.0 2.0 --seed 42 --batch-size 32 --overwrite 2>&1 | Tee-Object -FilePath logs\\final_augmentation_ratio_ablation_top3_bs32.log
```
"""


def build_preliminary_report(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    report_path = root / "reports" / "preliminary_report.md"
    ensure_dir(report_path.parent)

    benchmark_exists = (root / "results" / "metrics" / "final_benchmark_results.csv").exists()
    f2_exists = (root / "results" / "metrics" / "final_preprocessing_results.csv").exists()
    stability_frame = _read_frame(root / "results" / "metrics" / "final_preprocessing_stability_summary.csv")
    low_data_exists = (root / "results" / "metrics" / "final_low_data_results.csv").exists()
    augmentation_frame = _read_frame(root / "results" / "metrics" / "final_augmentation_results.csv")
    f5_exists = not augmentation_frame.empty
    augmentation_ablation_frame = _read_frame(
        root / "results" / "metrics" / "final_augmentation_ratio_ablation_results.csv"
    )
    f51_exists = not augmentation_ablation_frame.empty
    decision_exists = (root / "docs" / "final_preprocessing_decision.md").exists()
    benchmark_selection_exists = (root / "docs" / "final_benchmark_selection.md").exists()
    selected_row = _selected_stability_row(stability_frame)
    selected_preprocessing = str(selected_row["preprocessing"]) if selected_row is not None else None

    report_text = f"""# Wi-Fi CSI HAR Workflow Status Report

This report reflects only the clean F1-F5.1 workflow status. Old prototype artifacts are not used as final evidence.

## Workflow Rules

- F1 uses `preprocessing=none/raw` and `training_mode=original_epoch`.
- F1 model selection uses validation `Macro F1`; test `Macro F1` is confirmation only.
- F2/F3/F4/F5/F5.1 use `training_mode=controlled_generalization`.
- F2 uses the benchmark rank 1 model from F1.
- F3 selects final preprocessing primarily by mean validation `Macro F1` across seeds.
- Test `Macro F1` is confirmation only for preprocessing selection.
- F4 uses benchmark top3 models by default.
- final report should use only official final workflow outputs.

{_current_official_status(benchmark_exists, f2_exists, not stability_frame.empty, low_data_exists, f5_exists, f51_exists, selected_preprocessing)}

{_dataset_section(root)}

{_benchmark_section(root)}

Benchmark selection document:

- {"`docs/final_benchmark_selection.md` exists." if benchmark_selection_exists else "`docs/final_benchmark_selection.md` has not been created yet."}

{_f2_section(root, decision_exists)}

{_f3_section(root, stability_frame, decision_exists)}

{_low_data_section(root)}

{_augmentation_section(root)}

{_augmentation_ablation_section(root)}

{_augmentation_ablation_aggregate_section(root)}

## F6. Final Report

F6 final report can now be generated from the official F1-F5.1 outputs. Test-set figures and tables above should be reused as final evidence.

{_next_step_section(f51_exists)}
"""

    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main() -> None:
    report_path = build_preliminary_report()
    print(f"Saved workflow status report: {report_path}")


if __name__ == "__main__":
    main()
