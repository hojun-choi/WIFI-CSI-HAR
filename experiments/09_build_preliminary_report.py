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


def _dataset_section(root: Path) -> str:
    figure_path = root / "results" / "figures" / "class_distribution_by_activity.png"
    activity_names = ", ".join(UT_HAR_CLASS_NAMES[class_id] for class_id in UT_HAR_CLASS_ORDER)
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
    ]
    if figure_path.exists():
        lines.extend(
            [
                "",
                "Dataset figure:",
                f"- `{figure_path.relative_to(root)}` uses activity names instead of numeric-only labels.",
            ]
        )
    return "\n".join(lines) + "\n"


def _current_official_status(
    benchmark_exists: bool,
    f2_single_exists: bool,
    stability_frame: pd.DataFrame,
    selected_preprocessing: str | None,
) -> str:
    f3_completed = not stability_frame.empty
    lines = [
        "## Current Official Status",
        "",
        f"- F1 benchmark completed: {'yes' if benchmark_exists else 'no'}",
        f"- F2 preprocessing comparison completed: {'yes' if f2_single_exists else 'no'}",
        f"- F3 multi-seed stability check completed: {'yes' if f3_completed else 'no'}",
        (
            f"- Final preprocessing selected: `{selected_preprocessing}`"
            if selected_preprocessing
            else "- Final preprocessing selected: not available yet"
        ),
        "- F4 low-data robustness: not completed yet." if True else "",
        "- F5 augmentation recovery: not completed yet." if True else "",
    ]
    return "\n".join(line for line in lines if line) + "\n"


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

    lines.append(
        "- `docs/final_preprocessing_decision.md` exists."
        if decision_exists
        else "- `docs/final_preprocessing_decision.md` has not been created yet."
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
        seeds_value = seeds_value.replace(",", ", ")
        selected_summary = "\n".join(
            [
                "- final selected preprocessing = `moving_average_smoothing+minmax_scaling`"
                if str(selected_row["preprocessing"]) == "moving_average_smoothing+minmax_scaling"
                else f"- final selected preprocessing = `{selected_row['preprocessing']}`",
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


def _low_data_section(rows: list[dict[str, str]]) -> str:
    if not rows:
        return """## F4. Low-data Robustness

F4 low-data robustness has not been completed yet.
"""

    ordered = sorted(rows, key=lambda row: (row["model"], float(row["real_ratio"])))
    lines = [
        "## F4. Low-data Robustness",
        "",
        "| model | real_ratio | test_macro_f1 | test_accuracy | macro_f1_drop | macro_f1_retention | accuracy_drop | accuracy_retention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['model']} | {_format_float(row['real_ratio'])} | {_format_float(row['test_macro_f1'])} | "
            f"{_format_float(row['test_accuracy'])} | {_format_float(row.get('macro_f1_drop'))} | "
            f"{_format_float(row.get('macro_f1_retention'))} | {_format_float(row.get('accuracy_drop'))} | "
            f"{_format_float(row.get('accuracy_retention'))} |"
        )
    return "\n".join(lines) + "\n"


def _augmentation_section(rows: list[dict[str, str]]) -> str:
    if not rows:
        return """## F5. Augmentation Recovery

F5 augmentation recovery has not been completed yet.
"""

    ordered = sorted(rows, key=lambda row: (row["model"], float(row["real_ratio"])))
    lines = [
        "## F5. Augmentation Recovery",
        "",
        "| model | real_ratio | test_macro_f1 | test_accuracy | augmentation_gain_macro_f1 | augmentation_gain_accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['model']} | {_format_float(row['real_ratio'])} | {_format_float(row['test_macro_f1'])} | "
            f"{_format_float(row['test_accuracy'])} | {_format_float(row.get('augmentation_gain_macro_f1'))} | "
            f"{_format_float(row.get('augmentation_gain_accuracy'))} |"
        )
    return "\n".join(lines) + "\n"


def _next_step_section() -> str:
    return """## Next Step

F4 low-data robustness should be run with:

- benchmark top3 models from F1
- final preprocessing = `moving_average_smoothing+minmax_scaling`
- training_mode = `controlled_generalization`

Command:

```powershell
python -u experiments/10_run_low_data_robustness.py --use-benchmark-top3 --preprocessing moving_average_smoothing+minmax_scaling --seed 42 --batch-size 64 2>&1 | Tee-Object -FilePath logs\\final_low_data_top3.log
```

OOM fallback:

```powershell
python -u experiments/10_run_low_data_robustness.py --use-benchmark-top3 --preprocessing moving_average_smoothing+minmax_scaling --seed 42 --batch-size 32 --overwrite 2>&1 | Tee-Object -FilePath logs\\final_low_data_top3_bs32.log
```
"""


def build_preliminary_report(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    report_path = root / "reports" / "preliminary_report.md"
    ensure_dir(report_path.parent)

    benchmark_exists = (root / "results" / "metrics" / "final_benchmark_results.csv").exists()
    f2_single_exists = (root / "results" / "metrics" / "final_preprocessing_results.csv").exists()
    stability_frame = _read_frame(root / "results" / "metrics" / "final_preprocessing_stability_summary.csv")
    low_data_rows = _read_csv_rows(root / "results" / "metrics" / "final_low_data_results.csv")
    augmentation_rows = _read_csv_rows(root / "results" / "metrics" / "final_augmentation_results.csv")
    decision_exists = (root / "docs" / "final_preprocessing_decision.md").exists()
    benchmark_selection_exists = (root / "docs" / "final_benchmark_selection.md").exists()
    selected_row = _selected_stability_row(stability_frame)
    selected_preprocessing = str(selected_row["preprocessing"]) if selected_row is not None else None

    report_text = f"""# Wi-Fi CSI HAR Workflow Status Report

This report reflects only the clean F1-F6 workflow status. Old prototype artifacts are not used as final evidence.

## Workflow Rules

- F1 uses `preprocessing=none/raw` and `training_mode=original_epoch`.
- F1 model selection uses validation `Macro F1`; test `Macro F1` is confirmation only.
- F2/F3/F4/F5 use `training_mode=controlled_generalization`.
- F2 uses the benchmark rank 1 model from F1.
- F3 selects final preprocessing primarily by mean validation `Macro F1` across seeds.
- Test `Macro F1` is confirmation only for preprocessing selection.
- F4/F5 use benchmark top3 by default and benchmark top5 only as an optional extension.
- final report should use only official final workflow outputs.

{_current_official_status(benchmark_exists, f2_single_exists, stability_frame, selected_preprocessing)}

{_dataset_section(root)}

{_benchmark_section(root)}

Benchmark selection document:

- {"`docs/final_benchmark_selection.md` exists." if benchmark_selection_exists else "`docs/final_benchmark_selection.md` has not been created yet."}

{_f2_section(root, decision_exists)}

{_f3_section(root, stability_frame, decision_exists)}

{_low_data_section(low_data_rows)}

{_augmentation_section(augmentation_rows)}

## F6. Final Report

F6 final report should be generated only after F1, F2/F3, F4, and F5 official outputs are available.

{_next_step_section()}
"""

    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main() -> None:
    report_path = build_preliminary_report()
    print(f"Saved workflow status report: {report_path}")


if __name__ == "__main__":
    main()
