from __future__ import annotations

import csv
import sys
from pathlib import Path

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


def _format_float(value: str | float | None, digits: int = 4) -> str:
    if value in ("", None):
        return "-"
    return f"{float(value):.{digits}f}"


def _build_dataset_section(root: Path) -> str:
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


def _build_benchmark_section(root: Path) -> str:
    results_path = root / "results" / "metrics" / "final_benchmark_results.csv"
    if not results_path.exists():
        return """## F1. Original Benchmark Full Run

F1 original benchmark full run has not been completed yet.
"""

    full_rows = _read_csv_rows(results_path)
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

    unsupported_rows = [
        row for row in full_rows if row.get("support_status", "supported") != "supported"
    ]
    if unsupported_rows:
        lines.extend(["", "Unsupported benchmark models:"])
        for row in unsupported_rows:
            lines.append(
                f"- `{row['model']}`: {row.get('unsupported_reason', 'no reason recorded')}"
            )

    return "\n".join(lines) + "\n"


def _build_preprocessing_section(rows: list[dict[str, str]], decision_exists: bool) -> str:
    if not rows:
        return """## F2. Preprocessing Comparison

F2 preprocessing comparison has not been completed yet.
"""

    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["val_macro_f1"]),
            float(row["test_macro_f1"]),
            float(row["val_accuracy"]),
        ),
        reverse=True,
    )
    best_row = ordered[0]
    lines = [
        "## F2. Preprocessing Comparison",
        "",
        "- selection metric = validation Macro F1",
        "- test Macro F1 is confirmation only",
        "",
        "| preprocessing_group | model | preprocessing | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['preprocessing_group']} | {row['model']} | {row['preprocessing']} | "
            f"{_format_float(row['val_macro_f1'])} | {_format_float(row['test_macro_f1'])} | "
            f"{_format_float(row['val_accuracy'])} | {_format_float(row['test_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            f"- current best preprocessing by validation `Macro F1`: `{best_row['preprocessing']}`",
            (
                "- `docs/final_preprocessing_decision.md` exists."
                if decision_exists
                else "- `docs/final_preprocessing_decision.md` has not been created yet."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _build_low_data_section(rows: list[dict[str, str]]) -> str:
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


def _build_augmentation_section(rows: list[dict[str, str]]) -> str:
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


def build_preliminary_report(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    report_path = root / "reports" / "preliminary_report.md"
    ensure_dir(report_path.parent)

    preprocessing_rows = _read_csv_rows(
        root / "results" / "metrics" / "final_preprocessing_results.csv"
    )
    low_data_rows = _read_csv_rows(root / "results" / "metrics" / "final_low_data_results.csv")
    augmentation_rows = _read_csv_rows(
        root / "results" / "metrics" / "final_augmentation_results.csv"
    )
    decision_exists = (root / "docs" / "final_preprocessing_decision.md").exists()
    benchmark_selection_exists = (root / "docs" / "final_benchmark_selection.md").exists()

    report_text = f"""# Wi-Fi CSI HAR Workflow Status Report

This report reflects only the clean F1-F6 workflow status. Old prototype artifacts are not used as final evidence.

## Workflow Rules

- F1 uses `preprocessing=none/raw` and `training_mode=original_epoch`.
- F1 model selection uses validation `Macro F1`; test `Macro F1` is confirmation only.
- F2/F3/F4/F5 use `training_mode=controlled_generalization`.
- F2 uses the benchmark rank 1 model from F1.
- F4/F5 use benchmark top3 by default and benchmark top5 only as an optional extension.
- final report should use only official final workflow outputs.

{_build_dataset_section(root)}

{_build_benchmark_section(root)}

Benchmark selection document:

- {"`docs/final_benchmark_selection.md` exists." if benchmark_selection_exists else "`docs/final_benchmark_selection.md` has not been created yet."}

{_build_preprocessing_section(preprocessing_rows, decision_exists)}

{_build_low_data_section(low_data_rows)}

{_build_augmentation_section(augmentation_rows)}

## F6. Final Report

F6 final report should be generated only after F1, F2/F3, F4, and F5 official outputs are available.
"""

    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main() -> None:
    report_path = build_preliminary_report()
    print(f"Saved workflow status report: {report_path}")


if __name__ == "__main__":
    main()
