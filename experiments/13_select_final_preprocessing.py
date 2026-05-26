from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing_selection import apply_single_seed_ranking, apply_stability_ranking
from src.utils import ensure_dir


DEFAULT_CLOSE_TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the final preprocessing policy from official F2/F3 results."
    )
    parser.add_argument(
        "--close-tolerance",
        type=float,
        default=DEFAULT_CLOSE_TOLERANCE,
        help="Close-tolerance for stability-aware validation selection.",
    )
    return parser.parse_args()


def _load_stability_summary(root: Path) -> pd.DataFrame:
    path = root / "results" / "metrics" / "final_preprocessing_stability_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_f2_frame(root: Path) -> pd.DataFrame:
    paths = [
        root / "results" / "metrics" / "final_preprocessing_results.csv",
        root / "results" / "metrics" / "final_preprocessing_combination_results.csv",
    ]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _format_float(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{float(value):.4f}"


def _stability_table(frame: pd.DataFrame) -> str:
    lines = [
        "| rank | preprocessing | mean_val_macro_f1 | std_val_macro_f1 | mean_test_macro_f1 | std_test_macro_f1 | mean_val_test_macro_f1_gap | num_seeds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {int(row['selection_rank'])} | {row['preprocessing']} | "
            f"{_format_float(row['mean_val_macro_f1'])} | {_format_float(row['std_val_macro_f1'])} | "
            f"{_format_float(row['mean_test_macro_f1'])} | {_format_float(row['std_test_macro_f1'])} | "
            f"{_format_float(row['mean_val_test_macro_f1_gap'])} | {int(row['num_seeds'])} |"
        )
    return "\n".join(lines)


def _single_seed_table(frame: pd.DataFrame) -> str:
    lines = [
        "| rank | preprocessing | model | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['preprocessing']} | {row['model']} | "
            f"{_format_float(row['val_macro_f1'])} | {_format_float(row['test_macro_f1'])} | "
            f"{_format_float(row['val_accuracy'])} | {_format_float(row['test_accuracy'])} |"
        )
    return "\n".join(lines)


def _write_stability_decision_doc(
    output_path: Path,
    ranked_frame: pd.DataFrame,
    selected_row: pd.Series,
    raw_best_row: pd.Series,
    close_tolerance: float,
) -> None:
    ensure_dir(output_path.parent)
    seeds_text = selected_row["seeds"]
    try:
        parsed_seeds = json.loads(str(seeds_text))
        seeds_text = ", ".join(str(seed) for seed in parsed_seeds)
    except Exception:
        seeds_text = str(seeds_text)

    markdown = f"""# Final Preprocessing Decision

## Selection Source

- Decision source: multi-seed preprocessing stability check
- Summary file: `results/metrics/final_preprocessing_stability_summary.csv`

## Selection Rule

- Primary: mean validation Macro F1 across seeds
- Stability tie-break: lower std of validation Macro F1 within close tolerance
- Close tolerance: {close_tolerance:.4f}
- Test Macro F1: confirmation only

## Selected Final Preprocessing

- selected preprocessing: `{selected_row['preprocessing']}`
- model: `{selected_row['model']}`
- number of seeds: {int(selected_row['num_seeds'])}
- seeds: {seeds_text}
- mean_val_macro_f1: {_format_float(selected_row['mean_val_macro_f1'])}
- std_val_macro_f1: {_format_float(selected_row['std_val_macro_f1'])}
- mean_test_macro_f1: {_format_float(selected_row['mean_test_macro_f1'])}
- std_test_macro_f1: {_format_float(selected_row['std_test_macro_f1'])}
- mean_val_test_macro_f1_gap: {_format_float(selected_row['mean_val_test_macro_f1_gap'])}

## Best Validation Candidate

- best candidate by raw mean validation Macro F1: `{raw_best_row['preprocessing']}`
- raw best mean_val_macro_f1: {_format_float(raw_best_row['mean_val_macro_f1'])}

## Ranked Stability Results

{_stability_table(ranked_frame)}

## Leakage and Implementation Checks

- Train-statistics-based preprocessing was fit only on the selected train split.
- Fitted train statistics were applied to train/val/test.
- Deterministic smoothing was applied consistently to train/val/test.
- Augmentation was disabled in this stability check.

## Limitations

- Only selected top candidates were checked.
- Additional candidates or more seeds may change the ranking.
- Test is not used for selection.
"""
    output_path.write_text(markdown, encoding="utf-8")


def _write_single_seed_decision_doc(
    output_path: Path,
    ranked_frame: pd.DataFrame,
    selected_row: pd.Series,
    close_tolerance: float,
) -> None:
    ensure_dir(output_path.parent)
    markdown = f"""# Final Preprocessing Decision

## Selection Source

- Decision source: single-seed F2 result fallback
- Stability summary file was not found, so the current official F2 result files were used.

## Selection Rule

- Primary: validation Macro F1 from the official F2 run
- Close tolerance: {close_tolerance:.4f}
- Simplicity preference applies only when validation scores are very close.
- Test Macro F1: confirmation only

## Selected Final Preprocessing

- selected preprocessing: `{selected_row['preprocessing']}`
- model: `{selected_row['model']}`
- number of seeds: 1
- seeds: {selected_row.get('seed', '-')}
- mean_val_macro_f1: {_format_float(selected_row['val_macro_f1'])}
- std_val_macro_f1: -
- mean_test_macro_f1: {_format_float(selected_row['test_macro_f1'])}
- std_test_macro_f1: -
- mean_val_test_macro_f1_gap: {_format_float(float(selected_row['val_macro_f1']) - float(selected_row['test_macro_f1']))}

## Best Validation Candidate

- best candidate by raw validation Macro F1: `{ranked_frame.iloc[0]['preprocessing']}`
- raw best val_macro_f1: {_format_float(ranked_frame.iloc[0]['val_macro_f1'])}

## Ranked Stability Results

Stability summary has not been created yet. The current fallback ranking is shown below.

{_single_seed_table(ranked_frame)}

## Leakage and Implementation Checks

- Train-statistics-based preprocessing was fit only on the selected train split.
- Fitted train statistics were applied to train/val/test.
- Deterministic smoothing was applied consistently to train/val/test.
- Augmentation was disabled in preprocessing comparison.

## Limitations

- Only one seed is available in this fallback decision.
- A multi-seed stability check should be preferred when available.
- Test is not used for selection.
"""
    output_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_path = PROJECT_ROOT / "docs" / "final_preprocessing_decision.md"

    stability_frame = _load_stability_summary(PROJECT_ROOT)
    if not stability_frame.empty:
        ranked_frame, selected_row, raw_best_row = apply_stability_ranking(
            stability_frame,
            close_tolerance=args.close_tolerance,
        )
        _write_stability_decision_doc(
            output_path=output_path,
            ranked_frame=ranked_frame,
            selected_row=selected_row,
            raw_best_row=raw_best_row,
            close_tolerance=args.close_tolerance,
        )
        print(f"Selected preprocessing from stability summary: {selected_row['preprocessing']}")
        print(f"Saved decision document: {output_path}")
        return

    f2_frame = _load_f2_frame(PROJECT_ROOT)
    if f2_frame.empty:
        print(
            "Warning: no official preprocessing result files were found. "
            "Run F2 or the F3 stability check before selection."
        )
        return

    ranked_frame, selected_row = apply_single_seed_ranking(
        f2_frame,
        tolerance=args.close_tolerance,
    )
    _write_single_seed_decision_doc(
        output_path=output_path,
        ranked_frame=ranked_frame,
        selected_row=selected_row,
        close_tolerance=args.close_tolerance,
    )
    print(f"Selected preprocessing from single-seed fallback: {selected_row['preprocessing']}")
    print(f"Saved decision document: {output_path}")


if __name__ == "__main__":
    main()
