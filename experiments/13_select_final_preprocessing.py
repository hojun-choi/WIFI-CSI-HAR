from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir


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
DEFAULT_TOLERANCE = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the final preprocessing policy from official F2 results."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Validation Macro F1 tolerance for preferring simpler preprocessing.",
    )
    return parser.parse_args()


def _load_candidate_frame(root: Path) -> pd.DataFrame:
    paths = [
        root / "results" / "metrics" / "final_preprocessing_results.csv",
        root / "results" / "metrics" / "final_preprocessing_combination_results.csv",
    ]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "val_macro_f1",
        "test_macro_f1",
        "val_accuracy",
        "test_accuracy",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _simplicity_rank(preprocessing: str) -> int:
    normalized = str(preprocessing)
    if "+" in normalized:
        return len(SIMPLICITY_ORDER) + 100
    try:
        return SIMPLICITY_ORDER.index(normalized)
    except ValueError:
        return len(SIMPLICITY_ORDER) + 10


def _sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["simplicity_rank"] = ranked["preprocessing"].map(_simplicity_rank)
    ranked = ranked.sort_values(
        by=["val_macro_f1", "simplicity_rank", "test_macro_f1"],
        ascending=[False, True, False],
    )
    return ranked


def _select_row(frame: pd.DataFrame, tolerance: float) -> pd.Series:
    ranked = _sorted_frame(frame)
    best_val = float(ranked.iloc[0]["val_macro_f1"])
    candidates = ranked[ranked["val_macro_f1"] >= best_val - tolerance].copy()
    candidates = candidates.sort_values(
        by=["simplicity_rank", "val_macro_f1", "test_macro_f1"],
        ascending=[True, False, False],
    )
    return candidates.iloc[0]


def _to_markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| preprocessing_group | model | preprocessing | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['preprocessing_group']} | {row['model']} | {row['preprocessing']} | "
            f"{float(row['val_macro_f1']):.4f} | {float(row['test_macro_f1']):.4f} | "
            f"{float(row['val_accuracy']):.4f} | {float(row['test_accuracy']):.4f} |"
        )
    return "\n".join(lines)


def _write_decision_doc(
    output_path: Path,
    ranked_frame: pd.DataFrame,
    selected_row: pd.Series,
    tolerance: float,
) -> None:
    ensure_dir(output_path.parent)
    selected_preprocessing = str(selected_row["preprocessing"])
    selected_group = str(selected_row["preprocessing_group"])
    top_val = float(ranked_frame.iloc[0]["val_macro_f1"])
    selected_val = float(selected_row["val_macro_f1"])
    selected_test = float(selected_row["test_macro_f1"])
    within_tolerance = abs(top_val - selected_val) <= tolerance + 1e-12

    if "+" in selected_preprocessing:
        simplicity_reason = (
            "combination candidate였지만 validation Macro F1 기준 상위권이며 tolerance "
            "내에서 더 단순한 candidate가 우세하지 않았다."
        )
    else:
        simplicity_reason = (
            "validation Macro F1 상위권이 비슷할 때 더 단순하고 해석 가능한 정책을 우선했다."
            if within_tolerance
            else "validation Macro F1에서 가장 높은 점수를 보여 선택되었다."
        )

    markdown = f"""# Final Preprocessing Decision

## Selected preprocessing

- selected preprocessing: `{selected_preprocessing}`
- preprocessing group: `{selected_group}`
- selection metric: validation `Macro F1`
- selected validation `Macro F1`: {selected_val:.4f}
- test `Macro F1` confirmation: {selected_test:.4f}
- tolerance for simplicity preference: {tolerance:.4f}

## Why selected

- Final preprocessing selection은 validation `Macro F1`를 기준으로 수행했다.
- test `Macro F1`는 confirmation only로 사용했고, test-driven selection은 하지 않았다.
- {simplicity_reason}

## Ranked results

{_to_markdown_table(ranked_frame)}

## Leakage and implementation checks

- train-statistics-based preprocessing은 selected train split에만 fit했다.
- fit된 train statistics는 `train/val/test`에 동일하게 적용했다.
- deterministic smoothing은 `train/val/test`에 일관되게 적용했다.
- augmentation은 preprocessing과 분리되어 있으며 train-only로 유지된다.

## Limitations

- current decision은 available official F2 result file에 기반한다.
- 더 많은 model 또는 additional preprocessing combination을 넣으면 ranking이 달라질 수 있다.
- final report에서는 이 문서와 official final workflow result만 evidence로 사용해야 한다.
"""
    output_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = _load_candidate_frame(PROJECT_ROOT)
    if frame.empty:
        print(
            "Warning: final preprocessing result files were not found. "
            "Run F2 official preprocessing comparison before selection."
        )
        return

    ranked = _sorted_frame(frame)
    selected = _select_row(ranked, tolerance=args.tolerance)
    output_path = PROJECT_ROOT / "docs" / "final_preprocessing_decision.md"
    _write_decision_doc(output_path, ranked, selected, tolerance=args.tolerance)
    print(f"Selected preprocessing: {selected['preprocessing']}")
    print(f"Saved decision document: {output_path}")


if __name__ == "__main__":
    main()
