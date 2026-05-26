from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_selection import load_benchmark_results
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select benchmark rank 1 / top3 / top5 from F1 results.")
    parser.add_argument("--top-k", type=int, default=3, help="Default top-k set to highlight.")
    return parser.parse_args()


def _to_markdown_table(rows) -> str:
    lines = [
        "| rank | model | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy | preprocessing | training_mode | original_epoch_policy |",
        "|---:|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for index, (_, row) in enumerate(rows.iterrows(), start=1):
        lines.append(
            f"| {index} | {row['model']} | {float(row['val_macro_f1']):.4f} | "
            f"{float(row['test_macro_f1']):.4f} | {float(row['val_accuracy']):.4f} | "
            f"{float(row['test_accuracy']):.4f} | {row['preprocessing']} | "
            f"{row['training_mode']} | {row['original_epoch_policy']} |"
        )
    return "\n".join(lines)


def _unsupported_section(frame) -> str:
    if "support_status" not in frame.columns:
        return "## Unsupported Models\n\n- none\n"

    unsupported = frame[frame["support_status"].fillna("supported") != "supported"].copy()
    if unsupported.empty:
        return "## Unsupported Models\n\n- none\n"

    lines = [
        "## Unsupported Models",
        "",
        "| model | exists_in_original_baseline | wrapper_supported | unsupported_reason |",
        "|---|---|---|---|",
    ]
    for _, row in unsupported.iterrows():
        lines.append(
            f"| {row['model']} | {row.get('exists_in_original_baseline', '-')}"
            f" | {row.get('wrapper_supported', '-')}"
            f" | {row.get('unsupported_reason', '')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    results_path = PROJECT_ROOT / "results" / "metrics" / "final_benchmark_results.csv"
    if not results_path.exists():
        print(
            "Warning: final benchmark results were not found. "
            "Run experiments/14_run_original_benchmark_full.py first."
        )
        return

    full_frame = pd.read_csv(results_path)
    ranked_frame = load_benchmark_results(results_path)
    if ranked_frame.empty:
        print(
            "Warning: supported benchmark rows were not found in final_benchmark_results.csv. "
            "Check support_status and unsupported_reason."
        )
        return

    rank1_model = str(ranked_frame.iloc[0]["model"])
    top3_models = ranked_frame.head(min(3, len(ranked_frame)))["model"].astype(str).tolist()
    top5_models = ranked_frame.head(min(5, len(ranked_frame)))["model"].astype(str).tolist()
    default_top_k = ranked_frame.head(min(args.top_k, len(ranked_frame)))["model"].astype(str).tolist()

    doc_path = PROJECT_ROOT / "docs" / "final_benchmark_selection.md"
    ensure_dir(doc_path.parent)
    markdown = f"""# Final Benchmark Selection

## Selection Summary

- selection metric = validation Macro F1
- test Macro F1 is confirmation only
- benchmark rank 1 model by validation Macro F1: `{rank1_model}`
- benchmark top3 by validation Macro F1: {", ".join(f"`{model}`" for model in top3_models)}
- benchmark top5 by validation Macro F1: {", ".join(f"`{model}`" for model in top5_models)}
- default low-data model set suggestion: {", ".join(f"`{model}`" for model in default_top_k)}

## Selection Rule

- primary sort: `val_macro_f1` descending
- tie-break 1: `test_macro_f1` descending
- tie-break 2: `val_accuracy` descending
- F2 preprocessing comparison should use the benchmark rank 1 model.
- F4/F5 should use benchmark top3 by default and benchmark top5 only as an optional extension.

## Full Ranking Table

{_to_markdown_table(ranked_frame)}

{_unsupported_section(full_frame)}
"""
    doc_path.write_text(markdown, encoding="utf-8")
    print(f"Saved benchmark selection document: {doc_path}")
    print(f"Benchmark rank 1 model: {rank1_model}")


if __name__ == "__main__":
    main()
