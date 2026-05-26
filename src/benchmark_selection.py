"""Helpers for F1 benchmark ranking and downstream model selection."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    coerced = frame.copy()
    for column in ["val_macro_f1", "test_macro_f1", "val_accuracy", "test_accuracy"]:
        if column in coerced.columns:
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def sort_benchmark_frame(frame: pd.DataFrame) -> pd.DataFrame:
    sortable = _coerce_numeric_columns(frame)
    if "support_status" in sortable.columns:
        sortable = sortable[sortable["support_status"].fillna("supported") == "supported"]
    sortable = sortable.dropna(subset=["val_macro_f1", "test_macro_f1", "val_accuracy"])
    return sortable.sort_values(
        by=["val_macro_f1", "test_macro_f1", "val_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def load_benchmark_results(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    return sort_benchmark_frame(pd.read_csv(path))


def read_rank1_from_selection_doc(doc_path: str | Path) -> str | None:
    path = Path(doc_path)
    if not path.exists():
        return None

    pattern = re.compile(r"benchmark rank 1 model.*?:\s*`?([^`]+?)`?\s*$", re.IGNORECASE)
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line.strip())
        if match:
            return match.group(1).strip()
    return None


def resolve_rank1_model(project_root: str | Path) -> str | None:
    root = Path(project_root)
    rank1 = read_rank1_from_selection_doc(root / "docs" / "final_benchmark_selection.md")
    if rank1:
        return rank1

    frame = load_benchmark_results(root / "results" / "metrics" / "final_benchmark_results.csv")
    if frame.empty:
        return None
    return str(frame.iloc[0]["model"])


def resolve_top_models(project_root: str | Path, top_k: int) -> list[str]:
    root = Path(project_root)
    frame = load_benchmark_results(root / "results" / "metrics" / "final_benchmark_results.csv")
    if frame.empty:
        return []
    return frame.head(min(top_k, len(frame)))["model"].astype(str).tolist()
