"""Project metrics utilities.

Macro F1 and Weighted F1 are planned alongside Accuracy because accuracy alone
can hide class-wise degradation in low-data settings.
"""

# Rubric: Macro F1 is tracked because Accuracy can hide class-wise degradation,
# especially when low-data sampling affects minority classes unevenly.

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute report-ready classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
