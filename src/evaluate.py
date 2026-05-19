"""Evaluation helpers for validation/test reporting.

Centralizing evaluation helps keep metrics and saved outputs consistent across
all experiment stages.
"""

# Rubric: validation and test evaluation must be consistent so CSV metrics,
# confusion matrices, and classification reports are directly comparable.

from __future__ import annotations

from collections.abc import Iterable

import torch

from src.metrics import compute_classification_metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float], list[int], list[int]]:
    """Evaluate a model and return loss, metrics, and predictions."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    average_loss = total_loss / total_samples if total_samples else 0.0
    metrics = compute_classification_metrics(y_true, y_pred)
    return average_loss, metrics, y_true, y_pred
