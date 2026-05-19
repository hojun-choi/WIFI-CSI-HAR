"""Training entry points for the clean experimental pipeline.

Early stopping and best-validation checkpointing will live here so every model
is compared under the same training policy.
"""

# Rubric: validation-based early stopping should be centralized here so model
# comparison is fair and the test set remains reserved for final evaluation.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from src.evaluate import evaluate_model
from src.utils import ensure_dir


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Train the model for one epoch and return average training loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples if total_samples else 0.0


def run_training(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float = 1e-3,
    checkpoint_path: str | Path | None = None,
    use_early_stopping: bool = False,
    patience: int | None = None,
) -> dict[str, object]:
    """Run the dry-run training loop and return validation/test metrics."""
    # Rubric: the dry-run validates the full pipeline before spending time on
    # longer experiments, which reduces the risk of wasted 50/200 epoch runs.
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_macro_f1 = float("-inf")
    best_val_metrics: dict[str, float] | None = None
    best_state_dict = deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    actual_epochs_ran = 0

    checkpoint_file: Path | None = None
    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path)
        ensure_dir(checkpoint_file.parent)

    for epoch in range(1, epochs + 1):
        actual_epochs_ran = epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)

        print(
            "epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            "val_accuracy={val_accuracy:.4f} val_macro_f1={val_macro_f1:.4f} "
            "val_weighted_f1={val_weighted_f1:.4f}".format(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_metrics["accuracy"],
                val_macro_f1=val_metrics["macro_f1"],
                val_weighted_f1=val_metrics["weighted_f1"],
            )
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_metrics = val_metrics
            best_state_dict = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
            if checkpoint_file is not None:
                torch.save(best_state_dict, checkpoint_file)
        else:
            epochs_without_improvement += 1

        if use_early_stopping and patience is not None and epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(best_epoch={best_epoch}, patience={patience})"
            )
            break

    model.load_state_dict(best_state_dict)
    test_loss, test_metrics, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)

    return {
        "best_val_metrics": best_val_metrics or {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0},
        "best_epoch": best_epoch,
        "actual_epochs_ran": actual_epochs_ran,
        "test_metrics": test_metrics,
        "test_loss": test_loss,
        "test_y_true": y_true,
        "test_y_pred": y_pred,
    }
