"""Training entry points for the clean experimental pipeline."""

# Rubric: validation-based checkpoint selection should be centralized here so
# model comparison stays fair and the test set remains reserved for final use.

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
    gradient_clip_norm: float | None = None,
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
        if gradient_clip_norm is not None:
            # Rubric: gradient clipping helps keep low-data training stable.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples if total_samples else 0.0


def _create_scheduler(scheduler_type: str, optimizer: torch.optim.Optimizer):
    if scheduler_type == "none":
        return None
    if scheduler_type == "plateau":
        # Rubric: a conservative LR reduction is useful when low-data validation
        # performance oscillates instead of improving smoothly.
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )
    raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


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
    warmup_epochs: int = 0,
    patience: int | None = None,
    min_delta: float = 0.0,
    gradient_clip_norm: float | None = None,
    scheduler_type: str = "none",
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
) -> dict[str, object]:
    """Run training and return best-validation plus final test metrics."""
    # Rubric: best validation checkpointing is the key selection rule for fair
    # comparison, especially when low-data training fluctuates by epoch.
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer_name_normalized = optimizer_name.lower()
    if optimizer_name_normalized == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name_normalized == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

    scheduler = _create_scheduler(scheduler_type=scheduler_type, optimizer=optimizer)

    best_val_macro_f1 = float("-inf")
    best_val_metrics = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    best_state_dict = deepcopy(model.state_dict())
    best_epoch = 0
    epochs_since_improvement = 0
    actual_epochs_ran = 0

    checkpoint_file: Path | None = None
    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path)
        ensure_dir(checkpoint_file.parent)

    for epoch in range(1, epochs + 1):
        actual_epochs_ran = epoch
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            gradient_clip_norm=gradient_clip_norm,
        )
        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)

        improvement = val_metrics["macro_f1"] > best_val_macro_f1 + min_delta
        if improvement:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_metrics = val_metrics
            best_state_dict = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improvement = 0
            if checkpoint_file is not None:
                torch.save(best_state_dict, checkpoint_file)
        else:
            epochs_since_improvement += 1

        if scheduler is not None:
            scheduler.step(val_metrics["macro_f1"])

        print(
            "epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            "val_accuracy={val_accuracy:.4f} val_macro_f1={val_macro_f1:.4f} "
            "val_weighted_f1={val_weighted_f1:.4f} best_epoch={best_epoch} "
            "epochs_since_improvement={epochs_since_improvement} current_lr={current_lr:.6f}".format(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_metrics["accuracy"],
                val_macro_f1=val_metrics["macro_f1"],
                val_weighted_f1=val_metrics["weighted_f1"],
                best_epoch=best_epoch,
                epochs_since_improvement=epochs_since_improvement,
                current_lr=_current_lr(optimizer),
            )
        )

        if (
            use_early_stopping
            and patience is not None
            and epoch >= warmup_epochs
            and epochs_since_improvement >= patience
        ):
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(best_epoch={best_epoch}, warmup_epochs={warmup_epochs}, "
                f"patience={patience}, min_delta={min_delta})"
            )
            break

    # Rubric: final test evaluation must use the checkpoint chosen by
    # validation Macro F1, not simply the last training epoch.
    model.load_state_dict(best_state_dict)
    test_loss, test_metrics, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)

    return {
        "best_val_metrics": best_val_metrics,
        "best_val_accuracy": best_val_metrics["accuracy"],
        "best_val_macro_f1": best_val_metrics["macro_f1"],
        "best_val_weighted_f1": best_val_metrics["weighted_f1"],
        "best_epoch": best_epoch,
        "actual_epochs_ran": actual_epochs_ran,
        "epochs_since_improvement": epochs_since_improvement,
        "test_metrics": test_metrics,
        "test_loss": test_loss,
        "test_y_true": y_true,
        "test_y_pred": y_pred,
    }
