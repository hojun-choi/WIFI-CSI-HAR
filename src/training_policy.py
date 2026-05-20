"""Shared training-policy defaults for fair, reproducible experiment control."""

from __future__ import annotations


# Rubric: keep benchmark-style 200-epoch runs separate from generalization-
# oriented experiments so low-data comparisons are not biased by overtraining.
CONTROLLED_GENERALIZATION_DEFAULTS = {
    "max_epochs": 100,
    "warmup_epochs": 20,
    "patience": 15,
    "min_delta": 0.001,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "scheduler_type": "plateau",
    "selection_metric": "val_macro_f1",
}


ORIGINAL_EPOCH_POLICY_NOTE = (
    "original_epoch is preserved only for benchmark-style comparison with the "
    "original baseline. Future low-data experiments should prefer "
    "controlled_generalization."
)
