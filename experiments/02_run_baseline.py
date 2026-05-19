"""Stage 1 baseline experiment entry point.

This script will run the clean full-data baselines after dataset loading and
EDA are validated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.uthar_dataset import create_uthar_dataloaders
from src.models.gru import GRUClassifier
from src.train import run_training
from src.utils import ensure_dir, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UT-HAR GRU dry-run pipeline")
    parser.add_argument("--model", default="GRU")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--real-ratio", type=float, default=1.0)
    parser.add_argument("--augmentation", default="false")
    parser.add_argument(
        "--preprocessing",
        default="train_global_zscore",
        choices=["none", "train_global_zscore", "per_sample_zscore"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", default="data/UT_HAR")
    return parser.parse_args()


def _normalize_bool_string(value: str) -> str:
    lowered = value.lower()
    if lowered not in {"true", "false"}:
        raise ValueError("augmentation must be either 'true' or 'false'.")
    return lowered


def _append_result_row(csv_path: Path, row: dict[str, object]) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = [
        "experiment",
        "model",
        "real_ratio",
        "augmentation",
        "preprocessing",
        "seed",
        "epochs",
        "device",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    ]
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.augmentation = _normalize_bool_string(args.augmentation)

    if args.model.upper() != "GRU":
        raise ValueError("Only GRU is supported in this task.")
    if args.augmentation != "false":
        raise ValueError("augmentation=false is required in this task.")
    if args.real_ratio != 1.0:
        raise ValueError("Only real_ratio=1.0 is supported in this dry-run task.")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")

    print("Resolved config:")
    print(
        {
            "model": args.model,
            "epochs": args.epochs,
            "real_ratio": args.real_ratio,
            "augmentation": args.augmentation,
            "preprocessing": args.preprocessing,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "dry_run": args.dry_run,
            "data_root": args.data_root,
        }
    )

    # Rubric: reproducibility requires a fixed seed before sampling, shuffling,
    # and model initialization so dry-run metrics are traceable.
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, test_loader, metadata = create_uthar_dataloaders(
        data_root=PROJECT_ROOT / args.data_root,
        batch_size=args.batch_size,
        preprocessing=args.preprocessing,
        real_ratio=args.real_ratio,
        seed=args.seed,
        model_type=args.model,
    )
    print("Data metadata:", metadata)

    model = GRUClassifier()
    checkpoint_path = PROJECT_ROOT / "results" / "checkpoints" / "dry_run_gru_best.pt"
    result = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
    )

    best_val_metrics = result["best_val_metrics"]
    test_metrics = result["test_metrics"]
    csv_path = PROJECT_ROOT / "results" / "metrics" / "dry_run_results.csv"
    row = {
        "experiment": "dry_run_gru",
        "model": args.model,
        "real_ratio": args.real_ratio,
        "augmentation": args.augmentation,
        "preprocessing": args.preprocessing,
        "seed": args.seed,
        "epochs": args.epochs,
        "device": str(device),
        "val_accuracy": best_val_metrics["accuracy"],
        "val_macro_f1": best_val_metrics["macro_f1"],
        "val_weighted_f1": best_val_metrics["weighted_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
    }
    _append_result_row(csv_path, row)

    print("Best validation metrics:", best_val_metrics)
    print("Test metrics:", test_metrics)
    print(f"Saved metrics row to: {csv_path}")


if __name__ == "__main__":
    main()
