from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.uthar_dataset import create_uthar_dataloaders
from src.models.cnn import CNNClassifier
from src.models.cnn_gru import CNNGRUClassifier
from src.models.gru import GRUClassifier
from src.models.lstm import LSTMClassifier
from src.train import run_training
from src.utils import ensure_dir, get_device, set_seed
from src.visualization import plot_metric_bar


DEFAULT_MODELS = ["CNN", "GRU", "LSTM", "CNN_GRU"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UT-HAR full-data baselines")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preprocessing", default="per_sample_zscore")
    parser.add_argument("--real-ratio", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--data-root", default="data/UT_HAR")
    return parser.parse_args()


def build_model(model_name: str) -> torch.nn.Module:
    normalized = model_name.upper()
    if normalized == "CNN":
        return CNNClassifier()
    if normalized == "GRU":
        return GRUClassifier()
    if normalized == "LSTM":
        return LSTMClassifier()
    if normalized in {"CNN_GRU", "CNN+GRU"}:
        return CNNGRUClassifier()
    raise ValueError(f"Unsupported model: {model_name}")


def normalize_model_name(model_name: str) -> str:
    return "CNN_GRU" if model_name.upper() == "CNN+GRU" else model_name.upper()


def write_baseline_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = [
        "experiment",
        "model",
        "real_ratio",
        "augmentation",
        "preprocessing",
        "seed",
        "max_epochs",
        "best_epoch",
        "patience",
        "device",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "checkpoint_path",
        "selected_by",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    normalized_models = [normalize_model_name(model_name) for model_name in args.models]
    if args.real_ratio != 1.0:
        raise ValueError("M3 full-data baseline must use real_ratio=1.0.")
    if args.epochs <= 0 or args.epochs > 50:
        raise ValueError("Use max_epochs in the range 1..50 for this baseline task.")

    resolved_config = {
        "models": normalized_models,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "preprocessing": args.preprocessing,
        "real_ratio": args.real_ratio,
        "augmentation": "false",
        "patience": args.patience,
        "data_root": args.data_root,
    }
    print("Resolved config:")
    print(resolved_config)

    device = get_device()
    rows: list[dict[str, object]] = []

    for model_name in normalized_models:
        print(f"\nRunning model: {model_name}")
        set_seed(args.seed)
        train_loader, val_loader, test_loader, metadata = create_uthar_dataloaders(
            data_root=PROJECT_ROOT / args.data_root,
            batch_size=args.batch_size,
            preprocessing=args.preprocessing,
            real_ratio=args.real_ratio,
            seed=args.seed,
            model_type=model_name,
        )
        print("Data metadata:", metadata)

        checkpoint_path = (
            PROJECT_ROOT / "results" / "checkpoints" / f"baseline_{model_name.lower()}_best.pt"
        )
        model = build_model(model_name)
        try:
            result = run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                checkpoint_path=checkpoint_path,
                patience=args.patience,
            )
        except RuntimeError as exc:
            if torch.cuda.is_available() and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM while running {model_name}. Retry with --batch-size 32."
                ) from exc
            raise

        best_val_metrics = result["best_val_metrics"]
        test_metrics = result["test_metrics"]
        rows.append(
            {
                "experiment": "full_data_baseline",
                "model": model_name,
                "real_ratio": args.real_ratio,
                "augmentation": "false",
                "preprocessing": args.preprocessing,
                "seed": args.seed,
                "max_epochs": args.epochs,
                "best_epoch": result["best_epoch"],
                "patience": args.patience,
                "device": str(device),
                "val_accuracy": best_val_metrics["accuracy"],
                "val_macro_f1": best_val_metrics["macro_f1"],
                "val_weighted_f1": best_val_metrics["weighted_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "checkpoint_path": str(checkpoint_path),
                "selected_by": "validation_macro_f1",
            }
        )

    csv_path = PROJECT_ROOT / "results" / "metrics" / "baseline_results.csv"
    write_baseline_csv(csv_path, rows)

    figures_dir = PROJECT_ROOT / "results" / "figures"
    plot_metric_bar(
        rows,
        x_col="model",
        y_col="val_macro_f1",
        output_path=figures_dir / "baseline_macro_f1.png",
        title="Full-data Baseline Validation Macro F1",
        xlabel="Model",
        ylabel="Validation Macro F1",
    )
    plot_metric_bar(
        rows,
        x_col="model",
        y_col="val_accuracy",
        output_path=figures_dir / "baseline_accuracy.png",
        title="Full-data Baseline Validation Accuracy",
        xlabel="Model",
        ylabel="Validation Accuracy",
    )

    print(f"Saved baseline results to: {csv_path}")
    print(f"Saved figures under: {figures_dir}")


if __name__ == "__main__":
    main()
