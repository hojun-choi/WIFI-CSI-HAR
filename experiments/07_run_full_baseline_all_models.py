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
ORIGINAL_UT_HAR_EPOCHS = {
    # Mapping note: the project `CNN` baseline corresponds to the original
    # `LeNet`-style UT_HAR model, which is the clearest CNN baseline match.
    "CNN": 200,
    "GRU": 200,
    "LSTM": 200,
    # Mapping note: `CNN_GRU` maps directly to the original `CNN+GRU`.
    "CNN_GRU": 200,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UT-HAR full-data baselines")
    parser.add_argument(
        "--training-mode",
        choices=["original_epoch", "early_stopping"],
        default="original_epoch",
    )
    parser.add_argument("--epochs", type=int, default=None)
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
        "training_mode",
        "model",
        "real_ratio",
        "augmentation",
        "preprocessing",
        "seed",
        "requested_epochs",
        "actual_epochs_ran",
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
        "epoch_source",
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
    if args.training_mode == "early_stopping":
        if args.epochs is None:
            args.epochs = 50
        if args.epochs <= 0 or args.epochs > 50:
            raise ValueError("Use max_epochs in the range 1..50 for early_stopping mode.")
    elif args.epochs is not None and args.epochs <= 0:
        raise ValueError("epochs override must be positive.")

    resolved_config = {
        "training_mode": args.training_mode,
        "models": normalized_models,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "preprocessing": args.preprocessing,
        "real_ratio": args.real_ratio,
        "augmentation": "false",
        "patience": args.patience,
        "data_root": args.data_root,
        "original_ut_har_epochs": ORIGINAL_UT_HAR_EPOCHS,
    }
    print("Resolved config:")
    print(resolved_config)

    device = get_device()
    rows: list[dict[str, object]] = []

    for model_name in normalized_models:
        print(f"\nRunning model: {model_name}")
        if model_name not in ORIGINAL_UT_HAR_EPOCHS:
            raise ValueError(f"No original epoch mapping found for model: {model_name}")

        if args.training_mode == "original_epoch":
            if args.epochs is None:
                requested_epochs = ORIGINAL_UT_HAR_EPOCHS[model_name]
                epoch_source = "original_baseline_mapping"
            else:
                requested_epochs = args.epochs
                epoch_source = "user_override"
                print(
                    f"Overriding original epoch policy for {model_name}: "
                    f"{ORIGINAL_UT_HAR_EPOCHS[model_name]} -> {requested_epochs}"
                )
            use_early_stopping = False
            patience = None
        else:
            requested_epochs = args.epochs
            epoch_source = "user_override"
            use_early_stopping = True
            patience = args.patience

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
            PROJECT_ROOT
            / "results"
            / "checkpoints"
            / f"baseline_{args.training_mode}_{model_name.lower()}_best.pt"
        )
        model = build_model(model_name)
        try:
            result = run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=requested_epochs,
                checkpoint_path=checkpoint_path,
                use_early_stopping=use_early_stopping,
                patience=patience,
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
                "training_mode": args.training_mode,
                "model": model_name,
                "real_ratio": args.real_ratio,
                "augmentation": "false",
                "preprocessing": args.preprocessing,
                "seed": args.seed,
                "requested_epochs": requested_epochs,
                "actual_epochs_ran": result["actual_epochs_ran"],
                "best_epoch": result["best_epoch"],
                "patience": patience,
                "device": str(device),
                "val_accuracy": best_val_metrics["accuracy"],
                "val_macro_f1": best_val_metrics["macro_f1"],
                "val_weighted_f1": best_val_metrics["weighted_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "checkpoint_path": str(checkpoint_path),
                "selected_by": "validation_macro_f1",
                "epoch_source": epoch_source,
            }
        )

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    mode_csv_path = metrics_dir / f"baseline_results_{args.training_mode}.csv"
    write_baseline_csv(mode_csv_path, rows)
    # Backward compatibility: keep the generic filename pointing to the latest run.
    generic_csv_path = metrics_dir / "baseline_results.csv"
    write_baseline_csv(generic_csv_path, rows)

    plot_metric_bar(
        rows,
        x_col="model",
        y_col="val_macro_f1",
        output_path=figures_dir / f"baseline_{args.training_mode}_macro_f1.png",
        title=f"Full-data Baseline Validation Macro F1 ({args.training_mode})",
        xlabel="Model",
        ylabel="Validation Macro F1",
    )
    plot_metric_bar(
        rows,
        x_col="model",
        y_col="val_accuracy",
        output_path=figures_dir / f"baseline_{args.training_mode}_accuracy.png",
        title=f"Full-data Baseline Validation Accuracy ({args.training_mode})",
        xlabel="Model",
        ylabel="Validation Accuracy",
    )
    # Backward compatibility generic figure names.
    plot_metric_bar(
        rows,
        x_col="model",
        y_col="val_macro_f1",
        output_path=figures_dir / "baseline_macro_f1.png",
        title=f"Full-data Baseline Validation Macro F1 ({args.training_mode})",
        xlabel="Model",
        ylabel="Validation Macro F1",
    )
    plot_metric_bar(
        rows,
        x_col="model",
        y_col="val_accuracy",
        output_path=figures_dir / "baseline_accuracy.png",
        title=f"Full-data Baseline Validation Accuracy ({args.training_mode})",
        xlabel="Model",
        ylabel="Validation Accuracy",
    )

    print(f"Saved baseline results to: {mode_csv_path}")
    print(f"Saved figures under: {figures_dir}")


if __name__ == "__main__":
    main()
