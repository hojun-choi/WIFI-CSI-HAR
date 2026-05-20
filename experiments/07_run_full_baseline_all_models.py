from __future__ import annotations

import argparse
import csv
import importlib.util
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
from src.training_policy import CONTROLLED_GENERALIZATION_DEFAULTS
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
        choices=["original_epoch", "early_stopping", "controlled_generalization"],
        default="original_epoch",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preprocessing", default="per_sample_zscore")
    parser.add_argument("--real-ratio", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--scheduler-type", choices=["none", "plateau"], default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--data-root", default="data/UT_HAR")
    parser.add_argument(
        "--regenerate-figures",
        dest="regenerate_figures",
        action="store_true",
        help="Regenerate report-ready figures after a successful baseline run.",
    )
    parser.add_argument(
        "--disable-regenerate-figures",
        dest="regenerate_figures",
        action="store_false",
        help="Skip automatic figure regeneration after training.",
    )
    parser.add_argument(
        "--no-regenerate-figures",
        dest="regenerate_figures",
        action="store_false",
        help="Backward-compatible alias for disabling automatic figure regeneration.",
    )
    parser.set_defaults(regenerate_figures=True)
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
        "warmup_epochs",
        "patience",
        "min_delta",
        "weight_decay",
        "gradient_clip_norm",
        "scheduler_type",
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


def _load_regenerate_figures_callable():
    module_path = PROJECT_ROOT / "experiments" / "08_regenerate_figures.py"
    spec = importlib.util.spec_from_file_location("regenerate_figures_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load regeneration module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    regenerate = getattr(module, "regenerate_report_figures", None)
    if regenerate is None:
        raise AttributeError(
            "regenerate_report_figures() was not found in experiments/08_regenerate_figures.py"
        )
    return regenerate


def _resolve_training_policy(args: argparse.Namespace, model_name: str) -> dict[str, object]:
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
        return {
            "requested_epochs": requested_epochs,
            "use_early_stopping": False,
            "warmup_epochs": 0,
            "patience": None,
            "min_delta": 0.0,
            "weight_decay": 0.0,
            "gradient_clip_norm": None,
            "scheduler_type": "none",
            "epoch_source": epoch_source,
        }

    if args.training_mode == "early_stopping":
        requested_epochs = args.epochs if args.epochs is not None else 50
        if requested_epochs <= 0 or requested_epochs > 50:
            raise ValueError("Use max_epochs in the range 1..50 for early_stopping mode.")
        return {
            "requested_epochs": requested_epochs,
            "use_early_stopping": True,
            "warmup_epochs": args.warmup_epochs if args.warmup_epochs is not None else 0,
            "patience": args.patience if args.patience is not None else 8,
            "min_delta": args.min_delta if args.min_delta is not None else 0.0,
            "weight_decay": args.weight_decay if args.weight_decay is not None else 0.0,
            "gradient_clip_norm": args.gradient_clip_norm,
            "scheduler_type": args.scheduler_type if args.scheduler_type is not None else "none",
            "epoch_source": "user_override",
        }

    requested_epochs = args.epochs
    if requested_epochs is None:
        requested_epochs = int(CONTROLLED_GENERALIZATION_DEFAULTS["max_epochs"])
    if requested_epochs <= 0:
        raise ValueError("max_epochs must be positive for controlled_generalization mode.")
    return {
        "requested_epochs": requested_epochs,
        "use_early_stopping": True,
        "warmup_epochs": (
            args.warmup_epochs
            if args.warmup_epochs is not None
            else int(CONTROLLED_GENERALIZATION_DEFAULTS["warmup_epochs"])
        ),
        "patience": (
            args.patience
            if args.patience is not None
            else int(CONTROLLED_GENERALIZATION_DEFAULTS["patience"])
        ),
        "min_delta": (
            args.min_delta
            if args.min_delta is not None
            else float(CONTROLLED_GENERALIZATION_DEFAULTS["min_delta"])
        ),
        "weight_decay": (
            args.weight_decay
            if args.weight_decay is not None
            else float(CONTROLLED_GENERALIZATION_DEFAULTS["weight_decay"])
        ),
        "gradient_clip_norm": (
            args.gradient_clip_norm
            if args.gradient_clip_norm is not None
            else float(CONTROLLED_GENERALIZATION_DEFAULTS["gradient_clip_norm"])
        ),
        "scheduler_type": (
            args.scheduler_type
            if args.scheduler_type is not None
            else str(CONTROLLED_GENERALIZATION_DEFAULTS["scheduler_type"])
        ),
        "epoch_source": "user_override" if args.epochs is not None else "controlled_default",
    }


def main() -> None:
    args = parse_args()
    normalized_models = [normalize_model_name(model_name) for model_name in args.models]
    if args.real_ratio != 1.0:
        raise ValueError("This baseline runner is currently scoped to real_ratio=1.0.")
    if any(model_name not in ORIGINAL_UT_HAR_EPOCHS for model_name in normalized_models):
        missing = [model_name for model_name in normalized_models if model_name not in ORIGINAL_UT_HAR_EPOCHS]
        raise ValueError(f"No original epoch mapping found for model(s): {missing}")

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
        "warmup_epochs": args.warmup_epochs,
        "min_delta": args.min_delta,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "scheduler_type": args.scheduler_type,
        "data_root": args.data_root,
        "original_ut_har_epochs": ORIGINAL_UT_HAR_EPOCHS,
        "controlled_defaults": CONTROLLED_GENERALIZATION_DEFAULTS,
    }
    print("Resolved config:")
    print(resolved_config)

    device = get_device()
    rows: list[dict[str, object]] = []

    for model_name in normalized_models:
        print(f"\nRunning model: {model_name}")
        training_policy = _resolve_training_policy(args, model_name)
        print("Training policy:", training_policy)

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
                epochs=int(training_policy["requested_epochs"]),
                checkpoint_path=checkpoint_path,
                use_early_stopping=bool(training_policy["use_early_stopping"]),
                warmup_epochs=int(training_policy["warmup_epochs"]),
                patience=training_policy["patience"],
                min_delta=float(training_policy["min_delta"]),
                gradient_clip_norm=training_policy["gradient_clip_norm"],
                scheduler_type=str(training_policy["scheduler_type"]),
                optimizer_name="adam",
                weight_decay=float(training_policy["weight_decay"]),
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
                "requested_epochs": training_policy["requested_epochs"],
                "actual_epochs_ran": result["actual_epochs_ran"],
                "best_epoch": result["best_epoch"],
                "warmup_epochs": training_policy["warmup_epochs"],
                "patience": training_policy["patience"],
                "min_delta": training_policy["min_delta"],
                "weight_decay": training_policy["weight_decay"],
                "gradient_clip_norm": training_policy["gradient_clip_norm"],
                "scheduler_type": training_policy["scheduler_type"],
                "device": str(device),
                "val_accuracy": best_val_metrics["accuracy"],
                "val_macro_f1": best_val_metrics["macro_f1"],
                "val_weighted_f1": best_val_metrics["weighted_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "checkpoint_path": str(checkpoint_path),
                "selected_by": "validation_macro_f1",
                "epoch_source": training_policy["epoch_source"],
            }
        )

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    mode_csv_path = metrics_dir / f"baseline_results_{args.training_mode}.csv"
    write_baseline_csv(mode_csv_path, rows)
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

    if args.regenerate_figures:
        try:
            regenerate_report_figures = _load_regenerate_figures_callable()
            regenerated_files = regenerate_report_figures(PROJECT_ROOT)
            print("Automatically regenerated report figures:")
            for file_path in regenerated_files:
                print(f"- {file_path}")
        except Exception as exc:
            print(f"Warning: figure regeneration failed after training: {exc}")


if __name__ == "__main__":
    main()
