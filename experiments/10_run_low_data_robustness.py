from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_selection import resolve_top_models
from src.data.uthar_dataset import create_uthar_dataloaders
from src.models.cnn import CNNClassifier
from src.models.cnn_gru import CNNGRUClassifier
from src.models.benchmark_factory import build_benchmark_model, normalize_benchmark_model_name
from src.models.gru import GRUClassifier
from src.models.lstm import LSTMClassifier
from src.train import run_training
from src.training_policy import CONTROLLED_GENERALIZATION_DEFAULTS
from src.utils import ensure_dir, get_device, set_seed


DEFAULT_MODELS = ["CNN", "GRU", "LSTM", "CNN_GRU"]
DEFAULT_REAL_RATIOS = [1.0, 0.5, 0.25, 0.1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UT-HAR low-data robustness experiments")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--use-benchmark-top3", action="store_true")
    parser.add_argument("--use-benchmark-top5", action="store_true")
    parser.add_argument("--real-ratios", nargs="+", type=float, default=DEFAULT_REAL_RATIOS)
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(CONTROLLED_GENERALIZATION_DEFAULTS["max_epochs"]),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preprocessing", default="per_sample_zscore")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=int(CONTROLLED_GENERALIZATION_DEFAULTS["warmup_epochs"]),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=int(CONTROLLED_GENERALIZATION_DEFAULTS["patience"]),
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=float(CONTROLLED_GENERALIZATION_DEFAULTS["min_delta"]),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(CONTROLLED_GENERALIZATION_DEFAULTS["weight_decay"]),
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=float(CONTROLLED_GENERALIZATION_DEFAULTS["gradient_clip_norm"]),
    )
    parser.add_argument(
        "--scheduler-type",
        choices=["none", "plateau"],
        default=str(CONTROLLED_GENERALIZATION_DEFAULTS["scheduler_type"]),
    )
    parser.add_argument("--data-root", default="data/UT_HAR")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--disable-regenerate-figures", action="store_true")
    return parser.parse_args()


def normalize_model_name(model_name: str) -> str:
    upper_name = model_name.strip().upper()
    if upper_name in {"CNN", "CNN_GRU"}:
        return upper_name
    return normalize_benchmark_model_name(model_name)


def build_model(model_name: str) -> torch.nn.Module:
    normalized = normalize_model_name(model_name)
    if normalized == "CNN":
        return CNNClassifier()
    if normalized == "GRU":
        return GRUClassifier()
    if normalized == "LSTM":
        return LSTMClassifier()
    if normalized == "CNN_GRU":
        return CNNGRUClassifier()
    return build_benchmark_model(normalized)


def _resolve_models(args: argparse.Namespace) -> list[str]:
    if args.use_benchmark_top3 and args.use_benchmark_top5:
        raise ValueError("Choose only one of --use-benchmark-top3 or --use-benchmark-top5.")

    if args.use_benchmark_top3 or args.use_benchmark_top5:
        top_k = 3 if args.use_benchmark_top3 else 5
        resolved = resolve_top_models(PROJECT_ROOT, top_k=top_k)
        if not resolved:
            raise ValueError("final_benchmark_results.csv is missing. Run F1 original benchmark first.")
        return [normalize_model_name(model_name) for model_name in resolved]

    return [normalize_model_name(model_name) for model_name in args.models]


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


def _write_results_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = [
        "experiment",
        "model",
        "real_ratio",
        "augmentation",
        "preprocessing",
        "seed",
        "training_mode",
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
        "full_train_size",
        "selected_train_size",
        "val_size",
        "test_size",
        "class_counts_selected",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "macro_f1_drop",
        "macro_f1_retention",
        "accuracy_drop",
        "accuracy_retention",
        "checkpoint_path",
        "selected_by",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _add_degradation_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return rows

    frame["macro_f1_drop"] = pd.NA
    frame["macro_f1_retention"] = pd.NA
    frame["accuracy_drop"] = pd.NA
    frame["accuracy_retention"] = pd.NA

    for model_name, model_df in frame.groupby("model"):
        base_rows = model_df[model_df["real_ratio"].astype(float) == 1.0]
        if base_rows.empty:
            continue
        base_test_macro_f1 = float(base_rows.iloc[0]["test_macro_f1"])
        base_test_accuracy = float(base_rows.iloc[0]["test_accuracy"])
        model_mask = frame["model"] == model_name
        frame.loc[model_mask, "macro_f1_drop"] = (
            base_test_macro_f1 - frame.loc[model_mask, "test_macro_f1"].astype(float)
        )
        frame.loc[model_mask, "macro_f1_retention"] = (
            frame.loc[model_mask, "test_macro_f1"].astype(float) / base_test_macro_f1
        )
        frame.loc[model_mask, "accuracy_drop"] = (
            base_test_accuracy - frame.loc[model_mask, "test_accuracy"].astype(float)
        )
        frame.loc[model_mask, "accuracy_retention"] = (
            frame.loc[model_mask, "test_accuracy"].astype(float) / base_test_accuracy
        )

    return frame.to_dict(orient="records")


def _resolve_smoke_test(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    if args.models == DEFAULT_MODELS:
        args.models = ["GRU"]
    if args.real_ratios == DEFAULT_REAL_RATIOS:
        args.real_ratios = [0.25]
    if args.epochs == int(CONTROLLED_GENERALIZATION_DEFAULTS["max_epochs"]):
        args.epochs = 5


def main() -> None:
    args = parse_args()
    _resolve_smoke_test(args)

    normalized_models = _resolve_models(args)
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if any(ratio <= 0 or ratio > 1.0 for ratio in args.real_ratios):
        raise ValueError("All real_ratio values must be in the range (0, 1].")

    resolved_config = {
        "experiment": "low_data_robustness",
        "models": normalized_models,
        "real_ratios": args.real_ratios,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "preprocessing": args.preprocessing,
        "training_mode": "controlled_generalization",
        "augmentation": "false",
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "scheduler_type": args.scheduler_type,
        "smoke_test": args.smoke_test,
    }
    print("Resolved config:")
    print(resolved_config)

    device = get_device()
    rows: list[dict[str, object]] = []

    for model_name in normalized_models:
        for real_ratio in args.real_ratios:
            print(f"\nRunning model={model_name} real_ratio={real_ratio}")
            set_seed(args.seed)
            train_loader, val_loader, test_loader, metadata = create_uthar_dataloaders(
                data_root=PROJECT_ROOT / args.data_root,
                batch_size=args.batch_size,
                preprocessing=args.preprocessing,
                real_ratio=real_ratio,
                seed=args.seed,
                model_type=model_name,
            )
            print("Data metadata:", metadata)

            checkpoint_suffix = "smoke_test" if args.smoke_test else "full"
            checkpoint_path = (
                PROJECT_ROOT
                / "results"
                / "checkpoints"
                / f"final_low_data_{checkpoint_suffix}_{model_name.replace('+', '_').lower()}_{str(real_ratio).replace('.', 'p')}_best.pt"
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
                    use_early_stopping=True,
                    warmup_epochs=args.warmup_epochs,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    gradient_clip_norm=args.gradient_clip_norm,
                    scheduler_type=args.scheduler_type,
                    optimizer_name="adam",
                    weight_decay=args.weight_decay,
                )
            except RuntimeError as exc:
                if torch.cuda.is_available() and "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"CUDA OOM while running {model_name} at real_ratio={real_ratio}. "
                        "Retry with --batch-size 32."
                    ) from exc
                raise

            best_val_metrics = result["best_val_metrics"]
            test_metrics = result["test_metrics"]
            rows.append(
                {
                    "experiment": "low_data_robustness",
                    "model": model_name,
                    "real_ratio": real_ratio,
                    "augmentation": "false",
                    "preprocessing": args.preprocessing,
                    "seed": args.seed,
                    "training_mode": "controlled_generalization",
                    "requested_epochs": args.epochs,
                    "actual_epochs_ran": result["actual_epochs_ran"],
                    "best_epoch": result["best_epoch"],
                    "warmup_epochs": args.warmup_epochs,
                    "patience": args.patience,
                    "min_delta": args.min_delta,
                    "weight_decay": args.weight_decay,
                    "gradient_clip_norm": args.gradient_clip_norm,
                    "scheduler_type": args.scheduler_type,
                    "device": str(device),
                    "full_train_size": metadata["full_train_size"],
                    "selected_train_size": metadata["selected_train_size"],
                    "val_size": metadata["val_size"],
                    "test_size": metadata["test_size"],
                    "class_counts_selected": str(metadata["class_counts_selected"]),
                    "val_accuracy": best_val_metrics["accuracy"],
                    "val_macro_f1": best_val_metrics["macro_f1"],
                    "val_weighted_f1": best_val_metrics["weighted_f1"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_weighted_f1": test_metrics["weighted_f1"],
                    "macro_f1_drop": "",
                    "macro_f1_retention": "",
                    "accuracy_drop": "",
                    "accuracy_retention": "",
                    "checkpoint_path": str(checkpoint_path),
                    "selected_by": "validation_macro_f1",
                }
            )

    rows = _add_degradation_metrics(rows)

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    if args.smoke_test:
        csv_path = metrics_dir / "final_low_data_smoke_test_results.csv"
    else:
        csv_path = metrics_dir / "final_low_data_results.csv"
    _write_results_csv(csv_path, rows)
    print(f"Saved low-data results to: {csv_path}")

    if args.smoke_test:
        print("Smoke test mode: skipped full figure regeneration.")
        return

    if not args.disable_regenerate_figures:
        try:
            regenerate_report_figures = _load_regenerate_figures_callable()
            regenerated_files = regenerate_report_figures(PROJECT_ROOT)
            print("Automatically regenerated report figures:")
            for file_path in regenerated_files:
                print(f"- {file_path}")
        except Exception as exc:
            print(f"Warning: figure regeneration failed after low-data run: {exc}")


if __name__ == "__main__":
    main()
