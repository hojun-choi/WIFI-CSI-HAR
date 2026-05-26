from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_selection import resolve_rank1_model
from src.data.preprocessing import parse_preprocessing_pipeline
from src.data.uthar_dataset import create_uthar_dataloaders
from src.models.cnn import CNNClassifier
from src.models.cnn_gru import CNNGRUClassifier
from src.models.benchmark_factory import build_benchmark_model, normalize_benchmark_model_name
from src.train import run_training
from src.training_policy import CONTROLLED_GENERALIZATION_DEFAULTS
from src.utils import ensure_dir, get_device, set_seed


DEFAULT_SINGLE_PREPROCESSINGS = [
    "none",
    "train_global_zscore",
    "per_sample_zscore",
    "minmax_scaling",
    "robust_scaling",
    "savgol_smoothing",
    "moving_average_smoothing",
    "train_featurewise_zscore",
    "per_sample_featurewise_zscore",
]
DEFAULT_COMBINATIONS = [
    "savgol_smoothing+per_sample_zscore",
    "savgol_smoothing+train_global_zscore",
    "moving_average_smoothing+per_sample_zscore",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run F2 expanded preprocessing comparison on UT-HAR"
    )
    parser.add_argument("--model", default=None, help="Single benchmark rank 1 model alias.")
    parser.add_argument("--models", nargs="+", default=["GRU"])
    parser.add_argument(
        "--use-benchmark-rank1",
        action="store_true",
        help="Load the F1 benchmark rank 1 model from docs/final_benchmark_selection.md or final_benchmark_results.csv.",
    )
    parser.add_argument("--preprocessings", nargs="+", default=DEFAULT_SINGLE_PREPROCESSINGS)
    parser.add_argument("--combinations", nargs="+", default=DEFAULT_COMBINATIONS)
    parser.add_argument(
        "--comparison-mode",
        choices=["single", "combination", "all"],
        default="single",
    )
    parser.add_argument("--real-ratio", type=float, default=0.25)
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(CONTROLLED_GENERALIZATION_DEFAULTS["max_epochs"]),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--training-mode",
        choices=["controlled_generalization"],
        default="controlled_generalization",
    )
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
    parser.add_argument("--output-prefix", default="final")
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
    if normalized == "CNN_GRU":
        return CNNGRUClassifier()
    return build_benchmark_model(normalized)


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
        "preprocessing_group",
        "model",
        "preprocessing",
        "preprocessing_steps",
        "preprocessing_config",
        "real_ratio",
        "augmentation",
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
        "checkpoint_path",
        "selected_by",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_smoke_test(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.model = "GRU"
    args.models = ["GRU"]
    args.comparison_mode = "single"
    args.preprocessings = ["none", "per_sample_zscore", "savgol_smoothing"]
    args.real_ratio = 0.25
    args.epochs = 5


def _resolve_models(args: argparse.Namespace) -> list[str]:
    models = list(args.models)
    if args.use_benchmark_rank1:
        benchmark_rank1 = resolve_rank1_model(PROJECT_ROOT)
        if benchmark_rank1 is None:
            raise ValueError("Run F1 original benchmark and benchmark selection first.")
        models.append(benchmark_rank1)
    if args.model:
        models.append(args.model)
    normalized = [normalize_model_name(model_name) for model_name in models]
    unique_models: list[str] = []
    for model_name in normalized:
        if model_name not in unique_models:
            unique_models.append(model_name)
    return unique_models


def _resolve_experiment_plan(
    comparison_mode: str,
    preprocessings: list[str],
    combinations: list[str],
) -> list[tuple[str, str]]:
    valid_single: list[str] = []
    invalid_single: list[str] = []
    for step in preprocessings:
        try:
            parsed = parse_preprocessing_pipeline(step)
        except ValueError:
            invalid_single.append(step)
            continue
        if len(parsed) != 1:
            invalid_single.append(step)
            continue
        valid_single.append(parsed[0])
    if invalid_single:
        raise ValueError(f"Unsupported single preprocessing candidates: {invalid_single}")

    if comparison_mode == "single":
        return [("single", preprocessing) for preprocessing in valid_single]
    if comparison_mode == "combination":
        return [("combination", preprocessing) for preprocessing in combinations]
    if comparison_mode == "all":
        return [("single", preprocessing) for preprocessing in valid_single] + [
            ("combination", preprocessing) for preprocessing in combinations
        ]
    raise ValueError(f"Unsupported comparison_mode: {comparison_mode}")


def _serialize_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _output_csv_path(metrics_dir: Path, output_prefix: str, comparison_mode: str, smoke_test: bool) -> Path:
    if smoke_test:
        return metrics_dir / f"{output_prefix}_preprocessing_smoke_test_results.csv"
    if comparison_mode == "combination":
        return metrics_dir / f"{output_prefix}_preprocessing_combination_results.csv"
    return metrics_dir / f"{output_prefix}_preprocessing_results.csv"


def main() -> None:
    args = parse_args()
    _resolve_smoke_test(args)

    if args.smoke_test:
        args.use_benchmark_rank1 = False

    user_explicitly_set_model = ("--model" in sys.argv) or ("--models" in sys.argv)
    models = _resolve_models(args)
    if not models:
        raise ValueError("At least one model must be provided via --model or --models.")
    if not args.smoke_test and not args.use_benchmark_rank1 and not user_explicitly_set_model:
        raise ValueError(
            "F2 official workflow expects the benchmark rank 1 model. "
            "Use --use-benchmark-rank1 or pass --model/--models explicitly."
        )
    if any(ratio <= 0 or ratio > 1.0 for ratio in [args.real_ratio]):
        raise ValueError("real_ratio must be in the range (0, 1].")
    if args.training_mode != "controlled_generalization":
        raise ValueError("F2 preprocessing comparison must use controlled_generalization.")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")

    plan = _resolve_experiment_plan(
        comparison_mode=args.comparison_mode,
        preprocessings=args.preprocessings,
        combinations=args.combinations,
    )
    resolved_config = {
        "experiment": "expanded_preprocessing_comparison",
        "models": models,
        "use_benchmark_rank1": args.use_benchmark_rank1,
        "comparison_mode": args.comparison_mode,
        "real_ratio": args.real_ratio,
        "preprocessings": args.preprocessings,
        "combinations": args.combinations,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "training_mode": args.training_mode,
        "requested_epochs": args.epochs,
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

    for model_name in models:
        for preprocessing_group, preprocessing in plan:
            print(
                f"\nRunning model={model_name} preprocessing_group={preprocessing_group} "
                f"preprocessing={preprocessing}"
            )
            set_seed(args.seed)
            train_loader, val_loader, test_loader, metadata = create_uthar_dataloaders(
                data_root=PROJECT_ROOT / args.data_root,
                batch_size=args.batch_size,
                preprocessing=preprocessing,
                preprocessing_config=None,
                real_ratio=args.real_ratio,
                seed=args.seed,
                model_type=model_name,
                augmentation=False,
            )
            print("Data metadata:", metadata)

            checkpoint_suffix = "smoke_test" if args.smoke_test else preprocessing_group
            checkpoint_name = (
                f"{args.output_prefix}_preprocessing_{checkpoint_suffix}_"
                f"{model_name.lower()}_{preprocessing.replace('+', '__')}_best.pt"
            )
            checkpoint_path = PROJECT_ROOT / "results" / "checkpoints" / checkpoint_name
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
                        f"CUDA OOM while running {model_name} with preprocessing={preprocessing}. "
                        "Retry with --batch-size 32."
                    ) from exc
                raise

            best_val_metrics = result["best_val_metrics"]
            test_metrics = result["test_metrics"]
            rows.append(
                {
                    "experiment": "expanded_preprocessing_comparison",
                    "preprocessing_group": preprocessing_group,
                    "model": model_name,
                    "preprocessing": metadata["preprocessing"],
                    "preprocessing_steps": _serialize_json(metadata["preprocessing_steps"]),
                    "preprocessing_config": _serialize_json(metadata["preprocessing_config"]),
                    "real_ratio": args.real_ratio,
                    "augmentation": "false",
                    "seed": args.seed,
                    "training_mode": args.training_mode,
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
                    "class_counts_selected": _serialize_json(metadata["class_counts_selected"]),
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

    rows.sort(
        key=lambda row: (
            str(row["preprocessing_group"]),
            str(row["model"]),
            -float(row["val_macro_f1"]),
        )
    )
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    csv_path = _output_csv_path(
        metrics_dir=metrics_dir,
        output_prefix=args.output_prefix,
        comparison_mode=args.comparison_mode,
        smoke_test=args.smoke_test,
    )
    _write_results_csv(csv_path, rows)
    print(f"Saved preprocessing results to: {csv_path}")

    if args.smoke_test:
        print("Smoke test mode: skipped figure regeneration. These results are not final evidence.")
        return

    if not args.disable_regenerate_figures:
        try:
            regenerate_report_figures = _load_regenerate_figures_callable()
            regenerated_files = regenerate_report_figures(PROJECT_ROOT)
            print("Automatically regenerated report figures:")
            for file_path in regenerated_files:
                print(f"- {file_path}")
        except Exception as exc:
            print(f"Warning: figure regeneration failed after preprocessing run: {exc}")


if __name__ == "__main__":
    main()
