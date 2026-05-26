from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_selection import resolve_rank1_model
from src.data.preprocessing import parse_preprocessing_pipeline
from src.data.uthar_dataset import create_uthar_dataloaders
from src.models.benchmark_factory import build_benchmark_model, normalize_benchmark_model_name
from src.preprocessing_selection import (
    apply_stability_ranking,
    serialize_seed_list,
)
from src.train import run_training
from src.training_policy import CONTROLLED_GENERALIZATION_DEFAULTS
from src.utils import ensure_dir, get_device, set_seed


DEFAULT_CANDIDATES = [
    "savgol_smoothing+train_global_zscore",
    "moving_average_smoothing+minmax_scaling",
    "train_featurewise_zscore",
    "minmax_scaling",
    "moving_average_smoothing",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run F3 multi-seed preprocessing stability check on UT-HAR."
    )
    parser.set_defaults(use_benchmark_rank1=True)
    parser.add_argument("--use-benchmark-rank1", dest="use_benchmark_rank1", action="store_true")
    parser.add_argument("--no-use-benchmark-rank1", dest="use_benchmark_rank1", action="store_false")
    parser.add_argument("--model", default=None, help="Manual model override when benchmark rank1 is disabled.")
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--real-ratio", type=float, default=0.25)
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(CONTROLLED_GENERALIZATION_DEFAULTS["max_epochs"]),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", default="data/UT_HAR")
    parser.add_argument("--output-prefix", default="final")
    parser.add_argument("--close-tolerance", type=float, default=0.005)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-regenerate-figures", action="store_true")
    return parser.parse_args()


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


def _resolve_model(args: argparse.Namespace) -> tuple[str, str | None]:
    resolved_rank1_model = resolve_rank1_model(PROJECT_ROOT) if args.use_benchmark_rank1 else None
    if args.use_benchmark_rank1 and resolved_rank1_model is None:
        raise ValueError("Run F1 original benchmark and benchmark selection first.")
    if args.use_benchmark_rank1 and args.model is not None:
        raise ValueError(
            "--use-benchmark-rank1 cannot be combined with --model. "
            "The stability check must use the benchmark rank 1 model only."
        )
    if args.use_benchmark_rank1:
        normalized = normalize_benchmark_model_name(resolved_rank1_model)
        return normalized, normalized
    if not args.model:
        raise ValueError("Provide --model when --no-use-benchmark-rank1 is used.")
    normalized = normalize_benchmark_model_name(args.model)
    return normalized, resolved_rank1_model


def _resolve_candidates(candidates: list[str]) -> list[str]:
    resolved: list[str] = []
    invalid: list[str] = []
    for candidate in candidates:
        try:
            steps = parse_preprocessing_pipeline(candidate)
        except ValueError:
            invalid.append(candidate)
            continue
        resolved.append("+".join(steps))
    if invalid:
        raise ValueError(f"Unsupported preprocessing candidates: {invalid}")
    return resolved


def _write_csv(csv_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _output_paths(output_prefix: str) -> tuple[Path, Path]:
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    return (
        metrics_dir / f"{output_prefix}_preprocessing_stability_results.csv",
        metrics_dir / f"{output_prefix}_preprocessing_stability_summary.csv",
    )


def _check_output_paths(
    results_csv: Path,
    summary_csv: Path,
    overwrite: bool,
    dry_run: bool,
) -> None:
    existing_paths = [path for path in [results_csv, summary_csv] if path.exists()]
    if not existing_paths or overwrite:
        if overwrite and existing_paths:
            print("Overwrite enabled for existing stability output files:")
            for path in existing_paths:
                print(f"- {path}")
        return

    message = (
        "Stability output files already exist. "
        "Delete them or rerun with --overwrite:\n"
        f"- {results_csv}\n"
        f"- {summary_csv}"
    )
    if dry_run:
        print(f"Output file check: {message}")
        return
    raise RuntimeError(message)


def _build_summary_frame(rows: list[dict[str, object]], close_tolerance: float) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    grouped_rows: list[dict[str, object]] = []
    for (preprocessing, model), group in frame.groupby(["preprocessing", "model"], sort=False):
        ordered_group = group.sort_values(by="seed")
        mean_val_macro_f1 = float(ordered_group["val_macro_f1"].astype(float).mean())
        std_val_macro_f1 = float(ordered_group["val_macro_f1"].astype(float).std(ddof=0))
        mean_test_macro_f1 = float(ordered_group["test_macro_f1"].astype(float).mean())
        std_test_macro_f1 = float(ordered_group["test_macro_f1"].astype(float).std(ddof=0))
        mean_val_accuracy = float(ordered_group["val_accuracy"].astype(float).mean())
        mean_test_accuracy = float(ordered_group["test_accuracy"].astype(float).mean())
        grouped_rows.append(
            {
                "preprocessing": preprocessing,
                "model": model,
                "num_seeds": int(len(ordered_group)),
                "seeds": serialize_seed_list(ordered_group["seed"].astype(int).tolist()),
                "mean_val_macro_f1": mean_val_macro_f1,
                "std_val_macro_f1": std_val_macro_f1,
                "mean_test_macro_f1": mean_test_macro_f1,
                "std_test_macro_f1": std_test_macro_f1,
                "mean_val_accuracy": mean_val_accuracy,
                "mean_test_accuracy": mean_test_accuracy,
                "mean_val_test_macro_f1_gap": mean_val_macro_f1 - mean_test_macro_f1,
                "selected_by": "mean_validation_macro_f1_across_seeds",
            }
        )
    summary_frame = pd.DataFrame(grouped_rows)
    ranked_frame, _, _ = apply_stability_ranking(summary_frame, close_tolerance=close_tolerance)
    return ranked_frame


def main() -> None:
    args = parse_args()

    if args.real_ratio <= 0 or args.real_ratio > 1.0:
        raise ValueError("real_ratio must be in the range (0, 1].")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if not args.seeds:
        raise ValueError("At least one seed must be provided.")

    final_model_to_run, resolved_rank1_model = _resolve_model(args)
    resolved_candidates = _resolve_candidates(args.candidates)
    results_csv, summary_csv = _output_paths(args.output_prefix)

    resolved_config = {
        "experiment": "preprocessing_stability_check",
        "use_benchmark_rank1": args.use_benchmark_rank1,
        "resolved_rank1_model": resolved_rank1_model,
        "final_model_to_run": final_model_to_run,
        "candidates": resolved_candidates,
        "seeds": args.seeds,
        "real_ratio": args.real_ratio,
        "augmentation": False,
        "training_mode": "controlled_generalization",
        "requested_epochs": args.epochs,
        "batch_size": args.batch_size,
        "close_tolerance": args.close_tolerance,
        "dry_run": args.dry_run,
        "results_csv_path": str(results_csv),
        "summary_csv_path": str(summary_csv),
        "checkpoint_dir": str(PROJECT_ROOT / "results" / "checkpoints"),
    }
    print("Resolved config:")
    print(json.dumps(resolved_config, indent=2))

    _check_output_paths(
        results_csv=results_csv,
        summary_csv=summary_csv,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("Dry-run mode: exiting before training.")
        return

    device = get_device()
    rows: list[dict[str, object]] = []

    for seed in args.seeds:
        for preprocessing in resolved_candidates:
            print(
                f"\nRunning stability check model={final_model_to_run} preprocessing={preprocessing} seed={seed}"
            )
            set_seed(seed)
            train_loader, val_loader, test_loader, metadata = create_uthar_dataloaders(
                data_root=PROJECT_ROOT / args.data_root,
                batch_size=args.batch_size,
                preprocessing=preprocessing,
                preprocessing_config=None,
                real_ratio=args.real_ratio,
                seed=seed,
                model_type=final_model_to_run,
                augmentation=False,
            )
            print("Data metadata:", metadata)

            checkpoint_name = (
                f"{args.output_prefix}_preprocessing_stability_"
                f"{final_model_to_run.replace('+', '_').lower()}_"
                f"seed{seed}_{preprocessing.replace('+', '__')}_best.pt"
            )
            checkpoint_path = PROJECT_ROOT / "results" / "checkpoints" / checkpoint_name
            model = build_benchmark_model(final_model_to_run)

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
                    warmup_epochs=int(CONTROLLED_GENERALIZATION_DEFAULTS["warmup_epochs"]),
                    patience=int(CONTROLLED_GENERALIZATION_DEFAULTS["patience"]),
                    min_delta=float(CONTROLLED_GENERALIZATION_DEFAULTS["min_delta"]),
                    gradient_clip_norm=float(CONTROLLED_GENERALIZATION_DEFAULTS["gradient_clip_norm"]),
                    scheduler_type=str(CONTROLLED_GENERALIZATION_DEFAULTS["scheduler_type"]),
                    optimizer_name="adam",
                    weight_decay=float(CONTROLLED_GENERALIZATION_DEFAULTS["weight_decay"]),
                )
            except RuntimeError as exc:
                if torch.cuda.is_available() and "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"CUDA OOM while running preprocessing={preprocessing} seed={seed}. "
                        "Retry with --batch-size 32."
                    ) from exc
                raise

            best_val_metrics = result["best_val_metrics"]
            test_metrics = result["test_metrics"]
            rows.append(
                {
                    "experiment": "preprocessing_stability_check",
                    "model": final_model_to_run,
                    "preprocessing": preprocessing,
                    "seed": seed,
                    "real_ratio": args.real_ratio,
                    "augmentation": "false",
                    "training_mode": "controlled_generalization",
                    "requested_epochs": args.epochs,
                    "actual_epochs_ran": result["actual_epochs_ran"],
                    "best_epoch": result["best_epoch"],
                    "batch_size": args.batch_size,
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

    per_seed_fieldnames = [
        "experiment",
        "model",
        "preprocessing",
        "seed",
        "real_ratio",
        "augmentation",
        "training_mode",
        "requested_epochs",
        "actual_epochs_ran",
        "best_epoch",
        "batch_size",
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
    _write_csv(results_csv, rows, per_seed_fieldnames)
    print(f"Saved per-seed stability results to: {results_csv}")

    summary_frame = _build_summary_frame(rows, close_tolerance=args.close_tolerance)
    summary_fieldnames = [
        "preprocessing",
        "model",
        "num_seeds",
        "seeds",
        "mean_val_macro_f1",
        "std_val_macro_f1",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_val_accuracy",
        "mean_test_accuracy",
        "mean_val_test_macro_f1_gap",
        "selected_by",
        "selection_rank",
    ]
    _write_csv(summary_csv, summary_frame.to_dict(orient="records"), summary_fieldnames)
    print(f"Saved aggregate stability summary to: {summary_csv}")

    if not args.disable_regenerate_figures:
        try:
            regenerate_report_figures = _load_regenerate_figures_callable()
            regenerated_files = regenerate_report_figures(PROJECT_ROOT)
            print("Automatically regenerated report figures:")
            for file_path in regenerated_files:
                print(f"- {file_path}")
        except Exception as exc:
            print(f"Warning: figure regeneration failed after stability check: {exc}")


if __name__ == "__main__":
    main()
