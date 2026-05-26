from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.uthar_dataset import create_uthar_dataloaders
from src.models.benchmark_factory import (
    build_benchmark_model,
    get_original_uthar_model_spec,
    list_original_uthar_model_names,
    normalize_benchmark_model_name,
)
from src.train import run_training
from src.utils import ensure_dir, get_device, set_seed


DEFAULT_MODELS = list_original_uthar_model_names()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official F1 original benchmark full run.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Original UT_HAR_data supervised benchmark model names.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", default="data/UT_HAR")
    parser.add_argument("--real-ratio", type=float, default=1.0)
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--output-prefix", default="final")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _write_results_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = [
        "experiment",
        "model",
        "original_model_name",
        "dataset",
        "preprocessing",
        "real_ratio",
        "augmentation",
        "training_mode",
        "original_epoch_policy",
        "requested_epochs",
        "actual_epochs_ran",
        "best_epoch",
        "seed",
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
        "support_status",
        "unsupported_reason",
        "exists_in_original_baseline",
        "wrapper_supported",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_smoke_test(args: argparse.Namespace) -> int:
    if not args.smoke_test:
        return 0
    if args.models == DEFAULT_MODELS:
        args.models = ["GRU"]
    return 5


def _make_unsupported_row(
    model_name: str,
    seed: int,
    batch_size: int,
    device: str,
    reason: str,
    *,
    exists_in_original_baseline: bool,
    wrapper_supported: bool,
) -> dict[str, object]:
    return {
        "experiment": "original_benchmark_full",
        "model": model_name,
        "original_model_name": model_name,
        "dataset": "UT_HAR_data",
        "preprocessing": "none",
        "real_ratio": 1.0,
        "augmentation": "false",
        "training_mode": "original_epoch",
        "original_epoch_policy": "",
        "requested_epochs": "",
        "actual_epochs_ran": "",
        "best_epoch": "",
        "seed": seed,
        "batch_size": batch_size,
        "device": device,
        "val_accuracy": "",
        "val_macro_f1": "",
        "val_weighted_f1": "",
        "test_accuracy": "",
        "test_macro_f1": "",
        "test_weighted_f1": "",
        "checkpoint_path": "",
        "selected_by": "validation_macro_f1",
        "support_status": "unsupported",
        "unsupported_reason": reason,
        "exists_in_original_baseline": str(exists_in_original_baseline).lower(),
        "wrapper_supported": str(wrapper_supported).lower(),
    }


def main() -> None:
    args = parse_args()
    smoke_epochs = _resolve_smoke_test(args)

    if args.real_ratio != 1.0:
        raise ValueError("F1 official benchmark is scoped to real_ratio=1.0.")
    if args.preprocessing.lower() not in {"none", "raw"}:
        raise ValueError("F1 official benchmark must use preprocessing=none/raw.")

    normalized_models: list[str] = []
    for model_name in args.models:
        normalized_models.append(normalize_benchmark_model_name(model_name))

    resolved_config = {
        "experiment": "original_benchmark_full",
        "dataset": "UT_HAR_data",
        "models": normalized_models,
        "training_mode": "original_epoch",
        "preprocessing": "none",
        "real_ratio": 1.0,
        "augmentation": "false",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "smoke_test": args.smoke_test,
        "smoke_epochs": smoke_epochs if args.smoke_test else None,
    }
    print("Resolved config:")
    print(json.dumps(resolved_config, indent=2))

    device = get_device()
    rows: list[dict[str, object]] = []

    for model_name in normalized_models:
        spec = get_original_uthar_model_spec(model_name)
        requested_epochs = smoke_epochs if args.smoke_test else spec.epoch_policy
        print(
            f"\nRunning F1 model={model_name} original_model_name={spec.original_model_name} "
            f"requested_epochs={requested_epochs}"
        )
        set_seed(args.seed)
        train_loader, val_loader, test_loader, metadata = create_uthar_dataloaders(
            data_root=PROJECT_ROOT / args.data_root,
            batch_size=args.batch_size,
            preprocessing="none",
            preprocessing_config=None,
            real_ratio=1.0,
            seed=args.seed,
            model_type=model_name,
            augmentation=False,
        )
        print("Data metadata:", metadata)

        checkpoint_suffix = "smoke_test" if args.smoke_test else "full"
        checkpoint_path = (
            PROJECT_ROOT
            / "results"
            / "checkpoints"
            / f"{args.output_prefix}_benchmark_original_raw_{checkpoint_suffix}_{model_name.replace('+', '_').lower()}_best.pt"
        )

        try:
            model = build_benchmark_model(model_name)
        except Exception as exc:
            reason = (
                "exists_in_original_baseline=true; wrapper_supported=false; "
                f"model construction failed: {exc}"
            )
            print(f"Unsupported model detected: {model_name} -> {reason}")
            rows.append(
                _make_unsupported_row(
                    model_name=model_name,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    device=str(device),
                    reason=reason,
                    exists_in_original_baseline=spec.exists_in_original_baseline,
                    wrapper_supported=False,
                )
            )
            continue

        try:
            result = run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=requested_epochs,
                checkpoint_path=checkpoint_path,
                use_early_stopping=False,
                warmup_epochs=0,
                patience=None,
                min_delta=0.0,
                gradient_clip_norm=None,
                scheduler_type="none",
                optimizer_name="adam",
                weight_decay=0.0,
            )
        except RuntimeError as exc:
            if torch.cuda.is_available() and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                reason = (
                    "exists_in_original_baseline=true; wrapper_supported=true; "
                    "runtime failed with CUDA OOM in current environment. Retry with --batch-size 32."
                )
                print(f"Unsupported model detected at runtime: {model_name} -> {reason}")
                rows.append(
                    _make_unsupported_row(
                        model_name=model_name,
                        seed=args.seed,
                        batch_size=args.batch_size,
                        device=str(device),
                        reason=reason,
                        exists_in_original_baseline=spec.exists_in_original_baseline,
                        wrapper_supported=True,
                    )
                )
                continue
            raise
        except Exception as exc:
            reason = (
                "exists_in_original_baseline=true; wrapper_supported=true; "
                f"runtime failed in current wrapper: {exc}"
            )
            print(f"Unsupported model detected at runtime: {model_name} -> {reason}")
            rows.append(
                _make_unsupported_row(
                    model_name=model_name,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    device=str(device),
                    reason=reason,
                    exists_in_original_baseline=spec.exists_in_original_baseline,
                    wrapper_supported=True,
                )
            )
            continue

        best_val_metrics = result["best_val_metrics"]
        test_metrics = result["test_metrics"]
        rows.append(
            {
                "experiment": "original_benchmark_full",
                "model": model_name,
                "original_model_name": spec.original_model_name,
                "dataset": "UT_HAR_data",
                "preprocessing": "none",
                "real_ratio": 1.0,
                "augmentation": "false",
                "training_mode": "original_epoch",
                "original_epoch_policy": spec.epoch_policy,
                "requested_epochs": requested_epochs,
                "actual_epochs_ran": result["actual_epochs_ran"],
                "best_epoch": result["best_epoch"],
                "seed": args.seed,
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
                "support_status": "supported",
                "unsupported_reason": "",
                "exists_in_original_baseline": "true",
                "wrapper_supported": "true",
            }
        )

    supported_rows = [row for row in rows if row["support_status"] == "supported"]
    unsupported_rows = [row for row in rows if row["support_status"] != "supported"]
    supported_rows.sort(
        key=lambda row: (
            -float(row["val_macro_f1"]),
            -float(row["test_macro_f1"]),
            -float(row["val_accuracy"]),
        )
    )
    ordered_rows = supported_rows + unsupported_rows

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    csv_name = (
        f"{args.output_prefix}_benchmark_smoke_test_results.csv"
        if args.smoke_test
        else f"{args.output_prefix}_benchmark_results.csv"
    )
    csv_path = metrics_dir / csv_name
    _write_results_csv(csv_path, ordered_rows)
    print(f"Saved benchmark results to: {csv_path}")
    if unsupported_rows:
        print("Unsupported model summary:")
        for row in unsupported_rows:
            print(f"- {row['model']}: {row['unsupported_reason']}")
    if args.smoke_test:
        print("Smoke test mode: these results are not final evidence.")


if __name__ == "__main__":
    main()
