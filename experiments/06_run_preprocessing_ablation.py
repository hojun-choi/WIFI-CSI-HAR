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
from src.visualization import save_preprocessing_ablation_plot


PREPROCESSING_CANDIDATES = ["none", "train_global_zscore", "per_sample_zscore"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UT-HAR preprocessing ablation for GRU")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--real-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", default="data/UT_HAR")
    return parser.parse_args()


def _write_results_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
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
        "selected_train_size",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "selected_by",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(summary_path: Path, best_row: dict[str, object]) -> None:
    ensure_dir(summary_path.parent)
    summary_lines = [
        f"best_preprocessing: {best_row['preprocessing']}",
        f"best_val_macro_f1: {best_row['val_macro_f1']}",
        (
            "note: this preprocessing should be used for main experiments "
            "unless later evidence suggests otherwise."
        ),
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.real_ratio != 0.25:
        raise ValueError("This ablation task is scoped to real_ratio=0.25.")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")

    resolved_config = {
        "model": "GRU",
        "real_ratio": args.real_ratio,
        "augmentation": "false",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "data_root": args.data_root,
        "preprocessing_candidates": PREPROCESSING_CANDIDATES,
    }
    print("Resolved config:")
    print(resolved_config)

    set_seed(args.seed)
    device = get_device()

    rows: list[dict[str, object]] = []
    for preprocessing in PREPROCESSING_CANDIDATES:
        print(f"\nRunning preprocessing candidate: {preprocessing}")
        # Rubric: use the same seed and train subset policy for each candidate
        # so preprocessing is the only variable in this controlled comparison.
        set_seed(args.seed)
        train_loader, val_loader, test_loader, metadata = create_uthar_dataloaders(
            data_root=PROJECT_ROOT / args.data_root,
            batch_size=args.batch_size,
            preprocessing=preprocessing,
            real_ratio=args.real_ratio,
            seed=args.seed,
            model_type="GRU",
        )
        print("Selected train size:", metadata["selected_train_size"])
        print("Selected class counts:", metadata["class_counts_selected"])

        model = GRUClassifier()
        checkpoint_path = (
            PROJECT_ROOT
            / "results"
            / "checkpoints"
            / f"preprocessing_ablation_{preprocessing}_best.pt"
        )
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
        rows.append(
            {
                "experiment": "preprocessing_ablation",
                "model": "GRU",
                "real_ratio": args.real_ratio,
                "augmentation": "false",
                "preprocessing": preprocessing,
                "seed": args.seed,
                "epochs": args.epochs,
                "device": str(device),
                "selected_train_size": metadata["selected_train_size"],
                "val_accuracy": best_val_metrics["accuracy"],
                "val_macro_f1": best_val_metrics["macro_f1"],
                "val_weighted_f1": best_val_metrics["weighted_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "selected_by": "validation_macro_f1",
            }
        )

    rows.sort(key=lambda row: float(row["val_macro_f1"]), reverse=True)

    csv_path = PROJECT_ROOT / "results" / "metrics" / "preprocessing_ablation_results.csv"
    _write_results_csv(csv_path, rows)

    figure_path = PROJECT_ROOT / "results" / "figures" / "preprocessing_ablation_macro_f1.png"
    save_preprocessing_ablation_plot(csv_path, figure_path)

    best_row = rows[0]
    print(f"Best preprocessing by validation Macro F1: {best_row['preprocessing']}")
    summary_path = PROJECT_ROOT / "results" / "metrics" / "preprocessing_ablation_summary.txt"
    _write_summary(summary_path, best_row)

    print(f"Saved results CSV: {csv_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
