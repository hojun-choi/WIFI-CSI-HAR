from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import (
    save_baseline_original_epoch_plots,
    save_preprocessing_ablation_plot,
)


def main() -> None:
    baseline_csv = PROJECT_ROOT / "results" / "metrics" / "baseline_results_original_epoch.csv"
    preprocessing_csv = (
        PROJECT_ROOT / "results" / "metrics" / "preprocessing_ablation_results.csv"
    )
    figures_dir = PROJECT_ROOT / "results" / "figures"

    created_files: list[Path] = []

    if baseline_csv.exists():
        created_files.extend(
            save_baseline_original_epoch_plots(baseline_csv, figures_dir)
        )
    else:
        print(f"Skipped baseline figures because CSV was not found: {baseline_csv}")

    if preprocessing_csv.exists():
        created_files.extend(
            save_preprocessing_ablation_plot(
                preprocessing_csv,
                figures_dir / "preprocessing_ablation_macro_f1.png",
            )
        )
    else:
        print(f"Skipped preprocessing figures because CSV was not found: {preprocessing_csv}")

    print("Created figure files:")
    for file_path in created_files:
        print(f"- {file_path}")

    print("\nManual cleanup commands for old early-stopping artifacts:")
    print(
        "Remove-Item results\\metrics\\baseline_results_early_stopping.csv, "
        "results\\figures\\baseline_early_stopping_macro_f1.png, "
        "results\\figures\\baseline_early_stopping_accuracy.png"
    )
    print(
        "Remove-Item results\\metrics\\baseline_results.csv, "
        "results\\figures\\baseline_macro_f1.png, "
        "results\\figures\\baseline_accuracy.png"
    )


if __name__ == "__main__":
    main()
