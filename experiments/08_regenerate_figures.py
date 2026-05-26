from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import (
    save_augmentation_recovery_plots,
    save_baseline_original_epoch_plots,
    save_final_preprocessing_plots,
    save_low_data_plots,
    save_preprocessing_ablation_plot,
    save_sample_csi_lineplot,
)


def _load_numpy_binary(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        return np.load(file)


def regenerate_report_figures(project_root: Path | None = None) -> list[Path]:
    root = project_root or PROJECT_ROOT
    baseline_csv = root / "results" / "metrics" / "baseline_results_original_epoch.csv"
    low_data_csv = root / "results" / "metrics" / "low_data_results.csv"
    augmentation_csv = root / "results" / "metrics" / "augmentation_results.csv"
    preprocessing_csv = root / "results" / "metrics" / "preprocessing_ablation_results.csv"
    final_preprocessing_csv = root / "results" / "metrics" / "final_preprocessing_results.csv"
    final_preprocessing_combination_csv = (
        root / "results" / "metrics" / "final_preprocessing_combination_results.csv"
    )
    sample_path = root / "data" / "UT_HAR" / "data" / "X_train.csv"
    figures_dir = root / "results" / "figures"

    created_files: list[Path] = []

    if baseline_csv.exists():
        created_files.extend(save_baseline_original_epoch_plots(baseline_csv, figures_dir))
    else:
        print(f"Skipped baseline figures because CSV was not found: {baseline_csv}")

    if low_data_csv.exists():
        created_files.extend(save_low_data_plots(low_data_csv, figures_dir))
    else:
        print(f"Skipped low-data figures because CSV was not found: {low_data_csv}")

    if augmentation_csv.exists():
        created_files.extend(
            save_augmentation_recovery_plots(
                augmentation_csv,
                low_data_csv,
                figures_dir,
            )
        )
    else:
        print(f"Skipped augmentation figures because CSV was not found: {augmentation_csv}")

    if preprocessing_csv.exists():
        created_files.extend(
            save_preprocessing_ablation_plot(
                preprocessing_csv,
                figures_dir / "preprocessing_ablation_macro_f1.png",
            )
        )
    else:
        print(f"Skipped preprocessing figures because CSV was not found: {preprocessing_csv}")

    if final_preprocessing_csv.exists() or final_preprocessing_combination_csv.exists():
        if final_preprocessing_csv.exists() and final_preprocessing_combination_csv.exists():
            merged_frame = pd.concat(
                [
                    pd.read_csv(final_preprocessing_csv),
                    pd.read_csv(final_preprocessing_combination_csv),
                ],
                ignore_index=True,
            )
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".csv",
                delete=False,
                encoding="utf-8",
                newline="",
            ) as temp_file:
                merged_frame.to_csv(temp_file.name, index=False)
                created_files.extend(save_final_preprocessing_plots(temp_file.name, figures_dir))
            Path(temp_file.name).unlink(missing_ok=True)
        else:
            source_csv = (
                final_preprocessing_csv
                if final_preprocessing_csv.exists()
                else final_preprocessing_combination_csv
            )
            created_files.extend(save_final_preprocessing_plots(source_csv, figures_dir))
    else:
        print(
            "Skipped final preprocessing figures because neither official CSV was found: "
            f"{final_preprocessing_csv}"
        )

    if sample_path.exists():
        sample_array = _load_numpy_binary(sample_path)
        created_files.append(
            save_sample_csi_lineplot(
                sample_array[0],
                figures_dir / "sample_csi_lineplot.png",
            )
        )
    else:
        print(f"Skipped sample line plot because data was not found: {sample_path}")

    return created_files


def main() -> None:
    created_files = regenerate_report_figures()
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
