# Wi-Fi CSI-based HAR on UT-HAR

Project-specific experimental framework for Wi-Fi CSI-based Human Activity Recognition on `UT-HAR`.

Tested with Python `3.10.11`.

## Project Motivation

Wi-Fi CSI-based HAR is sensitive to environment change, subject difference, device placement, and noise. This makes generalization difficult, especially when real labeled data is limited. The official final workflow therefore starts from F1 again: benchmark models first, preprocessing second, low-data robustness third, and augmentation recovery fourth.

Old generated artifacts were cleaned and are not used as final evidence. The codebase, dataset, and `original_baseline/` are preserved.

## Environment Setup

Use Windows PowerShell and Python `3.10.11`.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional GPU check:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Dataset Placement

Place the UT-HAR files at:

```text
data/UT_HAR/data/X_train.csv
data/UT_HAR/data/X_val.csv
data/UT_HAR/data/X_test.csv
data/UT_HAR/label/y_train.csv
data/UT_HAR/label/y_val.csv
data/UT_HAR/label/y_test.csv
```

These files keep the `.csv` extension from the benchmark, but they are NumPy binary files and should be loaded with `np.load`.

## UT-HAR Dataset and Labels

- original benchmark README sample shape: `1 x 250 x 90`
- this repo loads arrays as `(N, 250, 90)`
- one sample = `250 CSI frame indices x 90 CSI features`
- timestep is `CSI frame index`, not directly seconds
- time conversion requires `sampling_rate`
- if `sampling_rate = fs` Hz, one sample duration is `250 / fs` seconds
- `100Hz` conversion is illustrative only and not confirmed ground truth

Activity classes:

- `lie down`
- `fall`
- `walk`
- `pickup`
- `run`
- `sit down`
- `stand up`

Dataset visualizations use activity names instead of numeric-only labels.

## Official Final Workflow from Scratch

The final workflow starts from a clean artifact state. F1 uses `raw/none` preprocessing and `original_epoch`. F2/F3/F4/F5 use `controlled_generalization`. Old generated artifacts are cleaned and are not used as final evidence.

1. Clean generated artifacts:

```powershell
python experiments/00_clean_generated_artifacts.py --dry-run
python experiments/00_clean_generated_artifacts.py --execute
```

2. F1 original benchmark smoke test:

```powershell
python experiments/14_run_original_benchmark_full.py --smoke-test --seed 42 --batch-size 64
```

3. F1 original benchmark full run:

```powershell
python -u experiments/14_run_original_benchmark_full.py --seed 42 --batch-size 64 2>&1 | Tee-Object -FilePath logs\final_benchmark_original_raw.log
```

4. Select benchmark models:

```powershell
python experiments/15_select_benchmark_models.py
```

5. F2 single preprocessing comparison:

```powershell
python experiments/12_run_expanded_preprocessing_comparison.py --use-benchmark-rank1 --comparison-mode single --seed 42 --batch-size 64
```

6. F3 combination preprocessing:

```powershell
python experiments/12_run_expanded_preprocessing_comparison.py --use-benchmark-rank1 --comparison-mode combination --seed 42 --batch-size 64
```

7. Select final preprocessing:

```powershell
python experiments/13_select_final_preprocessing.py
```

8. F4/F5:

- low-data robustness and augmentation recovery use benchmark top3 models by default
- benchmark top5 is optional if runtime and available models allow it

Workflow rules:

- F1 uses `preprocessing=none/raw`.
- F1 uses `training_mode=original_epoch`.
- F1 attempts the original UT_HAR_data supervised benchmark model set:
  `MLP`, `LeNet`, `ResNet18`, `ResNet50`, `ResNet101`, `RNN`, `GRU`, `LSTM`, `BiLSTM`, `CNN+GRU`, `ViT`.
- unsupported benchmark models, if any, are reported explicitly.
- benchmark rank1/top3/top5 selection uses validation `Macro F1`.
- test `Macro F1` is confirmation only.
- F2/F3/F4/F5 use `training_mode=controlled_generalization`.
- F2 should use the benchmark rank 1 model from F1.
- F2 single preprocessing comparison must be run before combination comparison.
- `final_` outputs are used so official results do not mix with old prototype files.

Official F4 low-data outputs:

- result CSV: `results/metrics/final_low_data_results.csv`
- figures:
  - `results/figures/final_low_data_macro_f1_by_ratio.png`
  - `results/figures/final_low_data_accuracy_by_ratio.png`
  - `results/figures/final_low_data_macro_f1_retention_by_ratio.png`
  - `results/figures/final_low_data_macro_f1_drop_by_ratio.png`
- reports automatically embed generated figures when they exist.

Official F5 augmentation outputs:

- result CSV: `results/metrics/final_augmentation_results.csv`
- figures:
  - `results/figures/final_augmentation_gain_macro_f1_by_ratio.png`
  - `results/figures/final_augmentation_gain_accuracy_by_ratio.png`
  - `results/figures/final_augmentation_macro_f1_aug_vs_no_aug.png`
  - `results/figures/final_augmentation_25_10_summary.png`
- augmentation is evaluated against the F4 no-augmentation baseline at the same `real_ratio`.
- official F5 uses `augmentation_mode=offline_append`, not on-the-fly augmentation.
- `augmentation_add_ratio=1.0` means one synthetic sample is appended for each selected real train sample.
- reports automatically embed generated F5 figures when they exist.

## F3 Multi-seed Preprocessing Stability Check

Single-seed F2 can leave close candidates unresolved. Instead of choosing by test performance, the official F3 follow-up runs a multi-seed validation stability check on the strongest candidates and selects preprocessing by mean validation `Macro F1`.

- Primary selection criterion: mean validation `Macro F1` across seeds
- Stability is only a meaningful tie-break, not a replacement for the best mean validation score
- A lower-mean candidate can override only if `std_val_macro_f1` improves by at least `0.003` within the close tolerance
- Test `Macro F1` is confirmation only
- Default seeds: `42 43 44`
- Default candidates:
  - `savgol_smoothing+train_global_zscore`
  - `moving_average_smoothing+minmax_scaling`
  - `train_featurewise_zscore`
  - `minmax_scaling`
  - `moving_average_smoothing`
- Final preprocessing after F3: `moving_average_smoothing+minmax_scaling`
- It was selected by mean validation `Macro F1` across seeds.
- Test `Macro F1` is confirmation only.

Dry run:

```powershell
python experiments/16_run_preprocessing_stability_check.py --dry-run
```

Official stability check:

```powershell
python -u experiments/16_run_preprocessing_stability_check.py --seeds 42 43 44 --batch-size 64 2>&1 | Tee-Object -FilePath logs\final_preprocessing_stability_resnet18.log
```

OOM fallback:

```powershell
python -u experiments/16_run_preprocessing_stability_check.py --seeds 42 43 44 --batch-size 32 --overwrite 2>&1 | Tee-Object -FilePath logs\final_preprocessing_stability_resnet18_bs32.log
```

Regenerate final decision:

```powershell
python experiments/13_select_final_preprocessing.py
```

## Official F1 Benchmark

F1 is the official restart point for the final project. It uses `raw/none` preprocessing only, uses the original benchmark epoch policy, and does not use early stopping. The runner attempts the original UT_HAR_data model set and records unsupported models explicitly if any wrapper/runtime limitation appears.

## Additional Notes

- `original_baseline/` is preserved for benchmark compatibility.
- `src/` and `experiments/` contain the project-specific implementation used for the final workflow.
- `docs/project_plan.md` describes the clean F1-F6 experiment order.
