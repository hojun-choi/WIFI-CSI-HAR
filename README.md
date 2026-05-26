# Wi-Fi CSI-based HAR on UT-HAR

Project-specific experimental framework for Wi-Fi CSI-based Human Activity Recognition on `UT-HAR`, focused on real training data ratio, data augmentation, and sequence model comparison.

Tested with Python `3.10.11`.

## Project Motivation

Wi-Fi CSI-based HAR often struggles with `generalization` because CSI signals are sensitive to environment changes, subject differences, device placement, room layout, and noise. In practice, collecting enough labeled CSI data for every new setup is expensive, so a useful system should remain reasonably robust even when real labeled training data is limited.

This project studies that low-data robustness problem on `UT-HAR` by intentionally reducing the real training data ratio and measuring performance degradation, then testing whether `train-only augmentation` can recover some of that loss. The original benchmark code is preserved under `original_baseline/`, while this repository builds a clean, report-friendly experimental pipeline focused on interpretable comparison.

## Preprocessing Strategy

The project will include a small preprocessing ablation because preprocessing can affect robustness under limited data. The main experiments will use one selected preprocessing policy consistently, and any normalization statistics will be computed only from the train split to avoid leakage.

## Environment Setup

Use Windows PowerShell and Python `3.10.11`.

Create and activate a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch for an NVIDIA GPU.

Note: Choose the CUDA-enabled PyTorch command from the official PyTorch Get Started page if this command changes.

Default command:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Fallback if `cu128` fails:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then install the remaining project dependencies:

```powershell
pip install -r requirements.txt
```

Verify GPU availability:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Do not run long training until the dry-run command succeeds.

The expected laptop GPU is `NVIDIA RTX 4050 Laptop GPU`, but the code must still fall back to CPU if CUDA is not available.

## Dataset Placement

Place the UT-HAR files at the following paths:

```text
data/UT_HAR/data/X_train.csv
data/UT_HAR/data/X_val.csv
data/UT_HAR/data/X_test.csv
data/UT_HAR/label/y_train.csv
data/UT_HAR/label/y_val.csv
data/UT_HAR/label/y_test.csv
```

These files keep the `.csv` extension from the benchmark, but they are NumPy binary files and must be loaded with `np.load`.

## UT-HAR Input Window

Each UT-HAR sample has shape `(250, 90)`.

- `250 timesteps` are interpreted in this processed benchmark as `CSI frame` indices.
- `90 CSI features` can be interpreted in `Intel 5300 NIC` style CSI layout as `30 subcarriers x 3 antenna pairs`.
- Therefore, one sample contains `250 CSI frames`.

The exact real-time duration depends on `sampling_rate`.

- `duration_seconds = 250 / sampling_rate`

This repository reports `CSI frame count` directly as the rigorous quantity. A `100Hz` conversion may be shown only as an optional illustrative estimate, not as a confirmed ground-truth timing for UT-HAR.

## Current Project Status

- The repository already contains working prototype pipelines for data checking / EDA, benchmark-style baselines, preprocessing ablation, low-data robustness, augmentation smoke testing, figure regeneration, and report generation.
- Existing outputs under `results/` and `reports/` should now be treated as prototype/development results, not final project evidence.
- The revised final project will reuse this codebase and regenerate official outputs through a clean sequential workflow.
- The F2 expanded preprocessing comparison infrastructure is implemented, but official F1/F2 full runs are still pending.

## Final Revised Workflow

Final results should be generated from the revised workflow below, not by mixing older prototype files.

1. F1 benchmark full run with `original_epoch`
2. F2 single preprocessing comparison with the benchmark rank 1 model
3. F2 combination preprocessing comparison after reviewing single-method results
4. F3 final preprocessing selection by validation `Macro F1`
5. F4 low-data robustness with final preprocessing and benchmark top3 by default
6. F5 augmentation recovery with the same final preprocessing and same top3/top5 model set
7. F6 final report from official outputs only

This workflow is cumulative:

- F1 uses the original benchmark epoch policy and identifies the benchmark rank 1 model plus benchmark top models.
- F2 and later stages use `controlled_generalization`.
- F2 must compare single preprocessing methods before any combination comparison.
- F3 freezes the final preprocessing policy before official low-data and augmentation experiments.
- F4 uses benchmark top3 models by default, with benchmark top5 only as an optional extension.
- F5 compares augmentation against the F4 no-augmentation baseline using the same model set.
- F6 uses only official final workflow outputs as final evidence.

The next official experiment task is the F1 benchmark full run, followed by F2 single preprocessing comparison with the benchmark rank 1 model.

Warning:

- The existing `per_sample_zscore` result is preliminary because `Min-Max Normalization`, `Robust Scaling`, `Savitzky-Golay Smoothing`, and other candidates have not yet been tested in the revised workflow.
- Do not mix prototype result files and final result files in the report.

## Quick Start

```bash
python experiments/01_check_data.py
```

Install dependencies first if needed:

```bash
pip install -r requirements.txt
```

## Run Full-data Baseline

Original benchmark epoch policy:

```bash
python experiments/07_run_full_baseline_all_models.py --training-mode original_epoch --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

This mode follows the original benchmark's model-specific epoch policy. Early stopping is disabled, but the best checkpoint is still selected by validation `Macro F1`.

Controlled early stopping mode:

```bash
python experiments/07_run_full_baseline_all_models.py --training-mode early_stopping --epochs 50 --patience 8 --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

This mode is useful for faster controlled experiments. It should not be mixed with `original_epoch` results without clear labeling.

If CUDA OOM occurs, retry with `--batch-size 32`.

After a successful full baseline run, report figures are regenerated automatically by default. Use `--no-regenerate-figures` only when you explicitly want to skip that post-processing step.

Expected outputs:

- `results/metrics/baseline_results_original_epoch.csv`
- `results/metrics/baseline_results_early_stopping.csv`
- `results/figures/baseline_original_epoch_macro_f1.png`
- `results/figures/baseline_original_epoch_accuracy.png`
- `results/figures/baseline_early_stopping_macro_f1.png`
- `results/figures/baseline_early_stopping_accuracy.png`
- `results/checkpoints/`

## Controlled Generalization Training

From `M4/M5` onward, use `controlled_generalization` mode by default instead of blindly reusing the `200 epoch` benchmark policy. This mode is designed to avoid both overfitting from very long fixed training and premature stopping from naive early stopping.

Recommended command:

```bash
python experiments/07_run_full_baseline_all_models.py --training-mode controlled_generalization --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

If CUDA OOM occurs, retry with:

```bash
python experiments/07_run_full_baseline_all_models.py --training-mode controlled_generalization --seed 42 --batch-size 32 --preprocessing per_sample_zscore
```

Key parameters:

- `max_epochs`: upper bound on training length, recommended default `100`
- `warmup_epochs`: no early stopping before this many epochs, recommended default `20`
- `patience`: wait after the best validation score before stopping, recommended default `15`
- `min_delta`: ignore tiny `Macro F1` fluctuations, recommended default `0.001`
- `weight_decay`: regularization term, recommended default `1e-4`
- `gradient_clip_norm`: stabilize training, recommended default `1.0`
- `scheduler_type`: `plateau` by default for conservative LR reduction

`original_epoch` mode is still available, but only for original benchmark-style comparison with the copied baseline.

## Run Expanded Preprocessing Comparison

F2 should use the benchmark rank 1 model from F1. First run single preprocessing methods. Then run only promising or logically complementary combinations. F2 and later stages use `controlled_generalization`, while F1 benchmark comparison uses `original_epoch`.

Validation `Macro F1` selects preprocessing. Test `Macro F1` is confirmation only. Official F2 outputs use `final_`-prefixed filenames so they do not mix with older prototype files.

Smoke test:

```bash
python experiments/12_run_expanded_preprocessing_comparison.py --smoke-test --seed 42 --batch-size 64
```

Official single-method F2 after benchmark rank 1 is known:

```bash
python experiments/12_run_expanded_preprocessing_comparison.py --models CNN --comparison-mode single --seed 42 --batch-size 64
```

Official combination F2 after reviewing single-method results:

```bash
python experiments/12_run_expanded_preprocessing_comparison.py --models CNN --comparison-mode combination --seed 42 --batch-size 64
```

Optional all-in-one:

```bash
python experiments/12_run_expanded_preprocessing_comparison.py --models CNN --comparison-mode all --seed 42 --batch-size 64
```

CUDA OOM fallback:

```bash
python experiments/12_run_expanded_preprocessing_comparison.py --models CNN --comparison-mode single --seed 42 --batch-size 32
```

Final preprocessing decision:

```bash
python experiments/13_select_final_preprocessing.py
```

## Run Low-data Robustness

Smoke test:

```bash
python experiments/10_run_low_data_robustness.py --smoke-test --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

Full `M4` run:

```bash
python experiments/10_run_low_data_robustness.py --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

CUDA OOM fallback:

```bash
python experiments/10_run_low_data_robustness.py --seed 42 --batch-size 32 --preprocessing per_sample_zscore
```

This experiment runs `CNN`, `GRU`, `LSTM`, and `CNN_GRU` over `real_ratio` values `1.0`, `0.5`, `0.25`, and `0.1`. It uses `controlled_generalization`, not `original_epoch`.

The validation and test sets remain unchanged. Only the train split is reduced, and the reduced train subset uses stratified sampling. Results are saved to `results/metrics/low_data_results.csv`, and the figures show how performance degrades as `real_ratio` decreases.

## Run Augmentation Recovery

Smoke test:

```bash
python experiments/11_run_augmentation_recovery.py --smoke-test --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

Full `M5` run:

```bash
python experiments/11_run_augmentation_recovery.py --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

CUDA OOM fallback:

```bash
python experiments/11_run_augmentation_recovery.py --seed 42 --batch-size 32 --preprocessing per_sample_zscore
```

This experiment runs `CNN`, `GRU`, `LSTM`, and `CNN_GRU` over `real_ratio` values `0.5`, `0.25`, and `0.1`. It uses `train-only augmentation` with `controlled_generalization`, compares the augmented results against `results/metrics/low_data_results.csv`, and never augments validation/test data.

Results are saved to `results/metrics/augmentation_results.csv`, and the figures summarize both augmentation recovery and augmentation gain:

- `results/figures/augmentation_recovery_macro_f1.png`
- `results/figures/augmentation_recovery_accuracy.png`
- `results/figures/augmentation_gain_macro_f1.png`
- `results/figures/augmentation_gain_accuracy.png`

## Revised Workflow Note

The old `M4` and `M5` artifacts remain useful for debugging and planning, but they should not be presented as the final project conclusion. The revised final workflow should prefer fresh `final_`-prefixed outputs when those paths are implemented, so prototype and official files stay separated.

## Regenerate Report Figures

```bash
python experiments/08_regenerate_figures.py
```

This command does not run training. It only reads saved CSV files and regenerates report-ready figures. The same regeneration logic can also be called automatically after `experiments/07_run_full_baseline_all_models.py` finishes successfully.

For high-score metrics, the regenerated figures use a zoomed y-axis and value labels so small differences remain visible. Supplementary gap plots use `1 - score` to make near-1.0 differences easier to explain in the report.

Representative figure outputs include:

- `results/figures/sample_csi_heatmap.png`
- `results/figures/sample_csi_lineplot.png`
- `results/figures/baseline_original_epoch_val_test_macro_f1_zoomed.png`
- `results/figures/baseline_original_epoch_val_test_accuracy_zoomed.png`
- `results/figures/preprocessing_ablation_val_test_macro_f1_zoomed.png`

`sample_csi_heatmap.png` shows the full `time step x CSI feature` matrix. `sample_csi_lineplot.png` shows selected CSI features over time, which is closer to a waveform view of the signal.

Accuracy and F1 figures use a zoomed y-axis plus value labels because the full-data baseline scores are very high and a raw `0~1` axis hides small but meaningful differences.

## Repository Layout

- `original_baseline/`: copied benchmark code preserved for direct comparison with the original implementation.
- `src/` and `experiments/`: clean project-specific code for the final course experiments and report-ready analysis.
