# Wi-Fi CSI HAR Workflow Status Report

This report reflects only the clean F1-F6 workflow status. Old prototype artifacts are not used as final evidence.

## Workflow Rules

- F1 uses `preprocessing=none/raw` and `training_mode=original_epoch`.
- F1 model selection uses validation `Macro F1`; test `Macro F1` is confirmation only.
- F2/F3/F4/F5 use `training_mode=controlled_generalization`.
- F2 uses the benchmark rank 1 model from F1.
- F3 selects final preprocessing primarily by mean validation `Macro F1` across seeds.
- Test `Macro F1` is confirmation only for preprocessing selection.
- F4/F5 use benchmark top3 by default and benchmark top5 only as an optional extension.
- final report should use only official final workflow outputs.

## Current Official Status
- F1 benchmark completed: yes
- F2 preprocessing comparison completed: yes
- F3 multi-seed stability check completed: yes
- Final preprocessing selected: `moving_average_smoothing+minmax_scaling`
- F4 low-data robustness: not completed yet.
- F5 augmentation recovery: not completed yet.


## Dataset and Labels

- dataset: `UT-HAR`
- array shape in this repo: `(N, 250, 90)`
- one sample = `250 CSI frame indices x 90 CSI features`
- activity labels: lie down, fall, walk, pickup, run, sit down, stand up
- timestep is `CSI frame index`, not directly seconds.
- if `sampling_rate = fs` Hz, one sample duration is `250 / fs` seconds.
- `100Hz` conversion is illustrative only and not confirmed ground truth.

Dataset figure:
- `results\figures\class_distribution_by_activity.png` uses activity names instead of numeric-only labels.


## F1. Original Benchmark Full Run

- selection metric = validation Macro F1
- test Macro F1 is confirmation only

| rank | model | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy | preprocessing | training_mode |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | ResNet18 | 0.9906 | 0.9807 | 0.9940 | 0.9880 | none | original_epoch |
| 2 | LeNet | 0.9887 | 0.9672 | 0.9940 | 0.9780 | none | original_epoch |
| 3 | ResNet101 | 0.9776 | 0.9230 | 0.9819 | 0.9500 | none | original_epoch |
| 4 | ResNet50 | 0.9773 | 0.9640 | 0.9798 | 0.9760 | none | original_epoch |
| 5 | ViT | 0.9707 | 0.9175 | 0.9738 | 0.9400 | none | original_epoch |
| 6 | MLP | 0.9471 | 0.8925 | 0.9496 | 0.9120 | none | original_epoch |
| 7 | GRU | 0.9196 | 0.8739 | 0.9294 | 0.9060 | none | original_epoch |
| 8 | LSTM | 0.9164 | 0.8748 | 0.9274 | 0.9120 | none | original_epoch |
| 9 | BiLSTM | 0.9063 | 0.8716 | 0.9274 | 0.9040 | none | original_epoch |
| 10 | RNN | 0.7322 | 0.6715 | 0.7440 | 0.7020 | none | original_epoch |
| 11 | CNN+GRU | 0.1461 | 0.1611 | 0.3750 | 0.4100 | none | original_epoch |


Benchmark selection document:

- `docs/final_benchmark_selection.md` exists.

## F2. Preprocessing Comparison

- F2 compares preprocessing candidates with the F1 benchmark rank 1 model.
- Validation Macro F1 is the selection metric.
- Test Macro F1 is confirmation only.

### F2 Single-method Results

| rank | preprocessing | model | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |
|---:|---|---|---:|---:|---:|---:|
| 1 | train_featurewise_zscore | ResNet18 | 0.9358 | 0.8955 | 0.9516 | 0.9300 |
| 2 | robust_scaling | ResNet18 | 0.9262 | 0.8680 | 0.9415 | 0.9120 |
| 3 | train_global_zscore | ResNet18 | 0.9154 | 0.8689 | 0.9274 | 0.9140 |
| 4 | savgol_smoothing | ResNet18 | 0.9153 | 0.8739 | 0.9274 | 0.9140 |
| 5 | moving_average_smoothing | ResNet18 | 0.9150 | 0.8920 | 0.9355 | 0.9260 |
| 6 | minmax_scaling | ResNet18 | 0.9121 | 0.9083 | 0.9274 | 0.9360 |
| 7 | none | ResNet18 | 0.9097 | 0.8874 | 0.9254 | 0.9220 |
| 8 | per_sample_zscore | ResNet18 | 0.9054 | 0.8763 | 0.9133 | 0.9180 |
| 9 | per_sample_featurewise_zscore | ResNet18 | 0.4546 | 0.3970 | 0.5302 | 0.4920 |

- Best single-method candidate in F2 by validation Macro F1: `train_featurewise_zscore`
- Final preprocessing is not selected from F2 single results alone. F3 multi-seed stability check determines the final preprocessing.

### F2 Combination Preprocessing Results

| rank | preprocessing | model | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |
|---:|---|---|---:|---:|---:|---:|
| 1 | savgol_smoothing+train_global_zscore | ResNet18 | 0.9472 | 0.8513 | 0.9536 | 0.9060 |
| 2 | moving_average_smoothing+minmax_scaling | ResNet18 | 0.9445 | 0.9008 | 0.9516 | 0.9400 |
| 3 | savgol_smoothing+train_featurewise_zscore | ResNet18 | 0.9287 | 0.8847 | 0.9355 | 0.9180 |
| 4 | savgol_smoothing+minmax_scaling | ResNet18 | 0.9262 | 0.8913 | 0.9415 | 0.9220 |
| 5 | moving_average_smoothing+train_featurewise_zscore | ResNet18 | 0.9227 | 0.8859 | 0.9415 | 0.9280 |

- These top combination candidates were forwarded into the F3 multi-seed stability check.

- `docs/final_preprocessing_decision.md` exists.


## F3. Multi-seed Preprocessing Stability Check

- F3 was added because single-seed F2 showed close preprocessing candidates.
- Final selection is based on mean validation Macro F1 across seeds.
- Stability is used only as a meaningful tie-break.
- Test Macro F1 is confirmation only.

| rank | preprocessing | mean_val_macro_f1 | std_val_macro_f1 | mean_test_macro_f1 | std_test_macro_f1 | mean_val_test_macro_f1_gap | num_seeds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | moving_average_smoothing+minmax_scaling | 0.9268 | 0.0178 | 0.9118 | 0.0120 | 0.0150 | 3 |
| 2 | savgol_smoothing+train_global_zscore | 0.9258 | 0.0222 | 0.8990 | 0.0338 | 0.0268 | 3 |
| 3 | moving_average_smoothing | 0.9227 | 0.0174 | 0.9067 | 0.0209 | 0.0159 | 3 |
| 4 | train_featurewise_zscore | 0.9160 | 0.0359 | 0.8799 | 0.0243 | 0.0361 | 3 |
| 5 | minmax_scaling | 0.9111 | 0.0142 | 0.9003 | 0.0114 | 0.0108 | 3 |

### Final Preprocessing Decision

- final selected preprocessing = `moving_average_smoothing+minmax_scaling`
- model = `ResNet18`
- seeds = 42,  43,  44
- mean_val_macro_f1 = 0.9268
- std_val_macro_f1 = 0.0178
- mean_test_macro_f1 = 0.9118
- std_test_macro_f1 = 0.0120

- `docs/final_preprocessing_decision.md` exists and records the official final preprocessing decision.


## F4. Low-data Robustness

F4 low-data robustness has not been completed yet.


## F5. Augmentation Recovery

F5 augmentation recovery has not been completed yet.


## F6. Final Report

F6 final report should be generated only after F1, F2/F3, F4, and F5 official outputs are available.

## Next Step

F4 low-data robustness should be run with:

- benchmark top3 models from F1
- final preprocessing = `moving_average_smoothing+minmax_scaling`
- training_mode = `controlled_generalization`

Command:

```powershell
python -u experiments/10_run_low_data_robustness.py --use-benchmark-top3 --preprocessing moving_average_smoothing+minmax_scaling --seed 42 --batch-size 64 2>&1 | Tee-Object -FilePath logs\final_low_data_top3.log
```

OOM fallback:

```powershell
python -u experiments/10_run_low_data_robustness.py --use-benchmark-top3 --preprocessing moving_average_smoothing+minmax_scaling --seed 42 --batch-size 32 --overwrite 2>&1 | Tee-Object -FilePath logs\final_low_data_top3_bs32.log
```

