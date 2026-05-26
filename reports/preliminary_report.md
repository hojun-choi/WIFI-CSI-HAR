# Wi-Fi CSI HAR Workflow Status Report

This report reflects only the clean F1-F6 workflow status. Old prototype artifacts are not used as final evidence.

## Workflow Rules

- F1 uses `preprocessing=none/raw` and `training_mode=original_epoch`.
- F1 model selection uses validation `Macro F1`; test `Macro F1` is confirmation only.
- F2/F3/F4/F5 use `training_mode=controlled_generalization`.
- F2 uses the benchmark rank 1 model from F1.
- F4/F5 use benchmark top3 by default and benchmark top5 only as an optional extension.
- final report should use only official final workflow outputs.

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

- selection metric = validation Macro F1
- test Macro F1 is confirmation only

| preprocessing_group | model | preprocessing | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |
|---|---|---|---:|---:|---:|---:|
| single | ResNet18 | train_featurewise_zscore | 0.9358 | 0.8955 | 0.9516 | 0.9300 |
| single | ResNet18 | robust_scaling | 0.9262 | 0.8680 | 0.9415 | 0.9120 |
| single | ResNet18 | train_global_zscore | 0.9154 | 0.8689 | 0.9274 | 0.9140 |
| single | ResNet18 | savgol_smoothing | 0.9153 | 0.8739 | 0.9274 | 0.9140 |
| single | ResNet18 | moving_average_smoothing | 0.9150 | 0.8920 | 0.9355 | 0.9260 |
| single | ResNet18 | minmax_scaling | 0.9121 | 0.9083 | 0.9274 | 0.9360 |
| single | ResNet18 | none | 0.9097 | 0.8874 | 0.9254 | 0.9220 |
| single | ResNet18 | per_sample_zscore | 0.9054 | 0.8763 | 0.9133 | 0.9180 |
| single | ResNet18 | per_sample_featurewise_zscore | 0.4546 | 0.3970 | 0.5302 | 0.4920 |

- current best preprocessing by validation `Macro F1`: `train_featurewise_zscore`
- `docs/final_preprocessing_decision.md` exists.


## F4. Low-data Robustness

F4 low-data robustness has not been completed yet.


## F5. Augmentation Recovery

F5 augmentation recovery has not been completed yet.


## F6. Final Report

F6 final report should be generated only after F1, F2/F3, F4, and F5 official outputs are available.
