# Final Preprocessing Decision

## Selection Source

- Decision source: single-seed F2 result fallback
- Stability summary file was not found, so the current official F2 result files were used.

## Selection Rule

- Primary: validation Macro F1 from the official F2 run
- Close tolerance: 0.0050
- Simplicity preference applies only when validation scores are very close.
- Test Macro F1: confirmation only

## Selected Final Preprocessing

- selected preprocessing: `savgol_smoothing+train_global_zscore`
- model: `ResNet18`
- number of seeds: 1
- seeds: 42
- mean_val_macro_f1: 0.9472
- std_val_macro_f1: -
- mean_test_macro_f1: 0.8513
- std_test_macro_f1: -
- mean_val_test_macro_f1_gap: 0.0959

## Best Validation Candidate

- best candidate by raw validation Macro F1: `savgol_smoothing+train_global_zscore`
- raw best val_macro_f1: 0.9472

## Ranked Stability Results

Stability summary has not been created yet. The current fallback ranking is shown below.

| rank | preprocessing | model | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |
|---:|---|---|---:|---:|---:|---:|
| 1 | savgol_smoothing+train_global_zscore | ResNet18 | 0.9472 | 0.8513 | 0.9536 | 0.9060 |
| 2 | moving_average_smoothing+minmax_scaling | ResNet18 | 0.9445 | 0.9008 | 0.9516 | 0.9400 |
| 3 | train_featurewise_zscore | ResNet18 | 0.9358 | 0.8955 | 0.9516 | 0.9300 |
| 4 | savgol_smoothing+train_featurewise_zscore | ResNet18 | 0.9287 | 0.8847 | 0.9355 | 0.9180 |
| 5 | robust_scaling | ResNet18 | 0.9262 | 0.8680 | 0.9415 | 0.9120 |
| 6 | savgol_smoothing+minmax_scaling | ResNet18 | 0.9262 | 0.8913 | 0.9415 | 0.9220 |
| 7 | moving_average_smoothing+train_featurewise_zscore | ResNet18 | 0.9227 | 0.8859 | 0.9415 | 0.9280 |
| 8 | train_global_zscore | ResNet18 | 0.9154 | 0.8689 | 0.9274 | 0.9140 |
| 9 | savgol_smoothing | ResNet18 | 0.9153 | 0.8739 | 0.9274 | 0.9140 |
| 10 | moving_average_smoothing | ResNet18 | 0.9150 | 0.8920 | 0.9355 | 0.9260 |
| 11 | minmax_scaling | ResNet18 | 0.9121 | 0.9083 | 0.9274 | 0.9360 |
| 12 | none | ResNet18 | 0.9097 | 0.8874 | 0.9254 | 0.9220 |
| 13 | per_sample_zscore | ResNet18 | 0.9054 | 0.8763 | 0.9133 | 0.9180 |
| 14 | per_sample_featurewise_zscore | ResNet18 | 0.4546 | 0.3970 | 0.5302 | 0.4920 |

## Leakage and Implementation Checks

- Train-statistics-based preprocessing was fit only on the selected train split.
- Fitted train statistics were applied to train/val/test.
- Deterministic smoothing was applied consistently to train/val/test.
- Augmentation was disabled in preprocessing comparison.

## Limitations

- Only one seed is available in this fallback decision.
- A multi-seed stability check should be preferred when available.
- Test is not used for selection.
