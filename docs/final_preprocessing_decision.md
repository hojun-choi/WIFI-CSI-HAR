# Final Preprocessing Decision

## Selection Source

- Decision source: multi-seed preprocessing stability check
- Summary file: `results/metrics/final_preprocessing_stability_summary.csv`

## Selection Rule

- Primary: mean validation Macro F1 across seeds
- Stability tie-break: lower std of validation Macro F1 within close tolerance
- Close tolerance: 0.0050
- Test Macro F1: confirmation only

## Selected Final Preprocessing

- selected preprocessing: `moving_average_smoothing`
- model: `ResNet18`
- number of seeds: 3
- seeds: 42, 43, 44
- mean_val_macro_f1: 0.9227
- std_val_macro_f1: 0.0174
- mean_test_macro_f1: 0.9067
- std_test_macro_f1: 0.0209
- mean_val_test_macro_f1_gap: 0.0159

## Best Validation Candidate

- best candidate by raw mean validation Macro F1: `moving_average_smoothing+minmax_scaling`
- raw best mean_val_macro_f1: 0.9268

## Ranked Stability Results

| rank | preprocessing | mean_val_macro_f1 | std_val_macro_f1 | mean_test_macro_f1 | std_test_macro_f1 | mean_val_test_macro_f1_gap | num_seeds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | moving_average_smoothing | 0.9227 | 0.0174 | 0.9067 | 0.0209 | 0.0159 | 3 |
| 2 | moving_average_smoothing+minmax_scaling | 0.9268 | 0.0178 | 0.9118 | 0.0120 | 0.0150 | 3 |
| 3 | savgol_smoothing+train_global_zscore | 0.9258 | 0.0222 | 0.8990 | 0.0338 | 0.0268 | 3 |
| 4 | train_featurewise_zscore | 0.9160 | 0.0359 | 0.8799 | 0.0243 | 0.0361 | 3 |
| 5 | minmax_scaling | 0.9111 | 0.0142 | 0.9003 | 0.0114 | 0.0108 | 3 |

## Leakage and Implementation Checks

- Train-statistics-based preprocessing was fit only on the selected train split.
- Fitted train statistics were applied to train/val/test.
- Deterministic smoothing was applied consistently to train/val/test.
- Augmentation was disabled in this stability check.

## Limitations

- Only selected top candidates were checked.
- Additional candidates or more seeds may change the ranking.
- Test is not used for selection.
