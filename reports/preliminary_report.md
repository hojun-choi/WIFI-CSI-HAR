# Wi-Fi CSI HAR Workflow Status Report

This report reflects only the clean F1-F5 workflow status. Old prototype artifacts are not used as final evidence.

## Workflow Rules

- F1 uses `preprocessing=none/raw` and `training_mode=original_epoch`.
- F1 model selection uses validation `Macro F1`; test `Macro F1` is confirmation only.
- F2/F3/F4/F5 use `training_mode=controlled_generalization`.
- F2 uses the benchmark rank 1 model from F1.
- F3 selects final preprocessing primarily by mean validation `Macro F1` across seeds.
- Test `Macro F1` is confirmation only for preprocessing selection.
- F4 uses benchmark top3 models by default.
- final report should use only official final workflow outputs.

## Current Official Status

- F1 benchmark completed: yes
- F2 preprocessing comparison completed: yes
- F3 multi-seed stability check completed: yes
- Final preprocessing selected: `moving_average_smoothing+minmax_scaling`
- F4 low-data robustness: completed
- F5 augmentation recovery: completed


## Dataset and Labels

- dataset: `UT-HAR`
- array shape in this repo: `(N, 250, 90)`
- one sample = `250 CSI frame indices x 90 CSI features`
- activity labels: lie down, fall, walk, pickup, run, sit down, stand up
- timestep is `CSI frame index`, not directly seconds.
- if `sampling_rate = fs` Hz, one sample duration is `250 / fs` seconds.
- `100Hz` conversion is illustrative only and not confirmed ground truth.

![UT-HAR class distribution by activity](../results/figures/class_distribution_by_activity.png)

![UT-HAR split size summary](../results/figures/split_size_summary.png)

![UT-HAR sample CSI heatmap](../results/figures/sample_csi_heatmap.png)

![UT-HAR sample CSI line plot](../results/figures/sample_csi_lineplot.png)

![UT-HAR sample heatmap by activity](../results/figures/sample_heatmap_by_activity.png)


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

![Final preprocessing validation Macro F1](../results/figures/final_preprocessing_val_macro_f1.png)

![Final preprocessing validation and test Macro F1](../results/figures/final_preprocessing_val_test_macro_f1.png)

![Final preprocessing accuracy comparison](../results/figures/final_preprocessing_accuracy.png)

![Preprocessing stability mean validation Macro F1](../results/figures/final_preprocessing_stability_mean_val_macro_f1.png)

![Preprocessing stability validation and test Macro F1](../results/figures/final_preprocessing_stability_val_test_macro_f1.png)

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
- seeds = 42, 43, 44
- mean_val_macro_f1 = 0.9268
- std_val_macro_f1 = 0.0178
- mean_test_macro_f1 = 0.9118
- std_test_macro_f1 = 0.0120

- `docs/final_preprocessing_decision.md` exists and records the official final preprocessing decision.


## F4. Low-data Robustness

| model | real_ratio | test_macro_f1 | test_accuracy | macro_f1_drop | macro_f1_retention | accuracy_drop | accuracy_retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| LeNet | 1.0000 | 0.9621 | 0.9740 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| LeNet | 0.5000 | 0.9333 | 0.9540 | 0.0288 | 0.9701 | 0.0200 | 0.9795 |
| LeNet | 0.2500 | 0.8368 | 0.8840 | 0.1253 | 0.8698 | 0.0900 | 0.9076 |
| LeNet | 0.1000 | 0.0649 | 0.2940 | 0.8972 | 0.0675 | 0.6800 | 0.3018 |
| ResNet101 | 1.0000 | 0.9763 | 0.9800 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| ResNet101 | 0.5000 | 0.9395 | 0.9560 | 0.0367 | 0.9624 | 0.0240 | 0.9755 |
| ResNet101 | 0.2500 | 0.8565 | 0.8980 | 0.1198 | 0.8773 | 0.0820 | 0.9163 |
| ResNet101 | 0.1000 | 0.7372 | 0.8060 | 0.2391 | 0.7551 | 0.1740 | 0.8224 |
| ResNet18 | 1.0000 | 0.9629 | 0.9740 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| ResNet18 | 0.5000 | 0.9361 | 0.9600 | 0.0268 | 0.9722 | 0.0140 | 0.9856 |
| ResNet18 | 0.2500 | 0.9008 | 0.9400 | 0.0621 | 0.9355 | 0.0340 | 0.9651 |
| ResNet18 | 0.1000 | 0.7543 | 0.8160 | 0.2085 | 0.7834 | 0.1580 | 0.8378 |

### F4 Interpretation

- At 25% training data, the strongest model by test Macro F1 is `ResNet18` (0.9008).
- At 10% training data, the strongest model by test Macro F1 is `ResNet18` (0.7543).
- `LeNet` collapses sharply at 10%, with test Macro F1 dropping to 0.0649.
- At 50% training data, all benchmark top3 models remain relatively stable.
- At 25%, model differences become clearer.
- At 10%, the best Macro F1 retention belongs to `ResNet18` (0.7834).

![Low-data robustness test Macro F1 by train ratio](../results/figures/final_low_data_macro_f1_by_ratio.png)

![Low-data robustness test accuracy by train ratio](../results/figures/final_low_data_accuracy_by_ratio.png)

![Macro F1 retention under reduced training data](../results/figures/final_low_data_macro_f1_retention_by_ratio.png)

![Macro F1 drop under reduced training data](../results/figures/final_low_data_macro_f1_drop_by_ratio.png)

![Low-data robustness summary at 25% and 10%](../results/figures/final_low_data_25_10_summary.png)


## F5. Augmentation Recovery

- This official F5 design uses offline appended synthetic samples, not on-the-fly augmentation.
- Synthetic samples are generated only from the selected real train subset. Validation/test are never augmented.

| model | real_ratio | selected_real_train_size | synthetic_train_size | effective_train_size | augmentation_add_ratio | no_aug_test_macro_f1 | test_macro_f1 | no_aug_test_accuracy | test_accuracy | augmentation_gain_macro_f1 | augmentation_gain_accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LeNet | 0.5000 | 1987 | 1987 | 3974 | 1.0000 | 0.9333 | 0.9086 | 0.9540 | 0.9380 | -0.0247 | -0.0160 |
| LeNet | 0.2500 | 992 | 992 | 1984 | 1.0000 | 0.8368 | 0.8444 | 0.8840 | 0.8840 | 0.0076 | 0.0000 |
| LeNet | 0.1000 | 395 | 395 | 790 | 1.0000 | 0.0649 | 0.7325 | 0.2940 | 0.7940 | 0.6676 | 0.5000 |
| ResNet101 | 0.5000 | 1987 | 1987 | 3974 | 1.0000 | 0.9395 | 0.9175 | 0.9560 | 0.9420 | -0.0220 | -0.0140 |
| ResNet101 | 0.2500 | 992 | 992 | 1984 | 1.0000 | 0.8565 | 0.8772 | 0.8980 | 0.9100 | 0.0207 | 0.0120 |
| ResNet101 | 0.1000 | 395 | 395 | 790 | 1.0000 | 0.7372 | 0.6868 | 0.8060 | 0.7600 | -0.0504 | -0.0460 |
| ResNet18 | 0.5000 | 1987 | 1987 | 3974 | 1.0000 | 0.9361 | 0.9398 | 0.9600 | 0.9600 | 0.0037 | 0.0000 |
| ResNet18 | 0.2500 | 992 | 992 | 1984 | 1.0000 | 0.9008 | 0.8664 | 0.9400 | 0.9040 | -0.0344 | -0.0360 |
| ResNet18 | 0.1000 | 395 | 395 | 790 | 1.0000 | 0.7543 | 0.7567 | 0.8160 | 0.8180 | 0.0024 | 0.0020 |

### F5 Interpretation

- F5 evaluates offline appended synthetic augmentation against the F4 no-augmentation baseline.
- Synthetic samples are generated only from the selected train subset. Validation/test are never augmented.
- Positive Macro F1 gain rows: 5; negative rows: 4; zero rows: 0.
- Largest positive Macro F1 gain: `LeNet` at `real_ratio=0.1` (0.6676).
- Largest negative Macro F1 gain: `ResNet101` at `real_ratio=0.1` (-0.0504).
- Overall, augmentation improves more model/ratio pairs than it degrades.

![Offline appended augmentation Macro F1 gain by train ratio](../results/figures/final_augmentation_gain_macro_f1_by_ratio.png)

![Offline appended augmentation accuracy gain by train ratio](../results/figures/final_augmentation_gain_accuracy_by_ratio.png)

![Real data plus synthetic augmentation versus no-augmentation Macro F1](../results/figures/final_augmentation_macro_f1_aug_vs_no_aug.png)

![Offline appended augmentation summary at 25% and 10%](../results/figures/final_augmentation_25_10_summary.png)

![Offline appended augmentation gain heatmap](../results/figures/final_augmentation_gain_heatmap.png)


## F6. Final Report

F6 final report can now be generated from the official F1-F5 outputs. Test-set figures and tables above should be reused as final evidence.

## Next Step

F5 is complete, so the next step is final report regeneration and packaging from the official F1-F5 outputs.

Command:

```powershell
python experiments/09_build_preliminary_report.py
```

