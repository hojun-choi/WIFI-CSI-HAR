# Wi-Fi CSI 기반 HAR Preliminary Report

## 1. 문서 성격

이 문서는 현재 저장소에 존재하는 실험 결과를 정리한 `preliminary_report`이다. 기존 codebase는 계속 재사용하지만, 현재 저장된 `results/`와 `reports/` 산출물은 revised final project 관점에서 prototype/development output으로 취급한다. 따라서 아래 표와 해석은 구현 검증과 계획 수립에는 유용하지만, final report의 공식 결론으로 직접 사용하지 않는다.

## 2. 문제 정의와 데이터

Wi-Fi CSI 기반 `Human Activity Recognition`은 `environment`, `subject`, `device position`, `room layout`, `noise` 변화에 민감하다. 이 프로젝트는 `UT-HAR`에서 `real_ratio`를 줄였을 때 성능이 얼마나 유지되는지, 그리고 preprocessing, model choice, `train-only augmentation`이 그 일반화 성능에 어떤 영향을 주는지를 분석하려고 한다.

- Dataset: `UT-HAR`
- Input shape: `X_train=(3977, 250, 90)`, `X_val=(496, 250, 90)`, `X_test=(500, 250, 90)`
- Label range: `0..6`
- One sample: `250 timesteps x 90 CSI features`
- Benchmark code preserved under `original_baseline/`

참고 figure:

- `results/figures/class_distribution.png`
- `results/figures/sample_csi_heatmap.png`
- `results/figures/sample_csi_lineplot.png`

## 3. CSI frame budget 해석

본 저장소에서는 sample 수와 함께 `CSI frame count`를 직접 기록한다. `real_ratio`가 줄어들면 selected train sample 수가 줄고, 각 sample이 `250 CSI frames`를 포함하므로 총 labeled training frame budget도 함께 줄어든다.

### Total CSI frame count by real_ratio

| real_ratio | selected_train_size | timesteps_per_sample | total_train_csi_frames |
|---:|---:|---:|---:|
| 1 | 3977 | 250 | 994250 |
| 0.5 | 1987 | 250 | 496750 |
| 0.25 | 992 | 250 | 248000 |
| 0.1 | 395 | 250 | 98750 |

### 100Hz 가정 시 예시 시간 길이

| real_ratio | selected_train_size | estimated_duration_at_100Hz_seconds | estimated_duration_at_100Hz_minutes |
|---:|---:|---:|---:|
| 1 | 3977 | 9942.5 | 165.7 |
| 0.5 | 1987 | 4967.5 | 82.8 |
| 0.25 | 992 | 2480.0 | 41.3 |
| 0.1 | 395 | 987.5 | 16.5 |

위 시간 환산은 `sampling_rate=100Hz` 가정에 기반한 보조 설명일 뿐이며, 본 프로젝트의 주 분석 단위는 시간보다 `CSI frame count`와 `real_ratio`다.

## 4. Prototype Benchmark Snapshot

현재 benchmark prototype은 `training_mode=original_epoch`와 `baseline_results_original_epoch.csv`를 기준으로 정리할 수 있다. 이 결과는 useful benchmark prototype이지만, revised final workflow에서는 F1 official benchmark run으로 다시 생성하거나 `final_` prefix output으로 분리할 수 있다.

- `training_mode=original_epoch`
- `augmentation=false`
- `real_ratio=1.0`
- current preprocessing in the saved prototype run: `per_sample_zscore`
- device in saved CSV: `cuda`

| model | best_epoch | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |
|---|---:|---:|---:|---:|---:|
| CNN | 200 | 0.9937 | 0.9708 | 0.9940 | 0.9820 |
| GRU | 74 | 0.9920 | 0.9622 | 0.9940 | 0.9740 |
| LSTM | 160 | 0.9852 | 0.9620 | 0.9899 | 0.9740 |
| CNN_GRU | 164 | 0.9914 | 0.9628 | 0.9940 | 0.9760 |

이 표는 prototype benchmark ranking snapshot으로는 유용하지만, 이후 preprocessing decision과 low-data/augmentation 결론까지 대신해 주지는 않는다.

## 5. Prototype Preprocessing Snapshot

현재 preprocessing 결과는 제한된 candidate만 비교한 preliminary snapshot이다. saved comparison은 `none`, `train_global_zscore`, `per_sample_zscore`만 포함하며, revised workflow에서 요구하는 `Min-Max Normalization`, `Robust Scaling`, `Savitzky-Golay Smoothing`, `Moving Average Smoothing`, `train_featurewise_zscore`, `per_sample_featurewise_zscore`는 아직 official comparison에 포함되지 않았다.

controlled setting:

- model=`GRU`
- `real_ratio=0.25`
- `augmentation=false`
- `seed=42`
- selection metric=`validation Macro F1`

| preprocessing | val_macro_f1 | test_macro_f1 |
|---|---:|---:|
| per_sample_zscore | 0.8851 | 0.8351 |
| train_global_zscore | 0.8569 | 0.7983 |
| none | 0.6918 | 0.6305 |

현재 저장된 limited candidate 비교에서는 `per_sample_zscore`가 가장 높은 validation `Macro F1`를 보인다. 그러나 이 결과는 어디까지나 limited candidate pool에서의 preliminary observation이다. 따라서 revised workflow에서는 expanded preprocessing comparison을 다시 수행하고, 그 결과가 나오기 전에는 `per_sample_zscore`나 `Savitzky-Golay Smoothing` 중 어느 것도 final best라고 주장하지 않는다.

참고 figure:

- `results/figures/preprocessing_ablation_macro_f1.png`
- `results/figures/preprocessing_ablation_val_test_macro_f1_zoomed.png`

## 7. Prototype Low-data Robustness Snapshot

본 섹션은 현재 저장된 `results/metrics/low_data_results.csv`를 요약하지만, 이는 revised workflow 기준 final evidence가 아니라 prototype/development snapshot이다. 특히 현재 run은 `per_sample_zscore` 기반이므로 F3에서 final preprocessing이 확정되면 F4 official low-data run을 다시 수행할 수 있다.

| model | real_ratio | test_macro_f1 | macro_f1_drop | macro_f1_retention | test_accuracy | accuracy_drop | accuracy_retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 1.0000 | 0.9595 | 0.0000 | 1.0000 | 0.9720 | 0.0000 | 1.0000 |
| CNN | 0.5000 | 0.9551 | 0.0044 | 0.9954 | 0.9700 | 0.0020 | 0.9979 |
| CNN | 0.2500 | 0.8627 | 0.0968 | 0.8991 | 0.9080 | 0.0640 | 0.9342 |
| CNN | 0.1000 | 0.7196 | 0.2399 | 0.7499 | 0.7840 | 0.1880 | 0.8066 |
| CNN_GRU | 1.0000 | 0.9730 | 0.0000 | 1.0000 | 0.9820 | 0.0000 | 1.0000 |
| CNN_GRU | 0.5000 | 0.9300 | 0.0431 | 0.9557 | 0.9540 | 0.0280 | 0.9715 |
| CNN_GRU | 0.2500 | 0.8761 | 0.0969 | 0.9004 | 0.9180 | 0.0640 | 0.9348 |
| CNN_GRU | 0.1000 | 0.7600 | 0.2131 | 0.7810 | 0.8020 | 0.1800 | 0.8167 |
| GRU | 1.0000 | 0.9515 | 0.0000 | 1.0000 | 0.9680 | 0.0000 | 1.0000 |
| GRU | 0.5000 | 0.9305 | 0.0210 | 0.9779 | 0.9520 | 0.0160 | 0.9835 |
| GRU | 0.2500 | 0.8611 | 0.0904 | 0.9050 | 0.9040 | 0.0640 | 0.9339 |
| GRU | 0.1000 | 0.7133 | 0.2382 | 0.7497 | 0.7740 | 0.1940 | 0.7996 |
| LSTM | 1.0000 | 0.9614 | 0.0000 | 1.0000 | 0.9720 | 0.0000 | 1.0000 |
| LSTM | 0.5000 | 0.9059 | 0.0555 | 0.9422 | 0.9260 | 0.0460 | 0.9527 |
| LSTM | 0.2500 | 0.8313 | 0.1301 | 0.8646 | 0.8800 | 0.0920 | 0.9053 |
| LSTM | 0.1000 | 0.7137 | 0.2478 | 0.7423 | 0.7720 | 0.2000 | 0.7942 |

현재 prototype low-data 결과에서 읽을 수 있는 제한적 관찰:
- `real_ratio=0.1`에서 highest `test_macro_f1` model은 `CNN_GRU`이다.
- `real_ratio=0.1`에서 smallest `macro_f1_drop` model은 `CNN_GRU`이다.

- 이 해석은 current prototype run에 대한 요약일 뿐이며, final report 결론으로 직접 사용하지 않는다.

참고 figure:

- `results/figures/low_data_macro_f1.png`
- `results/figures/low_data_accuracy.png`
- `results/figures/low_data_degradation_macro_f1.png`
- `results/figures/low_data_degradation_accuracy.png`


## 8. Augmentation Recovery Status

`results/metrics/augmentation_results.csv`가 아직 없으므로 본 섹션은 pending 상태다. revised workflow에서는 final preprocessing selection 이후 F4 no-augmentation baseline을 확정한 다음, `real_ratio=0.5, 0.25, 0.1`에서 train-only augmentation recovery를 official하게 다시 실행해야 한다.


## 9. Revised Final Workflow and Remaining Work

- Existing results are currently treated as prototype/development outputs.
- The final report should be based on official revised workflow outputs.
- Expanded preprocessing comparison is the next implementation task.
- Existing `per_sample_zscore` result is preliminary among limited candidates only.
- M4 low-data results may need to be rerun after final preprocessing is selected.
- M5 full augmentation recovery remains pending for the revised workflow.
- Future result files should preferably use `final_`-prefixed filenames.

official revised workflow:

1. F1 benchmark model comparison with `original_epoch`
2. F2 expanded preprocessing comparison
3. F3 final preprocessing selection by validation `Macro F1`
4. F4 low-data robustness with selected preprocessing
5. F5 augmentation recovery against F4 no-augmentation baseline
6. F6 final report from official revised outputs only

## 10. 한계와 주의사항

- 현재 문서는 final report가 아니라 prototype snapshot 정리 문서다.
- saved preprocessing comparison은 candidate pool이 제한적이다.
- saved low-data 결과는 current preprocessing choice에 종속되어 있으므로 F3 이후 rerun 필요성이 있다.
- `augmentation_results.csv`가 없으면 augmentation 결론은 비워 둔다.
- 현재 결과 대부분은 single dataset, single seed 관점의 development evidence다.

## 11. 참고 명령

```bash
python experiments/01_check_data.py
```

```bash
python experiments/06_run_preprocessing_ablation.py --epochs 30 --real-ratio 0.25 --seed 42 --batch-size 64
```

```bash
python experiments/07_run_full_baseline_all_models.py --training-mode original_epoch --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

```bash
python experiments/10_run_low_data_robustness.py --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

```bash
python experiments/11_run_augmentation_recovery.py --seed 42 --batch-size 64 --preprocessing per_sample_zscore
```

```bash
python experiments/08_regenerate_figures.py
```
