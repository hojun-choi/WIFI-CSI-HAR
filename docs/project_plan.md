# Wi-Fi CSI HAR Final Experiment Plan

이전 개발 과정에서 생성된 prototype artifacts는 정리되었고, 최종 실험은 아래 F1부터 다시 수행한다.

## 1. Problem Definition

Wi-Fi CSI 기반 HAR는 행동 인식에 활용될 수 있지만, labeled data가 부족하거나 환경이 바뀌면 일반화가 어렵다. 같은 행동이라도 사용자, 공간, 배치, 노이즈 조건이 달라지면 CSI 분포가 변할 수 있기 때문이다.

본 프로젝트는 `UT-HAR`를 사용해 다음을 순차적으로 검증한다.

- benchmark model comparison
- preprocessing comparison and final selection
- low-data robustness
- augmentation recovery

핵심 원칙은 한 번에 여러 결론을 섞지 않고, F1 결과가 F2 후보를 정하고, F2/F3 결과가 F4/F5의 입력 조건을 고정하는 누적형 workflow를 유지하는 것이다.

## 2. Dataset and Input Window

- Dataset: `UT-HAR`
- 이 저장소에서 로드하는 입력 array shape: `(N, 250, 90)`
- 한 sample shape: `(250, 90)`
- `250 timesteps`는 초 단위 시간이 아니라 `CSI frame index`로 해석한다.
- `90 CSI features`는 `Intel 5300 NIC` 스타일 해석에서 `30 subcarriers x 3 antenna pairs`로 볼 수 있다.
- timestep을 초 단위로 바꾸려면 `sampling_rate`가 필요하다.
- `sampling_rate = fs` Hz라면 한 sample duration은 `250 / fs` seconds이다.
- `100Hz` 환산은 설명용 illustrative assumption일 뿐이며 confirmed ground truth로 사용하지 않는다.

UT-HAR class names:

- `0`: `lie down`
- `1`: `fall`
- `2`: `walk`
- `3`: `pickup`
- `4`: `run`
- `5`: `sit down`
- `6`: `stand up`

Report-facing dataset visualization은 numeric label만 쓰지 않고 위 activity names를 사용한다.

## 3. Official Final Workflow

### F1. Original Benchmark Full Run

목적:

- `UT_HAR_data`에 대해 original benchmark model set을 최대한 그대로 실행한다.
- F1은 official final workflow의 시작점이다.
- F1 결과로 benchmark rank 1, top3, top5를 정한다.

규칙:

- `preprocessing = none/raw`
- `training_mode = original_epoch`
- no early stopping
- `real_ratio = 1.0`
- `augmentation = false`
- selection metric = validation `Macro F1`
- test `Macro F1` is confirmation only

F1 benchmark model set:

- `MLP`
- `LeNet`
- `ResNet18`
- `ResNet50`
- `ResNet101`
- `RNN`
- `GRU`
- `LSTM`
- `BiLSTM`
- `CNN+GRU`
- `ViT`

지원 원칙:

- runner는 위 original supervised model list를 우선 지원한다.
- 어떤 model이 현재 wrapper에서 실행되지 않으면 조용히 제외하지 않는다.
- unsupported model은 이유와 함께 `final_benchmark_results.csv` 및 `docs/final_benchmark_selection.md`에 명시한다.

output:

- `results/metrics/final_benchmark_results.csv`
- `docs/final_benchmark_selection.md`

selection:

- benchmark rank 1 model
- benchmark top3 models
- benchmark top5 optional

### F2. Single Preprocessing Comparison

목적:

- F1에서 선택된 benchmark rank 1 model 하나를 사용해 individual preprocessing candidates를 one-by-one 비교한다.

규칙:

- F2부터는 `controlled_generalization`을 사용한다.
- `original_epoch` fixed training은 F1까지만 사용한다.
- single preprocessing comparison이 combination comparison보다 먼저 수행되어야 한다.
- selection metric = validation `Macro F1`
- test `Macro F1` is confirmation only

individual candidates:

- `none / raw`
- `train_global_zscore`
- `per_sample_zscore`
- `minmax_scaling`
- `robust_scaling`
- `savgol_smoothing`
- `moving_average_smoothing`
- `train_featurewise_zscore`
- `per_sample_featurewise_zscore`

output:

- `results/metrics/final_preprocessing_results.csv`

### F3. Preprocessing Combination and Final Selection

목적:

- F2 single-method 결과를 본 뒤, 유망하거나 논리적으로 상보적인 preprocessing combinations만 추가 비교한다.
- blind combination search는 수행하지 않는다.
- close candidate가 남으면 multi-seed validation stability check로 하나의 final preprocessing policy를 고정한다.

candidate combinations:

- `savgol_smoothing + per_sample_zscore`
- `savgol_smoothing + train_global_zscore`
- `moving_average_smoothing + per_sample_zscore`
- `minmax_scaling + savgol_smoothing` if justified

selection rule:

- primary metric = validation `Macro F1`
- close candidate가 남으면 mean validation `Macro F1` across seeds를 우선한다.
- close tolerance 안에서는 highest mean validation `Macro F1` candidate를 기본으로 유지한다.
- 더 낮은 mean candidate는 `std_val_macro_f1` improvement가 `0.003` 이상일 때만 override할 수 있다.
- test `Macro F1` is confirmation only
- 점수가 비슷하면 더 단순하고 해석 가능한 preprocessing을 우선한다.

output:

- `docs/final_preprocessing_decision.md`

## F3 Multi-seed Preprocessing Stability Check

F2 single-seed 결과에서 close candidates가 남으면, test performance로 직접 선택하지 않고 F3 multi-seed stability check를 추가로 수행한다.

기본 원칙:

- benchmark rank 1 model 사용
- `training_mode = controlled_generalization`
- `real_ratio = 0.25`
- `augmentation = false`
- default seeds: `42`, `43`, `44`
- primary selection criterion: mean validation `Macro F1` across seeds
- stability is used only as a meaningful tie-break
- a lower-mean candidate can override only if `std_val_macro_f1` improves by at least `0.003` within the close tolerance
- test `Macro F1` is confirmation only

default candidates:

- `savgol_smoothing+train_global_zscore`
- `moving_average_smoothing+minmax_scaling`
- `train_featurewise_zscore`
- `minmax_scaling`
- `moving_average_smoothing`

artifacts:

- `results/metrics/final_preprocessing_stability_results.csv`
- `results/metrics/final_preprocessing_stability_summary.csv`
- `results/figures/final_preprocessing_stability_mean_val_macro_f1.png`
- `results/figures/final_preprocessing_stability_val_test_macro_f1.png`
- `docs/final_preprocessing_decision.md`

### F4. Low-data Robustness

목적:

- labeled train data가 줄어들 때 모델 성능이 어떻게 감소하는지 비교한다.

규칙:

- final selected preprocessing 사용
- benchmark top3 models를 기본으로 사용
- benchmark top5는 runtime 여유가 있을 때만 optional extension으로 사용
- `real_ratio = 1.0, 0.5, 0.25, 0.1`
- `augmentation = false`
- `training_mode = controlled_generalization`
- `validation/test`는 unchanged
- train split만 deterministic `stratified sampling`으로 축소

output:

- `results/metrics/final_low_data_results.csv`

### F5. Augmentation Recovery

목적:

- low-data 조건에서 train-only augmentation이 성능 회복에 도움이 되는지 확인한다.

규칙:

- final selected preprocessing 사용
- F4와 동일한 benchmark top3/top5 model set 사용
- `real_ratio = 0.5, 0.25, 0.1`
- `augmentation = true`
- augmentation은 train-only
- `validation/test`는 절대 augmentation하지 않는다
- F4 no-augmentation 결과와 직접 비교한다

output:

- `results/metrics/final_augmentation_results.csv`

### F6. Final Report

규칙:

- official final workflow outputs만 final evidence로 사용한다.
- old prototype 결과를 final conclusion과 섞지 않는다.
- report는 다음 내용을 포함한다.

포함 항목:

- problem definition
- dataset and input explanation
- benchmark model comparison
- preprocessing comparison
- final preprocessing decision
- low-data robustness
- augmentation recovery
- limitations
- future work

## 4. Preprocessing Candidate Pool

### `none / raw`

- normalization이나 smoothing을 적용하지 않는 기준선이다.
- preprocessing 효과를 판단하기 위한 baseline으로 사용한다.

### `train_global_zscore`

- selected train split에서 mean/std를 추정한다.
- 학습된 통계를 `train/val/test`에 동일하게 적용한다.
- validation/test leakage를 피할 수 있다.

### `per_sample_zscore`

- 각 sample을 독립적으로 normalize한다.
- sample-level amplitude variation을 줄이는 데 유용할 수 있다.
- absolute amplitude cue를 일부 제거할 수 있다는 한계가 있다.

### `Min-Max Normalization`

- 값을 고정된 범위로 rescale한다.
- gradient 안정성과 convergence 측면에서 도움이 될 수 있다.
- signal noise를 제거하는 것이 아니라 scale만 바꾼다.

### `Robust Scaling`

- median/IQR 기반 scaling이다.
- outlier에 더 강건할 수 있다.
- CSI 열화가 outlier보다 fluctuation/noise 중심이면 효과가 제한될 수 있다.

### `Savitzky-Golay Smoothing`

- local temporal structure를 비교적 유지하면서 high-frequency noise를 억제할 가능성이 있는 candidate이다.
- official F2/F3 결과 전에는 superiority를 주장하지 않는다.
- best method로 가정하지 않는다.

### `Moving Average Smoothing`

- 단순한 smoothing baseline이다.
- noise 감소에는 도움이 될 수 있지만 activity pattern을 blur할 수 있다.

### `train_featurewise_zscore`

- selected train split에서 feature별 mean/std를 학습한다.
- global z-score보다 feature-aware한 scaling이다.

### `per_sample_featurewise_zscore`

- 각 sample 내부에서 feature별 time-axis normalization을 수행한다.
- feature별 temporal scale variation을 줄일 수 있다.
- 유용한 amplitude cue를 제거할 수 있다.

Preprocessing implementation rules:

- train-stat 기반 transform은 selected train split에만 fit한다.
- fit된 train statistics를 `train/val/test`에 적용한다.
- deterministic smoothing은 `train/val/test`에 일관되게 적용한다.
- augmentation은 preprocessing과 분리되며 train-only로 유지한다.

## 5. Evaluation Metrics

- `Accuracy`
- `Macro F1`
- `Weighted F1`
- `Confusion Matrix` where useful

low-data degradation metrics:

- `macro_f1_drop`
- `macro_f1_retention`
- `accuracy_drop`
- `accuracy_retention`

augmentation gain metrics:

- `augmentation_gain_macro_f1`
- `augmentation_gain_accuracy`

## 6. Execution Checklist

- [ ] Clean old generated artifacts
- [ ] F1 original benchmark full run completed
- [ ] Benchmark rank 1 selected
- [ ] Benchmark top3/top5 selected
- [ ] F2 single preprocessing comparison completed
- [ ] F3 preprocessing combination comparison completed
- [ ] Final preprocessing selected
- [ ] F4 low-data robustness completed
- [ ] F5 augmentation recovery completed
- [ ] Final report generated
