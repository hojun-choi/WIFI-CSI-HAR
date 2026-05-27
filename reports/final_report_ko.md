# Wi-Fi CSI 기반 행동 인식에서 학습 데이터 절감 가능성 분석

## 초록

본 보고서는 `UT-HAR` 데이터셋을 사용하여 Wi-Fi CSI 기반 Human Activity Recognition(HAR)에서 전처리, 모델 선택, low-data robustness, 그리고 augmentation 전략이 성능 유지에 어떤 영향을 주는지 분석한다. 전체 실험은 F1 `original benchmark`, F2 preprocessing comparison, F3 multi-seed preprocessing stability check, F4 low-data robustness, F5 offline appended augmentation recovery, F5.1 augmentation add ratio ablation의 순차적 workflow로 수행되었다. 최종 전처리는 `moving_average_smoothing+minmax_scaling`으로 선정되었고, 이는 seed `42, 43, 44` 기준 mean validation `Macro F1 = 0.9268`, mean test `Macro F1 = 0.9118`을 기록하였다.

핵심 결과는 다음과 같다. F4에서 `ResNet18`은 `real_ratio=0.25` 조건에서 test `Macro F1 = 0.9008`, test `Accuracy = 0.9400`을 기록하며 가장 실용적인 low-data 모델로 나타났다. F5 offline appended augmentation에서는 일부 조건에서 성능 회복이 관찰되었고, F5.1 ablation에서는 `augmentation_add_ratio=0.5`가 평균적으로 가장 안정적인 설정으로 나타났다. 반면 `augmentation_add_ratio=2.0`은 과도한 synthetic data 비율로 인해 성능을 악화시키는 경우가 많았다. 즉, augmentation은 잠재력이 있지만 “많을수록 항상 좋다”는 결론으로 이어지지 않으며, 모델과 데이터 비율에 맞춘 조정이 필요하다.

## 1. 연구 배경 및 문제 정의

### 1.1 고령화와 돌봄 공백

한국은 빠르게 고령사회에서 초고령사회로 이동하고 있다. 통계청 장래인구추계 기준 65세 이상 인구 비중은 2025년 20.3%, 2036년 30.9%, 2050년 40% 초과로 전망된다. 이러한 변화는 단순한 인구구조 변동을 넘어 돌봄 수요 증가, 단독가구 확대, 일상 안전 관리 필요성 증대와 직결된다. 동시에 고령화는 생산가능인구 비중 축소를 동반하므로, 돌봄 인력을 사람 중심 방식만으로 지속적으로 확대하기는 어렵다.

사회적 고립과 고독사 문제도 이러한 배경에서 더욱 중요해지고 있다. 보건복지부 발표 기준 2024년 고독사 사망자는 3,924명으로 2023년 대비 증가했다. 이는 고령층의 이상행동, 낙상, 장시간 무활동 상태를 보다 조기에 감지할 수 있는 비침습적 모니터링 기술의 필요성을 보여준다.

### 1.2 기존 모니터링 방식의 한계

기존 모니터링 방식은 대체로 camera 기반, 버튼 호출 기반, 또는 사람 중심 점검 방식으로 구분할 수 있다. Camera 기반 방식은 정보량이 풍부하고 이상행동을 직접적으로 확인할 수 있다는 장점이 있지만, 사생활 침해 우려와 해킹 위험 때문에 실제 주거공간에서의 수용성이 낮을 수 있다. 특히 침실이나 거실 같은 생활공간에서는 이러한 우려가 더 크게 작동한다.

버튼 호출이나 정기 응답 확인 방식은 구조가 단순하지만 사용자의 지속적인 협조를 필요로 한다. 낙상이나 의식 저하처럼 사용자가 직접 응답할 수 없는 상황에서는 가장 필요한 순간에 실패할 수 있다. 사람 중심 상시 관찰은 비용과 인력 제약으로 인해 대규모 확장이 어렵다.

### 1.3 Wi-Fi CSI의 가능성

Wi-Fi CSI(Channel State Information)는 무선 신호가 공간을 통과하는 과정에서 발생하는 경로 변화와 위상 변화를 반영한다. 사람이 움직이거나 자세를 바꾸면 전파 경로가 변하고, 이 변화가 CSI 패턴에 나타난다. 따라서 CSI를 이용하면 사람의 외형을 직접 촬영하지 않고도 움직임과 활동 패턴을 감지할 수 있다.

이 점에서 Wi-Fi CSI 기반 HAR는 privacy-friendly한 행동 모니터링 기술로서 의미가 있다. 낙상, 보행, 앉기/서기 같은 행동을 비시각적으로 추정할 수 있기 때문이다. 그러나 CSI는 환경에 매우 민감하다. 가구 배치, 벽체 재질, 송수신기 위치, 사람의 체형, 잡음 조건에 따라 분포가 달라질 수 있으므로, 학습된 모델이 새로운 환경에서도 안정적으로 일반화된다고 보장하기 어렵다.

### 1.4 본 연구의 핵심 목표

실제 배치 환경에서는 site 또는 household 단위의 calibration/training이 필요할 가능성이 높다. 따라서 중요한 질문은 “새로운 환경에 적응하기 위해 얼마나 많은 labeled training data가 필요한가”이다. 본 연구는 이를 학습 데이터 절감 관점에서 다룬다.

> **Wi-Fi CSI 기반 HAR에서 전처리와 모델 선택을 통해 적은 학습 데이터로도 행동 인식 성능을 어느 정도 유지할 수 있는가?**

부가적으로 다음 질문도 함께 다룬다.

> **Data augmentation이 low-data 상황에서 성능 회복에 도움이 되는가?**

`real_ratio`는 train split에서 실제로 사용하는 샘플 비율을 의미한다. 이를 calibration time reduction의 잠재적 지표로 해석할 수는 있지만, 이는 엄밀한 실제 수집 시간 측정이 아니라 frame count 기반 단순 환산에 가깝다. 본 프로젝트에서 한 sample은 `250 CSI frame indices`로 구성되므로, 만약 `sampling_rate = 100Hz`라고 가정하면 sample당 약 `2.5s`, `50Hz`라면 약 `5s`에 해당한다. 다만 이는 illustrative estimate일 뿐이며, 데이터셋의 실제 sampling rate가 본 프로젝트에서 확인된 것은 아니다.

| train ratio | sample 수 | 100Hz 가정 | 50Hz 가정 |
|---|---:|---:|---:|
| 100% | 3977 | `3977 x 250 / 100 = 9942.5s` ≈ `165.7분` | `3977 x 250 / 50 = 19885s` ≈ `331.4분` |
| 25% | 992 | `992 x 250 / 100 = 2480s` ≈ `41.3분` | `992 x 250 / 50 = 4960s` ≈ `82.7분` |
| 10% | 395 | `395 x 250 / 100 = 987.5s` ≈ `16.5분` | `395 x 250 / 50 = 1975s` ≈ `32.9분` |

이는 연속 수집 시간의 엄밀한 측정이 아니라 frame count 기반 단순 환산이다.

## 2. 데이터셋 및 입력 구조

본 연구는 `UT-HAR` 데이터셋을 사용하였다. 본 프로젝트에서 입력은 `(N, 250, 90)` 형태로 로드되며, 각 sample은 `250 CSI frame indices x 90 CSI features`로 구성된다. timestep은 초 단위 시간이 아니라 `CSI frame index`이며, 시간으로 변환하려면 별도의 `sampling_rate` 정보가 필요하다. 본 보고서에서는 데이터 설명을 위해서만 예시적 환산을 제시하며, 실제 ground truth sampling rate를 확정적으로 주장하지 않는다.

`90`차원 feature는 benchmark 문맥상 `30 subcarriers x 3 antenna pairs`로 해석할 수 있으나, 본 연구의 분석은 우선 90차원 CSI feature 입력 자체에 초점을 둔다. 행동 label은 다음 7개이다.

- `lie down`
- `fall`
- `walk`
- `pickup`
- `run`
- `sit down`
- `stand up`

![UT-HAR class distribution by activity](../results/figures/class_distribution_by_activity.png)

![UT-HAR sample CSI heatmap](../results/figures/sample_csi_heatmap.png)

![UT-HAR sample CSI line plot](../results/figures/sample_csi_lineplot.png)

## 3. 실험 설계

### 3.1 전체 실험 흐름

전체 실험은 F1부터 F5.1까지 순차적으로 설계되었다.

- **F1**: `preprocessing=none/raw`와 `training_mode=original_epoch`로 original benchmark-compatible model을 비교하였다.
- **F2**: F1에서 validation `Macro F1` 기준 rank 1로 선정된 `ResNet18`을 사용하여 preprocessing 후보를 비교하였다.
- **F3**: F2에서 근접한 후보들이 나타났기 때문에 multi-seed stability check를 수행하여 최종 preprocessing을 확정하였다.
- **F4**: 최종 preprocessing을 고정한 뒤 benchmark top3인 `ResNet18`, `LeNet`, `ResNet101`에 대해 low-data robustness를 평가하였다.
- **F5**: F4 no-augmentation baseline과 같은 `real_ratio`를 유지한 상태에서 `offline_append` augmentation을 적용하여 성능 회복 가능성을 평가하였다.
- **F5.1**: F5에서 사용한 `augmentation_add_ratio=1.0`을 기준으로, `0.5`, `1.0`, `2.0`을 비교하는 synthetic data ratio ablation을 추가 수행하였다.

### 3.2 평가 지표

주요 평가 지표는 `Accuracy`와 `Macro F1`이다. 이 중 모델 및 preprocessing 선택의 핵심 기준은 `Macro F1`이다. `Macro F1`은 각 클래스의 성능을 균형 있게 반영하므로, 일부 클래스에 편향된 높은 정확도보다 행동 전반의 안정성을 평가하는 데 적합하다. `Accuracy`는 전체적인 정답률을 보여주는 보조 지표로 활용하였다.

### 3.3 학습 데이터 축소 방식

`real_ratio`는 train split에만 적용되며 validation/test split은 유지된다. 따라서 low-data 실험은 평가 세트 축소가 아니라 train data availability 감소에 대한 민감도를 분석하는 실험이다. train subset은 deterministic `stratified sampling`으로 구성하여 클래스 분포 왜곡을 최소화하였다.

또한 `seed 42, 43, 44`는 train/validation/test를 새로 나눈 것이 아니라, 동일 실험을 다른 랜덤 조건으로 반복해 안정성을 점검하기 위한 설정이다. 즉, 이는 K-fold split이 아니라 sampling, initialization, shuffle 차이에 대한 robustness 확인에 가깝다.

## 4. 모델 Benchmark 결과

F1 benchmark에서는 validation `Macro F1`을 기준으로 모델을 정렬하고, test `Macro F1`은 confirmation only로 사용하였다. 그 결과 `ResNet18`이 benchmark rank 1 model로 선정되었고, top3는 `ResNet18`, `LeNet`, `ResNet101`이었다.

| rank | model | val_macro_f1 | test_macro_f1 | val_accuracy | test_accuracy |
|---:|---|---:|---:|---:|---:|
| 1 | ResNet18 | 0.9906 | 0.9807 | 0.9940 | 0.9880 |
| 2 | LeNet | 0.9887 | 0.9672 | 0.9940 | 0.9780 |
| 3 | ResNet101 | 0.9776 | 0.9230 | 0.9819 | 0.9500 |
| 4 | ResNet50 | 0.9773 | 0.9640 | 0.9798 | 0.9760 |
| 5 | ViT | 0.9707 | 0.9175 | 0.9738 | 0.9400 |

따라서 이후 preprocessing 비교는 `ResNet18`을 중심으로 진행하였고, low-data 및 augmentation 실험은 benchmark top3를 사용하였다.

## 5. 전처리 비교 및 최종 전처리 선택

### 5.1 전처리 후보

본 연구에서 비교한 전처리 후보는 다음과 같다.

- `none/raw`
- `train_global_zscore`
- `per_sample_zscore`
- `minmax_scaling`
- `robust_scaling`
- `savgol_smoothing`
- `moving_average_smoothing`
- `train_featurewise_zscore`
- `per_sample_featurewise_zscore`
- 일부 조합형 preprocessing

### 5.2 F2 single/combination 결과

F2 single-method comparison에서는 `train_featurewise_zscore`가 validation `Macro F1` 기준 가장 높은 성능을 보였다. 그러나 combination 비교에서는 `savgol_smoothing+train_global_zscore`와 `moving_average_smoothing+minmax_scaling`이 매우 근접한 결과를 보였고, single-seed 결과만으로 최종 정책을 확정하기에는 불확실성이 있었다. 따라서 F2는 후보군을 좁히는 단계로 해석하는 것이 적절하다.

### 5.3 F3 multi-seed stability check

F3는 F2에서 근접했던 후보들의 seed 안정성을 검증하기 위해 수행되었다. 최종적으로 선택된 preprocessing은 `moving_average_smoothing+minmax_scaling`이다. 해당 조합은 seed `42, 43, 44` 기준 다음과 같은 결과를 보였다.

- `mean_val_macro_f1 = 0.9268`
- `mean_test_macro_f1 = 0.9118`
- `seeds = 42, 43, 44`

선정 규칙은 mean validation `Macro F1` 우선이며, stability는 meaningful tie-break에만 사용되었다. 따라서 test `Macro F1`은 selection이 아니라 confirmation에 해당한다.

![Preprocessing stability mean validation Macro F1](../results/figures/final_preprocessing_stability_mean_val_macro_f1.png)

![Preprocessing stability validation and test Macro F1](../results/figures/final_preprocessing_stability_val_test_macro_f1.png)

정리하면, 본 프로젝트의 final preprocessing은 `moving_average_smoothing+minmax_scaling`이다.

## 6. Low-data Robustness 결과

F4에서는 final preprocessing을 고정한 뒤, benchmark top3인 `ResNet18`, `LeNet`, `ResNet101`에 대해 `real_ratio = 1.0, 0.5, 0.25, 0.1`을 비교하였다.

| model | real_ratio | test_macro_f1 | test_accuracy | macro_f1_retention |
|---|---:|---:|---:|---:|
| LeNet | 1.00 | 0.9621 | 0.9740 | 1.0000 |
| LeNet | 0.50 | 0.9333 | 0.9540 | 0.9701 |
| LeNet | 0.25 | 0.8368 | 0.8840 | 0.8698 |
| LeNet | 0.10 | 0.0649 | 0.2940 | 0.0675 |
| ResNet101 | 1.00 | 0.9763 | 0.9800 | 1.0000 |
| ResNet101 | 0.50 | 0.9395 | 0.9560 | 0.9624 |
| ResNet101 | 0.25 | 0.8565 | 0.8980 | 0.8773 |
| ResNet101 | 0.10 | 0.7372 | 0.8060 | 0.7551 |
| ResNet18 | 1.00 | 0.9629 | 0.9740 | 1.0000 |
| ResNet18 | 0.50 | 0.9361 | 0.9600 | 0.9722 |
| ResNet18 | 0.25 | 0.9008 | 0.9400 | 0.9355 |
| ResNet18 | 0.10 | 0.7543 | 0.8160 | 0.7834 |

핵심 해석은 다음과 같다.

- `real_ratio=0.5`에서는 benchmark top3 모두 비교적 안정적인 성능을 유지한다.
- `real_ratio=0.25`에서는 `ResNet18`이 `test_macro_f1 = 0.9008`, `test_accuracy = 0.9400`, `macro_f1_retention = 0.9355`로 가장 강한 결과를 보인다.
- `real_ratio=0.1`에서도 `ResNet18`이 `test_macro_f1 = 0.7543`으로 가장 높은 성능을 유지한다.
- 반면 `LeNet`은 `real_ratio=0.1`에서 `test_macro_f1 = 0.0649`로 급격한 붕괴를 보인다.
- 여기서 raw `Macro F1`과 retention은 서로 다른 지표다. raw `Macro F1`은 실제 성능 자체를 보여주고, retention은 각 모델의 `real_ratio=1.0` 성능을 1.0으로 정규화한 상대 유지율을 의미한다.
- 따라서 retention plot에서 `100% = 1.0`으로 나타나는 것은 모든 모델이 동일한 raw 성능을 냈다는 뜻이 아니라, 각 모델의 full-data baseline을 자기 자신으로 나눈 정규화 결과라는 점을 분명히 해야 한다.

![Low-data robustness test Macro F1 by train ratio](../results/figures/final_low_data_macro_f1_by_ratio.png)

![Macro F1 retention under reduced training data](../results/figures/final_low_data_macro_f1_retention_by_ratio.png)

![Low-data robustness summary at 25% and 10%](../results/figures/final_low_data_25_10_summary.png)

## 7. Data Augmentation Recovery 결과

F5와 F5.1은 low-data setting에서 synthetic augmented data가 동일한 `real_ratio`의 실제 train subset을 얼마나 보완할 수 있는지 평가하기 위한 실험이다.

### 7.1 F5 offline appended augmentation

F5에서는 기존 on-the-fly augmentation이 아니라 `offline_append` synthetic augmentation을 사용하였다. 즉, `real_ratio`로 선택된 실제 train subset을 먼저 고정하고, 해당 subset에서 synthetic sample을 생성해 train set에 추가하였다. Validation/test set에는 augmentation을 적용하지 않았다.

또한 비교 기준은 반드시 같은 `real_ratio`의 F4 `no-augmentation` baseline이다. 따라서 `real_ratio=0.25`의 augmentation 결과는 `real_ratio=0.25`의 F4 결과와만 비교한다.

| model | real_ratio | selected_real_train_size | synthetic_train_size | effective_train_size | no_aug_test_macro_f1 | test_macro_f1 | augmentation_gain_macro_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LeNet | 0.5 | 1987 | 1987 | 3974 | 0.9333 | 0.9086 | -0.0247 |
| LeNet | 0.25 | 992 | 992 | 1984 | 0.8368 | 0.8444 | 0.0076 |
| LeNet | 0.1 | 395 | 395 | 790 | 0.0649 | 0.7325 | 0.6676 |
| ResNet101 | 0.5 | 1987 | 1987 | 3974 | 0.9395 | 0.9175 | -0.0220 |
| ResNet101 | 0.25 | 992 | 992 | 1984 | 0.8565 | 0.8772 | 0.0207 |
| ResNet101 | 0.1 | 395 | 395 | 790 | 0.7372 | 0.6868 | -0.0504 |
| ResNet18 | 0.5 | 1987 | 1987 | 3974 | 0.9361 | 0.9398 | 0.0037 |
| ResNet18 | 0.25 | 992 | 992 | 1984 | 0.9008 | 0.8664 | -0.0344 |
| ResNet18 | 0.1 | 395 | 395 | 790 | 0.7543 | 0.7567 | 0.0024 |

F5에서는 9개 조건 중 5개에서 positive gain, 4개에서 negative gain이 나타났다. 가장 큰 positive gain은 `LeNet`의 `real_ratio=0.1`에서 `+0.6676`, 가장 큰 negative gain은 `ResNet101`의 `real_ratio=0.1`에서 `-0.0504`였다. 즉, augmentation은 일부 low-data 조건에서 회복 효과를 보였지만, 모델과 ratio에 따라 방향과 크기가 크게 달랐다.

![Offline appended augmentation Macro F1 gain by train ratio](../results/figures/final_augmentation_gain_macro_f1_by_ratio.png)

![Real data plus synthetic augmentation versus no-augmentation Macro F1](../results/figures/final_augmentation_macro_f1_aug_vs_no_aug.png)

![Offline appended augmentation gain heatmap](../results/figures/final_augmentation_gain_heatmap.png)

### 7.2 F5.1 Augmentation Add Ratio Ablation

official F5에서 `augmentation_add_ratio=1.0`을 사용한 뒤, synthetic data 양이 augmentation 효과를 어떻게 바꾸는지 확인하기 위해 `augmentation_add_ratio = 0.5, 1.0, 2.0` ablation을 추가 수행하였다. 비교 규칙은 그대로 유지되어, 각 augmented 결과는 동일한 모델, 동일한 `real_ratio`, 동일한 preprocessing에 대한 F4 no-augmentation baseline과만 비교하였다. 또한 `augmentation_add_ratio=1.0` 행은 official F5 결과를 재사용했고, `0.5`와 `2.0`은 추가 ablation run으로 생성되었다.

| model | real_ratio | best augmentation_add_ratio by test_macro_f1 | best test_macro_f1 | no_aug_test_macro_f1 |
|---|---:|---:|---:|---:|
| ResNet18 | 0.50 | 0.5 | 0.9486 | 0.9361 |
| ResNet18 | 0.25 | 2.0 | 0.8836 | 0.9008 |
| ResNet18 | 0.10 | 0.5 | 0.8132 | 0.7543 |
| LeNet | 0.50 | 0.5 | 0.9287 | 0.9333 |
| LeNet | 0.25 | 0.5 | 0.8614 | 0.8368 |
| LeNet | 0.10 | 2.0 | 0.7567 | 0.0649 |
| ResNet101 | 0.50 | 0.5 | 0.9297 | 0.9395 |
| ResNet101 | 0.25 | 1.0 | 0.8772 | 0.8565 |
| ResNet101 | 0.10 | 0.5 | 0.7698 | 0.7372 |

F5.1의 요약은 다음과 같다.

- `augmentation_add_ratio=0.5`는 평균 `augmentation_gain_macro_f1`와 평균 `test_macro_f1` 관점에서 가장 안정적인 설정으로 나타났다.
- `augmentation_add_ratio=1.0`은 일부 조건에서 유용했지만 전반적으로 최적이라고 보기는 어려웠다.
- `augmentation_add_ratio=2.0`은 과도한 synthetic data 비율로 인해 성능을 떨어뜨리는 경우가 많았다.
- 즉, synthetic data는 “많을수록 좋다”기보다 모델과 `real_ratio`에 따라 적절한 양이 달라진다.
- `LeNet`의 `real_ratio=0.1`에서는 큰 개선이 나타났지만, 이는 baseline 붕괴 이후의 회복으로 해석해야 하며 universal superiority를 의미하지 않는다.
- `ResNet18`은 F4에서 가장 실용적인 low-data 모델이었으나, augmentation tuning이 항상 추가 이득으로 이어지지는 않았다.
- 특히 `ResNet18`, `real_ratio=0.5`, `augmentation_add_ratio=2.0`에서는 test `Macro F1`이 급락하는 severe collapse가 관찰되었다.

![Augmentation add ratio ablation gain](../results/figures/final_augmentation_ablation_gain_by_add_ratio.png)

![Augmentation add ratio heatmap](../results/figures/final_augmentation_ablation_heatmap.png)

![Best augmentation add ratio by condition](../results/figures/final_augmentation_ablation_best_add_ratio_by_condition.png)

### 7.3 F5.1 Aggregate Summary by augmentation_add_ratio

위 7.2 절은 각 `(model, real_ratio)` 조건별 결과를 보여준다. 이에 비해 본 절은 전체 `3 models x 3 real_ratios = 9`개 조건을 `augmentation_add_ratio`별로 다시 묶어, synthetic sample 비율이 전반적으로 어떤 경향을 보이는지 집계한 결과이다. 즉, condition-level 해석과 별도로 **add ratio 자체의 평균적 경향과 변동성**을 보는 요약 계층이다.

| augmentation_add_ratio | n_conditions | mean_test_macro_f1 | std_test_macro_f1 | mean_augmentation_gain_macro_f1 | std_augmentation_gain_macro_f1 | positive_gain_rate | source mix |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 9 | 0.8504 | 0.0873 | 0.0772 | 0.2020 | 55.6% | reused=0, trained=9 |
| 1.0 | 9 | 0.8367 | 0.0899 | 0.0634 | 0.2277 | 55.6% | reused=9, trained=0 |
| 2.0 | 9 | 0.7452 | 0.2661 | -0.0281 | 0.3925 | 22.2% | reused=0, trained=9 |

이 집계 결과는 다음과 같이 해석할 수 있다.

- `augmentation_add_ratio=0.5`는 평균 `test_macro_f1`와 평균 `augmentation_gain_macro_f1` 모두에서 가장 안정적인 설정으로 나타났다.
- `augmentation_add_ratio=1.0`도 일부 조건에서는 유효했지만, 전체 9개 조건 평균에서는 `0.5`를 일관되게 넘어서는 수준은 아니었다.
- `augmentation_add_ratio=2.0`는 평균 gain이 음수이고 표준편차도 가장 커서, synthetic data를 과도하게 늘릴 경우 성능과 안정성이 함께 악화될 수 있음을 시사한다.
- 즉, synthetic data는 "많을수록 좋다"기보다 **모델과 real_ratio에 따라 적정량을 조절해야 하는 자원**에 가깝다.
- 특히 CSI HAR에서는 과도한 synthetic sample이 원래 CSI signal distribution을 왜곡하거나, 모델의 inductive bias와 충돌하여 robustness를 해칠 수 있다.

집계 관점에서도 `0.5`가 가장 실용적인 기본값으로 보였지만, 이는 모든 조건에서 절대적으로 최적이라는 뜻은 아니다. 예를 들어 `LeNet`의 `real_ratio=0.1`에서는 더 높은 synthetic ratio가 큰 회복 효과를 보였으나, 이는 baseline 붕괴 이후의 회복으로 해석해야 하며 universal superiority로 일반화해서는 안 된다. 반대로 F4에서 가장 실용적이었던 `ResNet18`조차 augmentation tuning이 항상 이득을 보장하지는 않았다.

특히 `augmentation_add_ratio=0.5`는 평균적으로 가장 좋은 결과를 보였지만, 평균 `augmentation_gain_macro_f1`은 `+0.0772` 수준이고 표준편차도 커서 조건별 편차가 존재한다. 따라서 이 결과를 augmentation이 low-data 문제를 안정적으로 해결했다는 강한 성공으로 보기 어렵다. 오히려 특정 조건에서 성능 회복 가능성을 보인 초기적 시도로 해석하는 것이 더 타당하다.

특히 `LeNet`의 `real_ratio=0.1` 조건에서는 큰 성능 회복이 나타났지만, 이는 원래 baseline이 거의 붕괴한 조건에서 synthetic sample이 최소한의 학습 신호를 보완한 사례로 해석해야 한다. 따라서 이 결과를 augmentation의 보편적 우수성으로 일반화하기는 어렵다.

반대로 F4에서 가장 실용적인 모델로 나타난 `ResNet18`에서는 augmentation 효과가 제한적이었다. `real_ratio=0.25` 조건에서는 어떤 `augmentation_add_ratio`도 F4 no-augmentation baseline을 넘지 못했고, 이는 이미 low-data robustness가 높은 모델에서는 synthetic data가 추가 정보보다 분포 교란으로 작동할 수 있음을 시사한다.

![Aggregate F5.1 raw test Macro F1 and gain by augmentation_add_ratio](../results/figures/final_augmentation_ablation_add_ratio_summary_macro_f1.png)

위 aggregate 결과는 augmentation을 강한 성공 사례로 보기 어렵다는 점도 동시에 보여준다. `augmentation_add_ratio=0.5`와 `1.0`의 positive-gain rate가 모두 `5/9`에 그쳤고, 이는 augmentation이 일부 조건에서는 회복 효과를 보였지만 broad success나 robust universal fix로 해석될 정도는 아니라는 뜻이다. 특히 `augmentation_add_ratio=2.0`는 평균 gain이 음수이고 변동성도 가장 커서 over-augmentation risk가 분명하다.

또한 CSI HAR에서는 image-like augmentation 직관이 그대로 통하지 않을 수 있다. CSI는 propagation path와 channel response가 반영된 신호이므로, 단순 synthetic transform이 물리적으로 의미 있는 구조를 충분히 보존하지 못할 수 있고, synthetic sample 비율이 지나치게 커지면 실제 분포보다 augmentation 분포가 train set을 지배하여 모델이 잘못된 signal prior를 학습할 가능성이 있다. 따라서 향후에는 physics-aware augmentation, signal-structure-aware synthetic generation, class-conditional augmentation, model-specific augmentation strength가 필요하다.

<table>
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="../results/figures/final_augmentation_ablation_add_ratio_summary_gain.png" alt="Aggregate F5.1 Macro F1 gain by augmentation_add_ratio" width="100%"><br>
      <sub>Aggregate Macro F1 gain by add ratio</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <img src="../results/figures/final_augmentation_ablation_add_ratio_positive_rate.png" alt="Aggregate F5.1 positive-gain rate by augmentation_add_ratio" width="100%"><br>
      <sub>Positive-gain rate by add ratio</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <img src="../results/figures/final_augmentation_ablation_add_ratio_accuracy_summary.png" alt="Aggregate F5.1 raw test Accuracy by augmentation_add_ratio" width="100%"><br>
      <sub>Raw test Accuracy summary by add ratio</sub>
    </td>
  </tr>
</table>

본 실험에서 augmentation은 low-data 문제를 일관되게 해결하는 만능 방법은 아니었다. `augmentation_add_ratio=0.5`가 평균적으로 가장 안정적인 결과를 보였지만, 평균 gain은 제한적이었고 조건별 편차도 컸다. 특히 `augmentation_add_ratio=2.0`처럼 synthetic data 비율을 과도하게 높이면 성능이 악화되거나 collapse가 발생할 수 있었다. 따라서 Wi-Fi CSI HAR에서 augmentation은 “데이터를 많이 늘리면 좋아진다”는 방식으로 접근하기보다, CSI의 물리적 특성, class별 행동 패턴, model별 민감도를 반영하여 신중하게 설계해야 한다. 본 프로젝트의 augmentation 결과는 부분적 가능성을 보였지만, 최종 해결책이라기보다는 향후 physics-aware, class-conditional, model-specific augmentation으로 발전시켜야 할 출발점으로 해석하는 것이 타당하다.

CSI는 이미지와 달리 시간축, subcarrier축, antenna-pair 간 상관 구조가 행동 정보와 물리적 채널 정보를 동시에 담는다. 따라서 일반적인 noise, masking, shift, scaling 기반 augmentation은 실제 환경 변화나 행동 변화의 물리적 구조를 충분히 반영하지 못할 수 있다. 또한 synthetic sample 비율이 지나치게 커질 경우 모델이 실제 CSI 분포보다 augmentation artifact에 더 강하게 노출될 수 있다. 이는 `augmentation_add_ratio=2.0`에서 성능 저하와 collapse가 나타난 이유 중 하나로 해석할 수 있다.

모든 class에 동일한 augmentation policy를 적용한 것도 한계다. 행동별 CSI 패턴은 지속 시간, 움직임 속도, 반복성, 자세 변화 정도가 다르므로, class-conditional augmentation이 필요할 수 있다. 또한 augmentation 효과는 모델 구조에 따라 달랐다. `LeNet`처럼 low-data에서 붕괴하기 쉬운 모델은 synthetic sample로 회복 효과를 보였지만, `ResNet18`처럼 이미 robust한 모델에서는 augmentation이 항상 추가 이득으로 이어지지 않았다. 이는 model-specific augmentation strength가 필요함을 의미한다.

## 8. 종합 논의

본 연구는 Wi-Fi CSI HAR가 privacy-friendly한 고령자 돌봄 보조 기술로서 가능성을 가진다는 점을 보여준다. Camera 기반 시스템과 달리 시각 정보 자체를 수집하지 않으면서도 낙상이나 활동 변화를 감지할 수 있다는 점은 실제 수용성 측면에서 중요한 장점이다. 그러나 practical deployment의 핵심은 full-data benchmark 성능보다도 제한된 calibration data에서 얼마나 안정적인가에 달려 있다.

이 관점에서 `ResNet18 + moving_average_smoothing+minmax_scaling` 조합은 유의미한 결과를 보였다. `real_ratio=0.25`에서 test `Macro F1 = 0.9008`을 달성한 것은 비교적 적은 labeled data만으로도 usable한 수준의 성능 유지가 가능함을 시사한다. 반면 `real_ratio=0.1`은 여전히 어려운 구간이며, 모델 간 격차가 크게 벌어진다.

augmentation 측면에서는 두 가지 사실이 동시에 드러났다. 첫째, `offline_append` augmentation은 on-the-fly 방식보다 “고정된 실제 train subset에 synthetic sample을 추가하면 성능이 회복되는가”라는 연구 질문에 더 직접적으로 대응한다. 둘째, F5.1은 augmentation의 양 자체가 중요한 조절 변수임을 보여준다. 완만한 synthetic ratio인 `0.5`는 평균적으로 가장 안정적인 반면, 공격적인 `2.0`은 CSI 분포를 왜곡하거나 모델의 inductive bias와 충돌하여 성능을 악화시킬 수 있다. 즉, over-augmentation은 CSI HAR 성능을 해칠 수 있다.

## 9. 한계점

본 연구의 한계는 다음과 같다.

1. `UT-HAR` benchmark setting이 실제 가정 환경을 완전하게 대표한다고 보기 어렵다.
2. sampling rate 기반 시간 환산은 illustrative estimate일 뿐이며, 실제 수집 시간의 엄밀한 측정이 아니다.
3. 데이터 분할은 benchmark split이며, 실제 household adaptation 시나리오를 직접 재현한 것은 아니다.
4. F5 및 F5.1은 여전히 single-seed 결과이므로 augmentation 안정성을 충분히 검증하지 못했다.
5. ratio ablation은 `0.5`, `1.0`, `2.0`에 대해서만 수행되었고, 더 세밀한 값과 multi-seed 검증이 필요하다.
6. model-specific 및 class-specific augmentation policy는 아직 시험하지 않았다.
7. synthetic CSI sample이 실제 방 구조, 사람 간 차이, 신체 움직임 다양성을 물리적으로 충분히 반영한다고 보장할 수 없다.
8. 본 프로젝트는 실제 배치 환경이나 online adaptation을 포함하지 않는다.

## 10. 개선 방향

향후 개선 방향은 다음과 같다.

- finer ratio search: `0.25`, `0.5`, `0.75`, `1.0` 등 더 세밀한 synthetic ratio 탐색
- multi-seed F5.1 stability check
- model-specific augmentation strength 설계
- class-conditional augmentation
- physics-aware CSI augmentation
- environment-aware augmentation
- few-shot / domain adaptation
- self-supervised pretraining
- environment probing과 room metadata 활용
- privacy-aware multi-modal sensing
- household 또는 user 단위 personalization

즉, “augmentation_add_ratio ablation이 필요하다”는 단계는 지나갔고, 이제는 **추가 비율 탐색과 multi-seed 검증, 그리고 모델/환경 인지형 augmentation 설계**가 다음 과제에 해당한다.

## 11. 결론

official F1-F5.1 workflow를 종합하면, 최종 전처리는 `moving_average_smoothing+minmax_scaling`으로 확정되었고, practical low-data robustness 측면에서 가장 강한 모델은 `ResNet18`이었다. 특히 `ResNet18`은 `real_ratio=0.25`에서 test `Macro F1 = 0.9008`을 기록하여, 제한된 labeled training data로도 상당한 수준의 행동 인식 성능 유지가 가능함을 보여주었다.

augmentation 측면에서는 synthetic data가 일부 low-data 조건에서 실제로 성능 회복에 기여할 수 있었다. 그러나 F5.1은 augmentation의 양을 반드시 조정해야 한다는 점도 분명히 보여주었다. `augmentation_add_ratio=0.5`는 `2.0`보다 더 안정적이었고, 과도한 synthetic data는 오히려 성능을 해칠 수 있었다. 따라서 augmentation은 유망한 방향이지만, ratio와 모델 특성, 그리고 환경 차이를 고려한 정교한 설계와 multi-seed 검증이 필요하다.

## 참고 자료

- 통계청 장래인구추계
- 보건복지부 고독사 발생 실태조사
- `UT-HAR` / `WiFi-CSI-Sensing-Benchmark` repository
- 본 프로젝트의 official F1-F5.1 outputs
